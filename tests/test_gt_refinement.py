"""CPU integration checks for the GT-only diagnostic (no training loop).

Run from the repository root with:
    python -m unittest discover -s tests -p 'test_gt_refinement.py' -v
"""

import copy
import unittest

import torch
import torch.nn as nn

from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.gt_refinement import DFINEGTRefiner
from src.zoo.dfine.gt_refinement_criterion import (
    GTRefinementCriterion, aligned_generalized_box_iou,
)
from src.zoo.dfine.matcher import HungarianMatcher


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8))

    def forward(self, images):
        return [self.layers(images)]


class ToyEncoder(nn.Module):
    hidden_dim = 8

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(8, 8, 1)

    def forward(self, features):
        return [self.projection(features[0])]


class ToyDecoder(nn.Module):
    hidden_dim = 8
    num_classes = 2

    def __init__(self):
        super().__init__()
        self.boxes = nn.Parameter(
            torch.tensor([[0.35, 0.40, 0.20, 0.25], [0.80, 0.80, 0.10, 0.10],
                          [0.15, 0.80, 0.10, 0.10]])
        )
        self.logits = nn.Parameter(torch.tensor([[4.0, -4.0], [-4.0, 4.0], [-2.0, 2.0]]))
        self.query = nn.Parameter(torch.randn(3, self.hidden_dim))
        self.calls = []

    def forward(self, features, targets=None, return_query=False):
        self.calls.append((self.training, targets, return_query))
        batch = features[0].shape[0]
        outputs = {
            "pred_boxes": self.boxes.unsqueeze(0).expand(batch, -1, -1),
            "pred_logits": self.logits.unsqueeze(0).expand(batch, -1, -1),
        }
        if return_query:
            outputs["pred_queries"] = (
                self.query.unsqueeze(0) + features[0].mean((2, 3)).unsqueeze(1)
            )
        return outputs


class CountingMatcher(HungarianMatcher):
    def __init__(self):
        super().__init__({"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
                         use_focal_loss=True)
        self.calls = 0

    def forward(self, outputs, targets, **kwargs):
        self.calls += 1
        return super().forward(outputs, targets, **kwargs)


def make_model(loaded=True, **kwargs):
    model = DFINEGTRefiner(
        ToyBackbone(), ToyEncoder(), ToyDecoder(), CountingMatcher(),
        hidden_dim=8, num_points=5, kernel_size=3, **kwargs,
    )
    if loaded:
        # Synthetic state only: tests do not claim a pretrained checkpoint.
        model.load_state_dict(model.state_dict())
    return model


def normalized_targets(batch=1):
    return [
        {"boxes": torch.tensor([[0.40, 0.40, 0.25, 0.30]]),
         "labels": torch.tensor([0])}
        for _ in range(batch)
    ]


class GTRefinementIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.images = torch.randn(1, 3, 32, 48)

    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_requires_checkpoint_and_ground_truth(self):
        model = make_model(loaded=False)
        with self.assertRaises((RuntimeError, ValueError)):
            model(self.images, normalized_targets())
        model.load_state_dict(model.state_dict())
        for training in (False, True):
            model.train(training)
            with self.assertRaises((RuntimeError, ValueError)):
                model(self.images)

    def test_rejects_incomplete_baseline_checkpoint(self):
        model = make_model(loaded=False)
        state = model.state_dict()
        del state["decoder.boxes"]
        with self.assertRaises((RuntimeError, ValueError)):
            model.load_state_dict(state, strict=False)

    def test_accepts_complete_baseline_only_state(self):
        model = make_model(loaded=False)
        baseline = {
            name: value for name, value in model.state_dict().items()
            if name.startswith(("backbone.", "encoder.", "decoder."))
        }
        model.load_state_dict(baseline, strict=False)
        outputs = model(self.images, normalized_targets())
        self.assert_exact(outputs["pred_boxes"], outputs["baseline_boxes"])

    def test_zero_initialization_preserves_all_boxes_and_original_scores(self):
        model = make_model().train()
        outputs = model(self.images, normalized_targets())
        self.assert_exact(outputs["pred_boxes"], outputs["baseline_boxes"])
        self.assert_exact(outputs["pred_logits"], model.decoder.logits.unsqueeze(0))
        self.assertEqual(outputs["pred_boxes"].shape, (1, 3, 4))
        self.assertEqual(int(outputs["refinement_mask"].sum()), 1)
        self.assertTrue(model.refiner.training)
        for baseline in (model.backbone, model.encoder, model.decoder):
            self.assertTrue(all(not layer.training for layer in baseline.modules()))
            self.assertTrue(all(not p.requires_grad for p in baseline.parameters()))
        self.assertEqual(model.decoder.calls, [(False, None, True)])

    def test_only_matched_queries_change_and_scores_stay_fixed(self):
        model = make_model()
        # A known nonzero residual makes unmatched-query fallback observable.
        final_linear = [layer for layer in model.refiner.modules()
                        if isinstance(layer, nn.Linear)][-1]
        with torch.no_grad():
            final_linear.bias.copy_(torch.tensor([0.1, -0.1, 0.2, -0.2]))
        outputs = model(self.images, normalized_targets())
        mask = outputs["refinement_mask"]
        self.assert_exact(outputs["pred_boxes"][~mask], outputs["baseline_boxes"][~mask])
        self.assertFalse(torch.equal(outputs["pred_boxes"][mask], outputs["baseline_boxes"][mask]))
        self.assertTrue((outputs["pred_boxes"][mask][:, 2:] > 0).all())
        self.assert_exact(outputs["pred_logits"], model.decoder.logits.unsqueeze(0))

    def test_loss_reuses_fixed_matches_and_has_no_baseline_gradients(self):
        model = make_model().train()
        running_mean = model.backbone.layers[1].running_mean.clone()
        outputs = model(self.images, normalized_targets())
        self.assertEqual(model.matcher.calls, 1)
        criterion = GTRefinementCriterion()
        losses = criterion(outputs, normalized_targets())
        self.assertEqual(model.matcher.calls, 1)
        self.assertEqual(set(losses), {"loss_bbox", "loss_giou"})
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0)
        loss.backward()
        for baseline in (model.backbone, model.encoder, model.decoder):
            self.assertTrue(all(p.grad is None for p in baseline.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                            for p in model.refiner.parameters()))
        self.assert_exact(model.backbone.layers[1].running_mean, running_mean)

    def test_empty_gt_preserves_predictions_and_supports_backward(self):
        model = make_model().train()
        targets = [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}]
        outputs = model(self.images, targets)
        self.assert_exact(outputs["pred_boxes"], outputs["baseline_boxes"])
        self.assertFalse(outputs["refinement_mask"].any())
        losses = GTRefinementCriterion()(outputs, targets)
        loss = sum(losses.values())
        self.assertEqual(float(loss), 0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.refiner.parameters()
                            if p.requires_grad))

    def test_eval_xyxy_targets_use_input_size_without_mutating_originals(self):
        model = make_model().eval()
        targets = [{"boxes": torch.tensor([[12.0, 8.0, 36.0, 24.0]]),
                    "labels": torch.tensor([0]), "orig_size": torch.tensor([320, 480])}]
        original = copy.deepcopy(targets)
        outputs = model(self.images, targets)
        self.assert_exact(outputs["refinement_targets"][0]["boxes"],
                          torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
        for key in original[0]:
            self.assert_exact(targets[0][key], original[0][key])

    def test_explicit_normalized_format_in_eval(self):
        model = make_model().eval()
        targets = normalized_targets()
        outputs = model(self.images, targets, target_format="cxcywh_normalized")
        self.assert_exact(outputs["refinement_targets"][0]["boxes"], targets[0]["boxes"])

    def test_ignore_crowd_and_invalid_targets_are_not_matched(self):
        model = make_model().train()
        targets = [{
            "boxes": torch.tensor([[0.4, 0.4, 0.25, 0.3], [0.6, 0.6, 0.1, 0.1],
                                   [0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.0, 0.1],
                                   [float("nan"), 0.5, 0.1, 0.1]]),
            "labels": torch.zeros(5, dtype=torch.long),
            "iscrowd": torch.tensor([0, 1, 0, 0, 0]),
            "ignore": torch.tensor([0, 0, 1, 0, 0]),
        }]
        outputs = model(self.images, targets)
        self.assertEqual(len(outputs["refinement_targets"][0]["boxes"]), 1)
        self.assertEqual(len(outputs["matched_indices"][0][0]), 1)
        self.assertEqual(int(outputs["refinement_mask"].sum()), 1)


    def test_invalid_gt_sampling_preserves_matches_but_skips_refinement(self):
        model = make_model().train()
        targets = [{"boxes": torch.tensor([[1.5, 1.5, 0.2, 0.2]]),
                    "labels": torch.tensor([0])}]
        outputs = model(self.images, targets)
        self.assertEqual(len(outputs["matched_indices"][0][0]), 1)
        self.assertEqual(len(outputs["refinement_indices"][0][0]), 0)
        self.assertEqual(int(outputs["refinement_counts"]["invalid_gt_samples"]), 1)
        self.assert_exact(outputs["pred_boxes"], outputs["baseline_boxes"])
        loss = sum(GTRefinementCriterion()(outputs, targets).values())
        self.assertEqual(float(loss), 0)
        loss.backward()

    def test_mixed_empty_and_nonempty_batch_has_independent_fallback(self):
        model = make_model().train()
        targets = normalized_targets() + [
            {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
        ]
        outputs = model(self.images.expand(2, -1, -1, -1), targets)
        self.assertEqual(len(outputs["matched_indices"][0][0]), 1)
        self.assertEqual(len(outputs["matched_indices"][1][0]), 0)
        self.assertFalse(outputs["refinement_mask"][1].any())
        self.assert_exact(outputs["pred_boxes"][1], outputs["baseline_boxes"][1])
        loss = sum(GTRefinementCriterion()(outputs, targets).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_counters_preserve_each_image_and_sum_to_batch_totals(self):
        # Different image outcomes must stay separable when sampler padding
        # occupies only part of the final evaluation batch.
        for mode in ("real", "baseline"):
            with self.subTest(feature_mode=mode):
                model = make_model(feature_mode=mode).eval()
                original_forward = model.decoder.forward

                def forward_with_invalid_box(features, **kwargs):
                    result = original_forward(features, **kwargs)
                    result["pred_boxes"] = result["pred_boxes"].clone()
                    result["pred_boxes"][1, 2, 2] = 0
                    return result

                model.decoder.forward = forward_with_invalid_box
                targets = normalized_targets() + [
                    {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
                    {"boxes": torch.tensor([[0.4, 0.4, 0.25, 0.3], [1.5, 1.5, 0.2, 0.2]]),
                     "labels": torch.tensor([0, 0]), "ignore": torch.tensor([1, 0])},
                ]
                outputs = model(
                    self.images.expand(3, -1, -1, -1), targets,
                    target_format="cxcywh_normalized",
                )
                expected = {
                    "matched": [1, 0, 1],
                    "refined": [1, 0, 0] if mode == "real" else [0, 0, 0],
                    "ignored_gt": [0, 0, 1],
                    "total_gt": [1, 0, 1],
                    "unmatched_gt": [0, 0, 0],
                    "unmatched_queries": [2, 3, 2],
                    "invalid_gt_samples": [0, 0, 1] if mode == "real" else [0, 0, 0],
                    "invalid_baseline_boxes": [0, 1, 0],
                }
                self.assertEqual(set(outputs["refinement_counts_per_image"]), set(expected))
                for name, values in expected.items():
                    actual = outputs["refinement_counts_per_image"][name]
                    self.assert_exact(actual, torch.tensor(values, dtype=torch.long))
                    self.assert_exact(outputs["refinement_counts"][name], actual.sum())

    def test_real_decoder_backward_connects_all_head_parameters(self):
        for feature_mode in ("real", "constant", "random"):
            for empty in (False, True):
                with self.subTest(feature_mode=feature_mode, empty=empty):
                    decoder = DFINETransformer(
                        num_classes=2, hidden_dim=8, num_queries=3,
                        feat_channels=[8], feat_strides=[1], num_levels=1,
                        num_points=2, nhead=2, num_layers=1, dim_feedforward=16,
                        num_denoising=0, reg_max=4,
                    )
                    model = DFINEGTRefiner(
                        ToyBackbone(), ToyEncoder(), decoder, CountingMatcher(),
                        hidden_dim=8, feature_mode=feature_mode,
                    )
                    model.load_state_dict(model.state_dict())
                    model.train()
                    targets = (
                        [{"boxes": torch.empty(0, 4),
                          "labels": torch.empty(0, dtype=torch.long)}]
                        if empty else normalized_targets()
                    )
                    outputs = model(self.images, targets)
                    loss = sum(GTRefinementCriterion()(outputs, targets).values())
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if name.startswith("refiner."):
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        else:
                            self.assertIsNone(parameter.grad, name)


    def test_random_oracle_eval_accepts_dataset_image_ids_and_is_repeatable(self):
        model = make_model(feature_mode="random", random_seed=123).eval()
        with torch.no_grad():
            model.refiner.refiner[-1].weight.normal_(std=0.1)
        targets = [{"boxes": torch.tensor([[12.0, 8.0, 36.0, 24.0]]),
                    "labels": torch.tensor([0]), "image_id": torch.tensor([37])}]
        outputs = model(self.images, targets)
        repeated = model(self.images, targets)
        self.assert_exact(outputs["pred_boxes"], repeated["pred_boxes"])
        self.assert_exact(outputs["pred_logits"], repeated["pred_logits"])
        self.assertEqual(int(outputs["refinement_mask"].sum()), 1)

    def test_tiny_identical_boxes_have_zero_giou_loss(self):
        boxes = torch.tensor([[0.5, 0.5, 0.50001, 0.50001]])
        overlap = aligned_generalized_box_iou(boxes, boxes)
        torch.testing.assert_close(overlap, torch.ones_like(overlap), rtol=0, atol=1e-6)


class DecoderQueryCompatibilityTests(unittest.TestCase):
    def test_optional_query_preserves_regular_eval_outputs(self):
        torch.manual_seed(5)
        decoder = DFINETransformer(
            num_classes=2, hidden_dim=16, num_queries=3,
            feat_channels=[16], feat_strides=[8], num_levels=1,
            num_points=2, nhead=4, num_layers=2, dim_feedforward=32,
            num_denoising=0, reg_max=4,
        ).eval()
        features = [torch.randn(1, 16, 4, 4)]
        with torch.no_grad():
            regular = decoder(features)
            extended = decoder(features, return_query=True)
        self.assertEqual(set(regular), {"pred_logits", "pred_boxes"})
        self.assertEqual(extended["pred_queries"].shape, (1, 3, 16))
        for key in regular:
            torch.testing.assert_close(regular[key], extended[key], rtol=0, atol=0)
        decoder.train()
        with self.assertRaises(ValueError):
            decoder(features, return_query=True)


if __name__ == "__main__":
    unittest.main()
