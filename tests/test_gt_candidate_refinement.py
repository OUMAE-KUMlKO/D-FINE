"""CPU integration checks for the GT-selected candidate-strip oracle."""

import tempfile
from pathlib import Path
import unittest

import torch

from src.zoo.dfine import DFINEGTCandidateRefiner
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.gt_refinement import DFINEGTRefiner
from src.zoo.dfine.gt_refinement_criterion import GTRefinementCriterion
from test_gt_refinement import ToyBackbone, ToyEncoder, CountingMatcher, normalized_targets


def make_model(model_class=DFINEGTCandidateRefiner, **kwargs):
    decoder = DFINETransformer(
        num_classes=2, hidden_dim=8, num_queries=3,
        feat_channels=[8], feat_strides=[1], num_levels=1,
        num_points=2, nhead=2, num_layers=2, dim_feedforward=16,
        num_denoising=0, reg_max=4,
    )
    model = model_class(
        ToyBackbone(), ToyEncoder(), decoder, CountingMatcher(),
        hidden_dim=8, num_points=3, **kwargs,
    )
    if "baseline_checkpoint" not in kwargs:
        # Synthetic state checks interfaces; it is not an experiment baseline.
        model.load_state_dict(model.state_dict())
    return model


class GTCandidateRefinementTests(unittest.TestCase):
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

    def test_registered_oracle_requires_gt_in_train_and_eval(self):
        model = make_model()
        self.assertIsInstance(model, DFINEGTRefiner)
        self.assertTrue(model.requires_gt)
        self.assertTrue(model.oracle_evaluation)
        for training in (True, False):
            model.train(training)
            with self.assertRaisesRegex(ValueError, "requires GT"):
                model(self.images)

    def test_real_decoder_exports_only_matched_distribution_rows_to_head(self):
        model = make_model().train()
        baseline_outputs, head_inputs = [], []
        decoder_hook = model.decoder.register_forward_hook(
            lambda module, args, output: baseline_outputs.append(output)
        )
        head_hook = model.refiner.register_forward_pre_hook(
            lambda module, args, kwargs: head_inputs.append((args, kwargs)),
            with_kwargs=True,
        )
        try:
            outputs = model(self.images.expand(2, -1, -1, -1), normalized_targets(2))
        finally:
            decoder_hook.remove()
            head_hook.remove()
        self.assertEqual(len(baseline_outputs), 1)
        self.assertEqual(len(head_inputs), 1)
        baseline = baseline_outputs[0]
        args, kwargs = head_inputs[0]
        distribution = kwargs["distribution"]
        batch_ids = torch.cat([
            torch.full_like(source, index)
            for index, (source, _) in enumerate(outputs["matched_indices"])
        ])
        query_ids = torch.cat([source for source, _ in outputs["matched_indices"]])
        self.assertEqual(len(query_ids), 2)
        self.assertEqual(set(distribution), {"pred_corners", "ref_points", "up", "reg_scale"})
        for name in ("pred_corners", "ref_points"):
            torch.testing.assert_close(distribution[name], baseline[name][batch_ids, query_ids])
        for name in ("up", "reg_scale"):
            torch.testing.assert_close(distribution[name], baseline[name])
        self.assertTrue(all(not value.requires_grad for value in distribution.values()))
        torch.testing.assert_close(args[1], baseline["pred_queries"][batch_ids, query_ids])
        torch.testing.assert_close(args[2], baseline["pred_boxes"][batch_ids, query_ids])
        torch.testing.assert_close(outputs["pred_logits"], baseline["pred_logits"])
        self.assertEqual(model.matcher.calls, 2)
        GTRefinementCriterion()(outputs, normalized_targets(2))
        self.assertEqual(model.matcher.calls, 2)

    def test_zero_initialization_and_unmatched_query_fallback(self):
        model = make_model().train()
        outputs = model(self.images, normalized_targets())
        torch.testing.assert_close(outputs["pred_boxes"], outputs["baseline_boxes"], rtol=0, atol=0)
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (1, 3, 4))
        self.assertTrue(outputs["refinement_mask"].any())
        with torch.no_grad():
            model.refiner.refiner[-1].bias.fill_(0.1)
        refined = model(self.images, normalized_targets())
        mask = refined["refinement_mask"]
        self.assertTrue(mask.any())
        self.assertFalse(mask.all())
        torch.testing.assert_close(refined["pred_boxes"][~mask], refined["baseline_boxes"][~mask], rtol=0, atol=0)
        self.assertFalse(torch.equal(refined["pred_boxes"][mask], refined["baseline_boxes"][mask]))
        self.assertEqual(int(refined["refinement_counts"]["matched"]), 1)
        self.assertEqual(int(refined["refinement_counts"]["unmatched_queries"]), 2)

    def test_train_normalized_and_eval_absolute_targets_are_equivalent(self):
        model = make_model().train()
        targets = normalized_targets()
        training = model(self.images, targets)
        absolute = [{
            "boxes": box_cxcywh_to_xyxy(targets[0]["boxes"]) * torch.tensor([48, 32, 48, 32]),
            "labels": targets[0]["labels"],
            "orig_size": torch.tensor([480, 320]),
        }]
        original = absolute[0]["boxes"].clone()
        evaluation = model.eval()(self.images, absolute)
        torch.testing.assert_close(evaluation["refinement_targets"][0]["boxes"], targets[0]["boxes"])
        torch.testing.assert_close(evaluation["pred_boxes"], training["pred_boxes"])
        torch.testing.assert_close(evaluation["refinement_mask"], training["refinement_mask"])
        torch.testing.assert_close(absolute[0]["boxes"], original, rtol=0, atol=0)

    def test_baseline_stays_frozen_and_empty_gt_has_full_head_gradients(self):
        model = make_model().train()
        frozen_buffers = {name: value.clone() for name, value in model.named_buffers()}
        empty = [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}]
        for targets in (normalized_targets(), empty):
            model.zero_grad(set_to_none=True)
            outputs = model(self.images, targets)
            loss = sum(GTRefinementCriterion()(outputs, targets).values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if name.startswith("refiner."):
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertFalse(parameter.requires_grad, name)
                    self.assertIsNone(parameter.grad, name)
        self.assertEqual(float(loss), 0)
        self.assertFalse(outputs["refinement_mask"].any())
        torch.testing.assert_close(outputs["pred_boxes"], outputs["baseline_boxes"], rtol=0, atol=0)
        for module in (model.backbone, model.encoder, model.decoder, model.matcher):
            self.assertTrue(all(not child.training for child in module.modules()))
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, frozen_buffers[name], rtol=0, atol=0)

    def test_baseline_loader_does_not_import_existing_gt_head(self):
        original = make_model(model_class=DFINEGTRefiner)
        with torch.no_grad():
            original.refiner.refiner[-1].bias.fill_(0.8)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "g1.pth"
            torch.save({"ema": {"module": original.state_dict()}}, checkpoint)
            fresh = make_model(baseline_checkpoint=checkpoint)
        torch.testing.assert_close(fresh.refiner.refiner[-1].bias, torch.zeros(4), rtol=0, atol=0)
        for name, value in fresh.state_dict().items():
            if name.startswith(("backbone.", "encoder.", "decoder.")):
                torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
