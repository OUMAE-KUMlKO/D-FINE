"""CPU checks for the GT-free P1 model and its training integration."""

import tempfile
from pathlib import Path
import unittest

import torch
import torch.nn as nn

from src.core import YAMLConfig
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.matcher import HungarianMatcher
from src.zoo.dfine.pred_refinement import DFINEPredRefiner
from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
from test_gt_refinement import ToyBackbone, ToyEncoder, ToyDecoder, normalized_targets


def make_model(loaded=True, decoder=None, **kwargs):
    model = DFINEPredRefiner(
        ToyBackbone(), ToyEncoder(), decoder or ToyDecoder(), hidden_dim=8, **kwargs
    )
    if loaded:
        # Synthetic state is only used for interface tests, not an experiment.
        model.load_state_dict(model.state_dict())
    return model


class PredRefinementTests(unittest.TestCase):
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

    def test_requires_complete_baseline_and_zero_init_preserves_outputs(self):
        model = make_model(loaded=False)
        with self.assertRaisesRegex(RuntimeError, "baseline"):
            model(self.images)
        baseline = {
            key: value for key, value in model.state_dict().items()
            if key.startswith(("backbone.", "encoder.", "decoder."))
        }
        incomplete = dict(baseline)
        del incomplete["decoder.boxes"]
        with self.assertRaises(ValueError):
            model.load_state_dict(incomplete, strict=False)
        model.load_state_dict(baseline, strict=False)
        outputs = model(self.images)
        torch.testing.assert_close(outputs["pred_boxes"], outputs["baseline_boxes"], rtol=0, atol=0)
        torch.testing.assert_close(outputs["pred_logits"], model.decoder.logits[None], rtol=0, atol=0)
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (1, 3, 4))
        self.assertTrue(outputs["refinement_mask"].all())
        self.assertFalse(model.requires_gt)
        self.assertFalse(model.oracle_evaluation)

    def test_nonzero_refinement_ignores_targets_and_includes_unmatched_queries(self):
        model = make_model()
        with torch.no_grad():
            model.refiner.refiner[-1].bias.copy_(torch.tensor([0.1, -0.1, 0.2, -0.2]))
        empty = [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}]
        for training in (True, False):
            model.train(training)
            expected = model(self.images)
            for targets in (normalized_targets(), empty, [{"unread_target": object()}]):
                actual = model(self.images, targets=targets)
                for key in ("pred_boxes", "pred_logits", "refinement_mask"):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            self.assertTrue(expected["refinement_mask"].all())
            self.assertTrue((expected["pred_boxes"] != expected["baseline_boxes"]).any(-1).all())
        self.assertNotIn("refinement_targets", expected)
        self.assertNotIn("matched_indices", expected)
        self.assertTrue(all(call == (False, None, True) for call in model.decoder.calls))

    def test_invalid_prediction_is_not_refined(self):
        model = make_model()
        with torch.no_grad():
            model.refiner.refiner[-1].bias.fill_(0.1)
            model.decoder.boxes[1] = torch.tensor([1.5, 1.5, 0.1, 0.1])
            model.decoder.logits[2, 0] = float("nan")
        outputs = model(self.images)
        self.assertEqual(outputs["refinement_mask"].tolist(), [[True, False, False]])
        torch.testing.assert_close(outputs["pred_boxes"][0, 1:], outputs["baseline_boxes"][0, 1:])

    def test_checkpoint_loader_uses_baseline_only_and_resume_restores_head(self):
        original = make_model()
        with torch.no_grad():
            original.refiner.refiner[-1].bias.fill_(0.2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pth"
            torch.save({"ema": {"module": original.state_dict()}}, path)
            fresh = make_model(loaded=False, baseline_checkpoint=path)
        torch.testing.assert_close(fresh.refiner.refiner[-1].bias, torch.zeros(4))
        resumed = make_model(loaded=False)
        resumed.load_state_dict(original.state_dict())
        torch.testing.assert_close(resumed(self.images)["pred_boxes"], original(self.images)["pred_boxes"])

    def test_real_decoder_training_batch_freezes_detector_and_handles_empty_targets(self):
        decoder = DFINETransformer(
            num_classes=2, hidden_dim=8, num_queries=3,
            feat_channels=[8], feat_strides=[1], num_levels=1,
            num_points=2, nhead=2, num_layers=1, dim_feedforward=16,
            num_denoising=0, reg_max=4,
        )
        model = make_model(decoder=decoder).train()
        criterion = PredRefinementCriterion(HungarianMatcher(
            weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            alpha=0.25, gamma=2.0,
        ))
        optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=0.00025)
        frozen_buffers = {name: value.clone() for name, value in model.named_buffers()}
        empty = [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}]
        for targets in (normalized_targets(), empty):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(self.images, targets=targets)
            loss = sum(criterion(outputs, targets).values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if name.startswith("refiner."):
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertFalse(parameter.requires_grad, name)
                    self.assertIsNone(parameter.grad, name)
            optimizer.step()
        for module in (model.backbone, model.encoder, model.decoder):
            self.assertTrue(all(not child.training for child in module.modules()))
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, frozen_buffers[name], rtol=0, atol=0)
        with torch.no_grad():
            self.assertEqual(tuple(model.eval()(self.images)["pred_boxes"].shape), (1, 3, 4))

    def test_p1_config_builds_registered_model_and_only_optimizes_refiner(self):
        path = Path(__file__).resolve().parents[1] / "configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_pred_p1.yml"
        config = YAMLConfig(str(path))
        model, criterion = config.model, config.criterion
        self.assertIsInstance(model, DFINEPredRefiner)
        self.assertIsInstance(criterion, PredRefinementCriterion)
        self.assertFalse(model.requires_gt)
        trainable = {id(p) for p in model.refiner.parameters()}
        optimized = {id(p) for group in config.optimizer.param_groups for p in group["params"]}
        self.assertEqual(trainable, optimized)
        self.assertEqual(config.yaml_cfg["epochs"], 12)
        self.assertEqual(config.yaml_cfg["train_dataloader"]["total_batch_size"], 96)
        self.assertEqual(config.yaml_cfg["DFINEPredRefiner"]["sampling_mode"], "box")


if __name__ == "__main__":
    unittest.main()
