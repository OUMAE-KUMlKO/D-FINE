"""Integration checks for distribution-guided refinement with a real decoder."""

from pathlib import Path
import unittest

import torch

from src.core import YAMLConfig
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.matcher import HungarianMatcher
from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
from test_gt_refinement import normalized_targets
from test_pred_refinement import make_model


class PredP2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.images = torch.randn(1, 3, 16, 24)

    def make_p2(self):
        decoder = DFINETransformer(
            num_classes=2, hidden_dim=8, num_queries=3,
            feat_channels=[8], feat_strides=[1], num_levels=1,
            num_points=2, nhead=2, num_layers=2, dim_feedforward=16,
            num_denoising=0, reg_max=4,
        )
        return make_model(decoder=decoder, sampling_mode="distribution", query_chunk_size=2)

    def test_zero_init_preserves_baseline_and_exposes_detached_candidates(self):
        model = self.make_p2().eval()
        output = model(self.images)
        torch.testing.assert_close(output["pred_boxes"], output["baseline_boxes"], atol=0, rtol=0)
        diagnostics = output["sampling_diagnostics"]
        self.assertEqual(diagnostics["candidate_positions"].shape, (1, 3, 4, 5))
        self.assertEqual(diagnostics["candidate_valid"].dtype, torch.bool)
        self.assertEqual(diagnostics["excluded_probability_mass"].shape, (1, 3, 4))
        for value in diagnostics.values():
            self.assertFalse(value.requires_grad)
        self.assertNotIn("matched_indices", output)
        self.assertFalse(model.requires_gt)

    def test_nonzero_head_ignores_gt_and_only_trains_refiner(self):
        model = self.make_p2().train()
        with torch.no_grad():
            model.refiner.refiner[-1].bias.fill_(0.03)
        expected = model(self.images)
        for targets in (normalized_targets(), [{"unread_target": object()}]):
            actual = model(self.images, targets=targets)
            for name in ("pred_boxes", "pred_logits", "refinement_mask"):
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        self.assertTrue(expected["refinement_mask"].any())
        self.assertTrue((expected["pred_boxes"] != expected["baseline_boxes"]).any())
        criterion = PredRefinementCriterion(HungarianMatcher(
            weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            alpha=0.25, gamma=2.0,
        ))
        for targets in (normalized_targets(), [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}]):
            model.zero_grad(set_to_none=True)
            loss = sum(criterion(model(self.images), targets).values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if name.startswith("refiner."):
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertFalse(parameter.requires_grad, name)
                    self.assertIsNone(parameter.grad, name)
        self.assertFalse(model.decoder.training)

    def test_p2_config_keeps_p1_training_settings(self):
        root = Path(__file__).resolve().parents[1]
        config = YAMLConfig(str(root / "configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_pred_p2.yml"))
        self.assertEqual(config.model.refiner.sampling_mode, "distribution")
        trainable = {id(p) for p in config.model.refiner.parameters()}
        optimized = {id(p) for group in config.optimizer.param_groups for p in group["params"]}
        self.assertEqual(trainable, optimized)
        self.assertEqual(config.yaml_cfg["epochs"], 12)
        self.assertEqual(config.yaml_cfg["train_dataloader"]["total_batch_size"], 96)
        self.assertEqual(config.yaml_cfg["DFINEPredRefiner"]["offset_pixels"], 1.0)
        self.assertTrue(config.yaml_cfg["output_dir"].endswith("/p2"))


if __name__ == "__main__":
    unittest.main()
