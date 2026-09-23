"""The candidate oracle changes selection while retaining the P2 experiment setup."""

from pathlib import Path
import unittest

import torch

from src.core import YAMLConfig


class GTCandidateConfigTests(unittest.TestCase):
    def test_config_matches_p2_training_and_fresh_head_initialization(self):
        root = Path(__file__).resolve().parents[1]
        oracle_cfg = YAMLConfig(str(root / "configs/dfine/gt_refinement/dfine_hgnetv2_l_uavswarm_gt_candidate_oracle.yml"))
        p2_cfg = YAMLConfig(str(root / "configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_pred_p2.yml"))
        for name in ("epochs", "optimizer", "lr_scheduler", "lr_warmup_scheduler", "train_dataloader", "val_dataloader"):
            self.assertEqual(oracle_cfg.yaml_cfg[name], p2_cfg.yaml_cfg[name], name)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            oracle = oracle_cfg.model
            torch.manual_seed(7)
            p2 = p2_cfg.model
        self.assertTrue(oracle.requires_gt)
        self.assertTrue(oracle.oracle_evaluation)
        self.assertEqual(oracle_cfg.yaml_cfg["criterion"], "GTRefinementCriterion")
        self.assertTrue(oracle_cfg.yaml_cfg["output_dir"].endswith("/p2_gt_nearest"))
        self.assertEqual(set(oracle.refiner.state_dict()), set(p2.refiner.state_dict()))
        for name, value in oracle.refiner.state_dict().items():
            torch.testing.assert_close(value, p2.refiner.state_dict()[name], atol=0, rtol=0)
        trainable = {id(p) for p in oracle.parameters() if p.requires_grad}
        optimized = {id(p) for group in oracle_cfg.optimizer.param_groups for p in group["params"]}
        self.assertEqual(trainable, optimized)


if __name__ == "__main__":
    unittest.main()
