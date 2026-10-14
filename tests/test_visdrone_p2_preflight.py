"""Exercise the P2 preflight with a real decoder and optimizer on CPU."""

from pathlib import Path
from argparse import Namespace
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.core.yaml_utils import load_config
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.matcher import HungarianMatcher
from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
from test_gt_refinement import normalized_targets
from test_pred_refinement import make_model
from tools.analysis.check_visdrone_p2 import check_batch, run


class VisDroneP2PreflightTests(unittest.TestCase):
    def test_run_converts_real_validation_targets_before_updates(self):
        old_threads = torch.get_num_threads()
        try:
            decoder = DFINETransformer(
                num_classes=2, hidden_dim=8, num_queries=3,
                feat_channels=[8], feat_strides=[1], num_levels=1,
                num_points=2, nhead=2, num_layers=1, dim_feedforward=16,
                num_denoising=0, reg_max=4,
            )
            model = make_model(decoder=decoder, sampling_mode='distribution', query_chunk_size=2)
            model.backbone.layers.append(torch.nn.AvgPool2d(40))
            criterion = PredRefinementCriterion(HungarianMatcher(
                weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                alpha=0.25, gamma=2.0,
            ))
            cfg = SimpleNamespace(
                model=model, criterion=criterion, clip_max_norm=.1,
                optimizer=torch.optim.AdamW(model.refiner.parameters(), lr=.00025),
                val_dataloader=[(torch.randn(1, 3, 640, 640), [{
                    'boxes': torch.tensor([[180., 180., 340., 360.]]),
                    'labels': torch.tensor([0]), 'image_id': torch.tensor([1]),
                }])],
            )
            with tempfile.TemporaryDirectory() as folder, patch('src.core.YAMLConfig', return_value=cfg), \
                    patch('builtins.print'):
                summary = run(Namespace(
                    config='toy.yml', baseline_checkpoint='toy.pth', device='cpu',
                    output_file=str(Path(folder) / 'result.json'), batch_size=1,
                    workers=0, threads=1,
                ))
            self.assertEqual(summary['status'], 'passed')
            self.assertEqual(summary['valid_gt_counts'], [1])
        finally:
            torch.set_num_threads(old_threads)

    def test_real_updates_preserve_frozen_detector_and_ignore_gt(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(0)
            decoder = DFINETransformer(
                num_classes=2, hidden_dim=8, num_queries=3,
                feat_channels=[8], feat_strides=[1], num_levels=1,
                num_points=2, nhead=2, num_layers=2, dim_feedforward=16,
                num_denoising=0, reg_max=4,
            )
            model = make_model(decoder=decoder, sampling_mode='distribution', query_chunk_size=2)
            criterion = PredRefinementCriterion(HungarianMatcher(
                weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                alpha=0.25, gamma=2.0,
            ))
            optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=.00025)
            result = check_batch(
                model, criterion, optimizer, torch.randn(1, 3, 16, 24),
                normalized_targets(), use_amp=False, clip_max_norm=.1,
            )
            self.assertEqual(result['successful_updates'], 2)
            self.assertTrue(result['with_without_gt_identical'])
            self.assertTrue(result['frozen_parameters_and_buffers_unchanged'])
            self.assertTrue(result['classification_preserved_after_updates'])
            self.assertGreater(result['max_box_change_after_two_updates'], 0)
            self.assertGreater(result['refined_queries'], 0)
            self.assertEqual(result['total_queries'], 3)
        finally:
            torch.set_num_threads(old_threads)

    def test_visdrone_recipe_matches_oracle_budget_and_positive_ids(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(str(root / 'configs/dfine/pred_refinement/dfine_hgnetv2_l_visdrone_pred_p2.yml'), {})
        self.assertEqual(config['model'], 'DFINEPredRefiner')
        self.assertEqual(config['DFINEPredRefiner']['sampling_mode'], 'distribution')
        self.assertEqual(config['num_classes'], 10)
        self.assertEqual(config['epochs'], 12)
        self.assertEqual(config['train_dataloader']['total_batch_size'], 96)
        self.assertEqual(config['val_dataloader']['total_batch_size'], 96)
        self.assertIn('positive_ids', config['val_dataloader']['dataset']['ann_file'])
        self.assertTrue(config['use_amp'])
        self.assertFalse(config['use_ema'])
        self.assertFalse(config['sync_bn'])


if __name__ == '__main__':
    unittest.main()
