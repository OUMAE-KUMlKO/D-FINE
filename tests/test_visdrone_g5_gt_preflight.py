"""Disposable G5 oracle updates must use the actual model's parameters."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

import test_gt_all_layer_strip
from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.optim.ema import ModelEMA
from test_all_layer_strip import matcher


class G5GTPreflightTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.analysis.check_visdrone_g5_gt'))
        from tools.analysis import check_visdrone_g5_gt
        return check_visdrone_g5_gt

    def model(self):
        fixture = test_gt_all_layer_strip.GTAllLayerStripTests()
        fixture.setUp()
        return fixture.guided(fixture.plain())

    def test_strict_loading_requires_complete_ema_and_reports_epoch(self):
        api = self.api()
        model = self.model()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pth'
            torch.save({'ema': {'module': model.state_dict()}, 'last_epoch': 96, 'epoch': 3}, path)
            report = api.load_strict_ema(model, path)
            self.assertEqual(report['epoch'], 96)
            self.assertEqual(report['source'], 'ema.module')
            self.assertEqual(report['keys_loaded'], len(model.state_dict()))
            state = dict(model.state_dict())
            state.pop(next(iter(state)))
            torch.save({'ema': {'module': state}, 'epoch': 96}, path)
            with self.assertRaises(RuntimeError):
                api.load_strict_ema(model, path)
            torch.save({'model': model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, 'EMA'):
                api.load_strict_ema(model, path)

    def test_two_updates_train_every_group_and_reject_wrong_optimizer(self):
        api = self.api()
        model = self.model()
        criterion = DFINECriterion(
            matcher(), {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,
                        'loss_fgl': .15, 'loss_ddf': 1.5},
            ['vfl', 'boxes', 'local'], num_classes=2, reg_max=4)
        images = torch.randn(1, 3, 16, 24)
        targets = [{'boxes': torch.tensor([[7., 4., 14., 10.]]), 'labels': torch.tensor([0])}]
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        report = api.check_batch(model, criterion, optimizer, images, targets,
                                 use_amp=False, clip_max_norm=.1)
        self.assertEqual(report['successful_updates'], 2)
        self.assertTrue(report['empty_gt_matches_normal'])
        self.assertGreater(report['gt_guidance_max_box_change'], 0)
        self.assertEqual(set(report['updated_groups']),
                         {'backbone', 'encoder', 'decoder'} | {f'refiner.{i}' for i in range(6)})
        self.assertTrue(all(report['updated_groups'].values()))
        for step in report['steps']:
            self.assertTrue(step['gradients_finite'])
            self.assertTrue(all(g['nonzero'] for g in step['gradient_groups'].values()))
            self.assertTrue(all(g['with_grad'] == g['parameters']
                                for g in step['gradient_groups'].values()))
        other = self.model()
        wrong = torch.optim.AdamW(other.parameters(), lr=1e-4)
        with self.assertRaisesRegex(AssertionError, 'exactly'):
            api.assert_optimizer_partition(model, wrong)

    def test_fresh_zero_scores_train_on_separate_batch_and_update_gt_ema(self):
        api = self.api()
        model = self.model()
        with torch.no_grad():
            for head in model.refiner.heads:
                head.score_head[-1].weight.zero_()
                head.score_head[-1].bias.zero_()
        ema = ModelEMA(model, warmups=1)
        criterion = DFINECriterion(
            matcher(), {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,
                        'loss_fgl': .15, 'loss_ddf': 1.5},
            ['vfl', 'boxes', 'local'], num_classes=2, reg_max=4)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        val_images = torch.randn(1, 3, 16, 24)
        raw_targets = [{'boxes': torch.tensor([[7., 4., 14., 10.]]), 'labels': torch.tensor([0])}]
        train_images = torch.randn(1, 3, 20, 28)
        train_targets = [{'boxes': torch.tensor([[.4, .4, .2, .3]]), 'labels': torch.tensor([1])}]
        report = api.check_batch(
            model, criterion, optimizer, val_images, raw_targets,
            use_amp=False, clip_max_norm=.1, fresh_init=True,
            train_images=train_images, train_targets=train_targets, ema=ema)
        self.assertEqual(report['successful_updates'], 2)
        self.assertEqual(report['training_image_size'], [20, 28])
        self.assertEqual(report['gt_guidance_max_box_change'], 0)
        self.assertTrue(report['fresh_zero_score_identity'])
        self.assertGreater(report['post_update_gt_guidance_max_box_change'], 0)
        self.assertTrue(all(report['updated_groups'].values()))
        self.assertTrue(all(group['nonzero'] for group in report['steps'][-1]['gradient_groups'].values()))
        self.assertEqual(report['ema_updates'], 2)
        self.assertTrue(report['ema_gt_eval_passed'])
        self.assertTrue(report['ema_checked'])
        self.assertTrue(ema.module.requires_gt)
        self.assertFalse(ema.module.supports_gt_refinement_metrics)


if __name__ == '__main__':
    unittest.main()
