"""Focused checks for the isolated continuation-training protocol."""
import importlib.util
from pathlib import Path
import unittest
import torch
from torch import nn


class TrainingProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / 'tools/train_sparse_strip_query.py'
        spec = importlib.util.spec_from_file_location('sparse_training', path)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_parameter_groups_cover_trainable_parameters_once(self):
        model = nn.Module()
        model.backbone = nn.Linear(2, 2).requires_grad_(False)
        model.decoder = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
        model.sparse_refiner = nn.Linear(2, 2)
        groups = self.runner.optimizer_groups(model)
        by_id = {id(p): g for g in groups for p in g['params']}
        self.assertEqual(sum(len(g['params']) for g in groups), len(by_id))
        self.assertEqual(set(by_id), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual(by_id[id(model.decoder[0].weight)]['lr'], 2.5e-5)
        self.assertEqual(by_id[id(model.sparse_refiner.weight)]['lr'], 1e-3)
        self.assertEqual(by_id[id(model.decoder[1].weight)]['weight_decay'], 0)

    def test_scaler_skips_do_not_update_ema(self):
        class Scaler:
            scale = 8
            def get_scale(self): return self.scale
            def step(self, optimizer): pass
            def update(self): self.scale /= 2
        class EMA:
            updates = 0
            def update(self, model): self.updates += 1
        ema = EMA()
        self.assertFalse(self.runner.scaled_step(Scaler(), None, ema, None))
        self.assertEqual(ema.updates, 0)
        scaler = Scaler()
        scaler.update = lambda: None
        self.assertTrue(self.runner.scaled_step(scaler, None, ema, None))
        self.assertEqual(ema.updates, 1)

    def test_raw_nonfinite_loss_is_rejected_before_sanitization(self):
        class Criterion:
            weight_dict = {'loss_bbox': 5}
            def get_loss(self, *args, **kwargs): return {'loss_bbox': torch.tensor(float('nan'))}
        criterion = Criterion()
        self.runner.guard_raw_losses(criterion)
        with self.assertRaisesRegex(FloatingPointError, 'loss_bbox'):
            criterion.get_loss()

    def test_disabled_ddf_raw_nan_is_not_an_active_loss(self):
        class Criterion:
            weight_dict = {'loss_ddf': 0, 'loss_bbox': 5}
            def get_loss(self):
                return {'loss_ddf': torch.tensor(float('nan')), 'loss_bbox': torch.tensor(1.)}
        criterion = Criterion()
        self.runner.guard_raw_losses(criterion)
        self.assertEqual(criterion.get_loss()['loss_bbox'].item(), 1.)


if __name__ == '__main__':
    unittest.main()
