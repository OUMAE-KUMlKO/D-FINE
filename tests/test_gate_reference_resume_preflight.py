"""Full-state resume preflight preserves native scheduler and metadata semantics."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from src.optim.ema import ModelEMA
from src.optim.warmup import LinearWarmup
from tools.preflight_visdrone_gate_reference_resume import (
    restore_components, step_metadata, loader_overrides, execution_metadata,
)


def components():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500], gamma=.1)
    warmup = LinearWarmup(scheduler, warmup_duration=2)
    return {'model': model, 'criterion': torch.nn.MSELoss(),
            'postprocessor': torch.nn.Identity(), 'ema': ModelEMA(model, warmups=0),
            'scaler': torch.cuda.amp.GradScaler(enabled=False), 'optimizer': optimizer,
            'lr_scheduler': scheduler, 'lr_warmup_scheduler': warmup}


class ResumePreflightTests(unittest.TestCase):
    def test_distributed_loader_and_actual_bn_metadata_match_execution(self):
        self.assertEqual(loader_overrides(8)['train_dataloader'], {'total_batch_size': 32})
        self.assertEqual(loader_overrides(1)['train_dataloader'],
                         {'total_batch_size': 4, 'num_workers': 4})
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3))
        plain = execution_metadata(model, world_size=1, configured_sync_bn=True)
        self.assertFalse(plain['sync_bn'])
        self.assertEqual(plain['batch_norm_modules'], 1)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        synced = execution_metadata(model, world_size=8, configured_sync_bn=True)
        self.assertTrue(synced['sync_bn'])
        self.assertEqual(synced['sync_bn_modules'], 1)
        self.assertEqual(synced['world_size'], 8)

    def test_full_restore_loads_unwrapped_model_keys(self):
        source = components()
        state = {key: value.state_dict() for key, value in source.items()}
        state['last_epoch'] = 91
        target = components()
        target['model'] = torch.nn.DataParallel(target['model'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pth'
            torch.save(state, path)
            report = restore_components(target, path, SimpleNamespace(yaml_cfg={}))
        self.assertEqual(report['start_epoch'], 92)
        for key, value in source['model'].state_dict().items():
            torch.testing.assert_close(target['model'].module.state_dict()[key], value, atol=0, rtol=0)

    def test_restores_all_state_and_reuses_native_epoch92_lr_schedule(self):
        before = components()
        before['model'](torch.ones(1, 2)).sum().backward()
        before['optimizer'].step()
        before['lr_warmup_scheduler'].step()
        before['lr_warmup_scheduler'].step()
        before['ema'].update(before['model'])
        state = {key: value.state_dict() for key, value in before.items()}
        state['last_epoch'] = 91
        cfg = SimpleNamespace(resume='last.pth', yaml_cfg={
            'resume_lr_schedule': True,
            'lr_scheduler': {'type': 'MultiStepLR', 'milestones': [100], 'gamma': .1}})
        restored = components()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pth'
            torch.save(state, path)
            report = restore_components(restored, path, cfg)
        self.assertEqual(report['start_epoch'], 92)
        self.assertEqual(set(report['restored_keys']), set(before) | {'last_epoch'})
        self.assertEqual(restored['lr_scheduler'].last_epoch, 92)
        self.assertEqual(dict(restored['lr_scheduler'].milestones), {100: 1})
        self.assertIs(restored['lr_warmup_scheduler'].lr_scheduler, restored['lr_scheduler'])
        self.assertIs(restored['lr_scheduler'].optimizer, restored['optimizer'])
        self.assertEqual(restored['optimizer'].param_groups[0]['lr'], .01)
        self.assertEqual(restored['ema'].updates, before['ema'].updates)
        self.assertTrue(restored['optimizer'].state)
        for key, value in before['model'].state_dict().items():
            torch.testing.assert_close(restored['model'].state_dict()[key], value, atol=0, rtol=0)
        for key, value in before['ema'].module.state_dict().items():
            torch.testing.assert_close(restored['ema'].module.state_dict()[key], value, atol=0, rtol=0)

    def test_missing_optimizer_is_rejected_before_loading_model(self):
        original = components()
        copied = copy.deepcopy(original['model'].state_dict())
        state = {key: value.state_dict() for key, value in original.items() if key != 'optimizer'}
        state['last_epoch'] = 91
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'partial.pth'
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, 'optimizer'):
                restore_components(original, path, SimpleNamespace())
        for key, value in original['model'].state_dict().items():
            torch.testing.assert_close(value, copied[key], atol=0, rtol=0)

    def test_step_metadata_uses_original_eight_gpu_epoch_length(self):
        self.assertEqual(step_metadata(92, 1, dataset_size=6471, drop_last=True),
                         {'epoch': 92, 'step': 1, 'global_step': 92 * 202 + 1,
                          'epoch_step': 202})
        self.assertEqual(step_metadata(0, 0, dataset_size=6471, drop_last=False)['epoch_step'], 203)


if __name__ == '__main__':
    unittest.main()
