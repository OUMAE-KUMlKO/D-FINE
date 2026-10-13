"""Bounded diagnostic loading and train-mode replay preserve checkpoint state."""

from pathlib import Path
import contextlib
import io
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tools.diagnose_visdrone_prediction_safety import (
    load_model_weights, parse_args, capture_state, restore_state, forward_probe,
)


class PredictionSafetyDiagnosticTests(unittest.TestCase):
    def test_normal_boundary_mask_is_logged_but_decode_fallback_stops_probe(self):
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = SimpleNamespace(decoder=SimpleNamespace(last_prediction_safety_stats={}))
                self.refiner = SimpleNamespace(last_prediction_safety_stats={
                    'layer_0_head_input_invalid_queries': 2,
                    'layer_0_head_decode_fallback_queries': 0})

            def forward(self, images, targets=None):
                assert self.training and not torch.is_grad_enabled()
                return {'pred_boxes': torch.tensor([[[.5, .5, .2, .2]]]),
                        'pred_logits': torch.zeros(1, 1, 2)}

        model = Detector().train()
        report, diagnostic = forward_probe(model, torch.zeros(1, 3, 4, 4), [], amp=False)
        self.assertEqual(report['status'], 'valid')
        self.assertIsNone(diagnostic)
        model.refiner.last_prediction_safety_stats['layer_0_head_decode_fallback_queries'] = 1
        report, diagnostic = forward_probe(model, torch.zeros(1, 3, 4, 4), [], amp=False)
        self.assertEqual(report['status'], 'guard_fallback')
        self.assertEqual(diagnostic['stage'], 'guard_fallback')

    def test_loads_model_not_ema_and_removes_duplicate_ddp_names(self):
        model = torch.nn.Linear(2, 1)
        expected = {key: torch.full_like(value, 3.) for key, value in model.state_dict().items()}
        duplicated = {**expected, **{'module.' + key: value for key, value in expected.items()}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'NaN.pth'
            torch.save({'model': duplicated,
                        'ema': {'module': {key: torch.zeros_like(value)
                                           for key, value in expected.items()}},
                        'last_epoch': 91}, path)
            metadata = load_model_weights(model, path)
        self.assertEqual(metadata['weights'], 'model')
        self.assertEqual(metadata['duplicate_keys_removed'], 2)
        self.assertEqual(metadata['checkpoint_last_epoch'], 91)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)

    def test_restore_reproduces_rng_and_train_mode_buffers_without_parameter_updates(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(.5)).train()
        samples = torch.arange(12.).reshape(4, 3)
        original_buffers = {key: value.clone() for key, value in model.named_buffers()}
        original_parameters = {key: value.clone() for key, value in model.named_parameters()}
        state = capture_state(model)
        with torch.no_grad():
            first = model(samples)
        first_random = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(int(model[0].num_batches_tracked), 1)
        restore_state(model, state)
        for key, value in model.named_buffers():
            torch.testing.assert_close(value, original_buffers[key], atol=0, rtol=0)
        with torch.no_grad():
            second = model(samples)
        second_random = (random.random(), np.random.rand(), torch.rand(3))
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertEqual(first_random[:2], second_random[:2])
        torch.testing.assert_close(first_random[2], second_random[2], atol=0, rtol=0)
        for key, value in model.named_parameters():
            torch.testing.assert_close(value, original_parameters[key], atol=0, rtol=0)
        restore_state(model, state)

    def test_cli_limits_scan_and_never_exposes_training_options(self):
        required = ['--config', 'frozen.yml', '--checkpoint', 'NaN.pth',
                    '--output-dir', 'fresh-output']
        args = parse_args(required)
        self.assertEqual((args.max_batches, args.max_seconds, args.epoch, args.seed),
                         (16, 180., 92, 0))
        self.assertFalse(args.fp32_replay)
        for extra in (['--max-batches', '17'], ['--max-batches', '0'],
                      ['--max-seconds', '181'], ['--max-seconds', 'nan'],
                      ['--resume', 'last.pth']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(required + extra)


if __name__ == '__main__':
    unittest.main()
