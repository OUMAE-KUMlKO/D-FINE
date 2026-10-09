"""Read-only observation of the production per-layer distribution refiner."""

import unittest
from unittest import mock

import torch

from test_all_layer_strip import make_model
from tools.analysis.all_layer_strip_trace import trace_all_layer_strip


class AllLayerStripTraceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        self.images = torch.randn(1, 3, 64, 96)

    def make_model(self, mode):
        return make_model(
            normal_points=7, normal_kernel_size=3, offset_pixels=8,
            use_log_prior=False, query_fusion='channel_gate',
            bandwidth_mode=mode,
        ).eval()

    def test_trace_preserves_outputs_and_records_nonzero_local_effects(self):
        model = self.make_model('box_relative')
        with torch.no_grad():
            for head in model.refiner.heads:
                head.score_head[-1].weight.normal_(std=0.05)
            expected = model(self.images)
        traced = trace_all_layer_strip(model, self.images)
        self.assertEqual(list(traced['layers']), list(range(6)))
        for key in ('pred_boxes', 'pred_logits'):
            torch.testing.assert_close(traced['outputs'][key], expected[key], atol=0, rtol=0)
        torch.testing.assert_close(traced['layers'][5]['boxes_after'], expected['pred_boxes'])
        for record in traced['layers'].values():
            self.assertEqual(record['boxes_before'].shape, (1, 3, 4))
            self.assertEqual(record['valid'].shape, (1, 3))
            self.assertFalse(record['boxes_before'].requires_grad)
            expected_band = (record['boxes_before'][..., 2:] * torch.tensor([96, 64]) * .25).clamp(2, 8)
            torch.testing.assert_close(record['half_widths_px'], expected_band)
        self.assertTrue(any(not torch.equal(x['boxes_before'], x['boxes_after'])
                            for x in traced['layers'].values()))
        self.assertTrue(all(not h._forward_hooks for h in model.refiner.heads))

    def test_zero_initialized_fixed_heads_have_no_local_movement(self):
        model = self.make_model('fixed')
        for record in trace_all_layer_strip(model, self.images)['layers'].values():
            torch.testing.assert_close(record['boxes_before'], record['boxes_after'], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(record['half_widths_px'], torch.full((1, 3, 2), 8.0))

    def test_hooks_are_removed_after_forward_failure(self):
        model = self.make_model('fixed')
        with mock.patch.object(model.decoder, 'forward', side_effect=RuntimeError('expected failure')):
            with self.assertRaisesRegex(RuntimeError, 'expected failure'):
                trace_all_layer_strip(model, self.images)
        self.assertTrue(all(not h._forward_hooks for h in model.refiner.heads))

    def test_training_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'eval'):
            trace_all_layer_strip(self.make_model('fixed').train(), self.images)


if __name__ == '__main__':
    unittest.main()
