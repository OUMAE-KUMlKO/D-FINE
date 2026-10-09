"""Box-relative candidate bands retain support points and legacy checkpoints."""

from types import SimpleNamespace
import unittest

import torch

from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.pred_strip_head import PredictionStripHead
from src.zoo.dfine.strip_probability_head import StripProbabilityHead
from test_all_layer_strip import make_model, matcher
from test_gt_refinement import normalized_targets


class StripAdaptiveBandTests(unittest.TestCase):
    @staticmethod
    def geometry():
        # Input dimensions are H=160, W=320. Widths/heights exercise both clamps.
        boxes = torch.tensor([
            [0.45, 0.45, 4 / 320, 48 / 160],
            [0.55, 0.50, 16 / 320, 4 / 160],
            [0.50, 0.55, 64 / 320, 24 / 160],
        ], requires_grad=True)
        edges = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2), -1).detach()
        positions = edges[..., None] + torch.tensor([-0.01, 0.0, 0.01])
        return boxes, edges, positions

    @staticmethod
    def head(**kwargs):
        return StripProbabilityHead(
            2, 3, hidden_dim=4, normal_points=7, normal_kernel_size=3,
            offset_pixels=8.0, use_log_prior=False, query_fusion='channel_gate',
            **kwargs,
        )

    def test_grid_uses_each_box_width_height_and_clamps_in_input_pixels(self):
        # Exercise the shared grid directly: this test must detect old fixed bands.
        head = SimpleNamespace(
            num_points=5, normal_points=7, offset_pixels=8.0,
            bandwidth_mode='box_relative', bandwidth_ratio=0.25,
            min_offset_pixels=2.0,
        )
        boxes, edges, positions = self.geometry()
        grid, valid, candidates = PredictionStripHead._candidate_grid(
            head, positions, boxes, torch.ones(3, dtype=torch.bool), (160, 320),
        )
        self.assertEqual(grid.shape, (3, 4, 3, 5, 7, 2))
        self.assertTrue(valid.all())
        self.assertTrue(candidates.all())
        self.assertFalse(grid.requires_grad)
        coordinates = (grid + 1) / 2
        radii = ((2, 8), (4, 2), (8, 6))
        for row in range(3):
            for edge in range(4):
                axis = edge % 2
                radius = radii[row][axis]
                sign = 1 if edge < 2 else -1
                observed = (
                    coordinates[row, edge, ..., axis]
                    - positions[row, edge, :, None, None]
                ) * (320, 160)[axis]
                expected = sign * torch.linspace(radius, -radius, 7).expand(3, 5, 7)
                torch.testing.assert_close(observed, expected, atol=2e-5, rtol=0)
                # All candidate centers and the five tangential positions stay fixed.
                torch.testing.assert_close(
                    coordinates[row, edge, :, :, 3, axis],
                    positions[row, edge, :, None].expand(3, 5), atol=1e-7, rtol=0,
                )
                start, end = ((edges[row, 1], edges[row, 3]) if axis == 0
                              else (edges[row, 0], edges[row, 2]))
                tangent = torch.linspace(start, end, 5)[None, :, None].expand(3, 5, 7)
                torch.testing.assert_close(coordinates[row, edge, ..., 1 - axis], tangent)

    def test_fixed_grid_and_default_small_offset_remain_exactly_compatible(self):
        default = StripProbabilityHead(2, 3, hidden_dim=4)
        fixed = StripProbabilityHead(
            2, 3, hidden_dim=4, bandwidth_mode='fixed', min_offset_pixels=2.0,
        )
        fixed.load_state_dict(default.state_dict(), strict=True)
        self.assertEqual(default.bandwidth_mode, 'fixed')
        boxes, _, positions = self.geometry()
        args = (positions, boxes, torch.ones(3, dtype=torch.bool), (160, 320))
        legacy = SimpleNamespace(num_points=5, offset_pixels=1.0)
        expected = PredictionStripHead._candidate_grid(legacy, *args)
        for head in (default, fixed):
            for actual, old in zip(PredictionStripHead._candidate_grid(head, *args), expected):
                torch.testing.assert_close(actual, old, atol=0, rtol=0)

    def test_bandwidth_is_configuration_only_and_keeps_strict_state_loading(self):
        fixed = self.head()
        adaptive = self.head(bandwidth_mode='box_relative')
        adaptive.load_state_dict(fixed.state_dict(), strict=True)
        fixed.load_state_dict(adaptive.state_dict(), strict=True)
        self.assertEqual(list(fixed.state_dict()), list(adaptive.state_dict()))
        self.assertEqual(sum(p.numel() for p in fixed.parameters()),
                         sum(p.numel() for p in adaptive.parameters()))
        self.assertEqual(adaptive.bandwidth_ratio, 0.25)
        self.assertEqual(adaptive.min_offset_pixels, 2.0)

    def test_invalid_adaptive_configuration_is_rejected(self):
        invalid = [({'bandwidth_mode': 'unknown'}, 'bandwidth_mode')]
        for value in (0, -1, float('nan'), float('inf')):
            invalid.append(({'bandwidth_ratio': value}, 'bandwidth_ratio'))
            invalid.append(({'min_offset_pixels': value}, 'min_offset_pixels'))
        invalid.append(({'min_offset_pixels': 9}, 'min_offset_pixels'))
        for options, message in invalid:
            options = {'bandwidth_mode': 'box_relative', **options}
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                self.head(**options)

    def test_both_wrappers_forward_band_options_and_preserve_zero_init(self):
        torch.manual_seed(41)
        images = torch.randn(1, 3, 16, 24)
        common = dict(normal_points=7, normal_kernel_size=3, offset_pixels=8.0,
                      use_log_prior=False, query_fusion='channel_gate')
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                fixed = make_model(frozen=frozen, **common).eval()
                adaptive = make_model(
                    frozen=frozen, bandwidth_mode='box_relative',
                    bandwidth_ratio=0.3, min_offset_pixels=1.5, **common,
                ).eval()
                adaptive.load_state_dict(fixed.state_dict(), strict=True)
                for head in adaptive.refiner.heads:
                    self.assertEqual(head.bandwidth_mode, 'box_relative')
                    self.assertEqual(head.bandwidth_ratio, 0.3)
                    self.assertEqual(head.min_offset_pixels, 1.5)
                    self.assertEqual(head.offset_pixels, 8.0)
                with torch.no_grad():
                    first, second = fixed(images), adaptive(images)
                for key in ('pred_boxes', 'pred_logits'):
                    torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_checkpointed_training_reaches_all_six_visual_branches(self):
        torch.manual_seed(41)
        model = make_model(
            normal_points=7, normal_kernel_size=3, offset_pixels=8.0,
            use_log_prior=False, query_fusion='channel_gate',
            bandwidth_mode='box_relative',
        ).train()
        # The residual output is initially zero. Simulate its first learned update
        # so upstream visual encoders/query gates receive nonzero gradients.
        with torch.no_grad():
            for head in model.refiner.heads:
                head.score_head[-1].weight.normal_(std=0.02)
        calls = []
        hooks = [head.register_forward_pre_hook(
            lambda module, args, i=i: calls.append((i, args[1].shape[1])),
        ) for i, head in enumerate(model.refiner.heads)]
        try:
            output = model(torch.randn(1, 3, 64, 96), normalized_targets())
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(calls, [(i, 3) for i in range(6)])  # DN queries skip bands.
        self.assertEqual(len(output['aux_outputs']), 5)
        self.assertEqual(len(output['dn_outputs']), 6)
        criterion = DFINECriterion(
            matcher(), {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,
                        'loss_fgl': 0.15, 'loss_ddf': 1.5},
            ['vfl', 'boxes', 'local'], num_classes=2, reg_max=4,
        )
        sum(criterion(output, normalized_targets()).values()).backward()
        for i, head in enumerate(model.refiner.heads):
            for parameter in (head.strip_encoder[0].weight, head.projection.weight,
                              head.query_projection.weight):
                self.assertIsNotNone(parameter.grad, i)
                self.assertTrue(torch.isfinite(parameter.grad).all(), i)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, i)
        for module in (model.backbone, model.encoder, model.decoder):
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in gradients))
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gradients))


if __name__ == '__main__':
    unittest.main()
