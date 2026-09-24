"""Focused checks for boundary-aligned two-dimensional candidate bands."""

from types import SimpleNamespace
import unittest

import torch

from src.zoo.dfine.dfine_decoder import Integral
from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function
from src.zoo.dfine.pred_strip_head import PredictionStripHead
from src.zoo.dfine.strip_probability_head import StripProbabilityHead


torch.set_num_threads(1)


class StripProbabilityBandTests(unittest.TestCase):
    def make_head(self, **kwargs):
        return StripProbabilityHead(
            2, 3, hidden_dim=4, num_points=5, normal_points=7,
            kernel_size=3, normal_kernel_size=3, offset_pixels=8,
            query_chunk_size=1, **kwargs,
        )

    @staticmethod
    def edges(boxes):
        return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2), dim=-1)

    def test_variable_boxes_keep_five_by_seven_bands_and_inward_to_outward_order(self):
        head = self.make_head()
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.5], [0.6, 0.45, 0.2, 0.3]])
        edges = self.edges(boxes)
        positions = edges[..., None]
        grid, point_valid, candidate_valid = PredictionStripHead._candidate_grid(
            head, positions, boxes, torch.ones(2, dtype=torch.bool), (160, 320),
        )
        self.assertEqual(tuple(grid.shape), (2, 4, 1, 5, 7, 2))
        self.assertTrue(point_valid.all())
        self.assertTrue(candidate_valid.all())
        coordinates = (grid + 1) / 2
        normal_offsets = torch.linspace(8, -8, 7).expand(5, 7)
        for row in range(2):
            for edge in range(4):
                axis = edge % 2
                scale = (320, 160)[axis]
                observed = (coordinates[row, edge, 0, ..., axis] - edges[row, edge]) * scale
                sign = 1 if edge < 2 else -1
                torch.testing.assert_close(observed, sign * normal_offsets, atol=2e-5, rtol=0)
                start, end = (edges[row, 1], edges[row, 3]) if axis == 0 else (edges[row, 0], edges[row, 2])
                tangent = torch.linspace(start, end, 5)[:, None].expand(5, 7)
                torch.testing.assert_close(coordinates[row, edge, 0, ..., 1 - axis], tangent)

    def test_encoder_and_fusion_use_the_normal_dimension(self):
        head = self.make_head()
        self.assertEqual(head.strip_encoder[0].kernel_size, (3, 3))
        self.assertEqual(head.strip_encoder[0].padding, (1, 1))
        self.assertEqual(head.strip_encoder[0].groups, 4)
        self.assertEqual(head.strip_encoder[2].kernel_size, (1, 1))
        self.assertEqual(head.edge_fusion[0].in_features, 7 * 4)
        self.assertEqual(head.score_head[0].in_features, 2 * 4 + 8 + 2)

    def test_default_shapes_and_legacy_grid_users_keep_three_columns(self):
        head = StripProbabilityHead(2, 3, hidden_dim=4)
        explicit = StripProbabilityHead(2, 3, hidden_dim=4, normal_points=3, normal_kernel_size=1)
        explicit.load_state_dict(head.state_dict(), strict=True)
        self.assertEqual(tuple(head.strip_encoder[0].weight.shape), (4, 1, 3, 1))
        self.assertEqual(tuple(head.edge_fusion[0].weight.shape), (4, 12))
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.5]])
        edges = self.edges(boxes)
        legacy = SimpleNamespace(num_points=5, offset_pixels=1.0)
        args = (edges[..., None], boxes, torch.ones(1, dtype=torch.bool), (160, 320))
        actual = PredictionStripHead._candidate_grid(head, *args)
        old = PredictionStripHead._candidate_grid(legacy, *args)
        for value, expected in zip(actual, old):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        left = (actual[0][0, 0, 0, 0, :, 0] + 1) / 2
        expected = edges[0, 0] + torch.tensor([1.0, 0.0, -1.0]) / 320
        torch.testing.assert_close(left, expected)

    def test_band_zero_initialization_and_trainable_visual_gradients(self):
        torch.manual_seed(23)
        head = self.make_head(detach_inputs=False)
        features = torch.rand(1, 2, 12, 20, requires_grad=True)
        queries = torch.rand(1, 2, 3, requires_grad=True)
        distribution = {
            "pred_corners": (torch.randn(1, 2, 4 * 33) * 0.2).requires_grad_(),
            "ref_points": torch.tensor([[[0.5, 0.5, 0.3, 0.3], [0.4, 0.6, 0.2, 0.2]]], requires_grad=True),
            "up": torch.tensor([0.5]), "reg_scale": torch.tensor([4.0]),
        }
        support = weighting_function(32, distribution["up"], distribution["reg_scale"])
        boxes = distance2bbox(
            distribution["ref_points"], Integral(32)(distribution["pred_corners"], support),
            distribution["reg_scale"],
        ).detach().requires_grad_()
        sampled_shapes = []
        hook = head.strip_encoder.register_forward_pre_hook(
            lambda module, args: sampled_shapes.append(tuple(args[0].shape)),
        )
        try:
            result = head(features, queries, boxes, (160, 320), distribution)
        finally:
            hook.remove()
        self.assertTrue(result["refinement_mask"].all())
        self.assertTrue(all(shape[-2:] == (5, 7) for shape in sampled_shapes))
        torch.testing.assert_close(result["pred_boxes"], boxes, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result["calibrated_corners"], distribution["pred_corners"].reshape(1, 2, 4, 33), atol=0, rtol=0)
        result["pred_boxes"].square().sum().backward()
        self.assertGreater(float(head.score_head[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        head.zero_grad(set_to_none=True)
        for value in (features, queries, distribution["pred_corners"]):
            value.grad = None
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.fill_(0.1)
        result = head(features, queries, boxes, (160, 320), distribution)
        (-result["calibrated_corners"].log_softmax(-1)[..., 16].sum()).backward()
        for parameter in (head.projection.weight, head.strip_encoder[0].weight,
                          head.strip_encoder[2].weight, head.edge_fusion[0].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        for value in (features, queries, distribution["pred_corners"]):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)
        self.assertIsNone(boxes.grad)
        self.assertIsNone(distribution["ref_points"].grad)

    def test_normal_dimensions_reject_noninteger_or_even_values(self):
        for value in (1, 2, 4, 3.0):
            with self.subTest(normal_points=value), self.assertRaisesRegex(ValueError, "normal_points"):
                StripProbabilityHead(2, 3, normal_points=value)
        for value in (0, -1, 2, 3.0):
            with self.subTest(normal_kernel_size=value), self.assertRaisesRegex(ValueError, "normal_kernel_size"):
                StripProbabilityHead(2, 3, normal_kernel_size=value)


if __name__ == "__main__":
    unittest.main()
