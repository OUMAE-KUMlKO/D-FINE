"""Focused CPU checks for prediction-guided strip sampling."""

import unittest

import torch

from src.zoo.dfine.pred_strip_head import PredictionStripHead


class PredictionStripHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.features = torch.randn(1, 2, 8, 10)
        self.queries = torch.randn(2, 3)
        self.boxes = torch.tensor([[0.5, 0.5, 0.4, 0.5], [0.6, 0.45, 0.3, 0.2]])
        self.batch_indices = torch.zeros(2, dtype=torch.long)

    def make_head(self):
        return PredictionStripHead(2, 3, hidden_dim=2, num_points=3)

    def test_zero_initialization_retains_original_boxes_and_gradient_path(self):
        head = self.make_head()
        boxes = self.boxes.clone().requires_grad_()
        refined, valid = head(
            self.features, self.queries, boxes, self.batch_indices, (40, 80)
        )
        self.assertTrue(valid.all())
        torch.testing.assert_close(refined, boxes, rtol=0, atol=0)
        refined.sum().backward()
        torch.testing.assert_close(boxes.grad, torch.ones_like(boxes))
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
        ))

    def test_nonzero_residual_refines_valid_boxes_and_invalid_sampling_falls_back(self):
        head = self.make_head()
        boxes = self.boxes.clone()
        boxes[1] = torch.tensor([0.1, 0.5, 0.2, 0.4])
        delta = torch.tensor([0.2, -0.3, 0.1, -0.2])
        with torch.no_grad():
            head.refiner[-1].bias.copy_(delta)
        refined, valid = head(
            self.features, self.queries, boxes, self.batch_indices, (40, 80)
        )
        self.assertEqual(valid.tolist(), [True, False])
        expected = torch.cat((
            boxes[0, :2] + boxes[0, 2:] * delta[:2],
            boxes[0, 2:] * delta[2:].exp(),
        ))
        torch.testing.assert_close(refined[0], expected)
        torch.testing.assert_close(refined[1], boxes[1], rtol=0, atol=0)

    def test_prediction_controls_edges_spans_and_input_pixel_offsets(self):
        head = self.make_head()
        with torch.no_grad():
            head.projection.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
            head.projection.bias.zero_()
        y, x = torch.meshgrid(
            (torch.arange(8) + 0.5) / 8,
            (torch.arange(10) + 0.5) / 10,
            indexing="ij",
        )
        ramp = torch.stack((x, y))[None]
        boxes = self.boxes.clone().requires_grad_()
        sampled = []
        hook = head.strip_encoder.register_forward_pre_hook(
            lambda module, args: sampled.append(args[0])
        )
        try:
            _, valid = head(ramp, self.queries, boxes, self.batch_indices, (40, 80))
        finally:
            hook.remove()
        self.assertTrue(valid.all())
        strips = sampled[0].reshape(2, 4, 2, 3, 3)
        # Both examples change the edge position and its full tangent span.
        for index, (left, top, right, bottom) in enumerate((
            (0.3, 0.25, 0.7, 0.75), (0.45, 0.35, 0.75, 0.55),
        )):
            along_x = torch.linspace(left, right, 3)[:, None].expand(3, 3)
            along_y = torch.linspace(top, bottom, 3)[:, None].expand(3, 3)
            expected_columns = (
                [left + 1 / 80, left, left - 1 / 80],
                [top + 1 / 40, top, top - 1 / 40],
                [right - 1 / 80, right, right + 1 / 80],
                [bottom - 1 / 40, bottom, bottom + 1 / 40],
            )
            for edge in range(4):
                normal_channel = edge % 2
                expected_normal = torch.tensor(expected_columns[edge]).expand(3, 3)
                torch.testing.assert_close(strips[index, edge, normal_channel], expected_normal)
                torch.testing.assert_close(
                    strips[index, edge, 1 - normal_channel],
                    along_y if edge % 2 == 0 else along_x,
                )
        sampled[0].sum().backward()
        self.assertIsNone(boxes.grad, "Sampling coordinates must be detached from predictions.")

    def test_p1_accepts_only_box_sampling_with_real_features(self):
        head = PredictionStripHead(2, 3, sampling_mode="box")
        self.assertEqual(head.feature_mode, "real")
        for mode in ("distribution", "unknown"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "box"):
                PredictionStripHead(2, 3, sampling_mode=mode)


if __name__ == "__main__":
    unittest.main()
