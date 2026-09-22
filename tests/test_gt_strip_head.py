"""Small deterministic checks for the GT-only strip oracle (no training)."""

import importlib.util
from pathlib import Path
import unittest

import torch


# Load the standalone module without importing datasets/backbones and their
# optional dependencies, so these geometry checks also run on CPU installations.
_MODULE_PATH = Path(__file__).resolve().parents[1] / "src/zoo/dfine/gt_strip_head.py"
_SPEC = importlib.util.spec_from_file_location("gt_strip_head", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
GTStripHead = _MODULE.GTStripHead


class GTStripHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.features = torch.randn(2, 2, 8, 8)
        self.boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.6, 0.5, 0.3, 0.3]])
        self.queries = torch.randn(2, 3)
        self.batch_indices = torch.tensor([0, 1])

    def make_head(self, mode="real"):
        return GTStripHead(2, 3, hidden_dim=4, num_points=3, feature_mode=mode)

    def test_gt_edges_pixel_offsets_and_inside_outside_order(self):
        head = GTStripHead(2, 3, hidden_dim=2, num_points=3)
        with torch.no_grad():
            head.projection.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
            head.projection.bias.zero_()
        y, x = torch.meshgrid((torch.arange(8) + 0.5) / 8, (torch.arange(8) + 0.5) / 8, indexing="ij")
        ramp = torch.stack((x, y))[None]
        gt = torch.tensor([[0.5, 0.5, 0.5, 0.5]], requires_grad=True)
        strips, mask, valid = head.sample_strips(ramp, gt, torch.tensor([0]), (8, 8))
        self.assertTrue(valid.all())
        self.assertTrue(mask.all())
        torch.testing.assert_close(strips[0, 0, 0], torch.tensor([[0.375, 0.25, 0.125]]).expand(3, 3))
        torch.testing.assert_close(strips[0, 0, 1], torch.tensor([[0.25], [0.5], [0.75]]).expand(3, 3))
        torch.testing.assert_close(strips[0, 1, 1], torch.tensor([[0.375, 0.25, 0.125]]).expand(3, 3))
        torch.testing.assert_close(strips[0, 2, 0], torch.tensor([[0.625, 0.75, 0.875]]).expand(3, 3))
        torch.testing.assert_close(strips[0, 3, 1], torch.tensor([[0.625, 0.75, 0.875]]).expand(3, 3))
        strips.sum().backward()
        self.assertIsNone(gt.grad, "GT coordinates must not create a regression gradient path.")

    def test_zero_initialization_recovers_original_boxes(self):
        for mode in ("real", "constant", "random"):
            head = self.make_head(mode)
            refined, valid = head(self.features, self.queries, self.boxes, self.boxes, self.batch_indices, (32, 32))
            self.assertTrue(valid.all())
            torch.testing.assert_close(refined, self.boxes, rtol=0, atol=0)
            refined.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_sampling_ignores_predicted_box_and_query(self):
        head = self.make_head()
        sampled_inputs = []
        hook = head.strip_encoder.register_forward_pre_hook(lambda module, args: sampled_inputs.append(args[0].detach().clone()))
        head(self.features, self.queries, self.boxes, self.boxes, self.batch_indices, (32, 32))
        head(self.features, self.queries + 2, self.boxes * 0.75, self.boxes, self.batch_indices, (32, 32))
        hook.remove()
        torch.testing.assert_close(sampled_inputs[0], sampled_inputs[1], rtol=0, atol=0)

    def test_constant_control_does_not_pass_gt_geometry_to_regressor(self):
        head = self.make_head("constant")
        with torch.no_grad():
            head.refiner[-1].weight.normal_()
        first_gt = self.boxes
        second_gt = torch.tensor([[0.3, 0.35, 0.1, 0.2], [0.7, 0.65, 0.2, 0.1]])
        first, first_valid = head(self.features, self.queries, self.boxes, first_gt, self.batch_indices, (32, 32))
        second, second_valid = head(self.features + 100, self.queries, self.boxes, second_gt, self.batch_indices, (32, 32))
        self.assertTrue(first_valid.all() and second_valid.all())
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_all_modes_have_same_eligibility_and_invalid_queries_fall_back(self):
        gt = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.1, 0.5, 0.2, 0.4], [float("nan"), 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])
        pred = torch.tensor([[0.5, 0.5, 0.3, 0.3]]).expand(4, 4).clone()
        pred[-1, 2] = 0
        for mode in ("real", "constant", "random"):
            head = self.make_head(mode)
            with torch.no_grad():
                head.refiner[-1].bias.fill_(0.2)
            refined, valid = head(self.features, torch.ones(4, 3), pred, gt, torch.tensor([0, 0, 1, 1]), (32, 32))
            self.assertEqual(valid.tolist(), [True, False, False, False])
            torch.testing.assert_close(refined[1:], pred[1:], rtol=0, atol=0)
            self.assertFalse(torch.equal(refined[0], pred[0]))
            refined[valid].sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_empty_batch_has_full_parameter_gradient_connection(self):
        head = self.make_head()
        refined, valid = head(self.features, torch.empty(0, 3), torch.empty(0, 4), torch.empty(0, 4), torch.empty(0, dtype=torch.long), (32, 32))
        self.assertEqual(refined.shape, (0, 4))
        self.assertEqual(valid.shape, (0,))
        refined.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in head.parameters()))

    def test_random_fields_are_shared_and_evaluation_uses_image_ids(self):
        head = self.make_head("random").eval()
        gt = self.boxes[:1].expand(3, 4)
        batch = torch.tensor([0, 0, 1])
        strips, _, _ = head.sample_strips(self.features, gt, batch, (32, 32), image_ids=[11, 27])
        torch.testing.assert_close(strips[0], strips[1], rtol=0, atol=0)
        again, _, _ = head.sample_strips(self.features + 100, gt, batch, (32, 32), image_ids=[11, 27])
        torch.testing.assert_close(strips, again, rtol=0, atol=0)
        reordered, _, _ = head.sample_strips(self.features.flip(0), gt, 1 - batch, (32, 32), image_ids=[27, 11])
        torch.testing.assert_close(strips, reordered, rtol=0, atol=0)
        self.assertFalse(torch.equal(strips[0], strips[2]))
        head.train()
        first, _, _ = head.sample_strips(self.features, gt, batch, (32, 32))
        second, _, _ = head.sample_strips(self.features, gt, batch, (32, 32))
        self.assertFalse(torch.equal(first, second))

    def test_real_visual_path_receives_gradients_after_zero_init_is_released(self):
        head = self.make_head()
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.fill_(0.1)
        features = torch.ones_like(self.features, requires_grad=True)
        gt = self.boxes.clone().requires_grad_()
        refined, valid = head(features, torch.ones_like(self.queries), self.boxes, gt, self.batch_indices, (32, 32))
        refined[valid].sum().backward()
        self.assertGreater(torch.count_nonzero(features.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(head.projection.weight.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(head.strip_encoder[0].weight.grad).item(), 0)
        self.assertIsNone(gt.grad)

    def test_cpu_autocast_preserves_float32_geometry(self):
        head = self.make_head()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            refined, valid = head(self.features.bfloat16(), self.queries.bfloat16(), self.boxes, self.boxes, self.batch_indices, (32, 32))
        self.assertEqual(refined.dtype, torch.float32)
        self.assertTrue(valid.all())
        torch.testing.assert_close(refined, self.boxes, rtol=0, atol=0)
        refined.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_scale_clamp_stays_positive_and_scales_with_original_box(self):
        head = self.make_head()
        with torch.no_grad():
            head.refiner[-1].bias.copy_(torch.tensor([0.2, -0.3, 100.0, -100.0]))
        refined, valid = head(self.features, self.queries, self.boxes, self.boxes, self.batch_indices, (32, 32))
        torch.testing.assert_close(refined[:, :2], self.boxes[:, :2] + self.boxes[:, 2:] * torch.tensor([0.2, -0.3]))
        torch.testing.assert_close(refined[:, 2:], self.boxes[:, 2:] * torch.tensor([2.0, -2.0]).exp())
        self.assertTrue(valid.all() and (refined[:, 2:] > 0).all())


if __name__ == "__main__":
    unittest.main()
