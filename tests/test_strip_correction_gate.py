"""Adaptive per-edge gates for strip probability corrections."""

import unittest

import torch
import torch.nn as nn

from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.strip_probability_head import StripProbabilityHead
from test_all_layer_strip import make_model, matcher
from test_gt_refinement import normalized_targets


torch.set_num_threads(1)


class _FixedGate(nn.Module):
    def forward(self, context):
        gates = context.new_tensor([0.5, 1.0, 1.5, 1.75])
        raw = torch.atanh(gates - 1.0)
        return raw[None, :, None].expand(context.shape[0], -1, -1)


class StripCorrectionGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.features = torch.rand(1, 2, 8, 10, requires_grad=True)
        self.queries = torch.rand(1, 2, 3, requires_grad=True)
        self.distribution = {
            "pred_corners": (torch.randn(1, 2, 20) * 0.2).requires_grad_(),
            "ref_points": torch.tensor([
                [[0.5, 0.5, 0.4, 0.4], [0.45, 0.55, 0.3, 0.2]],
            ]),
            "up": torch.tensor([0.5]),
            "reg_scale": torch.tensor([4.0]),
        }
        from src.zoo.dfine.dfine_decoder import Integral
        from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function
        support = weighting_function(4, self.distribution["up"], self.distribution["reg_scale"])
        self.boxes = distance2bbox(
            self.distribution["ref_points"],
            Integral(4)(self.distribution["pred_corners"], support),
            self.distribution["reg_scale"],
        ).detach()

    @staticmethod
    def make_head(**kwargs):
        return StripProbabilityHead(
            2, 3, hidden_dim=8, num_points=3, query_chunk_size=1,
            detach_inputs=False, **kwargs,
        )

    def forward_head(self, head):
        return head(
            self.features, self.queries, self.boxes, (40, 80), self.distribution,
        )

    def test_disabled_gate_keeps_state_initialization_rng_and_nonzero_delta_exact(self):
        torch.manual_seed(101)
        disabled = self.make_head()
        after_disabled = torch.rand(5)
        torch.manual_seed(101)
        enabled = self.make_head(adaptive_correction_gate=True)
        after_enabled = torch.rand(5)

        self.assertFalse(disabled.adaptive_correction_gate)
        self.assertTrue(enabled.adaptive_correction_gate)
        self.assertFalse(any(name.startswith("correction_gate.") for name in disabled.state_dict()))
        self.assertEqual(torch.count_nonzero(enabled.correction_gate[-1].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(enabled.correction_gate[-1].bias).item(), 0)
        for name, value in disabled.state_dict().items():
            torch.testing.assert_close(enabled.state_dict()[name], value, atol=0, rtol=0)
        torch.testing.assert_close(after_enabled, after_disabled, atol=0, rtol=0)

        with torch.no_grad():
            disabled.score_head[-1].bias.fill_(0.4)
            enabled.score_head[-1].bias.fill_(0.4)
        first, second = self.forward_head(disabled), self.forward_head(enabled)
        self.assertGreater(float((first["calibrated_corners"] -
                                  self.distribution["pred_corners"].reshape(1, 2, 4, 5)).abs().sum()), 0)
        for key in ("pred_boxes", "calibrated_corners"):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_each_edge_gate_scales_only_valid_candidate_deltas(self):
        head = self.make_head(adaptive_correction_gate=True)
        with torch.no_grad():
            head.score_head[-1].weight.zero_()
            head.score_head[-1].bias.fill_(0.4)
        head.correction_gate = _FixedGate()
        result = self.forward_head(head)

        prior = self.distribution["pred_corners"].reshape(1, 2, 4, 5)
        valid = result["candidate_valid"] & result["input_valid"][..., None, None]
        gates = prior.new_tensor([0.5, 1.0, 1.5, 1.75])[None, None, :, None]
        expected = prior + torch.where(valid, 0.4 * gates, torch.zeros_like(prior))
        torch.testing.assert_close(result["calibrated_corners"], expected)
        torch.testing.assert_close(
            result["calibrated_corners"][~valid], prior[~valid], atol=0, rtol=0,
        )

    def test_gate_context_uses_masked_encoded_query_edge_and_delta_statistics(self):
        head = self.make_head(adaptive_correction_gate=True)
        with torch.no_grad():
            head.score_head[-1].weight.normal_(std=0.05)
        encoded, deltas, contexts = [], [], []
        hooks = [
            head.edge_fusion.register_forward_hook(
                lambda module, args, output: encoded.append(output.detach()),
            ),
            head.score_head.register_forward_hook(
                lambda module, args, output: deltas.append(output.detach()),
            ),
            head.correction_gate.register_forward_pre_hook(
                lambda module, args: contexts.append(args[0].detach()),
            ),
        ]
        try:
            result = self.forward_head(head)
        finally:
            for hook in hooks:
                hook.remove()

        candidates = torch.cat(encoded, 0).reshape(2, 4, 5, head.hidden_dim)
        delta = deltas[0].squeeze(-1).reshape(2, 4, 5)
        valid = (result["candidate_valid"] & result["input_valid"][..., None, None]).reshape(2, 4, 5)
        weight = valid[..., None].to(candidates)
        count = weight.sum(2).clamp_min(1)
        mean = (candidates * weight).sum(2) / count
        delta_mean = torch.where(valid, delta, 0).sum(2, keepdim=True) / count
        delta_mad = torch.where(valid, (delta - delta_mean).abs(), 0).sum(2, keepdim=True) / count
        expected = torch.cat((
            mean,
            head.query_projection(self.queries.reshape(2, 3)).detach()[:, None].expand(2, 4, -1),
            head.edge_embedding(torch.arange(4)).detach()[None].expand(2, -1, -1),
            delta_mean,
            delta_mad,
        ), -1)
        self.assertEqual(tuple(contexts[0].shape), (2, 4, 2 * head.hidden_dim + 10))
        torch.testing.assert_close(contexts[0], expected, atol=0, rtol=0)

    def test_real_gate_receives_finite_gradient(self):
        head = self.make_head(adaptive_correction_gate=True)
        with torch.no_grad():
            head.score_head[-1].weight.normal_(std=0.05)
        result = self.forward_head(head)
        result["calibrated_corners"].square().sum().backward()
        for parameter in head.correction_gate[-1].parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_small_model_training_backward_reaches_all_six_gates(self):
        torch.manual_seed(41)
        model = make_model(
            adaptive_correction_gate=True, use_log_prior=False,
            query_fusion="channel_gate", normal_points=7, normal_kernel_size=3,
            offset_pixels=8.0,
        ).train()
        with torch.no_grad():
            for head in model.refiner.heads:
                head.score_head[-1].weight.normal_(std=0.02)
        output = model(torch.randn(1, 3, 32, 48), normalized_targets())
        criterion = DFINECriterion(
            matcher(), {"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2,
                        "loss_fgl": 0.15, "loss_ddf": 1.5},
            ["vfl", "boxes", "local"], num_classes=2, reg_max=4,
        )
        sum(criterion(output, normalized_targets()).values()).backward()
        for index, head in enumerate(model.refiner.heads):
            gradient = head.correction_gate[-1].weight.grad
            self.assertIsNotNone(gradient, index)
            self.assertTrue(torch.isfinite(gradient).all(), index)
            self.assertGreater(float(gradient.abs().sum()), 0, index)

    def test_adaptive_correction_gate_requires_boolean(self):
        with self.assertRaisesRegex(ValueError, "adaptive_correction_gate"):
            self.make_head(adaptive_correction_gate=1)


if __name__ == "__main__":
    unittest.main()
