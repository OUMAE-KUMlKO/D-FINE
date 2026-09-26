"""Query channel gates modulate candidate features without a query concatenation."""

import unittest

import torch

from src.zoo.dfine.dfine_decoder import Integral
from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function
from src.zoo.dfine.strip_probability_head import StripProbabilityHead
from test_all_layer_strip import make_model


class StripProbabilityQueryGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.features = torch.rand(1, 2, 12, 20, requires_grad=True)
        self.queries = torch.tensor([[[0.2, 0.4, 0.6], [0.7, 0.5, 0.1]]], requires_grad=True)
        self.distribution = {
            "pred_corners": (torch.randn(1, 2, 20) * 0.2).requires_grad_(),
            "ref_points": torch.tensor([[[0.5, 0.5, 0.3, 0.3], [0.4, 0.6, 0.2, 0.2]]]),
            "up": torch.tensor([0.5]), "reg_scale": torch.tensor([4.0]),
        }
        support = weighting_function(4, self.distribution["up"], self.distribution["reg_scale"])
        self.boxes = distance2bbox(
            self.distribution["ref_points"], Integral(4)(self.distribution["pred_corners"], support),
            self.distribution["reg_scale"],
        ).detach()

    def make_head(self, use_log_prior=False):
        return StripProbabilityHead(
            2, 3, hidden_dim=4, normal_points=7, normal_kernel_size=3,
            offset_pixels=8, query_chunk_size=1, detach_inputs=False,
            use_log_prior=use_log_prior, query_fusion="channel_gate",
        )

    def forward_head(self, head, queries=None):
        return head(self.features, self.queries if queries is None else queries,
                    self.boxes, (160, 320), self.distribution)

    def capture(self, head, queries=None):
        encoded, inputs = [], []
        hooks = [
            head.edge_fusion.register_forward_hook(lambda module, args, output: encoded.append(output)),
            head.score_head.register_forward_pre_hook(lambda module, args: inputs.append(args[0])),
        ]
        try:
            result = self.forward_head(head, queries)
        finally:
            for hook in hooks:
                hook.remove()
        candidates = torch.cat(encoded, dim=0).reshape(2, 4, 5, head.hidden_dim)
        return result, candidates, inputs[0]

    def test_initial_gate_is_identity_and_preserves_complete_prior(self):
        for use_log_prior in (False, True):
            with self.subTest(use_log_prior=use_log_prior):
                head = self.make_head(use_log_prior)
                self.assertEqual(torch.count_nonzero(head.query_projection.weight).item(), 0)
                self.assertEqual(torch.count_nonzero(head.query_projection.bias).item(), 0)
                result, candidates, inputs = self.capture(head)
                self.assertEqual(inputs.shape[-1], 4 + 8 + 1 + int(use_log_prior))
                torch.testing.assert_close(inputs[..., :4], candidates, atol=0, rtol=0)
                prior = self.distribution["pred_corners"].reshape(1, 2, 4, 5)
                torch.testing.assert_close(result["calibrated_corners"], prior, atol=0, rtol=0)
                torch.testing.assert_close(result["pred_boxes"], self.boxes, atol=1e-6, rtol=1e-5)
                if use_log_prior:
                    torch.testing.assert_close(inputs[..., -1], prior[0].log_softmax(-1))
                self.distribution["pred_corners"].grad = None
                result["calibrated_corners"].sum().backward()
                torch.testing.assert_close(self.distribution["pred_corners"].grad,
                                           torch.ones_like(self.distribution["pred_corners"]), atol=0, rtol=0)

    def test_nonzero_gate_multiplies_each_channel_and_is_the_only_query_path(self):
        head = self.make_head()
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.fill_(0.1)
            head.query_projection.weight.copy_(torch.tensor(
                [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-0.5, 0.25, 1.0]],
            ))
            head.query_projection.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))
        result, candidates, inputs = self.capture(head)
        gate = 1 + head.query_projection(self.queries.reshape(2, 3)).tanh()
        self.assertGreater(float(candidates.abs().sum()), 0)
        self.assertFalse(torch.allclose(gate[0], gate[1]))
        torch.testing.assert_close(inputs[..., :4], candidates * gate[:, None, None, :])
        self.assertEqual(inputs.shape[-1], 13)
        expected_delta = head.score_head(inputs).squeeze(-1)
        mask = result["candidate_valid"][0] & result["input_valid"][0, :, None, None]
        self.assertGreater(float(expected_delta.abs().sum()), 0)
        expected = self.distribution["pred_corners"].reshape(1, 2, 4, 5) + torch.where(mask, expected_delta, 0)[None]
        torch.testing.assert_close(result["calibrated_corners"], expected)

        # Zero visual features remove the query contribution: there is neither
        # a query concatenation nor an additive query-conditioned feature bias.
        with torch.no_grad():
            head.edge_fusion[0].weight.zero_()
            head.edge_fusion[0].bias.zero_()
        first, _, first_inputs = self.capture(head)
        second, _, second_inputs = self.capture(head, self.queries + 2)
        torch.testing.assert_close(first_inputs, second_inputs, atol=0, rtol=0)
        torch.testing.assert_close(first["calibrated_corners"], second["calibrated_corners"], atol=0, rtol=0)

    def test_gate_and_visual_paths_receive_nonzero_finite_gradients(self):
        head = self.make_head()
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.fill_(0.1)
        result = self.forward_head(head)
        result["calibrated_corners"].sum().backward()
        for parameter in (
            head.query_projection.weight, head.query_projection.bias,
            head.projection.weight, head.strip_encoder[0].weight,
            head.strip_encoder[2].weight, head.edge_fusion[0].weight,
        ):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        for value in (self.features, self.queries):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_default_concat_keeps_initialization_shapes_and_strict_state_loading(self):
        torch.manual_seed(31)
        default = StripProbabilityHead(2, 3)
        torch.manual_seed(31)
        explicit = StripProbabilityHead(2, 3, query_fusion="concat")
        self.assertEqual(default.query_fusion, "concat")
        self.assertEqual(tuple(default.score_head[0].weight.shape), (64, 138))
        self.assertGreater(torch.count_nonzero(default.query_projection.weight).item(), 0)
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], atol=0, rtol=0)
        with torch.no_grad():
            default.score_head[-1].weight.fill_(0.1)
        explicit.load_state_dict(default.state_dict(), strict=True)
        first, second = self.forward_head(default), self.forward_head(explicit)
        for key in ("pred_boxes", "calibrated_corners", "candidate_positions", "candidate_valid"):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
        gate = StripProbabilityHead(2, 3, query_fusion="channel_gate", use_log_prior=False)
        self.assertEqual(tuple(gate.score_head[0].weight.shape), (64, 73))

    def test_both_wrappers_pass_channel_gate_to_all_six_heads(self):
        for frozen in (True, False):
            with self.subTest(frozen=frozen):
                model = make_model(
                    frozen=frozen, normal_points=7, normal_kernel_size=3,
                    offset_pixels=8.0, use_log_prior=False, query_fusion="channel_gate",
                )
                self.assertEqual(len(model.refiner.heads), 6)
                for head in model.refiner.heads:
                    self.assertEqual(head.query_fusion, "channel_gate")
                    self.assertEqual(head.score_head[0].in_features, 8 + 8 + 1)
                    self.assertEqual(torch.count_nonzero(head.query_projection.weight).item(), 0)
                    self.assertEqual(torch.count_nonzero(head.query_projection.bias).item(), 0)

    def test_unknown_query_fusion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "query_fusion"):
            StripProbabilityHead(2, 3, query_fusion="add")


if __name__ == "__main__":
    unittest.main()
