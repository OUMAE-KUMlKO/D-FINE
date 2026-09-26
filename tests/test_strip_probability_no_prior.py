"""The no-prior-input ablation keeps the original additive logit path."""

import unittest

import torch

from src.zoo.dfine.dfine_decoder import Integral
from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function
from src.zoo.dfine.strip_probability_head import StripProbabilityHead
from test_all_layer_strip import make_model


class StripProbabilityNoPriorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.features = torch.rand(1, 2, 10, 12)
        self.queries = torch.rand(1, 2, 3)
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

    def forward_head(self, head, distribution=None):
        return head(self.features, self.queries, self.boxes, (160, 320),
                    self.distribution if distribution is None else distribution)

    def test_nonzero_delta_is_prior_independent_but_additive_prior_gradient_is_one(self):
        head = StripProbabilityHead(
            2, 3, hidden_dim=8, normal_points=7, normal_kernel_size=3,
            offset_pixels=8, detach_inputs=False, use_log_prior=False,
        )
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.fill_(0.1)
        other = dict(self.distribution)
        other["pred_corners"] = self.distribution["pred_corners"].detach() + torch.linspace(-2, 2, 20)
        deltas = []
        hook = head.score_head.register_forward_hook(
            lambda module, args, output: deltas.append(output),
        )
        try:
            first, second = self.forward_head(head), self.forward_head(head, other)
        finally:
            hook.remove()
        self.assertGreater(float(deltas[0].abs().sum()), 0)
        torch.testing.assert_close(deltas[0], deltas[1], atol=0, rtol=0)
        for key in ("candidate_positions", "candidate_valid", "input_valid"):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
        expected_change = (other["pred_corners"] - self.distribution["pred_corners"]).reshape(1, 2, 4, 5)
        torch.testing.assert_close(
            second["calibrated_corners"] - first["calibrated_corners"],
            expected_change, atol=1e-6, rtol=1e-6,
        )
        first["calibrated_corners"].sum().backward()
        torch.testing.assert_close(
            self.distribution["pred_corners"].grad,
            torch.ones_like(self.distribution["pred_corners"]), atol=0, rtol=0,
        )

    def test_default_state_shapes_and_outputs_remain_compatible(self):
        default = StripProbabilityHead(2, 3)
        explicit = StripProbabilityHead(2, 3, use_log_prior=True)
        without = StripProbabilityHead(2, 3, use_log_prior=False)
        self.assertTrue(default.use_log_prior)
        self.assertEqual(tuple(default.score_head[0].weight.shape), (64, 138))
        self.assertEqual(tuple(without.score_head[0].weight.shape), (64, 137))
        with torch.no_grad():
            default.score_head[-1].weight.fill_(0.1)
        explicit.load_state_dict(default.state_dict(), strict=True)
        first, second = self.forward_head(default), self.forward_head(explicit)
        for key in ("pred_boxes", "calibrated_corners", "candidate_positions", "candidate_valid"):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_disabled_prior_input_preserves_zero_initialization_and_trainable_residual(self):
        head = StripProbabilityHead(2, 3, hidden_dim=8, use_log_prior=False)
        result = self.forward_head(head)
        self.assertTrue(result["refinement_mask"].all())
        torch.testing.assert_close(result["pred_boxes"], self.boxes, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            result["calibrated_corners"], self.distribution["pred_corners"].reshape(1, 2, 4, 5),
            atol=0, rtol=0,
        )
        result["pred_boxes"].square().sum().backward()
        self.assertGreater(float(head.score_head[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertIsNone(self.distribution["pred_corners"].grad)

    def test_both_wrappers_pass_the_switch_to_all_six_band_heads(self):
        for frozen in (True, False):
            with self.subTest(frozen=frozen):
                model = make_model(
                    frozen=frozen, normal_points=7, normal_kernel_size=3,
                    offset_pixels=8.0, use_log_prior=False,
                )
                self.assertEqual(len(model.refiner.heads), 6)
                for head in model.refiner.heads:
                    self.assertFalse(head.use_log_prior)
                    self.assertEqual(head.score_head[0].in_features, 2 * 8 + 8 + 1)
                    self.assertEqual(head.normal_points, 7)
                    self.assertEqual(head.strip_encoder[0].kernel_size, (3, 3))

    def test_switch_requires_a_boolean(self):
        with self.assertRaisesRegex(ValueError, "use_log_prior"):
            StripProbabilityHead(2, 3, use_log_prior="false")


if __name__ == "__main__":
    unittest.main()
