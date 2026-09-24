"""Focused CPU checks for full-support strip probability calibration."""

import unittest

import torch

from src.zoo.dfine.dfine_decoder import Integral
from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function
from src.zoo.dfine.strip_probability_head import StripProbabilityHead


class StripProbabilityHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.features = torch.rand(1, 2, 8, 10, requires_grad=True)
        self.queries = torch.rand(1, 2, 3, requires_grad=True)
        self.distribution = {
            "pred_corners": (torch.randn(1, 2, 20) * 0.2).requires_grad_(),
            "ref_points": torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.45, 0.55, 0.3, 0.2]]], requires_grad=True),
            "up": torch.tensor([0.5], requires_grad=True),
            "reg_scale": torch.tensor([4.0], requires_grad=True),
        }
        self.boxes = self.decode(self.distribution["pred_corners"]).detach().requires_grad_()

    def make_head(self, detach_inputs=True):
        return StripProbabilityHead(
            2, 3, hidden_dim=8, num_points=3, query_chunk_size=1,
            detach_inputs=detach_inputs,
        )

    def decode(self, logits):
        support = weighting_function(4, self.distribution["up"], self.distribution["reg_scale"])
        return distance2bbox(
            self.distribution["ref_points"], Integral(4)(logits.reshape(1, 2, 20), support),
            self.distribution["reg_scale"],
        )

    def forward_head(self, head):
        return head(self.features, self.queries, self.boxes, (40, 80), self.distribution)

    def test_zero_initialization_preserves_full_prior_and_box_gradient(self):
        head = self.make_head()
        result = self.forward_head(head)
        self.assertTrue(result["input_valid"].all())
        self.assertTrue(result["refinement_mask"].all())
        self.assertFalse(result["candidate_valid"].all())
        torch.testing.assert_close(result["pred_boxes"], self.boxes, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result["calibrated_corners"], self.distribution["pred_corners"].reshape(1, 2, 4, 5), atol=0, rtol=0)
        self.assertTrue((result["calibrated_corners"].softmax(-1)[~result["candidate_valid"]] > 0).all())
        result["pred_boxes"].square().sum().backward()
        self.assertGreater(head.score_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        for value in (self.features, self.queries, self.boxes, *self.distribution.values()):
            self.assertIsNone(value.grad)
        self.assertFalse(result["candidate_positions"].requires_grad)

    def test_residual_changes_only_supported_candidates_then_decodes_all_bins(self):
        head = self.make_head()
        with torch.no_grad():
            head.score_head[-1].bias.fill_(0.4)
        result = self.forward_head(head)
        prior = self.distribution["pred_corners"].detach().reshape(1, 2, 4, 5)
        expected = prior + result["candidate_valid"].float() * 0.4
        torch.testing.assert_close(result["calibrated_corners"], expected)
        torch.testing.assert_close(result["calibrated_corners"][~result["candidate_valid"]], prior[~result["candidate_valid"]], atol=0, rtol=0)
        torch.testing.assert_close(result["pred_boxes"], self.decode(expected), atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(result["pred_boxes"], self.boxes))

    def test_mlp_reads_true_normalized_offsets_and_log_prior_per_candidate(self):
        head = self.make_head()
        captured = []
        hook = head.score_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            result = self.forward_head(head)
        finally:
            hook.remove()
        inputs = captured[0].reshape(1, 2, 4, 5, -1)
        self.assertEqual(inputs.shape[-1], 2 * 8 + 8 + 2)
        boxes = self.boxes.detach()
        edges = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1)
        scale = boxes[..., 2:].repeat(1, 1, 2)
        expected_offset = (result["candidate_positions"] - edges[..., None]) / scale[..., None]
        torch.testing.assert_close(inputs[..., -2], expected_offset)
        torch.testing.assert_close(inputs[..., -1], self.distribution["pred_corners"].detach().reshape(1, 2, 4, 5).log_softmax(-1))
        self.assertEqual(head.edge_embedding.embedding_dim, 8)

    def test_invalid_inputs_fall_back_without_nan_gradients(self):
        for kind in ("query", "box", "logits", "refs", "no_samples", "features", "scale"):
            with self.subTest(kind=kind):
                head = self.make_head()
                features, queries, boxes = (value.detach().clone() for value in (self.features, self.queries, self.boxes))
                distribution = {name: value.detach().clone() for name, value in self.distribution.items()}
                if kind == "query":
                    queries[0, 0, 0] = float("nan")
                elif kind == "box":
                    boxes[0, 0, 2] = 0
                elif kind == "logits":
                    distribution["pred_corners"][0, 0, 0] = float("nan")
                elif kind == "refs":
                    distribution["ref_points"][0, 0, 0] = float("nan")
                elif kind == "no_samples":
                    distribution["ref_points"][0, 0, :2] = 4
                elif kind == "features":
                    features[0, 0, 0, 0] = float("nan")
                else:
                    distribution["reg_scale"].zero_()
                result = head(features, queries, boxes, (40, 80), distribution)
                self.assertFalse(result["input_valid"][0, 0])
                self.assertFalse(result["refinement_mask"][0, 0])
                torch.testing.assert_close(result["pred_boxes"][0, 0], boxes[0, 0], atol=0, rtol=0)
                self.assertTrue(torch.isfinite(result["calibrated_corners"]).all())
                (result["pred_boxes"].sum() + result["calibrated_corners"].sum() * 0).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_nonpositive_decoded_box_falls_back_but_distribution_keeps_gradient(self):
        head = self.make_head()
        with torch.no_grad():
            self.distribution["pred_corners"].copy_(torch.tensor([-10.0, 0, 0, 0, 0]).repeat(1, 2, 4))
            self.boxes.copy_(self.decode(self.distribution["pred_corners"]))
            head.score_head[0].weight.zero_()
            head.score_head[0].bias.zero_()
            head.score_head[0].weight[0, -1] = -1
            head.score_head[-1].weight.zero_()
            head.score_head[-1].weight[0, 0] = 2
        result = self.forward_head(head)
        self.assertTrue(result["input_valid"].all())
        self.assertFalse(result["refinement_mask"].any())
        torch.testing.assert_close(result["pred_boxes"], self.boxes, atol=0, rtol=0)
        (-result["calibrated_corners"].log_softmax(-1)[..., 2].sum()).backward()
        self.assertGreater(head.score_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(head.score_head[-1].weight.grad).all())

    def test_empty_queries_connect_every_parameter_to_zero_loss(self):
        head = self.make_head()
        distribution = dict(self.distribution)
        distribution["pred_corners"] = self.distribution["pred_corners"][:, :0]
        distribution["ref_points"] = self.distribution["ref_points"][:, :0]
        result = head(self.features, self.queries[:, :0], self.boxes[:, :0], (40, 80), distribution)
        self.assertEqual(tuple(result["pred_boxes"].shape), (1, 0, 4))
        self.assertEqual(tuple(result["calibrated_corners"].shape), (1, 0, 4, 5))
        (result["pred_boxes"].sum() + result["calibrated_corners"].sum()).backward()
        self.assertTrue(all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in head.parameters()))

    def test_visual_encoding_receives_gradients_after_zero_initialization(self):
        head = self.make_head()
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.fill_(0.1)
        result = self.forward_head(head)
        (-result["calibrated_corners"].log_softmax(-1)[..., 2].sum()).backward()
        self.assertGreater(head.projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.strip_encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.query_projection.weight.grad.abs().sum().item(), 0)

    def test_end_to_end_mode_backpropagates_through_features_query_and_prior(self):
        head = self.make_head(detach_inputs=False)
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.fill_(0.1)
        result = self.forward_head(head)

        (-result["calibrated_corners"].log_softmax(-1)[..., 2].sum()).backward()

        for value in (self.features, self.queries, self.distribution["pred_corners"]):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.0)
        # Sampling geometry remains fixed even during end-to-end training.
        for value in (
            self.boxes, self.distribution["ref_points"],
            self.distribution["up"], self.distribution["reg_scale"],
        ):
            self.assertIsNone(value.grad)

    def test_end_to_end_zero_init_preserves_direct_prior_gradient(self):
        head = self.make_head(detach_inputs=False)
        result = self.forward_head(head)
        prior = self.distribution["pred_corners"].reshape(1, 2, 4, 5)
        torch.testing.assert_close(result["calibrated_corners"], prior)

        result["calibrated_corners"].sum().backward()

        torch.testing.assert_close(
            self.distribution["pred_corners"].grad,
            torch.ones_like(self.distribution["pred_corners"]),
        )
        self.assertIsNone(self.distribution["ref_points"].grad)


if __name__ == "__main__":
    unittest.main()
