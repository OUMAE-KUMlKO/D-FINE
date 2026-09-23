import copy
import unittest

import torch

from src.zoo.dfine.dfine_utils import weighting_function
from src.zoo.dfine.pred_strip_head import PredictionStripHead


torch.set_num_threads(1)


def make_distribution(n=1, k=5):
    return {
        "pred_corners": torch.zeros(n, 4 * k),
        "ref_points": torch.tensor([[0.5, 0.5, 0.4, 0.2]]).repeat(n, 1),
        "up": torch.tensor([0.5]),
        "reg_scale": torch.tensor([4.0]),
    }


class PredictionDistributionStripTests(unittest.TestCase):
    def make_head(self, chunk=32):
        return PredictionStripHead(
            2, 3, hidden_dim=2, num_points=3,
            sampling_mode="distribution", query_chunk_size=chunk,
        )

    def test_candidate_geometry_preserves_bin_direction_and_decoder_support(self):
        head = self.make_head()
        distribution = make_distribution(k=5)
        positions, probabilities, valid = head._distribution_geometry(
            distribution, 1, torch.device("cpu")
        )
        support = weighting_function(4, distribution["up"], distribution["reg_scale"])
        refs = distribution["ref_points"][0]
        expected = torch.stack((
            refs[0] - refs[2] / 2 - support * refs[2] / 4,
            refs[1] - refs[3] / 2 - support * refs[3] / 4,
            refs[0] + refs[2] / 2 + support * refs[2] / 4,
            refs[1] + refs[3] / 2 + support * refs[3] / 4,
        ))
        torch.testing.assert_close(positions[0], expected)
        self.assertTrue(valid.all())
        torch.testing.assert_close(probabilities, torch.full((1, 4, 5), 0.2))
        self.assertTrue(torch.all(positions[0, 0, 1:] <= positions[0, 0, :-1]))
        self.assertTrue(torch.all(positions[0, 2, 1:] >= positions[0, 2, :-1]))

    def test_masked_softmax_selects_single_candidate_and_reports_excluded_mass(self):
        logits = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float16)
        candidate_valid = torch.tensor([[[False, True, False]]])
        weights, probabilities, excluded = PredictionStripHead._masked_candidate_weights(
            logits, candidate_valid
        )
        torch.testing.assert_close(weights, torch.tensor([[[0.0, 1.0, 0.0]]]))
        torch.testing.assert_close(probabilities, logits.float().softmax(-1))
        torch.testing.assert_close(
            excluded, probabilities[..., 0] + probabilities[..., 2]
        )
        self.assertEqual(weights.dtype, torch.float32)

    def test_single_valid_center_candidate_matches_box_mode_and_far_refs_fall_back(self):
        box_head = PredictionStripHead(
            2, 3, hidden_dim=2, num_points=3, sampling_mode="box"
        )
        distribution_head = self.make_head()
        distribution_head.load_state_dict(copy.deepcopy(box_head.state_dict()))
        with torch.no_grad():
            box_head.refiner[-1].weight.normal_(std=0.1)
            box_head.refiner[-1].bias.copy_(torch.tensor([0.1, -0.1, 0.05, -0.05]))
            distribution_head.refiner[-1].weight.copy_(box_head.refiner[-1].weight)
            distribution_head.refiner[-1].bias.copy_(box_head.refiner[-1].bias)
        features = torch.randn(1, 2, 8, 8)
        queries = torch.randn(1, 3)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        batches = torch.tensor([0])
        distribution = make_distribution(k=5)
        distribution["ref_points"] = boxes.clone()
        distribution["up"] = torch.tensor([5.0])

        box_refined, box_valid = box_head(
            features, queries, boxes, batches, (32, 32)
        )
        dist_refined, dist_valid, diagnostics = distribution_head(
            features, queries, boxes, batches, (32, 32), distribution, True
        )

        self.assertTrue(box_valid.all() and dist_valid.all())
        self.assertEqual(diagnostics["candidate_valid"].sum(dim=-1).tolist(), [[1, 1, 1, 1]])
        torch.testing.assert_close(dist_refined, box_refined)

        far_distribution = {key: value.clone() for key, value in distribution.items()}
        far_distribution["ref_points"][:, :2] = 10.0
        fallback, valid = distribution_head(
            features, queries, boxes, batches, (32, 32), far_distribution
        )
        self.assertFalse(valid.any())
        torch.testing.assert_close(fallback, boxes)

    def test_invalid_distribution_falls_back_and_diagnostics_are_finite(self):
        head = self.make_head()
        features = torch.randn(1, 2, 8, 8)
        queries = torch.randn(1, 3)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        distribution = make_distribution(k=5)
        distribution["pred_corners"][0, 0] = float("nan")

        refined, valid, diagnostics = head(
            features, queries, boxes, torch.tensor([0]), (32, 32),
            distribution=distribution, return_diagnostics=True,
        )

        self.assertFalse(valid.any())
        torch.testing.assert_close(refined, boxes)
        self.assertTrue(torch.isfinite(diagnostics["candidate_positions"]).all())
        self.assertTrue(torch.isfinite(diagnostics["candidate_probabilities"]).all())
        self.assertFalse(diagnostics["candidate_valid"].any())
        self.assertTrue(torch.isfinite(diagnostics["excluded_probability_mass"]).all())

    def test_empty_query_batch_keeps_parameter_connection(self):
        head = self.make_head()
        distribution = make_distribution(n=0, k=5)
        distribution["ref_points"] = torch.empty(0, 4)
        refined, valid, diagnostics = head(
            torch.randn(1, 2, 8, 8), torch.empty(0, 3), torch.empty(0, 4),
            torch.empty(0, dtype=torch.long), (32, 32), distribution, True,
        )
        self.assertEqual(refined.shape, (0, 4))
        self.assertEqual(valid.shape, (0,))
        self.assertEqual(diagnostics["candidate_positions"].shape, (0, 4, 5))
        refined.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))

    def test_zero_initialization_retains_boxes_and_detaches_sampling_geometry(self):
        head = self.make_head()
        features = torch.randn(1, 2, 8, 8)
        queries = torch.randn(1, 3)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]], requires_grad=True)
        distribution = make_distribution(k=5)
        distribution["pred_corners"].requires_grad_()
        distribution["ref_points"].requires_grad_()

        refined, valid, diagnostics = head(
            features, queries, boxes, torch.tensor([0]), (32, 32),
            distribution=distribution, return_diagnostics=True,
        )
        self.assertTrue(valid.all())
        torch.testing.assert_close(refined, boxes)
        self.assertFalse(diagnostics["candidate_positions"].requires_grad)
        refined.sum().backward()
        self.assertIsNone(distribution["pred_corners"].grad)
        self.assertIsNone(distribution["ref_points"].grad)
        self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))

    def test_chunking_preserves_outputs_diagnostics_and_gradients(self):
        full = self.make_head(chunk=32)
        chunked = self.make_head(chunk=1)
        chunked.load_state_dict(copy.deepcopy(full.state_dict()))
        with torch.no_grad():
            full.refiner[-1].weight.normal_(std=0.1)
            full.refiner[-1].bias.copy_(torch.tensor([0.1, -0.1, 0.05, -0.05]))
            chunked.refiner[-1].weight.copy_(full.refiner[-1].weight)
            chunked.refiner[-1].bias.copy_(full.refiner[-1].bias)
        features = torch.randn(2, 2, 7, 9)
        queries = torch.randn(3, 3)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2],
                              [0.4, 0.4, 0.3, 0.2],
                              [0.6, 0.6, 0.2, 0.3]])
        batches = torch.tensor([0, 0, 1])
        distribution = make_distribution(n=3, k=5)
        distribution["ref_points"] = boxes.clone()

        feature_full = features.clone().requires_grad_()
        feature_chunked = features.clone().requires_grad_()
        result_full = full(
            feature_full, queries, boxes, batches, (35, 45), distribution, True
        )
        result_chunked = chunked(
            feature_chunked, queries, boxes, batches, (35, 45), distribution, True
        )
        torch.testing.assert_close(result_full[0], result_chunked[0])
        torch.testing.assert_close(result_full[1], result_chunked[1])
        for key in result_full[2]:
            torch.testing.assert_close(result_full[2][key], result_chunked[2][key])
        result_full[0].sum().backward()
        result_chunked[0].sum().backward()
        torch.testing.assert_close(feature_full.grad, feature_chunked.grad)
        for left, right in zip(full.parameters(), chunked.parameters()):
            torch.testing.assert_close(left.grad, right.grad)


if __name__ == "__main__":
    unittest.main()
