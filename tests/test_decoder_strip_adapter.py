import copy
import unittest

import torch

from src.zoo.dfine.decoder_strip_adapter import DistributionStripQueryAdapter
from src.zoo.dfine.pred_strip_head import (
    PredictionStripHead,
    encode_distribution_candidates,
)


torch.set_num_threads(1)


def make_inputs(batch=1, queries=2, k=5):
    features = torch.randn(batch, 2, 8, 8)
    query = torch.randn(batch, queries, 3)
    boxes = torch.full((batch, queries, 4), 0.2)
    boxes[..., :2] = 0.5
    distribution = {
        "pred_corners": torch.zeros(batch, queries, 4 * k),
        "ref_points": boxes.clone(),
        "up": torch.tensor([0.5]),
        "reg_scale": torch.tensor([4.0]),
    }
    return features, query, boxes, distribution


class DistributionStripQueryAdapterTests(unittest.TestCase):
    def make_adapter(self, chunk=32):
        return DistributionStripQueryAdapter(
            2, 3, hidden_dim=2, num_points=3, query_chunk_size=chunk
        )

    def test_zero_initialization_returns_zero_delta_with_gradient_path(self):
        adapter = self.make_adapter()
        features, queries, boxes, distribution = make_inputs()

        delta, valid = adapter(features, queries, boxes, (32, 32), distribution)

        self.assertTrue(valid.all())
        torch.testing.assert_close(delta, torch.zeros_like(delta))
        delta.sum().backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in adapter.parameters()
        ))

    def test_nonzero_output_matches_explicit_probability_fusion(self):
        adapter = self.make_adapter()
        features, queries, boxes, distribution = make_inputs(queries=1)
        distribution["pred_corners"] = torch.tensor([[[
            0.0, 1.0, 2.0, 1.0, 0.0,
        ] * 4]])
        with torch.no_grad():
            adapter.query_projection.weight.copy_(torch.tensor([
                [0.2] * 8, [-0.1] * 8, [0.05] * 8,
            ]))

        delta, valid = adapter(features, queries, boxes, (32, 32), distribution)

        flat_boxes = boxes.reshape(-1, 4)
        positions, _, distribution_valid = PredictionStripHead._distribution_geometry(
            {key: value.reshape(-1, value.shape[-1]) if key in {"pred_corners", "ref_points"} else value
             for key, value in distribution.items()}, 1, features.device
        )
        helper = PredictionStripHead(
            2, 3, hidden_dim=2, num_points=3, sampling_mode="distribution"
        )
        grid, point_valid, candidate_valid = helper._candidate_grid(
            positions, flat_boxes, distribution_valid, (32, 32)
        )
        candidate_features = encode_distribution_candidates(
            adapter.projection(features.float()), grid, point_valid,
            torch.tensor([0]), adapter.strip_encoder, adapter.edge_fusion,
            adapter.hidden_dim, adapter.num_points, adapter.query_chunk_size,
        )
        weights, _, _ = PredictionStripHead._masked_candidate_weights(
            distribution["pred_corners"].reshape(1, 4, 5), candidate_valid
        )
        expected_features = (candidate_features * weights[..., None]).sum(2).reshape(1, -1)
        expected = adapter.query_projection(expected_features).reshape_as(delta)
        self.assertTrue(valid.all())
        torch.testing.assert_close(delta, expected)

    def test_invalid_and_empty_inputs_return_zero_and_keep_parameters_connected(self):
        adapter = self.make_adapter()
        features, queries, boxes, distribution = make_inputs(queries=1)
        distribution["pred_corners"][0, 0, 0] = float("nan")
        with torch.no_grad():
            adapter.query_projection.bias.fill_(1.0)
        delta, valid = adapter(features, queries, boxes, (32, 32), distribution)
        self.assertFalse(valid.any())
        torch.testing.assert_close(delta, torch.zeros_like(delta))
        delta.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in adapter.parameters()))

        empty_distribution = {
            "pred_corners": torch.empty(1, 0, 20),
            "ref_points": torch.empty(1, 0, 4),
            "up": torch.tensor([0.5]), "reg_scale": torch.tensor([4.0]),
        }
        empty, empty_valid = adapter(
            features, torch.empty(1, 0, 3), torch.empty(1, 0, 4),
            (32, 32), empty_distribution,
        )
        self.assertEqual(empty.shape, (1, 0, 3))
        self.assertEqual(empty_valid.shape, (1, 0))

    def test_queries_from_different_images_use_only_their_own_feature_map(self):
        adapter = self.make_adapter()
        with torch.no_grad():
            adapter.query_projection.weight.normal_(std=0.2)
        features, queries, boxes, distribution = make_inputs(batch=2, queries=1)
        features[0].fill_(0.0)
        features[1].fill_(2.0)
        first, _ = adapter(features, queries, boxes, (32, 32), distribution)
        changed = features.clone()
        changed[1].fill_(7.0)
        second, _ = adapter(changed, queries, boxes, (32, 32), distribution)
        torch.testing.assert_close(first[0], second[0])
        self.assertFalse(torch.allclose(first[1], second[1]))

    def test_chunking_preserves_outputs_and_nonzero_gradients(self):
        full, chunked = self.make_adapter(32), self.make_adapter(1)
        chunked.load_state_dict(copy.deepcopy(full.state_dict()))
        with torch.no_grad():
            full.query_projection.weight.normal_(std=0.2)
            full.projection.bias.fill_(1.0)
            for module in (*full.strip_encoder, *full.edge_fusion):
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.fill_(1.0)
            chunked.load_state_dict(copy.deepcopy(full.state_dict()))
        features, queries, boxes, distribution = make_inputs(batch=2, queries=3)
        left_features = features.clone().requires_grad_()
        right_features = features.clone().requires_grad_()
        left = full(left_features, queries, boxes, (32, 32), distribution)
        right = chunked(right_features, queries, boxes, (32, 32), distribution)
        torch.testing.assert_close(left[0], right[0])
        torch.testing.assert_close(left[1], right[1])
        left[0].sum().backward()
        right[0].sum().backward()
        torch.testing.assert_close(left_features.grad, right_features.grad)
        self.assertGreater(float(left_features.grad.abs().sum()), 0.0)
        for left_parameter, right_parameter in zip(full.parameters(), chunked.parameters()):
            torch.testing.assert_close(left_parameter.grad, right_parameter.grad)

    def test_distribution_geometry_and_probabilities_are_detached(self):
        adapter = self.make_adapter()
        features, queries, boxes, distribution = make_inputs(queries=1)
        distribution["pred_corners"].requires_grad_()
        distribution["ref_points"].requires_grad_()
        with torch.no_grad():
            adapter.query_projection.weight.normal_(std=0.2)
        delta, _ = adapter(features, queries, boxes, (32, 32), distribution)
        delta.sum().backward()
        self.assertIsNone(distribution["pred_corners"].grad)
        self.assertIsNone(distribution["ref_points"].grad)


if __name__ == "__main__":
    unittest.main()
