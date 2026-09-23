import copy
import unittest

import torch

from src.zoo.dfine.gt_candidate_strip_head import GTCandidateStripHead
from src.zoo.dfine.pred_strip_head import PredictionStripHead


torch.set_num_threads(1)


def distribution(refs, k=5, up=5.0):
    return {
        "pred_corners": torch.zeros(len(refs), 4 * k),
        "ref_points": refs.clone(),
        "up": torch.tensor([up]),
        "reg_scale": torch.tensor([4.0]),
    }


class GTCandidateStripHeadTests(unittest.TestCase):
    def make_head(self):
        return GTCandidateStripHead(2, 3, hidden_dim=2, num_points=3)

    def test_selects_nearest_valid_candidate_instead_of_nearest_invalid(self):
        positions = torch.tensor([[[0.40, 0.30, 0.80],
                                   [0.40, 0.20, 0.70],
                                   [0.60, 0.50, 0.90],
                                   [0.60, 0.45, 0.90]]])
        valid = torch.ones_like(positions, dtype=torch.bool)
        valid[..., 0] = False
        gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])

        selected = GTCandidateStripHead._select_candidate_indices(
            positions, valid, gt
        )

        self.assertEqual(selected.tolist(), [[1, 1, 1, 1]])

    def test_gt_changes_selection_without_changing_candidates_or_prediction_span(self):
        head = self.make_head()
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
        dist = distribution(boxes, up=0.5)
        positions, _, distribution_valid = head._distribution_geometry(
            dist, 1, boxes.device
        )
        grid, _, candidate_valid = head._candidate_grid(
            positions, boxes, distribution_valid, (100, 100)
        )
        gt_left = torch.tensor([[0.35, 0.5, 0.1, 0.2]])
        gt_right = torch.tensor([[0.65, 0.5, 0.1, 0.2]])

        left_choice = head._select_candidate_indices(positions, candidate_valid, gt_left)
        right_choice = head._select_candidate_indices(positions, candidate_valid, gt_right)

        self.assertFalse(torch.equal(left_choice, right_choice))
        # Every vertical candidate retains the prediction's [top, bottom] span.
        normalized_grid = (grid + 1) / 2
        expected_y = torch.linspace(0.3, 0.7, 3)
        expected_spans = expected_y[None].expand(5, 3)
        torch.testing.assert_close(normalized_grid[0, 0, :, :, 1, 1], expected_spans)
        torch.testing.assert_close(normalized_grid[0, 2, :, :, 1, 1], expected_spans)

    def test_zero_initialization_returns_baseline_not_selected_candidate_box(self):
        head = self.make_head()
        features = torch.randn(1, 2, 8, 8)
        queries = torch.randn(1, 3)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        gt = torch.tensor([[0.4, 0.4, 0.1, 0.1]])

        refined, valid = head(
            features, queries, boxes, gt, torch.tensor([0]), (32, 32),
            distribution=distribution(boxes, up=0.5),
        )

        self.assertTrue(valid.all())
        torch.testing.assert_close(refined, boxes)

    def test_single_valid_candidate_matches_p2_with_same_nonzero_state(self):
        gt_head = self.make_head()
        pred_head = PredictionStripHead(
            2, 3, hidden_dim=2, num_points=3, sampling_mode="distribution"
        )
        pred_head.load_state_dict(copy.deepcopy(gt_head.state_dict()))
        with torch.no_grad():
            gt_head.refiner[-1].weight.normal_(std=0.1)
            gt_head.refiner[-1].bias.copy_(torch.tensor([0.1, -0.1, 0.05, -0.05]))
            pred_head.refiner[-1].weight.copy_(gt_head.refiner[-1].weight)
            pred_head.refiner[-1].bias.copy_(gt_head.refiner[-1].bias)
        features = torch.randn(1, 2, 8, 8)
        queries = torch.randn(1, 3)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        dist = distribution(boxes)

        selected, selected_valid = gt_head(
            features, queries, boxes, boxes, torch.tensor([0]), (32, 32),
            distribution=dist,
        )
        probabilistic, probabilistic_valid = pred_head(
            features, queries, boxes, torch.tensor([0]), (32, 32), dist
        )

        self.assertTrue(selected_valid.all() and probabilistic_valid.all())
        torch.testing.assert_close(selected, probabilistic)

    def test_invalid_and_empty_gt_fall_back_with_parameter_gradients(self):
        head = self.make_head()
        features = torch.randn(1, 2, 8, 8)
        queries = torch.randn(1, 3)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        refined, valid = head(
            features, queries, boxes, torch.full((1, 4), float("nan")),
            torch.tensor([0]), (32, 32), distribution=distribution(boxes),
        )
        self.assertFalse(valid.any())
        torch.testing.assert_close(refined, boxes)
        refined.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))

        empty_dist = distribution(torch.empty(0, 4))
        empty, empty_valid = head(
            features, torch.empty(0, 3), torch.empty(0, 4), torch.empty(0, 4),
            torch.empty(0, dtype=torch.long), (32, 32), distribution=empty_dist,
        )
        self.assertEqual(empty.shape, (0, 4))
        self.assertEqual(empty_valid.shape, (0,))


if __name__ == "__main__":
    unittest.main()
