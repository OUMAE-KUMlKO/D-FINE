import unittest
from unittest import mock

import torch
import torch.nn as nn

from src.zoo.dfine.probability_refinement_criterion import ProbabilityRefinementCriterion
import src.zoo.dfine.probability_refinement_criterion as criterion_module


torch.set_num_threads(1)


class FixedMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, outputs, targets, **kwargs):
        self.calls.append((outputs, targets))
        device = outputs["pred_boxes"].device
        count = min(outputs["pred_boxes"].shape[1], len(targets[0]["boxes"]))
        indices = torch.arange(count, device=device)
        return {"indices": [(
            indices, indices.clone()
        )]}


def make_case(candidate_positions, gt_box, candidate_valid=None):
    positions = torch.tensor(candidate_positions, dtype=torch.float32).reshape(1, 1, 4, -1)
    k = positions.shape[-1]
    if candidate_valid is None:
        candidate_valid = torch.ones_like(positions, dtype=torch.bool)
    calibrated = torch.zeros_like(positions, requires_grad=True)
    baseline = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    zero_parameter = nn.Parameter(torch.tensor(1.0))
    outputs = {
        "pred_boxes": baseline.clone(), "baseline_boxes": baseline,
        "pred_logits": torch.tensor([[[2.0]]]),
        "refinement_mask": torch.tensor([[False]]),
        "refinement_zero_loss": zero_parameter * 0,
        "calibrated_corners": calibrated,
        "candidate_positions": positions,
        "candidate_valid": candidate_valid,
        "input_valid": torch.tensor([[True]]),
    }
    targets = [{"boxes": torch.tensor([gt_box]), "labels": torch.tensor([0])}]
    return outputs, targets, calibrated, zero_parameter


class ProbabilityRefinementCriterionTests(unittest.TestCase):
    def test_interpolates_in_coordinate_order_and_maps_back_to_reversed_bins(self):
        # GT left edge is 0.45: original bin 1 (0.6) gets .25, bin 2 (0.4) gets .75.
        positions = [[0.9, 0.6, 0.4], [0.8, 0.9, 1.0],
                     [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]
        outputs, targets, calibrated, _ = make_case(
            positions, [0.55, 0.5, 0.2, 0.2]
        )
        calibrated.data[0, 0, 0] = torch.tensor([2.0, 1.0, -1.0])
        criterion = ProbabilityRefinementCriterion(FixedMatcher())

        losses = criterion(outputs, targets)

        log_prob = calibrated[0, 0, 0].log_softmax(-1)
        expected = -(0.25 * log_prob[1] + 0.75 * log_prob[2]) * 0.05
        torch.testing.assert_close(losses["loss_candidate"], expected)
        self.assertEqual(criterion.last_candidate_stats["candidate_edges"], 1.0)
        self.assertEqual(criterion.last_candidate_stats["candidate_edge_fraction"], 0.25)
        losses["loss_candidate"].backward()
        self.assertIsNotNone(calibrated.grad)
        expected_gradient = 0.05 * (
            calibrated.detach()[0, 0, 0].softmax(-1) - torch.tensor([0.0, 0.25, 0.75])
        )
        torch.testing.assert_close(calibrated.grad[0, 0, 0], expected_gradient)

    def test_exact_candidate_is_onehot_and_invalid_required_candidate_skips_edge(self):
        positions = [[0.8, 0.6, 0.4], [0.8, 0.9, 1.0],
                     [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]
        outputs, targets, calibrated, _ = make_case(
            positions, [0.7, 0.5, 0.2, 0.2]
        )
        calibrated.data[0, 0, 0] = torch.tensor([0.0, 2.0, -1.0])
        criterion = ProbabilityRefinementCriterion(FixedMatcher())
        loss = criterion(outputs, targets)["loss_candidate"]
        expected = -calibrated[0, 0, 0].log_softmax(-1)[1] * 0.05
        torch.testing.assert_close(loss, expected)

        outputs["candidate_valid"][0, 0, 0, 1] = False
        skipped = criterion(outputs, targets)
        self.assertEqual(float(skipped["loss_candidate"]), 0.0)
        self.assertEqual(criterion.last_candidate_stats["candidate_edges"], 0.0)

    def test_candidate_supervision_uses_input_valid_not_refined_box_or_mask(self):
        positions = [[0.6, 0.4], [0.8, 1.0], [0.0, 0.2], [0.0, 0.2]]
        outputs, targets, calibrated, _ = make_case(
            positions, [0.6, 0.5, 0.2, 0.2]
        )
        outputs["pred_boxes"][0, 0, 2] = -1.0
        outputs["refinement_mask"][0, 0] = False
        criterion = ProbabilityRefinementCriterion(FixedMatcher())

        losses = criterion(outputs, targets)

        self.assertEqual(float(losses["loss_bbox"]), 0.0)
        self.assertEqual(float(losses["loss_giou"]), 0.0)
        self.assertGreater(float(losses["loss_candidate"]), 0.0)
        losses["loss_candidate"].backward()
        self.assertTrue(torch.isfinite(calibrated.grad).all())

    def test_empty_candidate_edges_keep_new_branch_connected_and_stats_zero(self):
        positions = [[0.8, 0.6], [0.8, 1.0], [0.0, 0.2], [0.0, 0.2]]
        outputs, targets, calibrated, zero_parameter = make_case(
            positions, [0.6, 0.5, 0.2, 0.2]
        )
        outputs["input_valid"][0, 0] = False
        # The model-provided zero includes the new branch in real integration.
        outputs["refinement_zero_loss"] = calibrated.sum() * 0 + zero_parameter * 0
        criterion = ProbabilityRefinementCriterion(FixedMatcher())

        losses = criterion(outputs, targets)
        sum(losses.values()).backward()

        self.assertIsNotNone(calibrated.grad)
        self.assertIsNotNone(zero_parameter.grad)
        self.assertEqual(criterion.last_candidate_stats, {
            "candidate_loss_unweighted": 0.0,
            "candidate_loss_weighted": 0.0,
            "candidate_edges": 0.0,
            "candidate_edge_fraction": 0.0,
        })

    def test_ddp_normalizer_does_not_clamp_positive_global_count_after_averaging(self):
        positions = [[0.8, 0.6, 0.4], [0.8, 0.9, 1.0],
                     [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]
        outputs, targets, calibrated, _ = make_case(
            positions, [0.7, 0.5, 0.2, 0.2]
        )
        criterion = ProbabilityRefinementCriterion(FixedMatcher())
        with mock.patch.object(criterion_module.dist, "is_available", return_value=True), \
             mock.patch.object(criterion_module.dist, "is_initialized", return_value=True), \
             mock.patch.object(criterion_module.dist, "get_world_size", return_value=4), \
             mock.patch.object(criterion_module.dist, "all_reduce", side_effect=lambda value: value):
            loss = criterion(outputs, targets)["loss_candidate"]

        local_cross_entropy = -calibrated[0, 0, 0].log_softmax(-1)[1]
        torch.testing.assert_close(loss, local_cross_entropy * 0.05 * 4)

    def test_multiple_matches_are_batched_and_empty_gt_is_safe(self):
        positions = [[0.8, 0.6, 0.4], [0.8, 0.9, 1.0],
                     [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]
        outputs, targets, _, _ = make_case(positions, [0.7, 0.5, 0.2, 0.2])
        for key in ("pred_boxes", "baseline_boxes", "pred_logits", "refinement_mask",
                    "calibrated_corners", "candidate_positions", "candidate_valid",
                    "input_valid"):
            outputs[key] = torch.cat((outputs[key], outputs[key].detach().clone()), dim=1)
        outputs["calibrated_corners"].requires_grad_()
        targets[0]["boxes"] = targets[0]["boxes"].repeat(2, 1)
        targets[0]["labels"] = targets[0]["labels"].repeat(2)
        matcher = FixedMatcher()
        criterion = ProbabilityRefinementCriterion(matcher)
        losses = criterion(outputs, targets)
        self.assertEqual(len(matcher.calls), 1)
        self.assertEqual(criterion.last_candidate_stats["candidate_edges"], 2.0)
        self.assertEqual(criterion.last_candidate_stats["candidate_edge_fraction"], 0.25)
        self.assertGreater(float(losses["loss_candidate"]), 0.0)

        targets[0] = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
        empty = criterion(outputs, targets)
        self.assertEqual(float(empty["loss_candidate"]), 0.0)
        self.assertEqual(criterion.last_candidate_stats["candidate_edges"], 0.0)
if __name__ == "__main__":
    unittest.main()
