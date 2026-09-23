import unittest

import torch
import torch.nn as nn

from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion


torch.set_num_threads(1)


class RecordingMatcher(nn.Module):
    def __init__(self, source_index=0, target_index=0):
        super().__init__()
        self.source_index = source_index
        self.target_index = target_index
        self.calls = []

    def forward(self, outputs, targets, **kwargs):
        self.calls.append((
            {name: value.detach().clone() for name, value in outputs.items()},
            [{name: value.detach().clone() for name, value in target.items()}
             for target in targets],
        ))
        device = outputs["pred_boxes"].device
        return {"indices": [(
            torch.tensor([self.source_index], device=device),
            torch.tensor([self.target_index], device=device),
        )]}


class PredRefinementCriterionTest(unittest.TestCase):
    def test_matches_baseline_with_original_logits_then_supervises_refined_box(self):
        matcher = RecordingMatcher(source_index=1)
        criterion = PredRefinementCriterion(matcher)
        baseline = torch.tensor([[[0.2, 0.2, 0.1, 0.1],
                                  [0.6, 0.6, 0.2, 0.2]]])
        refined = baseline.clone()
        refined[0, 1, 0] = 0.7
        logits = torch.tensor([[[2.0, -1.0], [-2.0, 3.0]]])
        outputs = {
            "pred_boxes": refined,
            "pred_logits": logits,
            "baseline_boxes": baseline,
            "refinement_mask": torch.tensor([[False, True]]),
            "refinement_zero_loss": refined.sum() * 0,
        }
        target = {"boxes": torch.tensor([[0.6, 0.6, 0.2, 0.2]]),
                  "labels": torch.tensor([1])}

        losses = criterion(outputs, [target])

        matched_outputs, _ = matcher.calls[0]
        torch.testing.assert_close(matched_outputs["pred_boxes"], baseline)
        torch.testing.assert_close(matched_outputs["pred_logits"], logits)
        self.assertAlmostEqual(float(losses["loss_bbox"]), 0.5, places=6)
        self.assertGreater(float(losses["loss_giou"]), 0.0)

    def test_filters_invalid_predictions_and_maps_match_to_original_query(self):
        matcher = RecordingMatcher(source_index=1)
        criterion = PredRefinementCriterion(matcher)
        baseline = torch.tensor([[[0.1, 0.1, 0.1, 0.1],
                                  [0.2, 0.2, -0.1, 0.1],
                                  [0.3, 0.3, 0.1, 0.1],
                                  [0.4, 0.4, 0.1, 0.1]]])
        logits = torch.tensor([[[1.0], [2.0], [float("nan")], [4.0]]])
        refined = baseline.clone()
        refined[0, 3] = torch.tensor([0.5, 0.4, 0.1, 0.1])
        outputs = {
            "pred_boxes": refined,
            "pred_logits": logits,
            "baseline_boxes": baseline,
            "refinement_mask": torch.tensor([[False, False, False, True]]),
            "refinement_zero_loss": refined.nan_to_num().sum() * 0,
        }
        target = {"boxes": torch.tensor([[0.4, 0.4, 0.1, 0.1]]),
                  "labels": torch.tensor([0])}

        losses = criterion(outputs, [target])

        matched_outputs, _ = matcher.calls[0]
        torch.testing.assert_close(
            matched_outputs["pred_boxes"], baseline[:, [0, 3]])
        self.assertAlmostEqual(float(losses["loss_bbox"]), 0.5, places=6)

    def test_filters_invalid_and_ignored_targets_before_matching(self):
        matcher = RecordingMatcher()
        criterion = PredRefinementCriterion(matcher)
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
        outputs = {
            "pred_boxes": boxes,
            "pred_logits": torch.tensor([[[1.0, 0.0]]]),
            "baseline_boxes": boxes,
            "refinement_mask": torch.tensor([[True]]),
            "refinement_zero_loss": boxes.sum() * 0,
        }
        targets = [{
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2],
                                   [float("nan"), 0.2, 0.1, 0.1],
                                   [0.2, 0.2, 0.0, 0.1],
                                   [0.3, 0.3, 0.1, 0.1],
                                   [0.4, 0.4, 0.1, 0.1]]),
            "labels": torch.tensor([1, 99, 99, 0, 0]),
            "iscrowd": torch.tensor([0, 0, 0, 1, 0]),
            "ignore": torch.tensor([0, 0, 0, 0, 1]),
        }]

        criterion(outputs, targets)

        _, matched_targets = matcher.calls[0]
        torch.testing.assert_close(
            matched_targets[0]["boxes"], torch.tensor([[0.5, 0.5, 0.2, 0.2]]))
        torch.testing.assert_close(matched_targets[0]["labels"], torch.tensor([1]))

    def test_mask_excludes_a_successful_match_from_loss(self):
        matcher = RecordingMatcher()
        criterion = PredRefinementCriterion(matcher)
        refined = torch.tensor([[[0.9, 0.9, 0.1, 0.1]]], requires_grad=True)
        outputs = {
            "pred_boxes": refined,
            "pred_logits": torch.tensor([[[1.0]]]),
            "baseline_boxes": torch.tensor([[[0.5, 0.5, 0.1, 0.1]]]),
            "refinement_mask": torch.tensor([[False]]),
            "refinement_zero_loss": refined.sum() * 0,
        }

        losses = criterion(outputs, [{
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "labels": torch.tensor([0]),
        }])

        self.assertEqual(len(matcher.calls), 1)
        self.assertEqual(float(losses["loss_bbox"]), 0.0)
        self.assertEqual(float(losses["loss_giou"]), 0.0)

    def test_empty_valid_pairs_keep_zero_loss_connected_for_backward(self):
        matcher = RecordingMatcher()
        criterion = PredRefinementCriterion(matcher)
        parameter = nn.Parameter(torch.tensor(2.0))
        boxes = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]])
        losses = criterion({
            "pred_boxes": boxes,
            "pred_logits": torch.tensor([[[1.0]]]),
            "baseline_boxes": boxes,
            "refinement_mask": torch.tensor([[False]]),
            "refinement_zero_loss": parameter * 0,
        }, [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}])

        sum(losses.values()).backward()

        self.assertIsNotNone(parameter.grad)
        self.assertEqual(float(parameter.grad), 0.0)

    def test_targets_are_normalized_cxcywh_and_need_no_image_size(self):
        matcher = RecordingMatcher()
        criterion = PredRefinementCriterion(matcher)
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
        target_boxes = torch.tensor([[0.25, 0.75, 0.1, 0.3]])
        criterion({
            "pred_boxes": boxes,
            "pred_logits": torch.tensor([[[1.0]]]),
            "baseline_boxes": boxes,
            "refinement_mask": torch.tensor([[True]]),
            "refinement_zero_loss": boxes.sum() * 0,
        }, [{"boxes": target_boxes, "labels": torch.tensor([0])}])

        _, matched_targets = matcher.calls[0]
        torch.testing.assert_close(matched_targets[0]["boxes"], target_boxes)


if __name__ == "__main__":
    unittest.main()
