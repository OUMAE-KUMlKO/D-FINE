import unittest

import torch

from tools.analysis.detector_geometry_metrics import DetectorGeometryComparison


def boxes(rows):
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, 4)


def indices(queries, targets):
    return (torch.tensor(queries, dtype=torch.long), torch.tensor(targets, dtype=torch.long))


class DetectorGeometryComparisonTests(unittest.TestCase):
    def test_known_pixels_non_square_input_and_query_permutation(self):
        metric = DetectorGeometryComparison()
        target = boxes([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.4, 0.4]])
        # Shift small box 2 px right and 2 px down on a 100 x 200 input.
        shifted = boxes([[0.51, 0.52, 0.1, 0.1]])[0]
        baseline = torch.stack((target[1], shifted))[None]
        model = target[None]
        metric.update({"pred_boxes": baseline}, {"pred_boxes": model},
                      [{"boxes": target}], [indices([0, 1], [1, 0])],
                      [indices([0, 1], [0, 1])], (100, 200))
        result = metric.compute()
        small = result["models"]["baseline"]["groups"]["small"]
        for error in small["edge_mae_px_ltrb"]:
            self.assertAlmostEqual(error, 2.0)
        self.assertAlmostEqual(small["mean_edge_mae_px"], 2.0)
        self.assertAlmostEqual(small["mean_iou"], 144 / 256)
        self.assertEqual(result["models"]["baseline"]["groups"]["medium"]["pairs"], 1)
        common = result["common_gt_comparison"]["groups"]["all"]
        self.assertEqual(common["pairs"], 2)
        self.assertAlmostEqual(common["baseline"]["mean_edge_mae_px"], 1.0)
        self.assertEqual(common["model"]["mean_edge_mae_px"], 0.0)

    def test_nonoverlapping_matches_and_empty_gt(self):
        metric = DetectorGeometryComparison()
        target = boxes([[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1],
                        [0.8, 0.8, 0.1, 0.1]])
        metric.update({"pred_boxes": target[None]}, {"pred_boxes": target[None]},
                      [{"boxes": target}], [indices([0], [0])],
                      [indices([1], [1])], (100, 200))
        metric.update({"pred_boxes": target[None]}, {"pred_boxes": target[None]},
                      [{"boxes": boxes([])}], [indices([], [])],
                      [indices([], [])], (100, 200))
        result = metric.compute()
        self.assertEqual(result["counts"], dict(total_gt=3, baseline_matched=1,
                         model_matched=1, common_gt=0, baseline_only=1,
                         model_only=1, neither=1))
        common = result["common_gt_comparison"]["groups"]["all"]
        self.assertEqual(common["pairs"], 0)
        self.assertIsNone(common["baseline"]["mean_edge_mae_px"])
        self.assertEqual(result["models"]["model"]["groups"]["all"]["pairs"], 1)

    def test_invalid_pairs_are_filtered_and_predictions_are_not_clipped(self):
        metric = DetectorGeometryComparison()
        target = boxes([[0.0, 0.5, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]])
        baseline = boxes([[-0.1, 0.5, 0.2, 0.2], [float("nan"), 0.5, 0.1, 0.1]])
        metric.update({"pred_boxes": baseline[None]}, {"pred_boxes": target[None]},
                      [{"boxes": target}], [indices([0, 1], [0, 1])],
                      [indices([0, 1], [0, 1])], (100, 200))
        result = metric.compute()
        own = result["models"]["baseline"]
        self.assertEqual(own["invalid_metric_pairs"], 1)
        self.assertAlmostEqual(own["groups"]["all"]["mean_edge_mae_px"], 10.0)
        self.assertEqual(result["models"]["model"]["groups"]["all"]["pairs"], 2)
        self.assertEqual(result["common_gt_comparison"]["groups"]["all"]["pairs"], 1)


if __name__ == "__main__":
    unittest.main()
