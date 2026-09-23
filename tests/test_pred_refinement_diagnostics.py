import unittest

import torch

from src.solver.pred_refinement_diagnostics import PredRefinementDiagnostics


torch.set_num_threads(1)


def make_outputs(before, after, logits, mask):
    return {
        "baseline_boxes": torch.tensor([before], dtype=torch.float64),
        "pred_boxes": torch.tensor([after], dtype=torch.float64),
        "pred_logits": torch.tensor([logits], dtype=torch.float64),
        "refinement_mask": torch.tensor([mask], dtype=torch.bool),
    }


class PredRefinementDiagnosticsTest(unittest.TestCase):
    def test_one_pixel_shift_has_expected_band_and_tangent_coverage(self):
        target = [0.5, 0.5, 0.2, 0.4]
        outputs = make_outputs(
            [[0.51, 0.5, 0.2, 0.4]], [target], [[0.0]], [True]
        )
        diagnostics = PredRefinementDiagnostics(offset_pixels=1.0)

        diagnostics.update(
            outputs,
            [{"boxes": torch.tensor([target]), "labels": torch.tensor([0])}],
            [(torch.tensor([0]), torch.tensor([0]))],
            input_size=(100, 100),
        )

        report = diagnostics.compute()
        coverage = report["sampling_coverage"]
        self.assertAlmostEqual(coverage["edge_error_px"]["left"]["mean"], 1.0, places=6)
        self.assertAlmostEqual(coverage["edge_error_px"]["right"]["p95"], 1.0, places=6)
        self.assertAlmostEqual(coverage["edge_error_px"]["top"]["mean"], 0.0, places=6)
        self.assertEqual(coverage["inside_band_fraction"]["left"], 1.0)
        self.assertEqual(coverage["all_edges_inside_band_fraction"], 1.0)
        self.assertAlmostEqual(coverage["mean_tangent_coverage"], 0.975)
        group = report["coverage_groups"]["all_edges_inside_band"]
        self.assertEqual(group["pairs"], 1)
        self.assertAlmostEqual(group["baseline_iou"], 19 / 21)
        self.assertAlmostEqual(group["refined_iou"], 1.0)
        self.assertEqual(group["improved"], 1)

    def test_reports_refined_and_fallback_matches_in_coverage_groups(self):
        targets = [[0.3, 0.5, 0.2, 0.2], [0.7, 0.5, 0.2, 0.2]]
        before = [targets[0], [0.73, 0.5, 0.2, 0.2]]
        after = [targets[0], before[1]]
        outputs = make_outputs(before, after, [[0.0], [0.0]], [True, False])
        diagnostics = PredRefinementDiagnostics(offset_pixels=1.0)

        diagnostics.update(
            outputs,
            [{"boxes": torch.tensor(targets), "labels": torch.tensor([0, 0])}],
            [(torch.tensor([0, 1]), torch.tensor([0, 1]))],
            input_size=(100, 100),
        )

        report = diagnostics.compute()
        self.assertEqual(report["prediction_counts"], {
            "matched_pairs": 2, "refined_matched_pairs": 1,
            "fallback_matched_pairs": 1, "invalid_metric_pairs": 0,
        })
        self.assertEqual(
            report["coverage_groups"]["all_edges_inside_band"]["pairs"], 1
        )
        outside = report["coverage_groups"]["at_least_one_edge_outside_band"]
        self.assertEqual(outside["pairs"], 1)
        self.assertEqual(outside["refined_pairs"], 0)
        self.assertEqual(outside["mean_delta"], 0.0)

    def test_unmatched_queries_are_grouped_by_score_and_movement(self):
        before = [[0.2, 0.2, 0.1, 0.1],
                  [0.5, 0.5, 0.2, 0.2],
                  [0.8, 0.8, 0.1, 0.1]]
        after = [before[0], [0.52, 0.5, 0.2, 0.2], before[2]]
        logits = [[5.0], [0.0], [-1.3862943611198906]]
        outputs = make_outputs(before, after, logits, [True, True, False])
        diagnostics = PredRefinementDiagnostics()

        diagnostics.update(
            outputs,
            [{"boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1]])}],
            [(torch.tensor([0]), torch.tensor([0]))],
            input_size=(100, 100),
        )

        unmatched = diagnostics.compute()["unmatched_queries"]
        self.assertEqual(unmatched["all"]["count"], 2)
        self.assertEqual(unmatched["all"]["refined"], 1)
        self.assertEqual(unmatched["all"]["moved"], 1)
        self.assertAlmostEqual(unmatched["all"]["mean_edge_displacement_px"], 0.5)
        self.assertEqual(unmatched["score_ge_0_1"]["count"], 2)
        self.assertEqual(unmatched["score_ge_0_3"]["count"], 1)
        self.assertEqual(unmatched["score_ge_0_5"]["count"], 1)

    def test_empty_gt_reports_none_instead_of_nan(self):
        box = [0.5, 0.5, 0.2, 0.2]
        diagnostics = PredRefinementDiagnostics()
        diagnostics.update(
            make_outputs([box], [box], [[0.0]], [False]),
            [{"boxes": torch.empty(0, 4)}],
            [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))],
            input_size=(100, 100),
        )

        report = diagnostics.compute()
        self.assertEqual(report["prediction_counts"]["matched_pairs"], 0)
        self.assertIsNone(report["sampling_coverage"]["edge_error_px"]["left"]["mean"])
        self.assertIsNone(report["sampling_coverage"]["all_edges_inside_band_fraction"])
        self.assertIsNone(
            report["coverage_groups"]["all_edges_inside_band"]["baseline_iou"]
        )
        self.assertEqual(report["unmatched_queries"]["all"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
