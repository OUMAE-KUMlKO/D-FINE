import unittest

import torch

from tools.analysis.categorized_geometry_metrics import CategorizedGeometryComparison


def indices(queries, targets):
    return (torch.tensor(queries, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


class CategorizedGeometryComparisonTests(unittest.TestCase):
    def test_nonconsecutive_categories_keep_gt_identity_and_partition_counts(self):
        metric = CategorizedGeometryComparison()
        target_boxes = torch.tensor([
            [0.1, 0.2, 0.1, 0.1], [0.2, 0.3, 0.1, 0.1],
            [0.3, 0.4, 0.1, 0.1], [0.4, 0.5, 0.1, 0.1],
            [0.5, 0.6, 0.1, 0.1], [0.6, 0.7, 0.1, 0.1],
        ], dtype=torch.float64)
        baseline = target_boxes.clone()
        model = target_boxes.clone()
        baseline_queries, baseline_targets = indices([4, 3, 0, 2], [0, 1, 2, 4])
        model_queries, model_targets = indices([5, 1, 4, 0], [1, 2, 3, 4])
        baseline[baseline_queries] = target_boxes[baseline_targets]
        model[model_queries] = target_boxes[model_targets]
        # Model query 1 matches original GT 2, which becomes category-7 GT 1.
        # A (0.01, 0.02) center shift is 2 px on each axis at input 100 x 200.
        model[1, :2] += torch.tensor([0.01, 0.02], dtype=torch.float64)
        metric.update(
            {"pred_boxes": baseline[None]}, {"pred_boxes": model[None]},
            [{"boxes": target_boxes, "labels": torch.tensor([7, 2, 7, 2, 13, 13])}],
            [(baseline_queries, baseline_targets)], [(model_queries, model_targets)],
            (100, 200),
        )
        result = metric.compute()
        self.assertEqual(set(result["by_category"]), {"2", "7", "13"})
        overall = result["overall"]
        self.assertEqual(overall["counts"], dict(
            total_gt=6, baseline_matched=4, model_matched=4, common_gt=3,
            baseline_only=1, model_only=1, neither=1,
        ))
        for name, count in overall["counts"].items():
            self.assertEqual(sum(group["counts"][name]
                                 for group in result["by_category"].values()), count)
        self.assertEqual(result["by_category"]["7"]["counts"]["baseline_only"], 1)
        self.assertEqual(result["by_category"]["2"]["counts"]["model_only"], 1)
        self.assertEqual(result["by_category"]["13"]["counts"]["neither"], 1)
        cat7 = result["by_category"]["7"]["common_gt_comparison"]["groups"]["all"]
        self.assertEqual(cat7["pairs"], 1)
        self.assertEqual(cat7["baseline"]["mean_edge_mae_px"], 0.0)
        self.assertAlmostEqual(cat7["model"]["mean_edge_mae_px"], 2.0)
        for model_name in ("baseline", "model"):
            for size in ("all", "small", "medium", "large"):
                self.assertEqual(
                    sum(group["models"][model_name]["groups"][size]["pairs"]
                        for group in result["by_category"].values()),
                    overall["models"][model_name]["groups"][size]["pairs"],
                )

    def test_empty_images_and_categories_absent_from_other_images(self):
        metric = CategorizedGeometryComparison()
        empty_target = {"boxes": torch.empty((0, 4)),
                        "labels": torch.empty((0,), dtype=torch.long)}
        predictions = {"pred_boxes": torch.tensor([[[0.5, 0.5, 0.1, 0.1]]])}
        metric.update(predictions, predictions, [empty_target],
                      [indices([], [])], [indices([], [])], (640, 640))
        self.assertEqual(metric.compute()["by_category"], {})
        targets = [
            {"boxes": predictions["pred_boxes"][0], "labels": torch.tensor([7])},
            empty_target,
            {"boxes": predictions["pred_boxes"][0], "labels": torch.tensor([2])},
        ]
        batch_predictions = {"pred_boxes": predictions["pred_boxes"].repeat(3, 1, 1)}
        metric.update(batch_predictions, batch_predictions, targets,
                      [indices([0], [0]), indices([], []), indices([], [])],
                      [indices([0], [0]), indices([], []), indices([0], [0])],
                      (640, 640))
        result = metric.compute()
        self.assertEqual(result["overall"]["counts"]["total_gt"], 2)
        self.assertEqual(result["by_category"]["7"]["counts"]["total_gt"], 1)
        self.assertEqual(result["by_category"]["2"]["counts"]["total_gt"], 1)
        self.assertEqual(result["by_category"]["7"]["counts"]["neither"], 0)
        self.assertEqual(result["by_category"]["2"]["counts"]["neither"], 0)


if __name__ == "__main__":
    unittest.main()
