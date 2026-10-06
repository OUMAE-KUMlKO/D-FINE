import unittest

import numpy as np

from tools.analysis.analyze_detector_density import analyze_cache, spatial_density


def record(boxes, baseline_indices=None, model_indices=None, labels=None, area=None):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    count = len(boxes)
    default = (np.arange(count), np.arange(count))
    return {
        "image_id": 1, "input_size": [100, 200], "orig_size": [100, 200],
        "target": {"boxes": boxes, "labels": np.asarray(labels if labels is not None else [2] * count),
                   "area_original": np.asarray(area if area is not None else boxes[:, 2] * 200 * boxes[:, 3] * 100)},
        "baseline": {"pred_boxes": boxes.copy()}, "model": {"pred_boxes": boxes.copy()},
        "matches": {"geometry_only": {
            "baseline": baseline_indices if baseline_indices is not None else default,
            "model": model_indices if model_indices is not None else default,
        }},
    }


class DetectorDensityTests(unittest.TestCase):
    def test_known_boundary_gaps_and_single_gt(self):
        boxes = np.array([[0, 0, 4, 4], [4, 0, 8, 4], [16, 0, 20, 4], [36, 0, 40, 4]])
        neighbors, nearest = spatial_density(boxes)
        np.testing.assert_array_equal(neighbors, [1, 2, 1, 0])
        np.testing.assert_allclose(nearest, [0, 0, 8, 16])
        neighbors, nearest = spatial_density(boxes[:1])
        np.testing.assert_array_equal(neighbors, [0])
        self.assertTrue(np.isinf(nearest[0]))

    def test_unmatched_gt_still_determines_density_and_common_queries_align(self):
        rec = record([[0.1, 0.2, 0.02, 0.04], [0.12, 0.2, 0.02, 0.04],
                      [0.9, 0.8, 0.02, 0.04]],
                     baseline_indices=(np.array([2, 0]), np.array([0, 2])),
                     model_indices=(np.array([1]), np.array([0])))
        rec["baseline"]["pred_boxes"][2] = rec["target"]["boxes"][0]
        rec["model"]["pred_boxes"][1] = rec["target"]["boxes"][0]
        rec["model"]["pred_boxes"][1, :2] += [0.01, 0.02]
        result = analyze_cache({"records": [rec]})
        self.assertEqual(result["counts"]["valid_gt"], 3)
        self.assertEqual(result["counts"]["common_matched_gt"], 1)
        group = result["groups"]["local_neighbor_density"]["1"]
        self.assertEqual(group["pairs"], 1)
        self.assertEqual(group["baseline"]["mean_edge_mae_px"], 0.0)
        self.assertAlmostEqual(group["model"]["mean_edge_mae_px"], 2.0)
        self.assertAlmostEqual(group["model"]["mean_iou"], 4 / 28)
        self.assertEqual(result["groups"]["nearest_gap"]["overlap_or_touch"]["pairs"], 1)
        self.assertEqual(result["groups"]["overall"]["pairs"], 1)

    def test_crossed_input_and_original_sizes_and_invalid_pairs(self):
        rec = record([[0.1, 0.2, 0.02, 0.04], [0.5, 0.5, 0.2, 0.4],
                      [0.7, 0.8, 0.1, 0.1]], labels=[7, 2, 7], area=[10000, 16, 16])
        rec["model"]["pred_boxes"][2, 0] = np.nan
        result = analyze_cache({"records": [rec]})
        self.assertEqual(result["counts"]["invalid_common_pairs"], 1)
        groups = result["groups"]
        self.assertEqual(groups["category_by_original_area_size"]["7"]["large"]["pairs"], 1)
        self.assertEqual(groups["category_by_input_area_size"]["7"]["small"]["pairs"], 1)
        self.assertEqual(groups["category_by_original_area_size"]["2"]["small"]["pairs"], 1)
        self.assertEqual(groups["category_by_input_area_size"]["2"]["medium"]["pairs"], 1)
        self.assertEqual(result["standardized_local_neighbor_density"]["eligible_cell_count"], 0)
        self.assertIsNone(result["standardized_local_neighbor_density"]["by_density"])

    def test_standardization_applies_identical_pooled_cell_weights(self):
        # Two eligible category/short-side cells have different frequency per
        # density. Pooled counts are equal, so every density must use 1/2 each.
        records = []
        for category, frequencies in ((2, [20, 40, 20, 40]), (7, [40, 20, 40, 20])):
            for density_index, (neighbors, count) in enumerate(zip([0, 1, 2, 4], frequencies)):
                for _ in range(count):
                    boxes = [[0.1, 0.2, 0.02, 0.04]] * (neighbors + 1)
                    rec = record(boxes, baseline_indices=(np.array([0]), np.array([0])),
                                 model_indices=(np.array([0]), np.array([0])),
                                 labels=[category] * (neighbors + 1))
                    # Four-sided MAE equals x shift / 2: category 2 => 1px,
                    # category 7 => 3px, independent of density.
                    rec["model"]["pred_boxes"][0, 0] += 0.01 if category == 2 else 0.03
                    rec["image_id"] = len(records)
                    records.append(rec)
        result = analyze_cache({"records": records})
        standardized = result["standardized_local_neighbor_density"]
        self.assertEqual(standardized["eligible_cell_count"], 2)
        self.assertEqual(standardized["eligible_pairs"], 240)
        self.assertEqual([cell["weight"] for cell in standardized["cells"]], [0.5, 0.5])
        for density in ("0", "1", "2_3", "4_plus"):
            group = standardized["by_density"][density]
            self.assertEqual(group["retained_pairs"], 60)
            self.assertEqual(group["coverage"], 1.0)
            self.assertAlmostEqual(group["model"]["mean_edge_mae_px"], 2.0)
            self.assertAlmostEqual(group["delta_model_minus_baseline"]["mean_edge_mae_px"], 2.0)

    def test_original_coco_area_thresholds_are_inclusive_in_both_groups(self):
        rec = record([[0.1, 0.2, 0.02, 0.04], [0.5, 0.5, 0.02, 0.04],
                      [0.8, 0.8, 0.02, 0.04]], labels=[2, 7, 2],
                     area=[1024, 9216, 1e10 + 1])
        result = analyze_cache({"records": [rec]})
        original = result["groups"]["original_area_size"]
        self.assertEqual([original[size]["pairs"] for size in ("small", "medium", "large")],
                         [1, 2, 1])
        categories = result["groups"]["category_by_original_area_size"]
        self.assertEqual(categories["2"]["small"]["pairs"], 1)
        self.assertEqual(categories["2"]["medium"]["pairs"], 1)
        self.assertEqual(categories["7"]["medium"]["pairs"], 1)
        self.assertEqual(categories["7"]["large"]["pairs"], 1)
        self.assertEqual(result["counts"]["invalid_original_area_pairs"], 1)
        self.assertEqual(result["groups"]["input_area_size"]["small"]["pairs"], 3)


if __name__ == "__main__":
    unittest.main()
