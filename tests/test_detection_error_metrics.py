"""Detection error composition from fixed official COCO matching."""

import json
import unittest

from faster_coco_eval import COCO, COCOeval_faster

from tools.analysis.detection_error_metrics import analyze_predictions
from tools.analysis.evaluate_density_coco import extract_metrics


def make_coco():
    coco = COCO()
    coco.dataset = {
        "images": [
            {"id": 1, "width": 1000, "height": 200},
            {"id": 2, "width": 1000, "height": 200},
        ],
        "categories": [
            {"id": 1, "name": "one"},
            {"id": 2, "name": "two"},
            {"id": 3, "name": "no_gt"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1,
             "bbox": [40, 0, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 3, "image_id": 1, "category_id": 1,
             "bbox": [80, 0, 10, 10], "area": 100, "iscrowd": 1},
            {"id": 4, "image_id": 1, "category_id": 2,
             "bbox": [0, 40, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 5, "image_id": 2, "category_id": 1,
             "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
        ],
    }
    coco.createIndex()
    return coco


def make_predictions():
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .99},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .98},
        {"image_id": 1, "category_id": 1, "bbox": [0, 40, 10, 10], "score": .97},
        {"image_id": 1, "category_id": 1, "bbox": [45, 0, 10, 10], "score": .96},
        {"image_id": 1, "category_id": 1, "bbox": [5, 40, 10, 10], "score": .95},
        {"image_id": 1, "category_id": 1, "bbox": [100, 100, 10, 10], "score": .94},
        {"image_id": 1, "category_id": 1, "bbox": [80, 0, 10, 10], "score": .93},
        {"image_id": 1, "category_id": 1, "bbox": [40, 0, 10, 10], "score": .89},
        {"image_id": 1, "category_id": 2, "bbox": [0, 40, 10, 10], "score": .88},
        {"image_id": 1, "category_id": 3, "bbox": [120, 100, 10, 10], "score": .87},
    ]
    # Stable same-score ordering must retain these 100 background detections
    # and truncate the true positive placed at position 101.
    predictions.extend(
        {"image_id": 2, "category_id": 1,
         "bbox": [20 + 8 * index, 100, 5, 5], "score": .2}
        for index in range(100)
    )
    predictions.append(
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .2}
    )
    return predictions


class DetectionErrorMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coco = make_coco()
        cls.predictions = make_predictions()
        cls.summary = analyze_predictions(cls.coco, cls.predictions)

    def test_original_metrics_match_official_faster_eval_and_are_json_safe(self):
        evaluator = COCOeval_faster(
            self.coco, self.coco.loadRes(self.predictions), "bbox",
            print_function=lambda message: None,
        )
        evaluator.evaluate()
        evaluator.accumulate()
        expected = extract_metrics(evaluator)
        self.assertEqual(self.summary["metrics"], expected)
        self.assertLessEqual(
            self.summary["precision_validation"]["pycocotools_max_abs"], 1e-7,
        )
        self.assertLessEqual(
            self.summary["precision_validation"]["faster_coco_eval_max_abs"], 1e-7,
        )
        json.dumps(self.summary, allow_nan=False)

    def test_counts_cover_all_errors_ignore_crowd_and_keep_fn(self):
        counts = self.summary["counts"]["0.5"]["0.50"]["overall"]
        self.assertEqual(counts, {
            "tp": 3, "duplicate": 1, "classification": 1,
            "localization": 1, "classification_localization": 1,
            "background": 2, "fp": 6, "fn": 1,
        })
        category = self.summary["counts"]["0.5"]["0.50"]["categories"]
        self.assertEqual(category["1"]["tp"], 2)
        self.assertEqual(category["1"]["fn"], 1)
        self.assertEqual(category["3"]["background"], 1)
        self.assertEqual(category["3"]["fn"], 0)
        self.assertEqual(self.summary["categories"]["3"]["gt_count"], 0)
        self.assertIsNone(self.summary["categories"]["3"]["AP"])

    def test_max_dets_and_stable_ties_exclude_the_101st_detection(self):
        counts = self.summary["counts"]["0.1"]["0.50"]
        self.assertEqual(counts["categories"]["1"]["tp"], 2)
        self.assertEqual(counts["categories"]["1"]["fn"], 1)
        self.assertEqual(counts["categories"]["1"]["background"], 101)
        self.assertEqual(counts["overall"]["background"], 102)

    def test_independent_fp_removal_improves_ap_without_repairing_fn(self):
        original = self.summary["original"]["AP"]
        for error in (
            "duplicate", "classification", "localization",
            "classification_localization", "background",
        ):
            oracle = self.summary["oracle"][error]["overall"]
            self.assertGreater(oracle["AP"], original, error)
            self.assertGreater(oracle["AP_gain_pp"], 0, error)
            self.assertLess(oracle["AP"], 1, error)
        self.assertEqual(self.summary["counts"]["0"]["0.50"]["overall"]["fn"], 1)

    def test_nonpositive_gt_ids_are_normalized_without_mutating_input(self):
        coco = COCO()
        coco.dataset = {
            "images": [{"id": 7, "width": 20, "height": 20}],
            "categories": [{"id": 3, "name": "car"}],
            "annotations": [{
                "id": 0, "image_id": 7, "category_id": 3,
                "bbox": [2, 2, 8, 8], "area": 64, "iscrowd": 0,
            }],
        }
        coco.createIndex()
        summary = analyze_predictions(coco, [{
            "image_id": 7, "category_id": 3,
            "bbox": [2, 2, 8, 8], "score": .9,
        }])

        self.assertEqual(summary["legacy_metrics"]["size_groups"]["all"]["AP"], 0)
        self.assertEqual(summary["metrics"]["size_groups"]["all"]["AP"], 1)
        self.assertEqual(summary["annotation_id_normalization"], {
            "applied": True, "offset": 1, "nonpositive_gt_count": 1,
            "strategy": "add one constant offset to every ground-truth annotation id",
        })
        self.assertEqual(coco.dataset["annotations"][0]["id"], 0)
        self.assertEqual(coco.anns[0]["id"], 0)
        counts = summary["counts"]["0"]["0.50"]["overall"]
        self.assertEqual(counts["tp"], 1)
        self.assertEqual(counts["fp"], 0)
        self.assertEqual(counts["fn"], 0)
        self.assertLessEqual(summary["precision_validation"]["pycocotools_max_abs"], 1e-7)
        self.assertLessEqual(summary["precision_validation"]["faster_coco_eval_max_abs"], 1e-7)


if __name__ == "__main__":
    unittest.main()
