import json
import unittest

import torch

from tools.analysis.layer_strip_effect_metrics import LayerEffectMetrics


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


class LayerEffectMetricsTests(unittest.TestCase):
    def test_fixed_nontrivial_pairs_and_rectangular_pixel_units(self):
        before = tensor([[[.1, .1, .1, .1], [.5, .5, .2, .4], [.7, .6, .2, .4]]])
        after = before.clone()
        after[0, 1, 0] += .01
        after[0, 2, 1] += .02
        targets = [{"boxes": tensor([[.7, .62, .2, .4], [.5, .5, .2, .4]]),
                    "labels": torch.tensor([3, 8]), "area_original": tensor([1200, 10000])}]
        metrics = LayerEffectMetrics()
        metrics.update(before, after, targets,
                       [(torch.tensor([2, 1]), torch.tensor([0, 1]))], (100, 200))
        result = metrics.compute()["groups"]
        self.assertEqual(result["all"]["pairs"], 2)
        self.assertAlmostEqual(result["all"]["before"]["mean_edge_mae_px"], .5)
        self.assertAlmostEqual(result["all"]["after"]["mean_edge_mae_px"], .5)
        self.assertAlmostEqual(result["all"]["mean_edge_movement_px"], 1.)
        self.assertAlmostEqual(result["category_3"]["after"]["mean_edge_mae_px"], 0.)
        self.assertEqual(result["category_8"]["iou_degraded"], 1)
        side = result["category_8"]["after"]["mean_edge_mae_by_side_px"]
        self.assertAlmostEqual(side["left"], 2.)
        self.assertAlmostEqual(side["right"], 2.)
        self.assertAlmostEqual(side["top"], 0.)

    def test_original_area_closed_ranges_and_input_short_side(self):
        boxes = tensor([[[.5, .5, .01, .01], [.5, .5, .016, .016],
                         [.5, .5, .032, .032], [.5, .5, .064, .064]]])
        targets = [{"boxes": boxes[0], "labels": torch.tensor([0, 1, 2, 3]),
                    "area_original": tensor([1024, 9216, 100, 20000])}]
        metrics = LayerEffectMetrics()
        metrics.update(boxes, boxes, targets, [(torch.arange(4), torch.arange(4))], (500, 500))
        groups = metrics.compute()["groups"]
        self.assertEqual(groups["original_small"]["pairs"], 2)
        self.assertEqual(groups["original_medium"]["pairs"], 2)
        self.assertEqual(groups["original_large"]["pairs"], 2)
        for name in ("short_lt8", "short_8_to_lt16", "short_16_to_lt32", "short_ge32"):
            self.assertEqual(groups[name]["pairs"], 1)

    def test_iou_directions_crossings_and_pre_iou_groups(self):
        target = tensor([[.5, .5, .2, .2]]).repeat(4, 1)
        before = target.clone()
        before[0, 0] += .1  # IoU 1/3 -> 1, crosses both thresholds.
        after = before.clone()
        after[0] = target[0]
        after[1, 0] += .1   # IoU 1 -> 1/3, crosses both thresholds.
        after[3, 0] += 1e-9 # Within the 1e-6 IoU stability tolerance.
        metrics = LayerEffectMetrics()
        metrics.update(before[None], after[None],
                       [{"boxes": target, "labels": torch.zeros(4, dtype=torch.long),
                         "area_original": tensor([100] * 4)}],
                       [(torch.arange(4), torch.arange(4))], (100, 100))
        groups = metrics.compute()["groups"]
        total = groups["all"]
        self.assertEqual((total["iou_improved"], total["iou_degraded"], total["iou_stable"]), (1, 1, 2))
        self.assertEqual(total["iou_improved_fraction"], .25)
        self.assertEqual(total["iou_threshold_crossings"]["0.5"], {"up": 1, "down": 1})
        self.assertEqual(total["iou_threshold_crossings"]["0.75"], {"up": 1, "down": 1})
        self.assertEqual(groups["before_iou_lt0_5"]["pairs"], 1)
        self.assertEqual(groups["before_iou_ge0_75"]["pairs"], 3)

    def test_fallback_mask_does_not_remove_pairs_and_width_summaries(self):
        boxes = tensor([[[.5, .5, .2, .2], [.4, .4, .1, .1]]])
        metrics = LayerEffectMetrics()
        metrics.update(boxes, boxes,
                       [{"boxes": boxes[0], "labels": torch.tensor([0, 1]),
                         "area_original": tensor([400, 100])}],
                       [(torch.arange(2), torch.arange(2))], (100, 100),
                       valid=torch.tensor([[False, True]]),
                       half_widths_px=tensor([[[2., 8.], [4., 2.]]]))
        total = metrics.compute()["groups"]["all"]
        self.assertEqual(total["pairs"], 2)
        self.assertEqual(total["valid_fraction"], .5)
        self.assertEqual(total["iou_stable"], 2)
        self.assertEqual(total["mean_half_width_x_px"], 3.)
        self.assertEqual(total["mean_half_width_y_px"], 5.)
        self.assertEqual(total["half_width_x_min_fraction"], .5)
        self.assertEqual(total["half_width_y_max_fraction"], .5)

    def test_invalid_pairs_and_empty_groups(self):
        before = tensor([[[.5, .5, .2, .2], [.5, .5, float("nan"), .2],
                          [.5, .5, .2, .2], [.5, .5, .2, .2]]])
        after = before.clone()
        after[0, 2, 2] = 0.
        gt = tensor([[.5, .5, .2, .2]]).repeat(4, 1)
        gt[3, 3] = -1.
        metrics = LayerEffectMetrics()
        metrics.update(before, after, [{"boxes": gt, "labels": torch.zeros(4, dtype=torch.long),
                                        "area_original": tensor([100] * 4)}],
                       [(torch.arange(4), torch.arange(4))], (100, 100))
        result = metrics.compute()
        self.assertEqual(result["invalid_pairs"], 3)
        self.assertEqual(result["groups"]["all"]["pairs"], 1)
        self.assertIsNone(result["groups"]["all"]["valid_fraction"])
        self.assertIsNone(result["groups"]["original_large"]["before"]["mean_iou"])
        self.assertIsNone(result["groups"]["original_large"]["mean_edge_movement_px"])
        self.assertEqual(result["groups"]["original_large"]["iou_improved"], 0)
        empty = LayerEffectMetrics().compute()["groups"]["all"]
        self.assertEqual(empty["pairs"], 0)
        self.assertIsNone(empty["mean_iou_change"])

    def test_multiple_updates_keep_weighted_means_and_empty_matches(self):
        metrics = LayerEffectMetrics()
        gt = tensor([[[.5, .5, .2, .2]]])
        shifted = gt.clone()
        shifted[0, 0, 0] += .1
        target = [{"boxes": gt[0], "labels": torch.tensor([12]),
                   "area_original": tensor([100])}]
        metrics.update(shifted, gt, target, [(torch.tensor([0]), torch.tensor([0]))], (100, 200))
        metrics.update(gt, gt, target, [(torch.tensor([0]), torch.tensor([0]))], (100, 200))
        metrics.update(gt, gt, target, [(torch.tensor([], dtype=torch.long),
                                        torch.tensor([], dtype=torch.long))], (100, 200))
        result = metrics.compute()
        total = result["groups"]["all"]
        self.assertEqual(total["pairs"], 2)
        self.assertAlmostEqual(total["before"]["mean_edge_mae_px"], 5.)
        self.assertAlmostEqual(total["edge_movement_p90_px"], 9.)
        self.assertEqual(result["groups"]["category_12"]["pairs"], 2)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False))["invalid_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
