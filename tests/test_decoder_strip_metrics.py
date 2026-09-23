import unittest

import torch

from tools.analysis.decoder_strip_metrics import FixedPairComparison


torch.set_num_threads(1)


class FixedPairComparisonTests(unittest.TestCase):
    def test_reference_iou_controls_groups_even_when_before_box_differs(self):
        metric = FixedPairComparison()
        target = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        metric.update(
            target.reshape(1, 1, 4),
            torch.tensor([[[0.1, 0.1, 0.2, 0.2]]]),
            target.reshape(1, 1, 4),
            [{"boxes": target}],
            [(torch.tensor([0]), torch.tensor([0]))],
            (200, 200),
        )

        groups = metric.compute()["groups"]
        self.assertEqual(groups["baseline_iou_ge_0_75"]["pairs"], 1)
        self.assertEqual(groups["baseline_iou_lt_0_50"]["pairs"], 0)
        self.assertEqual(groups["all"]["before"]["mean_iou"], 0.0)
        self.assertEqual(groups["all"]["after"]["mean_iou"], 1.0)

    def test_uses_fixed_indices_and_non_square_pixel_scaling(self):
        metric = FixedPairComparison()
        targets = torch.tensor([[0.2, 0.2, 0.1, 0.1],
                                [0.5, 0.5, 0.2, 0.4]])
        reference = torch.tensor([[[0.9, 0.9, 0.1, 0.1],
                                   [0.1, 0.1, 0.1, 0.1],
                                   [0.5, 0.5, 0.2, 0.4]]])
        before = reference.clone()
        before[0, 2, 0] += 0.05
        after = reference.clone()
        metric.update(
            reference, before, after, [{"boxes": targets}],
            [(torch.tensor([2]), torch.tensor([1]))], (100, 200),
        )

        group = metric.compute()["groups"]["all"]
        self.assertEqual(group["pairs"], 1)
        self.assertAlmostEqual(group["before"]["mean_edge_mae_px"], 5.0, places=5)
        self.assertEqual(group["after"]["mean_edge_mae_px"], 0.0)
        self.assertAlmostEqual(group["mean_edge_movement_px"], 5.0, places=5)
        self.assertAlmostEqual(group["mean_edge_mae_change_px"], -5.0, places=5)

    def test_p90_uses_all_pairs_across_updates(self):
        metric = FixedPairComparison()
        target = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        for horizontal_shift in (0.02, 0.06):
            before = target.reshape(1, 1, 4).clone()
            after = before.clone()
            after[0, 0, 0] += horizontal_shift
            metric.update(
                before, before, after, [{"boxes": target}],
                [(torch.tensor([0]), torch.tensor([0]))], (100, 100),
            )

        group = metric.compute()["groups"]["all"]
        self.assertEqual(group["pairs"], 2)
        self.assertAlmostEqual(group["mean_edge_movement_px"], 2.0, places=5)
        self.assertAlmostEqual(group["edge_movement_p90_px"], 2.8, places=5)

    def test_empty_and_invalid_pairs_report_none_without_nan(self):
        metric = FixedPairComparison()
        empty = torch.empty(1, 0, 4)
        metric.update(
            empty, empty, empty, [{"boxes": torch.empty(0, 4)}],
            [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))],
            (100, 100),
        )
        bad = torch.tensor([[[0.5, 0.5, -0.1, 0.2]]])
        metric.update(
            bad, bad, bad,
            [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}],
            [(torch.tensor([0]), torch.tensor([0]))], (100, 100),
        )

        report = metric.compute()
        self.assertEqual(report["invalid_pairs"], 1)
        group = report["groups"]["all"]
        self.assertEqual(group["pairs"], 0)
        self.assertIsNone(group["before"]["mean_iou"])
        self.assertIsNone(group["edge_movement_p90_px"])


if __name__ == "__main__":
    unittest.main()
