"""The GT-selected diagnostic must preserve normal inference outputs."""

import unittest

import torch

from tools.analysis.diagnose_pred_refinement import build_diagnostic_variants


class DiagnosticVariantsTests(unittest.TestCase):
    def test_p2_name_preserves_refined_boxes_and_gt_selection(self):
        before = torch.zeros(1, 2, 4)
        after = torch.ones_like(before)
        logits = torch.zeros(1, 2, 1)
        outputs = {"baseline_boxes": before, "pred_boxes": after, "pred_logits": logits}
        variants = build_diagnostic_variants(
            outputs, [(torch.tensor([0]), torch.tensor([0]))], refinement_name="p2"
        )
        self.assertEqual(set(variants), {"baseline", "p2", "matched_only_oracle"})
        self.assertIs(variants["p2"]["pred_boxes"], after)
        self.assertIs(variants["p2"]["pred_logits"], logits)
        torch.testing.assert_close(variants["matched_only_oracle"]["pred_boxes"][0, 0], after[0, 0])
        torch.testing.assert_close(variants["matched_only_oracle"]["pred_boxes"][0, 1], before[0, 1])

    def test_only_matched_variant_changes_the_selected_query(self):
        before = torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2],
                                [0.8, 0.8, 0.1, 0.1]]])
        after = before + 0.01
        logits = torch.tensor([[[2.0], [3.0], [-1.0]]])
        outputs = {"baseline_boxes": before, "pred_boxes": after, "pred_logits": logits}
        variants = build_diagnostic_variants(
            outputs, [(torch.tensor([1]), torch.tensor([0]))]
        )
        expected = before.clone()
        expected[:, 1] = after[:, 1]
        torch.testing.assert_close(variants["baseline"]["pred_boxes"], before)
        torch.testing.assert_close(variants["p1"]["pred_boxes"], after)
        torch.testing.assert_close(variants["matched_only_oracle"]["pred_boxes"], expected)
        torch.testing.assert_close(outputs["pred_boxes"], before + 0.01)
        for variant in variants.values():
            self.assertIs(variant["pred_logits"], logits)

    def test_empty_gt_and_mixed_batch_keep_separate_selection(self):
        before = torch.zeros(2, 2, 4)
        after = torch.ones_like(before)
        empty = torch.empty(0, dtype=torch.long)
        outputs = {"baseline_boxes": before, "pred_boxes": after,
                   "pred_logits": torch.zeros(2, 2, 1)}
        variants = build_diagnostic_variants(
            outputs, [(empty, empty), (torch.tensor([0]), torch.tensor([0]))]
        )
        selected = variants["matched_only_oracle"]["pred_boxes"]
        torch.testing.assert_close(selected[0], before[0])
        torch.testing.assert_close(selected[1, 0], after[1, 0])
        torch.testing.assert_close(selected[1, 1], before[1, 1])


if __name__ == "__main__":
    unittest.main()
