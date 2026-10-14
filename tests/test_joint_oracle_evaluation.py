"""GT-conditioned joint models need oracle labels, not frozen-pair metrics."""

import contextlib
import io
import unittest

import torch
from faster_coco_eval.core.coco import COCO

from src.data import CocoEvaluator
from src.solver.det_engine import evaluate
from src.zoo.dfine.postprocessor import DFINEPostProcessor


class ToyDetector(torch.nn.Module):
    def __init__(self, *, requires_gt, fixed_baseline=False):
        super().__init__()
        self.requires_gt = requires_gt
        self.fixed_baseline = fixed_baseline
        self.received_targets = None

    def forward(self, images, targets=None):
        self.received_targets = targets
        boxes = images.new_tensor([[[.4375, .4375, .25, .25]]])
        outputs = {"pred_boxes": boxes, "pred_logits": images.new_tensor([[[8.]]])}
        if self.fixed_baseline:
            index = torch.tensor([0], device=images.device)
            outputs.update({
                "baseline_boxes": boxes.clone(),
                "matched_indices": [(index, index)],
                "refinement_targets": [{"boxes": boxes[0]}],
                "refinement_mask": torch.ones((1, 1), dtype=torch.bool, device=images.device),
                "refinement_counts": {"matched": 1, "refined": 1, "total_gt": 1},
            })
        return outputs


class JointOracleEvaluationTests(unittest.TestCase):
    def run_evaluation(self, model):
        torch.set_num_threads(1)
        coco = COCO()
        coco.dataset = {
            "images": [{"id": 1, "width": 32, "height": 32}],
            "categories": [{"id": 0, "name": "object"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                             "bbox": [10, 10, 8, 8], "area": 64, "iscrowd": 0}],
        }
        target = {
            "boxes": torch.tensor([[10., 10., 18., 18.]]),
            "labels": torch.tensor([0]), "image_id": torch.tensor([1]),
            "orig_size": torch.tensor([32, 32]),
        }
        original_boxes = target["boxes"].clone()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            coco.createIndex()
            evaluator = CocoEvaluator(coco, ["bbox"])
            stats, returned_evaluator = evaluate(
                model, torch.nn.Identity(),
                DFINEPostProcessor(num_classes=1, num_top_queries=1),
                [(torch.zeros(1, 3, 32, 32), [target])], evaluator,
                torch.device("cpu"), epoch=0, use_wandb=False,
            )
        self.assertIs(returned_evaluator, evaluator)
        self.assertAlmostEqual(stats["coco_eval_bbox"][0], 1., places=6)
        torch.testing.assert_close(target["boxes"], original_boxes)
        return stats, buffer.getvalue()

    def test_joint_gt_model_skips_fixed_pair_metrics_but_remains_oracle(self):
        model = ToyDetector(requires_gt=True)
        model.oracle_evaluation = True
        model.supports_gt_refinement_metrics = False
        stats, log = self.run_evaluation(model)
        self.assertIsNotNone(model.received_targets)
        self.assertTrue(stats["oracle_evaluation"])
        self.assertNotIn("oracle_geometry", stats)
        self.assertIn("ORACLE COCO results", log)
        self.assertIn("not GT-free", log)
        self.assertNotIn("baseline scores", log)

    def test_legacy_oracle_keeps_default_fixed_pair_metrics(self):
        model = ToyDetector(requires_gt=True, fixed_baseline=True)
        stats, log = self.run_evaluation(model)
        self.assertIsNotNone(model.received_targets)
        self.assertTrue(stats["oracle_evaluation"])
        self.assertEqual(stats["oracle_geometry"]["groups"]["all"]["pairs"], 1)
        self.assertIn("fixed baseline scores", log)

    def test_regular_detector_receives_no_targets_and_no_oracle_stats(self):
        model = ToyDetector(requires_gt=False)
        stats, log = self.run_evaluation(model)
        self.assertIsNone(model.received_targets)
        self.assertNotIn("oracle_evaluation", stats)
        self.assertNotIn("oracle_geometry", stats)
        self.assertNotIn("ORACLE", log)


if __name__ == "__main__":
    unittest.main()
