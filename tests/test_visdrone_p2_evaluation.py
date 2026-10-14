"""Both P2 evaluation modes share one real model forward and COCO pipeline."""

import contextlib
import io
from types import SimpleNamespace
import unittest

import torch

from tools.analysis.evaluate_visdrone_p2 import evaluate


class VisDroneP2EvaluationTests(unittest.TestCase):
    def test_matched_only_preserves_unmatched_boxes_and_classification(self):
        from faster_coco_eval.core.coco import COCO
        from src.zoo.dfine.matcher import HungarianMatcher
        from src.zoo.dfine.postprocessor import DFINEPostProcessor
        from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
        from test_pred_p2_integration import PredP2IntegrationTests

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(0)
        try:
            model = PredP2IntegrationTests().make_p2().eval()
            with torch.no_grad():
                model.refiner.refiner[-1].bias.fill_(.03)
            coco = COCO()
            coco.dataset = {
                'images': [{'id': 1, 'width': 32, 'height': 32}],
                'categories': [{'id': 0, 'name': 'object'}, {'id': 1, 'name': 'other'}],
                'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0,
                                 'bbox': [10, 10, 8, 8], 'area': 64, 'iscrowd': 0}],
            }
            coco.createIndex()

            class Dataset:
                def __len__(self):
                    return 1

            class Loader(list):
                pass

            loader = Loader([(torch.rand(1, 3, 32, 32), [{
                'boxes': torch.tensor([[10., 10., 18., 18.]]),
                'labels': torch.tensor([0]), 'area': torch.tensor([64.]),
                'image_id': torch.tensor([1]), 'orig_size': torch.tensor([32, 32]),
            }])])
            loader.dataset = Dataset()
            loader.dataset.coco = coco
            cfg = SimpleNamespace(
                val_dataloader=loader,
                postprocessor=DFINEPostProcessor(num_classes=2, num_top_queries=3),
                criterion=PredRefinementCriterion(HungarianMatcher(
                    {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                    use_focal_loss=True, alpha=.25, gamma=2.,
                )),
            )
            forwards, post_inputs = [], []
            model_handle = model.register_forward_hook(lambda _, __, out: forwards.append(out))
            post_handle = cfg.postprocessor.register_forward_pre_hook(lambda _, inputs: post_inputs.append(inputs[0]))
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(model, cfg, SimpleNamespace(device='cpu', expected_images=1))
            model_handle.remove()
            post_handle.remove()
            self.assertEqual(len(forwards), 1)
            self.assertEqual(len(post_inputs), 2)
            full, matched = post_inputs
            torch.testing.assert_close(full['pred_boxes'], forwards[0]['pred_boxes'], atol=0, rtol=0)
            torch.testing.assert_close(full['pred_logits'], matched['pred_logits'], atol=0, rtol=0)
            mask = matched['refinement_mask']
            self.assertLessEqual(int(mask.sum()), 1)
            torch.testing.assert_close(matched['pred_boxes'][~mask], full['baseline_boxes'][~mask], atol=0, rtol=0)
            torch.testing.assert_close(matched['pred_boxes'][mask], full['pred_boxes'][mask], atol=0, rtol=0)
            self.assertTrue(result['full_validation'])
            self.assertEqual(result['images'], 1)
            self.assertEqual(set(result['results']), {'all_queries', 'matched_only'})
            for mode in result['results'].values():
                self.assertEqual(mode['fixed_geometry']['counts']['matched'], 1)
                self.assertIn('AP', mode['coco_percent'])
            self.assertEqual(result['results']['all_queries']['fixed_geometry'],
                             result['results']['matched_only']['fixed_geometry'])
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
