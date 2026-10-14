"""Exercise oracle pooling evaluation with real model and COCO metrics."""

import contextlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from tools.analysis.diagnose_candidate_selection import evaluate_oracle


class CandidateSelectionRunnerTests(unittest.TestCase):
    def test_oracle_modes_keep_scores_matches_and_restore_head(self):
        from faster_coco_eval.core.coco import COCO
        from src.zoo.dfine.postprocessor import DFINEPostProcessor
        from test_gt_candidate_refinement import make_model

        torch.set_num_threads(1)
        torch.manual_seed(9)
        model = make_model().eval()
        with torch.no_grad():
            model.refiner.refiner[-1].weight.normal_(std=.05)
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
        cfg = SimpleNamespace(val_dataloader=loader,
                              postprocessor=DFINEPostProcessor(num_classes=2, num_top_queries=3))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_oracle(model, cfg, SimpleNamespace(
                device='cpu', max_batches=None, expected_ap=None, ap_tolerance=.02,
                prediction_cache=str(Path(folder)/'predictions.pth')))
            self.assertTrue(Path(folder,'predictions.pth').exists())
        self.assertEqual(result['images'], 1)
        self.assertTrue(result['full_validation'])
        self.assertTrue(result['first_batch_restore_exact'])
        self.assertEqual(set(result['results']), {'gt_nearest', 'prior_argmax', 'prior_weighted'})
        for mode in result['results'].values():
            self.assertEqual(mode['fixed_geometry']['counts']['matched'], 1)
        self.assertNotIn('_distribution_forward', model.refiner.__dict__)


if __name__ == '__main__':
    unittest.main()
