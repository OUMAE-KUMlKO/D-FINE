"""Perturbations must retain the normal model's query/GT assignment."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from src.zoo.dfine.matcher import HungarianMatcher
from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
from tools.analysis.layer_strip_effect_metrics import LayerEffectMetrics


class CandidateFeatureDiagnosticTests(unittest.TestCase):
    def test_feature_and_scale_pipelines_with_real_coco_and_prediction_cache(self):
        from faster_coco_eval.core.coco import COCO
        from src.zoo.dfine.postprocessor import DFINEPostProcessor
        from test_all_layer_strip import make_model
        from tools.analysis.diagnose_candidate_features import run
        from tools.analysis.diagnose_correction_scale import run as run_scales

        coco = COCO()
        coco.dataset = {
            'images': [{'id': 1, 'width': 24, 'height': 16}],
            'categories': [{'id': 0, 'name': 'object'}, {'id': 1, 'name': 'other'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0,
                             'bbox': [10, 6, 4, 4], 'area': 16, 'iscrowd': 0}],
        }
        coco.createIndex()

        class Dataset:
            def __len__(self):
                return 1

        class Loader(list):
            pass

        loader = Loader([(torch.randn(1, 3, 16, 24), [{
            'boxes': torch.tensor([[10., 6., 14., 10.]]),
            'labels': torch.tensor([0]), 'area': torch.tensor([16.]),
            'image_id': torch.tensor([1]), 'orig_size': torch.tensor([24, 16]),
        }])])
        loader.dataset = Dataset()
        loader.dataset.coco = coco
        cfg = SimpleNamespace(
            val_dataloader=loader,
            postprocessor=DFINEPostProcessor(num_classes=2, num_top_queries=3),
            yaml_cfg={'num_classes': 2, 'val_dataloader': {'dataset': {'synthetic': True}}},
        )
        model = make_model(use_log_prior=False, query_fusion='channel_gate',
                           normal_points=7, normal_kernel_size=3).eval()
        with torch.no_grad():
            for head in model.refiner.heads:
                head.score_head[-1].weight.normal_(std=.05)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                config='synthetic.yml', checkpoint='synthetic.pth', device='cpu',
                batch_size=1, workers=0, threads=1, seed=0, max_batches=None,
                expected_ap=None, ap_tolerance=.02,
                output_file=str(Path(directory) / 'summary.json'),
                prediction_cache=str(Path(directory) / 'predictions.pth'),
            )
            # Replace only loading boundaries; model, postprocess, matching and
            # COCO evaluation all run normally on the synthetic sample.
            with mock.patch('src.core.YAMLConfig', return_value=cfg), mock.patch(
                'tools.analysis.diagnose_candidate_features.load_ema_model',
                return_value=(model, {'weights': 'ema.module', 'checkpoint': 'synthetic.pth',
                                     'checkpoint_last_epoch': 96,
                                     'model_class': 'DFINEAllLayerStrip'}),
            ):
                summary = run(args)
            self.assertTrue(summary['full_validation'])
            self.assertTrue(summary['first_batch_restore_exact'])
            self.assertEqual(summary['fixed_pairs'], 1)
            self.assertEqual(set(summary['results']), {'normal', 'zero', 'noise'})
            self.assertEqual(json.loads(Path(args.output_file).read_text())['images'], 1)
            cache = torch.load(args.prediction_cache, weights_only=False)
            self.assertEqual(len(cache['records']), 1)
            self.assertEqual(cache['records'][0]['image_id'], 1)
            for result in summary['results'].values():
                self.assertEqual(result['fixed_geometry']['groups']['all']['pairs'], 1)

            scale_args = SimpleNamespace(
                config=args.config, checkpoint=args.checkpoint, device='cpu',
                batch_size=1, workers=0, threads=1,
                baseline_summary=args.output_file, baseline_cache=args.prediction_cache,
                output_file=str(Path(directory) / 'scales.json'),
                prediction_cache=str(Path(directory) / 'scales.pth'),
            )
            with mock.patch('src.core.YAMLConfig', return_value=cfg), mock.patch(
                'tools.analysis.diagnose_correction_scale.load_ema_model',
                return_value=(model, summary['model']),
            ):
                scaled = run_scales(scale_args)
            self.assertTrue(scaled['first_batch_reference_close'])
            self.assertEqual(scaled['new_inference_scales'], {'alpha_0': 0., 'alpha_0.5': .5})
            self.assertEqual(scaled['results']['alpha_1']['coco_percent'],
                             summary['results']['normal']['coco_percent'])
            self.assertEqual(scaled['fixed_pairs'], 1)
            self.assertEqual(set(scaled['results']), {'alpha_0', 'alpha_0.5', 'alpha_1'})
            scale_cache = torch.load(scale_args.prediction_cache, weights_only=False)
            self.assertEqual(len(scale_cache['records']), 1)
            self.assertEqual(set(scale_cache['records'][0]['predictions']),
                             {'alpha_0', 'alpha_0.5'})

    def test_swapped_queries_remain_wrong_under_fixed_geometry_pairs(self):
        from tools.analysis.diagnose_candidate_features import update_fixed_geometry

        boxes = torch.tensor([[[.25, .5, .1, .2], [.75, .5, .1, .2]]])
        logits = torch.tensor([[[10., -10.], [-10., 10.]]])
        outputs = {
            'normal': {'pred_boxes': boxes, 'pred_logits': logits},
            'zero': {'pred_boxes': boxes.flip(1), 'pred_logits': logits.flip(1)},
            'noise': {'pred_boxes': boxes, 'pred_logits': -logits},
        }
        targets = [{'boxes': boxes[0], 'labels': torch.tensor([0, 1]),
                    'area_original': torch.tensor([200., 200.])}]
        matcher = PredRefinementCriterion(HungarianMatcher(
            {'cost_class': 0, 'cost_bbox': 5, 'cost_giou': 2}, use_focal_loss=True
        )).eval()
        meters = {mode: LayerEffectMetrics() for mode in outputs}
        indices = update_fixed_geometry(outputs, targets, (100, 200), matcher, meters)
        self.assertEqual(indices[0][0].tolist(), [0, 1])
        self.assertEqual(indices[0][1].tolist(), [0, 1])
        swapped = meters['zero'].compute()['groups']['all']
        self.assertAlmostEqual(swapped['before']['mean_edge_mae_px'], 0.)
        self.assertAlmostEqual(swapped['after']['mean_edge_mae_px'], 50.)
        self.assertEqual(swapped['pairs'], 2)
        self.assertAlmostEqual(meters['noise'].compute()['groups']['all']
                               ['after']['mean_edge_mae_px'], 0.)

    def test_nonfinite_or_invalid_perturbed_boxes_are_not_silently_filtered(self):
        from tools.analysis.diagnose_candidate_features import validate_outputs

        for box in ([.5, .5, float('nan'), .2], [.5, .5, 0., .2]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_outputs({'pred_boxes': torch.tensor([[box]]),
                                  'pred_logits': torch.zeros(1, 1, 2)}, 'noise')
        with self.assertRaises(ValueError):
            validate_outputs({'pred_boxes': torch.tensor([[[.5, .5, .2, .2]]]),
                              'pred_logits': torch.tensor([[[float('inf'), 0.]]])}, 'zero')


if __name__ == '__main__':
    unittest.main()
