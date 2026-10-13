import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tools.analysis.cache_detector_predictions import check_record, run


class CacheDetectorTests(unittest.TestCase):
    def test_real_postprocess_coco_and_reference_cache_alignment(self):
        from faster_coco_eval import COCO
        from src.zoo.dfine.postprocessor import DFINEPostProcessor

        coco = COCO()
        coco.dataset = {
            'images': [{'id': 1, 'width': 24, 'height': 16}],
            'categories': [{'id': 0, 'name': 'object'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 0,
                             'bbox': [6, 4, 12, 8], 'area': 96, 'iscrowd': 0}],
        }
        coco.createIndex()

        class Dataset:
            def __len__(self):
                return 1

        class Loader(list):
            pass

        class Detector(torch.nn.Module):
            def forward(self, images):
                return {'pred_boxes': torch.tensor([[[.5, .5, .5, .5]]]),
                        'pred_logits': torch.tensor([[[9.]]])}

        original = {'boxes': torch.tensor([[6., 4., 18., 12.]]),
                    'labels': torch.tensor([0]), 'area': torch.tensor([96.]),
                    'image_id': torch.tensor([1]), 'orig_size': torch.tensor([24, 16])}
        loader = Loader([(torch.ones(1, 3, 16, 24), [original])])
        loader.dataset = Dataset()
        loader.dataset.coco = coco
        cfg = SimpleNamespace(val_dataloader=loader,
                              postprocessor=DFINEPostProcessor(num_classes=1, num_top_queries=1),
                              yaml_cfg={'val_dataloader': {'dataset': {'synthetic': True}}})
        target = {'boxes': torch.tensor([[.5, .5, .5, .5]]), 'labels': torch.tensor([0]),
                  'area_original': torch.tensor([96.])}
        record = {'image_id': 1, 'input_size_hw': [16, 24],
                  'orig_size_wh': [24, 16], 'target': target}
        for changed in ({'orig_size_wh': [16, 24]}, {'input_size_hw': [24, 16]},
                        {'target': {**target, 'labels': torch.tensor([1])}}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_record({**record, **changed}, target, original, (16, 24))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {'checkpoint': 'synthetic.pth', 'checkpoint_last_epoch': 96,
                        'weights': 'ema.module', 'model_class': 'Detector'}
            (root / 'reference.json').write_text(json.dumps({
                'model': metadata, 'full_validation': True, 'inference_precision': 'float32'}))
            torch.save({'schema_version': 1, 'model': metadata, 'records': [record]},
                       root / 'reference.pth')
            args = SimpleNamespace(
                config='synthetic.yml', checkpoint='synthetic.pth', device='cpu',
                batch_size=1, workers=0, threads=1, expected_ap=100.,
                output_file=str(root / 'summary.json'), prediction_cache=str(root / 'predictions.pth'),
                reference_summary=str(root / 'reference.json'),
                reference_cache=str(root / 'reference.pth'))
            with mock.patch('src.core.YAMLConfig', return_value=cfg), mock.patch(
                    'tools.analysis.cache_detector_predictions.load_ema_model',
                    return_value=(Detector().eval(), metadata)):
                summary = run(args)
            self.assertAlmostEqual(summary['results']['normal']['coco_percent']['AP'], 100.)
            self.assertTrue(summary['all_image_sizes_and_gt_equal'])
            cached = torch.load(args.prediction_cache, weights_only=False)['records'][0]
            self.assertEqual(cached['orig_size_wh'], [24, 16])
            self.assertEqual(set(cached['predictions']), {'normal'})
            torch.testing.assert_close(cached['predictions']['normal']['pred_boxes'], target['boxes'])


if __name__ == '__main__':
    unittest.main()
