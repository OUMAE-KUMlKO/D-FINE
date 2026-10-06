import unittest
import torch

from tools.analysis.evaluate_density_coco import density_bin, postprocess_records


class DensityCocoTests(unittest.TestCase):
    def test_density_boundaries(self):
        self.assertEqual([density_bin(n) for n in (0, 30, 31, 60, 61, 100, 101)],
                         ['le30', 'le30', '31to60', '31to60', '61to100', '61to100', 'gt100'])

    def test_true_postprocessor_keeps_multiple_classes_per_query_and_scales_xy(self):
        logits = torch.full((300, 10), -10.0)
        logits[7, 2], logits[7, 4] = 10.0, 9.0
        boxes = torch.tensor([.5, .5, .2, .4]).repeat(300, 1)
        record = {'image_id': 42, 'orig_size': [100, 200],
                  'baseline': {'pred_boxes': boxes, 'pred_logits': logits}}
        results = postprocess_records([record], 'baseline')
        self.assertEqual(len(results), 300)
        self.assertEqual([r['category_id'] for r in results[:2]], [2, 4])
        for item in results[:2]:
            self.assertEqual(item['image_id'], 42)
            torch.testing.assert_close(torch.tensor(item['bbox']), torch.tensor([80., 30., 40., 40.]))


if __name__ == '__main__':
    unittest.main()
