import unittest

import torch

from tools.analysis.compare_detection_errors import postprocess, recall_precision_decomposition


class PostprocessCacheTests(unittest.TestCase):
    def test_recall_ceiling_uses_per_class_grid_and_excludes_no_gt_classes(self):
        def analysis(ap, recall):
            return {'original': {'AP': ap}, 'iou_metrics': {'categories': {
                'a': {'0.50': {'AR100': recall}},
                'b': {'0.50': {'AR100': 0.}},
                'no_gt': {'0.50': {'AR100': None}},
            }}}
        result = recall_precision_decomposition({'g0': analysis(.2, .5),
                                                 'g5': analysis(.19, .4)})
        self.assertAlmostEqual(result['g0']['recall_ceiling_percent'], 100 * 51 / 202)
        self.assertAlmostEqual(result['g5']['recall_ceiling_percent'], 100 * 41 / 202)
        self.assertAlmostEqual(result['delta_ap_pp'], -1.)
        self.assertAlmostEqual(result['delta_ap_pp'], result['delta_recall_ceiling_pp']
                               - result['delta_fp_precision_loss_pp'])

    def test_original_width_height_and_multiple_classes_per_query_are_preserved(self):
        logits = torch.full((30, 10), -10.)
        logits[0, 0], logits[0, 1] = 9., 8.
        record = {'orig_size_wh': [200, 100], 'predictions': {'normal': {
            'pred_boxes': torch.tensor([[.5, .5, .2, .4]]).repeat(30, 1),
            'pred_logits': logits}}}
        rows = postprocess({7: record})
        self.assertEqual(len(rows), 300)
        self.assertEqual([r['category_id'] for r in rows[:2]], [0, 1])
        self.assertEqual([r['image_id'] for r in rows[:2]], [7, 7])
        for row in rows[:2]:
            torch.testing.assert_close(torch.tensor(row['bbox']), torch.tensor([80., 30., 40., 40.]))


if __name__ == '__main__':
    unittest.main()
