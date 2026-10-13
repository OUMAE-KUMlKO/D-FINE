"""Cached normal references must follow image IDs and unchanged GT."""

import unittest

import torch


class CorrectionScaleReferenceTests(unittest.TestCase):
    def records(self):
        records, originals, targets = {}, [], []
        for image_id in (22, 11):
            boxes = torch.tensor([[image_id / 100., .5, .1, .2]])
            target = {'boxes': boxes, 'labels': torch.tensor([0]),
                      'area_original': torch.tensor([20.])}
            records[image_id] = {
                'input_size_hw': [100, 200], 'orig_size_wh': [400, 200],
                'target': target,
                'normal_geometry_matches': (torch.tensor([0]), torch.tensor([0])),
                'predictions': {'normal': {'pred_boxes': boxes, 'pred_logits': torch.zeros(1, 2)}},
            }
            originals.append({'image_id': torch.tensor([image_id]),
                              'orig_size': torch.tensor([400, 200])})
            targets.append({key: value.clone() for key, value in target.items()})
        return records, originals, targets

    def test_reference_lookup_uses_image_ids_and_preserves_cached_pairs(self):
        from tools.analysis.diagnose_correction_scale import reference_batch
        records, originals, targets = self.records()
        # Cache insertion order differs from the current batch order.
        records = dict(reversed(list(records.items())))
        outputs, indices = reference_batch(records, originals, targets, (100, 200), 'cpu')
        self.assertEqual(len(indices), 2)
        torch.testing.assert_close(outputs['pred_boxes'][:, 0, 0], torch.tensor([.22, .11]))
        self.assertEqual(indices[0][0].tolist(), [0])
        self.assertEqual(indices[1][1].tolist(), [0])

    def test_reference_rejects_unknown_image_changed_size_or_changed_gt(self):
        from tools.analysis.diagnose_correction_scale import reference_batch
        for change in ('id', 'size', 'gt'):
            records, originals, targets = self.records()
            if change == 'id':
                originals[0]['image_id'] = torch.tensor([999])
            elif change == 'size':
                records[22]['input_size_hw'] = [200, 200]
            else:
                targets[0]['boxes'][0, 0] += .1
            with self.subTest(change=change), self.assertRaises(ValueError):
                reference_batch(records, originals, targets, (100, 200), 'cpu')


if __name__ == '__main__':
    unittest.main()
