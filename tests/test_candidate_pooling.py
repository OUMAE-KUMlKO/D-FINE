"""Inference-only pooling interventions preserve the oracle model outside scope."""

import unittest

import torch

from tools.analysis.candidate_pooling import candidate_pooling, pooling_indices


class CandidatePoolingTests(unittest.TestCase):
    def test_argmax_excludes_invalid_candidates_and_weighted_uses_native_pool(self):
        logits = torch.tensor([[[99., 2., 1.], [3., 4., 8.]]])
        valid = torch.tensor([[[False, True, True], [False, False, False]]])
        selected = pooling_indices('prior_argmax', logits, valid)
        self.assertEqual(selected.tolist(), [[1, -1]])
        self.assertIsNone(pooling_indices('prior_weighted', logits, valid))

    def test_modes_restore_original_method_and_preserve_frozen_scores(self):
        from test_gt_candidate_refinement import make_model
        from test_gt_refinement import normalized_targets

        torch.manual_seed(9)
        model = make_model().eval()
        with torch.no_grad():
            model.refiner.refiner[-1].weight.normal_(std=.05)
        images = torch.rand(1, 3, 32, 32)
        gt = normalized_targets()
        with torch.inference_mode():
            original = model(images, gt, target_format='cxcywh_normalized')
            boxes_by_mode = {}
            for mode in ('gt_nearest', 'prior_argmax', 'prior_weighted'):
                with candidate_pooling(model, mode):
                    result = model(images, gt, target_format='cxcywh_normalized')
                self.assertNotIn('_distribution_forward', model.refiner.__dict__)
                torch.testing.assert_close(result['pred_logits'], original['pred_logits'], rtol=0, atol=0)
                torch.testing.assert_close(result['baseline_boxes'], original['baseline_boxes'], rtol=0, atol=0)
                boxes_by_mode[mode] = result['pred_boxes']
            self.assertFalse(torch.equal(boxes_by_mode['gt_nearest'], boxes_by_mode['prior_weighted']))
            restored = model(images, gt, target_format='cxcywh_normalized')
            torch.testing.assert_close(restored['pred_boxes'], original['pred_boxes'], rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with candidate_pooling(model, 'prior_weighted'):
                raise RuntimeError('probe')
        self.assertNotIn('_distribution_forward', model.refiner.__dict__)
        model.train()
        with self.assertRaises(ValueError):
            with candidate_pooling(model, 'prior_argmax'):
                pass


if __name__ == '__main__':
    unittest.main()
