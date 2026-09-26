"""Train all six query gates through the native decoder/DN losses."""

import unittest

import torch

from src.zoo.dfine.dfine_criterion import DFINECriterion
from test_all_layer_strip import make_model, matcher
from test_gt_refinement import normalized_targets


class AllLayerStripQueryGateTests(unittest.TestCase):
    def test_joint_training_reaches_all_gates_visual_encoders_and_detector(self):
        torch.manual_seed(41)
        model = make_model(
            normal_points=7, normal_kernel_size=3, offset_pixels=8.0,
            use_log_prior=False, query_fusion='channel_gate',
        ).train()
        self.assertEqual(len(model.refiner.heads), 6)
        with torch.no_grad():
            for head in model.refiner.heads:
                self.assertEqual(head.query_fusion, 'channel_gate')
                self.assertEqual(head.score_head[0].in_features, 8 + 8 + 1)
                self.assertEqual(torch.count_nonzero(head.query_projection.weight).item(), 0)
                self.assertEqual(torch.count_nonzero(head.query_projection.bias).item(), 0)
                # The zero-initialized residual output learns first. Verify the
                # upstream gates and visual features once that output is nonzero.
                head.score_head[-1].weight.normal_(std=0.02)
        images = torch.randn(1, 3, 64, 96)
        output = model(images, normalized_targets())
        self.assertEqual(len(output['aux_outputs']), 5)
        self.assertEqual(len(output['dn_outputs']), 6)
        criterion = DFINECriterion(
            matcher(), {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,
                        'loss_fgl': 0.15, 'loss_ddf': 1.5},
            ['vfl', 'boxes', 'local'], num_classes=2, reg_max=4,
        )
        losses = criterion(output, normalized_targets())
        sum(losses.values()).backward()
        for i, head in enumerate(model.refiner.heads):
            for parameter in (head.query_projection.weight, head.query_projection.bias,
                              head.strip_encoder[0].weight):
                self.assertIsNotNone(parameter.grad, i)
                self.assertTrue(torch.isfinite(parameter.grad).all(), i)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, i)
        for module in (model.backbone, model.encoder, model.decoder):
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in grads))


if __name__ == '__main__':
    unittest.main()
