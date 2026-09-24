"""Integration checks for boundary-aligned 2-D bands in all decoder layers."""

import unittest
from pathlib import Path

import torch

from src.core import YAMLConfig
from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
from test_all_layer_strip import make_model, matcher
from test_gt_refinement import normalized_targets


class AllLayerStripBandTests(unittest.TestCase):
    def test_band_configuration_reaches_all_heads_in_both_wrappers(self):
        for frozen in (True, False):
            with self.subTest(frozen=frozen):
                model = make_model(
                    frozen=frozen, normal_points=7,
                    normal_kernel_size=3, offset_pixels=8.0,
                )
                self.assertEqual(len(model.refiner.heads), 6)
                for head in model.refiner.heads:
                    self.assertEqual(head.num_points, 5)
                    self.assertEqual(head.normal_points, 7)
                    self.assertEqual(head.offset_pixels, 8.0)
                    self.assertEqual(head.strip_encoder[0].kernel_size, (3, 3))
                    self.assertEqual(head.edge_fusion[0].in_features, 7 * 8)
                    self.assertEqual(head.score_head[0].in_features, 2 * 8 + 8 + 2)

    def test_checkpointed_band_preserves_b0_at_init_and_trains_all_six_heads(self):
        torch.manual_seed(41)
        images = torch.randn(1, 3, 64, 96)
        model = make_model(
            frozen=True, normal_points=7,
            normal_kernel_size=3, offset_pixels=8.0,
        ).train()
        output = model(images)
        torch.testing.assert_close(
            output['pred_boxes'], output['baseline_boxes'], atol=1e-6, rtol=1e-6,
        )
        losses = PredRefinementCriterion(matcher())(output, normalized_targets())
        sum(losses.values()).backward()
        for i, head in enumerate(model.refiner.heads):
            self.assertGreater(head.score_head[-1].weight.grad.abs().sum().item(), 0, i)
        for name, parameter in model.named_parameters():
            if name.startswith('refiner.'):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)

    def test_experiment_inherits_twelve_epoch_training_setup(self):
        root = Path(__file__).resolve().parents[1]
        config = YAMLConfig(str(root / 'configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_all_layer_band2d_12ep.yml'))
        values = config.yaml_cfg
        self.assertEqual(values['model'], 'DFINEAllLayerStripRefiner')
        self.assertEqual(values['criterion'], 'PredRefinementCriterion')
        self.assertEqual(values['epochs'], 12)
        self.assertEqual(values['train_dataloader']['total_batch_size'], 96)
        self.assertFalse(values['HGNetv2']['pretrained'])
        self.assertEqual(values['DFINEAllLayerStripRefiner']['normal_points'], 7)
        self.assertEqual(values['DFINEAllLayerStripRefiner']['normal_kernel_size'], 3)
        self.assertEqual(values['DFINEAllLayerStripRefiner']['offset_pixels'], 8.0)
        self.assertTrue(values['output_dir'].endswith('/all_layer_band2d_12ep'))

    def test_joint_band_training_reaches_detector_and_all_six_visual_encoders(self):
        torch.manual_seed(41)
        model = make_model(
            normal_points=7, normal_kernel_size=3, offset_pixels=8.0,
        ).train()
        # After the zero-initialized output projection starts learning, gradients
        # must reach the spatial encoders as well as the original detector.
        with torch.no_grad():
            for head in model.refiner.heads:
                head.score_head[-1].weight.normal_(std=0.02)
        output = model(torch.randn(1, 3, 64, 96), normalized_targets())
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
            grad = head.strip_encoder[0].weight.grad
            self.assertTrue(torch.isfinite(grad).all(), i)
            self.assertGreater(grad.abs().sum().item(), 0, i)
        for module in (model.backbone, model.encoder, model.decoder):
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in grads))


if __name__ == '__main__':
    unittest.main()
