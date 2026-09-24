"""Per-layer strip integration for frozen-B0 and backbone-only training."""

import unittest
from pathlib import Path

import torch

from src.core import YAMLConfig
from src.zoo.dfine.all_layer_strip import DFINEAllLayerStrip, DFINEAllLayerStripRefiner
from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.matcher import HungarianMatcher
from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
from test_gt_refinement import ToyBackbone, ToyEncoder, normalized_targets


torch.set_num_threads(1)


def make_model(frozen=False, checkpoint_strips=True, **head_kwargs):
    decoder = DFINETransformer(
        num_classes=2, hidden_dim=8, num_queries=3,
        feat_channels=[8], feat_strides=[1], num_levels=1,
        num_points=2, nhead=2, num_layers=6, dim_feedforward=16,
        num_denoising=4, reg_max=4,
    )
    with torch.no_grad():
        for head in decoder.dec_bbox_head:
            head.layers[-1].weight.normal_(std=0.04)
    cls = DFINEAllLayerStripRefiner if frozen else DFINEAllLayerStrip
    model = cls(
        ToyBackbone(), ToyEncoder(), decoder, hidden_dim=8,
        query_chunk_size=2, checkpoint_strips=checkpoint_strips,
        **head_kwargs,
    )
    if frozen:
        detector = {k: v for k, v in model.state_dict().items()
                    if k.startswith(('backbone.', 'encoder.', 'decoder.'))}
        model.load_state_dict(detector, strict=False)
    return model


def matcher():
    return HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2})


class AllLayerStripTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.images = torch.randn(1, 3, 16, 24)

    def test_frozen_zero_init_calls_all_six_independent_heads_and_matches_b0(self):
        model = make_model(frozen=True).train()
        calls = []
        hooks = [head.register_forward_hook(
            lambda module, args, out, index=i: calls.append(index)
        ) for i, head in enumerate(model.refiner.heads)]
        try:
            output = model(self.images)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(calls, list(range(6)))
        self.assertEqual(len({id(h.score_head[-1].weight) for h in model.refiner.heads}), 6)
        torch.testing.assert_close(output['pred_boxes'], output['baseline_boxes'], atol=1e-6, rtol=1e-6)
        with torch.no_grad():
            baseline = model.decoder(model.encoder(model.backbone(self.images)))
        torch.testing.assert_close(output['pred_logits'], baseline['pred_logits'], atol=0, rtol=0)
        changed_gt = model(self.images, [{'unused': object()}])
        torch.testing.assert_close(output['pred_boxes'], changed_gt['pred_boxes'], atol=0, rtol=0)

    def test_frozen_checkpointed_training_reaches_every_head_but_not_detector(self):
        model = make_model(frozen=True).train()
        criterion = PredRefinementCriterion(matcher())
        losses = criterion(model(self.images), normalized_targets())
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

    def test_full_training_keeps_native_aux_dn_losses_and_strip_feature_gradients(self):
        model = make_model().train()
        with torch.no_grad():
            for head in model.refiner.heads:
                head.score_head[-1].weight.normal_(std=0.02)
        query_counts = []
        hook = model.refiner.heads[0].register_forward_pre_hook(
            lambda module, args: query_counts.append(args[1].shape[1])
        )
        try:
            output = model(self.images, normalized_targets())
        finally:
            hook.remove()
        self.assertEqual(query_counts, [3])  # Native denoising queries skip strips.
        self.assertEqual(len(output['aux_outputs']), 5)
        self.assertEqual(len(output['dn_outputs']), 6)
        criterion = DFINECriterion(
            matcher(), {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,
                        'loss_fgl': 0.15, 'loss_ddf': 1.5},
            ['vfl', 'boxes', 'local'], num_classes=2, reg_max=4,
        )
        losses = criterion(output, normalized_targets())
        self.assertIn('loss_fgl_aux_0', losses)
        self.assertIn('loss_fgl_dn_0', losses)
        sum(losses.values()).backward()
        for i, head in enumerate(model.refiner.heads):
            self.assertGreater(head.score_head[-1].weight.grad.abs().sum().item(), 0, i)
        for module in (model.backbone, model.encoder, model.decoder):
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in gradients))
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gradients))

    def test_checkpointing_preserves_outputs_and_gradients(self):
        plain = make_model(frozen=True, checkpoint_strips=False).train()
        checked = make_model(frozen=True, checkpoint_strips=True).train()
        checked.load_state_dict(plain.state_dict())
        first, second = plain(self.images), checked(self.images)
        torch.testing.assert_close(first['pred_boxes'], second['pred_boxes'], atol=0, rtol=0)
        first['pred_boxes'].square().sum().backward()
        second['pred_boxes'].square().sum().backward()
        for a, b in zip(plain.refiner.parameters(), checked.refiner.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)

    def test_configurations_select_two_initialization_and_training_modes(self):
        root = Path(__file__).resolve().parents[1]
        short = YAMLConfig(str(root / 'configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_all_layer_strip_12ep.yml'))
        full = YAMLConfig(str(root / 'configs/dfine/dfine_hgnetv2_l_uavswarm_all_layer_strip_full.yml'))
        self.assertEqual(short.yaml_cfg['model'], 'DFINEAllLayerStripRefiner')
        self.assertEqual(short.yaml_cfg['criterion'], 'PredRefinementCriterion')
        self.assertFalse(short.yaml_cfg['HGNetv2']['pretrained'])
        self.assertEqual(short.yaml_cfg['epochs'], 12)
        self.assertEqual(short.yaml_cfg['train_dataloader']['total_batch_size'], 96)
        self.assertEqual(full.yaml_cfg['model'], 'DFINEAllLayerStrip')
        self.assertEqual(full.yaml_cfg['criterion'], 'DFINECriterion')
        self.assertTrue(full.yaml_cfg['HGNetv2']['pretrained'])
        self.assertEqual(full.yaml_cfg['epochs'], 80)
        self.assertEqual(full.yaml_cfg['train_dataloader']['total_batch_size'], 64)
        self.assertNotIn('baseline_checkpoint', full.yaml_cfg['DFINEAllLayerStrip'])


if __name__ == '__main__':
    unittest.main()
