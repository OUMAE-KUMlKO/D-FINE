"""Frozen-detector integration for decoder-internal strip feedback."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.core import YAMLConfig
from src.zoo.dfine.decoder_strip_refinement import DFINEDecoderStripRefiner
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.matcher import HungarianMatcher
from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
from test_gt_refinement import ToyBackbone, ToyEncoder, normalized_targets


torch.set_num_threads(1)


def make_decoder_strip(loaded=True, **kwargs):
    decoder = DFINETransformer(
        num_classes=2, hidden_dim=8, num_queries=3,
        feat_channels=[8], feat_strides=[1], num_levels=1,
        num_points=2, nhead=2, num_layers=3, dim_feedforward=16,
        num_denoising=0, reg_max=4,
    )
    # Real pretrained bbox heads are nonzero; release the test decoder's zero heads.
    with torch.no_grad():
        for head in decoder.dec_bbox_head:
            head.layers[-1].weight.normal_(std=0.07)
    model = DFINEDecoderStripRefiner(
        ToyBackbone(), ToyEncoder(), decoder, hidden_dim=8,
        query_chunk_size=2, **kwargs,
    )
    if loaded:
        model.load_state_dict(model.state_dict())
    return model


class DecoderStripRefinementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.images = torch.randn(1, 3, 16, 24)

    def test_zero_projection_reproduces_baseline_and_uses_last_two_layers(self):
        model = make_decoder_strip().eval()
        calls = []
        handle = model.refiner.register_forward_hook(lambda module, args, out: calls.append(args))
        try:
            outputs = model(self.images)
        finally:
            handle.remove()
        self.assertEqual(model.refine_layers, (1, 2))
        self.assertEqual(len(calls), 2)
        self.assertFalse(model.requires_gt)
        self.assertFalse(model.oracle_evaluation)
        torch.testing.assert_close(outputs['pred_boxes'], outputs['baseline_boxes'], atol=0, rtol=0)
        with torch.no_grad():
            baseline = model.decoder(model.encoder(model.backbone(self.images)))
        torch.testing.assert_close(outputs['pred_logits'], baseline['pred_logits'], atol=0, rtol=0)
        self.assertEqual(outputs['refinement_mask'].shape, (1, 3))

    def test_nonzero_feedback_is_gt_independent_and_keeps_original_scores(self):
        model = make_decoder_strip().train()
        with torch.no_grad():
            model.refiner.query_projection.bias.copy_(torch.linspace(-0.2, 0.2, 8))
        expected = model(self.images)
        self.assertTrue(expected['refinement_mask'].any())
        self.assertFalse(torch.allclose(expected['pred_boxes'], expected['baseline_boxes']))
        for targets in (normalized_targets(), [{'unread_target': object()}]):
            actual = model(self.images, targets)
            for name in ('pred_boxes', 'pred_logits', 'baseline_boxes', 'refinement_mask'):
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        with torch.no_grad():
            baseline = model.decoder(model.encoder(model.backbone(self.images)))
        torch.testing.assert_close(expected['pred_logits'], baseline['pred_logits'], atol=0, rtol=0)
        self.assertFalse(model.decoder.training)

    def test_box_loss_crosses_frozen_decoder_and_empty_gt_is_ddp_safe(self):
        model = make_decoder_strip().train()
        criterion = PredRefinementCriterion(HungarianMatcher(
            {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}, alpha=0.25, gamma=2.0,
        ))
        for targets in (normalized_targets(), [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}]):
            model.zero_grad(set_to_none=True)
            losses = criterion(model(self.images), targets)
            sum(losses.values()).backward()
            for name, parameter in model.named_parameters():
                if name.startswith('refiner.'):
                    self.assertTrue(parameter.requires_grad, name)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                else:
                    self.assertFalse(parameter.requires_grad, name)
                    self.assertIsNone(parameter.grad, name)
            if len(targets[0]['boxes']):
                self.assertGreater(model.refiner.query_projection.weight.grad.abs().sum().item(), 0)

    def test_baseline_only_checkpoint_preserves_new_adapter(self):
        model = make_decoder_strip(loaded=False)
        with self.assertRaisesRegex(RuntimeError, 'baseline'):
            model(self.images)
        original = {k: v.clone() for k, v in model.refiner.state_dict().items()}
        detector = {k: v for k, v in model.state_dict().items() if not k.startswith('refiner.')}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'b0.pth'
            torch.save({'ema': {'module': detector}}, path)
            model.load_baseline_checkpoint(path)
        for name, tensor in model.refiner.state_dict().items():
            torch.testing.assert_close(tensor, original[name], atol=0, rtol=0)
        model(self.images)

    def test_layer_selection_rejects_first_duplicate_or_unexecuted_layers(self):
        for layers in ([], [0], [1, 1], [3], [-1]):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                make_decoder_strip(refine_layers=layers)
        model = make_decoder_strip(refine_layers=[2])
        self.assertEqual(model.refine_layers, (2,))

    def test_config_keeps_budget_and_optimizes_only_query_adapter(self):
        root = Path(__file__).resolve().parents[1]
        config = YAMLConfig(str(root / 'configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_decoder_strip.yml'))
        p2 = YAMLConfig(str(root / 'configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_pred_p2.yml'))
        for key in ('epochs', 'optimizer', 'lr_scheduler', 'lr_warmup_scheduler', 'train_dataloader', 'val_dataloader'):
            self.assertEqual(config.yaml_cfg[key], p2.yaml_cfg[key], key)
        model = config.model
        self.assertEqual(model.refine_layers, (4, 5))
        trainable = {id(p) for p in model.refiner.parameters()}
        optimized = {id(p) for g in config.optimizer.param_groups for p in g['params']}
        self.assertEqual(trainable, optimized)
        self.assertTrue(config.output_dir.endswith('/decoder_strip_last2'))


if __name__ == '__main__':
    unittest.main()
