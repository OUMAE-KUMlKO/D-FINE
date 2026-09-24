"""Final-distribution calibration integrated with a frozen real decoder."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.core import YAMLConfig
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.matcher import HungarianMatcher
from src.zoo.dfine.probability_refinement import DFINEProbabilityRefiner
from src.zoo.dfine.probability_refinement_criterion import ProbabilityRefinementCriterion
from test_gt_refinement import ToyBackbone, ToyEncoder


torch.set_num_threads(1)


def make_probability_refiner(loaded=True):
    decoder = DFINETransformer(
        num_classes=2, hidden_dim=8, num_queries=3,
        feat_channels=[8], feat_strides=[1], num_levels=1,
        num_points=2, nhead=2, num_layers=3, dim_feedforward=16,
        num_denoising=0, reg_max=4,
    )
    with torch.no_grad():
        for head in decoder.dec_bbox_head:
            head.layers[-1].weight.normal_(std=0.07)
    model = DFINEProbabilityRefiner(
        ToyBackbone(), ToyEncoder(), decoder, hidden_dim=8, query_chunk_size=2,
    )
    if loaded:
        model.load_state_dict(model.state_dict())
    return model


class ProbabilityRefinementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.images = torch.randn(1, 3, 16, 24)

    def test_zero_calibration_preserves_baseline_in_one_decoder_pass(self):
        model = make_probability_refiner().train()
        calls = []
        handle = model.decoder.register_forward_hook(
            lambda module, args, out: calls.append(out)
        )
        try:
            output = model(self.images)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)
        baseline = calls[0]
        torch.testing.assert_close(output['pred_boxes'], baseline['pred_boxes'], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(output['pred_logits'], baseline['pred_logits'], atol=0, rtol=0)
        torch.testing.assert_close(
            output['calibrated_corners'], baseline['pred_corners'].reshape(1, 3, 4, 5),
            atol=0, rtol=0,
        )
        self.assertTrue(output['calibrated_corners'].requires_grad)
        self.assertFalse(output['candidate_positions'].requires_grad)
        self.assertTrue(output['input_valid'].any())
        self.assertFalse(model.decoder.training)
        self.assertFalse(model.requires_gt)
        self.assertFalse(model.oracle_evaluation)

    def test_nonzero_calibration_ignores_gt_and_keeps_classification(self):
        model = make_probability_refiner().train()
        with torch.no_grad():
            model.refiner.score_head[-1].weight.normal_(std=0.2)
        expected = model(self.images)
        self.assertGreater((expected['pred_boxes'] - expected['baseline_boxes']).abs().max().item(), 1e-7)
        actual = model(self.images, [{'unused_target': object()}])
        for name in ('pred_boxes', 'pred_logits', 'baseline_boxes', 'calibrated_corners', 'refinement_mask'):
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        with torch.no_grad():
            baseline = model.decoder(model.encoder(model.backbone(self.images)))
        torch.testing.assert_close(expected['pred_logits'], baseline['pred_logits'], atol=0, rtol=0)

    def test_candidate_loss_reaches_only_new_head_and_empty_gt_is_connected(self):
        model = make_probability_refiner().train()
        criterion = ProbabilityRefinementCriterion(HungarianMatcher(
            {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2}, alpha=0.25, gamma=2.0,
        ))
        initial = model(self.images)
        q = torch.where(initial['input_valid'][0])[0][0]
        target = [{
            'boxes': initial['baseline_boxes'][0, q:q + 1].clone(),
            'labels': initial['pred_logits'][0, q:q + 1].argmax(-1),
        }]
        empty = [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}]
        for targets in (target, empty):
            model.zero_grad(set_to_none=True)
            losses = criterion(model(self.images), targets)
            self.assertEqual(set(losses), {'loss_bbox', 'loss_giou', 'loss_candidate'})
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
                self.assertGreater(losses['loss_candidate'].item(), 0)
                self.assertGreater(model.refiner.score_head[-1].weight.grad.abs().sum().item(), 0)
            else:
                self.assertEqual(sum(losses.values()).item(), 0)

    def test_loading_baseline_leaves_new_head_fresh_and_requires_complete_detector(self):
        model = make_probability_refiner(loaded=False)
        with self.assertRaisesRegex(RuntimeError, 'baseline'):
            model(self.images)
        fresh = {key: value.clone() for key, value in model.refiner.state_dict().items()}
        detector = {key: value for key, value in model.state_dict().items() if not key.startswith('refiner.')}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'baseline.pth'
            torch.save({'model': detector}, path)
            model.load_baseline_checkpoint(path)
        for name, value in model.refiner.state_dict().items():
            torch.testing.assert_close(value, fresh[name], atol=0, rtol=0)
        model(self.images)

    def test_registered_config_preserves_budget_and_optimizes_only_refiner(self):
        root = Path(__file__).resolve().parents[1]
        config = YAMLConfig(str(root / 'configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_pred_p3.yml'))
        p2 = YAMLConfig(str(root / 'configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_pred_p2.yml'))
        for key in ('epochs', 'optimizer', 'lr_scheduler', 'lr_warmup_scheduler', 'train_dataloader', 'val_dataloader'):
            self.assertEqual(config.yaml_cfg[key], p2.yaml_cfg[key], key)
        model = config.model
        self.assertIsInstance(model, DFINEProbabilityRefiner)
        self.assertIsInstance(config.criterion, ProbabilityRefinementCriterion)
        self.assertEqual(config.criterion.weight_dict, {
            'loss_bbox': 5, 'loss_giou': 2, 'loss_candidate': 0.05,
        })
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        optimized = {id(p) for group in config.optimizer.param_groups for p in group['params']}
        self.assertEqual(trainable, {id(p) for p in model.refiner.parameters()})
        self.assertEqual(trainable, optimized)
        self.assertTrue(config.output_dir.endswith('/p3_probability_calibration'))


if __name__ == '__main__':
    unittest.main()
