"""Decoder contract and checkpoint boundaries for trainable sparse query reads."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from src.zoo.dfine.dfine_decoder import DFINETransformer
from test_gt_refinement import ToyBackbone, ToyEncoder, normalized_targets


def decoder():
    model = DFINETransformer(
        num_classes=2, hidden_dim=8, num_queries=3, feat_channels=[8],
        feat_strides=[1], num_levels=1, num_points=2, nhead=2, num_layers=6,
        dim_feedforward=16, num_denoising=4, reg_max=4,
    )
    with torch.no_grad():
        for head in model.dec_bbox_head:
            head.layers[-1].weight.normal_(std=.04)
    return model


class SparseDecoderContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_training_requires_explicit_flag_and_exports_dn_count(self):
        model = decoder().train()
        features = [torch.randn(1, 8, 4, 6)]
        calls = []
        def callback(i, q, b, state):
            calls.append((i, state['num_denoising'], q.shape[1]))
            return q
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            model(features, normalized_targets(), query_refiner=callback)
        torch.manual_seed(5)
        expected = model(features, normalized_targets())
        torch.manual_seed(5)
        actual = model(features, normalized_targets(), query_refiner=callback,
                       query_refiner_trainable=True)
        self.assertEqual([c[0] for c in calls], list(range(1, 6)))
        for _, dn, count in calls:
            self.assertEqual(count - dn, 3)
            self.assertGreater(dn, 0)
        for key in ('pred_boxes', 'pred_logits', 'pred_corners'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        self.assertEqual(len(actual['dn_outputs']), 6)
        self.assertEqual(len(actual['aux_outputs']), 5)


class SparseWrapperTests(unittest.TestCase):
    def make_model(self, enabled=True):
        from src.zoo.dfine.sparse_strip_query_refinement import DFINESparseStripQuery
        return DFINESparseStripQuery(ToyBackbone(), ToyEncoder(), decoder(),
                                    hidden_dim=8, enabled=enabled)

    def test_wrapper_validates_before_returning(self):
        model=self.make_model().eval()
        with torch.no_grad():
            model.sparse_refiner.projection.bias.fill_(float('nan'))
            with self.assertRaisesRegex(FloatingPointError,'projected features'):
                model(torch.randn(1,3,16,24))

    def test_eval_compilation_option_keeps_checkpoint_keys_and_ema_independent(self):
        from src.zoo.dfine.sparse_strip_query_refinement import DFINESparseStripQuery
        model = DFINESparseStripQuery(ToyBackbone(), ToyEncoder(), decoder(),
                                     hidden_dim=8, compile_eval=True).eval()
        ema = copy.deepcopy(model)
        self.assertEqual(model.state_dict().keys(), ema.state_dict().keys())
        for a,b in zip(model.parameters(),ema.parameters()):
            self.assertNotEqual(a.data_ptr(),b.data_ptr())
        image = torch.randn(1,3,16,24)
        with torch.no_grad():
            a = model(image)
            ema.sparse_refiner.compile_eval = False
            b = ema(image)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)

    def test_zero_initialization_single_decoder_and_eval_gt_independence(self):
        torch.manual_seed(41)
        model = self.make_model().eval()
        image = torch.randn(1, 3, 16, 24)
        calls = []
        hook = model.decoder.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            changed = model(image, return_distribution=True, return_query=True)
            self.assertEqual(len(calls), 1)
            altered_gt = model(image, targets=[{'unused': object()}],
                               return_distribution=True, return_query=True)
            model.enabled = False
            plain = model(image, return_distribution=True, return_query=True)
        hook.remove()
        for key in ('pred_boxes','pred_logits','pred_corners','pred_queries'):
            torch.testing.assert_close(changed[key], plain[key], atol=0, rtol=0)
            torch.testing.assert_close(changed[key], altered_gt[key], atol=0, rtol=0)

    def test_dn_bypass_frozen_visual_modules_and_native_loss_gradients(self):
        from src.zoo.dfine.dfine_criterion import DFINECriterion
        from src.zoo.dfine.matcher import HungarianMatcher
        torch.manual_seed(7)
        model = self.make_model().train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.decoder.training)
        original = {k:v.clone() for k,v in model.state_dict().items()
                    if k.startswith(('backbone.','encoder.'))}
        calls = []
        hook = model.sparse_refiner.register_forward_pre_hook(
            lambda m, args: calls.append((args[0], args[2].shape[1])))
        criterion = DFINECriterion(
            HungarianMatcher({'cost_class':2,'cost_bbox':5,'cost_giou':2}),
            {'loss_vfl':1,'loss_bbox':5,'loss_giou':2,'loss_fgl':.15,'loss_ddf':0},
            ['vfl','boxes','local'],num_classes=2,reg_max=4)
        output = model(torch.randn(1,3,16,24), normalized_targets())
        hook.remove()
        self.assertEqual(calls, [(i,3) for i in range(1,6)])
        losses = criterion(output, normalized_targets())
        self.assertIn('loss_fgl_dn_0', losses)
        self.assertIn('loss_fgl_aux_0', losses)
        for k,v in losses.items():
            self.assertTrue(torch.isfinite(v), k)
            if k.startswith('loss_ddf'):
                self.assertEqual(v.item(), 0)
        sum(losses.values()).backward()
        grads = [p.grad for p in model.sparse_refiner.parameters()]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
        for k,v in original.items():
            torch.testing.assert_close(model.state_dict()[k],v,atol=0,rtol=0)
        for module in (model.backbone,model.encoder):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))

    def test_strict_common_checkpoint_loading_rejects_missing_or_unrelated_keys(self):
        model = self.make_model()
        common = {k:v.clone() for k,v in model.state_dict().items()
                  if not k.startswith('sparse_refiner.')}
        old = {**common, 'refiner.heads.0.fake':torch.ones(1)}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'checkpoint.pth'
            torch.save({'ema':{'module':old}},path)
            report = model.load_detector_checkpoint(path)
            self.assertEqual(report['source'], 'ema')
            self.assertEqual(report['ignored_keys'], ['refiner.heads.0.fake'])
            bad = copy.copy(old)
            bad.pop(next(iter(common)))
            torch.save({'ema':{'module':bad}},path)
            with self.assertRaisesRegex(ValueError,'missing'):
                model.load_detector_checkpoint(path)
            torch.save({'model':{**old,'unrelated.weight':torch.ones(1)}},path)
            with self.assertRaisesRegex(ValueError,'unexpected'):
                model.load_detector_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
