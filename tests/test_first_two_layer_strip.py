"""Keep six native decoder layers while correcting only the first two."""

import unittest

import torch

from src.zoo.dfine.dfine_criterion import DFINECriterion
from test_all_layer_strip import make_model, matcher
import test_all_layer_strip_feedback as feedback_tests
from test_gt_refinement import normalized_targets


class FirstTwoLayerStripTests(unittest.TestCase):
    assert_nested_equal = feedback_tests.AllLayerStripFeedbackTests.assert_nested_equal

    def setUp(self):
        torch.manual_seed(41)
        self.images = torch.randn(1, 3, 64, 96)

    @staticmethod
    def model(**kwargs):
        return make_model(
            num_refine_layers=2, normal_points=7, normal_kernel_size=3,
            offset_pixels=8.0, use_log_prior=False, query_fusion='channel_gate',
            bandwidth_mode='box_relative', **kwargs,
        )

    def test_only_two_heads_run_and_zero_init_preserves_native_train_eval(self):
        model = self.model(checkpoint_strips=False)
        self.assertEqual(len(model.refiner.heads), 2)
        self.assertEqual(len(model.decoder.decoder.layers), 6)
        self.assertFalse(any(key.startswith('refiner.heads.2.') for key in model.state_dict()))
        for training in (True, False):
            with self.subTest(training=training):
                model.train(training)
                targets = normalized_targets() if training else None
                calls = []
                hook = model.refiner.register_forward_pre_hook(
                    lambda module, args: calls.append(args[0])
                )
                try:
                    torch.manual_seed(17)
                    actual = model(self.images, targets)
                finally:
                    hook.remove()
                torch.manual_seed(17)
                expected = model.decoder(model.encoder(model.backbone(self.images)), targets)
                self.assert_nested_equal(actual, expected)
                self.assertEqual(calls, [0, 1])
                if training:
                    self.assertEqual(len(actual['aux_outputs']), 5)
                    self.assertEqual(len(actual['dn_outputs']), 6)

    def test_later_distributions_keep_native_updates_and_training_reaches_all_layers(self):
        model = self.model(checkpoint_strips=True).train()
        with torch.no_grad():
            for head in model.refiner.heads:
                head.score_head[-1].weight.normal_(std=0.02)
        increments, corners, query_counts = {}, {}, []
        hooks = []
        for i, head in enumerate(model.decoder.dec_bbox_head):
            hooks.append(head.register_forward_hook(
                lambda module, args, out, index=i: increments.update({index: out})
            ))
        for i, lqe in enumerate(model.decoder.decoder.lqe_layers):
            hooks.append(lqe.register_forward_pre_hook(
                lambda module, args, index=i: corners.update({index: args[1]})
            ))
        for head in model.refiner.heads:
            hooks.append(head.register_forward_pre_hook(
                lambda module, args: query_counts.append(args[1].shape[1])
            ))
        try:
            output = model(self.images, normalized_targets())
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(query_counts, [3, 3])  # DN queries never enter strips.
        self.assertEqual(set(corners), set(range(6)))
        for i in range(2, 6):
            torch.testing.assert_close(corners[i], increments[i] + corners[i - 1], atol=0, rtol=0)
        dn_count = output['dn_meta']['dn_num_split'][0]
        for i in range(2):
            prior = increments[i] if i == 0 else increments[i] + corners[i - 1]
            torch.testing.assert_close(corners[i][:, :dn_count], prior[:, :dn_count], atol=0, rtol=0)
            self.assertFalse(torch.equal(corners[i][:, dn_count:], prior[:, dn_count:]))
        criterion = DFINECriterion(
            matcher(), {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,
                        'loss_fgl': 0.15, 'loss_ddf': 1.5},
            ['vfl', 'boxes', 'local'], num_classes=2, reg_max=4,
        )
        losses = criterion(output, normalized_targets())
        self.assertIn('loss_fgl_aux_4', losses)
        self.assertIn('loss_fgl_dn_5', losses)
        sum(losses.values()).backward()
        for i, head in enumerate(model.refiner.heads):
            for parameter in (head.score_head[-1].weight, head.query_projection.weight,
                              head.strip_encoder[0].weight):
                self.assertIsNotNone(parameter.grad, i)
                self.assertTrue(torch.isfinite(parameter.grad).all(), i)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, i)
        for i, head in enumerate(model.decoder.dec_bbox_head):
            grad = head.layers[-1].weight.grad
            self.assertIsNotNone(grad, i)
            self.assertTrue(torch.isfinite(grad).all(), i)
            self.assertGreater(grad.abs().sum().item(), 0, i)

    def test_default_retains_six_heads_and_legacy_state_keys(self):
        default = make_model()
        explicit = make_model(num_refine_layers=6)
        self.assertEqual(len(default.refiner.heads), 6)
        explicit.load_state_dict(default.state_dict(), strict=True)

    def test_invalid_refinement_layer_count_is_rejected(self):
        for value in (0, -1, 7, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'num_refine_layers'):
                make_model(num_refine_layers=value)


if __name__ == '__main__':
    unittest.main()
