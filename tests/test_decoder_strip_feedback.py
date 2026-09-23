"""Behavior checks for query feedback inside a frozen real D-FINE decoder."""

import unittest

import torch
import torch.nn as nn

from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function


class DecoderStripFeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.features = [torch.randn(2, 8, 4, 5)]

    def make_model(self, eval_idx=-1):
        model = DFINETransformer(
            num_classes=2, hidden_dim=8, num_queries=3,
            feat_channels=[8], feat_strides=[1], num_levels=1,
            num_points=2, nhead=2, num_layers=3, dim_feedforward=16,
            num_denoising=0, reg_max=4, eval_idx=eval_idx, aux_loss=False,
        ).eval().requires_grad_(False)
        with torch.no_grad():
            for index, head in enumerate(model.dec_bbox_head):
                head.layers[-1].weight.normal_(std=0.2)
                head.layers[-1].bias.copy_(torch.linspace(-0.4, 0.4, 20) * (index + 1))
        return model

    @staticmethod
    def decode_distribution(distribution):
        support = weighting_function(4, distribution["up"], distribution["reg_scale"])
        logits = distribution["pred_corners"]
        expectation = (logits.reshape(*logits.shape[:-1], 4, 5).softmax(-1) * support).sum(-1)
        return distance2bbox(distribution["ref_points"], expectation, distribution["reg_scale"])

    def test_zero_feedback_preserves_outputs_and_state_exactly(self):
        model = self.make_model()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        original = model(self.features, return_query=True, return_distribution=True)
        feedback = model(
            self.features, return_query=True, return_distribution=True,
            query_refiner=lambda index, query, boxes, distribution: query + torch.zeros_like(query),
        )
        self.assertEqual(set(original), set(feedback))
        for key in original:
            torch.testing.assert_close(feedback[key], original[key], rtol=0, atol=0)
        plain = model(self.features)
        explicit_none = model(self.features, query_refiner=None)
        self.assertEqual(set(plain), {"pred_boxes", "pred_logits"})
        for key in plain:
            torch.testing.assert_close(explicit_none[key], plain[key], rtol=0, atol=0)
        self.assertEqual(set(state), set(model.state_dict()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_feedback_reads_prior_cumulative_distribution_after_current_layer(self):
        model = self.make_model()
        layer_outputs, corner_updates, callbacks = {}, [], []
        hooks = [
            layer.register_forward_hook(
                lambda module, args, output, index=index: layer_outputs.update({index: output})
            )
            for index, layer in enumerate(model.decoder.layers)
        ]
        hooks += [
            head.register_forward_hook(lambda module, args, output: corner_updates.append(output))
            for head in model.dec_bbox_head
        ]

        def feedback(index, query, boxes, distribution):
            self.assertEqual(len(corner_updates), index)
            torch.testing.assert_close(query, layer_outputs[index], rtol=0, atol=0)
            torch.testing.assert_close(distribution["pred_corners"], sum(corner_updates), rtol=0, atol=0)
            torch.testing.assert_close(boxes, self.decode_distribution(distribution), atol=1e-6, rtol=1e-5)
            self.assertEqual(tuple(distribution["pred_corners"].shape), (2, 3, 20))
            self.assertEqual(tuple(distribution["ref_points"].shape), (2, 3, 4))
            callbacks.append((index, distribution))
            return query

        try:
            outputs = model(self.features, return_distribution=True, query_refiner=feedback)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual([index for index, _ in callbacks], [1, 2])
        for _, distribution in callbacks:
            torch.testing.assert_close(distribution["ref_points"], outputs["ref_points"], rtol=0, atol=0)
        torch.testing.assert_close(outputs["pred_corners"], sum(corner_updates), rtol=0, atol=0)
        self.assertFalse(torch.equal(callbacks[-1][1]["pred_corners"], outputs["pred_corners"]))

    def test_early_feedback_changes_regression_and_next_layer_query_and_reference(self):
        model = self.make_model()
        original = model(self.features, return_query=True, return_distribution=True)
        layer_outputs, layer_inputs, head_inputs, corner_updates, injected = {}, {}, {}, [], {}
        hooks = []
        for index, layer in enumerate(model.decoder.layers):
            hooks.append(layer.register_forward_pre_hook(
                lambda module, args, index=index: layer_inputs.update({index: args})
            ))
            hooks.append(layer.register_forward_hook(
                lambda module, args, output, index=index: layer_outputs.update({index: output})
            ))
        for index, head in enumerate(model.dec_bbox_head):
            hooks.append(head.register_forward_pre_hook(
                lambda module, args, index=index: head_inputs.update({index: args[0]})
            ))
            hooks.append(head.register_forward_hook(
                lambda module, args, output: corner_updates.append(output)
            ))

        def feedback(index, query, boxes, distribution):
            if index == 1:
                query = query + torch.linspace(-0.5, 0.5, 8)
                injected[index] = query
            return query

        try:
            changed = model(self.features, return_query=True, return_distribution=True, query_refiner=feedback)
        finally:
            for hook in hooks:
                hook.remove()
        torch.testing.assert_close(head_inputs[1], injected[1] + layer_outputs[0].detach(), rtol=0, atol=0)
        torch.testing.assert_close(layer_inputs[2][0], injected[1], rtol=0, atol=0)
        torch.testing.assert_close(head_inputs[2], layer_outputs[2] + injected[1].detach(), rtol=0, atol=0)
        prior = dict(changed, pred_corners=sum(corner_updates[:2]))
        torch.testing.assert_close(layer_inputs[2][1].squeeze(2), self.decode_distribution(prior), atol=1e-6, rtol=1e-5)
        for key in ("pred_queries", "pred_corners", "pred_boxes"):
            self.assertFalse(torch.allclose(changed[key], original[key]), key)

    def test_frozen_regression_heads_preserve_callback_and_prior_logit_gradients(self):
        model = self.make_model()
        delta = nn.Parameter(torch.linspace(-0.2, 0.3, 8))
        retained = {}

        def feedback(index, query, boxes, distribution):
            if index == 1:
                return query + delta
            self.assertTrue(query.requires_grad)
            self.assertTrue(distribution["pred_corners"].requires_grad)
            query.retain_grad()
            distribution["pred_corners"].retain_grad()
            retained.update(query=query, corners=distribution["pred_corners"])
            return query

        outputs = model(self.features, query_refiner=feedback)
        outputs["pred_boxes"].square().sum().backward()
        for tensor in (delta, retained["query"], retained["corners"]):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(torch.count_nonzero(tensor.grad).item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_feedback_respects_eval_idx_and_never_runs_on_first_layer(self):
        for eval_idx in (0, 1, 2):
            with self.subTest(eval_idx=eval_idx):
                model = self.make_model(eval_idx)
                calls = []

                def feedback(index, query, boxes, distribution):
                    calls.append(index)
                    return query

                expected = model(self.features, return_query=True, return_distribution=True)
                actual = model(self.features, return_query=True, return_distribution=True, query_refiner=feedback)
                self.assertEqual(calls, list(range(1, eval_idx + 1)))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_feedback_is_rejected_when_outer_or_inner_decoder_is_training(self):
        model = self.make_model().train()
        feedback = lambda index, query, boxes, distribution: query
        with self.assertRaisesRegex(ValueError, "query_refiner.*eval"):
            model(self.features, query_refiner=feedback)
        model.eval()
        model.decoder.train()
        with self.assertRaisesRegex(ValueError, "query_refiner.*eval"):
            model(self.features, query_refiner=feedback)


if __name__ == "__main__":
    unittest.main()
