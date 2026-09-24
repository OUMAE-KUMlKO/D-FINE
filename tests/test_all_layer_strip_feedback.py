"""Real-decoder checks for probability feedback at every FDR layer."""

import unittest

import torch
import torch.nn as nn

from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function


class AllLayerStripFeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.features = [torch.randn(2, 8, 4, 5)]
        self.targets = [
            {"boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]), "labels": torch.tensor([1])}
            for _ in range(2)
        ]

    def make_model(self, eval_idx=-1, num_denoising=2):
        model = DFINETransformer(
            num_classes=2, hidden_dim=8, num_queries=3,
            feat_channels=[8], feat_strides=[1], num_levels=1,
            num_points=2, nhead=2, num_layers=6, dim_feedforward=16,
            num_denoising=num_denoising, reg_max=4, eval_idx=eval_idx,
        )
        with torch.no_grad():
            for head in model.dec_bbox_head:
                head.layers[-1].weight.normal_(std=0.1)
        return model

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for first, second in zip(actual, expected):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(actual, expected)

    @staticmethod
    def decode(model, corners, distribution):
        support = weighting_function(model.reg_max, distribution["up"], distribution["reg_scale"])
        return distance2bbox(
            distribution["ref_points"], model.integral(corners, support), distribution["reg_scale"]
        )

    def test_identity_callback_keeps_train_eval_outputs_and_state_unchanged(self):
        model = self.make_model()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        for training in (True, False):
            with self.subTest(training=training):
                model.train(training)
                kwargs = {} if training else {"return_query": True, "return_distribution": True}
                torch.manual_seed(11)
                expected = model(self.features, self.targets, **kwargs)
                torch.manual_seed(11)
                actual = model(
                    self.features, self.targets, **kwargs,
                    distribution_refiner=lambda index, query, boxes, distribution: distribution["pred_corners"],
                )
                self.assert_nested_equal(actual, expected)
        self.assertEqual(set(state), set(model.state_dict()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], atol=0, rtol=0)

    def test_all_six_training_layers_include_dn_and_export_corrected_aux_logits(self):
        model = self.make_model().train()
        torch.manual_seed(11)
        native = model(self.features, self.targets)
        callbacks, corrected, lqe_inputs = {}, {}, {}
        hooks = [
            layer.register_forward_pre_hook(
                lambda module, args, index=index: lqe_inputs.update({index: args[1]})
            )
            for index, layer in enumerate(model.decoder.lqe_layers)
        ]

        def feedback(index, query, boxes, distribution):
            self.assertEqual(query.shape[:2], distribution["pred_corners"].shape[:2])
            torch.testing.assert_close(boxes, self.decode(model, distribution["pred_corners"], distribution))
            callbacks[index] = distribution
            corrected[index] = distribution["pred_corners"] + torch.linspace(-0.1, 0.2, 20) * (index + 1)
            return corrected[index]

        try:
            torch.manual_seed(11)
            outputs = model(self.features, self.targets, distribution_refiner=feedback)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(set(callbacks), set(range(6)))
        dn_count = outputs["dn_meta"]["dn_num_split"][0]
        self.assertGreater(dn_count, 0)
        regular_layers = outputs["aux_outputs"] + [outputs]
        self.assertEqual(len(regular_layers), 6)
        self.assertEqual(len(outputs["dn_outputs"]), 6)
        for index in range(6):
            self.assertEqual(callbacks[index]["num_denoising"], dn_count)
            self.assertEqual(corrected[index].shape, (2, dn_count + 3, 20))
            torch.testing.assert_close(callbacks[index]["ref_points"], callbacks[0]["ref_points"], atol=0, rtol=0)
            torch.testing.assert_close(lqe_inputs[index], corrected[index], atol=0, rtol=0)
            torch.testing.assert_close(regular_layers[index]["pred_corners"], corrected[index][:, dn_count:], atol=0, rtol=0)
            torch.testing.assert_close(outputs["dn_outputs"][index]["pred_corners"], corrected[index][:, :dn_count], atol=0, rtol=0)
            decoded = self.decode(model, corrected[index], callbacks[index])
            torch.testing.assert_close(regular_layers[index]["pred_boxes"], decoded[:, dn_count:], atol=0, rtol=0)
        self.assert_nested_equal(outputs["pre_outputs"], native["pre_outputs"])
        self.assert_nested_equal(outputs["dn_pre_outputs"], native["dn_pre_outputs"])

    def test_first_layer_change_feeds_next_cumulative_logits_and_attention_reference(self):
        model = self.make_model(num_denoising=0).eval()
        native = model(self.features, return_distribution=True)
        layer_inputs, head_outputs, before, after = {}, {}, {}, {}
        hooks = []
        for index, layer in enumerate(model.decoder.layers):
            hooks.append(layer.register_forward_pre_hook(
                lambda module, args, index=index: layer_inputs.update({index: args})
            ))
        for index, head in enumerate(model.dec_bbox_head):
            hooks.append(head.register_forward_hook(
                lambda module, args, output, index=index: head_outputs.update({index: output})
            ))

        def feedback(index, query, boxes, distribution):
            before[index] = distribution
            self.assertEqual(distribution["num_denoising"], 0)
            torch.testing.assert_close(boxes, self.decode(model, distribution["pred_corners"], distribution))
            after[index] = distribution["pred_corners"]
            if index == 0:
                after[index] = after[index] + torch.linspace(-0.6, 0.6, 20)
            return after[index]

        try:
            outputs = model(self.features, return_distribution=True, distribution_refiner=feedback)
        finally:
            for hook in hooks:
                hook.remove()
        torch.testing.assert_close(before[0]["pred_corners"], head_outputs[0], atol=0, rtol=0)
        for index in range(1, 6):
            torch.testing.assert_close(before[index]["pred_corners"], head_outputs[index] + after[index - 1], atol=0, rtol=0)
            torch.testing.assert_close(
                layer_inputs[index][1].squeeze(2), self.decode(model, after[index - 1], before[index - 1]),
                atol=0, rtol=0,
            )
        self.assertFalse(torch.allclose(outputs["pred_corners"], native["pred_corners"]))
        self.assertFalse(torch.allclose(outputs["pred_boxes"], native["pred_boxes"]))

    def test_final_box_loss_backpropagates_through_first_layer_calibration(self):
        model = self.make_model(num_denoising=0).train().requires_grad_(False)
        delta = nn.Parameter(torch.linspace(-0.1, 0.2, 20))
        early = []

        def feedback(index, query, boxes, distribution):
            corners = distribution["pred_corners"]
            if index == 0:
                corners = corners + delta
                corners.retain_grad()
                early.append(corners)
            else:
                self.assertTrue(corners.requires_grad)
            return corners

        outputs = model(self.features, self.targets, distribution_refiner=feedback)
        outputs["pred_boxes"].square().sum().backward()
        for value in (delta, early[0]):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(torch.count_nonzero(value.grad).item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_eval_idx_stops_callbacks_after_requested_output_layer(self):
        for eval_idx in (0, 2, 5):
            with self.subTest(eval_idx=eval_idx):
                model = self.make_model(eval_idx=eval_idx).eval()
                calls = []

                def feedback(index, query, boxes, distribution):
                    calls.append(index)
                    return distribution["pred_corners"]

                expected = model(self.features)
                actual = model(self.features, distribution_refiner=feedback)
                self.assertEqual(calls, list(range(eval_idx + 1)))
                self.assert_nested_equal(actual, expected)

    def test_query_and_distribution_feedback_cannot_be_combined(self):
        model = self.make_model().eval()
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            model(
                self.features,
                query_refiner=lambda index, query, boxes, distribution: query,
                distribution_refiner=lambda index, query, boxes, distribution: distribution["pred_corners"],
            )


if __name__ == "__main__":
    unittest.main()
