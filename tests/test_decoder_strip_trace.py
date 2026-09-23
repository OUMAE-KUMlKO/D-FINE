"""Read-only tracing checks using the real small decoder feedback model."""

import unittest

import torch

from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function
from test_decoder_strip_refinement import make_decoder_strip
from tools.analysis.decoder_strip_trace import trace_decoder_feedback


class DecoderStripTraceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.images = torch.randn(1, 3, 16, 24)
        self.model = make_decoder_strip().eval()
        with torch.inference_mode():
            self.features = self.model.encoder(self.model.backbone(self.images))

    def enable_feedback(self):
        with torch.no_grad():
            self.model.refiner.query_projection.bias.copy_(torch.linspace(-0.2, 0.2, 8))

    def test_default_trace_matches_raw_decoder_and_wrapper_fallback(self):
        self.enable_feedback()
        raw = []
        hook = self.model.decoder.register_forward_hook(
            lambda module, args, output: raw.append(output)
        )
        try:
            with torch.inference_mode():
                expected = self.model(self.images)
        finally:
            hook.remove()
        self.assertEqual(len(raw), 2)
        result = trace_decoder_feedback(self.model, self.features, self.images.shape[-2:])
        for key in ("pred_boxes", "pred_logits"):
            torch.testing.assert_close(result[key], raw[1][key], rtol=0, atol=0)
        boxes, baseline = result["pred_boxes"], raw[0]["pred_boxes"]
        valid = (
            result["refinement_mask"] & torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 0).all(-1)
            & torch.isfinite(baseline).all(-1) & (baseline[..., 2:] > 0).all(-1)
            & torch.isfinite(raw[0]["pred_logits"]).all(-1)
        )
        torch.testing.assert_close(valid, expected["refinement_mask"])
        torch.testing.assert_close(torch.where(valid[..., None], boxes, baseline), expected["pred_boxes"], rtol=0, atol=0)

    def test_zero_initialization_and_empty_switch_reproduce_baseline(self):
        with torch.inference_mode():
            baseline = self.model.decoder(self.features)
        for active in (None, ()):
            with self.subTest(active=active):
                result = trace_decoder_feedback(self.model, self.features, self.images.shape[-2:], active)
                for key in baseline:
                    torch.testing.assert_close(result[key], baseline[key], rtol=0, atol=0)
                self.assertEqual(set(result["layers"]), set(self.model.refine_layers))
                for record in result["layers"].values():
                    self.assertEqual(tuple(record["query_before"].shape), (1, 3, 8))
                    self.assertEqual(tuple(record["logits_before"].shape), (1, 3, 20))
                    self.assertEqual(tuple(record["boxes_before"].shape), (1, 3, 4))
                    self.assertEqual(torch.count_nonzero(record["query_delta"]).item(), 0)
                    for field in ("boxes", "logits"):
                        torch.testing.assert_close(record[f"{field}_before"], record[f"{field}_after"], rtol=0, atol=0)
                    if active == ():
                        self.assertFalse(record["valid"].any())
                if active == ():
                    self.assertFalse(result["refinement_mask"].any())

    def test_local_before_after_matches_current_layer_actual_regression(self):
        self.enable_feedback()
        heads, inputs, callbacks = {}, {}, {}
        hooks = []
        for index, head in enumerate(self.model.decoder.dec_bbox_head):
            hooks.append(head.register_forward_pre_hook(
                lambda module, args, index=index: inputs.update({index: args[0]})
            ))
            hooks.append(head.register_forward_hook(
                lambda module, args, output, index=index: heads.update({index: output})
            ))

        def feedback(index, query, previous_boxes, distribution):
            delta, valid = self.model.refiner(
                self.features[self.model.feature_level], query, previous_boxes,
                self.images.shape[-2:], distribution,
            )
            callbacks[index] = (query, delta, valid, distribution, previous_boxes)
            return query + delta

        try:
            with torch.inference_mode():
                actual = self.model.decoder(self.features, query_refiner=feedback)
        finally:
            for hook in hooks:
                hook.remove()
        traced = trace_decoder_feedback(self.model, self.features, self.images.shape[-2:])
        support = weighting_function(4, self.model.decoder.up, self.model.decoder.reg_scale)
        for index, record in traced["layers"].items():
            query, delta, valid, distribution, previous_boxes = callbacks[index]
            with torch.inference_mode():
                logits_before = self.model.decoder.dec_bbox_head[index](inputs[index] - delta) + distribution["pred_corners"]
                logits_after = sum(heads[i] for i in range(index + 1))
                for label, logits in (("before", logits_before), ("after", logits_after)):
                    boxes = distance2bbox(
                        distribution["ref_points"], self.model.decoder.integral(logits, support),
                        distribution["reg_scale"],
                    )
                    torch.testing.assert_close(record[f"logits_{label}"], logits, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(record[f"boxes_{label}"], boxes, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(record["query_before"], query, rtol=0, atol=0)
            torch.testing.assert_close(record["query_delta"], delta, rtol=0, atol=0)
            torch.testing.assert_close(record["valid"], valid)
            self.assertFalse(torch.allclose(record["boxes_before"], previous_boxes))
        torch.testing.assert_close(traced["layers"][2]["boxes_after"], actual["pred_boxes"], rtol=0, atol=0)

    def test_first_last_and_empty_layer_switches_match_direct_callback(self):
        self.enable_feedback()
        traces = {}
        for active in ((), (1,), (2,), (1, 2)):
            with self.subTest(active=active):
                injected = torch.zeros(1, 3, dtype=torch.bool)

                def feedback(index, query, boxes, distribution):
                    nonlocal injected
                    if index not in active:
                        return query
                    delta, valid = self.model.refiner(
                        self.features[self.model.feature_level], query, boxes,
                        self.images.shape[-2:], distribution,
                    )
                    injected |= valid
                    return query + delta

                with torch.inference_mode():
                    expected = self.model.decoder(self.features, query_refiner=feedback)
                result = trace_decoder_feedback(self.model, self.features, self.images.shape[-2:], active)
                traces[active] = result
                for key in expected:
                    torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)
                torch.testing.assert_close(result["refinement_mask"], injected)
                for index in set(self.model.refine_layers) - set(active):
                    record = result["layers"][index]
                    self.assertFalse(record["valid"].any())
                    self.assertEqual(torch.count_nonzero(record["query_delta"]).item(), 0)
                    torch.testing.assert_close(record["boxes_before"], record["boxes_after"], rtol=0, atol=0)
        self.assertFalse(torch.allclose(traces[(1,)]["pred_boxes"], traces[(2,)]["pred_boxes"]))
        with torch.inference_mode():
            baseline = self.model.decoder(self.features)
        torch.testing.assert_close(traces[()]["pred_boxes"], baseline["pred_boxes"], rtol=0, atol=0)

    def test_trace_is_fp32_inference_and_removes_only_its_hooks_even_on_error(self):
        self.enable_feedback()
        parameters = {name: value.clone() for name, value in self.model.state_dict().items()}
        inference_calls = []
        existing = self.model.decoder.decoder.layers[0].register_forward_hook(
            lambda module, args, output: inference_calls.append(torch.is_inference_mode_enabled())
        )
        initial_hooks = {id(module): tuple(module._forward_hooks) for module in self.model.modules()}
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                traced = trace_decoder_feedback(self.model, self.features, self.images.shape[-2:])
            self.assertEqual(traced["pred_boxes"].dtype, torch.float32)
            self.assertEqual(traced["pred_logits"].dtype, torch.float32)
            for record in traced["layers"].values():
                self.assertTrue(all(not value.requires_grad for value in record.values()))
            self.assertEqual(inference_calls, [True])
            with self.assertRaises((RuntimeError, ValueError)):
                trace_decoder_feedback(self.model, [self.features[0][:, :7]], self.images.shape[-2:])
            for module in self.model.modules():
                self.assertEqual(tuple(module._forward_hooks), initial_hooks[id(module)])
            for name, value in self.model.state_dict().items():
                torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
        finally:
            existing.remove()


if __name__ == "__main__":
    unittest.main()
