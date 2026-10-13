"""Numerical safety at the decoder distribution-refiner boundary."""

import unittest

import torch

from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy
from src.zoo.dfine.prediction_safety import (
    PredictionSafetyError,
    select_safe_correction,
    validate_prior_boxes,
)
from test_all_layer_strip import make_model
from test_gt_refinement import normalized_targets


torch.set_num_threads(1)


class PredictionSafetyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)
        self.features = [torch.randn(1, 8, 4, 5)]

    @staticmethod
    def make_decoder(num_denoising=0):
        model = DFINETransformer(
            num_classes=2, hidden_dim=8, num_queries=3,
            feat_channels=[8], feat_strides=[1], num_levels=1,
            num_points=2, nhead=2, num_layers=6, dim_feedforward=16,
            num_denoising=num_denoising, reg_max=4,
        )
        with torch.no_grad():
            for head in model.dec_bbox_head:
                head.layers[-1].weight.normal_(std=0.06)
        return model

    @staticmethod
    def targets():
        return [{
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "labels": torch.tensor([1]),
        }]

    def test_fp16_conversion_overflow_falls_back_before_decode_with_finite_gradients(self):
        prior = torch.zeros(1, 2, 20, dtype=torch.float16, requires_grad=True)
        prior_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2],
                                     [0.4, 0.6, 0.3, 0.1]]], dtype=torch.float16)
        corrected = torch.zeros(1, 2, 20, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            corrected[0, 0, 0] = 1e5
        decode_inputs = []

        def decode(logits):
            decode_inputs.append(logits.detach().clone())
            zero = logits[..., 0] * 0
            return torch.stack((zero + 0.5, zero + 0.5,
                                zero + 0.2, zero + 0.2), -1)

        selected, boxes, stats = select_safe_correction(
            0, prior, prior_boxes, corrected,
            torch.full((1, 2, 4), 0.5, dtype=torch.float16), decode,
        )
        self.assertEqual(selected.dtype, torch.float16)
        self.assertTrue(all(torch.isfinite(value).all() for value in decode_inputs))
        torch.testing.assert_close(selected[0, 0], prior[0, 0], atol=0, rtol=0)
        torch.testing.assert_close(selected[0, 1], corrected[0, 1].half(), atol=0, rtol=0)
        self.assertEqual(stats, {"checked_queries": 2, "fallback_queries": 1})

        (selected.float().sum() + boxes.float().sum()).backward()
        for value in (prior.grad, corrected.grad):
            self.assertIsNotNone(value)
            self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(torch.count_nonzero(corrected.grad[0, 0]).item(), 0)

    def run_negative_box(self, source):
        torch.manual_seed(101)
        model = self.make_decoder().train()
        records = {"prior": None, "boxes": None, "lqe": None, "next_ref": None}
        hooks = [
            model.decoder.lqe_layers[0].register_forward_pre_hook(
                lambda module, args: records.update(lqe=args[1].detach().clone()),
            ),
            model.decoder.layers[1].register_forward_pre_hook(
                lambda module, args: records.update(next_ref=args[1].detach().clone()),
            ),
        ]

        def refine(index, query, boxes, distribution):
            prior = distribution["pred_corners"]
            if index:
                return prior
            records["prior"] = prior.detach().clone()
            records["boxes"] = boxes.detach().clone()
            corrected = prior.clone().reshape(1, 3, 4, 5)
            corrected[0, 0].fill_(-20)
            corrected[0, 0, :, 0] = 20
            return corrected.flatten(-2)

        try:
            torch.manual_seed(13)
            output = model(
                self.features, distribution_refiner=refine,
                next_reference_source=source,
            )
        finally:
            for hook in hooks:
                hook.remove()
        records["output"] = output
        records["stats"] = dict(model.decoder.last_prediction_safety_stats)
        return records

    def test_negative_width_correction_falls_back_whole_query_for_outputs_and_lqe(self):
        records = self.run_negative_box("corrected")
        first = records["output"]["aux_outputs"][0]
        torch.testing.assert_close(first["pred_corners"][0, 0], records["prior"][0, 0])
        torch.testing.assert_close(first["pred_boxes"][0, 0], records["boxes"][0, 0])
        torch.testing.assert_close(records["lqe"][0, 0], records["prior"][0, 0])
        self.assertEqual(records["stats"]["layer_0_checked_queries"], 3)
        self.assertEqual(records["stats"]["layer_0_fallback_queries"], 1)

    def test_fallback_reference_is_consistent_for_both_reference_sources(self):
        corrected = self.run_negative_box("corrected")
        pre = self.run_negative_box("pre_correction")
        for records in (corrected, pre):
            torch.testing.assert_close(
                records["next_ref"].squeeze(2)[0, 0], records["boxes"][0, 0],
                atol=0, rtol=0,
            )
        torch.testing.assert_close(
            corrected["next_ref"][0, 0], pre["next_ref"][0, 0], atol=0, rtol=0,
        )

    @staticmethod
    def force_prior_geometry(model, peak_bin):
        def force(module, args, output):
            result = output.clone().reshape(1, 3, 4, 5)
            result[0, 0].fill_(-100)
            result[0, 0, :, peak_bin] = 100
            return result.flatten(-2)

        return model.dec_bbox_head[0].register_forward_hook(force)

    def test_finite_nonpositive_priors_preserve_native_outputs_and_gradients(self):
        for peak_bin in (0, 1):
            for source in ("corrected", "pre_correction"):
                with self.subTest(peak_bin=peak_bin, source=source):
                    torch.manual_seed(101)
                    model = self.make_decoder().train()
                    hook = self.force_prior_geometry(model, peak_bin)
                    try:
                        native = model(self.features)
                        native["pred_boxes"].square().sum().backward()
                        native_grads = {
                            name: value.grad.clone() for name, value in model.named_parameters()
                            if value.grad is not None
                        }
                        model.zero_grad(set_to_none=True)
                        guarded = model(
                            self.features, next_reference_source=source,
                            distribution_refiner=lambda index, query, boxes, distribution:
                            distribution["pred_corners"],
                        )
                        guarded["pred_boxes"].square().sum().backward()
                    finally:
                        hook.remove()
                    first_box = native["aux_outputs"][0]["pred_boxes"][0, 0]
                    if peak_bin == 0:
                        self.assertTrue((first_box[2:] < 0).all())
                    else:
                        self.assertTrue((first_box[2:] == 0).all())
                    xyxy = box_cxcywh_to_xyxy(first_box)
                    torch.testing.assert_close(xyxy[:2], xyxy[2:], atol=0, rtol=0)
                    native_layers = [*native["aux_outputs"], native]
                    guarded_layers = [*guarded["aux_outputs"], guarded]
                    for expected, actual in zip(native_layers, guarded_layers):
                        for key in ("pred_boxes", "pred_corners", "pred_logits"):
                            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                    for name, value in model.named_parameters():
                        if name in native_grads:
                            self.assertTrue(torch.isfinite(value.grad).all())
                            torch.testing.assert_close(value.grad, native_grads[name])
                    self.assertEqual(
                        model.decoder.last_prediction_safety_stats[
                            "layer_0_prior_invalid_geometry_queries"
                        ], 1,
                    )

    def test_refiner_can_repair_finite_negative_prior_with_each_reference_source(self):
        for source in ("corrected", "pre_correction"):
            with self.subTest(source=source):
                model = self.make_decoder().train()
                records = {}

                def repair(index, query, boxes, distribution):
                    prior = distribution["pred_corners"]
                    if index:
                        return prior
                    records["prior_boxes"] = boxes.detach().clone()
                    corrected = prior.clone().reshape(1, 3, 4, 5)
                    corrected[0, 0].fill_(-20)
                    corrected[0, 0, :, 2] = 20
                    records["corrected"] = corrected.flatten(-2).detach().clone()
                    return corrected.flatten(-2)

                hooks = [
                    self.force_prior_geometry(model, 0),
                    model.decoder.layers[1].register_forward_pre_hook(
                        lambda module, args: records.update(next_ref=args[1].detach().clone()),
                    ),
                ]
                try:
                    output = model(
                        self.features, distribution_refiner=repair,
                        next_reference_source=source,
                    )
                finally:
                    for hook in hooks:
                        hook.remove()
                first = output["aux_outputs"][0]
                self.assertTrue((records["prior_boxes"][0, 0, 2:] < 0).all())
                self.assertTrue((first["pred_boxes"][0, 0, 2:] > 0).all())
                torch.testing.assert_close(first["pred_corners"], records["corrected"])
                expected_ref = first["pred_boxes"] if source == "corrected" else records["prior_boxes"]
                torch.testing.assert_close(records["next_ref"].squeeze(2), expected_ref)
                stats = model.decoder.last_prediction_safety_stats
                self.assertEqual(stats["layer_0_prior_invalid_geometry_queries"], 1)
                self.assertEqual(stats["layer_0_fallback_queries"], 0)

    def test_prior_geometry_count_excludes_dn_and_nonfinite_boxes_still_raise(self):
        prior = torch.zeros(1, 3, 20)
        refs = torch.full((1, 3, 4), 0.5)
        boxes = torch.tensor([[[0.5, 0.5, float("nan"), -1.0],
                               [0.5, 0.5, 0.0, 0.1],
                               [0.5, 0.5, 0.2, -0.01]]])
        self.assertEqual(validate_prior_boxes(2, prior, boxes, refs, dn_count=1), 2)
        boxes[0, 2, 0] = float("inf")
        with self.assertRaises(PredictionSafetyError) as caught:
            validate_prior_boxes(2, prior, boxes, refs, dn_count=1)
        self.assertEqual(caught.exception.diagnostic["stage"], "prior_boxes")
        torch.testing.assert_close(
            caught.exception.diagnostic["query_indices"], torch.tensor([[0, 2]]),
        )

    def test_nonfinite_prior_raises_before_refiner_and_keeps_cpu_diagnostic(self):
        model = self.make_decoder().eval()
        called = []

        def corrupt(module, args, output):
            output = output.clone()
            output[0, 1, 2] = float("nan")
            return output

        hook = model.dec_bbox_head[0].register_forward_hook(corrupt)
        try:
            with self.assertRaises(PredictionSafetyError) as caught:
                model(
                    self.features,
                    distribution_refiner=lambda *args: called.append(True) or args[3]["pred_corners"],
                )
        finally:
            hook.remove()
        self.assertFalse(called)
        diagnostic = caught.exception.diagnostic
        self.assertEqual(diagnostic["layer_index"], 0)
        self.assertEqual(diagnostic["stage"], "prior_logits")
        torch.testing.assert_close(diagnostic["query_indices"], torch.tensor([[0, 1]]))
        self.assertEqual(diagnostic["prior_logits"].device.type, "cpu")
        self.assertTrue(torch.isnan(diagnostic["prior_logits"][0, 1, 2]))
        self.assertIn("ref_points", diagnostic)
        self.assertEqual(model.decoder.last_prediction_safety_stats["layer_0_checked_queries"], 3)

    def test_invalid_dn_padding_is_unchanged_and_excluded_from_guard_stats(self):
        model = self.make_decoder(num_denoising=2).train()
        dn_counts = []

        def corrupt_dn(module, args, output):
            output = output.clone()
            output[0, 0, 2] = float("nan")
            return output

        def identity(index, query, boxes, distribution):
            dn_counts.append(distribution["num_denoising"])
            return distribution["pred_corners"]

        hook = model.dec_bbox_head[-1].register_forward_hook(corrupt_dn)
        try:
            torch.manual_seed(29)
            native = model(self.features, self.targets())
            torch.manual_seed(29)
            guarded = model(
                self.features, self.targets(), distribution_refiner=identity,
            )
        finally:
            hook.remove()
        self.assertGreater(dn_counts[0], 0)
        self.assertTrue(torch.isnan(guarded["dn_outputs"][-1]["pred_corners"][0, 0, 2]))
        for key in ("pred_corners", "pred_boxes"):
            torch.testing.assert_close(
                guarded["dn_outputs"][-1][key], native["dn_outputs"][-1][key],
                atol=0, rtol=0, equal_nan=True,
            )
        self.assertEqual(
            model.decoder.last_prediction_safety_stats["layer_5_checked_queries"], 3,
        )
        self.assertEqual(
            model.decoder.last_prediction_safety_stats["layer_5_fallback_queries"], 0,
        )

    def test_invalid_matching_prior_still_raises_with_full_query_index_and_dn_count(self):
        model = self.make_decoder(num_denoising=2).train()
        called = []
        observed_dn = []

        def corrupt_matching(module, args, output):
            output = output.clone()
            # The denoising group generated by this fixture precedes three
            # matching queries. Determine its size from the total query count.
            dn_count = output.shape[1] - 3
            observed_dn.append(dn_count)
            output[0, dn_count + 1, 2] = float("nan")
            return output

        hook = model.dec_bbox_head[0].register_forward_hook(corrupt_matching)
        try:
            with self.assertRaises(PredictionSafetyError) as caught:
                torch.manual_seed(29)
                model(
                    self.features, self.targets(),
                    distribution_refiner=lambda *args: called.append(True) or args[3]["pred_corners"],
                )
        finally:
            hook.remove()
        self.assertFalse(called)
        diagnostic = caught.exception.diagnostic
        self.assertGreater(observed_dn[0], 0)
        self.assertEqual(diagnostic["dn_count"], observed_dn[0])
        torch.testing.assert_close(
            diagnostic["query_indices"],
            torch.tensor([[0, observed_dn[0] + 1]]),
        )
        self.assertEqual(
            model.decoder.last_prediction_safety_stats["layer_0_checked_queries"], 3,
        )

    def test_valid_refiner_and_no_refiner_paths_preserve_outputs_and_state(self):
        model = self.make_decoder().eval()
        state_keys = set(model.state_dict())
        native = model(self.features, return_distribution=True)
        self.assertEqual(model.decoder.last_prediction_safety_stats, {})
        self.assertNotIn("prediction_safety", native)
        refined = model(
            self.features, return_distribution=True,
            distribution_refiner=lambda index, query, boxes, distribution:
            distribution["pred_corners"],
        )
        for key, expected in native.items():
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(refined[key], expected, atol=0, rtol=0)
        self.assertEqual(set(model.state_dict()), state_keys)
        for index in range(6):
            self.assertEqual(
                model.decoder.last_prediction_safety_stats[f"layer_{index}_checked_queries"], 3,
            )
            self.assertEqual(
                model.decoder.last_prediction_safety_stats[f"layer_{index}_fallback_queries"], 0,
            )

    def test_head_rejection_stats_are_assigned_once_per_layer_under_checkpointing(self):
        torch.manual_seed(41)
        model = make_model(checkpoint_strips=True).train()
        images = torch.randn(1, 3, 16, 24)

        def force_rejections(module, args, result):
            result = dict(result)
            result["input_valid"] = torch.ones_like(result["input_valid"])
            result["input_valid"][0, 0] = False
            result["refinement_mask"] = torch.ones_like(result["refinement_mask"])
            result["refinement_mask"][0, 0] = False
            result["refinement_mask"][0, 1] = False
            return result

        hook = model.refiner.heads[0].register_forward_hook(force_rejections)
        try:
            output = model(images, normalized_targets())
            output["pred_boxes"].square().sum().backward()
        finally:
            hook.remove()
        stats = model.refiner.last_prediction_safety_stats
        self.assertEqual(stats["layer_0_head_input_invalid_queries"], 1)
        self.assertEqual(stats["layer_0_head_decode_fallback_queries"], 1)
        self.assertLessEqual(stats["layer_0_head_input_invalid_queries"], 3)
        self.assertLessEqual(stats["layer_0_head_decode_fallback_queries"], 3)


if __name__ == "__main__":
    unittest.main()
