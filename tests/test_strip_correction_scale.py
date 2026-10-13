"""Inference-only scaling of all-layer strip correction deltas."""

import unittest

import torch

from test_all_layer_strip import make_model
from tools.analysis.strip_correction_scale import strip_correction_scale


torch.set_num_threads(1)


class StripCorrectionScaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        self.images = torch.randn(1, 3, 16, 24)
        self.model = make_model(
            use_log_prior=False, query_fusion="channel_gate",
            normal_points=7, normal_kernel_size=3,
        ).eval()
        with torch.no_grad():
            for head in self.model.refiner.heads:
                head.score_head[-1].weight.normal_(std=0.05)

    def infer(self):
        with torch.no_grad():
            return self.model(self.images)

    def test_alpha_one_matches_plain_inference_exactly(self):
        plain = self.infer()
        with strip_correction_scale(self.model, 1):
            scaled = self.infer()
        for key in ("pred_boxes", "pred_logits"):
            torch.testing.assert_close(plain[key], scaled[key], atol=0, rtol=0)

    def test_alpha_zero_matches_same_models_native_decoder(self):
        with torch.no_grad():
            native = self.model.decoder(
                self.model.encoder(self.model.backbone(self.images))
            )
        with strip_correction_scale(self.model, 0):
            disabled = self.infer()
        torch.testing.assert_close(
            disabled["pred_boxes"], native["pred_boxes"], atol=1e-6, rtol=1e-6,
        )
        torch.testing.assert_close(
            disabled["pred_logits"], native["pred_logits"], atol=0, rtol=0,
        )

    def test_half_scales_every_score_and_only_halves_first_layer_correction(self):
        first_head = self.model.refiner.heads[0]
        normal_prior, normal_output = [], []
        hooks = [
            first_head.register_forward_pre_hook(
                lambda module, args: normal_prior.append(
                    args[4]["pred_corners"].detach().clone()
                )
            ),
            first_head.register_forward_hook(
                lambda module, args, output: normal_output.append(
                    output["calibrated_corners"].detach().clone()
                )
            ),
        ]
        try:
            self.infer()
        finally:
            for hook in hooks:
                hook.remove()

        raw_scores = [[] for _ in self.model.refiner.heads]
        scaled_scores = [[] for _ in self.model.refiner.heads]
        half_prior, half_output = [], []
        raw_hooks = [
            head.score_head.register_forward_hook(
                lambda module, args, output, index=index:
                raw_scores[index].append(output.detach().clone())
            )
            for index, head in enumerate(self.model.refiner.heads)
        ]
        with strip_correction_scale(self.model, 0.5):
            scaled_hooks = [
                head.score_head.register_forward_hook(
                    lambda module, args, output, index=index:
                    scaled_scores[index].append(output.detach().clone())
                )
                for index, head in enumerate(self.model.refiner.heads)
            ]
            head_hooks = [
                first_head.register_forward_pre_hook(
                    lambda module, args: half_prior.append(
                        args[4]["pred_corners"].detach().clone()
                    )
                ),
                first_head.register_forward_hook(
                    lambda module, args, output: half_output.append(
                        output["calibrated_corners"].detach().clone()
                    )
                ),
            ]
            try:
                self.infer()
            finally:
                for hook in scaled_hooks + head_hooks:
                    hook.remove()
        for hook in raw_hooks:
            hook.remove()

        for index in range(6):
            self.assertTrue(raw_scores[index], index)
            self.assertEqual(len(raw_scores[index]), len(scaled_scores[index]), index)
            for raw, scaled in zip(raw_scores[index], scaled_scores[index]):
                torch.testing.assert_close(scaled, raw * 0.5, atol=0, rtol=0)

        normal_prior = normal_prior[0].reshape_as(normal_output[0])
        half_prior = half_prior[0].reshape_as(half_output[0])
        torch.testing.assert_close(half_prior, normal_prior, atol=0, rtol=0)
        torch.testing.assert_close(
            half_output[0] - half_prior,
            (normal_output[0] - normal_prior) * 0.5,
            atol=1e-6, rtol=1e-6,
        )

    def test_hooks_are_removed_after_regular_and_exceptional_exit(self):
        modules = [head.score_head for head in self.model.refiner.heads]
        before = [len(module._forward_hooks) for module in modules]
        with strip_correction_scale(self.model, 0.5):
            self.assertEqual(
                [len(module._forward_hooks) for module in modules],
                [count + 1 for count in before],
            )
        self.assertEqual([len(module._forward_hooks) for module in modules], before)

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with strip_correction_scale(self.model, 0):
                raise RuntimeError("sentinel")
        self.assertEqual([len(module._forward_hooks) for module in modules], before)

    def test_rejects_invalid_alpha_training_and_ground_truth_models(self):
        for alpha in (-0.1, float("inf"), float("-inf"), float("nan")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    with strip_correction_scale(self.model, alpha):
                        pass
        with self.assertRaisesRegex(ValueError, "eval"):
            with strip_correction_scale(self.model.train(), 0.5):
                pass
        self.model.eval()
        self.model.requires_gt = True
        with self.assertRaisesRegex(ValueError, "requires_gt"):
            with strip_correction_scale(self.model, 0.5):
                pass


if __name__ == "__main__":
    unittest.main()
