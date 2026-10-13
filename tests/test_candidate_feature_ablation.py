"""Inference-only ablation of encoded candidate visual features."""

import unittest

import torch

from test_all_layer_strip import make_model
from tools.analysis.candidate_feature_ablation import candidate_feature_ablation


torch.set_num_threads(1)


class CandidateFeatureAblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
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

    @staticmethod
    def assert_output_equal(first, second):
        for key in ("pred_boxes", "pred_logits"):
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def capture_edge_outputs(self, mode, seed=0):
        captured = [[] for _ in self.model.refiner.heads]
        with candidate_feature_ablation(self.model, mode, seed):
            hooks = [
                head.edge_fusion.register_forward_hook(
                    lambda module, args, output, index=index:
                    captured[index].append(output.detach().clone())
                )
                for index, head in enumerate(self.model.refiner.heads)
            ]
            try:
                result = self.infer()
            finally:
                for hook in hooks:
                    hook.remove()
        return result, [torch.cat(chunks) for chunks in captured]

    def test_normal_matches_plain_inference_exactly(self):
        plain = self.infer()
        with candidate_feature_ablation(self.model, "normal", seed=123):
            normal = self.infer()
        self.assert_output_equal(plain, normal)

    def test_zero_clears_only_first_score_visual_input_and_reaches_all_layers(self):
        head = self.model.refiner.heads[0]
        baseline_inputs, zero_inputs = [], []
        baseline_distribution, zero_distribution = [], []

        def capture_distribution(destination):
            def hook(module, args):
                destination.append({
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in args[4].items()
                })
            return hook

        hooks = [
            head.score_head.register_forward_pre_hook(
                lambda module, args: baseline_inputs.append(args[0].detach().clone())
            ),
            head.register_forward_pre_hook(capture_distribution(baseline_distribution)),
        ]
        try:
            self.infer()
        finally:
            for hook in hooks:
                hook.remove()

        with candidate_feature_ablation(self.model, "zero"):
            hooks = [
                head.score_head.register_forward_pre_hook(
                    lambda module, args: zero_inputs.append(args[0].detach().clone())
                ),
                head.register_forward_pre_hook(capture_distribution(zero_distribution)),
            ]
            edge_outputs = [[] for _ in self.model.refiner.heads]
            hooks.extend(
                layer.edge_fusion.register_forward_hook(
                    lambda module, args, output, index=index:
                    edge_outputs[index].append(output.detach().clone())
                )
                for index, layer in enumerate(self.model.refiner.heads)
            )
            try:
                self.infer()
            finally:
                for hook in hooks:
                    hook.remove()

        visual_width = head.hidden_dim
        self.assertGreater(float(baseline_inputs[0][..., :visual_width].abs().sum()), 0)
        torch.testing.assert_close(
            zero_inputs[0][..., :visual_width],
            torch.zeros_like(zero_inputs[0][..., :visual_width]), atol=0, rtol=0,
        )
        torch.testing.assert_close(
            zero_inputs[0][..., visual_width:],
            baseline_inputs[0][..., visual_width:], atol=0, rtol=0,
        )
        for key in baseline_distribution[0]:
            if torch.is_tensor(baseline_distribution[0][key]):
                torch.testing.assert_close(
                    zero_distribution[0][key], baseline_distribution[0][key], atol=0, rtol=0,
                )
        self.assertEqual(len(edge_outputs), 6)
        for index, chunks in enumerate(edge_outputs):
            self.assertTrue(chunks, index)
            self.assertEqual(sum(torch.count_nonzero(chunk).item() for chunk in chunks), 0, index)

    def test_noise_is_seeded_per_layer_and_does_not_consume_global_rng(self):
        global_state = torch.random.get_rng_state().clone()
        first_result, first = self.capture_edge_outputs("noise", seed=19)
        torch.testing.assert_close(torch.random.get_rng_state(), global_state, atol=0, rtol=0)

        second_result, second = self.capture_edge_outputs("noise", seed=19)
        third_result, third = self.capture_edge_outputs("noise", seed=20)
        for index in range(6):
            torch.testing.assert_close(first[index], second[index], atol=0, rtol=0)
            self.assertFalse(torch.equal(first[index], third[index]), index)
        self.assert_output_equal(first_result, second_result)
        self.assertFalse(torch.equal(first_result["pred_boxes"], third_result["pred_boxes"]))

    def test_context_removes_hooks_after_normal_and_exceptional_exit(self):
        modules = [head.edge_fusion for head in self.model.refiner.heads]
        before = [len(module._forward_hooks) for module in modules]
        with candidate_feature_ablation(self.model, "zero"):
            self.assertEqual(
                [len(module._forward_hooks) for module in modules],
                [count + 1 for count in before],
            )
        self.assertEqual([len(module._forward_hooks) for module in modules], before)

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with candidate_feature_ablation(self.model, "noise"):
                raise RuntimeError("sentinel")
        self.assertEqual([len(module._forward_hooks) for module in modules], before)

    def test_rejects_training_or_ground_truth_models_and_unknown_modes(self):
        with self.assertRaisesRegex(ValueError, "eval"):
            with candidate_feature_ablation(self.model.train(), "zero"):
                pass
        self.model.eval()
        self.model.requires_gt = True
        with self.assertRaisesRegex(ValueError, "requires_gt"):
            with candidate_feature_ablation(self.model, "zero"):
                pass
        self.model.requires_gt = False
        with self.assertRaisesRegex(ValueError, "mode"):
            with candidate_feature_ablation(self.model, "random"):
                pass


if __name__ == "__main__":
    unittest.main()
