"""CPU checks for optional same-layer D-FINE distribution export."""

import unittest

import torch

from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function


class DFineDistributionExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(31)
        self.features = [torch.randn(2, 8, 4, 5)]

    def make_model(self, eval_idx=-1):
        model = DFINETransformer(
            num_classes=2, hidden_dim=8, num_queries=3,
            feat_channels=[8], feat_strides=[1], num_levels=1,
            num_points=2, nhead=2, num_layers=3, dim_feedforward=16,
            num_denoising=0, reg_max=4, eval_idx=eval_idx, aux_loss=False,
        ).eval()
        # Release the uniform zero-initialized distributions so exporting a
        # wrong layer or the final boxes as references cannot pass by accident.
        with torch.no_grad():
            for index, head in enumerate(model.dec_bbox_head):
                head.layers[-1].weight.normal_(std=0.05)
                head.layers[-1].bias.copy_(
                    torch.linspace(-0.4, 0.4, 20) * (index + 1)
                )
        return model

    @torch.no_grad()
    def test_export_reconstructs_boxes_and_keeps_first_layer_references(self):
        full_model = self.make_model()
        full = full_model(self.features, return_distribution=True)
        for eval_idx in (0, 1, -1):
            with self.subTest(eval_idx=eval_idx):
                model = self.make_model(eval_idx)
                model.load_state_dict(full_model.state_dict())
                outputs = model(self.features, return_distribution=True)
                self.assertEqual(tuple(outputs["pred_corners"].shape), (2, 3, 20))
                self.assertEqual(tuple(outputs["ref_points"].shape), (2, 3, 4))
                probabilities = outputs["pred_corners"].reshape(2, 3, 4, 5).softmax(-1)
                support = weighting_function(
                    model.reg_max, outputs["up"], outputs["reg_scale"]
                )
                expectation = (probabilities * support).sum(-1)
                reconstructed = distance2bbox(
                    outputs["ref_points"], expectation, outputs["reg_scale"]
                )
                torch.testing.assert_close(
                    reconstructed, outputs["pred_boxes"], atol=1e-6, rtol=1e-5
                )
                torch.testing.assert_close(outputs["ref_points"], full["ref_points"])
                self.assertFalse(torch.allclose(outputs["ref_points"], outputs["pred_boxes"]))
                if eval_idx != -1:
                    self.assertFalse(torch.allclose(outputs["pred_boxes"], full["pred_boxes"]))

    @torch.no_grad()
    def test_query_and_cumulative_distribution_follow_early_stop_layer(self):
        model = self.make_model(eval_idx=1)
        layer_outputs = {}
        corner_updates = []
        hooks = [
            layer.register_forward_hook(
                lambda module, args, output, index=index: layer_outputs.update({index: output})
            )
            for index, layer in enumerate(model.decoder.layers)
        ]
        hooks += [
            head.register_forward_hook(
                lambda module, args, output: corner_updates.append(output)
            )
            for head in model.dec_bbox_head
        ]
        try:
            outputs = model(self.features, return_query=True, return_distribution=True)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(set(layer_outputs), {0, 1})
        self.assertEqual(len(corner_updates), 2)
        torch.testing.assert_close(outputs["pred_queries"], layer_outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(
            outputs["pred_corners"], sum(corner_updates), rtol=0, atol=0
        )
        query_only = model(self.features, return_query=True)
        self.assertEqual(set(query_only), {"pred_boxes", "pred_logits", "pred_queries"})
        for key in query_only:
            torch.testing.assert_close(outputs[key], query_only[key], rtol=0, atol=0)

    @torch.no_grad()
    def test_default_outputs_and_state_dict_are_unchanged(self):
        model = self.make_model()
        state = {name: value.clone() for name, value in model.state_dict().items()}
        default = model(self.features)
        disabled = model(self.features, return_distribution=False)
        exported = model(self.features, return_distribution=True)
        self.assertEqual(set(default), {"pred_logits", "pred_boxes"})
        self.assertEqual(set(disabled), set(default))
        self.assertEqual(
            set(exported),
            set(default) | {"pred_corners", "ref_points", "up", "reg_scale"},
        )
        for key in default:
            torch.testing.assert_close(default[key], disabled[key], rtol=0, atol=0)
            torch.testing.assert_close(default[key], exported[key], rtol=0, atol=0)
        self.assertEqual(set(model.state_dict()), set(state))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        training = model.train()(self.features)
        training_disabled = model(self.features, return_distribution=False)
        self.assertEqual(
            set(training),
            {"pred_logits", "pred_boxes", "pred_corners", "ref_points", "up", "reg_scale"},
        )
        for key in training:
            torch.testing.assert_close(training[key], training_disabled[key], rtol=0, atol=0)

    def test_distribution_export_is_eval_only(self):
        model = self.make_model().train()
        with self.assertRaisesRegex(ValueError, "return_distribution.*eval"):
            model(self.features, return_distribution=True)


if __name__ == "__main__":
    unittest.main()
