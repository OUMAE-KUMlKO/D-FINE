"""Choice of corrected or pre-correction boxes for the next decoder layer."""

import unittest
from unittest import mock

import torch

from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.dfine_utils import distance2bbox, weighting_function
from test_all_layer_strip import make_model


torch.set_num_threads(1)


class StripReferenceSourceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.features = [torch.randn(1, 8, 4, 5)]
        self.images = torch.randn(1, 3, 16, 24)

    @staticmethod
    def make_decoder():
        model = DFINETransformer(
            num_classes=2, hidden_dim=8, num_queries=3,
            feat_channels=[8], feat_strides=[1], num_levels=1,
            num_points=2, nhead=2, num_layers=6, dim_feedforward=16,
            num_denoising=0, reg_max=4,
        )
        with torch.no_grad():
            for head in model.dec_bbox_head:
                head.layers[-1].weight.normal_(std=0.08)
        return model.eval()

    @staticmethod
    def decode(model, corners, distribution):
        support = weighting_function(
            model.reg_max, distribution["up"], distribution["reg_scale"],
        )
        return distance2bbox(
            distribution["ref_points"], model.integral(corners, support),
            distribution["reg_scale"],
        )

    def run_source(self, model, source):
        model.train()
        records = {"callbacks": {}, "layer_refs": {}, "query_refs": {}, "lqe": {}}
        hooks = [
            model.decoder.layers[1].register_forward_pre_hook(
                lambda module, args: records["layer_refs"].update({1: args[1].detach().clone()}),
            ),
            model.query_pos_head.register_forward_pre_hook(
                lambda module, args: records["query_refs"].setdefault(
                    len(records["query_refs"]), args[0].detach().clone(),
                ),
            ),
            model.decoder.lqe_layers[0].register_forward_pre_hook(
                lambda module, args: records["lqe"].update({0: args[1].detach().clone()}),
            ),
        ]
        head_one = []
        hooks.append(model.dec_bbox_head[1].register_forward_hook(
            lambda module, args, output: head_one.append(output.detach().clone()),
        ))
        correction = torch.linspace(-0.3, 0.3, 20)

        def refine(index, query, boxes, distribution):
            prior = distribution["pred_corners"]
            corrected = prior + correction.to(prior) if index == 0 else prior
            records["callbacks"][index] = {
                "boxes": boxes.detach().clone(),
                "prior": prior.detach().clone(),
                "corrected": corrected.detach().clone(),
                "distribution": distribution,
            }
            return corrected

        try:
            torch.manual_seed(19)
            output = model(
                self.features, distribution_refiner=refine,
                next_reference_source=source,
            )
        finally:
            for hook in hooks:
                hook.remove()
        records["output"] = output
        records["head_one"] = head_one[0]
        return records

    def test_reference_and_query_position_switch_but_feedback_lqe_stay_corrected(self):
        model = self.make_decoder()
        corrected = self.run_source(model, "corrected")
        pre = self.run_source(model, "pre_correction")

        for records, source in ((corrected, "corrected"), (pre, "pre_correction")):
            first = records["callbacks"][0]
            corrected_box = self.decode(model, first["corrected"], first["distribution"])
            expected_reference = corrected_box if source == "corrected" else first["boxes"]
            torch.testing.assert_close(
                records["layer_refs"][1].squeeze(2), expected_reference, atol=0, rtol=0,
            )
            torch.testing.assert_close(records["query_refs"][1], expected_reference, atol=0, rtol=0)
            torch.testing.assert_close(records["lqe"][0], first["corrected"], atol=0, rtol=0)
            torch.testing.assert_close(
                records["output"]["aux_outputs"][0]["pred_corners"],
                first["corrected"], atol=0, rtol=0,
            )
            torch.testing.assert_close(
                records["output"]["aux_outputs"][0]["pred_boxes"],
                corrected_box, atol=0, rtol=0,
            )
            torch.testing.assert_close(
                records["callbacks"][1]["prior"],
                records["head_one"] + first["corrected"], atol=0, rtol=0,
            )

        torch.testing.assert_close(
            corrected["callbacks"][0]["corrected"],
            pre["callbacks"][0]["corrected"], atol=0, rtol=0,
        )
        torch.testing.assert_close(corrected["lqe"][0], pre["lqe"][0], atol=0, rtol=0)
        self.assertFalse(torch.equal(corrected["layer_refs"][1], pre["layer_refs"][1]))

    def test_default_and_no_refiner_paths_remain_exactly_compatible(self):
        model = self.make_decoder().eval()
        default = model(self.features, return_distribution=True)
        explicit = model(
            self.features, return_distribution=True,
            next_reference_source="corrected",
        )
        pre = model(
            self.features, return_distribution=True,
            next_reference_source="pre_correction",
        )
        for key in default:
            if isinstance(default[key], torch.Tensor):
                torch.testing.assert_close(default[key], explicit[key], atol=0, rtol=0)
                torch.testing.assert_close(default[key], pre[key], atol=0, rtol=0)

    def test_wrapper_passes_only_nondefault_source_and_rejects_unknown_value(self):
        default = make_model()
        with mock.patch.object(default.decoder, "forward", return_value={}) as forward:
            default(self.images)
        self.assertNotIn("next_reference_source", forward.call_args.kwargs)

        pre = make_model(next_reference_source="pre_correction")
        with mock.patch.object(pre.decoder, "forward", return_value={}) as forward:
            pre(self.images)
        self.assertEqual(forward.call_args.kwargs["next_reference_source"], "pre_correction")

        with self.assertRaisesRegex(ValueError, "next_reference_source"):
            make_model(next_reference_source="unknown")


if __name__ == "__main__":
    unittest.main()
