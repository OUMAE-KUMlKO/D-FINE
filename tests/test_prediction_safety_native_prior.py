"""A finite negative prior retains the native G5 skip and loss behavior."""

import unittest

import torch

from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.box_ops import box_cxcywh_to_xyxy
from test_all_layer_strip import make_model, matcher, normalized_targets


class NativePriorSafetyTests(unittest.TestCase):
    def test_negative_prior_uses_real_strip_skip_and_finite_native_losses(self):
        for gate, reference in ((True, 'corrected'), (False, 'pre_correction'),
                                (True, 'pre_correction')):
            with self.subTest(gate=gate, reference=reference):
                torch.manual_seed(41)
                model = make_model(adaptive_correction_gate=gate,
                                   next_reference_source=reference).train()
                images, targets = torch.randn(1, 3, 16, 24), normalized_targets()
                observed = {}

                def negative_height(module, args, output):
                    output = output.clone()
                    # Keep DN and all other matching queries unchanged.
                    query = output.shape[1] - 3 + 1
                    corners = output[0, query].reshape(4, 5)
                    for edge in (1, 3):
                        corners[edge].fill_(-30)
                        corners[edge, 0] = 30
                    return output

                corrupt = model.decoder.dec_bbox_head[-1].register_forward_hook(negative_height)
                capture = model.refiner.heads[-1].register_forward_hook(
                    lambda module, args, result: observed.update(
                        input_valid=result['input_valid'].detach().clone()))
                try:
                    torch.manual_seed(29)
                    native = model.decoder(model.encoder(model.backbone(images)), targets)
                    torch.manual_seed(29)
                    guarded = model(images, targets)
                finally:
                    corrupt.remove()
                    capture.remove()

                self.assertLess(float(guarded['pred_boxes'][0, 1, 3]), 0)
                self.assertFalse(bool(observed['input_valid'][0, 1]))
                for key in ('pred_boxes', 'pred_corners'):
                    torch.testing.assert_close(guarded[key], native[key], atol=1e-6, rtol=1e-6)
                xyxy = box_cxcywh_to_xyxy(guarded['pred_boxes'])
                self.assertTrue(torch.isfinite(xyxy).all())
                self.assertTrue((xyxy[..., 2:] >= xyxy[..., :2]).all())

                criterion = DFINECriterion(
                    matcher(), {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,
                                'loss_fgl': .15, 'loss_ddf': 1.5},
                    ['vfl', 'boxes', 'local'], num_classes=2, reg_max=4)
                loss = sum(criterion(guarded, targets).values())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))


if __name__ == '__main__':
    unittest.main()
