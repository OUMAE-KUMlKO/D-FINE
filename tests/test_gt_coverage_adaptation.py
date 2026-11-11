import unittest
import torch

from test_all_layer_strip import make_model, matcher
from test_gt_refinement import normalized_targets
from src.zoo.dfine.dfine_criterion import DFINECriterion
from tools.train_gt_coverage_adaptation import (
    GTCoverageAdaptation, gt_candidate_selection, fixed_digest, DecoderEMA)


class GTAdaptationTests(unittest.TestCase):
    def test_gt_selects_missing_instance_without_editing_content_or_box(self):
        memory = torch.arange(12, dtype=torch.float).reshape(1, 3, 4)
        logits = torch.tensor([[[3., 0], [2., 0], [0., 1]]])
        boxes = torch.tensor([[[.2, .2, .1, .1], [.2, .2, .1, .1], [.8, .8, .1, .1]]])
        bbox = torch.nn.Linear(4, 4)
        torch.nn.init.zeros_(bbox.weight); torch.nn.init.zeros_(bbox.bias)
        targets = [dict(boxes=boxes[0, [0, 2]], labels=torch.tensor([0, 1]))]
        result, chosen, records = gt_candidate_selection(memory, logits, torch.logit(boxes), bbox, 2, targets, 3, 1)
        torch.testing.assert_close(result[0], memory[:, [0, 2]])
        torch.testing.assert_close(result[2], torch.logit(boxes[:, [0, 2]]))
        self.assertIsNone(result[3])
        self.assertEqual(records[0]['native_covered'], 1)
        self.assertEqual(records[0]['selected_covered'], 2)
        self.assertEqual(chosen.tolist(), [[0, 2]])

    def test_decoder_learns_with_dn_while_candidate_generator_stays_frozen(self):
        torch.manual_seed(0); torch.set_num_threads(1)
        model = GTCoverageAdaptation(make_model(), candidates=20, budget=2)
        before = fixed_digest(model)
        criterion = DFINECriterion(matcher(), dict(loss_vfl=1, loss_bbox=5, loss_giou=2,
            loss_fgl=.15, loss_ddf=0), ['vfl', 'boxes', 'local'], num_classes=2, reg_max=4)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        output = model(torch.randn(1, 3, 16, 24), normalized_targets())
        self.assertIn('dn_outputs', output)
        self.assertNotIn('query_aux_outputs', output)
        self.assertEqual(output['pred_boxes'].shape, (1, 3, 4))
        sum(criterion(output, normalized_targets()).values()).backward()
        self.assertGreater(model.detector.decoder.denoising_class_embed.weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.detector.decoder.decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        optimizer.step()
        self.assertEqual(fixed_digest(model), before)
        ema = DecoderEMA(model)
        ema.update(model)
        with ema.apply(model):
            self.assertEqual(fixed_digest(model), before)
        self.assertFalse(model.detector.backbone.training)
        self.assertFalse(model.detector.decoder.enc_output.training)
        model.eval()
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 16, 24))
        result = model(torch.randn(1, 3, 16, 24), normalized_targets())
        self.assertEqual(result['pred_boxes'].shape, (1, 3, 4))


if __name__ == '__main__':
    unittest.main()
