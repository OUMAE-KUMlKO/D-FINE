"""GT guidance changes G5 visual inputs while preserving its decoder and losses."""

import copy
import importlib.util
import unittest

import torch

from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.all_layer_strip import LayerStripHeads
from test_all_layer_strip import make_model, matcher
from test_gt_refinement import normalized_targets
import test_strip_probability_head


class GTAllLayerStripTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(184)
        self.images = torch.randn(1, 3, 16, 24)

    def guided(self, plain):
        self.assertIsNotNone(importlib.util.find_spec('src.zoo.dfine.gt_all_layer_strip'))
        from src.zoo.dfine.gt_all_layer_strip import DFINEGTAllLayerStrip
        guided = DFINEGTAllLayerStrip(
            copy.deepcopy(plain.backbone), copy.deepcopy(plain.encoder),
            copy.deepcopy(plain.decoder), matcher(), hidden_dim=8,
            query_chunk_size=2, checkpoint_strips=plain.checkpoint_strips,
            use_log_prior=False, query_fusion='channel_gate',
        )
        guided.load_state_dict(plain.state_dict(), strict=True)
        return guided

    def plain(self, checkpoint_strips=False):
        plain = make_model(checkpoint_strips=checkpoint_strips,
                           use_log_prior=False, query_fusion='channel_gate')
        with torch.no_grad():
            for head in plain.refiner.heads:
                head.score_head[-1].weight.normal_(std=.2)
        return plain

    def test_requires_gt_and_empty_gt_preserves_normal_six_layer_outputs(self):
        plain = self.plain().eval()
        guided = self.guided(plain).eval()
        with self.assertRaisesRegex(ValueError, 'requires GT'):
            guided(self.images)
        empty = [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}]
        with torch.no_grad():
            expected = plain(self.images)
            actual = guided(self.images, empty)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        self.assertTrue(guided.requires_gt)
        self.assertFalse(guided.supports_gt_refinement_metrics)

    def test_gt_changes_predictions_and_target_formats_agree(self):
        guided = self.guided(self.plain()).eval()
        calls = []
        handles = [head.register_forward_hook(
            lambda module, args, output, index=i: calls.append(index)
        ) for i, head in enumerate(guided.refiner.heads)]
        targets = normalized_targets()
        boxes = targets[0]['boxes']
        absolute = [{'boxes': torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                                        boxes[:, :2] + boxes[:, 2:] / 2), -1)
                                  * torch.tensor([24, 16, 24, 16]),
                     'labels': targets[0]['labels']}]
        try:
            with torch.no_grad():
                first = guided(self.images, targets, target_format='cxcywh_normalized')
                second = guided(self.images, absolute)
                other = normalized_targets()
                other[0]['boxes'] = torch.tensor([[.75, .75, .15, .15]])
                changed = guided(self.images, other, target_format='cxcywh_normalized')
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, list(range(6)) * 3)
        torch.testing.assert_close(first['pred_boxes'], second['pred_boxes'])
        self.assertFalse(torch.equal(first['pred_boxes'], changed['pred_boxes']))

    def test_gt_training_retains_aux_dn_losses_and_joint_finite_gradients(self):
        guided = self.guided(self.plain(checkpoint_strips=True)).train()
        counts = []
        hook = guided.refiner.heads[0].register_forward_pre_hook(
            lambda module, args: counts.append(args[1].shape[1]))
        try:
            outputs = guided(self.images, normalized_targets())
        finally:
            hook.remove()
        self.assertEqual(counts, [3])
        self.assertEqual(len(outputs['aux_outputs']), 5)
        self.assertEqual(len(outputs['dn_outputs']), 6)
        criterion = DFINECriterion(
            matcher(), {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2,
                        'loss_fgl': .15, 'loss_ddf': 1.5},
            ['vfl', 'boxes', 'local'], num_classes=2, reg_max=4)
        losses = criterion(outputs, normalized_targets())
        self.assertIn('loss_fgl_dn_0', losses)
        sum(losses.values()).backward()
        for head in guided.refiner.heads:
            self.assertGreater(head.score_head[-1].weight.grad.abs().sum(), 0)
        for module in (guided.backbone, guided.encoder, guided.decoder, guided.refiner):
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in grads))

    def test_training_stale_zero_crowd_metadata_matches_no_flags_without_mutation(self):
        guided = self.guided(self.plain()).train()
        clean = normalized_targets()
        stale = normalized_targets()
        flags = torch.zeros(3, dtype=torch.long)
        stale[0]['iscrowd'] = flags
        original_boxes = stale[0]['boxes']
        decoder_targets = []
        hook = guided.decoder.register_forward_pre_hook(
            lambda module, args: decoder_targets.append(args[1]))
        try:
            with torch.no_grad():
                torch.manual_seed(94)
                expected = guided(self.images, clean)
                torch.manual_seed(94)
                actual = guided(self.images, stale)
        finally:
            hook.remove()
        for key in ('pred_boxes', 'pred_logits', 'pred_corners'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        self.assertIs(decoder_targets[1], stale)
        self.assertIs(stale[0]['iscrowd'], flags)
        self.assertIs(stale[0]['boxes'], original_boxes)
        self.assertEqual(flags.tolist(), [0, 0, 0])

    def test_misaligned_nonzero_crowd_or_eval_metadata_is_still_rejected(self):
        guided = self.guided(self.plain())
        for training, flags in ((True, [0, 1]), (False, [0, 0]), (False, [0, 1])):
            with self.subTest(training=training, flags=flags):
                targets = normalized_targets()
                targets[0]['iscrowd'] = torch.tensor(flags)
                guided.train(training)
                with self.assertRaisesRegex(ValueError, 'iscrowd must have one value'):
                    guided(self.images, targets, target_format='cxcywh_normalized')

    def test_gt_mask_selects_only_nearest_valid_visual_channels(self):
        fixture = test_strip_probability_head.StripProbabilityHeadTests()
        fixture.setUp()
        head = fixture.make_head()
        with torch.no_grad():
            head.score_head[-1].weight.normal_(std=.2)
        captured = []
        hook = head.score_head.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        gt_boxes = torch.tensor([[[.42, .48, .2, .25], [.6, .6, .2, .2]]])
        try:
            normal = fixture.forward_head(head)
            guided = head(fixture.features, fixture.queries, fixture.boxes, (40, 80),
                          fixture.distribution, gt_boxes=gt_boxes,
                          gt_matched=torch.tensor([[True, False]]))
        finally:
            hook.remove()
        before, after = captured
        gt_edges = torch.cat((gt_boxes[..., :2] - gt_boxes[..., 2:] / 2,
                              gt_boxes[..., :2] + gt_boxes[..., 2:] / 2), -1)
        distance = (normal['candidate_positions'] - gt_edges[..., None]).abs()
        nearest = distance.masked_fill(~normal['candidate_valid'], float('inf')).argmin(-1)
        keep = torch.arange(5)[None, :] == nearest[0, 0, :, None]
        torch.testing.assert_close(after[0, ..., :8], before[0, ..., :8] * keep[..., None])
        torch.testing.assert_close(after[0, ..., 8:], before[0, ..., 8:], atol=0, rtol=0)
        torch.testing.assert_close(after[1], before[1], atol=0, rtol=0)
        for key in ('candidate_positions', 'candidate_valid'):
            torch.testing.assert_close(guided[key], normal[key], atol=0, rtol=0)
        torch.testing.assert_close(guided['pred_boxes'][:, 1], normal['pred_boxes'][:, 1], atol=0, rtol=0)

    def test_denoising_logits_bypass_gt_visual_guidance(self):
        fixture = test_strip_probability_head.StripProbabilityHeadTests()
        fixture.setUp()
        heads = LayerStripHeads(1, in_channels=2, query_dim=3, hidden_dim=8,
                                num_points=3, query_chunk_size=1).eval()
        with torch.no_grad():
            heads.heads[0].score_head[-1].weight.normal_(std=.2)
        distribution = dict(fixture.distribution, num_denoising=1)
        actual = heads(0, fixture.features, fixture.queries, fixture.boxes, (40, 80),
                       distribution, gt_boxes=torch.tensor([[[.4, .4, .2, .2]]]),
                       gt_matched=torch.ones(1, 1, dtype=torch.bool))
        torch.testing.assert_close(actual[:, :1], fixture.distribution['pred_corners'][:, :1],
                                   atol=0, rtol=0)

    def test_checkpointing_preserves_guided_outputs_and_gradients(self):
        plain = self.plain()
        first = self.guided(plain).train()
        second = self.guided(plain).train()
        second.checkpoint_strips = True
        torch.manual_seed(71)
        a = first(self.images, normalized_targets())
        torch.manual_seed(71)
        b = second(self.images, normalized_targets())
        torch.testing.assert_close(a['pred_boxes'], b['pred_boxes'], atol=0, rtol=0)
        a['pred_boxes'].square().sum().backward()
        b['pred_boxes'].square().sum().backward()
        for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5, msg=name)


if __name__ == '__main__':
    unittest.main()
