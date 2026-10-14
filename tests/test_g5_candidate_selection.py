"""Small deterministic checks for the read-only G5 candidate diagnostic."""

import importlib.util
import json
import unittest
from unittest import mock

import torch

from test_all_layer_strip import make_model


class CandidateSelectionTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('tools.analysis.g5_candidate_selection'))
        from tools.analysis.g5_candidate_selection import CandidateSelectionMetrics, trace_candidate_selection
        return CandidateSelectionMetrics, trace_candidate_selection

    def test_trace_preserves_outputs_and_replays_only_zero_visual_inputs(self):
        _, trace = self.api()
        torch.manual_seed(75)
        model = make_model(use_log_prior=False, query_fusion='channel_gate').eval()
        images = torch.randn(1, 3, 16, 24)
        score_inputs = {}
        with torch.no_grad():
            for head in model.refiner.heads:
                head.score_head[-1].weight.normal_(std=.2)
            expected = model(images)
        handles = [head.score_head.register_forward_pre_hook(
            lambda module, args, i=i: score_inputs.setdefault(i, []).append(args[0].clone())
        ) for i, head in enumerate(model.refiner.heads)]
        try:
            result = trace(model, images)
        finally:
            for handle in handles:
                handle.remove()
        for key in ('pred_boxes', 'pred_logits'):
            torch.testing.assert_close(result['outputs'][key], expected[key], atol=0, rtol=0)
        self.assertEqual(list(result['layers']), list(range(6)))
        changed = False
        with torch.no_grad():
            for i, layer in result['layers'].items():
                head = model.refiner.heads[i]
                self.assertEqual(len(score_inputs[i]), 1)
                inputs = score_inputs[i][0].clone()
                inputs[..., :head.hidden_dim] = 0
                delta = head.score_head(inputs).squeeze(-1).reshape_as(layer['prior_logits'])
                mask = layer['candidate_valid'] & layer['input_valid'][..., None, None]
                expected_zero = layer['prior_logits'] + torch.where(mask, delta, 0)
                torch.testing.assert_close(layer['zero_visual_logits'], expected_zero)
                changed |= not torch.equal(layer['zero_visual_logits'], layer['calibrated_logits'])
                self.assertFalse(layer['prior_logits'].requires_grad)
                self.assertFalse(head._forward_hooks)
                self.assertFalse(head.score_head._forward_hooks)
        self.assertTrue(changed)

    def test_hooks_are_removed_after_failure(self):
        _, trace = self.api()
        model = make_model(use_log_prior=False, query_fusion='channel_gate').eval()
        with mock.patch.object(model.decoder, 'forward', side_effect=RuntimeError('expected failure')):
            with self.assertRaisesRegex(RuntimeError, 'expected failure'):
                trace(model, torch.randn(1, 3, 16, 24))
        for head in model.refiner.heads:
            self.assertFalse(head._forward_hooks)
            self.assertFalse(head.score_head._forward_hooks)

    @staticmethod
    def sample():
        boxes = torch.tensor([[[.5, .5, .04, .04]]], dtype=torch.float64)
        edges = torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                           boxes[..., :2] + boxes[..., 2:] / 2), -1)
        positions = edges[..., None] + torch.tensor([-.01, .01, .10])
        logits = torch.tensor([.2, .3, .5], dtype=torch.float64).log().expand(1, 1, 4, 3).clone()
        layer = {
            'boxes_before': boxes.clone(), 'boxes_after': boxes.clone(),
            'prior_logits': logits, 'calibrated_logits': logits.clone(),
            'zero_visual_logits': logits.clone(), 'candidate_positions': positions,
            'candidate_valid': torch.tensor([True, True, False]).expand(1, 1, 4, 3).clone(),
            'input_valid': torch.ones(1, 1, dtype=torch.bool),
            'refinement_mask': torch.ones(1, 1, dtype=torch.bool),
        }
        return layer, [{'boxes': boxes[0]}], [(torch.tensor([0]), torch.tensor([0]))]

    def test_all_bin_expectation_and_invalid_probability_are_not_renormalized(self):
        Metrics, _ = self.api()
        layer, targets, indices = self.sample()
        metrics = Metrics()
        metrics.update(layer, targets, indices, (100, 100))
        result = metrics.compute()
        group = result['groups']['all']
        self.assertEqual((group['total_pairs'], group['pairs'], group['edges']), (1, 1, 4))
        self.assertEqual(group['excluded_edges'], 0)
        self.assertAlmostEqual(group['mean_nearest_valid_error_px'], 1., places=6)
        self.assertEqual(group['within1px_coverage'], 1.)
        for mode in ('prior', 'post', 'zero_visual'):
            self.assertAlmostEqual(group[mode]['mean_nearest_valid_probability'], .2)
            self.assertAlmostEqual(group[mode]['mean_within1px_probability'], .5)
            self.assertAlmostEqual(group[mode]['mean_invalid_candidate_probability'], .5)
            self.assertAlmostEqual(group[mode]['mean_expected_edge_abs_error_px'], 5.1, places=6)
            self.assertAlmostEqual(group[mode]['mean_argmax_valid_edge_abs_error_px'], 1., places=6)
            self.assertEqual(group[mode]['top1_nearest_match_rate'], 0.)
        self.assertEqual(result['groups']['short_lt8']['pairs'], 1)
        self.assertIsNone(result['groups']['short_ge32']['mean_nearest_valid_error_px'])
        json.dumps(result, allow_nan=False)

    def test_invalid_inputs_refinement_candidates_and_nonfinite_edges_are_excluded(self):
        Metrics, _ = self.api()
        metrics = Metrics()
        for reason in ('input_valid', 'refinement_mask', 'candidate_valid', 'prior_logits', 'candidate_positions'):
            layer, targets, indices = self.sample()
            if reason == 'prior_logits':
                layer[reason][0, 0, 0, 0] = float('nan')
            elif reason == 'candidate_positions':
                # Even an invalid candidate participates in the all-K expectation.
                layer[reason][0, 0, 0, 2] = float('inf')
            else:
                layer[reason].fill_(False)
            metrics.update(layer, targets, indices, (100, 100))
        layer, targets, _ = self.sample()
        metrics.update(layer, targets, [(torch.tensor([], dtype=torch.long),) * 2], (100, 100))
        result = metrics.compute()
        group = result['groups']['all']
        self.assertEqual((group['total_pairs'], group['pairs'], group['excluded_pairs']), (5, 2, 3))
        self.assertEqual((group['total_edges'], group['edges'], group['excluded_edges']), (20, 6, 14))
        self.assertEqual(group['partially_excluded_pairs'], 2)
        self.assertEqual(sum(group['excluded_edges_by_reason'].values()), 14)
        json.dumps(result, allow_nan=False)

    def test_short_side_groups_use_input_pixels_and_half_open_thresholds(self):
        Metrics, _ = self.api()
        metrics = Metrics()
        for short in (7, 8, 15, 16, 31, 32):
            layer, targets, indices = self.sample()
            targets[0]['boxes'][0, 2:] = torch.tensor([short / 200., short / 100.], dtype=torch.float64)
            metrics.update(layer, targets, indices, (100, 200))
        groups = metrics.compute()['groups']
        self.assertEqual([groups[name]['total_pairs'] for name in Metrics.GROUP_NAMES], [6, 1, 2, 2, 1])


if __name__ == '__main__':
    unittest.main()
