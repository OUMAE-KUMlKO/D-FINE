"""Read-only G5 candidate observation with caller-fixed query/GT assignment.

GT is used only by the streaming metrics, never by the model or score replay.
Candidate probabilities and expectations always include all K original bins.
"""

import math

import torch


@torch.inference_mode()
def trace_candidate_selection(model, images):
    """Observe normal inference and locally replay score heads with zero vision.

For the original E0 channel-gated head, only the first hidden_dim score-input
channels contain visual features. Zeroing them preserves the query/geometry,
edge embedding, offset and original prior. Replay does not alter the decoder
trajectory, and replayed logits are not passed back into the detector.
"""
    if model.training or getattr(model, 'requires_gt', False):
        raise ValueError('Candidate tracing requires eval-mode GT-free inference')
    for head in model.refiner.heads:
        if (head.query_fusion != 'channel_gate' or head.use_log_prior
                or head.adaptive_correction_gate):
            raise ValueError('Candidate tracing expects original E0 channel_gate heads')
    layers, zero_deltas, handles = {}, {}, []

    def replay(index, head, module, args, result):
        inputs = args[0].detach().clone()
        inputs[..., :head.hidden_dim] = 0
        # Bypass this Sequential's hooks to avoid recursive replay. The normal
        # score output and its input tensor remain untouched.
        zero_deltas[index] = module.forward(inputs).squeeze(-1).detach()

    def observe(index, head, args, result):
        if index in layers:
            raise RuntimeError(f'Strip layer {index} ran more than once')
        prior = args[4]['pred_corners'].detach().float().reshape_as(result['calibrated_corners'])
        finite = torch.isfinite(prior).flatten(2).all(-1)
        safe_prior = torch.where(finite[..., None, None], prior, torch.zeros_like(prior))
        delta = zero_deltas.pop(index, torch.zeros_like(prior)).reshape_as(prior)
        correction_valid = result['candidate_valid'] & result['input_valid'][..., None, None]
        zero_logits = safe_prior + torch.where(correction_valid, delta, torch.zeros_like(delta))
        layers[index] = {
            'prior_logits': prior.clone(),
            'calibrated_logits': result['calibrated_corners'].detach().float().clone(),
            'zero_visual_logits': zero_logits.clone(),
            'candidate_positions': result['candidate_positions'].detach().float().clone(),
            'candidate_valid': result['candidate_valid'].detach().clone(),
            'input_valid': result['input_valid'].detach().clone(),
            'refinement_mask': result['refinement_mask'].detach().clone(),
            'boxes_before': args[2].detach().float().clone(),
            'boxes_after': result['pred_boxes'].detach().float().clone(),
        }

    try:
        for index, head in enumerate(model.refiner.heads):
            handles.append(head.score_head.register_forward_hook(
                lambda module, args, result, i=index, h=head: replay(i, h, module, args, result)
            ))
            handles.append(head.register_forward_hook(
                lambda module, args, result, i=index: observe(i, module, args, result)
            ))
        outputs = model(images)
    finally:
        for handle in handles:
            handle.remove()
    expected = list(range(model.decoder.eval_idx + 1))
    if list(layers) != expected:
        raise RuntimeError(f'Expected layers {expected}, got {list(layers)}')
    return {'outputs': outputs, 'layers': layers}


class CandidateSelectionMetrics:
    """Accumulate means over valid corrected edges, with explicit exclusions.

Targets use normalized cxcywh boxes; image_size is input (H,W). Every stage
uses identical eligible edges. Empty means are None, and no samples are cached.
"""

    GROUP_NAMES = ('all', 'short_lt8', 'short_8_to_lt16', 'short_16_to_lt32', 'short_ge32')
    MODES = {'prior': 'prior_logits', 'post': 'calibrated_logits',
             'zero_visual': 'zero_visual_logits'}
    EXCLUSIONS = ('invalid_geometry', 'invalid_input', 'invalid_refinement',
                  'nonfinite_positions', 'nonfinite_logits', 'no_valid_candidate')
    MODE_METRICS = ('mean_nearest_valid_probability', 'mean_nearest_valid_rank',
                    'top1_nearest_match_rate', 'mean_within1px_probability',
                    'mean_invalid_candidate_probability', 'mean_expected_edge_abs_error_px',
                    'mean_argmax_valid_edge_abs_error_px')

    def __init__(self):
        self._groups = {name: {
            'total_pairs': 0, 'pairs': 0, 'excluded_pairs': 0,
            'partially_excluded_pairs': 0, 'total_edges': 0, 'edges': 0,
            'excluded_edges': 0,
            'excluded_edges_by_reason': {reason: 0 for reason in self.EXCLUSIONS},
            'mean_nearest_valid_error_px': 0., 'within1px_coverage': 0.,
            **{mode: dict.fromkeys(self.MODE_METRICS, 0.) for mode in self.MODES},
        } for name in self.GROUP_NAMES}

    @staticmethod
    def _xyxy(boxes):
        return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                          boxes[..., :2] + boxes[..., 2:] / 2), -1)

    @torch.no_grad()
    def update(self, layer, targets, indices, image_size):
        height, width = map(float, image_size)
        if not all(math.isfinite(x) and x > 0 for x in (height, width)):
            raise ValueError('image_size must be positive finite (height, width)')
        shape = layer['candidate_positions'].shape
        if len(shape) != 4 or shape[-2] != 4 or shape[-1] < 1:
            raise ValueError('Candidate positions must have shape [B,Q,4,K]')
        if len(targets) != shape[0] or len(indices) != shape[0]:
            raise ValueError('targets and indices must have one entry per image')
        for key in (*self.MODES.values(), 'candidate_valid'):
            if layer[key].shape != shape:
                raise ValueError(f'{key} must have the same shape as candidate_positions')
        for key in ('input_valid', 'refinement_mask'):
            if layer[key].shape != shape[:2]:
                raise ValueError(f'{key} must have shape [B,Q]')
        for key in ('boxes_before', 'boxes_after'):
            if layer[key].shape != (*shape[:2], 4):
                raise ValueError(f'{key} must have shape [B,Q,4]')

        for batch, (query_ids, target_ids) in enumerate(indices):
            if query_ids.ndim != 1 or target_ids.shape != query_ids.shape:
                raise ValueError('Matched indices must be equal-length vectors')
            if not query_ids.numel():
                continue

            def selected(key, dtype=torch.float64):
                value = layer[key]
                return value[batch, query_ids.to(value.device)].detach().to(device='cpu', dtype=dtype)

            target_boxes = targets[batch]['boxes']
            gt = target_boxes[target_ids.to(target_boxes.device)].detach().to(device='cpu', dtype=torch.float64)
            gt_valid = torch.isfinite(gt).all(-1) & (gt[:, 2:] > 0).all(-1)
            geometry_valid = gt_valid.clone()
            for key in ('boxes_before', 'boxes_after'):
                boxes = selected(key)
                geometry_valid &= torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 0).all(-1)
            positions = selected('candidate_positions')
            candidate_valid = selected('candidate_valid', torch.bool)
            logits = {mode: selected(key) for mode, key in self.MODES.items()}
            failures = {
                'invalid_geometry': ~geometry_valid[:, None].expand(-1, 4),
                'invalid_input': ~selected('input_valid', torch.bool)[:, None].expand(-1, 4),
                'invalid_refinement': ~selected('refinement_mask', torch.bool)[:, None].expand(-1, 4),
                'nonfinite_positions': ~torch.isfinite(positions).all(-1),
                'nonfinite_logits': ~torch.stack([torch.isfinite(v).all(-1) for v in logits.values()]).all(0),
                'no_valid_candidate': ~candidate_valid.any(-1),
            }
            usable = torch.ones((len(gt), 4), dtype=torch.bool)
            excluded = {}
            for reason, failure in failures.items():
                excluded[reason] = usable & failure
                usable &= ~failure
            short = torch.minimum(gt[:, 2] * width, gt[:, 3] * height)
            selections = {
                'all': torch.ones(len(gt), dtype=torch.bool),
                'short_lt8': gt_valid & (short < 8),
                'short_8_to_lt16': gt_valid & (short >= 8) & (short < 16),
                'short_16_to_lt32': gt_valid & (short >= 16) & (short < 32),
                'short_ge32': gt_valid & (short >= 32),
            }
            scale = gt.new_tensor((width, height, width, height)).expand(len(gt), -1)[usable]
            truth = self._xyxy(gt)[usable]
            pos, valid = positions[usable], candidate_valid[usable]
            errors = (pos - truth[:, None]).abs() * scale[:, None]
            nearest_error, nearest_index = errors.masked_fill(~valid, float('inf')).min(-1)
            # Numerical tolerance only addresses floating-point threshold roundoff.
            within = valid & (errors <= 1. + 1e-7)
            mode_values = {}
            for mode, value in logits.items():
                scores = value[usable]
                probabilities = scores.softmax(-1)
                nearest_prob = probabilities.gather(-1, nearest_index[:, None]).squeeze(-1)
                nearest_score = scores.gather(-1, nearest_index[:, None])
                top = scores.argmax(-1)
                valid_top = scores.masked_fill(~valid, float('-inf')).argmax(-1)
                mode_values[mode] = {
                    'mean_nearest_valid_probability': nearest_prob,
                    'mean_nearest_valid_rank': 1 + (scores > nearest_score).sum(-1),
                    'top1_nearest_match_rate': valid.gather(-1, top[:, None]).squeeze(-1) & (
                        errors.gather(-1, top[:, None]).squeeze(-1) <= nearest_error + 1e-7),
                    'mean_within1px_probability': (probabilities * within).sum(-1),
                    'mean_invalid_candidate_probability': (probabilities * ~valid).sum(-1),
                    'mean_expected_edge_abs_error_px': ((probabilities * pos).sum(-1) - truth).abs() * scale,
                    'mean_argmax_valid_edge_abs_error_px': errors.gather(-1, valid_top[:, None]).squeeze(-1),
                }
            for name, selection in selections.items():
                group = self._groups[name]
                count = int(selection.sum())
                has_edge = usable.any(-1)
                edge_selection = selection[:, None].expand_as(usable)
                included = edge_selection[usable]
                edge_count = int(included.sum())
                group['total_pairs'] += count
                group['pairs'] += int((selection & has_edge).sum())
                group['excluded_pairs'] += int((selection & ~has_edge).sum())
                group['partially_excluded_pairs'] += int((selection & has_edge & ~usable.all(-1)).sum())
                group['total_edges'] += 4 * count
                group['edges'] += edge_count
                group['excluded_edges'] += 4 * count - edge_count
                for reason, mask in excluded.items():
                    group['excluded_edges_by_reason'][reason] += int((mask & edge_selection).sum())
                group['mean_nearest_valid_error_px'] += float(nearest_error[included].sum())
                group['within1px_coverage'] += int(within.any(-1)[included].sum())
                for mode, values in mode_values.items():
                    for key, values_by_edge in values.items():
                        group[mode][key] += float(values_by_edge[included].sum())

    def compute(self):
        groups = {}
        for name, totals in self._groups.items():
            count = totals['edges']
            result = {key: value for key, value in totals.items()
                      if key not in self.MODES and key not in ('mean_nearest_valid_error_px', 'within1px_coverage')}
            result['excluded_edges_by_reason'] = dict(totals['excluded_edges_by_reason'])
            for key in ('mean_nearest_valid_error_px', 'within1px_coverage'):
                result[key] = totals[key] / count if count else None
            for mode in self.MODES:
                result[mode] = {key: total / count if count else None for key, total in totals[mode].items()}
            groups[name] = result
        return {
            'groups': groups,
            'definitions': {
                'assignment': 'Caller-fixed final-output geometric Hungarian query/GT pairs; no score filtering.',
                'eligibility': 'Finite geometry, input_valid and refinement_mask, finite positions/logits in all modes, at least one valid candidate per edge.',
                'aggregation': 'Means over eligible edges. pairs counts pairs with at least one eligible edge; exclusions use the listed first failing reason.',
                'probability': 'Original softmax over all K bins, including invalid candidates; no valid-bin renormalization.',
                'expected_edge_error': 'Absolute error of sum_k softmax(logits)[k]*position[k], in input pixels.',
                'nearest': 'Nearest valid candidate; first index breaks distance ties for probability/rank, while top1 accepts any equally near valid bin.',
                'rank': 'One plus the number of all-K logits strictly larger than the nearest valid candidate logit.',
                'zero_visual': 'Local score-head replay at unchanged query/offset/prior/geometry; no downstream trajectory replay.',
                'interpretation': 'Low nearest-bin probability alone is not evidence of an error: neighboring bins can interpolate continuous targets.',
                'empty': 'Empty means are null. Invalid GT geometry is counted in all but excluded from short-side groups.',
            },
        }
