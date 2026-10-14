"""Temporarily change feature pooling in a trained GT-candidate oracle."""

from contextlib import contextmanager
from types import MethodType

import torch


def pooling_indices(mode, logits, valid):
    if mode == 'prior_weighted':
        return None  # Native prediction head masks and renormalizes probabilities.
    if mode != 'prior_argmax':
        raise ValueError(f'Unsupported prior pooling mode: {mode}')
    valid = valid & torch.isfinite(logits).all(-1, keepdim=True)
    selected = logits.masked_fill(~valid, float('-inf')).argmax(-1)
    return torch.where(valid.any(-1), selected, torch.full_like(selected, -1))


@contextmanager
def candidate_pooling(model, mode):
    if mode not in {'gt_nearest', 'prior_argmax', 'prior_weighted'}:
        raise ValueError(f'Unknown candidate pooling mode: {mode}')
    if model.training or type(model).__name__ != 'DFINEGTCandidateRefiner':
        raise ValueError('Pooling diagnostic requires eval-mode DFINEGTCandidateRefiner')
    if mode == 'gt_nearest':
        yield model
        return
    head = model.refiner
    original = head._distribution_forward
    had_override = '_distribution_forward' in head.__dict__

    def forward(self, feature_map, queries, pred_boxes, batch_indices,
                image_size, distribution, return_diagnostics, candidate_indices=None):
        if mode == 'prior_weighted':
            chosen = None
        else:
            positions, _, distribution_valid = self._distribution_geometry(
                distribution, len(pred_boxes), feature_map.device)
            _, _, valid = self._candidate_grid(
                positions, pred_boxes, distribution_valid, image_size)
            logits = distribution['pred_corners'].detach().float().reshape(positions.shape)
            chosen = pooling_indices(mode, logits, valid)
        return original(feature_map, queries, pred_boxes, batch_indices,
                        image_size, distribution, return_diagnostics,
                        candidate_indices=chosen)

    head._distribution_forward = MethodType(forward, head)
    try:
        yield model
    finally:
        if had_override:
            head._distribution_forward = original
        else:
            del head._distribution_forward
