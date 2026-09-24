"""Calibrate final boundary distributions while preserving the frozen detector."""

import torch

from ...core import register
from .pred_refinement import DFINEPredRefiner
from .strip_probability_head import StripProbabilityHead


@register()
class DFINEProbabilityRefiner(DFINEPredRefiner):
    """Train the strip probability head after one GT-free decoder forward."""

    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(
        self, backbone, encoder, decoder, feature_level=0, hidden_dim=64,
        num_points=5, kernel_size=3, offset_pixels=1.0, query_chunk_size=32,
        baseline_checkpoint=None,
    ):
        super().__init__(
            backbone, encoder, decoder, feature_level=feature_level,
            hidden_dim=hidden_dim, num_points=num_points, kernel_size=kernel_size,
            offset_pixels=offset_pixels, sampling_mode='distribution',
            query_chunk_size=query_chunk_size, baseline_checkpoint=baseline_checkpoint,
        )

    def _build_refiner(self, **kwargs):
        kwargs.pop('max_log_scale')
        kwargs.pop('sampling_mode')
        return StripProbabilityHead(**kwargs)

    def forward(self, images, targets=None):
        # Targets are consumed exclusively by ProbabilityRefinementCriterion.
        if not self._baseline_loaded:
            raise RuntimeError('Load a complete baseline checkpoint before probability refinement')
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError('Expected a nonempty BCHW image batch')
        with torch.no_grad():
            features = self.encoder(self.backbone(images))
            if self.feature_level >= len(features):
                raise ValueError('feature_level is outside the Encoder output list')
            baseline = self.decoder(features, return_query=True, return_distribution=True)
            baseline_boxes = baseline['pred_boxes'].detach().float()
            baseline_logits = baseline['pred_logits'].detach()
            distribution = {
                name: baseline[name].detach()
                for name in ('pred_corners', 'ref_points', 'up', 'reg_scale')
            }

        outputs = self.refiner(
            features[self.feature_level].detach(), baseline['pred_queries'].detach(),
            baseline_boxes, images.shape[-2:], distribution,
        )
        score_valid = torch.isfinite(baseline_logits).all(-1)
        outputs['input_valid'] = outputs['input_valid'] & score_valid
        valid = outputs['refinement_mask'] & score_valid
        outputs['pred_boxes'] = torch.where(
            valid[..., None], outputs['pred_boxes'], baseline_boxes,
        )
        outputs.update({
            'pred_logits': baseline_logits,
            'baseline_boxes': baseline_boxes,
            'refinement_mask': valid,
            'refinement_zero_loss': self.refiner.zero_loss(),
        })
        # calibrated_corners stays attached: the candidate loss trains this branch.
        return outputs
