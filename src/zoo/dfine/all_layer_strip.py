"""Strip correction inside every decoder layer, with two training regimes."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...core import register
from .dfine import DFINE
from .pred_refinement import DFINEPredRefiner
from .strip_probability_head import StripProbabilityHead


class LayerStripHeads(nn.Module):
    """Independent layer heads; native denoising queries keep their original path."""

    def __init__(self, num_layers, **head_kwargs):
        super().__init__()
        self.heads = nn.ModuleList([
            StripProbabilityHead(**head_kwargs, detach_inputs=False)
            for _ in range(num_layers)
        ])

    def zero_loss(self):
        return sum(head.zero_loss() for head in self.heads)

    def forward(
        self, layer_index, feature, queries, boxes, image_size, distribution,
        checkpoint_strips=False,
    ):
        head = self.heads[layer_index]
        dn_count = distribution['num_denoising']
        corners = distribution['pred_corners']
        refs = distribution['ref_points']

        def calibrate(feature, query, box, prior, reference, up, scale):
            result = head(feature, query, box, image_size, {
                'pred_corners': prior, 'ref_points': reference,
                'up': up, 'reg_scale': scale,
            })
            calibrated = result['calibrated_corners'].flatten(-2).to(prior.dtype)
            # Keep logits and decoded boxes consistent when calibration falls back.
            return torch.where(result['refinement_mask'][..., None], calibrated, prior)

        args = (
            feature, queries[:, dn_count:], boxes[:, dn_count:],
            corners[:, dn_count:], refs[:, dn_count:],
            distribution['up'], distribution['reg_scale'],
        )
        if checkpoint_strips and self.training and torch.is_grad_enabled():
            corrected = checkpoint(calibrate, *args, use_reentrant=False)
        else:
            corrected = calibrate(*args)
        return torch.cat((corners[:, :dn_count], corrected), dim=1)


@register()
class DFINEAllLayerStrip(DFINE):
    """Jointly train D-FINE and strip heads with native per-layer/DN losses."""

    __inject__ = ['backbone', 'encoder', 'decoder']
    requires_gt = False
    oracle_evaluation = False

    def __init__(
        self, backbone, encoder, decoder, feature_level=0, hidden_dim=64,
        num_points=5, kernel_size=3, offset_pixels=1.0, query_chunk_size=32,
        checkpoint_strips=True, normal_points=3, normal_kernel_size=1,
    ):
        super().__init__(backbone, encoder, decoder)
        if feature_level < 0:
            raise ValueError('feature_level must be nonnegative')
        self.feature_level = feature_level
        self.checkpoint_strips = checkpoint_strips
        self.refiner = LayerStripHeads(
            decoder.num_layers, in_channels=encoder.hidden_dim,
            query_dim=decoder.hidden_dim, hidden_dim=hidden_dim,
            num_points=num_points, kernel_size=kernel_size,
            offset_pixels=offset_pixels, query_chunk_size=query_chunk_size,
            normal_points=normal_points, normal_kernel_size=normal_kernel_size,
        )

    def forward(self, images, targets=None):
        features = self.encoder(self.backbone(images))
        if self.feature_level >= len(features):
            raise ValueError('feature_level is outside the Encoder output list')

        def refine(layer_index, queries, boxes, distribution):
            return self.refiner(
                layer_index, features[self.feature_level], queries, boxes,
                images.shape[-2:], distribution, self.checkpoint_strips,
            )

        # Targets enter only the original training denoising mechanism and loss.
        return self.decoder(features, targets, distribution_refiner=refine)


@register()
class DFINEAllLayerStripRefiner(DFINEPredRefiner):
    """Frozen-B0 experiment: train all layer heads through final box supervision."""

    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(
        self, backbone, encoder, decoder, feature_level=0, hidden_dim=64,
        num_points=5, kernel_size=3, offset_pixels=1.0, query_chunk_size=32,
        checkpoint_strips=True, baseline_checkpoint=None,
        normal_points=3, normal_kernel_size=1,
    ):
        # The base constructor invokes _build_refiner before loading B0.
        self.normal_points = normal_points
        self.normal_kernel_size = normal_kernel_size
        super().__init__(
            backbone, encoder, decoder, feature_level=feature_level,
            hidden_dim=hidden_dim, num_points=num_points, kernel_size=kernel_size,
            offset_pixels=offset_pixels, sampling_mode='distribution',
            query_chunk_size=query_chunk_size, baseline_checkpoint=baseline_checkpoint,
        )
        self.checkpoint_strips = checkpoint_strips

    def _build_refiner(self, **kwargs):
        kwargs.pop('max_log_scale')
        kwargs.pop('sampling_mode')
        return LayerStripHeads(
            self.decoder.num_layers, normal_points=self.normal_points,
            normal_kernel_size=self.normal_kernel_size, **kwargs,
        )

    def forward(self, images, targets=None):
        if not self._baseline_loaded:
            raise RuntimeError('Load a complete baseline checkpoint before all-layer refinement')
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError('Expected a nonempty BCHW image batch')
        with torch.no_grad():
            features = self.encoder(self.backbone(images))
            if self.feature_level >= len(features):
                raise ValueError('feature_level is outside the Encoder output list')
            baseline = self.decoder(features)
            baseline_boxes = baseline['pred_boxes'].detach().float()
            baseline_logits = baseline['pred_logits'].detach()

        def refine(layer_index, queries, boxes, distribution):
            return self.refiner(
                layer_index, features[self.feature_level], queries, boxes,
                images.shape[-2:], distribution, self.checkpoint_strips,
            )

        output = self.decoder(features, distribution_refiner=refine)
        boxes = output['pred_boxes'].float()
        valid = (
            torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 0).all(-1)
            & torch.isfinite(baseline_boxes).all(-1) & (baseline_boxes[..., 2:] > 0).all(-1)
            & torch.isfinite(baseline_logits).all(-1)
        )
        return {
            'pred_boxes': torch.where(valid[..., None], boxes, baseline_boxes),
            'pred_logits': baseline_logits,
            'baseline_boxes': baseline_boxes,
            'refinement_mask': valid,
            'refinement_zero_loss': self.refiner.zero_loss(),
        }
