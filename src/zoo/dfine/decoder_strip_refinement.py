"""Inject prediction-guided boundary evidence into frozen decoder queries."""

import torch

from ...core import register
from .decoder_strip_adapter import DistributionStripQueryAdapter
from .pred_refinement import DFINEPredRefiner


@register()
class DFINEDecoderStripRefiner(DFINEPredRefiner):
    """Train a shared query adapter while keeping detector weights and scores fixed."""

    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(
        self, backbone, encoder, decoder, feature_level=0, hidden_dim=64,
        num_points=5, kernel_size=3, offset_pixels=1.0, query_chunk_size=32,
        refine_layers=None, baseline_checkpoint=None,
    ):
        if refine_layers is None:
            refine_layers = list(range(max(1, decoder.eval_idx - 1), decoder.eval_idx + 1))
        layers = tuple(refine_layers)
        if (not layers or any(type(i) is not int or i < 1 or i > decoder.eval_idx for i in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError('refine_layers must select unique executed decoder layers after layer 0')
        super().__init__(
            backbone, encoder, decoder, feature_level=feature_level,
            hidden_dim=hidden_dim, num_points=num_points, kernel_size=kernel_size,
            offset_pixels=offset_pixels, sampling_mode='distribution',
            query_chunk_size=query_chunk_size, baseline_checkpoint=baseline_checkpoint,
        )
        self.refine_layers = tuple(sorted(layers))

    def _build_refiner(self, **kwargs):
        # Reuse the prediction wrapper's checkpoint/freeze lifecycle with a query head.
        kwargs.pop('max_log_scale')
        kwargs.pop('sampling_mode')
        return DistributionStripQueryAdapter(**kwargs)

    def forward(self, images, targets=None):
        # GT is consumed only by the separate criterion, never by decoder feedback.
        if not self._baseline_loaded:
            raise RuntimeError('Load a complete baseline checkpoint before decoder strip refinement')
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError('Expected a nonempty BCHW image batch')
        with torch.no_grad():
            features = self.encoder(self.backbone(images))
            if self.feature_level >= len(features):
                raise ValueError('feature_level is outside the Encoder output list')
            baseline = self.decoder(features)
            baseline_boxes = baseline['pred_boxes'].detach().float()
            baseline_logits = baseline['pred_logits'].detach()

        injected = torch.zeros(baseline_boxes.shape[:2], dtype=torch.bool, device=images.device)

        def refine_query(layer_index, queries, previous_boxes, distribution):
            nonlocal injected
            if layer_index not in self.refine_layers:
                return queries
            delta, valid = self.refiner(
                features[self.feature_level], queries, previous_boxes,
                images.shape[-2:], distribution,
            )
            injected = injected | valid
            return queries + delta.to(queries.dtype)

        # Frozen parameters still propagate gradients from bbox outputs to the adapter.
        refined = self.decoder(features, query_refiner=refine_query)
        boxes = refined['pred_boxes'].float()
        valid = (
            injected & torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 0).all(-1)
            & torch.isfinite(baseline_boxes).all(-1) & (baseline_boxes[..., 2:] > 0).all(-1)
            & torch.isfinite(baseline_logits).all(-1)
        )
        return {
            'pred_logits': baseline_logits,
            'pred_boxes': torch.where(valid[..., None], boxes, baseline_boxes),
            'baseline_boxes': baseline_boxes,
            'refinement_mask': valid,
            'refinement_zero_loss': self.refiner.zero_loss(),
        }
