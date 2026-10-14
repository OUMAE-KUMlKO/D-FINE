"""G5 oracle that reveals only the GT-nearest candidate's visual features."""

import torch

from ...core import register
from .all_layer_strip import DFINEAllLayerStrip
from .gt_refinement import DFINEGTRefiner


@register()
class DFINEGTAllLayerStrip(DFINEAllLayerStrip):
    """Retain G5's six-layer scoring/decoding and native joint-training losses.

    Matching uses each layer's pre-correction predictions solely to select
    visual inputs. The criterion retains its independent native matching.
    Unlike final-output oracle wrappers, this model has no fixed baseline pair
    geometry, and classification scores may change through the decoder/LQE.
    """

    __inject__ = ['backbone', 'encoder', 'decoder', 'matcher']
    requires_gt = True
    oracle_evaluation = True
    supports_gt_refinement_metrics = False

    def __init__(
        self, backbone, encoder, decoder, matcher, feature_level=0, hidden_dim=64,
        num_points=5, kernel_size=3, offset_pixels=1.0, query_chunk_size=32,
        checkpoint_strips=True, normal_points=3, normal_kernel_size=1,
        use_log_prior=True, query_fusion='concat', bandwidth_mode='fixed',
        bandwidth_ratio=0.25, min_offset_pixels=2.0, num_refine_layers=None,
        adaptive_correction_gate=False, next_reference_source='corrected',
        train_target_format='cxcywh_normalized', eval_target_format='xyxy_absolute',
    ):
        super().__init__(
            backbone, encoder, decoder, feature_level=feature_level,
            hidden_dim=hidden_dim, num_points=num_points, kernel_size=kernel_size,
            offset_pixels=offset_pixels, query_chunk_size=query_chunk_size,
            checkpoint_strips=checkpoint_strips, normal_points=normal_points,
            normal_kernel_size=normal_kernel_size, use_log_prior=use_log_prior,
            query_fusion=query_fusion, bandwidth_mode=bandwidth_mode,
            bandwidth_ratio=bandwidth_ratio, min_offset_pixels=min_offset_pixels,
            num_refine_layers=num_refine_layers,
            adaptive_correction_gate=adaptive_correction_gate,
            next_reference_source=next_reference_source,
        )
        self.matcher = matcher
        self.train_target_format = train_target_format
        self.eval_target_format = eval_target_format
        self.last_gt_guidance_stats = {}

    @torch.no_grad()
    def _guide_layer(self, layer_index, queries, boxes, distribution, targets):
        dn_count = distribution['num_denoising']
        boxes = boxes[:, dn_count:].detach().float()
        with torch.autocast(device_type=queries.device.type, enabled=False):
            raw_scores = self.decoder.dec_score_head[layer_index](
                queries[:, dn_count:].detach().float())
            prior_scores = self.decoder.decoder.lqe_layers[layer_index](
                raw_scores, distribution['pred_corners'][:, dn_count:].detach().float())
            indices, _ = DFINEGTRefiner._match_baseline(self, boxes, prior_scores, targets)
        aligned = torch.zeros_like(boxes)
        matched = torch.zeros(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
        for batch_index, (src, dst) in enumerate(indices):
            aligned[batch_index, src] = targets[batch_index]['boxes'][dst]
            matched[batch_index, src] = True
        self.last_gt_guidance_stats[f'layer_{layer_index}_matched_queries'] = int(matched.sum())
        return aligned, matched

    def forward(self, images, targets=None, target_format=None):
        if targets is None:
            raise ValueError('DFINEGTAllLayerStrip requires GT targets for training and evaluation')
        if images.ndim != 4 or not len(images) or len(targets) != len(images):
            raise ValueError('Expected a nonempty BCHW batch and one target per image')
        target_format = target_format or (
            self.train_target_format if self.training else self.eval_target_format)
        guidance_targets = targets
        if self.training:
            guidance_targets = []
            for target in targets:
                crowd = target.get('iscrowd')
                if crowd is not None:
                    crowd = torch.as_tensor(crowd).reshape(-1)
                    if len(crowd) != len(target['boxes']) and not bool(crowd.any()):
                        # The dataset removes crowds before augmentation, but
                        # box sanitization can leave the all-zero flags stale.
                        # Only canonical guidance omits this redundant metadata;
                        # decoder and criterion still receive the original target.
                        target = {key: value for key, value in target.items() if key != 'iscrowd'}
                guidance_targets.append(target)
        canonical, _ = DFINEGTRefiner._prepare_targets(
            guidance_targets, images.shape[-2:], images.device, target_format,
            self.decoder.num_classes)
        self.refiner.last_prediction_safety_stats = {}
        self.last_gt_guidance_stats = {}
        features = self.encoder(self.backbone(images))
        if self.feature_level >= len(features):
            raise ValueError('feature_level is outside the Encoder output list')

        def refine(layer_index, queries, boxes, distribution):
            if layer_index >= self.num_refine_layers:
                return distribution['pred_corners']
            gt_boxes, gt_matched = self._guide_layer(
                layer_index, queries, boxes, distribution, canonical)
            return self.refiner(
                layer_index, features[self.feature_level], queries, boxes,
                images.shape[-2:], distribution, self.checkpoint_strips,
                gt_boxes=gt_boxes, gt_matched=gt_matched)

        decoder_kwargs = {'distribution_refiner': refine}
        if self.next_reference_source != 'corrected':
            decoder_kwargs['next_reference_source'] = self.next_reference_source
        return self.decoder(features, targets, **decoder_kwargs)
