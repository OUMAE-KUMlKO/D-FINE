"""Box-only supervision for the fixed matches of a GT-only diagnostic."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .box_ops import box_cxcywh_to_xyxy


def aligned_generalized_box_iou(boxes1, boxes2):
    """Linear-memory GIoU for aligned, valid xyxy box pairs."""
    lower = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    upper = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (upper - lower).clamp(min=0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).prod(-1)
    union = area1 + area2 - intersection
    enclosing = (
        torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
        - torch.minimum(boxes1[:, :2], boxes2[:, :2])
    ).prod(-1)
    eps = torch.finfo(boxes1.dtype).tiny
    return intersection / union.clamp(min=eps) - (
        enclosing - union
    ) / enclosing.clamp(min=eps)


@register()
class GTRefinementCriterion(nn.Module):
    """Use the model's fixed valid pairs, without rematching or classification loss."""

    def __init__(self, weight_dict=None):
        super().__init__()
        self.weight_dict = {"loss_bbox": 5.0, "loss_giou": 2.0}
        if weight_dict is not None:
            if set(weight_dict) - set(self.weight_dict):
                raise ValueError("Stage one supports only loss_bbox and loss_giou")
            self.weight_dict.update(weight_dict)

    def forward(self, outputs, targets=None, **kwargs):
        indices = outputs["refinement_indices"]
        canonical_targets = outputs["refinement_targets"]
        prediction = outputs["pred_boxes"].float()
        batch_ids = torch.cat([
            torch.full_like(src, batch_index)
            for batch_index, (src, _) in enumerate(indices)
        ])
        query_ids = torch.cat([src for src, _ in indices])
        count = prediction.new_tensor([len(query_ids)])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count)
            count /= dist.get_world_size()
        normalizer = count.clamp(min=1).squeeze(0)
        # Keep every added parameter connected on empty/constant-feature batches.
        zero = outputs["refinement_zero_loss"]
        if not len(query_ids):
            return {name: zero * weight for name, weight in self.weight_dict.items()}
        source = prediction[batch_ids, query_ids]
        target = torch.cat([
            item["boxes"][dst]
            for item, (_, dst) in zip(canonical_targets, indices)
        ]).float()
        loss_bbox = F.l1_loss(source, target, reduction="sum") / normalizer
        loss_giou = (1 - aligned_generalized_box_iou(
            box_cxcywh_to_xyxy(source), box_cxcywh_to_xyxy(target)
        )).sum() / normalizer
        return {
            "loss_bbox": self.weight_dict["loss_bbox"] * (loss_bbox + zero),
            "loss_giou": self.weight_dict["loss_giou"] * (loss_giou + zero),
        }
