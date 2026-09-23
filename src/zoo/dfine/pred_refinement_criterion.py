"""Box supervision for prediction-guided strip refinement."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .box_ops import box_cxcywh_to_xyxy
from .gt_refinement_criterion import aligned_generalized_box_iou


@register()
class PredRefinementCriterion(nn.Module):
    """Match frozen baseline predictions, then supervise accepted refinements."""

    __inject__ = ["matcher"]

    def __init__(self, matcher, weight_dict=None):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = {"loss_bbox": 5.0, "loss_giou": 2.0}
        if weight_dict is not None:
            if set(weight_dict) - set(self.weight_dict):
                raise ValueError("Pred refinement supports only loss_bbox and loss_giou")
            self.weight_dict.update(weight_dict)

    @staticmethod
    def _prepare_targets(targets, device, num_classes):
        prepared = []
        for target in targets:
            boxes = torch.as_tensor(target["boxes"], device=device).float().reshape(-1, 4)
            boxes = boxes.as_subclass(torch.Tensor).detach()
            labels = torch.as_tensor(
                target["labels"], device=device, dtype=torch.long
            ).reshape(-1)
            if len(boxes) != len(labels):
                raise ValueError("Each target must have the same number of boxes and labels")
            keep = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 0).all(-1)
            for flag in ("iscrowd", "ignore"):
                if flag in target:
                    ignored = torch.as_tensor(target[flag], device=device).reshape(-1)
                    if len(ignored) != len(boxes):
                        raise ValueError(f"{flag} must have one value per target box")
                    keep &= ~ignored.bool()
            boxes, labels = boxes[keep], labels[keep]
            if ((labels < 0) | (labels >= num_classes)).any():
                raise ValueError("GT labels must use the baseline's contiguous class indices")
            prepared.append({"boxes": boxes, "labels": labels})
        return prepared

    @torch.no_grad()
    def _match_baseline(self, boxes, logits, targets):
        indices = []
        valid = (
            torch.isfinite(boxes).all(-1)
            & (boxes[..., 2:] > 0).all(-1)
            & torch.isfinite(logits).all(-1)
        )
        for batch_index, target in enumerate(targets):
            query_ids = torch.where(valid[batch_index])[0]
            if not len(query_ids) or not len(target["boxes"]):
                empty = torch.empty(0, dtype=torch.long, device=boxes.device)
                indices.append((empty, empty))
                continue
            matches = self.matcher(
                {
                    "pred_boxes": boxes[batch_index, query_ids].float().unsqueeze(0),
                    "pred_logits": logits[batch_index, query_ids].float().unsqueeze(0),
                },
                [target],
            )["indices"][0]
            source, destination = (value.to(boxes.device) for value in matches)
            indices.append((query_ids[source], destination))
        return indices

    def forward(self, outputs, targets, **kwargs):
        prediction = outputs["pred_boxes"].float()
        baseline = outputs["baseline_boxes"]
        logits = outputs["pred_logits"]
        if len(targets) != len(prediction):
            raise ValueError("Expected one target dictionary per batch item")
        canonical_targets = self._prepare_targets(
            targets, prediction.device, logits.shape[-1]
        )
        indices = self._match_baseline(baseline, logits, canonical_targets)

        selected_batch = []
        selected_query = []
        selected_target = []
        mask = outputs["refinement_mask"]
        for batch_index, ((source, destination), target) in enumerate(
            zip(indices, canonical_targets)
        ):
            accepted = mask[batch_index, source].bool()
            source = source[accepted]
            destination = destination[accepted]
            selected_batch.append(torch.full_like(source, batch_index))
            selected_query.append(source)
            selected_target.append(target["boxes"][destination])

        batch_ids = torch.cat(selected_batch)
        query_ids = torch.cat(selected_query)
        count = prediction.new_tensor([len(query_ids)])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count)
            count /= dist.get_world_size()
        normalizer = count.clamp(min=1).squeeze(0)
        zero = outputs["refinement_zero_loss"]
        if not len(query_ids):
            return {name: zero * weight for name, weight in self.weight_dict.items()}

        source = prediction[batch_ids, query_ids]
        target = torch.cat(selected_target).float()
        loss_bbox = F.l1_loss(source, target, reduction="sum") / normalizer
        loss_giou = (1 - aligned_generalized_box_iou(
            box_cxcywh_to_xyxy(source), box_cxcywh_to_xyxy(target)
        )).sum() / normalizer
        return {
            "loss_bbox": self.weight_dict["loss_bbox"] * (loss_bbox + zero),
            "loss_giou": self.weight_dict["loss_giou"] * (loss_giou + zero),
        }
