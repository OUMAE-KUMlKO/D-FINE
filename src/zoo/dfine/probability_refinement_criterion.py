"""Fixed-match box and candidate-distribution supervision for P3."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ...core import register
from .box_ops import box_cxcywh_to_xyxy
from .gt_refinement_criterion import aligned_generalized_box_iou
from .pred_refinement_criterion import PredRefinementCriterion


@register()
class ProbabilityRefinementCriterion(PredRefinementCriterion):
    """Supervise calibrated edge distributions on fixed baseline matches."""

    def __init__(self, matcher, weight_dict=None):
        super().__init__(matcher)
        self.weight_dict = {
            "loss_bbox": 5.0, "loss_giou": 2.0, "loss_candidate": 0.05,
        }
        if weight_dict is not None:
            if set(weight_dict) - set(self.weight_dict):
                raise ValueError(
                    "Probability refinement supports only loss_bbox, loss_giou, "
                    "and loss_candidate"
                )
            self.weight_dict.update(weight_dict)
        self.last_candidate_stats = {
            "candidate_loss_unweighted": 0.0,
            "candidate_loss_weighted": 0.0,
            "candidate_edges": 0.0,
            "candidate_edge_fraction": 0.0,
        }

    @staticmethod
    def _gt_edges(boxes):
        x, y, width, height = boxes.unbind(-1)
        return torch.stack((
            x - width / 2, y - height / 2,
            x + width / 2, y + height / 2,
        ), dim=-1)

    @staticmethod
    def _distributed_sum(value):
        result = value.detach().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(result)
        return result

    def forward(self, outputs, targets, **kwargs):
        prediction = outputs["pred_boxes"].float()
        baseline = outputs["baseline_boxes"]
        class_logits = outputs["pred_logits"]
        if len(targets) != len(prediction):
            raise ValueError("Expected one target dictionary per batch item")
        canonical_targets = self._prepare_targets(
            targets, prediction.device, class_logits.shape[-1]
        )
        indices = self._match_baseline(baseline, class_logits, canonical_targets)
        zero = outputs["refinement_zero_loss"]

        box_batch, box_query, box_targets = [], [], []
        mask = outputs["refinement_mask"]
        for batch_index, ((source, destination), target) in enumerate(
            zip(indices, canonical_targets)
        ):
            accepted = mask[batch_index, source].bool()
            box_batch.append(torch.full_like(source[accepted], batch_index))
            box_query.append(source[accepted])
            box_targets.append(target["boxes"][destination[accepted]])
        batch_ids, query_ids = torch.cat(box_batch), torch.cat(box_query)
        box_count = prediction.new_tensor([len(query_ids)])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(box_count)
            box_count /= dist.get_world_size()
        box_normalizer = box_count.clamp(min=1).squeeze(0)
        if len(query_ids):
            source_boxes = prediction[batch_ids, query_ids]
            target_boxes = torch.cat(box_targets).float()
            bbox = F.l1_loss(source_boxes, target_boxes, reduction="sum") / box_normalizer
            giou = (1 - aligned_generalized_box_iou(
                box_cxcywh_to_xyxy(source_boxes), box_cxcywh_to_xyxy(target_boxes)
            )).sum() / box_normalizer
        else:
            bbox = giou = zero

        calibrated = outputs["calibrated_corners"].float()
        positions = outputs["candidate_positions"].detach().to(calibrated.device)
        candidate_valid = outputs["candidate_valid"].detach().to(
            device=calibrated.device, dtype=torch.bool
        )
        input_valid = outputs["input_valid"].detach().to(
            device=calibrated.device, dtype=torch.bool
        )
        if calibrated.shape != positions.shape or calibrated.shape != candidate_valid.shape:
            raise ValueError("calibrated logits, candidate positions, and validity must share [B,Q,4,K]")
        if input_valid.shape != calibrated.shape[:2]:
            raise ValueError("input_valid must have shape [B,Q]")

        matched_batch = torch.cat([
            torch.full_like(source, batch_index)
            for batch_index, (source, _) in enumerate(indices)
        ])
        matched_query = torch.cat([source for source, _ in indices])
        matched_targets = torch.cat([
            target["boxes"][destination]
            for target, (_, destination) in zip(canonical_targets, indices)
        ])
        local_matched_edges = matched_query.numel() * 4
        if matched_query.numel():
            matched_logits = calibrated[matched_batch, matched_query]
            matched_positions = positions[matched_batch, matched_query]
            matched_valid = candidate_valid[matched_batch, matched_query]
            matched_input_valid = input_valid[matched_batch, matched_query]
            target_edges = self._gt_edges(matched_targets).to(matched_positions.dtype)

            sorted_positions, sorted_to_original = matched_positions.sort(dim=-1)
            finite = torch.isfinite(sorted_positions).all(-1) & torch.isfinite(target_edges)
            inside = (
                (target_edges >= sorted_positions[..., 0])
                & (target_edges <= sorted_positions[..., -1])
            )
            exact_mask = torch.isclose(
                sorted_positions, target_edges[..., None], rtol=1e-6, atol=1e-7
            )
            has_exact = exact_mask.any(-1)
            exact_sorted = exact_mask.to(torch.long).argmax(-1)
            upper = torch.searchsorted(
                sorted_positions, target_edges[..., None]
            ).squeeze(-1).clamp(1, calibrated.shape[-1] - 1)
            lower = upper - 1
            lower_coordinate = sorted_positions.gather(-1, lower[..., None]).squeeze(-1)
            upper_coordinate = sorted_positions.gather(-1, upper[..., None]).squeeze(-1)
            interval = upper_coordinate - lower_coordinate
            upper_weight = torch.where(
                interval > 0, (target_edges - lower_coordinate) / interval,
                torch.zeros_like(interval),
            )
            lower_weight = 1 - upper_weight
            lower_original = sorted_to_original.gather(-1, lower[..., None]).squeeze(-1)
            upper_original = sorted_to_original.gather(-1, upper[..., None]).squeeze(-1)
            exact_original = sorted_to_original.gather(
                -1, exact_sorted[..., None]
            ).squeeze(-1)
            first_index = torch.where(has_exact, exact_original, lower_original)
            second_index = torch.where(has_exact, exact_original, upper_original)
            first_weight = torch.where(
                has_exact, torch.ones_like(lower_weight), lower_weight
            )
            second_weight = torch.where(
                has_exact, torch.zeros_like(upper_weight), upper_weight
            )
            first_valid = matched_valid.gather(-1, first_index[..., None]).squeeze(-1)
            second_valid = matched_valid.gather(-1, second_index[..., None]).squeeze(-1)
            supervised = (
                matched_input_valid[:, None] & finite & inside & first_valid
                & ((second_weight == 0) | second_valid)
                & (has_exact | (interval > 0))
            )
            flat_logits = matched_logits.reshape(-1, matched_logits.shape[-1])
            flat_supervised = supervised.reshape(-1)
            flat_first = first_index.reshape(-1)[flat_supervised]
            flat_second = second_index.reshape(-1)[flat_supervised]
            flat_first_weight = first_weight.reshape(-1)[flat_supervised]
            flat_second_weight = second_weight.reshape(-1)[flat_supervised]
            log_probability = flat_logits[flat_supervised].log_softmax(-1)
            candidate_sum = -(
                log_probability.gather(-1, flat_first[:, None]).squeeze(-1)
                * flat_first_weight
                + log_probability.gather(-1, flat_second[:, None]).squeeze(-1)
                * flat_second_weight
            ).sum()
            local_edges = flat_supervised.sum().detach().to(calibrated.dtype)
        else:
            candidate_sum = calibrated.sum() * 0.0
            local_edges = calibrated.new_zeros(())

        stat_values = torch.stack((
            candidate_sum.detach(), local_edges,
            calibrated.new_tensor(float(local_matched_edges)),
        ))
        stat_values = self._distributed_sum(stat_values)
        global_loss_sum, global_edges, global_matched_edges = stat_values
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        candidate_normalizer = global_edges.clamp(min=1) / world_size
        candidate_loss = candidate_sum / candidate_normalizer

        global_loss_sum, stats_edges, matched_edges = stat_values.tolist()
        unweighted = global_loss_sum / stats_edges if stats_edges else 0.0
        self.last_candidate_stats = {
            "candidate_loss_unweighted": unweighted,
            "candidate_loss_weighted": unweighted * self.weight_dict["loss_candidate"],
            "candidate_edges": stats_edges,
            "candidate_edge_fraction": stats_edges / matched_edges if matched_edges else 0.0,
        }
        return {
            "loss_bbox": self.weight_dict["loss_bbox"] * (bbox + zero),
            "loss_giou": self.weight_dict["loss_giou"] * (giou + zero),
            "loss_candidate": self.weight_dict["loss_candidate"] * (
                candidate_loss + zero
            ),
        }
