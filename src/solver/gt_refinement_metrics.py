"""Geometry diagnostics for GT-guided refinement with fixed baseline matching.

All pixel errors and object-size groups use the network input coordinate system.
The accumulator contains only additive tensors, so distributed evaluation reduces
the same fixed-size state even on ranks without any matched objects. Padding
from the standard DistributedSampler is excluded before accumulation.
"""

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler


class GTRefinementMetrics:
    GROUP_NAMES = (
        "all", "small", "medium", "large",
        "baseline_iou_lt_0_50", "baseline_iou_0_50_to_0_75", "baseline_iou_ge_0_75",
    )
    COUNT_NAMES = (
        "matched", "refined", "ignored_gt", "total_gt", "unmatched_gt",
        "unmatched_queries", "invalid_gt_samples", "invalid_baseline_boxes",
        "invalid_metric_pairs",
    )
    # count, refined count, 8 baseline errors, 8 refined errors, 2 IoUs,
    # improved/degraded/stable counts.
    GROUP_WIDTH = 23

    def __init__(self, device=None, iou_tolerance=1e-6, sampler=None):
        self.iou_tolerance = iou_tolerance
        # DistributedSampler appends padding before selecting rank-strided
        # indices. Each rank therefore has a prefix of original samples,
        # even with shuffle=True. Do not assume this for custom samplers.
        self._unpadded_samples = (
            len(range(sampler.rank, len(sampler.dataset), sampler.num_replicas))
            if type(sampler) is DistributedSampler else None
        )
        self._seen_samples = 0
        self.values = torch.zeros(
            len(self.GROUP_NAMES) * self.GROUP_WIDTH + len(self.COUNT_NAMES),
            dtype=torch.float64, device=device,
        )

    @property
    def groups(self):
        return self.values[:len(self.GROUP_NAMES) * self.GROUP_WIDTH].view(
            len(self.GROUP_NAMES), self.GROUP_WIDTH
        )

    @property
    def counts(self):
        return self.values[len(self.GROUP_NAMES) * self.GROUP_WIDTH:]

    @staticmethod
    def _xyxy(boxes):
        return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2), dim=-1)

    @staticmethod
    def _paired_iou(boxes, targets):
        intersection = (
            torch.minimum(boxes[:, 2:], targets[:, 2:])
            - torch.maximum(boxes[:, :2], targets[:, :2])
        ).clamp(min=0).prod(dim=-1)
        area = (boxes[:, 2:] - boxes[:, :2]).prod(dim=-1)
        target_area = (targets[:, 2:] - targets[:, :2]).prod(dim=-1)
        return intersection / (area + target_area - intersection).clamp(min=1e-12)

    @torch.no_grad()
    def update(self, outputs, input_size):
        """Accumulate all fixed pairs, including pairs that were left unchanged.

        ``matched_indices`` addresses ``refinement_targets`` (the wrapper's
        canonical, filtered GT), not the original dataset targets. Boxes are
        normalized cxcywh, and ``input_size`` is (height, width).
        """
        before = outputs["baseline_boxes"]
        after = outputs["pred_boxes"]
        if self.values.device != before.device:
            self.values = self.values.to(before.device)
        batch_size = len(before)
        valid_size = batch_size if self._unpadded_samples is None else min(
            batch_size, max(0, self._unpadded_samples - self._seen_samples)
        )
        self._seen_samples += batch_size
        # Keep every model forward intact (including padding-only ranks) so DDP
        # sees equal evaluation steps. Only discard their metric contributions.
        if valid_size == 0:
            return
        per_image_counts = outputs.get("refinement_counts_per_image")
        if valid_size != batch_size and per_image_counts is None:
            raise ValueError("Excluding sampler padding requires refinement_counts_per_image")
        for index, name in enumerate(self.COUNT_NAMES[:-1]):
            if per_image_counts is not None:
                value = per_image_counts[name][:valid_size].sum()
            else:
                value = outputs.get("refinement_counts", {}).get(name, 0)
            self.counts[index] += torch.as_tensor(
                value, device=self.values.device, dtype=self.values.dtype
            )

        height, width = input_size
        scale = self.values.new_tensor((width, height, width, height))
        for batch_index, (query_ids, target_ids) in enumerate(outputs["matched_indices"][:valid_size]):
            if query_ids.numel() == 0:
                continue
            original = before[batch_index, query_ids].to(torch.float64)
            refined = after[batch_index, query_ids].to(torch.float64)
            target = outputs["refinement_targets"][batch_index]["boxes"][target_ids].to(
                device=before.device, dtype=torch.float64
            )
            valid = (
                torch.isfinite(original).all(dim=-1)
                & torch.isfinite(refined).all(dim=-1)
                & torch.isfinite(target).all(dim=-1)
                & (original[:, 2:] > 0).all(dim=-1)
                & (refined[:, 2:] > 0).all(dim=-1)
                & (target[:, 2:] > 0).all(dim=-1)
            )
            self.counts[-1] += (~valid).sum()
            original, refined, target = original[valid], refined[valid], target[valid]
            if original.shape[0] == 0:
                continue
            original_xyxy, refined_xyxy, target_xyxy = (
                self._xyxy(boxes) for boxes in (original, refined, target)
            )
            before_iou = self._paired_iou(original_xyxy, target_xyxy)
            after_iou = self._paired_iou(refined_xyxy, target_xyxy)
            difference = after_iou - before_iou
            errors_before = torch.cat((
                (original_xyxy - target_xyxy).abs() * scale,
                (original - target).abs() * scale,
            ), dim=-1)
            errors_after = torch.cat((
                (refined_xyxy - target_xyxy).abs() * scale,
                (refined - target).abs() * scale,
            ), dim=-1)
            was_refined = outputs["refinement_mask"][batch_index, query_ids][valid]
            rows = torch.cat((
                torch.ones_like(before_iou[:, None]), was_refined[:, None],
                errors_before, errors_after, before_iou[:, None], after_iou[:, None],
                (difference > self.iou_tolerance)[:, None],
                (difference < -self.iou_tolerance)[:, None],
                (difference.abs() <= self.iou_tolerance)[:, None],
            ), dim=-1)
            target_area = target[:, 2] * width * target[:, 3] * height
            selections = (
                torch.ones_like(before_iou, dtype=torch.bool),
                target_area < 32 ** 2,
                (target_area >= 32 ** 2) & (target_area < 96 ** 2),
                target_area >= 96 ** 2,
                before_iou < 0.5,
                (before_iou >= 0.5) & (before_iou < 0.75),
                before_iou >= 0.75,
            )
            for index, selection in enumerate(selections):
                self.groups[index] += rows[selection].sum(dim=0)

    def synchronize_between_processes(self):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.values, op=dist.ReduceOp.SUM)

    def compute(self):
        def geometry(row, start, iou_index):
            count = row[0]
            if count == 0:
                return {
                    "edge_mae_px_ltrb": None, "mean_edge_mae_px": None,
                    "center_mae_px_xy": None, "size_mae_px_wh": None, "mean_iou": None,
                }
            errors = [value / count for value in row[start:start + 8]]
            return {
                "edge_mae_px_ltrb": errors[:4],
                "mean_edge_mae_px": sum(errors[:4]) / 4,
                "center_mae_px_xy": errors[4:6],
                "size_mae_px_wh": errors[6:8],
                "mean_iou": row[iou_index] / count,
            }

        groups = {}
        for name, row in zip(self.GROUP_NAMES, self.groups.cpu().tolist()):
            count = row[0]
            groups[name] = {
                "pairs": int(count), "refined_pairs": int(row[1]),
                "baseline": geometry(row, 2, 18), "refined": geometry(row, 10, 19),
                "mean_iou_change": (row[19] - row[18]) / count if count else None,
                "iou_improved": int(row[20]), "iou_degraded": int(row[21]),
                "iou_stable": int(row[22]),
            }
        counts = dict(zip(self.COUNT_NAMES, (int(v) for v in self.counts.cpu().tolist())))
        counts["unrefined_matched"] = counts["matched"] - counts["refined"]
        return {
            "coordinate_system": "network_input_pixels",
            "size_grouping": "GT input area: small < 32^2, medium [32^2, 96^2), large >= 96^2",
            "pairing": "fixed_baseline_hungarian",
            "counts": counts, "groups": groups,
        }
