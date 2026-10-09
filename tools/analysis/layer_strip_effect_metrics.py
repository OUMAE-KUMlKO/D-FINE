"""Measure a strip head's immediate effect using caller-fixed query/GT pairs.

No matching or filtering by detection score is performed here. ``valid`` marks
whether the strip head applied a correction; fallback boxes remain in geometry
statistics. All geometry is accumulated on CPU in float64. Pre-IoU groups are
conditional descriptions, not estimates of causal effects.
"""

import math

import torch

from tools.analysis.decoder_strip_metrics import FixedPairComparison


class LayerEffectMetrics:
    GROUP_NAMES = (
        "all", "original_small", "original_medium", "original_large",
        "short_lt8", "short_8_to_lt16", "short_16_to_lt32", "short_ge32",
        "before_iou_lt0_5", "before_iou_0_5_to0_75", "before_iou_ge0_75",
        *(f"category_{category}" for category in range(10)),
    )
    SIDE_NAMES = ("left", "top", "right", "bottom")

    def __init__(self, iou_tolerance=1e-6, half_width_bounds_px=(2., 8.)):
        self.iou_tolerance = float(iou_tolerance)
        self.half_width_bounds_px = tuple(float(v) for v in half_width_bounds_px)
        self.invalid_pairs = 0
        self._groups = {name: self._new_group() for name in self.GROUP_NAMES}

    @staticmethod
    def _new_group():
        return {
            "pairs": 0, "before_iou": 0., "after_iou": 0.,
            "before_edges": torch.zeros(4, dtype=torch.float64),
            "after_edges": torch.zeros(4, dtype=torch.float64),
            "improved": 0, "degraded": 0, "stable": 0,
            "crossings": {"0.5": {"up": 0, "down": 0},
                          "0.75": {"up": 0, "down": 0}},
            "movements": [], "mask_observed": 0, "mask_valid": 0,
            "width_count": [0, 0], "width_sum": [0., 0.],
            "width_min_count": [0, 0], "width_max_count": [0, 0],
        }

    def _accumulate(self, name, selection, before_iou, after_iou, before_edges,
                    after_edges, movement, mask, widths):
        count = int(selection.sum())
        if not count:
            return
        group = self._groups.setdefault(name, self._new_group())
        old, new = before_iou[selection], after_iou[selection]
        difference = new - old
        group["pairs"] += count
        group["before_iou"] += float(old.sum())
        group["after_iou"] += float(new.sum())
        group["before_edges"] += before_edges[selection].sum(0)
        group["after_edges"] += after_edges[selection].sum(0)
        group["improved"] += int((difference > self.iou_tolerance).sum())
        group["degraded"] += int((difference < -self.iou_tolerance).sum())
        group["stable"] += int((difference.abs() <= self.iou_tolerance).sum())
        for threshold, crossings in group["crossings"].items():
            threshold = float(threshold)
            crossings["up"] += int(((old < threshold) & (new >= threshold)).sum())
            crossings["down"] += int(((old >= threshold) & (new < threshold)).sum())
        group["movements"].extend(movement[selection].tolist())
        if mask is not None:
            group["mask_observed"] += count
            group["mask_valid"] += int(mask[selection].sum())
        if widths is not None:
            for direction in range(2):
                values = widths[selection, direction]
                values = values[torch.isfinite(values) & (values >= 0)]
                group["width_count"][direction] += len(values)
                group["width_sum"][direction] += float(values.sum())
                for bound, value in zip(("min", "max"), self.half_width_bounds_px):
                    group[f"width_{bound}_count"][direction] += int(
                        torch.isclose(values, values.new_tensor(value), atol=1e-6, rtol=0).sum()
                    )

    @torch.no_grad()
    def update(self, before_boxes, after_boxes, targets, indices, image_size,
               valid=None, half_widths_px=None):
        """Accumulate normalized cxcywh boxes using the same pairs before/after.

        ``image_size`` is input (H, W). Targets contain normalized ``boxes``,
        ``labels`` and original COCO ``area_original`` aligned in the same order.
        Optional ``half_widths_px`` is [B,Q,2] with x/y half-widths in input pixels.
        """
        before = before_boxes.detach().to(device="cpu", dtype=torch.float64)
        after = after_boxes.detach().to(device="cpu", dtype=torch.float64)
        if before.shape != after.shape or before.ndim != 3 or before.shape[-1] != 4:
            raise ValueError("before and after boxes must share shape [B,Q,4].")
        if len(targets) != len(before) or len(indices) != len(before):
            raise ValueError("targets and indices must have one entry per image.")
        height, width = (float(value) for value in image_size)
        if not all(math.isfinite(value) and value > 0 for value in (height, width)):
            raise ValueError("image_size must be finite positive (height, width).")
        scale = before.new_tensor((width, height, width, height))
        if valid is not None:
            valid = valid.detach().to(device="cpu", dtype=torch.bool)
            if valid.shape != before.shape[:2]:
                raise ValueError("valid must have shape [B,Q].")
        if half_widths_px is not None:
            half_widths_px = half_widths_px.detach().to(device="cpu", dtype=torch.float64)
            if half_widths_px.shape != (*before.shape[:2], 2):
                raise ValueError("half_widths_px must have shape [B,Q,2].")

        for batch_index, (query_ids, target_ids) in enumerate(indices):
            query_ids = query_ids.detach().to(device="cpu", dtype=torch.long)
            target_ids = target_ids.detach().to(device="cpu", dtype=torch.long)
            if query_ids.ndim != 1 or query_ids.shape != target_ids.shape:
                raise ValueError("query_ids and target_ids must be equal-length vectors.")
            if not len(query_ids):
                continue
            target = targets[batch_index]
            gt = target["boxes"].detach().to(device="cpu", dtype=torch.float64)[target_ids]
            labels = target["labels"].detach().to(device="cpu", dtype=torch.long)[target_ids]
            original_area = target["area_original"].detach().to(
                device="cpu", dtype=torch.float64
            )[target_ids]
            old, new = before[batch_index, query_ids], after[batch_index, query_ids]
            geometry_valid = torch.ones(len(query_ids), dtype=torch.bool)
            for boxes in (old, new, gt):
                geometry_valid &= torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 0).all(-1)
            self.invalid_pairs += int((~geometry_valid).sum())
            if not geometry_valid.any():
                continue
            old, new, gt = (boxes[geometry_valid] for boxes in (old, new, gt))
            labels, original_area = labels[geometry_valid], original_area[geometry_valid]
            mask = valid[batch_index, query_ids][geometry_valid] if valid is not None else None
            widths = (half_widths_px[batch_index, query_ids][geometry_valid]
                      if half_widths_px is not None else None)
            old_xyxy, new_xyxy, gt_xyxy = (FixedPairComparison._xyxy(boxes)
                                         for boxes in (old, new, gt))
            old_iou = FixedPairComparison._paired_iou(old_xyxy, gt_xyxy)
            new_iou = FixedPairComparison._paired_iou(new_xyxy, gt_xyxy)
            before_edges = (old_xyxy - gt_xyxy).abs() * scale
            after_edges = (new_xyxy - gt_xyxy).abs() * scale
            movement = ((new_xyxy - old_xyxy).abs() * scale).mean(-1)
            short_side = torch.minimum(gt[:, 2] * width, gt[:, 3] * height)
            selections = {
                "all": torch.ones_like(old_iou, dtype=torch.bool),
                # COCO includes both endpoints; exact threshold areas overlap.
                "original_small": (original_area >= 0) & (original_area <= 1024),
                "original_medium": (original_area >= 1024) & (original_area <= 9216),
                "original_large": (original_area >= 9216) & (original_area <= 1e10),
                "short_lt8": short_side < 8,
                "short_8_to_lt16": (short_side >= 8) & (short_side < 16),
                "short_16_to_lt32": (short_side >= 16) & (short_side < 32),
                "short_ge32": short_side >= 32,
                "before_iou_lt0_5": old_iou < .5,
                "before_iou_0_5_to0_75": (old_iou >= .5) & (old_iou < .75),
                "before_iou_ge0_75": old_iou >= .75,
            }
            selections.update({f"category_{category}": labels == category
                               for category in labels.unique().tolist()})
            for name, selection in selections.items():
                self._accumulate(name, selection, old_iou, new_iou, before_edges,
                                 after_edges, movement, mask, widths)

    def compute(self):
        groups = {}
        for name, accumulated in self._groups.items():
            count = accumulated["pairs"]
            result = {"pairs": count, "n": count}
            for stage in ("before", "after"):
                edges = accumulated[f"{stage}_edges"] / count if count else None
                result[stage] = {
                    "mean_iou": accumulated[f"{stage}_iou"] / count if count else None,
                    "mean_edge_mae_px": float(edges.mean()) if count else None,
                    "mean_edge_mae_by_side_px": {
                        side: float(edges[index]) if count else None
                        for index, side in enumerate(self.SIDE_NAMES)
                    },
                }
            result["mean_iou_change"] = (result["after"]["mean_iou"]
                                         - result["before"]["mean_iou"] if count else None)
            result["mean_edge_mae_change_px"] = (result["after"]["mean_edge_mae_px"]
                                                 - result["before"]["mean_edge_mae_px"]
                                                 if count else None)
            for change in ("improved", "degraded", "stable"):
                result[f"iou_{change}"] = accumulated[change]
                result[f"iou_{change}_fraction"] = accumulated[change] / count if count else None
            result["iou_threshold_crossings"] = {
                key: dict(value) for key, value in accumulated["crossings"].items()
            }
            movements = torch.tensor(accumulated["movements"], dtype=torch.float64)
            result["mean_edge_movement_px"] = float(movements.mean()) if count else None
            result["edge_movement_p90_px"] = float(torch.quantile(movements, .9)) if count else None
            observed = accumulated["mask_observed"]
            result["valid_fraction"] = accumulated["mask_valid"] / observed if observed else None
            result["valid_mask_observed_pairs"] = observed
            for direction, axis in enumerate(("x", "y")):
                observed = accumulated["width_count"][direction]
                result[f"half_width_{axis}_observed_pairs"] = observed
                result[f"mean_half_width_{axis}_px"] = (
                    accumulated["width_sum"][direction] / observed if observed else None
                )
                for bound in ("min", "max"):
                    result[f"half_width_{axis}_{bound}_fraction"] = (
                        accumulated[f"width_{bound}_count"][direction] / observed
                        if observed else None
                    )
            groups[name] = result
        return {
            "invalid_pairs": self.invalid_pairs,
            "metadata": {
                "iou_tolerance": self.iou_tolerance,
                "half_width_summary_bounds_px": list(self.half_width_bounds_px),
                "half_width_bound_tolerance_px": 1e-6,
                "geometry_units": "network input pixels; normalized boxes converted with (H,W)",
                "original_area_groups": "COCO closed intervals [0,1024], [1024,9216], [9216,1e10]",
                "before_iou_groups": "Condition on the same pair's pre-correction IoU; descriptive only",
            },
            "groups": groups,
        }
