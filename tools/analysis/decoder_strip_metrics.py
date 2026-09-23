"""Accumulate geometry changes on caller-supplied, fixed B0 query/GT pairs.

Matching is performed by the diagnostic entry point. IoU groups always use
the final B0 reference boxes, while before/after may describe any traced stage.
Edge errors, movements, and GT-size groups use network-input pixel units.
"""

import torch


class FixedPairComparison:
    GROUP_NAMES = (
        "all", "small", "medium", "large",
        "baseline_iou_lt_0_50", "baseline_iou_0_50_to_0_75",
        "baseline_iou_ge_0_75",
    )

    def __init__(self, iou_tolerance=1e-6):
        self.iou_tolerance = float(iou_tolerance)
        self.invalid_pairs = 0
        self._groups = {
            name: {
                "pairs": 0, "before_iou": 0.0, "after_iou": 0.0,
                "before_edge_mae": 0.0, "after_edge_mae": 0.0,
                "improved": 0, "degraded": 0, "stable": 0,
                "movements": [],
            }
            for name in self.GROUP_NAMES
        }

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

    def _accumulate(self, name, selection, before_iou, after_iou,
                    before_error, after_error, movement):
        if not selection.any():
            return
        group = self._groups[name]
        difference = after_iou - before_iou
        group["pairs"] += int(selection.sum())
        group["before_iou"] += float(before_iou[selection].sum())
        group["after_iou"] += float(after_iou[selection].sum())
        group["before_edge_mae"] += float(before_error[selection].sum())
        group["after_edge_mae"] += float(after_error[selection].sum())
        group["improved"] += int((difference[selection] > self.iou_tolerance).sum())
        group["degraded"] += int((difference[selection] < -self.iou_tolerance).sum())
        group["stable"] += int((difference[selection].abs() <= self.iou_tolerance).sum())
        group["movements"].extend(movement[selection].tolist())

    @torch.no_grad()
    def update(self, reference_boxes, before_boxes, after_boxes,
               targets, indices, image_size):
        """Compare normalized cxcywh boxes using existing pairs without rematching.

        ``targets`` contains canonical normalized GT boxes and ``image_size``
        is the network input's (height, width), not the original image size.
        """
        reference = reference_boxes.detach().to(device="cpu", dtype=torch.float64)
        before = before_boxes.detach().to(device="cpu", dtype=torch.float64)
        after = after_boxes.detach().to(device="cpu", dtype=torch.float64)
        if reference.shape != before.shape or reference.shape != after.shape:
            raise ValueError("reference, before, and after boxes must share [B, Q, 4].")
        if reference.ndim != 3 or reference.shape[-1] != 4:
            raise ValueError("boxes must have shape [B, Q, 4].")
        if len(targets) != len(reference) or len(indices) != len(reference):
            raise ValueError("targets and indices must contain one item per image.")
        height, width = image_size
        if height <= 0 or width <= 0:
            raise ValueError("image_size must contain positive (height, width).")
        scale = reference.new_tensor((width, height, width, height))
        for batch_index, (query_ids, target_ids) in enumerate(indices):
            query_ids = query_ids.detach().to(device="cpu", dtype=torch.long)
            target_ids = target_ids.detach().to(device="cpu", dtype=torch.long)
            if query_ids.numel() == 0:
                continue
            target_boxes = targets[batch_index]["boxes"].detach().to(
                device="cpu", dtype=torch.float64
            )[target_ids]
            ref = reference[batch_index, query_ids]
            old = before[batch_index, query_ids]
            new = after[batch_index, query_ids]
            valid = (
                torch.isfinite(ref).all(-1) & torch.isfinite(old).all(-1)
                & torch.isfinite(new).all(-1) & torch.isfinite(target_boxes).all(-1)
                & (ref[:, 2:] > 0).all(-1) & (old[:, 2:] > 0).all(-1)
                & (new[:, 2:] > 0).all(-1) & (target_boxes[:, 2:] > 0).all(-1)
            )
            self.invalid_pairs += int((~valid).sum())
            ref, old, new, target_boxes = (
                value[valid] for value in (ref, old, new, target_boxes)
            )
            if ref.numel() == 0:
                continue
            ref_xyxy, old_xyxy, new_xyxy, target_xyxy = (
                self._xyxy(value) for value in (ref, old, new, target_boxes)
            )
            reference_iou = self._paired_iou(ref_xyxy, target_xyxy)
            before_iou = self._paired_iou(old_xyxy, target_xyxy)
            after_iou = self._paired_iou(new_xyxy, target_xyxy)
            before_error = ((old_xyxy - target_xyxy).abs() * scale).mean(-1)
            after_error = ((new_xyxy - target_xyxy).abs() * scale).mean(-1)
            movement = ((new_xyxy - old_xyxy).abs() * scale).mean(-1)
            target_area = target_boxes[:, 2] * width * target_boxes[:, 3] * height
            selections = (
                torch.ones_like(reference_iou, dtype=torch.bool),
                target_area < 32 ** 2,
                (target_area >= 32 ** 2) & (target_area < 96 ** 2),
                target_area >= 96 ** 2,
                reference_iou < 0.5,
                (reference_iou >= 0.5) & (reference_iou < 0.75),
                reference_iou >= 0.75,
            )
            for name, selection in zip(self.GROUP_NAMES, selections):
                self._accumulate(
                    name, selection, before_iou, after_iou,
                    before_error, after_error, movement,
                )

    def compute(self):
        groups = {}
        for name, accumulated in self._groups.items():
            count = accumulated["pairs"]
            if count:
                before_iou = accumulated["before_iou"] / count
                after_iou = accumulated["after_iou"] / count
                before_error = accumulated["before_edge_mae"] / count
                after_error = accumulated["after_edge_mae"] / count
                movements = torch.tensor(accumulated["movements"], dtype=torch.float64)
                movement_mean = float(movements.mean())
                movement_p90 = float(torch.quantile(movements, 0.9))
            else:
                before_iou = after_iou = before_error = after_error = None
                movement_mean = movement_p90 = None
            groups[name] = {
                "pairs": count,
                "before": {
                    "mean_iou": before_iou, "mean_edge_mae_px": before_error,
                },
                "after": {
                    "mean_iou": after_iou, "mean_edge_mae_px": after_error,
                },
                "mean_iou_change": after_iou - before_iou if count else None,
                "mean_edge_mae_change_px": after_error - before_error if count else None,
                "iou_improved": accumulated["improved"],
                "iou_degraded": accumulated["degraded"],
                "iou_stable": accumulated["stable"],
                "mean_edge_movement_px": movement_mean,
                "edge_movement_p90_px": movement_p90,
            }
        return {"invalid_pairs": self.invalid_pairs, "groups": groups}
