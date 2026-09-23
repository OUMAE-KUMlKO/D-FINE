"""Fixed-match diagnostics for prediction-guided strip refinement."""

import torch

from .gt_refinement_metrics import GTRefinementMetrics


class PredRefinementDiagnostics:
    EDGE_NAMES = ("left", "top", "right", "bottom")
    COVERAGE_GROUPS = (
        "all_edges_inside_band", "at_least_one_edge_outside_band",
    )
    SCORE_GROUPS = (
        ("all", None), ("score_ge_0_1", 0.1),
        ("score_ge_0_3", 0.3), ("score_ge_0_5", 0.5),
    )

    def __init__(self, offset_pixels=1.0, iou_tolerance=1e-6):
        self.offset_pixels = float(offset_pixels)
        self.iou_tolerance = float(iou_tolerance)
        self.geometry = GTRefinementMetrics(iou_tolerance=iou_tolerance)
        self.prediction_counts = {
            "matched_pairs": 0, "refined_matched_pairs": 0,
            "fallback_matched_pairs": 0, "invalid_metric_pairs": 0,
        }
        self._valid_pairs = 0
        self._edge_errors = [[] for _ in self.EDGE_NAMES]
        self._inside_band = [0] * 4
        self._all_inside = 0
        self._tangent_sum = 0.0
        self._coverage_groups = {
            name: self._new_pair_group() for name in self.COVERAGE_GROUPS
        }
        self._unmatched = {
            name: {"count": 0, "refined": 0, "moved": 0,
                   "edge_displacement_sum": 0.0}
            for name, _ in self.SCORE_GROUPS
        }

    @staticmethod
    def _new_pair_group():
        return {
            "pairs": 0, "refined_pairs": 0, "baseline_iou_sum": 0.0,
            "refined_iou_sum": 0.0, "improved": 0, "degraded": 0,
        }

    @staticmethod
    def _xyxy(boxes):
        return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2), dim=-1)

    def _append_errors(self, errors):
        for edge_index, values in enumerate(errors.transpose(0, 1)):
            self._edge_errors[edge_index].extend(values.tolist())

    def _update_pair_group(self, name, before_iou, after_iou, refined):
        group = self._coverage_groups[name]
        difference = after_iou - before_iou
        group["pairs"] += int(before_iou.numel())
        group["refined_pairs"] += int(refined.sum())
        group["baseline_iou_sum"] += float(before_iou.sum())
        group["refined_iou_sum"] += float(after_iou.sum())
        group["improved"] += int((difference > self.iou_tolerance).sum())
        group["degraded"] += int((difference < -self.iou_tolerance).sum())

    def _update_unmatched(self, before, after, logits, mask, matched, scale):
        unmatched = torch.ones(len(before), dtype=torch.bool)
        unmatched[matched] = False
        if not unmatched.any():
            return
        before_xyxy = self._xyxy(before[unmatched]) * scale
        after_xyxy = self._xyxy(after[unmatched]) * scale
        displacement = (after_xyxy - before_xyxy).abs()
        mean_displacement = displacement.mean(dim=-1)
        moved = displacement.max(dim=-1).values > 1e-6
        scores = logits[unmatched].sigmoid().max(dim=-1).values
        refined = mask[unmatched]
        for name, threshold in self.SCORE_GROUPS:
            selected = torch.ones_like(scores, dtype=torch.bool)
            if threshold is not None:
                selected = scores >= threshold
            group = self._unmatched[name]
            group["count"] += int(selected.sum())
            group["refined"] += int(refined[selected].sum())
            group["moved"] += int(moved[selected].sum())
            finite = selected & torch.isfinite(mean_displacement)
            group["edge_displacement_sum"] += float(mean_displacement[finite].sum())

    @torch.no_grad()
    def update(self, outputs, targets, indices, input_size):
        before = outputs["baseline_boxes"].detach().to(device="cpu", dtype=torch.float64)
        after = outputs["pred_boxes"].detach().to(device="cpu", dtype=torch.float64)
        logits = outputs["pred_logits"].detach().to(device="cpu", dtype=torch.float64)
        masks = outputs["refinement_mask"].detach().to(device="cpu", dtype=torch.bool)
        canonical_targets = [
            {"boxes": target["boxes"].detach().to(device="cpu", dtype=torch.float64)}
            for target in targets
        ]
        fixed_indices = [
            (source.detach().to(device="cpu", dtype=torch.long),
             destination.detach().to(device="cpu", dtype=torch.long))
            for source, destination in indices
        ]
        matched = sum(source.numel() for source, _ in fixed_indices)
        refined_matched = sum(
            int(masks[batch_index, source].sum())
            for batch_index, (source, _) in enumerate(fixed_indices)
        )
        self.prediction_counts["matched_pairs"] += matched
        self.prediction_counts["refined_matched_pairs"] += refined_matched
        self.prediction_counts["fallback_matched_pairs"] += matched - refined_matched

        geometry_outputs = {
            "baseline_boxes": before, "pred_boxes": after,
            "refinement_mask": masks, "refinement_targets": canonical_targets,
            "matched_indices": fixed_indices,
            "refinement_counts": {
                "matched": matched, "refined": refined_matched, "total_gt": matched,
            },
        }
        self.geometry.update(geometry_outputs, input_size)

        height, width = input_size
        scale = before.new_tensor((width, height, width, height))
        for batch_index, (source, destination) in enumerate(fixed_indices):
            self._update_unmatched(
                before[batch_index], after[batch_index], logits[batch_index],
                masks[batch_index], source, scale,
            )
            if source.numel() == 0:
                continue
            original = before[batch_index, source]
            refined_boxes = after[batch_index, source]
            target = canonical_targets[batch_index]["boxes"][destination]
            valid = (
                torch.isfinite(original).all(dim=-1)
                & torch.isfinite(refined_boxes).all(dim=-1)
                & torch.isfinite(target).all(dim=-1)
                & (original[:, 2:] > 0).all(dim=-1)
                & (refined_boxes[:, 2:] > 0).all(dim=-1)
                & (target[:, 2:] > 0).all(dim=-1)
            )
            self.prediction_counts["invalid_metric_pairs"] += int((~valid).sum())
            original, refined_boxes, target = (
                value[valid] for value in (original, refined_boxes, target)
            )
            if original.numel() == 0:
                continue
            was_refined = masks[batch_index, source][valid]
            original_xyxy = self._xyxy(original)
            refined_xyxy = self._xyxy(refined_boxes)
            target_xyxy = self._xyxy(target)
            errors = (original_xyxy - target_xyxy).abs() * scale
            inside = errors <= self.offset_pixels + 1e-6
            all_inside = inside.all(dim=-1)
            self._valid_pairs += len(original)
            self._append_errors(errors)
            for edge_index in range(4):
                self._inside_band[edge_index] += int(inside[:, edge_index].sum())
            self._all_inside += int(all_inside.sum())

            horizontal_overlap = (
                torch.minimum(original_xyxy[:, 2], target_xyxy[:, 2])
                - torch.maximum(original_xyxy[:, 0], target_xyxy[:, 0])
            ).clamp(min=0) / (target_xyxy[:, 2] - target_xyxy[:, 0])
            vertical_overlap = (
                torch.minimum(original_xyxy[:, 3], target_xyxy[:, 3])
                - torch.maximum(original_xyxy[:, 1], target_xyxy[:, 1])
            ).clamp(min=0) / (target_xyxy[:, 3] - target_xyxy[:, 1])
            self._tangent_sum += float(
                (2 * horizontal_overlap + 2 * vertical_overlap).sum() / 4
            )
            before_iou = GTRefinementMetrics._paired_iou(original_xyxy, target_xyxy)
            after_iou = GTRefinementMetrics._paired_iou(refined_xyxy, target_xyxy)
            self._update_pair_group(
                "all_edges_inside_band", before_iou[all_inside],
                after_iou[all_inside], was_refined[all_inside],
            )
            outside = ~all_inside
            self._update_pair_group(
                "at_least_one_edge_outside_band", before_iou[outside],
                after_iou[outside], was_refined[outside],
            )

    @staticmethod
    def _mean(total, count):
        return total / count if count else None

    def compute(self):
        edge_error = {}
        for name, values in zip(self.EDGE_NAMES, self._edge_errors):
            if not values:
                edge_error[name] = {key: None for key in ("mean", "p50", "p90", "p95")}
                continue
            tensor = torch.tensor(values, dtype=torch.float64)
            quantiles = torch.quantile(
                tensor, torch.tensor([0.5, 0.9, 0.95], dtype=tensor.dtype)
            )
            edge_error[name] = {
                "mean": float(tensor.mean()), "p50": float(quantiles[0]),
                "p90": float(quantiles[1]), "p95": float(quantiles[2]),
            }
        groups = {}
        for name, accumulated in self._coverage_groups.items():
            count = accumulated["pairs"]
            baseline = self._mean(accumulated["baseline_iou_sum"], count)
            refined = self._mean(accumulated["refined_iou_sum"], count)
            groups[name] = {
                "pairs": count, "refined_pairs": accumulated["refined_pairs"],
                "baseline_iou": baseline, "refined_iou": refined,
                "mean_delta": refined - baseline if count else None,
                "improved": accumulated["improved"],
                "degraded": accumulated["degraded"],
            }
        unmatched = {}
        for name, accumulated in self._unmatched.items():
            count = accumulated["count"]
            unmatched[name] = {
                "count": count, "refined": accumulated["refined"],
                "moved": accumulated["moved"],
                "mean_edge_displacement_px": self._mean(
                    accumulated["edge_displacement_sum"], count
                ),
            }
        geometry = self.geometry.compute()["groups"]
        return {
            "coordinate_system": "network_input_pixels",
            "pairing": "fixed_baseline_hungarian",
            "coverage_note": (
                "Coordinate-strip proxy only; it does not measure feature receptive field."
            ),
            "offset_pixels": self.offset_pixels,
            "prediction_counts": dict(self.prediction_counts),
            "geometry_groups": geometry,
            "sampling_coverage": {
                "edge_error_px": edge_error,
                "inside_band_fraction": {
                    name: self._mean(value, self._valid_pairs)
                    for name, value in zip(self.EDGE_NAMES, self._inside_band)
                },
                "all_edges_inside_band_fraction": self._mean(
                    self._all_inside, self._valid_pairs
                ),
                "mean_tangent_coverage": self._mean(
                    self._tangent_sum, self._valid_pairs
                ),
                "quantile_sample_count": self._valid_pairs,
            },
            "coverage_groups": groups,
            "unmatched_queries": unmatched,
        }
