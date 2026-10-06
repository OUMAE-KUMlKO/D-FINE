"""Analyze cached, independently matched detections by size and GT density."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LOCAL_DENSITIES = ("0", "1", "2_3", "4_plus")
IMAGE_DENSITIES = ("le30", "31_60", "61_100", "gt100")
NEAREST_GAPS = ("overlap_or_touch", "gt0_le8", "gt8_le16", "gt16")
SIZES = ("small", "medium", "large")
SHORT_SIDES = ("lt8", "8_to_lt16", "16_to_lt32", "ge32")
GAP_TOLERANCE = 1e-3


def _array(value, dtype=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _xyxy(boxes):
    return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2), axis=1)


def spatial_density(boxes_xyxy):
    """Return other-GT counts within 8px and nearest rectangle-boundary gaps."""
    boxes = _array(boxes_xyxy, np.float64).reshape(-1, 4)
    if not len(boxes):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    horizontal = np.maximum(np.maximum(boxes[None, :, 0] - boxes[:, None, 2],
                                      boxes[:, None, 0] - boxes[None, :, 2]), 0)
    vertical = np.maximum(np.maximum(boxes[None, :, 1] - boxes[:, None, 3],
                                    boxes[:, None, 1] - boxes[None, :, 3]), 0)
    gaps = np.hypot(horizontal, vertical)
    np.fill_diagonal(gaps, np.inf)
    return (gaps <= 8 + GAP_TOLERANCE).sum(axis=1), gaps.min(axis=1)


def _valid_boxes(boxes):
    return np.isfinite(boxes).all(axis=1) & (boxes[:, 2:] > 0).all(axis=1)


def _paired_iou(boxes, targets):
    intersection = np.maximum(np.minimum(boxes[:, 2:], targets[:, 2:])
                              - np.maximum(boxes[:, :2], targets[:, :2]), 0).prod(axis=1)
    union = ((boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
             + (targets[:, 2:] - targets[:, :2]).prod(axis=1) - intersection)
    return intersection / np.maximum(union, 1e-12)


def _size(area):
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def _metrics(sums):
    count = int(sums[0])
    means = sums[1:] / count if count else [None] * 4
    baseline = dict(mean_edge_mae_px=means[0], mean_iou=means[1])
    model = dict(mean_edge_mae_px=means[2], mean_iou=means[3])
    return {
        "pairs": count, "baseline": baseline, "model": model,
        "delta_model_minus_baseline": {
            key: model[key] - baseline[key] if count else None for key in baseline
        },
    }


def _standardize(group_sums):
    cells = defaultdict(dict)
    for key, sums in group_sums.items():
        if key[0] == "local_density_by_category_by_short_side":
            _, density, category, short_side = key
            cells[(category, short_side)][density] = sums
    eligible = {
        cell: values for cell, values in cells.items()
        if all(density in values and values[density][0] >= 20 for density in LOCAL_DENSITIES)
    }
    pooled_total = sum(sum(values[density][0] for density in LOCAL_DENSITIES)
                       for values in eligible.values())
    all_pairs = sum(group_sums[("local_neighbor_density", density)][0]
                    for density in LOCAL_DENSITIES)
    result = {
        "minimum_pairs_per_cell_per_density": 20,
        "cell_definition": "canonical_category_x_input_short_side_bin",
        "weight_definition": "pooled_common_valid_gt_count_per_cell_across_four_density_bins",
        "eligible_cell_count": len(eligible), "eligible_pairs": int(pooled_total),
        "total_pairs": int(all_pairs),
        "coverage": pooled_total / all_pairs if all_pairs else None,
        "interpretation": "partially_adjusted_descriptive_comparison_not_causal",
        "cells": [], "by_density": None,
    }
    if not eligible:
        result["reason"] = "No category/short-side cell has >=20 common valid GTs in every density bin."
        return result
    weighted = {density: np.zeros(4, dtype=np.float64) for density in LOCAL_DENSITIES}
    retained = {density: 0 for density in LOCAL_DENSITIES}
    for (category, short_side), values in sorted(eligible.items()):
        pooled_count = sum(values[density][0] for density in LOCAL_DENSITIES)
        weight = pooled_count / pooled_total
        result["cells"].append({
            "category": category, "short_side_bin": short_side,
            "pooled_pairs": int(pooled_count), "weight": weight,
            "pairs_by_density": {density: int(values[density][0]) for density in LOCAL_DENSITIES},
        })
        for density in LOCAL_DENSITIES:
            sums = values[density]
            weighted[density] += weight * sums[1:] / sums[0]
            retained[density] += int(sums[0])
    result["by_density"] = {}
    for density in LOCAL_DENSITIES:
        total = int(group_sums[("local_neighbor_density", density)][0])
        values = _metrics(np.concatenate(([1], weighted[density])))
        del values["pairs"]  # Weighted means have no synthetic observation count.
        values.update(retained_pairs=retained[density], total_pairs=total,
                      coverage=retained[density] / total if total else None)
        result["by_density"][density] = values
    return result


def analyze_cache(cache):
    sums = defaultdict(lambda: np.zeros(5, dtype=np.float64))
    sums[("overall",)]
    for section, keys in (("image_density", IMAGE_DENSITIES),
                          ("local_neighbor_density", LOCAL_DENSITIES),
                          ("nearest_gap", NEAREST_GAPS),
                          ("input_area_size", SIZES), ("original_area_size", SIZES)):
        for key in keys:
            sums[(section, key)]
    counts = dict(images=0, canonical_gt=0, valid_gt=0, invalid_gt=0,
                  baseline_matched_gt=0, model_matched_gt=0, common_matched_gt=0,
                  baseline_only=0, model_only=0, neither=0,
                  common_valid_pairs=0, invalid_common_pairs=0, invalid_original_area_pairs=0)
    per_image = []
    for record in cache["records"]:
        target = record["target"]
        target_boxes = _array(target["boxes"], np.float64)
        labels = _array(target["labels"], np.int64)
        areas_original = _array(target["area_original"], np.float64)
        height, width = record["input_size"]
        scale = np.array([width, height, width, height], dtype=np.float64)
        valid_gt = _valid_boxes(target_boxes)
        target_xyxy = _xyxy(target_boxes) * scale
        gt_count = int(valid_gt.sum())
        image_density = IMAGE_DENSITIES[int(gt_count > 30) + int(gt_count > 60) + int(gt_count > 100)]
        neighbors = np.zeros(len(target_boxes), dtype=np.int64)
        nearest = np.full(len(target_boxes), np.inf)
        neighbors[valid_gt], nearest[valid_gt] = spatial_density(target_xyxy[valid_gt])
        matches = record["matches"]["geometry_only"]
        baseline_map = dict(zip(_array(matches["baseline"][1], np.int64),
                                _array(matches["baseline"][0], np.int64)))
        model_map = dict(zip(_array(matches["model"][1], np.int64),
                             _array(matches["model"][0], np.int64)))
        baseline_ids, model_ids = set(baseline_map), set(model_map)
        common = np.array(sorted(baseline_ids & model_ids), dtype=np.int64)
        baseline = _array(record["baseline"]["pred_boxes"], np.float64)[
            np.array([baseline_map[index] for index in common], dtype=np.int64)]
        model = _array(record["model"]["pred_boxes"], np.float64)[
            np.array([model_map[index] for index in common], dtype=np.int64)]
        valid_pairs = valid_gt[common] & _valid_boxes(baseline) & _valid_boxes(model)
        image_counts = {
            "image_id": int(record["image_id"]), "canonical_gt": len(target_boxes),
            "valid_gt": gt_count, "invalid_gt": len(target_boxes) - gt_count,
            "baseline_matched_gt": len(baseline_ids), "model_matched_gt": len(model_ids),
            "common_matched_gt": len(common), "baseline_only": len(baseline_ids - model_ids),
            "model_only": len(model_ids - baseline_ids),
            "neither": len(target_boxes) - len(baseline_ids | model_ids),
            "common_valid_pairs": int(valid_pairs.sum()),
            "invalid_common_pairs": int((~valid_pairs).sum()),
            "image_density_bin": image_density,
        }
        counts["images"] += 1
        for name in counts.keys() - {"images", "invalid_original_area_pairs"}:
            counts[name] += image_counts[name]
        per_image.append(image_counts)
        common = common[valid_pairs]
        baseline_xyxy = _xyxy(baseline[valid_pairs]) * scale
        model_xyxy = _xyxy(model[valid_pairs]) * scale
        truth = target_xyxy[common]
        values = np.column_stack((
            np.ones(len(common)), np.abs(baseline_xyxy - truth).mean(axis=1),
            _paired_iou(baseline_xyxy, truth), np.abs(model_xyxy - truth).mean(axis=1),
            _paired_iou(model_xyxy, truth),
        ))
        sums[("overall",)] += values.sum(axis=0)
        sums[("image_density", image_density)] += values.sum(axis=0)
        for index, row in zip(common, values):
            category = str(int(labels[index]))
            number = neighbors[index]
            local = "0" if number == 0 else "1" if number == 1 else "2_3" if number < 4 else "4_plus"
            gap = nearest[index]
            gap_bin = ("overlap_or_touch" if gap <= GAP_TOLERANCE else "gt0_le8"
                       if gap <= 8 + GAP_TOLERANCE else "gt8_le16"
                       if gap <= 16 + GAP_TOLERANCE else "gt16")
            side_lengths = target_boxes[index, 2:] * [width, height]
            input_size = _size(side_lengths.prod())
            short_side = side_lengths.min()
            short_bin = SHORT_SIDES[int(short_side >= 8) + int(short_side >= 16) + int(short_side >= 32)]
            paths = [
                ("local_neighbor_density", local), ("nearest_gap", gap_bin),
                ("input_area_size", input_size),
                ("category_by_input_area_size", category, input_size),
                ("local_density_by_input_area_size", local, input_size),
                ("local_density_by_category_by_short_side", local, category, short_bin),
            ]
            original_area = areas_original[index]
            if np.isfinite(original_area) and 0 <= original_area <= 1e10:
                # COCO uses closed ranges: boundary GTs belong to both groups.
                for original_size, lower, upper in (("small", 0, 1024),
                                                     ("medium", 1024, 9216),
                                                     ("large", 9216, 1e10)):
                    if lower <= original_area <= upper:
                        paths.extend([("original_area_size", original_size),
                                      ("category_by_original_area_size", category, original_size)])
            else:
                counts["invalid_original_area_pairs"] += 1
            for path in paths:
                sums[path] += row
    groups = {}
    for path, values in sums.items():
        parent = groups
        for key in path[:-1]:
            parent = parent.setdefault(key, {})
        parent[path[-1]] = _metrics(values)
    return {
        "schema_version": 1,
        "pairing": "cached_geometry_only_hungarian_matches_intersected_by_gt_identity",
        "coordinate_system": "network_input_pixels",
        "density_population": "all_canonical_gt_with_finite_positive_geometry_in_each_image",
        "geometry_filter": "common_GT_and_both_matched_predictions_finite_and_positive_width_height",
        "density_definition": {
            "image_density": "valid_gt_count: <=30, 31-60, 61-100, >100",
            "local_neighbor_density": "number_of_other_GT_rectangles_with_boundary_gap<=8px+tolerance",
            "nearest_gap": "minimum_Euclidean_rectangle_boundary_gap; overlap/touch=0; single_GT=infinity",
            "tolerance_px": GAP_TOLERANCE,
            "input_area_bins": "small<32^2; medium>=32^2_and_<96^2; large>=96^2",
            "original_area_bins": "COCO closed ranges: small=[0,1024]; medium=[1024,9216]; large=[9216,1e10]",
            "original_area_overlap": "thresholds 1024 and 9216 each belong to both adjacent groups",
            "invalid_original_area_pairs": "nonfinite, negative, or >1e10 area; excluded only from original-area groups",
            "short_side_bins_px": "<8; >=8_and_<16; >=16_and_<32; >=32",
        },
        "counts": counts, "per_image": per_image, "groups": groups,
        "standardized_local_neighbor_density": _standardize(sums),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    if args.output_file.exists():
        parser.error(f"Output already exists: {args.output_file}")
    import torch

    cache = torch.load(args.prediction_cache, map_location="cpu", weights_only=False)
    result = analyze_cache(cache)
    result["prediction_cache"] = str(args.prediction_cache.resolve())
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                          allow_nan=False) + "\n")
    print(json.dumps({"output_file": str(args.output_file), "counts": result["counts"],
                      "overall": result["groups"]["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
