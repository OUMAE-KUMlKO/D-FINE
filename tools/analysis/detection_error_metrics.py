"""Decompose detector false positives using fixed official COCO matching."""

import contextlib
import copy
import io

import numpy as np
from faster_coco_eval import COCOeval_faster
from pycocotools.cocoeval import COCOeval

from tools.analysis.evaluate_density_coco import extract_metrics


ERROR_TYPES = (
    "duplicate", "classification", "localization",
    "classification_localization", "background",
)
SCORE_THRESHOLDS = (0.0, 0.1, 0.3, 0.5)
COUNT_IOUS = (0.5, 0.75)
TOLERANCE = 1e-7


def _bbox_ious(detections, ground_truths):
    if not detections or not ground_truths:
        return np.zeros((len(detections), len(ground_truths)), dtype=np.float64)
    d = np.asarray([ann["bbox"] for ann in detections], dtype=np.float64)
    g = np.asarray([ann["bbox"] for ann in ground_truths], dtype=np.float64)
    d_xy2 = d[:, :2] + d[:, 2:]
    g_xy2 = g[:, :2] + g[:, 2:]
    intersection = (
        np.maximum(0, np.minimum(d_xy2[:, None], g_xy2[None])
                   - np.maximum(d[:, None, :2], g[None, :, :2]))
    ).prod(axis=2)
    union = d[:, 2:].prod(axis=1)[:, None] + g[:, 2:].prod(axis=1)[None] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _precision_curve(tp, fp, gt_count, rec_thrs):
    if not gt_count:
        return np.full(len(rec_thrs), -1.0, dtype=np.float64)
    tp_sum = np.cumsum(tp, dtype=np.float64)
    fp_sum = np.cumsum(fp, dtype=np.float64)
    if not len(tp_sum):
        return np.zeros(len(rec_thrs), dtype=np.float64)
    recall = tp_sum / gt_count
    precision = tp_sum / (tp_sum + fp_sum + np.spacing(1))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, rec_thrs, side="left")
    result = np.zeros(len(rec_thrs), dtype=np.float64)
    valid = indices < len(precision)
    result[valid] = precision[indices[valid]]
    return result


def _ap_values(precision, i50, i75):
    valid = precision >= 0
    if not valid.any():
        return {"AP": None, "AP50": None, "AP75": None}
    return {
        "AP": float(precision[valid].mean()),
        "AP50": float(precision[i50][precision[i50] >= 0].mean()),
        "AP75": float(precision[i75][precision[i75] >= 0].mean()),
    }


def _with_gains(values, original):
    result = dict(values)
    for name in ("AP", "AP50", "AP75"):
        result[f"{name}_gain_pp"] = (
            None if values[name] is None or original[name] is None
            else (values[name] - original[name]) * 100.0
        )
    return result


def _per_iou_metrics(evaluator):
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    params = evaluator.params
    area = list(params.areaRngLbl).index("all")
    maxdet = list(params.maxDets).index(100)

    def mean(values):
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else None

    overall, categories = {}, {str(cat): {} for cat in params.catIds}
    for t, threshold in enumerate(params.iouThrs):
        key = f"{threshold:.2f}"
        overall[key] = {
            "AP": mean(precision[t, :, :, area, maxdet]),
            "AR100": mean(recall[t, :, area, maxdet]),
        }
        for k, category_id in enumerate(params.catIds):
            categories[str(category_id)][key] = {
                "AP": mean(precision[t, :, k, area, maxdet]),
                "AR100": mean(np.asarray([recall[t, k, area, maxdet]])),
            }
    return {"overall": overall, "categories": categories}


def _faster_evaluation(coco, predictions):
    detections = coco.loadRes(predictions)
    faster = COCOeval_faster(
        coco, detections, "bbox", print_function=lambda message: None,
    )
    faster.params.maxDets = [1, 10, 100]
    faster.evaluate()
    faster.accumulate()
    return detections, faster


def _official_evaluation(coco, detections, faster):
    official = COCOeval(coco, detections, "bbox")
    official.params.imgIds = list(faster.params.imgIds)
    official.params.catIds = list(faster.params.catIds)
    official.params.iouThrs = np.asarray(faster.params.iouThrs)
    official.params.recThrs = np.asarray(faster.params.recThrs)
    official.params.areaRng = [list(faster.params.areaRng[0])]
    official.params.areaRngLbl = ["all"]
    official.params.maxDets = [100]
    with contextlib.redirect_stdout(io.StringIO()):
        official.evaluate()
        official.accumulate()
    return official


def _normalized_ground_truth(coco):
    annotations = coco.dataset.get("annotations", [])
    nonpositive = sum(int(annotation["id"]) <= 0 for annotation in annotations)
    minimum = min((int(annotation["id"]) for annotation in annotations), default=1)
    offset = 1 - minimum if minimum <= 0 else 0
    metadata = {
        "applied": bool(offset), "offset": int(offset),
        "nonpositive_gt_count": int(nonpositive),
        "strategy": "add one constant offset to every ground-truth annotation id",
    }
    if not offset:
        return coco, metadata
    dataset = copy.deepcopy(coco.dataset)
    for annotation in dataset["annotations"]:
        annotation["id"] = int(annotation["id"]) + offset
    normalized = type(coco)()
    normalized.dataset = dataset
    normalized.createIndex()
    return normalized, metadata


def _evaluation_records(official):
    image_count = len(official.params.imgIds)
    records = {}
    for category_index, category_id in enumerate(official.params.catIds):
        start = category_index * image_count
        records[int(category_id)] = [
            entry for entry in official.evalImgs[start:start + image_count]
            if entry is not None
        ]
    return records


def _geometry_cache(coco, detections, records):
    valid_gt_ids = {int(image_id): [] for image_id in coco.getImgIds()}
    detection_ids = {int(image_id): [] for image_id in coco.getImgIds()}
    seen_detections = {int(image_id): set() for image_id in coco.getImgIds()}
    for entries in records.values():
        for entry in entries:
            image_id = int(entry["image_id"])
            valid_gt_ids[image_id].extend(
                int(gt_id) for gt_id, ignored in zip(entry["gtIds"], entry["gtIgnore"])
                if not ignored
            )
            for detection_id in entry["dtIds"]:
                detection_id = int(detection_id)
                if detection_id not in seen_detections[image_id]:
                    seen_detections[image_id].add(detection_id)
                    detection_ids[image_id].append(detection_id)

    cache = {}
    for image_id in valid_gt_ids:
        gt_ids = list(dict.fromkeys(valid_gt_ids[image_id]))
        dt_ids = detection_ids[image_id]
        gt_annotations = [coco.anns[gt_id] for gt_id in gt_ids]
        dt_annotations = [detections.anns[dt_id] for dt_id in dt_ids]
        cache[image_id] = {
            "gt_ids": np.asarray(gt_ids, dtype=np.int64),
            "gt_categories": np.asarray(
                [ann["category_id"] for ann in gt_annotations], dtype=np.int64,
            ),
            "dt_rows": {dt_id: index for index, dt_id in enumerate(dt_ids)},
            "ious": _bbox_ious(dt_annotations, gt_annotations),
        }
    return cache


def _error_type(entry, threshold_index, detection_id, category_id, geometry):
    image_geometry = geometry[int(entry["image_id"])]
    row = image_geometry["ious"][image_geometry["dt_rows"][int(detection_id)]]
    gt_ids = image_geometry["gt_ids"]
    categories = image_geometry["gt_categories"]
    same = categories == category_id
    other = ~same
    matched_ids = {
        int(gt_id) for gt_id, match, ignored in zip(
            entry["gtIds"], entry["gtMatches"][threshold_index], entry["gtIgnore"],
        ) if match and not ignored
    }
    matched_same = same & np.isin(gt_ids, list(matched_ids))
    threshold = float(entry_thresholds(entry)[threshold_index])
    if matched_same.any() and row[matched_same].max(initial=0) >= threshold:
        return "duplicate"
    if other.any() and row[other].max(initial=0) >= threshold:
        return "classification"
    if same.any() and row[same].max(initial=0) >= 0.1:
        return "localization"
    if other.any() and row[other].max(initial=0) >= 0.1:
        return "classification_localization"
    return "background"


def entry_thresholds(entry):
    """Thresholds are attached once while records are assembled."""
    return entry["_iou_thresholds"]


def _category_data(official, records, geometry):
    data = {}
    thresholds = np.asarray(official.params.iouThrs)
    for category_id, entries in records.items():
        for entry in entries:
            entry["_iou_thresholds"] = thresholds
        gt_count = sum(np.count_nonzero(np.asarray(entry["gtIgnore"]) == 0) for entry in entries)
        if entries:
            scores = np.concatenate([np.asarray(entry["dtScores"]) for entry in entries])
            matches = np.concatenate([entry["dtMatches"] for entry in entries], axis=1)
            ignored = np.concatenate([entry["dtIgnore"] for entry in entries], axis=1).astype(bool)
            detection_ids = np.concatenate([
                np.asarray(entry["dtIds"], dtype=np.int64) for entry in entries
            ])
            error_chunks = []
            for entry in entries:
                errors = np.full((len(thresholds), len(entry["dtIds"])), "", dtype=object)
                for t in range(len(thresholds)):
                    for d, detection_id in enumerate(entry["dtIds"]):
                        if not entry["dtIgnore"][t, d] and not entry["dtMatches"][t, d]:
                            errors[t, d] = _error_type(
                                entry, t, detection_id, category_id, geometry,
                            )
                error_chunks.append(errors)
            errors = np.concatenate(error_chunks, axis=1)
            order = np.argsort(-scores, kind="mergesort")
            scores, matches, ignored = scores[order], matches[:, order], ignored[:, order]
            detection_ids, errors = detection_ids[order], errors[:, order]
        else:
            scores = np.empty(0)
            matches = np.zeros((len(thresholds), 0))
            ignored = np.zeros((len(thresholds), 0), dtype=bool)
            detection_ids = np.empty(0, dtype=np.int64)
            errors = np.empty((len(thresholds), 0), dtype=object)
        data[category_id] = {
            "gt_count": int(gt_count), "scores": scores,
            "tp": (matches > 0) & ~ignored,
            "fp": (matches == 0) & ~ignored,
            "errors": errors, "detection_ids": detection_ids,
        }
    return data


def _validate_precision(official, faster, category_data):
    reconstructed = np.full_like(official.eval["precision"][:, :, :, 0, 0], -1.0)
    for k, category_id in enumerate(official.params.catIds):
        values = category_data[int(category_id)]
        for t in range(len(official.params.iouThrs)):
            reconstructed[t, :, k] = _precision_curve(
                values["tp"][t], values["fp"][t], values["gt_count"],
                official.params.recThrs,
            )
    py_precision = official.eval["precision"][:, :, :, 0, 0]
    faster_area = list(faster.params.areaRngLbl).index("all")
    faster_maxdet = list(faster.params.maxDets).index(100)
    faster_precision = faster.eval["precision"][:, :, :, faster_area, faster_maxdet]
    py_difference = float(np.max(np.abs(reconstructed - py_precision), initial=0))
    faster_difference = float(np.max(np.abs(reconstructed - faster_precision), initial=0))
    if py_difference > TOLERANCE or faster_difference > TOLERANCE:
        raise ValueError(
            f"Recomputed precision differs from COCO evaluation: "
            f"pycocotools={py_difference}, faster_coco_eval={faster_difference}"
        )
    return {
        "tolerance": TOLERANCE,
        "pycocotools_max_abs": py_difference,
        "faster_coco_eval_max_abs": faster_difference,
    }


def _empty_counts():
    return {
        "tp": 0, **{error: 0 for error in ERROR_TYPES},
        "fp": 0, "fn": 0,
    }


def _counts(official, category_data):
    result = {}
    for score_threshold in SCORE_THRESHOLDS:
        score_key = f"{score_threshold:g}"
        result[score_key] = {}
        for iou in COUNT_IOUS:
            t = int(np.flatnonzero(np.isclose(official.params.iouThrs, iou))[0])
            categories = {}
            overall = _empty_counts()
            for category_id in official.params.catIds:
                values = category_data[int(category_id)]
                selected = values["scores"] >= score_threshold
                counts = _empty_counts()
                counts["tp"] = int(np.count_nonzero(values["tp"][t] & selected))
                for error in ERROR_TYPES:
                    counts[error] = int(np.count_nonzero(
                        values["fp"][t] & selected & (values["errors"][t] == error)
                    ))
                counts["fp"] = sum(counts[error] for error in ERROR_TYPES)
                counts["fn"] = values["gt_count"] - counts["tp"]
                categories[str(category_id)] = counts
                for name, count in counts.items():
                    overall[name] += count
            result[score_key][f"{iou:.2f}"] = {
                "overall": overall, "categories": categories,
            }
    return result


def _oracle_metrics(official, category_data, original, original_categories):
    i50 = int(np.flatnonzero(np.isclose(official.params.iouThrs, 0.5))[0])
    i75 = int(np.flatnonzero(np.isclose(official.params.iouThrs, 0.75))[0])
    result = {}
    for error in ERROR_TYPES:
        category_precision = {}
        category_metrics = {}
        for category_id in official.params.catIds:
            values = category_data[int(category_id)]
            precision = np.stack([
                _precision_curve(
                    values["tp"][t][values["errors"][t] != error],
                    values["fp"][t][values["errors"][t] != error],
                    values["gt_count"], official.params.recThrs,
                )
                for t in range(len(official.params.iouThrs))
            ])
            category_precision[int(category_id)] = precision
            category_metrics[str(category_id)] = _with_gains(
                _ap_values(precision, i50, i75),
                original_categories[str(category_id)],
            )
        combined = np.stack([
            category_precision[int(category_id)] for category_id in official.params.catIds
        ], axis=2)
        result[error] = {
            "overall": _with_gains(_ap_values(combined, i50, i75), original),
            "categories": category_metrics,
        }
    return result


def analyze_predictions(coco, predictions):
    """Return JSON-safe COCO metrics, error counts, and fixed-match FP oracles."""
    normalized_coco, normalization = _normalized_ground_truth(coco)
    if normalization["applied"]:
        _, legacy_evaluator = _faster_evaluation(coco, predictions)
        legacy_metrics = extract_metrics(legacy_evaluator)
        detections, faster = _faster_evaluation(normalized_coco, predictions)
    else:
        detections, faster = _faster_evaluation(coco, predictions)
        legacy_metrics = None
    official = _official_evaluation(normalized_coco, detections, faster)
    metrics = extract_metrics(faster)
    if legacy_metrics is None:
        legacy_metrics = metrics
    records = _evaluation_records(official)
    geometry = _geometry_cache(normalized_coco, detections, records)
    category_data = _category_data(official, records, geometry)
    precision_validation = _validate_precision(official, faster, category_data)
    original = dict(metrics["size_groups"]["all"])
    original_categories = {
        str(category_id): {
            name: metrics["categories"][str(category_id)]["all"][name]
            for name in ("AP", "AP50", "AP75")
        }
        for category_id in official.params.catIds
    }
    categories = {
        str(category_id): {
            "gt_count": category_data[int(category_id)]["gt_count"],
            **metrics["categories"][str(category_id)]["all"],
        }
        for category_id in official.params.catIds
    }
    return {
        "units": "AP/AR in [0,1]; gains in percentage points; counts are integers",
        "max_detections": 100,
        "iou_thresholds": [float(value) for value in official.params.iouThrs],
        "score_thresholds": list(SCORE_THRESHOLDS),
        "metrics": metrics,
        "legacy_metrics": legacy_metrics,
        "annotation_id_normalization": normalization,
        "iou_metrics": _per_iou_metrics(faster),
        "original": original,
        "categories": categories,
        "precision_validation": precision_validation,
        "counts": _counts(official, category_data),
        "oracle": _oracle_metrics(
            official, category_data, original, original_categories,
        ),
        "oracle_definition": (
            "Each result independently removes one false-positive type from the fixed "
            "official score order and matching; maxDets truncation, true positives, and "
            "the ground-truth denominator remain fixed. Gains are not additive."
        ),
    }
