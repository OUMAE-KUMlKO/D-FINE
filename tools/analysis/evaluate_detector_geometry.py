"""Evaluate independent detectors with the historical GT-strip geometry metric."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_ema_model(config, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Training validates EMA, so require those exact weights rather than silently
    # falling back to the instantaneous training model.
    weights = checkpoint["ema"]["module"]
    model = config.model
    model.load_state_dict(weights, strict=True)
    if getattr(model, "requires_gt", False):
        raise ValueError("This evaluator expects GT-free detector inference")
    metadata = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_last_epoch": checkpoint.get("last_epoch"),
        "weights": "ema.module",
        "model_class": type(model).__name__,
    }
    return model.to(device).eval(), metadata


@torch.inference_mode()
def run(args):
    from src.core import YAMLConfig
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner
    from src.zoo.dfine.matcher import HungarianMatcher
    from src.zoo.dfine.pred_refinement_criterion import PredRefinementCriterion
    from tools.analysis.detector_geometry_metrics import DetectorGeometryComparison

    output_file = Path(args.output_file)
    if output_file.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_file}")
    cache_file = Path(args.prediction_cache) if args.prediction_cache else None
    if cache_file is not None and cache_file.exists():
        raise FileExistsError(f"Prediction cache already exists: {cache_file}")
    prediction_records = []
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    baseline_config = YAMLConfig(args.baseline_config, HGNetv2={"pretrained": False})
    baseline, baseline_metadata = load_ema_model(
        baseline_config, args.baseline_checkpoint, args.device
    )
    model_config = YAMLConfig(
        args.model_config,
        HGNetv2={"pretrained": False},
        val_dataloader={"total_batch_size": args.batch_size, "num_workers": args.workers},
    )
    model, model_metadata = load_ema_model(model_config, args.model_checkpoint, args.device)
    loader = model_config.val_dataloader
    # These are the historical oracle diagnostic's matching costs, fixed for
    # both independently trained models, irrespective of their training losses.
    matching_config = {
        "weight_dict": {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        "use_focal_loss": True, "alpha": 0.25, "gamma": 2.0,
    }
    matching = PredRefinementCriterion(HungarianMatcher(**matching_config)).to(args.device).eval()
    metrics_factory = DetectorGeometryComparison
    if args.detailed_diagnostics:
        from tools.analysis.categorized_geometry_metrics import CategorizedGeometryComparison
        metrics_factory = CategorizedGeometryComparison
    metrics = metrics_factory()
    geometry_matching_config = {
        **matching_config,
        "weight_dict": {"cost_class": 0, "cost_bbox": 5, "cost_giou": 2},
    }
    geometry_matching = (
        PredRefinementCriterion(HungarianMatcher(**geometry_matching_config)).to(args.device).eval()
        if args.detailed_diagnostics else None
    )
    geometry_only_metrics = metrics_factory() if args.detailed_diagnostics else None
    images_seen = ignored_targets = 0
    input_sizes = set()
    started = time.monotonic()
    for batch_index, (images, targets) in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images = images.to(args.device)
        # No GT is passed to either detector. GT is used only for measurement.
        baseline_outputs = baseline(images)
        model_outputs = model(images)
        if baseline_outputs["pred_logits"].shape[-1] != model_outputs["pred_logits"].shape[-1]:
            raise ValueError("Both models must use the same class mapping")
        canonical_targets, ignored = DFINEGTRefiner._prepare_targets(
            targets, images.shape[-2:], images.device, "xyxy_absolute",
            model_outputs["pred_logits"].shape[-1],
        )
        indices = [matching._match_baseline(
            outputs["pred_boxes"], outputs["pred_logits"], canonical_targets
        ) for outputs in (baseline_outputs, model_outputs)]
        metrics.update(baseline_outputs, model_outputs, canonical_targets,
                       *indices, images.shape[-2:])
        if geometry_matching is not None:
            geometry_indices = [geometry_matching._match_baseline(
                outputs["pred_boxes"], outputs["pred_logits"], canonical_targets
            ) for outputs in (baseline_outputs, model_outputs)]
            geometry_only_metrics.update(
                baseline_outputs, model_outputs, canonical_targets,
                *geometry_indices, images.shape[-2:],
            )
        if cache_file is not None:
            for image_index, (original_target, target) in enumerate(zip(targets, canonical_targets)):
                # Preserve original COCO annotation areas, not resized box areas.
                # Additional canonical filtering would require an explicit remap.
                if ignored[image_index] or len(original_target["area"]) != len(target["boxes"]):
                    raise ValueError("Prediction cache requires unchanged canonical GT ordering")
                original_wh = original_target["orig_size"].tolist()
                record = {
                    "image_id": int(original_target["image_id"].item()),
                    "input_size": list(images.shape[-2:]),
                    "orig_size": [int(original_wh[1]), int(original_wh[0])],
                    "target": {
                        "boxes": target["boxes"].detach().cpu(),
                        "labels": target["labels"].detach().cpu(),
                        "area_original": original_target["area"].detach().cpu(),
                    },
                    "matches": {},
                }
                for name, outputs in (("baseline", baseline_outputs), ("model", model_outputs)):
                    record[name] = {
                        key: outputs[key][image_index].detach().float().cpu()
                        for key in ("pred_boxes", "pred_logits")
                    }
                for match_name, matched in (("class_aware", indices), ("geometry_only", geometry_indices)):
                    record["matches"][match_name] = {
                        name: tuple(value.detach().cpu() for value in model_indices[image_index])
                        for name, model_indices in zip(("baseline", "model"), matched)
                    }
                prediction_records.append(record)
        images_seen += len(images)
        ignored_targets += sum(ignored)
        input_sizes.add(tuple(images.shape[-2:]))
        if batch_index % 25 == 0 or images_seen == len(loader.dataset):
            print(f"Geometry: {images_seen}/{len(loader.dataset)} images; "
                  f"{time.monotonic() - started:.1f}s", flush=True)

    if images_seen == 0:
        raise ValueError("No validation images were processed")
    measured = metrics.compute()
    summary = {
        "baseline": {**baseline_metadata, "config": str(Path(args.baseline_config).resolve())},
        "model": {**model_metadata, "config": str(Path(args.model_config).resolve())},
        "dataset": model_config.yaml_cfg["val_dataloader"]["dataset"],
        "images": images_seen,
        "full_validation": images_seen == len(loader.dataset),
        "input_sizes_hw": sorted(input_sizes),
        "ignored_targets": ignored_targets,
        "elapsed_seconds": time.monotonic() - started,
        "inference_uses_gt": False,
        "inference_precision": "float32",
        "matching": matching_config,
        "score_threshold": None,
        "iou_threshold": None,
        "clip_boxes": False,
        "geometry": measured["overall"] if args.detailed_diagnostics else measured,
    }
    if args.detailed_diagnostics:
        summary["geometry_by_category"] = measured["by_category"]
        summary["geometry_only_matching"] = {
            "matching": geometry_matching_config,
            "interpretation": "GT-based diagnostic of all raw query boxes, ignoring predicted class and confidence; not detection AP or deployable selection.",
            **geometry_only_metrics.compute(),
        }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema_version": 1,
            "orig_size_order": "height,width",
            "input_size_order": "height,width",
            "baseline": summary["baseline"],
            "model": summary["model"],
            "dataset": summary["dataset"],
            "full_validation": summary["full_validation"],
            "records": prediction_records,
        }, cache_file)
        summary["prediction_cache"] = str(cache_file.resolve())
        print(f"Saved prediction cache: {cache_file}", flush=True)
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for name, result in summary["geometry"]["models"].items():
        print(f"{name}: {json.dumps(result['groups']['all'])}", flush=True)
    print(f"Saved geometry: {output_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default="configs/dfine/dfine_hgnetv2_l_uavswarm.yml")
    parser.add_argument("--model-config", default="configs/dfine/dfine_hgnetv2_l_uavswarm_all_layer_band2d_query_gate_full.yml")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--detailed-diagnostics", action="store_true",
                        help="also group geometry by category and repeat matching without classification cost")
    parser.add_argument("--prediction-cache", help="save raw outputs, targets and both matchings for offline grouping")
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.workers < 0:
        parser.error("batch size/threads must be positive, and workers nonnegative")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("max-batches must be positive")
    if args.prediction_cache and not args.detailed_diagnostics:
        parser.error("prediction-cache requires detailed-diagnostics to retain both matchings")
    run(args)


if __name__ == "__main__":
    main()
