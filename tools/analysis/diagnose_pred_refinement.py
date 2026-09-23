"""Evaluate P1/P2 geometry, candidate coverage, and GT-selected diagnostics."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_diagnostic_variants(outputs, indices, refinement_name="p1"):
    """Keep scores fixed; only the oracle variant uses GT-derived selection."""
    baseline, refined = outputs["baseline_boxes"], outputs["pred_boxes"]
    matched = torch.zeros(baseline.shape[:2], dtype=torch.bool, device=baseline.device)
    for batch_index, (query_ids, _) in enumerate(indices):
        matched[batch_index, query_ids] = True
    return {
        name: {"pred_boxes": boxes, "pred_logits": outputs["pred_logits"]}
        for name, boxes in (
            ("baseline", baseline),
            (refinement_name, refined),
            ("matched_only_oracle", torch.where(matched[..., None], refined, baseline)),
        )
    }


@torch.inference_mode()
def run(args):
    from src.core import YAMLConfig
    from src.data import CocoEvaluator, get_coco_api_from_dataset
    from src.solver.distribution_coverage import DistributionCoverageMetrics
    from src.solver.pred_refinement_diagnostics import PredRefinementDiagnostics
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner

    output_file = Path(args.output_file)
    if output_file.exists():
        raise FileExistsError(f"Diagnostic output already exists: {output_file}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    config = YAMLConfig(
        args.config,
        DFINEPredRefiner={"baseline_checkpoint": None},
        val_dataloader={"total_batch_size": args.batch_size, "num_workers": args.workers},
    )
    if config.yaml_cfg["model"] != "DFINEPredRefiner" or not config.yaml_cfg["use_focal_loss"]:
        raise ValueError("This diagnostic expects a prediction refiner with sigmoid classification scores")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = config.model
    model.load_state_dict(checkpoint["model"])
    checkpoint_epoch = checkpoint.get("last_epoch")
    del checkpoint
    model.to(args.device).eval()
    refinement_name = "p2" if model.refiner.sampling_mode == "distribution" else "p1"
    coverage = (
        DistributionCoverageMetrics(offset_pixels=model.refiner.offset_pixels)
        if refinement_name == "p2" else None
    )
    criterion = config.criterion.to(args.device).eval()
    postprocessor = config.postprocessor.to(args.device).eval()
    loader = config.val_dataloader
    coco_gt = get_coco_api_from_dataset(loader.dataset)
    evaluators = {
        name: CocoEvaluator(coco_gt=coco_gt, iou_types=["bbox"])
        for name in ("baseline", refinement_name, "matched_only_oracle")
    }
    diagnostics = PredRefinementDiagnostics(offset_pixels=model.refiner.offset_pixels)
    images_seen = 0
    ignored_targets = 0
    started = time.monotonic()
    for batch_index, (images, targets) in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        images = images.to(args.device)
        # This is the only model call: no GT is passed to the predictor.
        outputs = model(images)
        canonical_targets, ignored = DFINEGTRefiner._prepare_targets(
            targets, images.shape[-2:], images.device, "xyxy_absolute",
            outputs["pred_logits"].shape[-1],
        )
        indices = criterion._match_baseline(
            outputs["baseline_boxes"], outputs["pred_logits"], canonical_targets
        )
        diagnostics.update(outputs, canonical_targets, indices, images.shape[-2:])
        if coverage is not None:
            coverage.update(
                outputs["sampling_diagnostics"], outputs["baseline_boxes"],
                canonical_targets, indices, images.shape[-2:],
            )
        ignored_targets += sum(ignored)
        sizes = torch.stack([target["orig_size"] for target in targets]).to(images.device)
        for name, variant in build_diagnostic_variants(outputs, indices, refinement_name).items():
            predictions = postprocessor(variant, sizes)
            evaluators[name].update({
                int(target["image_id"].item()): prediction
                for target, prediction in zip(targets, predictions)
            })
        images_seen += len(images)
        if batch_index % 50 == 0:
            print(f"Diagnostic: {images_seen}/{len(loader.dataset)} images", flush=True)

    if images_seen == 0:
        raise ValueError("No validation images were processed")
    coco_metrics = {}
    for name, evaluator in evaluators.items():
        print(f"COCO variant: {name}", flush=True)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        coco_metrics[name] = [float(value) for value in evaluator.coco_eval["bbox"].stats]

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "config": str(Path(args.config).resolve()),
        "images": images_seen,
        "full_validation": images_seen == len(loader.dataset),
        "ignored_targets": ignored_targets,
        "elapsed_seconds": time.monotonic() - started,
        "inference_uses_gt": False,
        "sampling_mode": model.refiner.sampling_mode,
        "diagnostic_notes": [
            "All three COCO variants reuse the same model outputs and unchanged scores.",
            "matched_only_oracle uses baseline Hungarian GT matches to select refinements; it is not a deployment result.",
            "Unmatched queries are not synonymous with false positives.",
            "Normal-band coverage measures sampling coordinates, not the feature receptive field.",
        ],
        "coco_metric_order": [
            "AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
            "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge",
        ],
        "coco_eval_bbox": coco_metrics,
        "diagnostics": diagnostics.compute(),
    }
    if coverage is not None:
        summary["distribution_coverage"] = coverage.compute()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Saved diagnostic: {output_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_pred_p1.yml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.workers < 0:
        parser.error("batch size/threads must be positive, and workers nonnegative")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("max-batches must be positive")
    run(args)


if __name__ == "__main__":
    main()
