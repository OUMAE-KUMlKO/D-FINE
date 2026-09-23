"""Check fresh P2 integration and candidate coverage without a training run."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def run(args):
    from src.core import YAMLConfig
    from src.solver.distribution_coverage import DistributionCoverageMetrics
    from src.zoo.dfine.gt_refinement import DFINEGTRefiner

    output_file = Path(args.output_file)
    if output_file.exists():
        raise FileExistsError(f"Output already exists: {output_file}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    config = YAMLConfig(
        args.config,
        DFINEPredRefiner={"baseline_checkpoint": args.baseline_checkpoint},
        val_dataloader={"total_batch_size": args.batch_size, "num_workers": args.workers},
    )
    model = config.model.to(args.device).eval()
    if model.refiner.sampling_mode != "distribution":
        raise ValueError("This check requires the P2 distribution sampling mode")
    criterion = config.criterion.to(args.device).eval()
    loader = config.val_dataloader
    coverage = DistributionCoverageMetrics(model.refiner.offset_pixels)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    images_seen, ignored_targets = 0, 0
    max_decode_error = 0.0
    first_batch = None
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            images = images.to(device)
            # No targets enter the model; GT is used only by the diagnostic.
            outputs = model(images)
            torch.testing.assert_close(outputs["pred_boxes"], outputs["baseline_boxes"], atol=0, rtol=0)
            diag = outputs["sampling_diagnostics"]
            edges = (diag["candidate_positions"] * diag["candidate_probabilities"]).sum(-1)
            decoded = torch.cat(((edges[..., :2] + edges[..., 2:]) / 2, edges[..., 2:] - edges[..., :2]), -1)
            torch.testing.assert_close(decoded, outputs["baseline_boxes"], atol=1e-6, rtol=1e-5)
            max_decode_error = max(max_decode_error, (decoded - outputs["baseline_boxes"]).abs().max().item())
            canonical, ignored = DFINEGTRefiner._prepare_targets(
                targets, images.shape[-2:], device, "xyxy_absolute", outputs["pred_logits"].shape[-1],
            )
            indices = criterion._match_baseline(outputs["baseline_boxes"], outputs["pred_logits"], canonical)
            coverage.update(diag, outputs["baseline_boxes"], canonical, indices, images.shape[-2:])
            ignored_targets += sum(ignored)
            images_seen += len(images)
            if first_batch is None and args.check_backward:
                first_batch = (images, canonical)
            if batch_index % 50 == 0:
                print(f"Candidate coverage: {images_seen}/{len(loader.dataset)} images", flush=True)
    if not images_seen:
        raise ValueError("No validation images were processed")

    # Two disposable optimizer steps validate AMP and gradients beyond the
    # zero-initialized final layer. These weights are never saved.
    step_losses = []
    if args.check_backward:
        model.train()
        optimizer = config.optimizer
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        images, targets = first_batch
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
            # Match det_engine.py: autocast for the model, FP32 for the loss.
            with torch.autocast(device_type=device.type, enabled=False):
                loss = sum(criterion(outputs, targets).values())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite P2 loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, parameter in model.named_parameters():
                if name.startswith("refiner."):
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise ValueError(f"Missing or nonfinite refiner gradient: {name}")
                elif parameter.requires_grad or parameter.grad is not None:
                    raise ValueError(f"Frozen detector received gradients: {name}")
            scaler.step(optimizer)
            scaler.update()
            step_losses.append(float(loss.detach()))
        print(f"Disposable AMP/backward checks passed: losses={step_losses}", flush=True)

    summary = {
        "config": str(Path(args.config).resolve()),
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
        "images": images_seen,
        "full_validation": images_seen == len(loader.dataset),
        "ignored_targets": ignored_targets,
        "inference_uses_gt": False,
        "zero_initialization_preserves_boxes": True,
        "max_distribution_decode_error": max_decode_error,
        "disposable_train_step_losses": step_losses,
        "batch_size": args.batch_size,
        "query_chunk_size": model.refiner.query_chunk_size,
        "candidate_count_per_edge": int(diag["candidate_positions"].shape[-1]),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "coverage": coverage.compute(),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Saved P2 check: {output_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_pred_p2.yml")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=1, help="Default 1 for a short check; 0 processes the full validation set")
    parser.add_argument("--check-backward", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or min(args.workers, args.max_batches) < 0:
        parser.error("batch size/threads must be positive; workers/max-batches must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
