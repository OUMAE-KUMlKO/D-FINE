#!/usr/bin/env python3
"""Disposable preflight for the UAVSwarm G5 DDF on/off ablation.

This script never saves a checkpoint.  It deliberately builds a batch-one,
zero-worker loader while leaving the configured augmentation pipeline intact.
The formal run remains an independent 8-GPU, global-batch-64 process.
"""

import argparse
import copy
import json
import random
import sys
import traceback
import gc
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core import YAMLConfig  # noqa: E402
from src.core.yaml_utils import load_config  # noqa: E402


ON_CONFIG = REPO / "configs/dfine/ablation/dfine_hgnetv2_l_uavswarm_g5_ddf_on.yml"
OFF_CONFIG = REPO / "configs/dfine/ablation/dfine_hgnetv2_l_uavswarm_g5_ddf_off.yml"
SEED = 0
DDF_KEYS = tuple(f"loss_ddf_aux_{i}" for i in range(5)) + tuple(
    f"loss_ddf_dn_{i}" for i in range(6)
)
STAT_KEYS = tuple(f"ddf_{kind}_{group}_{i}" for kind in ("raw", "weighted")
                  for group, count in (("aux", 5), ("dn", 6)) for i in range(count))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_snapshot():
    return {
        "python": repr(random.getstate()),
        "numpy": repr(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
    }


def rng_equal(a, b):
    return (a["python"] == b["python"] and a["numpy"] == b["numpy"]
            and torch.equal(a["torch"], b["torch"]))


def json_annotation_summary(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "image_ids": {item["id"] for item in data["images"]},
        "categories": data.get("categories", []),
    }


def without_allowed_differences(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop("__include__", None)
    cfg.pop("output_dir", None)
    cfg.pop("exp_name", None)
    cfg["DFINECriterion"]["weight_dict"].pop("loss_ddf", None)
    return cfg


def static_checks(report, seed=SEED):
    on = load_config(str(ON_CONFIG), {})
    off = load_config(str(OFF_CONFIG), {})
    require(without_allowed_differences(on) == without_allowed_differences(off),
            "effective configs differ beyond DDF weight, output_dir, and exp_name")
    weights_on = on["DFINECriterion"]["weight_dict"]
    weights_off = off["DFINECriterion"]["weight_dict"]
    require(weights_on["loss_ddf"] == 1.5 and weights_off["loss_ddf"] == 0.0,
            "unexpected DDF weights")
    require(weights_on["loss_fgl"] == weights_off["loss_fgl"] == 0.15,
            "FGL must remain enabled at 0.15")
    require(on["DFINECriterion"].get("log_ddf_diagnostics") is True
            and off["DFINECriterion"].get("log_ddf_diagnostics") is True,
            "DDF diagnostics must be enabled")
    require(on["epochs"] == 80, "formal configuration must run 80 epochs")
    require(on["train_dataloader"]["total_batch_size"] == 64,
            "formal global train batch must be 64")
    require(on["val_dataloader"]["total_batch_size"] == 64,
            "formal global validation batch must be 64")
    require(on.get("use_amp") and on.get("use_ema") and on.get("sync_bn"),
            "formal AMP, EMA, and SyncBN must remain enabled")
    require(on["DFINEAllLayerStrip"].get("checkpoint_strips") is True,
            "G5 strip gradient checkpointing must remain enabled")
    train_path = Path(on["train_dataloader"]["dataset"]["ann_file"])
    val_path = Path(on["val_dataloader"]["dataset"]["ann_file"])
    train = json_annotation_summary(train_path)
    val = json_annotation_summary(val_path)
    require((train["images"], train["annotations"]) == (3450, 25396),
            "unexpected full train annotation counts")
    require((val["images"], val["annotations"]) == (3394, 24265),
            "unexpected full validation annotation counts")
    require(train["image_ids"].isdisjoint(val["image_ids"]),
            "train and validation image IDs overlap")
    require(train["categories"] == val["categories"], "category metadata differs")
    require(len(train["categories"]) == 1 and train["categories"][0]["id"] == 1
            and train["categories"][0]["name"] == "pedestrian",
            "expected the single pedestrian category with category_id=1")
    pretrain = REPO / on["HGNetv2"]["local_model_dir"] / "PPHGNetV2_B4_stage1.pth"
    require(pretrain.is_file(), f"missing local backbone pretraining: {pretrain}")
    report["static"] = {
        "effective_configs_equal_except": ["DFINECriterion.weight_dict.loss_ddf",
                                             "output_dir", "exp_name"],
        "formal_scope": {"epochs": 80, "gpus": 8, "global_batch": 64, "seed": seed},
        "annotations": {
            "train": {"images": train["images"], "annotations": train["annotations"]},
            "val": {"images": val["images"], "annotations": val["annotations"]},
            "image_ids_disjoint": True,
        },
        "backbone_pretrain": str(pretrain),
        "gradient_checkpointing_enabled": True,
    }
    return on, off


def cpu_initialization_checks(report, seed=SEED):
    seed_all(seed)
    on_cfg = YAMLConfig(str(ON_CONFIG))
    on_model = on_cfg.model.cpu()
    on_state = {key: value.detach().clone() for key, value in on_model.state_dict().items()}
    on_rng = rng_snapshot()
    del on_model, on_cfg
    gc.collect()

    seed_all(seed)
    off_cfg = YAMLConfig(str(OFF_CONFIG))
    off_model = off_cfg.model.cpu()
    off_state = off_model.state_dict()
    require(on_state.keys() == off_state.keys(), "seeded CPU model state keys differ")
    unequal = [key for key in on_state if not torch.equal(on_state[key], off_state[key])]
    require(not unequal, f"seeded CPU model tensors differ: {unequal[:3]}")
    off_rng = rng_snapshot()
    require(rng_equal(on_rng, off_rng),
            "seeded CPU model construction consumes different RNG")
    report["cpu_initialization"] = {
        "seed": seed, "state_tensors_compared": len(on_state),
        "state_equal": True, "post_construction_rng_equal": True,
        "models_constructed": 2,
    }
    del off_model, off_cfg, off_state, on_state
    gc.collect()


def move_targets(targets, device):
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in target.items()} for target in targets]


def scalar_map(values):
    return {key: float(value.detach().cpu()) for key, value in values.items()}


def compare_non_ddf(on_losses, off_losses):
    on = {k: v for k, v in on_losses.items() if "ddf" not in k}
    off = {k: v for k, v in off_losses.items() if "ddf" not in k}
    require(on.keys() == off.keys(), "non-DDF loss keys differ")
    for key in on:
        require(torch.equal(on[key], off[key]), f"non-DDF loss differs: {key}")


def gradients(loss, params, retain_graph):
    return torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)


def gradient_difference_metrics(left_grads, right_grads, named_params):
    per_parameter = []
    diff_sq = 0.0
    reference_sq = 0.0
    nonzero_grad_parameters = 0
    presence_mismatches = []
    for (name, _), left, right in zip(named_params, left_grads, right_grads):
        if (left is None) != (right is None):
            presence_mismatches.append(name)
            continue
        if left is None:
            continue
        require(torch.isfinite(left).all().item() and torch.isfinite(right).all().item(),
                f"non-finite gradient: {name}")
        difference = left - right
        parameter_diff_sq = float(difference.double().square().sum().detach().cpu())
        parameter_reference_sq = float(right.double().square().sum().detach().cpu())
        parameter_diff_norm = parameter_diff_sq ** 0.5
        parameter_reference_norm = parameter_reference_sq ** 0.5
        max_abs = float(difference.abs().max().detach().cpu())
        per_parameter.append({
            "name": name,
            "max_abs_difference": max_abs,
            "relative_l2_difference": parameter_diff_norm / (parameter_reference_norm + 1e-6),
        })
        diff_sq += parameter_diff_sq
        reference_sq += parameter_reference_sq
        nonzero_grad_parameters += int(torch.count_nonzero(right).item() > 0)
    diff_norm = diff_sq ** 0.5
    reference_norm = reference_sq ** 0.5
    return {
        "difference_l2": diff_norm,
        "reference_l2": reference_norm,
        "relative_l2_difference": diff_norm / (reference_norm + 1e-6),
        "max_abs_difference": max(
            (item["max_abs_difference"] for item in per_parameter), default=0.0
        ),
        "presence_mismatches": presence_mismatches,
        "nonzero_gradient_parameters": nonzero_grad_parameters,
        "per_parameter": per_parameter,
    }


def compare_zero_ddf_gradients(off_losses, named_params, diagnostics):
    params = [parameter for _, parameter in named_params]
    ddf_total = sum(value for key, value in off_losses.items() if "loss_ddf" in key)
    off_total = sum(off_losses.values())
    removed_total = sum(value for key, value in off_losses.items() if "loss_ddf" not in key)
    ddf_grads = gradients(ddf_total, params, retain_graph=True)
    ddf_nonzero = []
    ddf_max_abs = 0.0
    for (name, _), gradient in zip(named_params, ddf_grads):
        if gradient is not None:
            require(torch.isfinite(gradient).all().item(), f"non-finite zero-weight DDF gradient: {name}")
            maximum = float(gradient.abs().max().detach().cpu())
            ddf_max_abs = max(ddf_max_abs, maximum)
            if torch.count_nonzero(gradient).item() != 0:
                ddf_nonzero.append(name)
    diagnostics["direct_zero_weight_ddf_gradient"] = {
        "exactly_zero_for_all_parameters": not ddf_nonzero,
        "max_abs": ddf_max_abs,
        "nonzero_parameter_names": ddf_nonzero,
    }
    require(not ddf_nonzero, "direct zero-weight DDF gradient is nonzero")

    grads_with_zero = gradients(off_total, params, retain_graph=True)
    grads_removed = gradients(removed_total, params, retain_graph=True)
    comparison = gradient_difference_metrics(grads_with_zero, grads_removed, named_params)
    diagnostics["zero_weight_included_vs_removed"] = comparison
    require(not comparison["presence_mismatches"], "gradient presence differs")
    require(comparison["nonzero_gradient_parameters"] > 0,
            "ordinary losses produced no parameter gradients")
    primary_limit = 1e-6 + 1e-4 * comparison["reference_l2"]
    diagnostics["compatibility_primary_limit_l2"] = primary_limit
    compatible = comparison["difference_l2"] <= primary_limit
    diagnostics["compatibility_basis"] = "relative_l2_with_absolute_floor"

    if not compatible:
        repeated_removed = gradients(removed_total, params, retain_graph=False)
        noise = gradient_difference_metrics(grads_removed, repeated_removed, named_params)
        diagnostics["removed_vs_repeated_removed_noise_control"] = noise
        require(not noise["presence_mismatches"], "noise-control gradient presence differs")
        noise_limit = 1e-6 + 2.0 * noise["difference_l2"]
        diagnostics["compatibility_noise_control_limit_l2"] = noise_limit
        compatible = comparison["difference_l2"] <= noise_limit
        diagnostics["compatibility_basis"] = "within_twice_repeated_backward_noise"
    diagnostics["compatible"] = compatible
    require(compatible, "zero-weight-included gradient exceeds relative-L2 and noise-control limits")
    return comparison["max_abs_difference"], comparison["nonzero_gradient_parameters"]


def gpu_checks(report, device, seed=SEED):
    require(device.type == "cuda", "--device cuda is required for this preflight")
    require(torch.cuda.is_available(), "CUDA is not available")

    # The override affects only this disposable loader.  Its dataset transforms and
    # collate augmentation are inherited unchanged from the actual on configuration.
    seed_all(seed)
    cfg = YAMLConfig(str(ON_CONFIG), train_dataloader={"total_batch_size": 1, "num_workers": 0})
    loader = cfg.train_dataloader
    require(loader.batch_size == 1 and loader.num_workers == 0,
            "disposable loader override did not take effect")
    samples, targets = next(iter(loader))
    samples, targets = samples.to(device), move_targets(targets, device)

    seed_all(seed)
    model = cfg.model.to(device).train()
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    forward_rng = torch.cuda.get_rng_state(device).clone()
    model.load_state_dict(initial_state, strict=True)
    del initial_state
    torch.cuda.set_rng_state(forward_rng, device)
    outputs = model(samples, targets=targets)
    for key in ("pred_boxes", "pred_logits", "pred_corners"):
        require(torch.isfinite(outputs[key]).all().item(), f"non-finite primary output: {key}")

    on_criterion = cfg.criterion.to(device).train()
    off_cfg = YAMLConfig(str(OFF_CONFIG))
    off_criterion = off_cfg.criterion.to(device).train()
    on_losses = on_criterion(outputs, targets, epoch=0, step=0, global_step=0,
                             epoch_step=len(loader))
    on_stats = dict(on_criterion.last_ddf_stats)
    off_losses = off_criterion(outputs, targets, epoch=0, step=0, global_step=0,
                               epoch_step=len(loader))
    off_stats = dict(off_criterion.last_ddf_stats)
    compare_non_ddf(on_losses, off_losses)
    require(all(torch.isfinite(value).item() for value in on_losses.values()),
            "enabled criterion produced a non-finite loss")
    require(all(torch.isfinite(value).item() for value in off_losses.values()),
            "disabled criterion produced a non-finite loss")
    require(all(any(key.startswith(prefix) for key in off_losses)
                for prefix in ("loss_vfl", "loss_bbox", "loss_giou", "loss_fgl")),
            "off loss is missing VFL, box, GIoU, or FGL supervision")
    require(all(key in on_losses and key in off_losses for key in DDF_KEYS),
            "ordinary aux_0..4 or dn_0..5 DDF losses are missing")
    require(all(key in on_stats and key in off_stats for key in STAT_KEYS),
            "ordinary aux_0..4 or dn_0..5 DDF diagnostics are missing")
    require(all(not value.requires_grad for stats in (on_stats, off_stats)
                for value in stats.values()), "DDF diagnostics must be detached")
    for key in STAT_KEYS:
        require(torch.equal(on_stats[key], off_stats[key]) if "raw" in key else True,
                f"raw DDF diagnostic differs: {key}")
    raw_keys = [key for key in STAT_KEYS if "_raw_" in key]
    for key in raw_keys:
        suffix = key.removeprefix("ddf_raw_")
        require(torch.allclose(on_stats[f"ddf_weighted_{suffix}"], on_stats[key] * 1.5,
                               rtol=1e-6, atol=1e-10), f"on weighted/raw relation failed: {suffix}")
        require(torch.count_nonzero(off_stats[f"ddf_weighted_{suffix}"]).item() == 0,
                f"off weighted DDF is nonzero: {suffix}")
    initial_raw_nonzero = any(torch.count_nonzero(on_stats[key]).item() > 0 for key in raw_keys)
    require("loss_fgl" in on_criterion.weight_dict and on_criterion.weight_dict["loss_fgl"] == 0.15,
            "FGL was removed from the criterion")

    gpu_report = {
        "device": str(device), "real_batch": True, "batch_size": 1, "num_workers": 0,
        "model_instances_on_gpu": 1, "training_mode": True,
        "forward_equivalence_basis": (
            "One cached forward was reused for both actual criteria after CPU seeded-state and "
            "post-construction-RNG equality proved the configs do not change model construction."
        ),
        "non_ddf_losses_equal": True,
        "ddf_keys": list(DDF_KEYS),
        "diagnostic_keys": list(STAT_KEYS),
        "initial_raw_ddf_nonzero": initial_raw_nonzero,
        "initial_ddf_stats": scalar_map(on_stats),
        "checkpoint_saved": False,
    }
    report["gpu_preflight"] = gpu_report

    del outputs, on_losses, off_losses
    torch.cuda.empty_cache()

    # Disposable AMP steps use the configured optimizer and default scaler.  A
    # scale decrease is a skipped step; retry without changing training settings.
    optimizer = cfg.optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    updated = False
    attempts = []
    for attempt in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", cache_enabled=True):
            step_outputs = model(samples, targets=targets)
        with torch.autocast(device_type="cuda", enabled=False):
            step_losses = on_criterion(step_outputs, targets, epoch=0, step=attempt,
                                       global_step=attempt, epoch_step=len(loader))
        loss = sum(step_losses.values())
        require(torch.isfinite(loss).item(), "non-finite AMP loss")
        raw_nonzero = any(torch.count_nonzero(value).item() > 0 for key, value
                          in on_criterion.last_ddf_stats.items() if "ddf_raw_" in key)
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        changed_candidate = next((value for value in model.parameters()
                                  if value.requires_grad and value.grad is not None
                                  and torch.count_nonzero(value.grad).item() > 0), None)
        require(changed_candidate is not None, "AMP backward produced no nonzero gradient")
        before = changed_candidate.detach().cpu().clone()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        changed = not torch.equal(before, changed_candidate.detach().cpu())
        effective = scale_after >= scale_before and changed
        attempts.append({"attempt": attempt + 1, "scale_before": scale_before,
                         "scale_after": scale_after, "parameter_changed": changed,
                         "raw_ddf_nonzero": raw_nonzero, "effective_update": effective})
        if effective:
            updated = True
            break
    gpu_report["amp_attempts"] = attempts
    gpu_report["amp_effective_update"] = updated
    require(updated, "no effective finite AMP update in ten attempts")

    # Zero-initialized strip heads legitimately produce zero DDF before training.
    # Check liveness and the off-vs-removed gradient identity on a fresh forward
    # only after the disposable optimizer update has changed model parameters.
    optimizer.zero_grad(set_to_none=True)
    post_outputs = model(samples, targets=targets)
    post_on_losses = on_criterion(post_outputs, targets, epoch=0, step=len(attempts),
                                  global_step=len(attempts), epoch_step=len(loader))
    post_on_stats = dict(on_criterion.last_ddf_stats)
    post_off_losses = off_criterion(post_outputs, targets, epoch=0, step=len(attempts),
                                    global_step=len(attempts), epoch_step=len(loader))
    post_off_stats = dict(off_criterion.last_ddf_stats)
    compare_non_ddf(post_on_losses, post_off_losses)
    for key in raw_keys:
        suffix = key.removeprefix("ddf_raw_")
        require(torch.equal(post_on_stats[key], post_off_stats[key]),
                f"post-update raw DDF diagnostic differs: {suffix}")
        require(torch.allclose(post_on_stats[f"ddf_weighted_{suffix}"],
                               post_on_stats[key] * 1.5, rtol=1e-6, atol=1e-10),
                f"post-update on weighted/raw relation failed: {suffix}")
        require(torch.count_nonzero(post_off_stats[f"ddf_weighted_{suffix}"]).item() == 0,
                f"post-update off weighted DDF is nonzero: {suffix}")
    post_raw_nonzero = any(torch.count_nonzero(post_on_stats[key]).item() > 0
                           for key in raw_keys)
    gpu_report["post_update_raw_ddf_nonzero"] = post_raw_nonzero
    gpu_report["post_update_ddf_stats"] = scalar_map(post_on_stats)
    require(post_raw_nonzero, "DDF remained zero after an effective disposable update")
    named_params = [(name, parameter) for name, parameter in model.named_parameters()
                    if parameter.requires_grad]
    gradient_diagnostics = {}
    gpu_report["gradient_diagnostics"] = gradient_diagnostics
    max_abs_diff, nonzero_grad_parameters = compare_zero_ddf_gradients(
        post_off_losses, named_params, gradient_diagnostics
    )
    gpu_report["off_zero_ddf_gradient_equals_removed"] = True
    gpu_report["gradient_max_abs_difference"] = max_abs_diff
    gpu_report["nonzero_ordinary_gradient_parameters"] = nonzero_grad_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "scope": "local real-batch GPU preflight; formal training is a separate 8-GPU process",
        "configs": {"on": str(ON_CONFIG), "off": str(OFF_CONFIG)},
    }
    try:
        static_checks(report, seed=args.seed)
        cpu_initialization_checks(report, seed=args.seed)
        gpu_checks(report, torch.device(args.device), seed=args.seed)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        print(f"preflight report: {output} ({report['status']})")


if __name__ == "__main__":
    main()
