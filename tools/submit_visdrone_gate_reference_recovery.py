"""Prepare and submit the approved VisDrone E1-E3 recovery chain."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.submit_controlled_experiments import freeze_source
from tools.submit_visdrone_gate_reference import BASELINE, EXPERIMENTS


PROJECT = Path(__file__).resolve().parents[1]
RESUME_CHECKPOINT = (
    PROJECT / "output/visdrone_gate_reference_20261010/results/visdrone/seed0/"
    "g5_adaptive_band_gate/continue120/last.pth"
)
HISTORICAL_BEST = (
    PROJECT / "output/visdrone_gate_reference_20261010/results/visdrone/seed0/"
    "g5_adaptive_band_gate/continue120/best_stg2.pth"
)
REQUIRED_RESUME_KEYS = (
    "model",
    "optimizer",
    "ema",
    "scaler",
    "lr_scheduler",
    "lr_warmup_scheduler",
)
MODES = ("resume", "fresh")


def validate_checkpoint(path, expected_epoch):
    """Require the complete state needed for an exact training continuation."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing = [key for key in REQUIRED_RESUME_KEYS if key not in checkpoint]
    if missing:
        raise RuntimeError(f"Checkpoint {path} is missing resume state: {', '.join(missing)}")
    if checkpoint.get("last_epoch") != expected_epoch:
        raise RuntimeError(
            f"Checkpoint {path} must be epoch{expected_epoch}, "
            f"got {checkpoint.get('last_epoch')!r}"
        )
    return {
        "path": str(path.resolve()),
        "last_epoch": checkpoint["last_epoch"],
        "required_keys": list(REQUIRED_RESUME_KEYS),
    }


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def prepare(project, run_root, mode):
    """Validate inputs and freeze a new, immutable source tree."""
    _check_mode(mode)
    project = Path(project).resolve()
    run_root = Path(run_root).resolve()
    if run_root.exists():
        raise FileExistsError(run_root)

    for _, group in EXPERIMENTS:
        for suffix in ("", "_continue120"):
            config = project / f"configs/controlled/visdrone_{group}{suffix}.yml"
            if not config.is_file():
                raise FileNotFoundError(config)
    baseline = project / BASELINE
    if not baseline.is_file():
        raise FileNotFoundError(baseline)

    recovery = None
    historical = None
    if mode == "resume":
        recovery = validate_checkpoint(RESUME_CHECKPOINT, expected_epoch=91)
        historical = validate_checkpoint(HISTORICAL_BEST, expected_epoch=89)

    snapshot = freeze_source(
        project, run_root, datasets=("visdrone",), groups=("g5_adaptive_band",)
    )
    extra_sources = (
        "tools/submit_visdrone_gate_reference.py",
        "tools/submit_visdrone_gate_reference_recovery.py",
        "slurm_files/run_resume_dfine_controlled.slurm",
        "slurm_files/run_train_visdrone_gate_reference.slurm",
        "slurm_files/run_resume_visdrone_gate_reference.slurm",
        "docs/research/visdrone_g5_gate_reference_plan_20261010.md",
    )
    for relative in extra_sources:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / relative, destination)

    selection = {
        "mode": mode,
        "dataset": "visdrone",
        "seed": 0,
        "batch_size": 32,
        "gpus": 8,
        "experiments": dict(EXPERIMENTS),
        "E0_reused_checkpoint": str(baseline),
        "E0_retrained": False,
        "epochs": {
            "E1": [92, 119] if mode == "resume" else [0, 119],
            "E2": [0, 119],
            "E3": [0, 119],
        },
        "stage2": {
            "milestones": [100],
            "resume_lr_schedule": True,
            "stage2_rollback": False,
        },
        "prediction_safety": {
            "implementation": "src/zoo/dfine/prediction_safety.py",
            "scope": "matching queries only; DN queries retain the native path",
            "version": "exact frozen snapshot recorded by git_head.txt and working_changes.patch",
        },
        "automatic_retry": False,
        "failure_policy": "no automatic retry; fresh fallback requires a new explicit fresh invocation",
        "resume_checkpoint": recovery,
        "historical_true_best": historical,
        "final_E1_selection": (
            "compare archive/pre_resume_best_stg2.pth with the new continue120/best_stg2.pth"
            if mode == "resume" else None
        ),
    }
    (run_root / "experiment_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    return snapshot


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def submit_chain(snapshot, run_root, mode):
    """Submit E1, E2 and E3 serially; never retry failed submissions."""
    _check_mode(mode)
    snapshot = Path(snapshot).resolve()
    run_root = Path(run_root).resolve()
    manifest_path = run_root / "jobs.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    run_root.mkdir(parents=True, exist_ok=True)
    logs = run_root / "logs"
    logs.mkdir(exist_ok=False)
    manifest = {
        "mode": mode,
        "status": "submitting",
        "automatic_retry": False,
        "jobs": [],
    }
    _write_manifest(manifest_path, manifest)

    previous = None
    for experiment, group in EXPERIMENTS:
        job_mode = "resume" if mode == "resume" and experiment == "E1" else "fresh"
        name = f"visdrone_{experiment.lower()}_gate_ref_recovery_s0"
        config80 = f"configs/controlled/visdrone_{group}.yml"
        config120 = f"configs/controlled/visdrone_{group}_continue120.yml"
        output = run_root / "results" / "visdrone" / "seed0" / group
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={name}",
            f"--output={logs / (name + '_%j.out')}",
            f"--error={logs / (name + '_%j.err')}",
            f"--chdir={snapshot}",
            "--kill-on-invalid-dep=yes",
        ]
        if previous:
            command.append(f"--dependency=afterok:{previous}")
        if job_mode == "resume":
            command += [
                str(snapshot / "slurm_files/run_resume_visdrone_gate_reference.slurm"),
                str(snapshot),
                config120,
                str(output),
                str(RESUME_CHECKPOINT),
                str(HISTORICAL_BEST),
                "0",
            ]
        else:
            command += [
                str(snapshot / "slurm_files/run_train_visdrone_gate_reference.slurm"),
                str(snapshot),
                config80,
                config120,
                str(output),
                "0",
            ]

        try:
            response = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            job_id = response.stdout.strip().split(";")[0]
            if not job_id.isdecimal():
                raise RuntimeError(f"Unexpected sbatch response: {response.stdout!r}")
        except Exception:
            manifest["status"] = "submission_failed"
            manifest["failed_experiment"] = experiment
            _write_manifest(manifest_path, manifest)
            raise

        job = {
            "job_id": job_id,
            "experiment": experiment,
            "group": group,
            "mode": job_mode,
            "seed": 0,
            "afterok": previous,
            "config120": config120,
            "output_dir": str(output),
            "stdout": str(logs / f"{name}_{job_id}.out"),
            "stderr": str(logs / f"{name}_{job_id}.err"),
        }
        if job_mode == "fresh":
            job["config80"] = config80
        else:
            job["resume_checkpoint"] = str(RESUME_CHECKPOINT)
            job["historical_true_best"] = str(HISTORICAL_BEST)
        manifest["jobs"].append(job)
        _write_manifest(manifest_path, manifest)
        print(
            f"{job_id}: {experiment}/{group} ({job_mode}); afterok={previous or 'none'}",
            flush=True,
        )
        previous = job_id

    manifest["status"] = "submitted"
    _write_manifest(manifest_path, manifest)
    return manifest["jobs"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    run_root = (
        args.run_root
        or PROJECT
        / "output"
        / datetime.now().strftime("visdrone_gate_reference_recovery_%Y%m%d_%H%M%S")
    ).resolve()
    print(f"Mode: {args.mode}; chain: E1 -> E2 -> E3")
    print(f"Run root: {run_root}")
    if args.submit:
        snapshot = prepare(PROJECT, run_root, args.mode)
        submit_chain(snapshot, run_root, args.mode)
        print(f"Job manifest: {run_root / 'jobs.json'}")
    else:
        print("Use --submit to freeze source and submit the approved chain.")


if __name__ == "__main__":
    main()
