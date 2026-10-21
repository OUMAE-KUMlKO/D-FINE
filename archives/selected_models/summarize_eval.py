#!/usr/bin/env python3
"""从冻结 runtime 输出的 eval.pth 恢复精确 COCO 指标并核对原日志。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = [
    "AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
    "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluator = COCOeval(iouType="bbox")
    evaluator.eval = torch.load(args.evaluation_dir / "eval.pth", map_location="cpu")
    evaluator.params = evaluator.eval["params"]
    evaluator.summarize()
    metrics = np.asarray(evaluator.stats, dtype=float) * 100.0
    expected = np.asarray(manifest["expected_metrics_percent"], dtype=float)
    if metrics.shape != (12,) or expected.shape != (12,):
        raise ValueError("复评指标和原日志指标都应包含 12 项 COCO bbox 指标")
    if not np.isfinite(metrics).all() or not np.isfinite(expected).all():
        raise ValueError("COCO 指标包含非有限数值")
    difference = metrics - expected
    matches = f"{metrics[0]:.2f}" == f"{expected[0]:.2f}"
    result = {
        "id": manifest["id"],
        "dataset": manifest["dataset"],
        "seed": manifest["seed"],
        "archive_branch": manifest["archive_branch"],
        "archive_tag": manifest["archive_tag"],
        "weights": manifest["weights"],
        "checkpoint": args.checkpoint or manifest["checkpoint"],
        "config": args.config,
        "metric_names": METRIC_NAMES,
        "metrics_percent": metrics.tolist(),
        "expected_metrics_percent": expected.tolist(),
        "delta_percent": difference.tolist(),
        "max_abs_delta_percent": float(np.abs(difference).max()),
        "ap_rounded_2dp_matches": matches,
        "all_metrics_rounded_2dp_match": all(
            f"{actual:.2f}" == f"{reference:.2f}"
            for actual, reference in zip(metrics, expected)
        ),
    }
    destination = args.evaluation_dir / "metrics.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"复评 AP: {metrics[0]:.8f}，原日志 AP: {expected[0]:.8f}，差值: {difference[0]:+.8f}")
    print(f"AP 保留两位小数是否一致: {'是' if matches else '否'}；完整指标: {destination}")
    return 0 if matches else 2


if __name__ == "__main__":
    raise SystemExit(main())
