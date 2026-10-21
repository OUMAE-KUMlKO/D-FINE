#!/usr/bin/env python3
"""Check the native D-FINE ablation against its frozen controlled configuration."""

import argparse
import json
import os
from pathlib import Path
import sys

import yaml


def differences(reference, experiment, prefix=""):
    if isinstance(reference, dict) and isinstance(experiment, dict):
        return [
            item
            for key in sorted(reference.keys() | experiment.keys())
            for item in differences(
                reference.get(key), experiment.get(key),
                f"{prefix}.{key}" if prefix else key,
            )
        ]
    return [] if reference == experiment else [
        {"key": prefix, "reference": reference, "experiment": experiment}
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("uavswarm", "visdrone"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    from src.core import YAMLConfig

    overrides = dict(seed=args.seed, controlled_seed=args.seed, use_amp=True,
                     output_dir=str(output))
    reference = YAMLConfig(
        f"configs/controlled/{args.dataset}_g0.yml", **overrides,
    ).yaml_cfg
    config = YAMLConfig(
        f"configs/dfine/ablation/dfine_hgnetv2_l_{args.dataset}_baseline_ddf_off.yml",
        **overrides,
        exp_name=f"dfine_l_{args.dataset}_baseline_ddf_off_seed{args.seed}",
    )
    effective = config.yaml_cfg
    diff = differences(reference, effective)
    expected = {"__include__", "exp_name", "DFINECriterion.weight_dict.loss_ddf"}
    if args.dataset == "visdrone":
        expected.add("val_dataloader.dataset.ann_file")
    assert {item["key"] for item in diff} == expected, diff
    assert reference["DFINECriterion"]["weight_dict"]["loss_ddf"] == 1.5
    assert effective["model"] == "DFINE"
    assert effective["DFINETransformer"]["num_layers"] == 6
    assert effective["epochs"] == 80
    assert effective["seed"] == effective["controlled_seed"] == args.seed
    expected_batch = 64 if args.dataset == "uavswarm" else 32
    assert effective["train_dataloader"]["total_batch_size"] == expected_batch
    assert effective["val_dataloader"]["total_batch_size"] == expected_batch
    assert effective["train_dataloader"]["collate_fn"]["stop_epoch"] == 72
    assert effective["DFINECriterion"]["losses"] == ["vfl", "boxes", "local"]
    expected_weights = dict(loss_vfl=1, loss_bbox=5, loss_giou=2,
                            loss_fgl=0.15, loss_ddf=0.0)
    assert effective["DFINECriterion"]["weight_dict"] == expected_weights
    assert config.criterion.weight_dict == expected_weights
    assert not effective.get("resume") and not effective.get("tuning")
    assert effective["HGNetv2"]["pretrained"] is True
    assert Path("weight/hgnetv2/PPHGNetV2_B4_stage1.pth").is_file()
    for loader in ("train_dataloader", "val_dataloader"):
        dataset = effective[loader]["dataset"]
        assert Path(dataset["ann_file"]).is_file(), dataset["ann_file"]
        assert Path(dataset["img_folder"]).is_dir(), dataset["img_folder"]
    report_dir.mkdir(parents=True, exist_ok=True)
    report = dict(status="passed", dataset=args.dataset, seed=args.seed,
                  source_dir=str(source), output_dir=str(output),
                  model=effective["model"], epochs=80, global_batch=expected_batch,
                  loss_weights=expected_weights, protocol_diff=diff)
    (report_dir / "protocol_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    (report_dir / "effective_config.yml").write_text(
        yaml.safe_dump(effective, sort_keys=False)
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
