#!/usr/bin/env python3
"""Copy the five approved runs and create independent evaluation bundles."""

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import yaml


BRANCH = "archive/selected-models-20261021"
TAG = "selected-models-20261021-v1"
BASE_COMMIT = "13d83be6c7cc9cdfee20e372afb6742a5629436b"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def file_inventory(folder):
    return {str(p.relative_to(folder)): p.stat().st_size
            for p in folder.rglob("*") if p.is_file()}


def expanded_config(source, relative_config, seed):
    # Separate process: each frozen src owns its own component registry.
    code = """
import json,sys
sys.path.insert(0,sys.argv[1])
from src.core import YAMLConfig
print(json.dumps(YAMLConfig(sys.argv[2],seed=int(sys.argv[3]),use_amp=True).yaml_cfg))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(source), str(source / relative_config), str(seed)],
        cwd=source, capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def model_readme(model):
    metrics = model["expected_metrics_percent"]
    return f"""# {model['title']}

本目录为用户指定实验的独立复评归档，目录ID：`{model['id']}`。

## 对应Git版本

- 代码仓库：`{REPO}`。
- 归档分支：`{BRANCH}`；验收标签：`{TAG}`（五模型复评完成后固定）。
- 整理阶段提交：`{BASE_COMMIT}`。
- 分支内索引：`archives/selected_models/README.md`；结构与来源清单：`models.json`。
- 最终归档提交号以本目录`manifest.json`及归档根目录`VERSION.json`为准。
- 本模型复评使用本目录`source/`的原实验源码，不导入研究工作区当前的`src/`。

## 指标、权重与实验身份

| AP | AP50 | AP75 | APsmall | APmedium | APlarge |
| ---: | ---: | ---: | ---: | ---: | ---: |
| {' | '.join(f'{v:.6f}' for v in metrics[:6])} |

以上为原训练日志中总AP最佳同一轮的指标，单位为百分数。
作业`{model['job_id']}`，seed={model['seed']}，完整日志{model['completed_epochs']}轮，最佳外层epoch={model['best_log_epoch']}（从0计数）。
checkpoint内部epoch={model['internal_epoch']}；评估使用`{model['weights']}`权重。

{model['selection_note']}

## 模型结构

{model['structure']}

完整实现位于`source/src/zoo/dfine/`；原训练配置为`source/{model['config_rel']}`。
`archive_eval.yml`已展开配置继承，仅调整归档路径、评估输出和预训练初始化依赖，保留模型参数、验证预处理、类别映射与权重类型。

## 文件位置

- `training/`：完整原训练目录，包含所有周期checkpoint、best、last、原日志、样本可视化与原评估产物。
- `{model['checkpoint']}`：默认最佳checkpoint；未裁剪优化器或其他原保存内容。
- `training/log.txt`：原逐轮指标，最佳记录在第{model['best_log_line']}行。
- `logs/`：训练目录之外的Slurm输出及错误日志；训练目录原有控制台日志仍保留在`training/`内。
- `source/`：原运行时完整src、配置和已有工具/启动脚本；`source/train.py`为原生评估入口。
- `annotations/train.json`、`annotations/val.json`：该实验实际标注文件的独立副本。
- `archive_eval.yml`、`eval.sh`、`summarize_eval.py`：本目录独立评估入口。
- `metadata/`：原作业与协议记录；`manifest.json`：本归档来源、配置调整和预期指标。
- `environment/`：本次复评环境的版本与pip依赖记录，不将其冒充历史环境锁文件。
- `evaluation/`：新复评输出；成功运行后通过`evaluation/latest/`查看最新日志及`metrics.json`。

原训练目录：`{model['original_training']}`。
原冻结源码：`{model['original_source']}`。

## 运行方法

在有GPU的计算节点上，使用现有dfine环境：

```bash
cd {model['archive_dir']}
bash eval.sh
```

默认8GPU，验证总batch={model['eval_global_batch']}，与原训练评估一致。脚本从自身位置定位文件，可从任意工作目录调用。
单GPU可用`NPROC=1 bash eval.sh`，脚本相应缩小验证batch；不同批次划分可能带来极小浮点差异，正式验收使用8GPU。
可通过`PYTHON_BIN`、`IMAGE_ROOT`、`CHECKPOINT`、`CONFIG`、`OUTPUT_DIR`覆盖相应路径。
也可直接运行原生入口：

```bash
cd {model['archive_dir']}
/home/zzy/.conda/envs/dfine/bin/python -m torch.distributed.run --nproc_per_node=8 \\
  source/train.py -c archive_eval.yml -r {model['checkpoint']} \\
  --test-only --use-amp --seed={model['seed']} \\
  --output-dir evaluation/manual --summary-dir evaluation/manual/summary
```

验证图像路径为`{model['validation_images']}`。图像数据集不复制进归档，迁移机器需提供同一验证集并设置`IMAGE_ROOT`。
验证标注已复制，不随旧output目录删除而丢失。VisDrone原版28.1196保留历史ID口径，其余三个VisDrone模型使用各自正值ID版本。
本目录承诺的是加载最终checkpoint复评；重新训练仍需原训练图像、初始化权重及原训练协议。

## 复评验收

`evaluation/latest/metrics.json`记录12项COCO指标、与原日志的逐项差值及两位小数AP是否吻合。
归档过程不重新训练、不更改预测或评估GT来凑齐指标；未验收前以原日志为历史指标，最终实测以该JSON为准。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    specs = json.loads((HERE / "models.json").read_text())
    # Preflight all destinations before copying any large file.
    assert not (destination / "index.json").exists(), "Existing archive index; do not overwrite"
    for spec in specs:
        assert not (destination / spec["id"]).exists(), spec["id"]
        origin = REPO / spec["experiment_root"]
        assert (origin / spec["training_rel"] / spec["checkpoint"]).is_file()
    for helper in ["eval.sh", "summarize_eval.py", "run_evaluations.slurm"]:
        assert (HERE / helper).is_file(), helper
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    packages = {}
    for name in ["torch", "torchvision", "numpy", "scipy", "pycocotools", "faster-coco-eval", "PyYAML"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = dict(python=sys.version, executable=sys.executable,
                       platform=platform.platform(), packages=packages,
                       recorded_at=datetime.now().astimezone().isoformat())
    index = dict(archive_branch=BRANCH, archive_tag=TAG, base_commit=BASE_COMMIT,
                 repository=str(REPO), destination=str(destination), models=[])
    (HERE / "configs").mkdir(exist_ok=True)
    for spec in specs:
        origin = REPO / spec["experiment_root"]
        source = origin / "source"
        training = origin / spec["training_rel"]
        target = destination / spec["id"]
        target.mkdir()
        print(f"Copying {spec['id']}", flush=True)
        shutil.copytree(training, target / "training")
        original_inventory = file_inventory(training)
        assert file_inventory(target / "training") == original_inventory
        (target / "source").mkdir()
        for name in ["src", "configs", "tools", "slurm_files", "tests"]:
            if (source / name).is_dir():
                shutil.copytree(source / name, target / "source" / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(source / "train.py", target / "source/train.py")
        for name in ["requirements.txt", "LICENSE"]:
            candidate = source / name if (source / name).is_file() else REPO / name
            if candidate.is_file():
                shutil.copy2(candidate, target / "source" / name)
        cfg = expanded_config(source, spec["config_rel"], spec["seed"])
        if "controlled_seed" in cfg:
            cfg["controlled_seed"] = spec["seed"]
        validation_images = cfg["val_dataloader"]["dataset"]["img_folder"]
        (target / "annotations").mkdir()
        annotation_sources = {}
        for phase, loader in [("train", "train_dataloader"), ("val", "val_dataloader")]:
            original_ann = Path(cfg[loader]["dataset"]["ann_file"])
            if not original_ann.is_absolute():
                original_ann = source / original_ann
            original_ann = original_ann.resolve()
            shutil.copy2(original_ann, target / "annotations" / f"{phase}.json")
            annotation_sources[phase] = str(original_ann)
            cfg[loader]["dataset"]["ann_file"] = f"./annotations/{phase}.json"
        cfg.pop("__include__", None)
        cfg.pop("resume", None)
        cfg.pop("tuning", None)
        cfg.update(use_ema=spec["weights"] == "ema", use_wandb=False,
                   output_dir="./evaluation/manual", summary_dir="./evaluation/manual/summary")
        cfg["HGNetv2"]["pretrained"] = False
        if "DFINEPredRefiner" in cfg:
            cfg["DFINEPredRefiner"]["baseline_checkpoint"] = None
        config_text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        (target / "archive_eval.yml").write_text(config_text)
        (HERE / "configs" / f"{spec['id']}.yml").write_text(config_text)
        records = [(i, json.loads(line)) for i, line in enumerate((training / "log.txt").open(), 1)
                   if line.strip()]
        line, best = max(records, key=lambda pair: pair[1]["test_coco_eval_bbox"][0])
        model = dict(spec, archive_branch=BRANCH, archive_tag=TAG, base_commit=BASE_COMMIT,
                     original_source=str(source), original_training=str(training),
                     archive_dir=str(target), checkpoint="training/" + spec["checkpoint"],
                     expected_metrics_percent=[v * 100 for v in best["test_coco_eval_bbox"]],
                     best_log_epoch=best["epoch"], best_log_line=line, completed_epochs=len(records),
                     eval_world_size=8, eval_global_batch=cfg["val_dataloader"]["total_batch_size"],
                     validation_images=validation_images, annotation_sources=annotation_sources,
                     training_inventory=original_inventory,
                     evaluation_status="pending")
        write_json(target / "manifest.json", model)
        (target / "README.md").write_text(model_readme(model))
        for helper in ["eval.sh", "summarize_eval.py"]:
            shutil.copy2(HERE / helper, target / helper)
        (target / "logs").mkdir()
        for logfile in (origin / "logs").glob(spec["log_pattern"]):
            shutil.copy2(logfile, target / "logs" / logfile.name)
        (target / "metadata").mkdir()
        for metadata in ["jobs.json", "experiment_selection.json", "preflight.json"]:
            if (origin / metadata).is_file():
                shutil.copy2(origin / metadata, target / "metadata" / metadata)
        if (origin / "manifest").is_dir():
            shutil.copytree(origin / "manifest", target / "metadata/run_manifest")
        (target / "environment").mkdir()
        (target / "environment/pip-freeze.txt").write_text(freeze)
        write_json(target / "environment/versions.json", environment)
        index["models"].append(model)
        write_json(destination / "index.json", index)
        print(f"Copied and checked {len(original_inventory)} training files: {spec['id']}", flush=True)
    shutil.copy2(HERE / "run_evaluations.slurm", destination / "run_evaluations.slurm")
    (destination / "logs").mkdir(exist_ok=True)
    write_json(HERE / "index.json", index)
    print("All five bundles copied; full GPU evaluation is still required.", flush=True)


if __name__ == "__main__":
    main()
