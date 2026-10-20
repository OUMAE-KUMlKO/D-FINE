#!/bin/bash
#SBATCH --job-name=vd-g5-ddf-off
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

# SOURCE_DIR is a copy of output/controlled_20260926_212304/source, with the
# new config and positive-ID validation annotations added. Keep its src intact.
# Submit with explicit --output/--error paths inside EXPERIMENT_ROOT/logs.
set -euo pipefail
: "${SOURCE_DIR:?Set SOURCE_DIR to the frozen job-15543-compatible source}"
: "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to a new independent experiment root}"
: "${BASELINE_CHECKPOINT:?Set BASELINE_CHECKPOINT to job 15543 best_stg2.pth}"

PYTHON_BIN=/home/zzy/.conda/envs/dfine/bin/python
CONFIG_FILE="$SOURCE_DIR/configs/dfine/ablation/dfine_hgnetv2_l_visdrone_g5_ddf_off.yml"
GROUP_DIR="$EXPERIMENT_ROOT/results/g5_ddf_off"
BASELINE_DIR="$EXPERIMENT_ROOT/results/g5_ddf_on_reference"
test -f "$CONFIG_FILE"
test -f "$BASELINE_CHECKPOINT"
test -f "$EXPERIMENT_ROOT/manifest/provenance.json"
mkdir -p "$EXPERIMENT_ROOT/results"
mkdir "$GROUP_DIR" "$BASELINE_DIR"
mkdir -p "$GROUP_DIR/manifest"

module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
cd "$SOURCE_DIR"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 8))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dfine-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/dfine-cache-${SLURM_JOB_ID}"
export CONFIG_FILE GROUP_DIR EXPERIMENT_ROOT
MASTER_PORT=$((20000 + 10#$SLURM_JOB_ID % 20000))

# Fail before spending GPU time if another training setting changed.
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
import yaml
from src.core import YAMLConfig

reference = YAMLConfig('configs/controlled/visdrone_g5.yml').yaml_cfg
config = YAMLConfig(os.environ['CONFIG_FILE']).yaml_cfg
def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for key in sorted(a.keys() | b.keys())
                for d in differences(a.get(key), b.get(key), f'{prefix}.{key}' if prefix else key)]
    return [] if a == b else [{'key': prefix, 'reference': a, 'experiment': b}]
diff = differences(reference, config)
expected = {'__include__', 'output_dir', 'exp_name', 'DFINECriterion.weight_dict.loss_ddf',
            'val_dataloader.dataset.ann_file'}
assert {d['key'] for d in diff} == expected, diff
assert config['DFINECriterion']['weight_dict']['loss_ddf'] == 0
assert config['DFINETransformer']['num_layers'] == 6
assert config['epochs'] == 80 and config['controlled_seed'] == 0
assert config['train_dataloader']['total_batch_size'] == 32
manifest = Path(os.environ['GROUP_DIR']) / 'manifest'
(manifest / 'protocol_diff.json').write_text(json.dumps(diff, indent=2) + '\n')
effective = YAMLConfig(os.environ['CONFIG_FILE'], seed=0, use_amp=True,
                       output_dir=os.environ['GROUP_DIR']).yaml_cfg
(manifest / 'effective_config.yml').write_text(yaml.safe_dump(effective, sort_keys=False))
print('Protocol check passed:', json.dumps(diff))
PY

# Evaluate the historical EMA with the same corrected annotation IDs first.
# Its checkpoint is used ONLY here; formal training below starts from backbone.
EVAL_COMMAND=("$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node=8
  --master_port="$MASTER_PORT" train.py -c "$CONFIG_FILE"
  --resume "$BASELINE_CHECKPOINT" --test-only --use-amp --seed=0
  --output-dir "$BASELINE_DIR" -u DFINECriterion.weight_dict.loss_ddf=1.5)
printf '%q ' "${EVAL_COMMAND[@]}" > "$BASELINE_DIR/run_command.txt"
printf '\n' >> "$BASELINE_DIR/run_command.txt"
"${EVAL_COMMAND[@]}" 2>&1 | tee "$BASELINE_DIR/evaluate.log"
"$PYTHON_BIN" - "$BASELINE_DIR" <<'PY'
import json
from pathlib import Path
import sys
import torch

folder = Path(sys.argv[1])
p = torch.load(folder / 'eval.pth', map_location='cpu')['precision']
def mean(x):
    return float(x[x > -1].mean())
metrics = dict(AP=mean(p[:, :, :, 0, -1]), AP50=mean(p[0, :, :, 0, -1]),
               AP75=mean(p[5, :, :, 0, -1]), APsmall=mean(p[:, :, :, 1, -1]))
(folder / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
print('Historical G5, corrected annotation IDs:', metrics)
PY

TRAIN_COMMAND=("$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node=8
  --master_port="$MASTER_PORT" train.py -c "$CONFIG_FILE"
  --output-dir "$GROUP_DIR" --use-amp --seed=0)
printf '%q ' "${TRAIN_COMMAND[@]}" > "$GROUP_DIR/manifest/run_command.txt"
printf '\n' >> "$GROUP_DIR/manifest/run_command.txt"
echo "Job=$SLURM_JOB_ID | fixed six-layer G5 | DDF=0 | seed=0 | fresh 80 epochs | batch=32"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "$GROUP_DIR/train.log"
