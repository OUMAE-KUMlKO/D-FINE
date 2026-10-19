#!/bin/bash
#SBATCH --job-name=uavswarm_g5_ddf_ablation
#SBATCH --output=logs/uavswarm_g5_ddf_ablation_%j.out
#SBATCH --error=logs/uavswarm_g5_ddf_ablation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

# Required submission interface:
# SOURCE_DIR=/frozen/source EXPERIMENT_ROOT=/path/uavswarm_g5_ddf_ablation_<run_id> \
#   DDF_MODE=on TRAIN_SEED=1 sbatch slurm_files/run_train_uavswarm_g5_ddf_ablation.slurm
set -euo pipefail

: "${SOURCE_DIR:?Set SOURCE_DIR to the frozen experiment source directory}"
: "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to the independent experiment root}"
: "${DDF_MODE:?Set DDF_MODE to on or off}"
TRAIN_SEED=${TRAIN_SEED:-0}
[[ "$TRAIN_SEED" =~ ^[0-9]+$ ]] || { echo "TRAIN_SEED must be a nonnegative integer" >&2; exit 2; }

case "$DDF_MODE" in
  on|off) ;;
  *) echo "DDF_MODE must be 'on' or 'off', got: $DDF_MODE" >&2; exit 2 ;;
esac

PYTHON_BIN=/home/zzy/.conda/envs/dfine/bin/python
CONFIG_FILE="$SOURCE_DIR/configs/dfine/ablation/dfine_hgnetv2_l_uavswarm_g5_ddf_${DDF_MODE}.yml"
GROUP_NAME="g5_ddf_${DDF_MODE}"
GROUP_DIR="$EXPERIMENT_ROOT/results/$GROUP_NAME"
MANIFEST_DIR="$GROUP_DIR/manifest"
TRAIN_LOG_DIR="$GROUP_DIR/logs"

test -d "$SOURCE_DIR"
test -f "$CONFIG_FILE"
test -x "$PYTHON_BIN"
test -f "$SOURCE_DIR/weight/hgnetv2/PPHGNetV2_B4_stage1.pth"

mkdir -p "$EXPERIMENT_ROOT/results"
if ! mkdir "$GROUP_DIR"; then
  echo "Result group already exists or cannot be created: $GROUP_DIR" >&2
  exit 1
fi
mkdir -p "$MANIFEST_DIR" "$TRAIN_LOG_DIR"

cd "$SOURCE_DIR"
export CONFIG_FILE GROUP_DIR TRAIN_SEED
"$PYTHON_BIN" - <<'PY'
import os
import yaml

from src.core import YAMLConfig

config_file = os.environ["CONFIG_FILE"]
group_dir = os.environ["GROUP_DIR"]
effective = YAMLConfig(
    config_file,
    seed=int(os.environ["TRAIN_SEED"]),
    use_amp=True,
    output_dir=group_dir,
).yaml_cfg
with open(os.path.join(group_dir, "manifest", "effective_config.yml"), "w") as stream:
    yaml.safe_dump(effective, stream, sort_keys=False)
PY

module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine

THREADS_PER_RANK=$((SLURM_CPUS_PER_TASK / 8))
export OMP_NUM_THREADS="$THREADS_PER_RANK"
export MKL_NUM_THREADS="$THREADS_PER_RANK"
export PYTHONUNBUFFERED=1
JOB_NUMBER=$((10#$SLURM_JOB_ID))
MASTER_PORT=$((20000 + JOB_NUMBER % 20000))

RUN_COMMAND=(
  torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT"
  train.py -c "$CONFIG_FILE" --output-dir "$GROUP_DIR" --use-amp --seed="$TRAIN_SEED"
)
printf '%q ' "${RUN_COMMAND[@]}" > "$MANIFEST_DIR/run_command.txt"
printf '\n' >> "$MANIFEST_DIR/run_command.txt"
{
  printf 'source_dir=%s\n' "$SOURCE_DIR"
  printf 'experiment_root=%s\n' "$EXPERIMENT_ROOT"
  printf 'group=%s\n' "$GROUP_NAME"
  printf 'config=%s\n' "$CONFIG_FILE"
  printf 'backbone_pretrained=%s\n' "$SOURCE_DIR/weight/hgnetv2/PPHGNetV2_B4_stage1.pth"
  printf 'seed=%s\nworld_size=8\ntotal_batch_size=64\nepochs=80\n' "$TRAIN_SEED"
} > "$MANIFEST_DIR/run.env"

# Both groups use the same frozen source and the already completed pairwise
# GPU preflight. Formal training starts fresh in a separate process below.
"$PYTHON_BIN" - "$EXPERIMENT_ROOT/preflight.json" "$SOURCE_DIR" "$GROUP_DIR" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys

report_path, source, group = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text())
assert report["status"] == "passed", "Pairwise GPU preflight has not passed"
assert report["static"]["formal_scope"]["seed"] == int(os.environ["TRAIN_SEED"]), "Preflight seed differs"
assert report["cpu_initialization"]["seed"] == int(os.environ["TRAIN_SEED"]), "Initialization-check seed differs"
for mode, config in report["configs"].items():
    expected = source / "configs/dfine/ablation" / f"dfine_hgnetv2_l_uavswarm_g5_ddf_{mode}.yml"
    assert Path(config).resolve() == expected.resolve(), "Preflight used another source snapshot"
shutil.copy2(report_path, group / "preflight.json")
PY

echo "Job: $SLURM_JOB_ID | Group: $GROUP_NAME | Seed: $TRAIN_SEED | Fresh 80-epoch G5 training | Port: $MASTER_PORT"
"${RUN_COMMAND[@]}" 2>&1 | tee "$TRAIN_LOG_DIR/train_${SLURM_JOB_ID}.log"
