#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

set -euo pipefail
PROJECT_DIR=${1:?frozen source required}
BASELINE_CHECKPOINT=${2:?VisDrone G0 best checkpoint required}
RESULT_ROOT=${3:?fresh result directory required}
PREFLIGHT_FILE=${4:?successful preflight summary required}
CONFIG_FILE=configs/dfine/gt_refinement/dfine_hgnetv2_l_visdrone_gt_stage1.yml
cd "$PROJECT_DIR"
test -f "$CONFIG_FILE"
test -f "$BASELINE_CHECKPOINT"
test -f "$PREFLIGHT_FILE"
test -f output/visdrone_oracle/instances_val_positive_ids.json
if [[ -e "$RESULT_ROOT" ]]; then
  echo "Refusing to reuse existing results: $RESULT_ROOT" >&2
  exit 1
fi
mkdir -p "$RESULT_ROOT"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
python - "$PREFLIGHT_FILE" "$BASELINE_CHECKPOINT" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text())
if report.get('status') != 'passed' or report.get('successful_updates') != 2:
    raise RuntimeError('Oracle preflight did not complete two valid updates')
if report.get('batch_size') != 12:
    raise RuntimeError('Preflight must verify the actual per-GPU batch size of 12')
if Path(report['baseline_checkpoint']).resolve() != Path(sys.argv[2]).resolve():
    raise RuntimeError('Preflight and training baseline checkpoints differ')
PY
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 8))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dfine-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/dfine-cache-${SLURM_JOB_ID}"
MASTER_PORT=$((20000 + 10#$SLURM_JOB_ID % 20000))

echo "Job=$SLURM_JOB_ID | VisDrone GT-boundary oracle | frozen baseline=$BASELINE_CHECKPOINT"
# Baseline mode computes fixed-pair geometry using GT, but leaves every
# baseline prediction and classification score unchanged; no training.
torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
  train.py -c "$CONFIG_FILE" --test-only --seed=0 \
  -u "DFINEGTRefiner.baseline_checkpoint=$BASELINE_CHECKPOINT" \
     DFINEGTRefiner.feature_mode=baseline \
  --output-dir "$RESULT_ROOT/baseline"

echo "Starting new GT strip head: 12 epochs, total batch96, seed0; frozen detector and scores."
exec torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
  train.py -c "$CONFIG_FILE" --seed=0 --use-amp \
  -u "DFINEGTRefiner.baseline_checkpoint=$BASELINE_CHECKPOINT" \
  --output-dir "$RESULT_ROOT/oracle"
