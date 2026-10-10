#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

set -euo pipefail
PROJECT_DIR=${1:?frozen project directory required}
CONFIG_FILE=${2:?continuation config required}
OUTPUT_DIR=${3:?fresh output directory required}
RESUME_CHECKPOINT=${4:?completed epoch79 checkpoint required}
EXPERIMENT_SEED=${5:-0}
cd "$PROJECT_DIR"
test -f "$CONFIG_FILE"
test -f "$RESUME_CHECKPOINT"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output already exists: $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
# Retain the evaluated starting weights if no continuation epoch improves AP.
# A real copy avoids writes through a symlink to the original checkpoint.
cp "$RESUME_CHECKPOINT" "$OUTPUT_DIR/best_stg2.pth"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 8))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$EXPERIMENT_SEED"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dfine-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/dfine-cache-${SLURM_JOB_ID}"
JOB_NUMBER=$((10#$SLURM_JOB_ID))
MASTER_PORT=$((20000 + JOB_NUMBER % 20000))
echo "Job=$SLURM_JOB_ID | config=$CONFIG_FILE | seed=$EXPERIMENT_SEED | resume=$RESUME_CHECKPOINT"
echo "Frozen source=$PROJECT_DIR | output=$OUTPUT_DIR | target epochs=120 | LR decay at epoch100"
exec torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
  train.py -c "$CONFIG_FILE" --output-dir "$OUTPUT_DIR" --use-amp \
  --resume "$RESUME_CHECKPOINT" --seed="$EXPERIMENT_SEED" -u "controlled_seed=$EXPERIMENT_SEED"
