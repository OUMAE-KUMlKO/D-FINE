#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Called by tools/submit_controlled_experiments.py with a frozen source tree.
set -euo pipefail
PROJECT_DIR=${1:?frozen project directory required}
CONFIG_FILE=${2:?controlled config required}
OUTPUT_DIR=${3:?fresh output directory required}
EXPERIMENT_SEED=${4:-0}
cd "$PROJECT_DIR"
test -f "$CONFIG_FILE"
test -f weight/hgnetv2/PPHGNetV2_B4_stage1.pth
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output already exists: $OUTPUT_DIR" >&2
  exit 1
fi
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
echo "Job=$SLURM_JOB_ID | config=$CONFIG_FILE | seed=$EXPERIMENT_SEED | epochs=80 | batch from config"
echo "Frozen source=$PROJECT_DIR | output=$OUTPUT_DIR"
exec torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
  train.py -c "$CONFIG_FILE" --output-dir "$OUTPUT_DIR" --use-amp \
  --seed="$EXPERIMENT_SEED" -u "controlled_seed=$EXPERIMENT_SEED"
