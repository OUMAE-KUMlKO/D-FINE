#!/bin/bash
#SBATCH --job-name=dfine_uavswarm_b0
#SBATCH --output=logs/dfine_uavswarm_b0/dfine_uavswarm_b0_%j.out
#SBATCH --error=logs/dfine_uavswarm_b0/dfine_uavswarm_b0_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# From the project directory:
# mkdir -p logs/dfine_uavswarm_b0
# sbatch slurm_files/run_train_dfine_uavswarm_baseline.slurm
set -euo pipefail
set -x

PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
CONDA_ENV=${CONDA_ENV:-dfine}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/cluster/home/zzy/D-FINE/output/uavswarm_strip_20260923}
OUTPUT_DIR=${OUTPUT_DIR:-$EXPERIMENT_ROOT/b0}
CONFIG_FILE=configs/dfine/dfine_hgnetv2_l_uavswarm.yml

cd "$PROJECT_DIR"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Fresh B0 output already exists: $OUTPUT_DIR. Set a new EXPERIMENT_ROOT to start another experiment." >&2
  exit 1
fi
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate "$CONDA_ENV"

test -f "$CONFIG_FILE"
test -f weight/hgnetv2/PPHGNetV2_B4_stage1.pth
test -f /cluster/home/zzy/dataset/UAVSwarm-dataset-master/annotations/train_half.json
test -f /cluster/home/zzy/dataset/UAVSwarm-dataset-master/annotations/val_half.json

THREADS_PER_RANK=$((SLURM_CPUS_PER_TASK / 8))
export OMP_NUM_THREADS="$THREADS_PER_RANK"
export MKL_NUM_THREADS="$THREADS_PER_RANK"
export PYTHONUNBUFFERED=1
JOB_NUMBER=$((10#$SLURM_JOB_ID))
MASTER_PORT=$((20000 + JOB_NUMBER % 20000))

# Backbone pretraining only: no detector --tuning or --resume checkpoint.
torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
  train.py \
  -c "$CONFIG_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --use-amp \
  --seed=0

# G1 and P1 both use this exact detector snapshot (EMA weights when present).
test -f "$OUTPUT_DIR/best_stg1.pth"
