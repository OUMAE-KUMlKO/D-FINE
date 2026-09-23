#!/bin/bash
#SBATCH --job-name=dfine_uavswarm_pred_p1
#SBATCH --output=logs/dfine_uavswarm_pred_p1/dfine_uavswarm_pred_p1_%j.out
#SBATCH --error=logs/dfine_uavswarm_pred_p1/dfine_uavswarm_pred_p1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Submit from the project directory after B0 and G1 complete:
# mkdir -p logs/dfine_uavswarm_pred_p1
# sbatch slurm_files/run_train_dfine_uavswarm_pred_p1.slurm
# Slurm opens log files before the script runs; create the directory first.
set -euo pipefail
set -x

PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
CONDA_ENV=${CONDA_ENV:-dfine}
CONFIG_FILE=${CONFIG_FILE:-configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_pred_p1.yml}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/cluster/home/zzy/D-FINE/output/uavswarm_strip_20260923}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:-$EXPERIMENT_ROOT/b0/best_stg1.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$EXPERIMENT_ROOT/p1}

cd "$PROJECT_DIR"
if [[ ! -f "$BASELINE_CHECKPOINT" ]]; then
  echo "Baseline checkpoint not found: $BASELINE_CHECKPOINT" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Fresh P1 output already exists: $OUTPUT_DIR. Use a new OUTPUT_DIR or resume through train.py." >&2
  exit 1
fi
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate "$CONDA_ENV"

test -f "$CONFIG_FILE"
test -f /cluster/home/zzy/dataset/UAVSwarm-dataset-master/annotations/train_half.json
test -f /cluster/home/zzy/dataset/UAVSwarm-dataset-master/annotations/val_half.json

THREADS_PER_RANK=$((SLURM_CPUS_PER_TASK / 8))
export OMP_NUM_THREADS="$THREADS_PER_RANK"
export MKL_NUM_THREADS="$THREADS_PER_RANK"
export PYTHONUNBUFFERED=1

JOB_NUMBER=$((10#$SLURM_JOB_ID))
MASTER_PORT=$((20000 + JOB_NUMBER % 20000))

echo "Job: $SLURM_JOB_ID | Model: l P1 | Dataset: UAVSwarm | Node: $(hostname) | Port: $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Load only the frozen baseline; the P1 refiner starts from its own initialization.
torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
  train.py \
  -c "$CONFIG_FILE" \
  -u "DFINEPredRefiner.baseline_checkpoint=$BASELINE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --use-amp \
  --seed=0
