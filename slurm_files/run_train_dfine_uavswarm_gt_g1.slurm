#!/bin/bash
#SBATCH --job-name=dfine_uavswarm_gt_g1
#SBATCH --output=logs/dfine_uavswarm_gt_g1/dfine_uavswarm_gt_g1_%j.out
#SBATCH --error=logs/dfine_uavswarm_gt_g1/dfine_uavswarm_gt_g1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Submit after B0 completes, or use --dependency=afterok:<B0_JOB_ID>.
# mkdir -p logs/dfine_uavswarm_gt_g1
# sbatch slurm_files/run_train_dfine_uavswarm_gt_g1.slurm
set -euo pipefail
set -x

PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
CONDA_ENV=${CONDA_ENV:-dfine}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/cluster/home/zzy/D-FINE/output/uavswarm_strip_20260923}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:-$EXPERIMENT_ROOT/b0/best_stg1.pth}
OUTPUT_DIR=${OUTPUT_DIR:-$EXPERIMENT_ROOT/g1}
CONFIG_FILE=configs/dfine/gt_refinement/dfine_hgnetv2_l_uavswarm_gt_stage1_cluster.yml

cd "$PROJECT_DIR"
test -f "$BASELINE_CHECKPOINT"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Fresh G1 output already exists: $OUTPUT_DIR. Use a new OUTPUT_DIR or resume through train.py." >&2
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

# Load only B0 detector weights; initialize the GT strip head independently.
torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
  train.py \
  -c "$CONFIG_FILE" \
  -u "DFINEGTRefiner.baseline_checkpoint=$BASELINE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --use-amp \
  --seed=0
