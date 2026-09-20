#!/bin/bash
#SBATCH --job-name=dfine_coco
#SBATCH --output=logs/dfine_coco/dfine_coco_%j.out
#SBATCH --error=logs/dfine_coco/dfine_coco_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

set -euo pipefail
set -x

PROJECT_DIR=/home/zzy/projects/occ_study/D-FINE_mine/D-FINE
MODEL=${MODEL:-l}
CONDA_ENV=${CONDA_ENV:-dfine}
CONFIG_FILE=configs/dfine/dfine_hgnetv2_${MODEL}_coco.yml

cd "$PROJECT_DIR"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate "$CONDA_ENV"

test -f "$CONFIG_FILE"

THREADS_PER_RANK=$((SLURM_CPUS_PER_TASK / 8))
export OMP_NUM_THREADS="$THREADS_PER_RANK"
export MKL_NUM_THREADS="$THREADS_PER_RANK"
export PYTHONUNBUFFERED=1

JOB_NUMBER=$((10#$SLURM_JOB_ID))
MASTER_PORT=$((20000 + JOB_NUMBER % 20000))

echo "Job: $SLURM_JOB_ID | Model: $MODEL | Node: $(hostname) | Port: $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
  train.py \
  -c "$CONFIG_FILE" \
  --output-dir "/cluster/home/zzy/D-FINE/output/dfine_hgnetv2_${MODEL}_coco" \
  --use-amp \
  --seed=0
