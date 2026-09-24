#!/bin/bash
#SBATCH --job-name=dfine_uavswarm_all_layer_strip_full
#SBATCH --output=logs/dfine_uavswarm_all_layer_strip_full/dfine_uavswarm_all_layer_strip_full_%j.out
#SBATCH --error=logs/dfine_uavswarm_all_layer_strip_full/dfine_uavswarm_all_layer_strip_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Submit from the project root:
# mkdir -p logs/dfine_uavswarm_all_layer_strip_full
# sbatch slurm_files/run_train_dfine_uavswarm_all_layer_strip_full.slurm
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
CONDA_ENV=${CONDA_ENV:-dfine}
CONFIG_FILE=${CONFIG_FILE:-configs/dfine/dfine_hgnetv2_l_uavswarm_all_layer_strip_full.yml}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/cluster/home/zzy/D-FINE/output/uavswarm_strip_beststg2_20260923}
OUTPUT_DIR=${OUTPUT_DIR:-$EXPERIMENT_ROOT/all_layer_strip_full}
cd "$PROJECT_DIR"
test -f "$CONFIG_FILE"
test -f weight/hgnetv2/PPHGNetV2_B4_stage1.pth
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Fresh all-layer strip output already exists: $OUTPUT_DIR. Use a new OUTPUT_DIR or resume through train.py." >&2
  exit 1
fi
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate "$CONDA_ENV"
THREADS_PER_RANK=$((SLURM_CPUS_PER_TASK / 8))
export OMP_NUM_THREADS="$THREADS_PER_RANK"
export MKL_NUM_THREADS="$THREADS_PER_RANK"
export PYTHONUNBUFFERED=1
JOB_NUMBER=$((10#$SLURM_JOB_ID))
MASTER_PORT=$((20000 + JOB_NUMBER % 20000))
echo "Job: $SLURM_JOB_ID | All-layer strip full training, 80 epochs | Local backbone pretraining | Port: $MASTER_PORT"
torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
  train.py -c "$CONFIG_FILE" \
  --output-dir "$OUTPUT_DIR" --use-amp --seed=0
