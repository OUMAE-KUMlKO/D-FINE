#!/bin/bash
#SBATCH --job-name=dfine_p2_check
#SBATCH --output=logs/dfine_p2_check/dfine_p2_check_%j.out
#SBATCH --error=logs/dfine_p2_check/dfine_p2_check_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=00:20:00
#SBATCH --partition=gpu

# Single real batch, two disposable optimizer steps, no model checkpoint saved.
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
CONDA_ENV=${CONDA_ENV:-dfine}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:-/cluster/home/zzy/D-FINE/output/uavswarm_strip_20260923/b0/best_stg2.pth}
CHECK_OUTPUT=${CHECK_OUTPUT:-output/p2_check_beststg2/summary.json}
cd "$PROJECT_DIR"
test -f "$BASELINE_CHECKPOINT"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate "$CONDA_ENV"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python tools/analysis/check_pred_distribution.py \
  --baseline-checkpoint "$BASELINE_CHECKPOINT" --output-file "$CHECK_OUTPUT" \
  --device cuda --batch-size 12 --workers 4 --threads 4 --max-batches 1 --check-backward
