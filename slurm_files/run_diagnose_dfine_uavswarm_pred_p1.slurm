#!/bin/bash
#SBATCH --job-name=dfine_p1_diagnostic
#SBATCH --output=logs/dfine_p1_diagnostic/dfine_p1_diagnostic_%j.out
#SBATCH --error=logs/dfine_p1_diagnostic/dfine_p1_diagnostic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

# mkdir -p logs/dfine_p1_diagnostic
# sbatch slurm_files/run_diagnose_dfine_uavswarm_pred_p1.slurm
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
CONDA_ENV=${CONDA_ENV:-dfine}
P1_CHECKPOINT=${P1_CHECKPOINT:-/cluster/home/zzy/D-FINE/output/uavswarm_strip_beststg2_20260923/p1/best_stg1.pth}
DIAGNOSTIC_OUTPUT=${DIAGNOSTIC_OUTPUT:-output/p1_diagnostics_beststg2/summary.json}
cd "$PROJECT_DIR"
test -f "$P1_CHECKPOINT"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate "$CONDA_ENV"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python tools/analysis/diagnose_pred_refinement.py \
  --checkpoint "$P1_CHECKPOINT" \
  --output-file "$DIAGNOSTIC_OUTPUT" \
  --device cuda --batch-size 8 --workers 4 --threads 4
