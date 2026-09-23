#!/bin/bash
#SBATCH --job-name=dfine_p2_diagnostic
#SBATCH --output=logs/dfine_p2_diagnostic/dfine_p2_diagnostic_%j.out
#SBATCH --error=logs/dfine_p2_diagnostic/dfine_p2_diagnostic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

# mkdir -p logs/dfine_p2_diagnostic
# sbatch slurm_files/run_diagnose_dfine_uavswarm_pred_p2.slurm
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
CONDA_ENV=${CONDA_ENV:-dfine}
P2_CHECKPOINT=${P2_CHECKPOINT:-/cluster/home/zzy/D-FINE/output/uavswarm_strip_beststg2_20260923/p2/best_stg1.pth}
DIAGNOSTIC_OUTPUT=${DIAGNOSTIC_OUTPUT:-output/p2_diagnostics_beststg2/summary.json}
cd "$PROJECT_DIR"
test -f "$P2_CHECKPOINT"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate "$CONDA_ENV"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

python tools/analysis/diagnose_pred_refinement.py \
  --config configs/dfine/pred_refinement/dfine_hgnetv2_l_uavswarm_pred_p2.yml \
  --checkpoint "$P2_CHECKPOINT" \
  --output-file "$DIAGNOSTIC_OUTPUT" \
  --device cuda --batch-size 8 --workers 4 --threads 4
