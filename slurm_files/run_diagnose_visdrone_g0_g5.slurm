#!/bin/bash
#SBATCH --job-name=dfine_visdrone_geometry
#SBATCH --output=logs/dfine_visdrone_diagnostics/geometry_%j.out
#SBATCH --error=logs/dfine_visdrone_diagnostics/geometry_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=00:40:00
#SBATCH --partition=gpu

# Three independent G0/G5 checkpoint comparisons; validation only.
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
DIAGNOSTIC_ROOT=${DIAGNOSTIC_ROOT:-output/visdrone_diagnostics_20261006}
cd "$PROJECT_DIR"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp
for experiment_seed in 0 1 2; do
  if [[ "$experiment_seed" == 0 ]]; then
    weights_root=output/controlled_20260926_212304/results/visdrone
  else
    weights_root=output/controlled_visdrone_20260929_seed12/results/visdrone/seed${experiment_seed}
  fi
  echo "VisDrone geometry seed=${experiment_seed} | checkpoints=${weights_root}"
  python tools/analysis/evaluate_detector_geometry.py \
    --baseline-config configs/controlled/visdrone_g0.yml \
    --model-config configs/controlled/visdrone_g5.yml \
    --baseline-checkpoint "${weights_root}/g0/best_stg2.pth" \
    --model-checkpoint "${weights_root}/g5/best_stg2.pth" \
    --output-file "${DIAGNOSTIC_ROOT}/geometry_seed${experiment_seed}.json" \
    --batch-size 8 --workers 4 --threads 4 --device cuda --detailed-diagnostics
done
