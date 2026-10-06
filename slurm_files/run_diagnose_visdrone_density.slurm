#!/bin/bash
#SBATCH --job-name=dfine_visdrone_density
#SBATCH --output=logs/dfine_visdrone_density/density_%j.out
#SBATCH --error=logs/dfine_visdrone_density/density_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

# Cache raw validation predictions, then reuse them for density diagnostics.
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
DIAGNOSTIC_ROOT=${DIAGNOSTIC_ROOT:-output/visdrone_density_diagnostics_20261006}
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
  cache_file="${DIAGNOSTIC_ROOT}/predictions_seed${experiment_seed}.pth"
  echo "VisDrone density seed=${experiment_seed}"
  python tools/analysis/evaluate_detector_geometry.py \
    --baseline-config configs/controlled/visdrone_g0.yml \
    --model-config configs/controlled/visdrone_g5.yml \
    --baseline-checkpoint "${weights_root}/g0/best_stg2.pth" \
    --model-checkpoint "${weights_root}/g5/best_stg2.pth" \
    --output-file "${DIAGNOSTIC_ROOT}/geometry_seed${experiment_seed}.json" \
    --prediction-cache "$cache_file" \
    --batch-size 8 --workers 4 --threads 4 --device cuda --detailed-diagnostics
  python tools/analysis/analyze_detector_density.py \
    --prediction-cache "$cache_file" \
    --output-file "${DIAGNOSTIC_ROOT}/density_seed${experiment_seed}.json"
  python tools/analysis/evaluate_density_coco.py \
    --prediction-cache "$cache_file" \
    --output-file "${DIAGNOSTIC_ROOT}/coco_density_seed${experiment_seed}.json" --threads 4
done
