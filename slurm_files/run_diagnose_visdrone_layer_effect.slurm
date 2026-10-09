#!/bin/bash
#SBATCH --job-name=dfine_visdrone_layer_effect
#SBATCH --output=logs/dfine_visdrone_layer_effect/layers_%j.out
#SBATCH --error=logs/dfine_visdrone_layer_effect/layers_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
DIAGNOSTIC_ROOT=${DIAGNOSTIC_ROOT:-output/visdrone_layer_effect_20261009}
cd "$PROJECT_DIR"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp

for variant in adaptive fixed; do
  if [[ "$variant" == adaptive ]]; then
    config=configs/controlled/visdrone_g5_adaptive_band.yml
    checkpoint=output/controlled_visdrone_adaptive_band_20261006_seed0/results/visdrone/seed0/g5_adaptive_band/best_stg2.pth
  else
    config=configs/controlled/visdrone_g5.yml
    checkpoint=output/controlled_20260926_212304/results/visdrone/g5/best_stg2.pth
  fi
  echo "Layer effect: variant=$variant, seed=0, checkpoint=$checkpoint"
  python tools/analysis/diagnose_all_layer_strip.py \
    --config "$config" --checkpoint "$checkpoint" \
    --output-file "${DIAGNOSTIC_ROOT}/${variant}.json" \
    --trace-cache "${DIAGNOSTIC_ROOT}/${variant}_traces.pth" \
    --batch-size 8 --workers 4 --threads 4 --device cuda
done
