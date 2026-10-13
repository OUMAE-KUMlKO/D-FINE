#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
set -euo pipefail
PROJECT_DIR=${1:?frozen source required}
DIAGNOSTIC_ROOT=${2:?fresh results directory required}
CHECKPOINT_FILE=${3:?G0 best checkpoint required}
REFERENCE_ROOT=${4:?completed feature-ablation results required}
cd "$PROJECT_DIR"
test -f "$CHECKPOINT_FILE"
test -f "$REFERENCE_ROOT/predictions.pth"
test ! -e "$DIAGNOSTIC_ROOT"
mkdir -p "$DIAGNOSTIC_ROOT"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp
exec python tools/analysis/cache_detector_predictions.py \
  --config configs/controlled/visdrone_g0_continue120.yml \
  --checkpoint "$CHECKPOINT_FILE" \
  --reference-summary "$REFERENCE_ROOT/summary.json" \
  --reference-cache "$REFERENCE_ROOT/predictions.pth" \
  --output-file "$DIAGNOSTIC_ROOT/summary.json" \
  --prediction-cache "$DIAGNOSTIC_ROOT/predictions.pth" \
  --expected-ap 28.622720199142976 --batch-size 4 --workers 4 --threads 4 --device cuda
