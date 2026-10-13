#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

set -euo pipefail
PROJECT_DIR=${1:?frozen source directory required}
DIAGNOSTIC_ROOT=${2:?fresh result directory required}
CHECKPOINT_FILE=${3:?existing best checkpoint required}
cd "$PROJECT_DIR"
test -f "$CHECKPOINT_FILE"
if [[ -e "$DIAGNOSTIC_ROOT" ]]; then
  echo "Output already exists: $DIAGNOSTIC_ROOT" >&2
  exit 1
fi
mkdir -p "$DIAGNOSTIC_ROOT"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp
echo "Job=$SLURM_JOB_ID | fixed EMA=$CHECKPOINT_FILE | normal/zero/noise inference only"
exec python tools/analysis/diagnose_candidate_features.py \
  --config configs/controlled/visdrone_g5_adaptive_band_continue120.yml \
  --checkpoint "$CHECKPOINT_FILE" \
  --output-file "$DIAGNOSTIC_ROOT/summary.json" \
  --prediction-cache "$DIAGNOSTIC_ROOT/predictions.pth" \
  --batch-size 4 --workers 4 --threads 4 --device cuda --seed 0 \
  --expected-ap 28.4190 --ap-tolerance 0.02
