#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
set -euo pipefail
PROJECT_DIR=${1:?frozen source required}
DIAGNOSTIC_ROOT=${2:?error diagnostic root required}
REFERENCE_ROOT=${3:?completed G5 feature-ablation results required}
RESULT_FILE=${4:-"$DIAGNOSTIC_ROOT/results/summary.json"}
cd "$PROJECT_DIR"
module load gcc/8.3.0
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp
exec python tools/analysis/compare_detection_errors.py \
  --g0-summary "$DIAGNOSTIC_ROOT/g0/summary.json" \
  --g0-cache "$DIAGNOSTIC_ROOT/g0/predictions.pth" \
  --g5-summary "$REFERENCE_ROOT/summary.json" \
  --g5-cache "$REFERENCE_ROOT/predictions.pth" \
  --ann-file output/visdrone_controlled/instances_val.json \
  --output-file "$RESULT_FILE" --threads 4
