#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

set -euo pipefail
PROJECT_DIR=${1:?new diagnostic source snapshot required}
CONFIG_FILE=${2:?original frozen config required}
CHECKPOINT_FILE=${3:?NaN.pth or last.pth model checkpoint required}
DIAGNOSTIC_ROOT=${4:?fresh diagnostic directory required}
cd "$PROJECT_DIR"
test -f "$CONFIG_FILE"
test -f "$CHECKPOINT_FILE"
test ! -e "$DIAGNOSTIC_ROOT"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp
exec python tools/diagnose_visdrone_prediction_safety.py \
  --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_FILE" \
  --output-dir "$DIAGNOSTIC_ROOT" --epoch 92 --seed 0 \
  --max-batches 16 --max-seconds 180 --fp32-replay
