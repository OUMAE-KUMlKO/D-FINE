#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

set -euo pipefail
PROJECT_DIR=${1:?frozen source required}
PREFLIGHT_ROOT=${2:?fresh preflight directory required}
cd "$PROJECT_DIR"
test ! -e "$PREFLIGHT_ROOT"
mkdir -p "$PREFLIGHT_ROOT"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp
for GROUP in g5_adaptive_band_gate g5_adaptive_band_pre_reference g5_adaptive_band_gate_pre_reference; do
  python tools/preflight_visdrone_gate_reference.py \
    --config "configs/controlled/visdrone_$GROUP.yml" \
    --output-file "$PREFLIGHT_ROOT/$GROUP.json"
done
