#!/bin/bash
#SBATCH --job-name=dfine_g5_geometry_15526
#SBATCH --output=logs/dfine_g5_geometry_15526/geometry_%j.out
#SBATCH --error=logs/dfine_g5_geometry_15526/geometry_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

# Full validation only; reads EMA checkpoints and never writes training weights.
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
GEOMETRY_OUTPUT=${GEOMETRY_OUTPUT:-output/g5_geometry_15526_20260929/summary.json}
cd "$PROJECT_DIR"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp
python tools/analysis/evaluate_detector_geometry.py \
  --baseline-checkpoint /cluster/home/zzy/D-FINE/output/uavswarm_strip_20260923/b0/best_stg2.pth \
  --model-checkpoint /cluster/home/zzy/D-FINE/output/uavswarm_strip_beststg2_20260923/all_layer_band2d_query_gate_full/best_stg2.pth \
  --output-file "$GEOMETRY_OUTPUT" --batch-size 8 --workers 4 --threads 4 --device cuda
