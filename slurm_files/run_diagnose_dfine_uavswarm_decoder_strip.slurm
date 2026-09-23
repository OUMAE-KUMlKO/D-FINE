#!/bin/bash
#SBATCH --job-name=dfine_decoder_strip_diagnostic
#SBATCH --output=logs/dfine_decoder_strip_diagnostic/dfine_decoder_strip_diagnostic_%j.out
#SBATCH --error=logs/dfine_decoder_strip_diagnostic/dfine_decoder_strip_diagnostic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=00:20:00
#SBATCH --partition=gpu

# Fixed 96-image geometry diagnostic. No training and no checkpoint writes.
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE}
CONDA_ENV=${CONDA_ENV:-dfine}
DIAGNOSTIC_OUTPUT=${DIAGNOSTIC_OUTPUT:-output/decoder_strip_diagnostics_beststg2/summary.json}
cd "$PROJECT_DIR"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate "$CONDA_ENV"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python tools/analysis/diagnose_decoder_strip.py \
  --output-file "$DIAGNOSTIC_OUTPUT" --samples 96 --batch-size 8 \
  --workers 4 --threads 4 --device cuda
