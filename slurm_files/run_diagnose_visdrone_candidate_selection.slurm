#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

set -euo pipefail
PROJECT_DIR=${1:?frozen source directory required}
RESULT_ROOT=${2:?fresh diagnostic result directory required}
G5_CHECKPOINT=${3:?G5 best EMA checkpoint required}
ORACLE_CHECKPOINT=${4:?trained GT candidate oracle checkpoint required}
cd "$PROJECT_DIR"
test -f "$G5_CHECKPOINT"
test -f "$ORACLE_CHECKPOINT"
if [[ -e "$RESULT_ROOT" ]]; then
  echo "Output already exists: $RESULT_ROOT" >&2
  exit 1
fi
mkdir -p "$RESULT_ROOT"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dfine-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/dfine-cache-${SLURM_JOB_ID}"
ANNOTATIONS=output/visdrone_oracle/instances_val_positive_ids.json
test -f "$ANNOTATIONS"
echo "Job=$SLURM_JOB_ID | candidate selection diagnostic | existing checkpoints, FP32 inference only"
python tools/analysis/diagnose_candidate_selection.py \
  --stage oracle \
  --config configs/dfine/gt_refinement/dfine_hgnetv2_l_visdrone_gt_candidate_oracle.yml \
  --checkpoint "$ORACLE_CHECKPOINT" --validation-annotations "$ANNOTATIONS" \
  --output-file "$RESULT_ROOT/oracle_pooling.json" \
  --prediction-cache "$RESULT_ROOT/oracle_predictions.pth" \
  --batch-size 4 --workers 4 --threads 4
exec python tools/analysis/diagnose_candidate_selection.py \
  --stage g5 \
  --config configs/controlled/visdrone_g5_adaptive_band_continue120.yml \
  --checkpoint "$G5_CHECKPOINT" --validation-annotations "$ANNOTATIONS" \
  --output-file "$RESULT_ROOT/g5_scores.json" \
  --batch-size 4 --workers 4 --threads 4
