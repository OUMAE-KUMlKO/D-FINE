#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

set -euo pipefail
PROJECT_DIR=${1:?frozen source required}
MODE=${2:?preflight or train required}
EXPERIMENT=${3:?p2 or g5_gt required}
INITIAL_CHECKPOINT=${4:?initial checkpoint required}
RESULT_ROOT=${5:?fresh result root required}
PREFLIGHT_FILE=${6:?preflight JSON required}
case "$EXPERIMENT" in
  p2)
    CONFIG_FILE=configs/dfine/pred_refinement/dfine_hgnetv2_l_visdrone_pred_p2.yml
    MODEL_CLASS=DFINEPredRefiner
    PER_GPU_BATCH=12
    ;;
  g5_gt)
    CONFIG_FILE=configs/dfine/gt_refinement/dfine_hgnetv2_l_visdrone_g5_gt_candidate.yml
    MODEL_CLASS=DFINEGTAllLayerStrip
    PER_GPU_BATCH=4
    ;;
  *) echo "Unknown experiment: $EXPERIMENT" >&2; exit 2 ;;
esac
cd "$PROJECT_DIR"
test -f "$CONFIG_FILE"
test -f "$INITIAL_CHECKPOINT"
test -f output/visdrone_oracle/instances_val_positive_ids.json
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dfine-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/dfine-cache-${SLURM_JOB_ID}"

if [[ "$MODE" == preflight ]]; then
  if [[ "$EXPERIMENT" == p2 ]]; then
    exec python tools/analysis/check_visdrone_p2.py \
      --config "$CONFIG_FILE" --baseline-checkpoint "$INITIAL_CHECKPOINT" \
      --output-file "$PREFLIGHT_FILE" --batch-size "$PER_GPU_BATCH" --device cuda
  else
    exec python tools/analysis/check_visdrone_g5_gt.py \
      --config "$CONFIG_FILE" --checkpoint "$INITIAL_CHECKPOINT" \
      --output-file "$PREFLIGHT_FILE" --batch-size "$PER_GPU_BATCH" --device cuda
  fi
fi
[[ "$MODE" == train ]] || { echo "Unknown mode: $MODE" >&2; exit 2; }
[[ ! -e "$RESULT_ROOT" ]] || { echo "Refusing existing results: $RESULT_ROOT" >&2; exit 1; }
python - "$PREFLIGHT_FILE" "$INITIAL_CHECKPOINT" "$CONFIG_FILE" "$MODEL_CLASS" "$PER_GPU_BATCH" <<'PY'
import json
from pathlib import Path
import sys
report = json.loads(Path(sys.argv[1]).read_text())
assert report['status'] == 'passed' and report['successful_updates'] == 2
assert report['model_class'] == sys.argv[4]
assert report['batch_size'] == int(sys.argv[5])
checkpoint = report.get('baseline_checkpoint', report.get('checkpoint'))
assert Path(checkpoint).resolve() == Path(sys.argv[2]).resolve()
assert Path(report['config']).resolve() == Path(sys.argv[3]).resolve()
PY
mkdir -p "$RESULT_ROOT"
MASTER_PORT=$((20000 + 10#$SLURM_JOB_ID % 20000))
echo "Job=$SLURM_JOB_ID experiment=$EXPERIMENT checkpoint=$INITIAL_CHECKPOINT"
if [[ "$EXPERIMENT" == p2 ]]; then
  torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
    train.py -c "$CONFIG_FILE" --seed=0 --use-amp \
    -u "DFINEPredRefiner.baseline_checkpoint=$INITIAL_CHECKPOINT" \
    --output-dir "$RESULT_ROOT/p2"
  # This evaluates the same trained head; it never modifies its checkpoint.
  python tools/analysis/evaluate_visdrone_p2.py \
    --config "$CONFIG_FILE" --checkpoint "$RESULT_ROOT/p2/best_stg1.pth" \
    --output-file "$RESULT_ROOT/p2_pooling_evaluation.json" --batch-size 4 --device cuda
else
  exec torchrun --nnodes=1 --nproc_per_node=8 --master_port="$MASTER_PORT" \
    train.py -c "$CONFIG_FILE" --seed=0 --use-amp \
    --tuning "$INITIAL_CHECKPOINT" --output-dir "$RESULT_ROOT/g5_gt"
fi
