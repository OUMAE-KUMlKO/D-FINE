#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

set -euo pipefail
PROJECT_DIR=${1:?frozen source required}
MODE=${2:?preflight or train required}
RESULT_ROOT=${3:?fresh experiment directory required}
PREFLIGHT_FILE=${4:?full-training preflight JSON required}
CONFIG_80=configs/controlled/visdrone_g5_gt_candidate.yml
CONFIG_120=configs/controlled/visdrone_g5_gt_candidate_continue120.yml
cd "$PROJECT_DIR"
test -f "$CONFIG_80"
test -f "$CONFIG_120"
test -f weight/hgnetv2/PPHGNetV2_B4_stage1.pth
test -f output/visdrone_oracle/instances_val_positive_ids.json
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dfine-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/dfine-cache-${SLURM_JOB_ID}"

if [[ "$MODE" == preflight ]]; then
  exec python tools/analysis/check_visdrone_g5_gt.py \
    --config "$CONFIG_80" --fresh-init --train-size 800 \
    --output-file "$PREFLIGHT_FILE" --batch-size 4 --device cuda
fi
[[ "$MODE" == train ]] || { echo "Unknown mode: $MODE" >&2; exit 2; }
python - "$PREFLIGHT_FILE" "$CONFIG_80" <<'PY'
import json
from pathlib import Path
import sys
report = json.loads(Path(sys.argv[1]).read_text())
assert report['status'] == 'passed' and report['successful_updates'] == 2
assert report['model_class'] == 'DFINEGTAllLayerStrip'
assert report['initialization'] == 'imagenet_fresh'
assert report['batch_size'] == 4
assert report['training_image_size'] == [800, 800]
assert report['ema_checked'] is True
assert Path(report['config']).resolve() == Path(sys.argv[2]).resolve()
PY
echo "Job=$SLURM_JOB_ID | G5 GT candidate guidance | full 80+40 epochs | seed0"
# Reuse the established stage transition and complete-state continuation.
exec bash slurm_files/run_train_visdrone_gate_reference.slurm \
  "$PROJECT_DIR" "$CONFIG_80" "$CONFIG_120" "$RESULT_ROOT" 0
