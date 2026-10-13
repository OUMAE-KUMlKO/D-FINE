#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18,gpu20
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

set -euo pipefail
PROJECT_DIR=${1:?frozen source required}
CONFIG_120=${2:?continuation config required}
EXPERIMENT_ROOT=${3:?fresh experiment directory required}
RESUME_CHECKPOINT=${4:?epoch91 checkpoint required}
HISTORICAL_BEST=${5:?epoch89 historical best required}
EXPERIMENT_SEED=${6:-0}
HISTORICAL_LOG=$(dirname "$HISTORICAL_BEST")/log.txt

cd "$PROJECT_DIR"
test -f "$CONFIG_120"
test -f "$RESUME_CHECKPOINT"
test -f "$HISTORICAL_BEST"
test -f "$HISTORICAL_LOG"
if [[ -e "$EXPERIMENT_ROOT" ]]; then
  echo "Output already exists: $EXPERIMENT_ROOT" >&2
  exit 1
fi

module load gcc/8.3.0
source /soft/anaconda3/bin/activate dfine
python - "$RESUME_CHECKPOINT" "$HISTORICAL_BEST" <<'PY'
import sys
import torch

required = ('model', 'optimizer', 'ema', 'scaler', 'lr_scheduler', 'lr_warmup_scheduler')
resume = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
historical = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
if resume['last_epoch'] != 91:
    raise RuntimeError(f"Expected resume epoch91, got {resume['last_epoch']}")
if historical['last_epoch'] != 89:
    raise RuntimeError(f"Expected historical best epoch89, got {historical['last_epoch']}")
for label, checkpoint in (('resume', resume), ('historical best', historical)):
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(f"{label} checkpoint missing state: {', '.join(missing)}")
PY

mkdir -p "$EXPERIMENT_ROOT/archive"
cp "$HISTORICAL_BEST" "$EXPERIMENT_ROOT/archive/pre_resume_best_stg2.pth"
python - "$EXPERIMENT_ROOT/resume_selection.json" "$RESUME_CHECKPOINT" "$HISTORICAL_BEST" <<'PY'
import json
from pathlib import Path
import sys

metadata = {
    'status': 'pending_training',
    'historical_true_best': {
        'source': sys.argv[3],
        'epoch': 89,
        'archive': 'archive/pre_resume_best_stg2.pth',
    },
    'resume_segment_selection': {
        'source': sys.argv[2],
        'epoch': 91,
        'detail': (
            'run_resume_dfine_controlled seeds continue120/best_stg2.pth from epoch91; '
            'DetSolver resets best_metric for this resume segment'
        ),
    },
    'final_selection': (
        'compare the historical_true_best archive with the new '
        'continue120/best_stg2.pth and select the higher metric'
    ),
    'failure_policy': 'no automatic retry',
}
Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2) + '\n')
PY

echo "Job=$SLURM_JOB_ID | E1 recovery | seed=$EXPERIMENT_SEED | epochs=92-119"
bash slurm_files/run_resume_dfine_controlled.slurm \
  "$PROJECT_DIR" "$CONFIG_120" "$EXPERIMENT_ROOT/continue120" \
  "$RESUME_CHECKPOINT" "$EXPERIMENT_SEED"

# The generic continuation selects only within epoch91-119. Compare that result
# with the archived epoch89 true best before allowing dependent jobs to start.
python - "$EXPERIMENT_ROOT/resume_selection.json" \
  "$EXPERIMENT_ROOT/archive/pre_resume_best_stg2.pth" \
  "$EXPERIMENT_ROOT/continue120/best_stg2.pth" \
  "$HISTORICAL_LOG" "$EXPERIMENT_ROOT/continue120/log.txt" \
  "$EXPERIMENT_ROOT/best_overall_stg2.pth" <<'PY'
import json
from pathlib import Path
import shutil
import sys

import torch


def metrics_by_epoch(*paths):
    metrics = {}
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                record = json.loads(line)
                values = record.get('test_coco_eval_bbox')
                if values is not None:
                    metrics[record['epoch']] = values[0]
    return metrics


metadata_path = Path(sys.argv[1])
historical_checkpoint = Path(sys.argv[2])
segment_checkpoint = Path(sys.argv[3])
metrics = metrics_by_epoch(sys.argv[4], sys.argv[5])
overall_checkpoint = Path(sys.argv[6])

segment_state = torch.load(segment_checkpoint, map_location='cpu', weights_only=False)
segment_epoch = segment_state.get('last_epoch')
if segment_epoch not in range(91, 120):
    raise RuntimeError(f'Unexpected resume-segment best epoch: {segment_epoch!r}')
for epoch in (89, segment_epoch):
    if epoch not in metrics:
        raise RuntimeError(f'Missing test_coco_eval_bbox for epoch{epoch}')

historical_ap = metrics[89]
segment_ap = metrics[segment_epoch]
if historical_ap >= segment_ap:
    selected_source = historical_checkpoint
    selected_label = 'historical_true_best'
    selected_epoch = 89
    selected_ap = historical_ap
else:
    selected_source = segment_checkpoint
    selected_label = 'resume_segment_selection'
    selected_epoch = segment_epoch
    selected_ap = segment_ap
shutil.copy2(selected_source, overall_checkpoint)

metadata = json.loads(metadata_path.read_text())
metadata['status'] = 'complete'
metadata['historical_true_best']['test_coco_eval_bbox_ap'] = historical_ap
metadata['resume_segment_selection'].update({
    'source': str(segment_checkpoint),
    'epoch': segment_epoch,
    'test_coco_eval_bbox_ap': segment_ap,
})
metadata['selected_checkpoint'] = {
    'candidate': selected_label,
    'source': str(selected_source),
    'output': str(overall_checkpoint),
    'epoch': selected_epoch,
    'test_coco_eval_bbox_ap': selected_ap,
}
metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
print(
    f'Selected overall E1 best: {selected_label}, epoch={selected_epoch}, AP={selected_ap}',
    flush=True,
)
PY
