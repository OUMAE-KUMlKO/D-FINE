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
CONFIG_80=${2:?80-epoch config required}
CONFIG_120=${3:?continuation config required}
EXPERIMENT_ROOT=${4:?fresh experiment directory required}
EXPERIMENT_SEED=${5:-0}
cd "$PROJECT_DIR"
test -f "$CONFIG_80"
test -f "$CONFIG_120"
if [[ -e "$EXPERIMENT_ROOT" ]]; then
  echo "Output already exists: $EXPERIMENT_ROOT" >&2
  exit 1
fi
mkdir -p "$EXPERIMENT_ROOT"
echo "Job=$SLURM_JOB_ID | approved new variant | seed=$EXPERIMENT_SEED | 80+40 epochs"
# Each child script execs its own torchrun; the parent must continue after stage 1.
bash slurm_files/run_train_dfine_controlled.slurm \
  "$PROJECT_DIR" "$CONFIG_80" "$EXPERIMENT_ROOT/epoch80" "$EXPERIMENT_SEED"
test -f "$EXPERIMENT_ROOT/epoch80/last.pth"
module load gcc/8.3.0
source /soft/anaconda3/bin/activate dfine
python - "$EXPERIMENT_ROOT/epoch80/last.pth" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
if checkpoint['last_epoch'] != 79:
    raise RuntimeError(f"Expected a completed epoch79, got {checkpoint['last_epoch']}")
for key in ('model', 'optimizer', 'ema', 'scaler', 'lr_scheduler', 'lr_warmup_scheduler'):
    if key not in checkpoint:
        raise RuntimeError(f'Missing resume state: {key}')
print('Stage 1 complete; starting the planned epoch80-119 continuation.', flush=True)
PY
exec bash slurm_files/run_resume_dfine_controlled.slurm \
  "$PROJECT_DIR" "$CONFIG_120" "$EXPERIMENT_ROOT/continue120" \
  "$EXPERIMENT_ROOT/epoch80/last.pth" "$EXPERIMENT_SEED"
