#!/bin/bash
#SBATCH --job-name=vd-sparse-strip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE_DIR:?Set SOURCE_DIR to the frozen implementation source}"
: "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to the paired experiment root}"
: "${GROUP:?Set GROUP to control or sparse}"
case "$GROUP" in control|sparse) ;; *) exit 2 ;; esac
PYTHON_BIN=/home/zzy/.conda/envs/dfine/bin/python
CHECKPOINT=${CHECKPOINT:-/home/zzy/projects/occ_study/D-FINE_mine/D-FINE/output/visdrone_g5_ddf_off_20261019_103206/results/g5_ddf_off/best_stg2.pth}
PREFLIGHT_JSON=${PREFLIGHT_JSON:-$EXPERIMENT_ROOT/preflight.json}
"$PYTHON_BIN" - "$PREFLIGHT_JSON" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    report = json.load(handle)
if report.get('status') != 'passed':
    raise SystemExit('Required sparse-strip preflight has not passed')
PY
test -f "$CHECKPOINT"
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
cd "$SOURCE_DIR"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 8))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/sparse-strip-inductor-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sparse-strip-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/sparse-strip-cache-${SLURM_JOB_ID}"
MASTER_PORT=$((20000 + 10#$SLURM_JOB_ID % 20000))
exec "$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --master_port="$MASTER_PORT" tools/train_sparse_strip_query.py \
  --config "configs/dfine/sparse_strip_query/$GROUP.yml" \
  --checkpoint "$CHECKPOINT" --output-dir "$EXPERIMENT_ROOT/results/$GROUP" --seed 0
