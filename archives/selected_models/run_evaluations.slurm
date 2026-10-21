#!/usr/bin/env bash
#SBATCH --job-name=archive-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

# 提交时用 sbatch --output=... --error=... 显式指定归档目录中的日志文件。
set -euo pipefail
ARCHIVE_ROOT=${ARCHIVE_ROOT:-/cluster/home/zzy/D-FINE/desired_results}
module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
export PYTHON_BIN=${PYTHON_BIN:-/home/zzy/.conda/envs/dfine/bin/python}
export NPROC=${NPROC:-8}
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$ARCHIVE_ROOT/logs"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
  > "$ARCHIVE_ROOT/logs/evaluation_hardware_${SLURM_JOB_ID}.txt"
MODEL_IDS=$("$PYTHON_BIN" - "$ARCHIVE_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
index = root / "index.json"
if not index.exists():
    index = root / "manifest.json"
data = json.loads(index.read_text(encoding="utf-8"))
models = data["models"] if isinstance(data, dict) else data
if not models:
    raise ValueError("归档索引没有模型")
for model in models:
    print(model["id"] if isinstance(model, dict) else model)
PY
)
mapfile -t MODELS <<< "$MODEL_IDS"
STATUS=0
for MODEL_ID in "${MODELS[@]}"; do
  echo "[$(date --iso-8601=seconds)] 开始复评: $MODEL_ID"
  if bash "$ARCHIVE_ROOT/$MODEL_ID/eval.sh"; then
    echo "[$(date --iso-8601=seconds)] 复评通过: $MODEL_ID"
  else
    MODEL_STATUS=$?
    echo "[$(date --iso-8601=seconds)] 复评失败: $MODEL_ID，退出码 $MODEL_STATUS；保留现场并继续" >&2
    STATUS=1
  fi
done
exit "$STATUS"
