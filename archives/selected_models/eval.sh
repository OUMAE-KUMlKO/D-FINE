#!/usr/bin/env bash
# 本文件复制到每个模型归档根目录后，可从任意工作目录运行。
set -euo pipefail
MODEL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$MODEL_ROOT"
PYTHON_BIN=${PYTHON_BIN:-/home/zzy/.conda/envs/dfine/bin/python}
NPROC=${NPROC:-8}
MANIFEST_VALUES=$("$PYTHON_BIN" - <<'PY'
import json
with open("manifest.json", encoding="utf-8") as handle:
    manifest = json.load(handle)
for key in ("checkpoint", "seed", "eval_world_size", "eval_global_batch", "weights"):
    print(manifest[key])
PY
)
mapfile -t SETTINGS <<< "$MANIFEST_VALUES"
CHECKPOINT=${CHECKPOINT:-${SETTINGS[0]}}
SEED=${SETTINGS[1]}
ORIGINAL_WORLD_SIZE=${SETTINGS[2]}
ORIGINAL_GLOBAL_BATCH=${SETTINGS[3]}
WEIGHTS=${SETTINGS[4]}
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || { echo 'NPROC 必须为正整数' >&2; exit 2; }
(( ORIGINAL_GLOBAL_BATCH % ORIGINAL_WORLD_SIZE == 0 ))
GLOBAL_BATCH=$((ORIGINAL_GLOBAL_BATCH / ORIGINAL_WORLD_SIZE * NPROC))
case "$WEIGHTS" in
  ema) USE_EMA=True ;;
  model) USE_EMA=False ;;
  *) echo "未知权重类型: $WEIGHTS" >&2; exit 2 ;;
esac
CONFIG=${CONFIG:-archive_eval.yml}
OUTPUT_DIR=${OUTPUT_DIR:-evaluation/run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}_$$}
[[ "$CONFIG" = /* ]] || CONFIG="$MODEL_ROOT/$CONFIG"
[[ "$CHECKPOINT" = /* ]] || CHECKPOINT="$MODEL_ROOT/$CHECKPOINT"
[[ "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="$MODEL_ROOT/$OUTPUT_DIR"
test -f "$CONFIG"
test -f "$CHECKPOINT"
test -f "$MODEL_ROOT/source/train.py"
if [[ -e "$OUTPUT_DIR/eval.pth" || -e "$OUTPUT_DIR/metrics.json" || -e "$OUTPUT_DIR/console.log" ]]; then
  echo "输出目录已有复评产物，请选择新 OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$SEED"
export PYTHONPATH="$MODEL_ROOT/source"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
COMMAND=("$PYTHON_BIN")
if (( NPROC > 1 )); then
  COMMAND+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC")
fi
COMMAND+=("$MODEL_ROOT/source/train.py" -c "$CONFIG" -r "$CHECKPOINT"
  --test-only --use-amp -d cuda --seed "$SEED"
  --output-dir "$OUTPUT_DIR" --summary-dir "$OUTPUT_DIR/summary"
  -u "use_ema=$USE_EMA" use_wandb=False HGNetv2.pretrained=False
  "val_dataloader.total_batch_size=$GLOBAL_BATCH")
if [[ -n "${IMAGE_ROOT:-}" ]]; then
  # JSON 字符串也是合法 YAML；保留路径中的空格及特殊字符。
  IMAGE_VALUE=$("$PYTHON_BIN" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$IMAGE_ROOT")
  COMMAND+=("val_dataloader.dataset.img_folder=$IMAGE_VALUE")
fi
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$MODEL_ROOT"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} > "$OUTPUT_DIR/command.sh"
"${COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/console.log"
STATUS=0
"$PYTHON_BIN" "$MODEL_ROOT/summarize_eval.py" \
  --manifest "$MODEL_ROOT/manifest.json" --evaluation-dir "$OUTPUT_DIR" \
  --checkpoint "$CHECKPOINT" --config "$CONFIG" \
  2>&1 | tee -a "$OUTPUT_DIR/console.log" || STATUS=$?
"$PYTHON_BIN" - "$MODEL_ROOT" "$OUTPUT_DIR" <<'PY'
import os
import sys
from pathlib import Path
root, output = map(Path, sys.argv[1:])
directory = root / "evaluation"
directory.mkdir(exist_ok=True)
temporary = directory / f".latest.{os.getpid()}"
temporary.symlink_to(os.path.relpath(output, directory))
temporary.replace(directory / "latest")
PY
exit "$STATUS"
