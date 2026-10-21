#!/bin/bash
#SBATCH --job-name=dfine-ddf-off
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu10,gpu18
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --no-requeue

# Submit with explicit --output/--error paths inside EXPERIMENT_ROOT/logs.
set -euo pipefail
: "${SOURCE_DIR:?Set SOURCE_DIR to the frozen controlled multi-seed runtime}"
: "${EXPERIMENT_ROOT:?Set EXPERIMENT_ROOT to an independent experiment root}"
: "${DATASET:?Set DATASET to uavswarm or visdrone}"
: "${TRAIN_SEED:?Set TRAIN_SEED to 0, 1 or 2}"
case "$DATASET" in uavswarm|visdrone) ;; *) exit 2 ;; esac
case "$TRAIN_SEED" in 0|1|2) ;; *) exit 2 ;; esac

PYTHON_BIN=/home/zzy/.conda/envs/dfine/bin/python
CONFIG_FILE="$SOURCE_DIR/configs/dfine/ablation/dfine_hgnetv2_l_${DATASET}_baseline_ddf_off.yml"
GROUP_DIR="$EXPERIMENT_ROOT/results/$DATASET/seed${TRAIN_SEED}"
test -f "$CONFIG_FILE"
test -f "$EXPERIMENT_ROOT/manifest/provenance.json"
mkdir -p "$EXPERIMENT_ROOT/results/$DATASET"
# Never overwrite or automatically resume an existing run.
mkdir "$GROUP_DIR"

module load gcc/8.3.0
module load cuda/cuda12.4
source /soft/anaconda3/bin/activate dfine
cd "$SOURCE_DIR"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 8))
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$TRAIN_SEED"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dfine-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/dfine-cache-${SLURM_JOB_ID}"
MASTER_PORT=$((20000 + 10#$SLURM_JOB_ID % 20000))

"$PYTHON_BIN" tools/check_baseline_ddf_off.py \
  --source-dir "$SOURCE_DIR" --dataset "$DATASET" --seed "$TRAIN_SEED" \
  --output-dir "$GROUP_DIR" --report-dir "$GROUP_DIR/manifest"

TRAIN_COMMAND=("$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node=8
  --master_port="$MASTER_PORT" train.py -c "$CONFIG_FILE"
  --output-dir "$GROUP_DIR" --use-amp --seed="$TRAIN_SEED"
  -u "controlled_seed=$TRAIN_SEED"
  "exp_name=dfine_l_${DATASET}_baseline_ddf_off_seed${TRAIN_SEED}")
printf '%q ' "${TRAIN_COMMAND[@]}" > "$GROUP_DIR/manifest/run_command.txt"
printf '\n' >> "$GROUP_DIR/manifest/run_command.txt"
echo "Job=$SLURM_JOB_ID | native D-FINE | dataset=$DATASET | seed=$TRAIN_SEED | DDF=0 | fresh 80 epochs"
"${TRAIN_COMMAND[@]}" 2>&1 | tee "$GROUP_DIR/train.log"
