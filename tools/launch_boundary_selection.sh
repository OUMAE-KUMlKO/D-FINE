#!/usr/bin/env bash
set -euo pipefail
# Run from the frozen source directory, prepared before launch.
export CUDA_VISIBLE_DEVICES=0,2,3,4,6,7
export OMP_NUM_THREADS=4
exec /root/anaconda3/envs/dfine_py3.10/bin/python -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=6 \
  tools/train_boundary_selection.py --config configs/dfine/boundary_selection/selection.yml
