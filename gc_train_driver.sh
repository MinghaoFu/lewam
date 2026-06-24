#!/usr/bin/env bash
# Per-task GC head-to-head trainer: GC-IDM (Markovian) then OURS (history), both
# on the SAME frozen SIGReg vit-tiny-192 base + the SAME shared latent cache.
# Usage: gc_train_driver.sh <task> <gpu>   e.g. gc_train_driver.sh lift 4
set -euo pipefail
TASK=$1; GPU=$2
B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro
PY=/var/lib/docker/data/minghao_home/lewm/bin/python
DEC=/mnt/minghao_data/.stable-wm/decoders
ENV="STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl_${TASK} TMPDIR=/tmp"
WEIGHTS=$DEC/${TASK}_lewm_weights.pt
cd $B

echo "[driver:$TASK] === GC-IDM (Markovian) on GPU$GPU ==="
env $ENV CUDA_VISIBLE_DEVICES=$GPU $PY train_gcidm.py \
  --dataset_name ${TASK}.h5 --keys_to_load pixels,action \
  --weights $WEIGHTS \
  --run_name ${TASK}_gcidm --cache_run ${TASK}_gcidm \
  --epochs 200 --H_max 50 2>&1
echo "[driver:$TASK] GC-IDM done"

echo "[driver:$TASK] === OURS (history GC) on GPU$GPU ==="
env $ENV CUDA_VISIBLE_DEVICES=$GPU $PY train_ours_gc.py \
  --weights $WEIGHTS \
  --run_name ${TASK}_gc_ours --cache_run ${TASK}_gcidm \
  --epochs 200 --H_max 50 2>&1
echo "[driver:$TASK] OURS done -- BOTH ARMS TRAINED"
