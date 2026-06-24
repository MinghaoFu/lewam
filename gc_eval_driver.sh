#!/usr/bin/env bash
# Per-task GC head-to-head EVAL: GC-IDM (mode=gcidm) + OURS (mode=policy goal_conditioned=true),
# N=50, 3 seeds {42,0,1}, on the task's TRAINING GPU (cross-GPU render gotcha).
# Usage: gc_eval_driver.sh <task> <gpu> <world_task> <dstats> <budget> <offset>
#   e.g. gc_eval_driver.sh lift 4 Lift lift 100 30
set -uo pipefail
TASK=$1; GPU=$2; WTASK=$3; DSTATS=$4; BUDGET=$5; OFFSET=$6
B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro
PY=/var/lib/docker/data/minghao_home/lewm/bin/python
ENV="STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl_e${TASK} TMPDIR=/tmp OMP_NUM_THREADS=1"
COMMON="--config-name robomimic world.task=$WTASK world.num_envs=10 dataset.stats=$DSTATS \
  eval.dataset_name=$DSTATS eval.num_eval=50 eval.eval_budget=$BUDGET eval.goal_offset_steps=$OFFSET"
cd $B

for SEED in 42 0 1; do
  echo "[eval:$TASK] === GC-IDM seed=$SEED (GPU$GPU) ==="
  env $ENV CUDA_VISIBLE_DEVICES=$GPU $PY eval_histbc_robomimic.py $COMMON \
    policy=${TASK}_gcidm +gip_eval.mode=gcidm seed=$SEED \
    hydra.run.dir=/tmp/eval_${TASK}_gcidm_s${SEED} 2>&1 | grep -aE 'GCIDM|chunk|RESULTS|success_rate|Traceback|Error|assert'
  echo "[eval:$TASK] === OURS seed=$SEED (GPU$GPU) ==="
  env $ENV CUDA_VISIBLE_DEVICES=$GPU $PY eval_histbc_robomimic.py $COMMON \
    policy=${TASK}_gc_ours +gip_eval.mode=policy +gip_eval.goal_conditioned=true seed=$SEED \
    hydra.run.dir=/tmp/eval_${TASK}_ours_s${SEED} 2>&1 | grep -aE 'HISTBC|chunk|RESULTS|success_rate|Traceback|Error|assert'
done
echo "[eval:$TASK] DONE all 3 seeds x 2 arms"
