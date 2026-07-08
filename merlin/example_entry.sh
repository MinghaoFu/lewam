#!/usr/bin/env bash
# Merged WAM v2 — cap-robust. Fixes vs v1: (1) background loop syncs checkpoints to HDFS
# every 5 min so a pod kill never loses progress; (2) smaller budget that completes in-cap
# for auto-eval; (3) per-task heartbeat FILES (v1's shared file lost concurrent writes).
# Still failure-isolated: one task crashing does not disturb the others.
set -uo pipefail
HROOT=/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam; HCODE=$HROOT/code
CK=$HROOT/ckpts/merged_wam2; mkdir -p "$CK" "$CK/ckpts_live"
ts(){ date '+%F %T'; }
JHB=$CK/job_heartbeat.log; jhb(){ echo "[$(ts)] $1" | tee -a "$JHB"; }
jhb "MERGED WAM v2 START host=$(hostname)"; nvidia-smi -L 2>&1 | head

WORK=/opt/tiger/merged_wam2; rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
tar xzf "$HCODE/lewam_repo_current.tar.gz" -C "$WORK"
export PYTHONPATH="$WORK"
python3 -c "import torch,stable_worldmodel,lightning,hydra" 2>/dev/null || {
  grep -vE '^torch==|^torchvision==|^nvidia-|^cuda-' requirements.txt | sed 's/==/<=/' > /tmp/r.txt
  python3 -m pip install --user -q -r /tmp/r.txt 2>/dev/null
  python3 -m pip install --user -q 'torchvision<=0.24.1' 'dm_control==1.0.43' lightning 2>/dev/null
}
jhb "ENV ready"

export STABLEWM_HOME=/opt/tiger/.swm MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl TMPDIR=/tmp
DS=$STABLEWM_HOME/datasets; mkdir -p "$DS/ogbench"
SSD=/mnt/hdfs/bi_algo_a2f/minghao.fu/lewam/data; HDD=$HROOT/datasets
lnk(){ local f=$1 sub=${2:-} src=$SSD; [ -r "$SSD/$sub$f" ] || src=$HDD; ln -sf "$src/$sub$f" "$DS/$sub$f"; }
lnk tworoom.h5; lnk reacher.h5; lnk pusht_expert_train.h5; lnk cube_single_expert.h5 ogbench/
jhb "data linked"

# --- background checkpoint sync: pod-local ckpts -> HDFS every 5 min (survives cap-kill) ---
( while true; do cp -ru "$STABLEWM_HOME/checkpoints/." "$CK/ckpts_live/" 2>/dev/null; sleep 300; done ) &
SYNC_PID=$!
jhb "ckpt-sync loop pid=$SYNC_PID (every 5 min -> $CK/ckpts_live)"

EPOCHS=15; LTB=1000   # in-cap budget (fuse-bound ~1.5-2 it/s -> ~2-3h); background sync covers overruns

run_task(){  # $1=gpu $2=T $3=dataconf $4=evalname
  local GPU=$1 T=$2 DC=$3 EV=$4
  local HB=$CK/hb_$T.log; thb(){ echo "[$(ts)] $1" | tee -a "$HB"; }
  : > "$HB"; thb "TRAIN $T START gpu=$GPU (merged ep=$EPOCHS ltb=$LTB)"
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/train.py \
    model=lewm_merged data=$DC \
    action_pred.enabled=true action_pred.goal_conditioned=true action_pred.horizon_conditioned=true \
    action_pred.detach_decoder=true action_pred.head=mse action_pred.w_act=1.0 action_pred.w_intent=1.0 \
    trainer.max_epochs=$EPOCHS +trainer.limit_train_batches=$LTB +trainer.limit_val_batches=15 \
    output_model_name=merged_$T subdir=merged_$T wandb.enabled=false \
    > /tmp/train_$T.log 2>&1
  local RC=$?; cp /tmp/train_$T.log "$CK/" 2>/dev/null
  cp -ru "$STABLEWM_HOME/checkpoints/merged_$T" "$CK/ckpts_live/" 2>/dev/null
  thb "TRAIN $T DONE rc=$RC $(grep -oiE 'val/act_loss[^0-9]*[0-9.]+' /tmp/train_$T.log | tail -1)"
  if [ $RC -ne 0 ]; then thb "TRAIN $T FAILED (rc=$RC) -> skip eval; others continue"; return 0; fi
  local RESF=$CK/merged_${T}_sr.txt; : > "$RESF"
  for S in 42 0 1; do
    thb "EVAL $T seed=$S"
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_gip.py --config-name $EV policy=merged_$T \
      +gip_eval.mode=policy +gip_eval.goal_conditioned=true eval.num_eval=50 seed=$S \
      > /tmp/eval_${T}_$S.log 2>&1
    cp /tmp/eval_${T}_$S.log "$CK/" 2>/dev/null
    local SR=$(grep -oiE "success_rate['\"]?[: ]+[0-9.]+" /tmp/eval_${T}_$S.log | grep -oE "[0-9.]+" | tail -1)
    echo "seed=$S SR=$SR" | tee -a "$RESF"; thb "EVAL $T s$S SR=$SR"
  done
  thb "$T ALL DONE: $(tr '\n' ' ' < "$RESF")"
}

run_task 0 tworoom tworoom tworoom &
run_task 1 reacher dmc     reacher &
run_task 2 pusht   pusht   pusht &
run_task 3 cube    ogb     cube &
wait
kill $SYNC_PID 2>/dev/null
cp -ru "$STABLEWM_HOME/checkpoints/." "$CK/ckpts_live/" 2>/dev/null
jhb "MERGED WAM v2 ALL DONE"
