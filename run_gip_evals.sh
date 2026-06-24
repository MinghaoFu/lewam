#!/usr/bin/env bash
# GIP success-rate eval matrix for one checkpoint: runs bc + guided, appends to summary.
# usage: run_gip_evals.sh <gpu> <config_name> <policy_run> [num_eval] [ckpt_epoch] [modes]
set -e
cd /home/minghao.fu/workspace/le-wm-repro
export STABLEWM_HOME=/mnt/data_nvme1/minghao.fu/.stable-wm
export MUJOCO_GL=egl
GPU=$1; CFG=$2; POL=$3; NE=${4:-50}; EP=${5:-}; MODES=${6:-bc guided}
LOGD=/mnt/data_nvme1/minghao.fu/le-wm-repro-logs
SUM=$LOGD/gip_eval_summary.txt
epflag=""; [ -n "$EP" ] && epflag="+gip_eval.ckpt_epoch=$EP"
for mode in $MODES; do
  log=$LOGD/eval_${POL}_${mode}${EP:+_ep$EP}.log
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python eval_gip.py --config-name $CFG     policy=$POL +gip_eval.mode=$mode eval.num_eval=$NE $epflag > $log 2>&1 || { echo "FAIL $POL $mode" | tee -a $SUM; continue; }
  sr=$(grep -aoE "'success_rate': [0-9.]+" $log | tail -1)
  echo "$(date -Iseconds) $POL ${EP:+ep$EP }$mode ne=$NE -> $sr" | tee -a $SUM
done
