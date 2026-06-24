#!/usr/bin/env bash
# Warm-start arm: load action-conditioned LeWM ckpt, finetune with action decoding (40 ep).
set -e
cd /home/minghao.fu/workspace/le-wm-repro
export STABLEWM_HOME=/mnt/data_nvme1/minghao.fu/.stable-wm
export MUJOCO_GL=egl
DEC=/mnt/data_nvme1/minghao.fu/.stable-wm/decoders
LOGD=/mnt/data_nvme1/minghao.fu/le-wm-repro-logs
launch(){ # gpu data name init
  CUDA_VISIBLE_DEVICES=$1 nohup .venv/bin/python train.py data=$2     action_pred.enabled=true init_from=$4 output_model_name=$3     trainer.max_epochs=40 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20     loader.batch_size=64 num_workers=6 > $LOGD/$3.log 2>&1 &
  echo $! > /tmp/$3.pid; echo "launched $3 gpu$1 pid=$(cat /tmp/$3.pid)"
}
launch ${1:-6} pusht   gip_pusht   $DEC/pusht_ours_lewm_weights.pt
launch ${2:-0} tworoom gip_tworoom $DEC/tworoom_ours_lewm_weights.pt
launch ${3:-2} dmc     gip_reacher $DEC/reacher_ours_lewm_weights.pt
