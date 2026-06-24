#!/usr/bin/env bash
# From-scratch end-to-end PiT arm: train intention transformer jointly from
# random init (init_from=null) in the general latent state space.
# Same per-epoch budget as the warm-start arm so ep6 is a matched-compute point.
set -e
cd /home/minghao.fu/workspace/le-wm-repro
export STABLEWM_HOME=/mnt/data_nvme1/minghao.fu/.stable-wm
export MUJOCO_GL=egl
LOGD=/mnt/data_nvme1/minghao.fu/le-wm-repro-logs

launch () {  # $1=gpu $2=data $3=name
  CUDA_VISIBLE_DEVICES=$1 nohup .venv/bin/python train.py     data=$2 action_pred.enabled=true     output_model_name=$3     trainer.max_epochs=60 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20     loader.batch_size=64 num_workers=6     > $LOGD/$3.log 2>&1 &
  echo $! > /tmp/$3.pid
  echo "launched $3 on GPU $1 pid=$(cat /tmp/$3.pid)"
}

launch ${1:-6} pusht   gip_scratch_pusht
launch ${2:-4} tworoom gip_scratch_tworoom
launch ${3:-3} dmc     gip_scratch_reacher
