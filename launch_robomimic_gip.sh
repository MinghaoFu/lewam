#!/usr/bin/env bash
# Warm-start GIP (intention/BC head) on RoboMimic expert demos. ADDITIVE: new run
# names gip_robomimic_<task>, warm-started from each task's existing LeWM world model.
# RoboMimic = expert data -> the action head learns a real policy (bc/guided >> 0%).
set -e
cd /home/minghao.fu/workspace/le-wm-repro
export STABLEWM_HOME=/mnt/data_nvme1/minghao.fu/.stable-wm
export MUJOCO_GL=egl
DEC=/mnt/data_nvme1/minghao.fu/.stable-wm/decoders
LOGD=/mnt/data_nvme1/minghao.fu/le-wm-repro-logs
launch(){ # gpu task
  CUDA_VISIBLE_DEVICES=$1 nohup .venv/bin/python train.py data=robomimic_$2     action_pred.enabled=true init_from=$DEC/$2_lewm_weights.pt     output_model_name=gip_robomimic_$2 trainer.max_epochs=40     +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20     loader.batch_size=64 num_workers=6 > $LOGD/gip_robomimic_$2.log 2>&1 &
  echo $! > /tmp/gip_robomimic_$2.pid; echo "launched gip_robomimic_$2 gpu$1 pid=$(cat /tmp/gip_robomimic_$2.pid)"
}
launch ${1:-2} lift
launch ${2:-3} can
launch ${3:-6} square
