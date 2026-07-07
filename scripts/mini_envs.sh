#!/usr/bin/env bash
# Train + eval one of the four mini envs (tworoom | reacher | pusht | cube).
#
#   bash scripts/mini_envs.sh tworoom quick    # ~15 min flow run: 3 epochs x 300 batches, eval N=20
#   bash scripts/mini_envs.sh tworoom full     # the 15-epoch recipe, eval N=50 x 3 seeds
#
#   MODEL=lewm_merged bash scripts/mini_envs.sh tworoom quick   # shared-backbone variant
#
# Datasets resolve via $STABLEWM_HOME/datasets/ (see README Data section). Results:
# checkpoints -> $STABLEWM_HOME/checkpoints/<run>/, eval SR -> $STABLEWM_HOME/gip_eval/policy/<run>/.
set -euo pipefail

TASK=${1:-tworoom}
BUDGET=${2:-quick}
MODEL=${MODEL:-lewm}

case "$TASK" in
  tworoom) DATA=tworoom ;;
  reacher) DATA=dmc ;;
  pusht)   DATA=pusht ;;
  cube)    DATA=ogb ;;
  *) echo "usage: $0 <tworoom|reacher|pusht|cube> [quick|full]"; exit 1 ;;
esac

if [ "$BUDGET" = quick ]; then
  EPOCHS=3; LTB=300; SEEDS="42"; NEVAL=20
else
  EPOCHS=15; LTB=1000; SEEDS="42 0 1"; NEVAL=50
fi

RUN=${MODEL}_${TASK}_${BUDGET}

python scripts/train.py \
  model=$MODEL data=$DATA \
  action_pred.enabled=true action_pred.goal_conditioned=true action_pred.horizon_conditioned=true \
  action_pred.detach_decoder=true action_pred.head=mse action_pred.w_act=1.0 action_pred.w_intent=1.0 \
  trainer.max_epochs=$EPOCHS +trainer.limit_train_batches=$LTB +trainer.limit_val_batches=15 \
  output_model_name=$RUN subdir=$RUN wandb.enabled=false

for S in $SEEDS; do
  python scripts/eval_gip.py --config-name $TASK policy=$RUN \
    +gip_eval.mode=policy +gip_eval.goal_conditioned=true eval.num_eval=$NEVAL seed=$S
done
