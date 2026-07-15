#!/usr/bin/env bash
# Evaluate a train_lewam_seq.py run (LeWAM-Seq) for goal-conditioned success rate.
# No checkpoint remap needed: gip.load_lewam_seq_model reads lewam_seq_config.json +
# lewam_seq_best.pt directly, so we just stage them under $STABLEWM_HOME/checkpoints/<policy>/
# (where the eval config's policy=<policy> resolves) and run eval_gip.py over 3 seeds.
#
#   bash scripts/eval_lewam_seq.sh <tworoom|reacher|pusht|cube> [seq_policy|seq_cem] [run_dir] [num_eval]
#
# run_dir defaults to $STABLEWM_HOME/checkpoints/<task>_seq (the trainer's --run_dir output).
#   seq_policy = model.get_action (action head / BC)    seq_cem = CEM over model.get_cost
# Validated tworoom: seq_policy 94.0% (94/98/90), seq_cem 71.3% (74/64/76) vs split 100.
set -euo pipefail

TASK=${1:?usage: $0 <task> [seq_policy|seq_cem] [run_dir] [num_eval]}
MODE=${2:-seq_policy}
SWM=${STABLEWM_HOME:-$HOME/.stable_worldmodel}
RUN_DIR=${3:-$SWM/checkpoints/${TASK}_seq}
NEVAL=${4:-50}
POLICY=${TASK}_seq
DST=$SWM/checkpoints/$POLICY
export MUJOCO_GL=${MUJOCO_GL:-egl}

[ -f "$RUN_DIR/lewam_seq_best.pt" ] || { echo "no lewam_seq_best.pt in $RUN_DIR"; exit 1; }
# stage in place if RUN_DIR already is the policy dir; else copy config + best weights across
if [ "$RUN_DIR" != "$DST" ]; then
  mkdir -p "$DST"
  cp "$RUN_DIR/lewam_seq_config.json" "$DST/"
  cp "$RUN_DIR/lewam_seq_best.pt"    "$DST/"
fi

for S in 42 0 1; do
  echo "==== eval $TASK $MODE seed=$S ===="
  python scripts/eval_gip.py --config-name "$TASK" policy="$POLICY" \
    +gip_eval.mode="$MODE" eval.num_eval="$NEVAL" seed=$S
done
