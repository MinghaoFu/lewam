#!/usr/bin/env bash
# Evaluate a train_lewam_unified.py run (a LeWAMUnified checkpoint) via the DIRECT
# unified_policy adapter -- reactive goal-conditioned over a state window, 3 seeds.
# Mirrors eval_lewam_split.sh; loads the LeWAMUnified checkpoint directly
# (gip.load_lewam_unified_model) and drives gip.LeWAMUnifiedPolicy. The single adapter is
# arm-agnostic: the checkpoint's config fully determines the architecture and load is strict.
#
#   bash scripts/eval_lewam_unified.sh <tworoom|reacher|pusht|cube> <run_name> [run_dir] [num_eval]
#
# run_name = policy= passed to eval_gip.py (also the dir under get_cache_dir(checkpoints)).
# run_dir defaults to $STABLEWM_HOME/checkpoints/<run_name>; staged into the cache dir if different.
set -euo pipefail

TASK=${1:?usage: $0 <task> <run_name> [run_dir] [num_eval]}
POLICY=${2:?usage: $0 <task> <run_name> [run_dir] [num_eval]}
SWM=${STABLEWM_HOME:-$HOME/.stable_worldmodel}
RUN_DIR=${3:-$SWM/checkpoints/${POLICY}}
NEVAL=${4:-50}
DST=$SWM/checkpoints/$POLICY
export MUJOCO_GL=${MUJOCO_GL:-egl}

[ -f "$RUN_DIR/lewam_unified_best.pt" ] || { echo "no lewam_unified_best.pt in $RUN_DIR"; exit 1; }

if [ "$RUN_DIR" != "$DST" ]; then
  mkdir -p "$DST"
  cp -f "$RUN_DIR"/lewam_unified_config.json "$RUN_DIR"/lewam_unified_best.pt "$DST"/
  [ -f "$RUN_DIR/lewam_unified_latest.pt" ] && cp -f "$RUN_DIR"/lewam_unified_latest.pt "$DST"/ || true
  echo "staged unified checkpoint -> $DST"
fi

for S in 42 0 1; do
  python scripts/eval_gip.py --config-name "$TASK" policy=$POLICY \
    +gip_eval.mode=unified_policy eval.num_eval="$NEVAL" seed=$S
done
