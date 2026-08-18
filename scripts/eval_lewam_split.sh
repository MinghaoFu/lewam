#!/usr/bin/env bash
# Evaluate a train_lewam_gc.py run (a LeWAMSplit checkpoint) via the DIRECT split_policy
# adapter -- reactive goal-conditioned, 3 seeds. Unlike eval_lewam_gc.sh (which remaps the
# checkpoint into the gcidm format and rebuilds a frozen-LeWM/JEPA whose projector no longer
# matches module.VisionEncoder -> size-mismatch, SRs=[]), this loads the LeWAMSplit checkpoint
# directly (gip.load_lewam_split_model) and drives gip.LeWAMSplitPolicy. No remap, no JEPA.
#
#   bash scripts/eval_lewam_split.sh <tworoom|reacher|pusht|cube> [run_dir] [num_eval]
#
# run_dir defaults to $STABLEWM_HOME/checkpoints/<task>_lewam_gc (the train_lewam_gc output).
# policy=<task>_lewam_gc; SR per seed is printed by eval_gip.py.
set -euo pipefail

TASK=${1:?usage: $0 <tworoom|reacher|pusht|cube> [run_dir] [num_eval]}
SWM=${STABLEWM_HOME:-$HOME/.stable_worldmodel}
RUN_DIR=${2:-$SWM/checkpoints/${TASK}_lewam_gc}
NEVAL=${3:-50}
POLICY=${TASK}_lewam_gc
DST=$SWM/checkpoints/$POLICY
export MUJOCO_GL=${MUJOCO_GL:-egl}

[ -f "$RUN_DIR/lewam_gc_best.pt" ] || { echo "no lewam_gc_best.pt in $RUN_DIR"; exit 1; }

# load_lewam_split_model reads get_cache_dir(checkpoints)/<policy>; stage there if run_dir differs.
if [ "$RUN_DIR" != "$DST" ]; then
  mkdir -p "$DST"
  cp -f "$RUN_DIR"/lewam_gc_config.json "$RUN_DIR"/lewam_gc_best.pt "$DST"/
  [ -f "$RUN_DIR/lewam_gc_latest.pt" ] && cp -f "$RUN_DIR"/lewam_gc_latest.pt "$DST"/ || true
  echo "staged split checkpoint -> $DST"
fi

for S in 42 0 1; do
  python scripts/eval_gip.py --config-name "$TASK" policy=$POLICY \
    +gip_eval.mode=split_policy eval.num_eval="$NEVAL" seed=$S
done
