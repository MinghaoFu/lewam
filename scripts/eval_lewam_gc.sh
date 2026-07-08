#!/usr/bin/env bash
# Evaluate a train_lewam_gc.py run (the §10 LeWAM results): remap the checkpoint to the
# gcidm eval format, stage it under $STABLEWM_HOME/checkpoints/, and run the reactive
# goal-conditioned eval (3 seeds).
#
#   bash scripts/eval_lewam_gc.sh <tworoom|reacher|pusht|cube> [run_dir] [num_eval]
#
# run_dir defaults to $STABLEWM_HOME/checkpoints/<task>_lewam_gc — i.e. the output of
#   python scripts/train_lewam_gc.py --dataset_name <task>.h5 --run_name <task>_lewam_gc \
#       --epochs 50 --H_max 50 --w_cyc 0.0        # w_cyc 0 = v2 (w/o consistency), 1.0 = v3
# SR per seed is printed by eval_gip.py and saved under $STABLEWM_HOME/gip_eval/gcidm/<policy>/.
set -euo pipefail

TASK=${1:?usage: $0 <tworoom|reacher|pusht|cube> [run_dir] [num_eval]}
SWM=${STABLEWM_HOME:-$HOME/.stable_worldmodel}
RUN_DIR=${2:-$SWM/checkpoints/${TASK}_lewam_gc}
NEVAL=${3:-50}
POLICY=${TASK}_lewamgc
DST=$SWM/checkpoints/$POLICY
export MUJOCO_GL=${MUJOCO_GL:-egl}

[ -f "$RUN_DIR/lewam_gc_best.pt" ] || { echo "no lewam_gc_best.pt in $RUN_DIR"; exit 1; }
mkdir -p "$DST"

# remap: lewam_gc checkpoint -> gcidm eval format
#   gc_head.* -> head.*, drop dynamics.* (eval is the reactive head only), keep encoder.*;
#   config: z_dim -> emb_dim, mark self-contained weights.
python - "$RUN_DIR" "$DST" <<'PY'
import json, shutil, sys
import torch

src, dst = sys.argv[1], sys.argv[2]
for s, d in [("lewam_gc_best.pt", "gcidm_full_model_best.pt"),
             ("lewam_gc_latest.pt", "gcidm_full_model_latest.pt")]:
    try:
        sd = torch.load(f"{src}/{s}", map_location="cpu", weights_only=False)
    except FileNotFoundError:
        continue
    nd = {}
    for k, v in sd.items():
        if k.startswith("gc_head."):
            nd["head." + k[len("gc_head."):]] = v
        elif k.startswith("dynamics."):
            continue
        else:
            nd[k] = v
    torch.save(nd, f"{dst}/{d}")
for s, d in [("gc_head_best.pt", "gcidm_head_best.pt"),
             ("gc_head_latest.pt", "gcidm_head_latest.pt")]:
    try:
        shutil.copy2(f"{src}/{s}", f"{dst}/{d}")
    except FileNotFoundError:
        pass
cfg = json.load(open(f"{src}/lewam_gc_config.json"))
cfg["from_scratch"] = True
cfg["weights"] = "self"
if "z_dim" in cfg and "emb_dim" not in cfg:
    cfg["emb_dim"] = cfg.pop("z_dim")
cfg.setdefault("head", "gcidm")
json.dump(cfg, open(f"{dst}/gcidm_config.json", "w"))
print(f"staged eval checkpoint -> {dst}")
PY

for S in 42 0 1; do
  python scripts/eval_gip.py --config-name "$TASK" policy=$POLICY \
    +gip_eval.mode=gcidm eval.num_eval="$NEVAL" seed=$S
done
