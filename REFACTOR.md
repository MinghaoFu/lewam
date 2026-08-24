# Refactor plan — post-finalization cleanup (do NOT execute on lewam-jointflow)

> Owner's call 2026-08-19: the lewam-jointflow branch deliberately carries the crossattn-tip port
> (`lewam_crossattn.py`, `dp_transformer.py`, `train_crossattn.py`, `grad_mpc.py`, `prompt_mpc.py`,
> the crossattn arms of `gip.py`/`train_lewam_unified.py`) because jointflow reuses pieces of it and
> the round-1 campaign is running against this exact tree. Once the model is finalized, ONE
> dedicated refactor pass produces the streamlined codebase from this plan. Per docs/STYLE.md:
> behavior-preserving first, one concern per commit, smoke-verify between steps.

## Target layout

```
lewam/
  models/
    module.py        # THE shared-blocks home: VisionEncoder, SIGReg, MLP, CrossAttnBlock,
                     # sinusoid, AdaLN helpers. One copy of every block (STYLE: one AdaLNBlock).
    jointflow.py     # the model only (slot_layout, attn_mask, JointFlow)
  train/
    datasets.py      # RawContextDataset, JointFlowDataset, load_cache (from train_crossattn.py)
    utils.py         # durable_sync, image-norm constants, lr schedules (from train_lewam_unified.py)
  eval/
    loaders.py       # load_jointflow_model (+ whichever legacy loaders survive)
    adapters.py      # JointFlowPolicy (+ surviving policies); BasePolicy stays swm's
    planners.py      # best-of-K joint sampling; warm-CEM w/ inpainting scorer; grad planner
                     # (port the grad_mpc.py skeleton: warm start, Adam, trust region,
                     # best-iterate floor, promise-vs-delivery diagnostics)
scripts/
  train_jointflow.py # thin: argparse + loop, importing lewam.train.*
  eval_cem.py        # note: main already renamed eval_gip.py -> eval_cem.py; the branch still
                     # carries eval_gip.py -- reconcile the name at merge time
  make_preload_cache.py
```

## Delete after their pieces move

- `lewam/models/lewam_crossattn.py` — after CrossAttnBlock + sinusoid move to module.py.
- `lewam/models/dp_transformer.py` — DPHead-only dependency; drop with it (keep only if a DP-head
  baseline stays in the paper).
- `scripts/train_crossattn.py` — after RawContextDataset/load_cache move to lewam/train/datasets.py.
- `lewam/models/grad_mpc.py`, `prompt_mpc.py` — port the planner skeleton into lewam/eval/planners.py
  against jointflow; delete the Unified-specific bodies.
- `lewam/models/flow_policy_head.py`, `lewam_seq` remnants, unused Unified heads — whatever the
  final ablation table does not need.
- `gip.py` — dissolve into lewam/eval/*; it is a 1500-line grab-bag of loaders + adapters +
  planners + dataset glue.

## Naming sweep (repo-wide, the owner's standing gripes)

- `context`/`ctx`/`ctx_cap` → **history** everywhere (canonical: history = the raw-consecutive
  conditioning frames, matching `policy_history_len`; **memory** = the cross-attended token bank
  built from them). Applies to RawContextDataset (→ HistoryDataset or similar), gip adapters,
  train_lewam_unified.
- No cryptic locals (`tr/va/hl/S/comps` class — already clean in jointflow files; sweep the rest).
- `imag`/`noimag` stay human arm-labels only; code says `actions_attend_states`.
- Legacy derived config fields (e.g. `action_dim` = fs*raw) are derived at load
  (load_jointflow_model), never stored — keep that convention.

## Policy-class merge (owner question 2026-08-24)

`JointFlowGCPolicy` is not structurally necessary: the checkpoint config already records
`goal_conditioning`/`goal_terminal`, so one `JointFlowPolicy` could branch on them in
`_propose` (encode goal + pick h source when conditioned, skip otherwise) — collapsing
modes `jointflow_policy`/`jointflow_gc` into one that trusts the ckpt. Deferred to this
refactor rather than done mid-campaign because the mode names are wired into every eval
entry/config/board record and the eval spine is under active measurement; folding also
removes today's silent-mis-eval hazard (a GC ckpt run under mode=jointflow_policy).
The training-side half is already done (2026-08-24: both datasets return one 8-tuple,
run_batch consumes the goal iff `--goal_conditioning`).

## Keep / don't touch

- Eval protocol machinery (full-traj pick-list, sharding, budget) — results depend on it byte-wise.
- `make_preload_cache.py` — cache-format source of truth.
- merlin/ entries + job YAMLs — operational history.

## Tests

Promote the scratch suites (`test_jointflow.py`, `test_jointflow_policy.py`, currently in the
session tmp dir) into `tests/`: slot layout, mask zero-gradient statements, dataset boundary
indexing, loss/backward reach, adapter replan cadence/denorm/flush. They are the regression net
for the whole refactor.

## In-code markers

The owner leaves `#RE:` comments in the code (style pass 71e5ef4 onward) marking refactor-time
renames: `grep -rn "#RE:" lewam/ scripts/` at refactor time and clear every one (e.g.
split_tau -> indep_schedule, dim -> embed_dim, slot_pos -> pos_emb, one shared sinusoid home).
