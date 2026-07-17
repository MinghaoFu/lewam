# LeWAM-Unified — session-2 handoff (2026-07-17): the trainer bug is FIXED, finish the sweep

Read `docs/LEWAM_UNIFIED_HANDOFF.md` first for the original direction (Part 1 seq post-mortem,
Part 2 the unified design, Part 3 base operational context). This doc is the continuation: the
model + tooling are built and a **real trainer bug was found and fixed** — the action now learns.
What remains is mostly running a clean sweep and two small follow-ups.

Branch: **`lewam-unified`** (off `lewam-seq`). Commit identity MUST be
`Minghao Fu <isminghaofu@gmail.com>`. `gh` not installed (PR via web link). Latest commit
**`90d333e`** (the BatchNorm fix). Prior: dff67f1 model, 48cf419 seq-parallel aggregator,
ed788d2 seq-parallel trainer, 4174886 flags, f9fa771/db24bf3 (mis-guided band-aids, reverted),
90d333e (the real fix).

## What was built (all committed, CPU-verified)

- **`lewam/models/lewam_unified.py`** — `LeWAMUnified` = ViT encoder → a **causal state
  aggregator** (`CausalStateAggregator`, sinusoidal PE, emits `c_τ` at every position) →
  shared `c_t` feeding `gc_head` (action) + `dynamics`. `forward_seq` = per-position training
  path; `gc_action` = last-position eval path. Flags: `agg_depth`(4), `agg_heads`(4),
  `agg_residual` (positive flag, default OFF; ON = zero-init identity boot = boots as the split),
  `agg_gate`, `agg_action_cond`. Heads reused verbatim from `lewam_split`.
- **`scripts/train_lewam_unified.py`** — sequence-parallel trainer. Samples a start→goal
  trajectory `[t..t+h]` (h~U[1,H_max]); goal = `z[-1]` (the window ENDPOINT, shared by all
  positions — correct, confirmed with the user); per-position horizon `h-k`; dynamics targets
  `z[1:]`; per-position masked act+dyn+SIGReg loss. `--ckpt_sync_dir` durable sync. ~1x-coverage
  per-epoch `RandomSampler` (avoids ~12x over-supervision → ~68s/epoch). `--H_max` is the context
  knob (OBS-STEPS; one frame per obs-step, NOT divided by frameskip).
- **`lewam/models/gip.py`** — `load_lewam_unified_model` + `LeWAMUnifiedPolicy` (mode
  `unified_policy`): full-causal-from-episode-start eval, caches per-frame latents, re-aggregates
  the growing history, reads the current position; arm-agnostic via `model.gc_action`.
- `scripts/eval_lewam_unified.sh`, `scripts/unified_vs_split_shadow.py` (near-goal probe).

## THE BUG (fixed) — and why it looked like collapse

Symptom: `act` loss stuck at ~1.0 (mean predictor), `dyn→0`, `reg` elevated ~9x the split, SR
~12 vs split 100 — looked exactly like encoder collapse. Two wrong hypotheses (SIGReg on
correlated latents; stop-grad the dyn target) did NOT fix it (both reverted — the working seq
trainer uses `sigreg(z.reshape(-1,D))` over ALL latents and NO stop-grad; match it).

**Root cause (user's instinct was right — a trainer bug):** the ViT projector has a
**BatchNorm**, and `collate_pad` padded variable-length trajectories with **ZERO frames**. In
train mode BatchNorm uses batch statistics, so the zero-padding frames corrupted the stats for
the real frames → the encoder output was non-discriminative → the action couldn't be learned.
The seq/split never hit this (fixed-length windows / triplets, no padding).

**Fix (`90d333e`):** in `collate_pad`, pad with the trajectory's **last real frame** (repeat the
endpoint) instead of zeros; padded positions stay masked out of the loss and causal-masked in
the aggregator, so they only serve to keep BatchNorm stats real.

**Confirmed working:** with the fix, `act` DROPS for the first time — `1.036 → 0.88` over ~19
epochs on tworoom_res (residual arm = split-equivalent), tracking toward the split's 0.85. The
action is learning.

## OPEN ITEMS (do these)

1. **Finish a CLEAN 6-arm sweep.** The current run is muddled (see infra note below). Kill any
   running worker, then launch ONE clean worker running `code/train_unified_res6.sh`
   (6 arms = {residual, no-residual, gated} × {tworoom, reacher}, serial on 1 GPU, batch 64,
   30 ep). Confirm tworoom_res reaches act ~0.85 + SR ~100, then read the residual comparison
   (does `--agg_residual` help vs no-residual vs gated) on tworoom AND reacher.
2. **`dyn` val spike** — during training, val `dyn` spikes wildly early (4→328→...) then decays;
   it's a BatchNorm train/eval-stats transient in the dynamics target (running-stat vs batch-stat
   mismatch, amplified because the dyn target is the moving encoder output). Confirm it settles
   over a full run; if it destabilizes best-ckpt selection (`combined_val = va_a + va_d`), select
   on `va_a` alone or fix the val-BN mismatch.
3. **BatchNorm + padding (user, follow-up):** the real-frame-padding fix is a workaround. If BN
   keeps hurting, do the cleaner thing — encode only the REAL frames (gather non-pad frames,
   encode, scatter back) so padding never reaches the encoder at all; or make the projector norm
   padding-invariant (e.g. LayerNorm, or a masked BN). Only if #1 shows residual BN artifacts.

## INFRA (learned the hard way this session — read before launching)

- **Use debug / live workers, NOT the congested Arnold guarantee queue** (user, 2026-07-17).
  The A100 arnold queue was ~29 min to schedule and pods intermittently failed (`status:FAILED`).
  V100 debug workers (28 free) schedule INSTANTLY but have a **small ~37 GB cgroup** (too small
  for tworoom's ~57 GB `Frames` → OOM). See `merlin/MERLIN.md` / handoff Part 3 for the live-worker
  launch; try `--resourcetype workspace`/`public-arnold` for debug pools. Launch template:
  `script -qfc "mlx worker launch --type <T> --gpu 1 --resourcetype arnold --usergroup bi_algorithm --cluster cloudnative-maliva --queuename compute-334-aliyun.va-cloudnative-ai-bi.algorithm-guarantee --no-input -- bash <hdfs_entry>.sh" /tmp/launch.log </dev/null`
  run as a background job. **Use `</dev/null` for stdin, NOT `tail -f /dev/null |`** (the latter
  started failing). **Do NOT `pkill` after launching** — it kills your own launch before it
  schedules (a mistake I made repeatedly). **Never launch two workers on the same entry script**
  — they clobber the shared HDFS heartbeat/ckpts.
- **cgroup RAM is the hard limit, not node RAM.** A100 --gpu1 = 247 GB worker / ~200 GB cgroup;
  --gpu2 = 495 GB / ~400 GB. V100 debug = ~37 GB. `free -g` shows the 2 TB node, misleading. The
  entry script prints `cgroup_mem.max` at START.
- **RAM-safe packing:** tworoom `Frames` ~57 GB, reacher@4000 ~60 GB, reacher@6000 ~74 GB (fp16,
  190k/246k frames × 301 KB). Preload holds it all. On a 200-400 GB cgroup, run arms SERIAL (1 at
  a time) or at most 1 tworoom + 1 reacher; 3-concurrent OOM-kills (rc=137).
- **Node disk (`/opt/tiger`, `/tmp`) can be ~2 GB free** on some nodes → `torch.save` fails
  (rc=1). The entry script writes STABLEWM_HOME/TMPDIR/MPLCONFIGDIR to **`/dev/shm`** (RAM,
  node-disk-independent). Keep this.
- `mlx worker kill <id>` takes ~100-170 s and needs `</dev/null` + a long timeout; it's async
  (worker lingers ~1 min after "killing"). Killing the launch-holder process does NOT release the
  worker — use `mlx worker kill`.
- Monitor via HDFS `ckpts/lewam_gc_repro/UNIFIED_RES6/{heartbeat.log, hb_<tag>.log,
  train_<tag>.log}` — job logs are unavailable.

## Current live state (as of handoff)

- Worker **1001508** (A100, 247 GB) is running `train_unified_res6.sh` and IS learning
  (tworoom_res act ~0.88 at ep 19). But its `heartbeat.log` was polluted by a second (V100) worker
  I accidentally launched in parallel (now killed). **Recommend: kill 1001508 and relaunch ONE
  clean worker** for trustworthy sweep numbers, rather than salvage the muddled logs.
- Before launching: rebuild+stage the tarball if you change code —
  `cd <worktree>; tar czf /tmp/t.tgz lewam scripts configs requirements.txt pyproject.toml;
  cp /tmp/t.tgz /mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/code/lewam_repo_seq.tar.gz`.
  The staged tarball currently has the BatchNorm fix (`90d333e`).
- Split baselines for comparison/shadow (durable, current-module): reacher SR 96, tworoom SR 100
  at `ckpts/lewam_gc_repro/SPLIT_RETRAIN2/ckpts_live/`. Split tworoom training reference:
  act 0.85/0.93, dyn 0.012, reg 2.89 (calibration for "is it learning").
