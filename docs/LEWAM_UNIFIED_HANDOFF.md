# LeWAM-Unified — consolidated handoff (updated 2026-07-19)

Single source of truth for the LeWAM-Unified track. It replaces the earlier per-session handoffs
(`LEWAM_UNIFIED_HANDOFF2/3.md`, now deleted). Seq history lives in `docs/LEWAM_SEQ_HANDOFF.md`
(seq is closed). Branch **`lewam-unified`** (off `lewam-seq`). Commit identity MUST be
`Minghao Fu <isminghaofu@gmail.com>`. `gh` is not installed — open PRs via the web link.

## Goal

Improve the working **split** model (a ViT encoder feeding two independent heads) by giving it
**temporal context** on the state side and moving it toward a **single shared representation feeding
both heads**, without adding many components and without re-introducing the seq model's failure mode.
The precise tasks (reacher, pusht) are what discriminate architectures; tworoom and cube are
saturated at SR 100 for both split and every unified arm, so they do not tell you if a change helped.

## 1. Background — why the earlier seq model was scrapped (condensed)

The seq model was one causal Transformer over interleaved `[z_0,e_0,z_1,e_1,…]` state/action tokens.
It underperformed the split badly on the precise tasks (seq vs split: reacher 37 vs 96, pusht 61 vs
88; tworoom and cube tied). The eval scaffold, the eval adapter, and training were all ruled out as
the cause (the seq even fit held-out actions *better* than the split, at val_act ≈ 0.76 vs 0.98). The
deficit is architectural and concentrated in the precise endgame: with history capped to a single
frame the seq is already 27 vs split 96, and a shadow diagnostic (drive the env with the working
split, ask whether the seq would pick the split's action on the split's own goal-reaching states)
showed agreement that **collapses as the goal approaches** (near-goal R² 0.24 vs far-goal 0.52) —
backwards from a competent policy. Likely mechanism: the action decode is routed through a **deep
(6-layer) predictor shared with the dynamics/FDM objective**, which rewards smooth, predictable
latent transitions and smooths away the fine endgame corrections. The split avoids this because its
`gc_head` reads **raw `z_t`** directly through independent MLP heads.

**Design consequences carried into the unified model.** Keep a near-direct path from `z_t` to the
action (the endgame precision lives there). Do not route the action through a deep predictor
entangled with dynamics. Do not interleave action tokens into the sequence (the seq's action context
netted only +10 SR from HS 1→8 and added self-generated covariate shift). Temporal context on the
STATE side is the part that helped.

## 2. The unified model (what is built)

`lewam/models/lewam_unified.py` — `LeWAMUnified` = `ViTEncoder` → a **shallow causal state
aggregator** (`CausalStateAggregator`, sinusoidal PE, emits a context latent `c_τ` at every position)
→ a shared `c_t` feeding both `gc_head` (action, off `c_t`, `z_goal`, `h_norm`) and `dynamics`
(`c_t`, `a_t`, `z_goal`). The heads are reused verbatim from `lewam_split`. `forward_seq` is the
per-position training path; `gc_action` is the last-position eval path.

Arm flags govern how `c_t` relates to `z_t`:
- **res** (`--agg_residual`, positive flag): `c_t = z_t + Aggr(z)` with a **zero-init** correction, so
  it boots as the split (identity at init) and cannot underperform the split's direct read by
  construction.
- **nores** (default): `c_t = Aggr(z)`.
- **gate** (`--agg_gate`): `c_t = z_t + g(z)·Aggr(z)`, an input-dependent sigmoid gate.
- `--agg_action_cond`, `--agg_depth` (4), `--agg_heads` (4) are also present so every checkpoint
  strict-loads under one eval adapter.

Trainer `scripts/train_lewam_unified.py` is sequence-parallel: it samples a start→goal trajectory
`[t..t+h]` with `h~U[1,H_max]`, the goal is the window endpoint `z[-1]` (shared by all positions,
confirmed with the user), per-position horizon is `h-k`, dynamics targets are `z[1:]`, and the loss is
a per-position masked `act + dyn + SIGReg` (match the seq trainer: `sigreg` over ALL latents, no
stop-grad). `--H_max` is the context length in OBS-STEPS (one frame per obs-step, not divided by
frameskip). Always pass `--ckpt_sync_dir <hdfs_dir>` so a released worker never costs a run.

Eval adapter: `lewam/models/gip.py` `load_lewam_unified_model` + `LeWAMUnifiedPolicy`
(mode `unified_policy`) — full-causal-from-episode-start rollout, caches per-frame latents,
re-aggregates the growing history, reads the current position, arm-agnostic via `model.gc_action`.
Run it with `scripts/eval_gip.py --config-name <task> policy=<run_name> +gip_eval.mode=unified_policy`
or `bash scripts/eval_lewam_unified.sh <task> <tag> <run_dir> <num_eval>`.

### The trainer bug that was found and fixed (`90d333e`)

Early unified runs looked like encoder collapse: `act` stuck at ~1.0 (mean predictor), `dyn→0`, SIGReg
elevated, SR ~12. The cause was **not** collapse. The ViT projector has a **BatchNorm**, and the
padding collate padded variable-length trajectories with **zero frames**; in train mode BatchNorm
uses batch statistics, so the zero pads corrupted the stats for the real frames and made the encoder
output non-discriminative. The fix pads with the trajectory's **last real frame** (repeat the
endpoint) instead of zeros; padded positions stay masked out of the loss and causal-masked in the
aggregator, so they only keep BatchNorm's stats real. With the fix the action learns for the first
time (`act` 1.036 → 0.88 tracking toward the split's 0.85). Two earlier hypotheses (SIGReg on
correlated latents; stop-grad the dynamics target) were wrong and reverted. A cleaner fix (encode only
the real frames, or a padding-invariant projector norm) is deferred — the real-frame padding is
sufficient and no residual BN artifacts have surfaced.

## 3. Results so far

### 3a. Residual sweep — res / nores / gate are indistinguishable

Screening budget, identical across all six arms so the comparison is valid: **30 epochs, batch 64**,
reacher `--max_eps 4000`, H_max 25 (tworoom) / 50 (reacher), the ~1×-coverage RandomSampler. Eval 3
seeds {42,0,1} × N=50 on `lewam_unified_best.pt`.

| arm — context `c_t` | tworoom SR | reacher SR (per seed) |
|---|---|---|
| **res**   `z_t + Aggr(z)`      | 100 [100,100,100] | 92.7 [94,94,90] |
| **nores** `Aggr(z)`            | 100 [100,100,100] | 90.7 [86,90,96] |
| **gate**  `z_t + g(z)·Aggr(z)` | 100 [100,100,100] | 91.3 [94,96,84] |
| *split (reference)*            | *100* | *96* |

tworoom is uninformative (all 100). On reacher the three arms fall within eval-seed noise of each
other. A 10-seed re-eval (500 rollouts/arm) confirmed it: **res 90.8±3.4, nores 89.6±3.1, gate
91.2±4.8** — means span 1.6 SR, all inside ±3–5 std, and the ranking flipped versus the 3-seed screen
(the signature of noise). The aggregator learns the goal-conditioned action map about equally with or
without the explicit `z_t` skip. Crucially the aggregator-only **nores** arm is **not worse**, which
is the opposite of the seq's failure — the shallow causal aggregator does not smooth the endgame the
way the seq's deep shared predictor did. **Choose res** on the principled basis (zero-init identity
boot ⇒ provably ≥ the split's directness), not on a measured SR win.

The real open signal is the **~3–5 SR gap to split** on reacher at 30ep (all three arms sit under 96).
The reacher act-val was still creeping down at ep30, and reacher act sits near the mean-predictor 1.0
for every method (the split itself was ~0.98), so reacher is a high-action-MSE-floor task where SR,
not act, is the metric. Whether the unified model matches or beats split on reacher is unresolved at
the screening budget.

### 3b. Context ablation on reacher — a wash (the policy is effectively Markovian)

Three probes agree that the aggregator's temporal history does not help on reacher:
- **ctx_cap SR sweep** (cap the aggregator to the last k obs-step latents, FIFO), serial run, res arm:
  SR 91 / 90 / 90 / 92 / 92 / 92 for k = 1/2/4/8/16/full — flat, a single frame controls as well as
  full history.
- **One-step BC probe** (`scripts/probe_ctx_bc.py`, open-loop, held-out demos, referenced to the
  dataset's recorded actions): `bc_mse` is context-flat across k at every horizon, res ≈ nores; the
  only effect is a small uptick in context usefulness at the shortest horizons (velocity mattering
  slightly for the last inch).
- **Trajectory-divergence probe** (`scripts/probe_divergence_analyze.py` + `dump_latents`): k=1 and
  k=full rollouts from matched starts track closely and both reach goal.

So the residual/gate/context axis has no SR lever on reacher, and the gap to split is a model/training
matter, not a context matter. Whether context *ever* helps needs a genuine-dynamics task with
contact and momentum — which is pusht.

> Data-integrity note: treat concurrent-eval outliers as suspect. An earlier ctx sweep run with 5 EGL
> renderers on one GPU reported a spurious `ctx_cap=1` SR of 12 (corrupted renders under GPU/EGL-init
> contention); the identical code + seeds run serially gives 91. Reproduce small-N eval numbers
> serially (conc=1) before believing them.

### 3c. PushT — screening result and the needed test (NEW, 2026-07-19)

pusht is the discriminating dynamics task and the one place the aggregator's temporal context could
finally matter. At the screening budget both models were trained on only ~4000 episodes (a fifth of
the full 18,685), and **both are under-data there**: the **unified res arm reaches SR ≈ 9** (30
epochs, 4000 episodes) and the **current-module split reaches SR ≈ 53** (50 epochs, 4000 episodes).
The older split-pusht figure of 88 predates the current module/setup and should not be trusted as the
baseline. The reference world-model baselines (DINO-WM, LeWM) use the full pusht dataset
(≈18,500–20,000 episodes).

**The needed next test is a FULL-DATA pusht run at 50 epochs** (full dataset, no `--max_eps` cap) for
**both** the unified res arm and the split baseline, so the comparison is fair at full data. Removing
the data deficit is the prerequisite before drawing any conclusion about pusht. **This full-data run
has not yet been completed** — it is the primary open experiment.

## 4. Next steps (priority order)

1. **Full-data pusht at 50 epochs, for BOTH the unified res arm and the split baseline** (full data,
   no `--max_eps`). This is the immediate open experiment (§3c) — there is no full-data split pusht
   baseline yet, so both must be run for a fair comparison. Eval both with 3 seeds × N=50 (unified:
   `unified_policy`; split: `split_policy`).
2. **Re-run the context probes on the full-data pusht checkpoint** (ctx_cap / one-step BC /
   divergence). pusht is the genuine-dynamics task; this is where the temporal-context hypothesis gets
   its real test, since reacher was Markovian.
3. **Finalization / headline** for the chosen res arm at the split's protocol (full data, H_max 50,
   matched effective supervision-per-epoch to the split rather than the fast ~1×-coverage sampler),
   to settle unified-vs-split on the precise tasks.

## 5. Operational context (read before running)

- **The working split (the base/reference).** Model `lewam/models/lewam_split.py`; train
  `scripts/train_lewam_gc.py` (50ep, `--w_cyc 0.0 --hidden_dim 512 --batch_size 256 --H_max 50`, plus
  `--ckpt_sync_dir`); eval `bash scripts/eval_lewam_split.sh <task>` or `eval_gip.py` mode
  `split_policy`. Durable current-module split checkpoints (strict-loadable): tworoom SR 100 and
  reacher SR 96 at `ckpts/lewam_gc_repro/SPLIT_RETRAIN2/ckpts_live/`; cube SR 100 (3000 eps) and pusht
  SR ~53 (4000 eps) at `ckpts/lewam_gc_repro/SPLIT_NEW/ckpts_live/`. Cube's 3000 eps is saturated; the
  pusht 4000-ep number (~53) is under-data, so a full-data split pusht baseline is still owed (§3c/§4).
  The older §10 `lewam_gc_v1/v2/v3` checkpoints are a different (JEPA-style) architecture and will NOT
  load into the current model — ignore them.
- **Unified checkpoints + curves.** `ckpts/lewam_gc_repro/UNIFIED_RES6/ckpts_live/{tworoom,reacher}_uni_{res,nores,gate}/`,
  per-arm curves in `hb_*.log` / `train_*.log`.
- **Datasets** (HDFS `HROOT=/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam`): `datasets/`
  holds `tworoom.h5`, `pusht_expert_train.h5` (18,685 episodes), `reacher.h5`,
  `ogbench/cube_single_expert.h5`. Trainers use `--keys_to_load pixels,action`; frameskip=5 →
  action_block = raw_adim × 5.
- **Compute.** Merlin/Arnold GPU launch mechanics live in `merlin/MERLIN.md` and the project
  `CLAUDE.md` (worker/job launch, cgroup RAM, entry-script pattern). Monitor runs via the HDFS
  heartbeat files, not job logs. reacher eval needs `dm_control==1.0.43`; headless render needs
  `MUJOCO_GL=egl`. Before any launch that includes code changes, rebuild and stage the repo tarball:
  `cd <worktree>; tar czf /tmp/t.tgz lewam scripts configs requirements.txt pyproject.toml; cp /tmp/t.tgz $HROOT/code/lewam_repo_seq.tar.gz`.
- **Reusable probes/scripts:** `scripts/probe_ctx_bc.py`, `scripts/probe_divergence_analyze.py`,
  `scripts/unified_vs_split_shadow.py`, `scripts/eval_lewam_unified.sh`.

## 6. Compute gotchas — FULL-DATA pusht launch (learned the hard way, 2026-07-19)

Launching the full-data pusht run hit a wall of Merlin/scheduler traps. Keep these in mind:
- **Full pusht preload ≈ 195 GB RAM** (144 GB fp16 frames @ 224² + ~44 GB h5 page cache). `flatten`
  is **1×, not 2×** (measured: `torch.empty` doesn't commit pages). Old runs OOM'd (`rc=137`) only
  because pods were under-provisioned AND logged `cgroup=?` — nobody checked the real limit. Request
  ≥256 GB and **log/guard the actual cgroup `memory.max`** at startup (abort if too small).
- **A10 trap.** The A100 queue `...va-cloudnative-ai-bi.algorithm-guarantee` has ONE `NVIDIA A10`
  (23 GB) mixed in, and the scheduler **deterministically** steers 1-GPU jobs onto it (node
  `n214-147-131`). That was the mysterious "22 GB GPU" that OOM'd earlier runs. **Guard on GPU
  memory <70 GB** and reroll. `gpuv: A100-SXM-80GB` is a soft hint, NOT enforced.
- **2-GPU / big-RAM (≥370 GB) A100 requests do not schedule** on the packed pool — they sit in
  `STARTED` for hours with no heartbeat. 2-GPU dodges the A10 but then won't place.
- **Escape hatch that worked: the H100 GCP pool** (`...va-cloudnative-aigcp-bi.algorithm-guarantee`,
  `gpuv: H100-SXM-80GB`, cluster 4) — separate pool, no A10, 1-GPU schedules. H100-80GB is fine
  substrate. Guard on memory (accepts A100+H100, rejects A10).
- **Unified bs256 preload prefetch OOM.** The unified's trajectory-window batches are ~3.9 GB each;
  DataLoader prefetch at 8 workers × 3 ≈ 90 GB on top of the 144 GB preload OOMs the pod. Bound it:
  `--num_workers 4 --prefetch_factor 2`. (Split's 3-frame items are tiny — 8 workers is fine.)
- **`train_lewam_gc.py` now has `--stream`** (h5-on-demand, low RAM) mirroring the unified trainer —
  a fallback if preload RAM is ever the blocker; stream is slow for the split (full-coverage epochs).
- **No `mlx job kill` in the CLI** — stuck/zombie jobs can only be killed from the web UI, so they
  accumulate. Launch scripts live in `~/lewam_project/jobs/` (`full_pusht_one.sh` + `_uni/_split.yaml`).
