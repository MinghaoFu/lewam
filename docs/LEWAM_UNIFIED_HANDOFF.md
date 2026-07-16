# LeWAM — next direction handoff (2026-07-17): unify the split, don't rebuild the seq

The LeWAM-Seq model is **scrapped** (decision by the user + Minghao, 2026-07-17). This doc is the
handoff for the next session. Part 1 is the seq post-mortem (what we learned, where it broke, so
we don't repeat it). Part 2 is the new direction. Part 3 is the operational context (machine, repo,
compute, checkpoints, eval) needed to run — read it before launching anything. The older seq/GC
context is in `docs/LEWAM_SEQ_HANDOFF.md` (kept for history; seq is closed).

---

## Part 1 — LeWAM-Seq post-mortem (why it failed, with evidence)

**What seq was.** One causal Transformer over interleaved `[z_0,e_0,z_1,e_1,…]` state/action
tokens (`lewam/models/lewam_seq.py`). State-token read-out → action head (IDM/BC); action-token
read-out → dynamics head (FDM). Eval `get_action` = reactive receding-horizon rollout over a
sliding `num_frames` window; `history_size` (HS) sets the eval context length.

**Headline (seq_policy, 3 seeds N=50) vs the split baseline:**

| task | seq_policy | split (this repo) |
|---|---|---|
| tworoom | 100 | 100 |
| cube | 92 | 100 |
| pusht | 61 | 88 |
| reacher | 37 | 96 (98/92/98) |

The gap grows with task precision. reacher is the sharpest failure and was the focus.

**What we RULED OUT (not the cause):**
- **The eval scaffold** — shared with the split, which now scores tworoom 100 / reacher 96 through
  the exact same `world.evaluate` path (via the new `split_policy` adapter). So env/goal/SR are fine.
- **The eval policy adapter (`LeWAMSeqPolicy`)** — audited and empirically verified faithful:
  the predictor is causal (perturbing the last action token changes **zero** action predictions;
  `e_1` leaves position 0 unchanged, moves position 2), and `get_action` reproduces the training
  `forward` at the decision position to **2e-7**. z-score is the exact inverse of training, goal
  conditioning matches (`goal_dropout_act=0` → always-GC), past-action feeding is aligned.
- **Training** — internally consistent, and NOT collapse / underfit / overfit: reacher **val_act
  ≈ 0.76** (LOWER, i.e. better, than the split's 0.98), **eff_rank 48%**, z_std ≈ 1.0. The seq fits
  held-out actions *better* than the split yet controls far worse.
- One minor training wrinkle (not the cause): a horizon OOD at low HS — window position 0 is only
  trained on horizons ≥ `num_frames/H_max` ≈ 0.16, but eval queries it down to 1/H_max = 0.02.
  Explains part of the HS=1→HS=8 gradient (27→37), not the gap.

**Where the problem IS (most likely) — the architecture, concentrated in the precise endgame.**
Two pieces of evidence:
1. **HS=1 isolates the state path** (no action tokens in context) and is already **27 vs split 96**.
   So most of the deficit exists before any action embedding enters — it's the routing of `z_t`
   through the predictor to decode the action, not the action context.
2. **Shadow diagnostic** (`scripts/seq_vs_split_shadow.py`): drive the env with the *working* split
   (drove at 90% here), and at each step shadow the seq's HS=1 action on the SAME frame/goal/horizon
   — measuring whether the seq would pick the split's action on the split's own goal-reaching states
   (self-consistent, competent distribution). Result (seq vs split action):

   | regime | cosine | R² | rel. RMSE |
   |---|---|---|---|
   | all | 0.62 | 0.41 | 0.75 |
   | far from goal | 0.69 | 0.52 | 0.66 |
   | **near goal** | **0.53** | **0.24** | **0.86** |

   Only moderate agreement overall (pure covariate shift would predict high agreement on good
   states), and agreement **collapses as the goal approaches** — backwards from a competent policy,
   since the correct action is *more* determined near the goal. The seq diverges from the competent
   policy exactly in the precise endgame.

**Synthesis.** The seq's action feature (predictor state-token output `h_act`) is a *lower-average-MSE
but worse-for-control* map: the average MSE is dominated by the easy far/coarse actions, where it
roughly tracks the split (R² 0.52); the precise endgame, where it falls apart (R² 0.24), is a small
slice of the MSE but the decisive slice for reaching. Likely mechanism: the predictor is **shared**
between the action decode and the dynamics/FDM objective (which rewards smooth, predictable latent
transitions), plus it's a deep (6-layer) stack — this shapes `h_act` toward a smoothed action that
loses the fine corrections. The split avoids this: its `gc_head` reads **raw `z_t`** directly, and its
two heads are independent MLPs (only the encoder is shared), so nothing pulls the action map toward
"smooth/predictable."

**One caveat on the shadow:** two independently trained good policies needn't agree in absolute
terms, so absolute cosine 0.62 isn't self-calibrating — the robust signal is the near-vs-far
*contrast* (which needs no calibration and goes the wrong way). A split-vs-demo (or 2nd-seed split)
reference run would pin the absolute scale if ever needed. Left un-run; seq is closed.

---

## Part 2 — The NEW direction (what to build)

Goal (user + Minghao, 2026-07-17): **improve the split** (the working base) — give it **longer
temporal context** and move it **closer to a unified predictor**, rather than "two independent heads
bolted on an encoder" — **without adding many new components**, and without re-introducing the seq's
failure mode.

**Keep (the split's wins):**
- The direct, precise read of the current state for the action (`gc_head` off `z_t`). The endgame
  precision the seq lost comes from this directness — preserve a near-direct path to the action.
- The reactive goal + horizon conditioning (AdaLN) that already works across all four tasks.

**Avoid (the seq's failures):**
- Do **not** route the action decode through a deep predictor entangled with the dynamics objective
  — that smooths the endgame. If a shared predictor feeds the action head, keep it **shallow** and/or
  **residual to `z_t`** so raw-state precision survives.
- Do **not** interleave action tokens into the sequence — that adds self-generated covariate shift at
  eval and entangles the action pathway. The seq's action context netted only slightly positive
  anyway (HS 1→8: +10). Prefer **state-side context**.

**Use (what helped even in the broken seq):**
- Temporal context on the STATE side helps (HS 1→8 gave +10). Add a short state-latent history.

**Concrete seed to consider (minimal-component unification):**
A single **shallow causal aggregator over the state-latent window** `[z_{t-k..t}]` → a context latent
`c_t = z_t + light_attn([z_{t-k..t}])` (residual, so `z_t`'s precision is preserved), where **one
shared `c_t` feeds BOTH** the action head `gc_head(c_t, z_goal, h)` and the dynamics head
`dyn(c_t, a, z_goal)`. That is "one predictor/representation feeding both read-outs over a temporal
window" — more unified than 2-heads-on-encoder, adds ~one attention block (few new components), keeps
the endgame-precise direct path via the residual, and drops the seq's action-token interleaving.
Variants worth a thought: no residual but a shallow (1–2 layer) state-only transformer; or an
explicit skip from `z_t` into the action head alongside `c_t`.

**Validate on reacher and pusht FIRST** — the precise tasks that discriminate architectures. tworoom
and cube are easy (both seq and split already pass), so they won't tell you if a change helped.

**Reusable probe:** `scripts/seq_vs_split_shadow.py` is the template for asking "does the new model
choose the split's actions on the split's trajectories, and does agreement hold near-goal?" — the
fastest read on whether a new model preserves endgame precision, before a full SR sweep. Adapt it to
shadow the new model instead of the seq.

---

## Part 3 — Operational context (read before running)

**Machine / repo.** Dev box `minghao4` (user 傅明浩 / minghao.fu / tiger), direct local access. Repo
`/home/tiger/lewam`; active worktree `/home/tiger/lewam/.claude/worktrees/lewam-gc-preload-fix`,
branch **`lewam-seq`** (despite the dir name). For the NEW model, branch off `lewam-seq` (it contains
all the working infra + the split + eval adapters). Commit identity **MUST** be
`Minghao Fu <isminghaofu@gmail.com>` (`git config user.email isminghaofu@gmail.com`). `gh` is NOT
installed — open PRs via the web link.

**The working split (your base).**
- Model: `lewam/models/lewam_split.py` (`LeWAMSplit` = `ViTEncoder` → `gc_head(z_t,z_goal,h_norm)` +
  `dynamics(z_t,a_t,z_goal)`). Encoder + heads defined in `lewam/models/module.py` (`ViTEncoder` with
  a BatchNorm projector — force `model.eval()` at eval; `Transformer`/`Attention` are causal by
  default; `GMMHead`/`DiffusionHead` exist there for a distributional action head).
- Train: `scripts/train_lewam_gc.py` — 50ep, `--w_cyc 0.0 --hidden_dim 512 --batch_size 256`,
  `--H_max 50`, per-position-free horizon `h~U[1,H_max]`. It now has: the **flatten OOM fix**
  (preallocate+free, no `torch.cat` doubling) and **`--ckpt_sync_dir <hdfs_dir>`** which mirrors
  config + best.pt to durable storage on each best-improvement — **always pass it** so a released
  worker never costs a run (a lost SPLIT_RETRAIN checkpoint is what motivated it).
- Eval: `bash scripts/eval_lewam_split.sh <task> [run_dir] [num_eval]`, or
  `python scripts/eval_gip.py --config-name <task> policy=<run_name> +gip_eval.mode=split_policy`.
  The `split_policy` path (`gip.load_lewam_split_model` + `gip.LeWAMSplitPolicy` +
  `build_policy` dispatch in `lewam/models/gip.py`) loads a `LeWAMSplit` directly and drives it
  reactively — it does NOT use the old gcidm/JEPA rebuild (that path is broken: its projector no
  longer matches `module.ViTEncoder`, size mismatch → `SRs=[]`). A NEW model needs its own adapter
  following this exact template (loader + `BasePolicy` subclass + a `build_policy` mode + an
  `eval_gip.py` branch).
- Durable current-module split ckpts (loadable, strict): `ckpts/lewam_gc_repro/SPLIT_RETRAIN2/
  ckpts_live/{tworoom,reacher}_lewam_gc/` (tworoom SR 100, reacher SR 96). §10 ckpts
  (`lewam_gc_v1/v2/v3`, `DBG2`, …) are an OLDER architecture (JEPA-style encoder, different ViT key
  names) and will NOT load into the current `LeWAMSplit` — ignore them.

**Compute (verified this session).** Cap-free headless worker launch:
```
tail -f /dev/null | script -qfc "mlx worker launch --type A100-SXM-80GB --gpu <N> \
  --resourcetype arnold --usergroup bi_algorithm --cluster cloudnative-maliva \
  --queuename compute-334-aliyun.va-cloudnative-ai-bi.algorithm-guarantee \
  --no-input -- bash <hdfs_entry_script.sh> <args>" /tmp/launch.log
```
run as a **background command** (the worker lives only while this command lives; ~40 min to allocate
+ image-pull, then it runs the entry script). Worker had **495 GB RAM** (the quota table's "206" was
misleading). `mlx worker list` / `mlx worker kill <id>` to manage; **never kill another user's
worker** (e.g. an H100 you didn't launch). Monitor progress via HDFS heartbeat files, not job logs.
Entry-script pattern: untar the repo tarball to `/opt/tiger`, `export PYTHONPATH`, symlink datasets,
`export STABLEWM_HOME=/opt/tiger/.swm_* MUJOCO_GL=egl HF_HUB_OFFLINE=1`, run the script, write
heartbeat FILES. reacher eval needs `dm_control==1.0.43`. Before any launch, rebuild + stage the
tarball: `cd <worktree>; tar czf /tmp/t.tgz lewam scripts configs requirements.txt pyproject.toml;
cp /tmp/t.tgz /mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/code/lewam_repo_seq.tar.gz`.

**HDFS layout** (`HROOT=/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam`): `code/` (entry
scripts + `lewam_repo_seq.tar.gz`), `datasets/` (tworoom.h5, pusht_expert_train.h5, reacher.h5,
ogbench/cube_single_expert.h5), `ckpts/lewam_gc_repro/<TAG>/` (outputs). Dataset→keys: seq/split
trainers use `--keys_to_load pixels,action`; frameskip=5 → action_block = raw_adim×5 (reacher
raw_adim=2 → block 10).

**Reusable helper scripts on HDFS `code/`:** `split_retrain2.sh` (train+sync+split_policy eval, the
template for training a new model with durable sync), `seq_split_shadow.sh` (the shadow probe entry),
`eval_lewam_split.sh` (in-repo). The commits from this session on `lewam-seq`: `2cca312` split_policy
eval adapter, `34204ab` flatten OOM fix, `5250f7d` `--ckpt_sync_dir`, `f95354b` shadow diagnostic,
plus the SESSION 3 verdict doc.
