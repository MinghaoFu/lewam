# LeWAM-Unified — session-3 handoff (2026-07-18): the residual sweep is DONE

Read `docs/LEWAM_UNIFIED_HANDOFF.md` (original direction + seq post-mortem) and
`docs/LEWAM_UNIFIED_HANDOFF2.md` (the BatchNorm-padding fix) first. This doc closes HANDOFF2's
open item #1: the 6-arm residual screening sweep is finished, with clean SR numbers, and both
follow-up items (#2 dyn-val spike, #3 BN-padding) are resolved or deferred with evidence.

Branch **`lewam-unified`**. Commit identity `Minghao Fu <isminghaofu@gmail.com>`. No repo *code*
changed this session; the model/trainer/eval-adapter are as of `90d333e` (staged tarball
`code/lewam_repo_seq.tar.gz`, 22:01). All sweep work was operational (HDFS entry scripts + eval).

## The result — residual sweep (SCREENING budget, not a headline)

Config, identical across all six arms (so the arm comparison is valid): 30 epochs, batch 64,
reacher `--max_eps 4000`, the ~1x-coverage RandomSampler, H_max 25 (tworoom) / 50 (reacher).
Eval = `scripts/eval_gip.py` `+gip_eval.mode=unified_policy`, loads `lewam_unified_best.pt`
(best combined val `va_a+va_d`; for these runs best = the last epoch), 3 seeds {42,0,1} × N=50.

| arm — context `c_t` | tworoom act(val) | tworoom SR | reacher act(val) | reacher SR (per seed) |
|---|---|---|---|---|
| **res**   `z_t + Aggr(z)`        | 0.874 | 100 [100,100,100] | 0.956 | **92.7** [94,94,90] |
| **nores** `Aggr(z)`              | 0.875 | 100 [100,100,100] | 0.959 | **90.7** [86,90,96] |
| **gate**  `z_t + g(z)·Aggr(z)`   | 0.872 | 100 [100,100,100] | 0.951 | **91.3** [94,96,84] |
| *split baseline (reference)*     | *0.85* | *100* | *~0.98* | *96* |

**What it says.** tworoom is uninformative: all three arms score 100 (the task is too easy to
separate architectures, as predicted). reacher is the discriminating task, and there the three
arms (92.7 / 90.7 / 91.3) fall **within eval-seed noise of each other** — each arm's own three
seeds span ±5–12 SR, wider than the between-arm gaps. So **the aggregator residual makes no clear
difference to SR** at this budget: the aggregator learns the goal-conditioned action map about
equally with the explicit `z_t` skip (res), without it (nores), or gated. The aggregator-only
`nores` arm is not worse, which is the *opposite* of the seq's failure (there, routing the action
through the deep shared predictor destroyed endgame precision). The shallow causal aggregator here
does not.

**The real signal is the gap to split.** All three unified reacher arms sit ~3–5 SR under the
split's 96, at 30ep. The reacher act-val curves were still creeping down at ep30 (0.958→0.956 over
the last five epochs), and reacher act sits near the mean-predictor 1.0 for every method (the split
itself was ~0.98), so this is a task with a high action-MSE floor where SR, not act, is the metric.
Whether the unified model matches or beats split on reacher is **unresolved at screening** and is
the question the finalization run answers.

## HANDOFF2 open items — status

- **#1 (the sweep): DONE**, above.
- **#2 (dyn-val spike): RESOLVED.** It self-settles over a full run. tworoom_res val `dyn` peaked
  328 at ep4 then decayed to 0.034 by ep30; reacher arms peaked ~6 then settled ~0.19. Best-ckpt
  selection was not corrupted: best landed at the last epoch where both `va_a` and `va_d` are low.
  No need to switch to `va_a`-only selection.
- **#3 (cleaner BN-vs-padding fix): DEFERRED, not needed.** No residual BN artifacts surfaced — all
  six arms trained cleanly and tworoom hit SR 100 across the board. The real-frame-padding
  workaround (`90d333e`) is sufficient. Revisit only if a finalization run shows BN trouble.

## Operational lessons (Merlin — these cost real time this session)

1. **The in-loop eval in `train_unified_res6.sh` is broken (stale cwd).** Its `run_arm` subshell
   `cd "$WORK"` once, and after a ~34-min training the cwd inode goes stale (`getcwd: ... No such
   file or directory` / `Stale file handle`), so the RELATIVE `bash scripts/eval_lewam_unified.sh`
   fails instantly and logs `SRs=[]`. Training survives (each arm re-`cd`s fresh). Fix baked into
   the new scripts: re-`cd "$WORK"` + ABSOLUTE script path right before eval. Because ckpts sync to
   `ckpts_live/$TAG` BEFORE the eval, no result is ever lost — eval the durable ckpt separately.
2. **A standalone eval works and is cheap.** `code/eval_unified_res6.sh <TAG...>` evals durable
   ckpts (own heartbeat dir `UNIFIED_RES6_EVAL`, absolute paths). The `unified_policy` adapter is
   verified end-to-end: tworoom all three arms SR 100. A100 eval workers **allocate in ~4 min** and
   **auto-exit gracefully** when the entry script ends.
3. **Parallel TRAINING workers do NOT auto-exit** — they linger idle holding their A100 after the
   script finishes (unlike eval workers). Two finished reacher workers idle-held 2 of the group's 4
   A100 slots for ~2h, which starved a third (gate) worker — it sat "Waiting to be scheduled" and
   the pod eventually died. **Kill each training worker explicitly once its arm is done**
   (`mlx worker kill <id>`); freeing the slots let the pending gate worker schedule instantly.
4. **Parallel single-arm runner:** `code/train_eval_unified_one.sh <TAG>` derives task/dataset/H_max/
   max_eps/agg-flags from the tag (bareword, no nested quoting through `mlx worker launch -- bash`),
   trains+syncs+evals ONE arm, writes per-TAG logs to `UNIFIED_RES6/` (parallel-safe: no shared-log
   `rm`). This is how the 3 reacher arms ran concurrently. Note `mlx worker kill` is gated by the
   auto-mode permission classifier — needs a Bash permission rule or the user's ok.

## Next steps

1. **Finalization run (per user, 2026-07-18).** Once a design is chosen, run it at the split's
   protocol for a fair, converged headline: **50 epochs, batch 256, FULL dataset (no `max_eps`
   cap), H_max 50**, and match effective supervision-per-epoch to the split (not the 1x-coverage
   fast sampler). **Params → ~15M** (LeWM's size), up from 9.30M (encoder 5.80M ViT-tiny + agg
   1.81M + gc_head 1.12M + dynamics 0.57M); bump encoder/width, keep the +1.8M aggregator. See the
   `unified-finalization-protocol` memory. This is the run that decides unified-vs-split on reacher.
2. **Which arm to finalize.** Screening does not separate res/nores/gate. Default to **res** (the
   residual = zero-init identity boot = provably ≥ split's directness, the design intent) unless a
   multi-seed reacher screen shows nores/gate ahead. If the residual question matters for the paper,
   run ≥3 TRAINING seeds per arm on reacher to get error bars the single-seed screen can't provide.
3. **pusht** is the other precise task (split 88); worth adding to a finalization sweep. tworoom and
   cube are saturated (100) and won't discriminate.

## Reusable assets on HDFS `code/`

`train_unified_res6.sh` (serial 6-arm, in-loop eval BROKEN — use for training only),
`train_eval_unified_one.sh` (parallel single-arm train+eval, eval FIXED),
`eval_unified_res6.sh` (standalone eval of durable ckpts, eval FIXED). Durable ckpts + logs:
`ckpts/lewam_gc_repro/UNIFIED_RES6/ckpts_live/{tworoom,reacher}_uni_{res,nores,gate}/`,
per-arm curves in `hb_*.log` / `train_*.log`, milestones in `par_heartbeat.log`.
