# LeWAM session handoff (2026-07-15)

Context for a fresh session picking up this work. Written thorough on purpose (project rule 9).

## Machine / repo / branches
- Running ON dev machine `minghao4` (user 傅明浩 / minghao.fu / tiger), direct local access. Repo `/home/tiger/lewam`.
- Active worktree: `/home/tiger/lewam/.claude/worktrees/lewam-gc-preload-fix` (this is where all edits happened; despite the name it currently sits on branch **`lewam-seq`**).
- GitHub `origin git@github.com:MinghaoFu/lewam.git` (PRIVATE). `gh` CLI is NOT installed (open PRs via the web link). Commit identity MUST be `Minghao Fu <isminghaofu@gmail.com>`.
- Branches on GitHub:
  - `main` — base.
  - `worktree-lewam-gc-preload-fix` — my infra fixes + the reproduction/ablation work (has an unopened draft PR; link: github.com/MinghaoFu/lewam/pull/new/worktree-lewam-gc-preload-fix).
  - `lewam-arch` — the USER's new models (LeWAMSeq/LeWAMSplit + module.py rework). Does NOT contain the infra fixes.
  - **`lewam-seq`** — consolidation = `lewam-arch` ⊕ my fixes ⊕ the new seq trainer. THIS is the working branch. Pushed.

## Compute mechanics (important, lots of friction this session)
- **Arnold** `mlx job submitv2 -p job.yaml`: the only launch path that works from a no-TTY background session. BUT ~4h soft cap (long jobs get STOPPED mid-train) + queue often backed up hours. `mlx job log/describe` broken — monitor via HDFS heartbeats. No `mlx job kill` (stop via web UI).
- **Workspace debug workers** (no cap, faster/bigger GPU): `mlx worker launch` needs a TTY → fails at "Login Worker" from background. Can only `mlx worker login <id> -- bash <script>` into an ALREADY-running worker (works non-TTY). Worker 982977 (8×A100, ~2TB RAM) was the debug box, now gone; 990113 (8×H100) is busy with someone else. Launching new workers must be done by the USER via `! mlx worker launch ...`.
- To run on a worker held-open: `mlx worker login <id> -- bash <hdfs script>` as a background command (the ssh session must stay up for the whole run; if it drops the tree is reaped). `setsid`-detached runs get killed on session exit.
- HDFS layout (`HROOT=/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam`):
  - `$HROOT/code/` — entry scripts + repo tarballs. Current seq tarball: `lewam_repo_seq.tar.gz` (rebuild from the worktree: `tar czf ... lewam scripts configs requirements.txt pyproject.toml`).
  - `$HROOT/datasets/` — tworoom.h5 (13G), pusht_expert_train.h5 (46G), reacher.h5 (99G), ogbench/cube_single_expert.h5 (102G).
  - `$HROOT/ckpts/lewam_gc_repro/<JOBTAG>/` — outputs (heartbeats, ckpts_live/, logs).
- Dataset name → keys: tworoom.h5|pixels,action,observation ; reacher.h5|... ; pusht_expert_train.h5|pixels,action ; ogbench/cube_single_expert.h5|... . frameskip=5, so action_block_dim = raw_adim*5.

## Work completed this session
### 1. Reproduced LeWAM-GC (§10) from scratch — DONE, matches dashboard
4 tasks × {v2 w_cyc=0, v3 w_cyc=1}, 50 ep, eval 3 seeds×N=50. Mean SR: tworoom 100/100, reacher 98/98, pusht 85.3/91.3, cube 100/100 (dashboard targets: 100/98.7/88/100). Needed 3 fixes (all in `worktree-lewam-gc-preload-fix`, carried into `lewam-seq`):
- `train_lewam_gc.py` flatten OOM: `torch.cat` of per-episode frames doubled RAM → OOM. Fixed to per-episode list indexing (numerically identical). Also added `--stream` (read frames from h5 on demand, no preload) + per-epoch collapse metrics.
- `train_gcidm.py`: package-qualified `jepa.JEPA`→`lewam.models.jepa.JEPA` etc. (eval broke on packaged repo).
- `gip.py` `load_gcidm_model`: re-added the dropped `from_scratch`/`weights=="self"` path (load end-to-end encoder from `gcidm_full_model_*.pt`).
Trained on debug box 982977; evaled on H100 990113. Ckpts at `ckpts/lewam_gc_repro/DBG2/ckpts_live/`.

### 2. SIGReg ablation (w_reg=0) + collapse metrics — mostly DONE
Question: does action/dynamics prediction alone prevent latent collapse? Answer: NO — SIGReg is doing the anti-collapse work.
- Without SIGReg: reacher fully collapses (eff_rank 2.7% of 192, z_std 1e-4, cos 1.0, SR ~3% vs 98%); tworoom collapses (eff_rank ~1%); cube/pusht drop hard too (eff_rank 3–8%) — cube keeps some structure/learning. Consistency loss (w_cyc) does not help.
- With SIGReg (converged, via `scripts/collapse_eval.py` offline probe on DBG2 ckpts): eff_rank 15–34% across tasks, z_std ~1.0, cos ~0.0003. Results in `ckpts/lewam_gc_repro/SIGREG_COLLAPSE/results.jsonl`.
- cube/pusht no-SIGReg relaunched to 50ep (JOBTAG=ABL2, preload, 1 big cube/node) — check `ckpts/lewam_gc_repro/ABL2/` for final numbers.
- Tooling added: `--stream`, per-epoch collapse metrics (`collapse_metrics.json`), `scripts/collapse_eval.py`.

### 3. NEW model LeWAM-Seq — reviewed, fixed, trainer written, pilot PASSED
`lewam/models/lewam_seq.py` (LeWAMSeq): one causal transformer over interleaved `[z_t,e_t]` tokens. State-token→action head (IDM/BC), action-token→dynamics head (FDM). `get_action` = AR policy rollout, `get_cost` = CEM oracle. (`lewam_split.py` LeWAMSplit = clean refactor of the split GC model; `train_lewam_gc.py` now trains it.)
- Bugs fixed (in lewam_seq.py): `get_cost` cold-start crash (reshape(-1) on 0-elem tensor when t0==1); `_apply_action_head` now takes per-position horizon (B,T) not just (B,) — each frame is a different distance from goal, rollout feeds current-frame horizon; `forward(return_z=True)` exposes z,e for FDM targets + SIGReg.
- New trainer `scripts/train_lewam_seq.py`. Recipe: full `num_frames` windows (every PE slot trained, no inference extrapolation), per-position horizon, goal-dropout (zeroed goal latent), losses w_act/w_dyn/w_reg(SIGReg)/w_rollout, `--stream`, per-epoch collapse metrics. Encodes B*num_frames frames/step (~nf× the split model → memory heavy: use small batch + `--grad_ckpt`).
- Ablation flags: `--w_reg` (SIGReg), `--w_rollout --rollout_steps --rollout_closed_loop` (rollout loss; `--rollout_stopgrad` parked, we chose end-to-end SIGReg-only), `--goal_dropout_p` (0=GC,1=BC,0.5=mixed), `--num_frames`, `--action_head` (mse now; gmm/diffusion reserved, GMMHead/DiffusionHead exist in module.py), `--w_act --w_dyn`.
- Model = 12.22M params (encoder 5.80M, predictor 5.61M, heads ~0.6M).
- Validated: CPU smoke (forward+all losses+rollout open/closed+backward) OK; pilot tworoom 5ep OK — act 1.00→0.92 (val 0.95), latent NOT collapsing (eff_rank 3%→10% rising, z_std ~0.84). First pilot OOM'd on a 22GB slice → fixed with batch 32 + `--grad_ckpt`. Pilot ckpts `ckpts/lewam_gc_repro/SEQ_PILOT/`.
- WATCH: dynamics train/val gap (train 0.05 vs val ~1.1) — likely BN-in-projector train/eval stats mismatch (affects FDM/CEM not BC); should tighten over 50ep. Revisit BN→LN in projector if it persists.

## Open items / next steps
1. **Breadth runs — LAUNCHED (all 4 tasks), recipe `--num_frames 8 --goal_dropout_p 0.5 --w_reg 0.04` rollout off, bs64 `--grad_ckpt`, 50ep.** Venue solved: script-wrapped `mlx worker launch` from a headless session (`tail -f /dev/null | script -qfc "mlx worker launch ... --no-input -- bash $HCODE/<entry>.sh" launch.log`, held open by a background command — worker dies if it exits; NO 4h cap on a worker). Two workers:
   - `SEQ_BREADTH` (worker 996941, 2×A100): tworoom (gpu0) + cube (gpu1), `--stream`. tworoom CONVERGED (best_val 0.648, SR eval'd — see item 2); cube training.
   - `SEQ_BREADTH2` (worker 997135, 2×A100, ~2TB RAM): pusht (gpu0) + reacher (gpu1), PRELOAD (user: "enough RAM"; preload ~141G pusht + ~121G reacher, faster epochs than stream, numerically identical). Entry `$HCODE/lewam_seq_breadth2.sh`. Uniform `--keys_to_load pixels,action` (pusht has no `observation` col; seq trainer only uses pixels+action). ~15 min/epoch preload-GPU-bound → ~10-12h.
   Monitor via HDFS `ckpts/lewam_gc_repro/SEQ_BREADTH{,2}/` collapse_metrics.json + hb_*.log (NOT the per-arm hb, which only logs START/DONE — read collapse_metrics.json for epoch progress).
2. **Eval adapter for LeWAM-Seq**: DONE + SR-validated end-to-end (both modes, tworoom). `gip.load_lewam_seq_model(run_name)` rebuilds LeWAMSeq from `lewam_seq_config.json` + loads `lewam_seq_best.pt`; `gip.LeWAMSeqPolicy` drives it. Two modes wired into `eval_gip.py`:
   - `+gip_eval.mode=seq_policy` -> `model.get_action` (AR/BC), reactive receding-horizon, one goal-conditioned block/replan.
   - `+gip_eval.mode=seq_cem` -> self-contained CEM over `model.get_cost` (knobs `gip_eval.cem_samples|cem_iters|cem_elites|cem_horizon`).
   Run: `python scripts/eval_gip.py --config-name <task> policy=<seq_run_name> +gip_eval.mode=seq_policy` (add `+seq_which=latest` for the latest ckpt). The policy shares ONE sliding num_frames-window history (frames + z-scored action blocks), z-scores past actions and un-z-scores the head output using the model's OWN stored stats (`action_mean/std`,`frameskip`,`action_raw_dim`), so it ignores `process['action']` (no double-normalization); `model.eval()` is forced (BN-in-projector). CPU-validated (strict load, exact z-score round-trip, finite actions) AND env-validated on the converged tworoom ckpt (3 seeds N=50, co-located on the freed GPU0 of the training worker via `mlx worker login <id> -- bash $HCODE/seq_eval_tworoom.sh [seq_policy|seq_cem]`):
   - **seq_policy (action head / BC): SR = 94.0%** (seeds 42/0/1 = 94/98/90) -- vs split baseline tworoom 100.
   - **seq_cem (CEM over FDM): SR = 71.3%** (74/64/76) with default CEM knobs (256 samples, 3 iters, 32 elites, horizon 5).
   So the action-head path clearly wins on tworoom (confirms the handoff hunch); seq_cem trails despite the FDM being accurate (tworoom final val_dyn=0.009) -- CEM knobs likely need tuning, or the BC head is just the better policy here. tworoom final: val_act 0.640, z_std 0.99, eff_rank 76.7 (40%), no collapse. STILL TODO: eval cube/pusht/reacher once they converge; compare to split baseline (reacher 98 / pusht 88 / cube 100). Eval-launch traps: use the plain `mlx worker login -- bash <script>` form (NOT `bash -c "..."` -> hits a missing mlxrc rc and never runs the script); the login blocks on the remote script so the session stays up for the whole eval; hardcode `CUDA_VISIBLE_DEVICES=0` in the script to pin the freed GPU (cube keeps GPU1).
3. **Depth ablations** (after breadth works): rollout off/open/closed, goal-dropout, num_frames — likely on cube+pusht where compounding error matters.
4. Split trainer `train_lewam_gc.py` (on lewam-seq) is lewam-arch's version — lost my streaming/collapse/flatten-fix in the merge; re-add if re-running the split model on capped nodes.

## Entry-script pattern (reuse for launches)
Tarball on HDFS → entry script untars to /opt/tiger, `export PYTHONPATH`, pip-install if imports fail (image usually has torch/swm), symlink datasets from `$HROOT/datasets`, `export STABLEWM_HOME=/opt/tiger/.swm* MUJOCO_GL=egl HF_HUB_OFFLINE=1`, run `python3 scripts/train_lewam_seq.py ...`, background ckpt-sync loop to `$CK/ckpts_live` every 5 min, per-arm heartbeat FILES. See `$HROOT/code/lewam_seq_pilot.sh` for a working example.
