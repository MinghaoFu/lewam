# Canonical training recipes (set in stone)

Every number on the board traces to one of these recipes. Cite the recipe name when
launching; deviations must be listed explicitly in the run's config note. Sources: the
trainer argparse defaults, the launch entries on HDFS `code/`, and the dumped configs
beside each checkpoint.

## Cell → protocol map (authoritative, owner-stated 2026-08-22)

- **TC** (task completion): toolhang, cube, drawer, transport. **DEFAULT (owner
  2026-08-24): goal-image conditioning ON, horizon OFF** — goal-blind reactive is an
  appendix ablation for cells that permit it (toolhang). Eval: `+gip_eval.full_traj=true`
  (start = episode frame 0, goal = terminal frame, per-episode budget 2× length) for
  toolhang-class evals; drawer/transport use their `configs/eval/<cell>.yaml` budgets.
  Success = env criterion.
  **CUBE EVAL WAS BROKEN (audit 2026-08-24): `configs/eval/cube.yaml` samples starts
  uniformly over ALL rows (gip.sample_eval_episodes default mode) with the env target set
  from `goal_privileged_block_0_pos` at start+25.** On full cube (69% post-success tail)
  ~69% of starts are trivial (target ≈ current resting position, terminate_at_goal) —
  jf's 66 ≈ the tail fraction, and its 19/50 failure videos ≈ the pre-success starts.
  Every cube SR measured this way (jf 66, uni-ViT 74.7, uni-r18sp 51.3) is protocol-
  confounded. The correct cube TC eval is full_traj (start 0, target = terminal-frame
  block placement, per-episode budget).
  **HONEST CUBE BOARD (full_traj, 3×50, 2026-08-24): uni-ViT (goal-cond) {38,48,34} =
  40.0 · jf (goal-blind) {18,22,26} = 22.0 · uni-r18sp (goal-cond) {14,18,8} = 13.3.**
  Ordering matches the broken protocol minus ~35-44 pts of tail inflation; goal
  conditioning ≈ doubles honest cube; the ViT-over-ResNet gap survives (+26.7). jf
  goal-cond arm (--goal_conditioning --goal_terminal, 75 ep) launched to fill the 4th row.
  **CUBE SOLVED BY GOAL-COND JF (2026-08-25): tc_cube_gcf_s42 (--goal_conditioning
  --goal_terminal --fp32, 75 ep, success-restricted aux) = {96,96,94} = 95.3 on honest
  full-traj — vs goal-blind jf 22.0, restricted uni-ViT 40.0, full-data unified 96.7.
  The goal image is the whole story on cube; the restriction is NOT binding for jf
  (95.3 ≈ the full-data ceiling), so the restricted-unified 40 is a unified-recipe/
  budget issue, and success+K extension is unnecessary. Board cube row = jf goal-cond
  95.3.**
  **CROSS-EVAL (2026-08-25): the OLD full-data unified (wf8_uni8/cube/res_base, the
  e2e-96 arm) scores 96.7 {96,98,96} under the NEW full-traj protocol on train episodes,
  and the new restricted-trained vit scores 40.0 on the old eval_tc split — protocol and
  split are BOTH irrelevant; the ~57-pt gap is training. The honest-protocol cube
  ceiling is ≥96, held by FULL-data training. The success-restriction cuts the settling
  segment (success key = grazing first-touch, 0.033 from target; final frame = settled,
  0.004) plus 69% of data; owner direction: consider restriction at success+K frames
  (~+10) or full-data TC training for the next cube round.**
- **TWOROOM CANARY (jointflow GR recipe, H_max 25, 50 ep s42, 2026-08-24): {96,100,100}
  = 98.7** — the flow model is near-perfect on the easiest 2D-action GR cell, so the
  pusht gap is pusht-specific, NOT a generic low-action-dim failure. Flow action loss
  sat flat at ~1.20 all run while SR ≈ 100: the loss floor is irreducible conditional
  action entropy — never compare it across datasets or read it as policy quality.
- **PUSHT SIGReg (anchor + w_reg 0.04, s42, 2026-08-24): {70,82,70} = 74.0 vs noreg
  anchor 65.0** (+9.0 at n=150, marginally past the ~8-pt noise line; single training
  seed). Latent spread held (zstd 0.86 vs 0.02 collapsed); best jf pusht to date.
  Supports the SIGReg-matters-more-at-low-action-dim hypothesis.
- **PUSHT WIDTH x SIGReg 2x2 (2026-08-25): 384-noreg 65.0 · 384-SIG 74.0 · 192-noreg
  58.0 · 192-SIG {56,78,68} = 67.3.** Additive, no interaction: SIGReg ≈ +9 at either
  width (spectral, width-independent); width ≈ +7 at either regularization (capacity
  matters on its own — smaller embeddings never win). ViT-tiny+SIG 62.0 (encoder swap
  loses -12 vs ResNet+SIG at matched losses). Single train seeds; slim rows carry the
  120-ep/lr-1e-4 slim budget.
- **PUSHT PLANNING LADDER (noreg ckpt, 2026-08-25, final): seeded same-pod controls put
  reactive 67.2 (n=250) · k=1 exec5 71.2 (n=250) · joint-scored BoK-32 67.3 · INPAINT-
  scored BoK-32 (shared-noise ranking) 70.7 — all one band; warm-CEM (std 0.2) 15.3;
  cold-CEM (unified-era) 2-8. Diversity probe: policy samples are action-diverse
  (act_div 0.34) but joint-sample imagination noise is 11.7 transitions vs an
  action-caused 1.14 (cost SNR ~0.1); fixing the noise (inpaint scoring) restores the
  ranking signal (0.77 top-1) yet SR stays in-band — pusht is POLICY-limited, not
  selection-limited. sig04's imagination is near action-blind (top1 .26): SIGReg trades
  dynamics causality for SR. sig04 planning (owner-requested despite the probe): inpaint
  67.3 / joint 70.7 vs reactive 74.0 — null there too; selection is null on BOTH ckpts.** Dynamics probe (probe_jf_dynamics.py, n=200, K=15): noreg
  imagination is action-causal (true-action top-1 0.77, sensitivity 1.14) with THIN
  margins (cost cv 0.069); sig04's is near action-blind (0.26/0.26) — SIGReg trades
  world-model causality for policy SR. CEM fails because it executes a refit MEAN of a
  multimodal flow policy and its goal-cost is progress-dominated; best-of-K executes a
  real sample. NO planning on TC cells unless the owner explicitly asks.
- **GR** (goal-reaching, goal+horizon-conditioned policy): pusht, tworoom, pointmaze,
  reacher. Eval: goal at `goal_offset_steps` (pusht 25 raw), budget from config,
  horizon countdown `min(steps_left, H_max)/H_max` in obs-steps, replan = 1 action block.
- `goal_offset_steps` appearing in a TC cell's eval yaml does NOT make it GR.

## Runtime settings (part of the recipe — the 2026-08-23 lesson)

The reference trainers (`train_lewam_gc.py`, `train_lewam_unified.py`) BOTH run:

```python
torch.backends.cudnn.benchmark = True
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # train AND val steps
```

`train_jointflow.py` / `train_crossattn.py` historically ran fp32 eager with neither —
a deviation from the references discovered 2026-08-23. VALIDATED SAME DAY: a bf16 rerun of
the pusht GR anchor scored {60,70,66}=65.3 vs fp32's {58,74,66}=66.0 (within noise), with
matched final loss and a cleaner val curve. bf16 + cudnn.benchmark is now the jointflow
default (`--fp32` restores the old behavior). Note the speedup is ~10%, not 2.5–3×: the
fp32 path already ran TF32 tensor-core convs on H100; bf16 buys bandwidth, not FLOPs.
Numbers before 2026-08-23 are fp32-trained; from the bf16 default onward, new trainings
are bf16 — both regimes are SR-equivalent per the validation arm.
**bf16 CAUTION for TC cells (2026-08-25): 2/2 fitting bf16 TC arms NaN'd mid-training
(cube goal-cond ep 45, toolhang ViT-small ep 33) while their fp32 reruns and all bf16
pusht-GR arms run clean. Until diagnosed, TC-cell trainings pass `--fp32`.**

## Recipe: lewam_gc (per-cell GR baseline; pusht board 82.7)

`scripts/train_lewam_gc.py --dataset_name <cell>.h5 --epochs 50 (pusht repro; default 100)
--H_max 50 (25 tworoom) --w_cyc 0.0` — ViT-tiny z_dim 192, SIGReg w_reg 0.04, w_dyn 1.0
(goal-conditioned dynamics), split LRs encoder 1e-4 / head 3e-4 / dynamics 3e-4,
batch 256, fp16 pre-normalized RAM-preloaded obs cache, bf16 autocast, cudnn.benchmark.
Goal sampling h~U[1,H_max] clamped to tail. Wall: ~1–1.5 h / 50 ep (A100-era, pusht).

## Recipe: lewam_unified `idm05` (prev pusht GR SOTA, 87.2)

`scripts/train_lewam_unified.py` — encoder scratch ViT-tiny z 192, context_len 5 (causal
aggregator agg_depth 4), head_type mse, w_reg 0.04 (SIGReg), **w_idm 0.5**, w_dyn 1.0,
dyn_action_embed_dim 128, H_max 50, goal sampling default RANDOM (p_terminal_goal 0,
p_shared 0, close_bias 0), split LRs as gc, bf16 autocast, cudnn.benchmark. Window
sampling: one item = ctx_len decision positions (epoch = decision points / avg_cover).
Wall: obs-cache cells ~1–1.5 h; wf8 raw-anchor pusht (200 ep, bs 48, mmap) 13.2 h.

## Recipe: jointflow TC (`noreg`; toolhang 89.0 pooled / 91.3 s42)

`scripts/train_jointflow.py --dataset_name <podname>.h5 --epochs 120 --warmup_epochs 10
--batch_size 64 --num_workers 6 --num_actions_pred 10 --num_states_pred 1 --w_reg 0
--seed 42` — raw cache (uint8, RawContextDataset, consecutive-frame history hl 2), lr 1e-4
uniform (UWM precedent; deviation from refs' split LRs), 384-wide 6-head depth-8 trunk
(~42M), tau_cond per_modality, fp32 eager (see runtime note). RAM-preload the cache when
it fits (drawer 42G, transport 59G); cube 282G needs --cache_mmap + the success-patched
aux (working set ~88G) or an episode subset.
Board: toolhang 89.0/91.3 · transport 86.7 · drawer 68.0 · cube (success-restricted,
75 ep) 66.0.

## Recipe: jointflow GR (pusht round-2; 65.0 pooled / 66.0 s42)

TC recipe plus `--fs_strided --goal_conditioning --num_actions_pred 10 --num_states_pred 1
--policy_history_len 2 --H_max 50 --p_drop_goal 0 --epochs 50 --warmup_epochs 5
--batch_size 128 --lr 1.5e-4` (1.5e-4 verified from the gr2 config dumps 2026-08-24 — an
earlier revision of this doc said 1e-4, which was the slim-192 arm's lr, not the anchor's),
u8 fs-strided cache RAM-preloaded. GCHeadMSE readout-only
conditioning; anchor-spaced history (matches eval adapter). Pixel scale: cache uint8 →
normalize on GPU (`was_uint8` branch) — the dataset must NEVER float frames (the ep-48
pixel-scale bug).

### Historical unified receipts (config dumps, HDFS `ckpts/`)

- `lewam_unified_final/ckpts_live/cube_uni_final_p00` (2026-07-24) — the old cube ~100
  run: scratch ViT-tiny CLS z 192, ctx 5, agg_residual, MSE head, teacher forcing only,
  random goals (p00), full episodes. Its dump predates the `rollout_k`/`encoder_backbone`/
  `head_type`/`anchor_rate` flags.
- `.../pusht_uni_roll2` (+`_s1`/`_nogoal`/`_ng_acons`, 2026-07-30) — scratch ViT + MSE +
  `rollout_k 2, w_rollout 1.0`, w_idm 0: the "roll2" element of the owner's best-config
  trio.
- Every wf8-era TC arm (`wf8_uni/*/rawchunk_*`, `matched_*`) ran `resnet18sp + flow
  flow_H 2` with `p_terminal_goal 1.0 + ablate_horizon` (with that pair, H_max is fully
  inert: terminal goals skip the U[1,H_max] draw and noh zeroes h_norm; without noh,
  H_max still saturates the countdown at min(dist, H_max)/H_max).
- Diagnostic RESULT 2026-08-24 (`uni_cube_tc_vit`/`uni_cube_tc_res`, 50 ep warmup 5,
  raw anchors on the success-restricted cube aux, MSE + rollout_k 2, eval 3 seeds x 50):
  **ViT {70,76,78} = 74.7 · resnet18sp {48,58,48} = 51.3**; jointflow board 66.0 sits
  between. Verdict: (a) the success restriction is the dominant factor — the owner-best
  recipe lands ~75, not ~100, so cube-TC is simply harder than old full-episode cube;
  (b) the encoder axis is real on cube: +23.4 ViT over resnet18sp under an otherwise
  identical recipe; jointflow's resnet18dp is partially implicated in its 66. Toolhang
  gives NO usable unified encoder signal on the board (full-traj) protocol: noh_t100 vs
  r18sp_t100 = 4.0 vs 8.0 there (s42 only, floor) — the ViT-ahead numbers (47.3 vs 33.3,
  3 seeds) are SEGMENT-protocol only. Label the protocol when citing that era. Val action loss inverted the SR order
  (ResNet best_val 0.080 < ViT 0.096) — never compare val loss across encoders.
  Wall (H100): ViT 213-217 s/ep, ResNet 266-269 s/ep; jointflow cube 75 ep = 481.6 s/ep
  — the 3.2x total gap decomposes as ~2.2x per-anchor compute (trunk size + window
  amortization) x 1.5x epochs, no residual.

## Recipe: DP-T baseline (budget-matched; toolhang board 71.0)

`wf8_dp_cell_v5` entry — convert wf8 h5 → robomimic hdf5 (`--no_proprio`,
`--env_args_json` placeholder; cube adds `--truncate_at_success success`), train
`train_diffusion_transformer_hybrid_workspace` with the cell's `*_ours_image_noprop.yaml`
(single 224 view, no proprio), `training.num_epochs` = the cell's budget (toolhang board
artifact = noprop snap_ep120), crop 202. Runs die at the 3 h util wall → checkpoint-resume
chain; sync copies MUST be atomic (cp→mv) and seed verification MUST use
`torch.load(..., weights_only=False)` (torch ≥2.6) with dill importable.
**`training.checkpoint_every` MUST be < epochs-per-link (v6c sets 2).** v5/v6 hardcoded 20
while links die at ~15–17 epochs: any cell whose resumed epoch counter restarts low never
reaches the next %20 event, so `latest.ckpt` re-freezes at the seed and the chain loops at
~+1 net epoch per 3 h link (cube and transport did exactly this, 2026-08-24; drawer escaped
because its counter stayed cumulative and crossed ep 100/120 — which also means drawer's
synced latest.ckpt IS the exact ep-120 budget artifact, epochs 121–124 were never saved).
v6b adds epoch-snapshot sync (epoch=*.ckpt) so budget artifacts survive the wall.

## Eval invariants

- Eval code tarball at-or-newer than the training commit (per-modality weights load
  silently into older wiring otherwise).
- Toolhang-family evals need `wf8_sim_src.tar.gz` on PYTHONPATH (robomimic 0.5.0 source;
  PyPI tops out at 0.3.0 = mujoco_py era) + the robosuite/mujoco pip block; drawer/
  transport use the dexmg env with repo-asset env metas and per-cell thresholds
  (transport eef 0.10 on purpose); cube's dataset symlink lives under `datasets/ogbench/`.
- **Flow-noise seeding (restored 2026-08-25, commit after 3b1649e):** jf adapters now draw
  ALL flow/sampling noise from a per-eval generator seeded by `cfg.seed` (the flow-head
  rule; the old unified CEM had it, the jf adapters missed it). Every jf SR measured
  before this fix used unseeded flow noise — statistically valid, but same-seed reruns
  jitter (observed ±4-6/seed, e.g. reactive s42 66.0 vs 62.0); marginal gaps (<~10)
  from that era carry that extra uncertainty.
- Model selection: pooled average across seeds; best-checkpoint = min val action loss.
- n=50 episodes per eval seed; eval seeds {42, 0, 1}; SR gaps < ~8 pts at n=150 are
  within noise — run controls before claiming gains.
