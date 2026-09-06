# Canonical training recipes (set in stone)

Every number on the board traces to one of these recipes. Cite the recipe name when
launching; deviations must be listed explicitly in the run's config note. Sources: the
trainer argparse defaults, the launch entries on HDFS `code/`, and the dumped configs
beside each checkpoint.

## BOARD INVALIDATED 2026-09-06 -- read before citing any number below
The mse dynamics head (every arm on the board: state_head mse) was trained with its state tokens
AdaLN-modulated on tau_s ~ U(0,1) and queried at tau_s = 1 at inference (imagine_step / sample); the context
tokens (history, clean actions, goal) were modulated by a constant tau = 1 in both. Owner: "ALL THE RESULTS
ARE INVALID." Every planner row (best-of-K, gradient, CEM, random-candidates) ran on that mismatched readout;
every mse-head checkpoint is a model that was not the one designed. DP rows are unaffected (different model).
FIX (commit 67e20e1, spec by the owner): tau only for a flow branch, on that branch's slots; the horizon on the
action slots whenever goal-reaching, regardless of head; NO conditioning on history / goal / clean actions --
a plain pre-LN transformer everywhere except the conditioned slots (classic AdaLN-Zero there). Guards: block ==
hand-written reference bit-for-bit; mse readouts independent of every tau; attention mask / layout identical
through the renames. Old checkpoints do not load on 67e20e1 (module and key names changed); old tarballs remain
for reading old checkpoints. EVERYTHING on the board is to be retrained on 67e20e1+ before any number is
cited; the tables below are the historical record of the invalid runs until then.
HARD RULE from the same day (memory design-before-code): no implementation without the exact design and
pseudocode verified first -- the pseudocode review caught a second mismatch (per-stream conditioning would
have modulated the clean-action tokens) before it shipped.

## Post-fix retrain wave 1 (2026-09-06, tarball 67e20e1) -- launch record and the audits behind it

### Protocol audit (owner TODOs 1-2; read from code, file:line)
GM = goal-state match against the dataset frame at start+offset; TP = the env's own task predicate.
- Success is LATCHED on every cell: `stable_worldmodel/world/world.py:543` (`episode_successes |= terminateds`);
  a succeeded env freezes (`reset_mode='wait'`, `world.py:443`). No cell judges the final frame; no hold
  requirement (`GOAL_HOLD_K=1` everywhere). Overshooting the goal after reaching it never costs a success.
- pusht GM: 4-D norm over (agent xy, block xy) < 20 px AND |dangle| < pi/9 (`envs/pusht/env.py:347-355`; the
  agent pose is inside the rule; `+gip_eval.success=strict` = block only, unused by any jf_* entry). tworoom GM:
  agent < 16 (`two_room/env.py:272`). pointmaze_large GM: xy < 1.0 (`lewam/envs/pointmaze_env.py:129-139`).
  reacher / reacher_policy GM: all |qpos - goal qpos| < 0.05 rad (`custom_tasks/reacher.py:27-33`; arm joints only).
  cube-as-GR GM on the cube only: <= 0.04 m (`cube_env.py:1318-1344`); the end-effector term EXISTS but is inert
  (`CUBE_EEF_THRESHOLD` defaults to inf, no jf_* entry sets it; `lewam/envs/cube_env.py:33-56`).
  toolhang / transport / drawer: GM OR TP (`lewam/envs/robomimic_gc_env.py:242-253`, `dexmimicgen_env.py:277-306`),
  thresholds 0.04 objects / 0.04 eef (toolhang, drawer), 0.04 / 0.10 (transport); the eef is inside the GM term;
  drawer's eef1 slice is the known-wrong one (`dexmimicgen_env.py:39-44`).
- "TC" is NOT goal-blind at the env: the goal callables are passed for every gip_eval.mode (`eval_gip.py:490-499`),
  so a goal-blind policy is scored against a goal it never saw. transport / drawer: goal = +50 raw steps from a
  random start, so those numbers are mostly a 50-step goal-reach OR task. toolhang (full_traj): goal = terminal
  frame, so GM ~= TP.
- GR cells (pusht / tworoom / pointmaze / reacher) are evaluated on the TRAIN h5 (`jf_grev_966463b.sh:35-39`);
  `--train_split 0.9` holds out decision points, not trajectories. Inherited from the LeWM protocol; say so in the paper.
- Budgets: 50 raw steps (pusht / tworoom / reacher), 100 (pointmaze / transport / drawer), 2 x (len-1) per episode
  under full_traj; 50 episodes x eval seeds 42 / 0 / 1; default start sampling is row-uniform over the h5.

### Train/eval audit (owner OQ1): no bug found
- Pixels: trainer `_IMG_MEAN/_IMG_STD` = ImageNet = eval `img_transform` (`lewam/models/gip.py:40-48`); the goal
  frame takes the same transform (`eval_gip.py:241`).
- Actions: z-scored at cache build (`scripts/make_preload_cache.py:162-198`), stats dumped into the checkpoint
  config (`train_jointflow.py:212, 441`), un-z-scored on the execution path (`gip.py:970-971, 1202`).
- History padding: repeat-edge + attention mask in training (`train_jointflow.py:258-265`); the adapter left-pads
  the same way (`gip.py:1196-1200`).
- Loss masking: action blocks valid while inside the episode (`train_jointflow.py:267-271`); `_masked_mse` divides
  by the valid count (`motflow.py:366-368`). There is NO goal-relative mask: at h = 1 the second action block is
  supervised by the demo's continuation past the goal (the overshoot in the videos). Since success is latched the
  overshoot is free; masking past-goal actions is a design decision, deferred until the wave-1 numbers.
- HORIZON UNITS (the owner's question): `H_max` is counted in fs-states (anchors), never raw steps, by all three
  trainers (`train_lewam_gc.py:157`, `train_lewam_unified.py:144-157`, `train_jointflow.py:282-284`) and by the
  adapters (`gip.py:513-516, 1206, 1353-1355`), so `--H_max 50` = 250 raw steps at fs 5, including for the headline
  lewam_gc numbers (`train_lewam_gc.py:81-83` keeps one frame per frameskip; `train_lewam_unified.py:197-211`
  counts prediction steps under both anchor rates; the crossattn/split trainer feeds horizon = 0 and a terminal
  goal, `train_crossattn.py:184, 297`; the GCIDM / split / unified adapters divide the raw goal offset by the
  frameskip, `gip.py:852-855, 942-943, 1013-1025`). The owner remembers raw steps ("otherwise the recipe would
  make no sense"); if so it was the L40S le-wm-repro lineage, unreadable from this machine. Checked against the
  counter-hypothesis "obs-step = raw step / h_max // fs somewhere": no occurrence of h_max/H_max in either trainer
  divides by frameskip (all listed above), and a ROW of their frame tensor is one observation = frameskip raw steps
  (preloads keep `frames[:n_obs+1]`, `n_obs = L // frameskip`, actions grouped into blocks of frameskip per row).
  Empirical, on the toolhang caches: strided row k == raw row 5k to |diff| 2e-4 (vs
  0.2-0.5 against raw row k) for k in {1,2,3,10,20,50}; first episode 136 strided rows vs 676 raw (ratio 4.97). The owner meant 50 RAW
  steps (= 10 fs-states). Measured on the pusht u8 cache index: episodes
  have 10..50 anchors (mean 25.4, median 25), the tail clamp `h = min(U[1,50], tail)` binds at 100% of decision
  points, 74% of training samples had the goal ON THE TERMINAL FRAME, effective h median 10 anchors, h >= 25 in 7%,
  while eval asks for 5 anchors (25 raw) or 10 (50 raw) counting down to 1. toolhang-strided: 27% terminal, mean
  tail 49.8. Decision: pass `--H_max 10` (fs 5) = 50 raw; no code change -- the flag overrides the entry's
  `--H_max 50` (argparse last wins) and lands in the checkpoint config, which the adapters read (`gip.py:964`).
  tworoom would be `--H_max 5`. Draw stays uniform-then-clamped; a uniform-over-the-feasible-range draw was
  proposed (one line at `train_jointflow.py:282`) and not approved yet.

### Width facts (dumped configs on HDFS)
pusht board arm `jointflow_gr_pusht_nm192` = 192 throughout (z 192 / proj 384 / d 192 / depth 4 / heads 4).
toolhang TC `jointflow_tc/tc_toolhang_nm192_s42` = 192 throughout (zstd floor 0.005, 120 ep, batch 64, lr 1e-4,
warmup 10, 3.1 h on H100); `tc_toolhang_mnm192_s42` = d 192 with z 384 / proj 768 (its launch string never set
--z_dim). The retrain uses 192 throughout on both cells.

### Wave 1 (owner 2026-09-06: "launch flow noreg and mse noreg directly on pusht, toolhang first (4 arms). If
they are for sure training and resources allow, stage sigreg flow and mse. If the first 4 fail due to collapse
specifically, replace them with pure sigreg (no projected)")
- Code: `code/lewam_jointflow_67e20e1.tar.gz` (git archive of 67e20e1, md5 91235be9f18235863505564a52808f36; vs the
  63d58f5 tarball: __pycache__ gone, + configs/eval/reacher_policy.yaml, lewam/envs/reacher_visible_target_env.py,
  .gitignore). Entries `jf_gr_67e20e1.sh` / `jf_tc_67e20e1.sh` / `jf_tc_ev_67e20e1.sh` = the 96618d4 entries
  verbatim except the tarball line (+ one header line), chmod 444.
- Verified before submit: the real `train_jointflow.py` CLI on CPU on 2-episode slices of the real toolhang caches
  (strided GR cache + raw TC cache) x {mse, flow} x {noreg, pw_zp} + a resume run: 9/9 rc 0, finite losses, every
  checkpoint file written; the toolhang recipe re-smoked at 192 throughout: 2/2; the eval-side loader
  (`gip.load_jointflow_model` strict load, `encode` on real frames, `sample`, `imagine_step`): 8/8 finite.
- Pods: 1 x H100, cpu 16, memory 120000, bi_research `compute-23-aliyun.va-cloudnative-aigcp-bi.research-guarantee`
  (10 H100 free at launch; A100 = 0 in both groups). YAMLs `~/lewam_project/jobs/fix_wave/`, staged by
  `stage_fix_wave.py` (scratch), submitted through submit_guard.sh.
- toolhang TC (jf_tc entry: raw cache, 120 ep, batch 64, warmup 10, lr 1e-4, `--w_reg 0 --fp32`, train-only,
  ckpt `ckpts/jointflow_tc/tc_toolhang_fx_<arm>_s42`), EXTRA = `--model motflow --grad_probe_every 250
  --zstd_floor 0.005 --z_dim 192 --proj_hidden 384 --d_model 192 --depth 4 --n_heads 4 --mot_state_head {mse|flow}`:
    th_fx_mnm192 (mse)  job acc3c8e66c52c1a7  caption mf-74825ef5   START 16:46 n124-136-240 H100 (verified)
        TRAIN_OK 19:59: ep 120/120 train act 0.276 state 0.00128 zstd 0.093 | val act 0.267 state 0.00114 zstd 0.094
        (the pre-fix nm192 ended at act 0.277 / state 0.0014). Evals (jf_tc_ev_5cca99f.sh, timing json on):
        a (EV_SEEDS "42 0") job 27185eecece7becd mf-bd9034ab; b (EV_SEEDS "1") job 194c4927eea2db7d mf-186fbd34, 20:01.
        FIRST POST-FIX NUMBER 21:16: JFTC toolhang evseed_1 success_rate 88.0 (job b, 70 min; invalidated nm192 row: 86.7).
        Timing (H100, reactive, 50 envs, full_traj budget 1482): block 6.5 ms amortized (call 76 ms over ~24 envs),
        episode total 0.96 +- 1.24 s per env, 103.6 replans/env (~518 raw steps), 215 replan calls; eval wall
        4055 s -> the policy is ~1% of a toolhang eval, the rest is MuJoCo + rendering.
    th_fx_mfl192 (flow) job 38216e5ecfaa0682  caption mf-133b6087   START 16:47 n124-112-071 H100 (verified)
        TRAIN_OK 20:09: ep 120/120 train act 0.318 state 0.0174 zstd 0.025 | val act 0.309 state 0.0153 zstd 0.024
        (latent scale drifted 0.055 -> 0.024 over training, 5x the 0.005 floor; no collapse kill). Evals:
        a (EV_SEEDS "42 0") job 825f2f3607e3ed99 mf-bcaf046e; b (EV_SEEDS "1") job 2f1a82e3b1f27dfa mf-3623c2b2, 20:11.
  Evals follow on `jf_tc_ev_67e20e1.sh toolhang tool_hang.h5 toolhang tc_toolhang_fx_<arm>_s42` (board protocol:
  mode jointflow_policy, full_traj on the eval split), two jobs per arm (`EV_SEEDS="42 0"` then `"1"`; ~80 min per
  seed; the 3 h util wall cut the board's 3-seed job after two seeds).
- pusht GR (jf_gr entry: the nm192 recipe + chained gc / best-of-K 32 / grad 50 x 0.05 / random-candidates evals,
  3 eval seeds x 50; ckpt `ckpts/jointflow_gr_pusht_fx_<arm>/fx_<arm>_s42`), EXTRA = `[--mot_state_head flow]
  --w_reg 0 --H_max 10 --zstd_floor 0.005 --z_dim 192 --proj_hidden 384 --d_model 192 --depth 4 --n_heads 4`
  (`--H_max 10` = the owner's stated cap, 50 raw steps = 10 fs-states; launched under the "4 noreg arms" go once the
  unit question was settled with evidence, 2026-09-06 17:0x):
    pu_fx_nm192 (mse)  job a4c959015467f38d  caption mf-d1d52bca
    pu_fx_fl192 (flow) job 260ef01f66b8053b  caption mf-9e11cf78
- SIGReg arms (pw_zp = `--w_reg 0.04 --sep_policy_state`; the projection is automatic at w_reg > 0,
  `train_jointflow.py:436-438`), launched once the noreg arms were verified training (owner's conditional go):
    th_fx_msig192 (mse, pw_zp)   job 68eae863917763ce  caption mf-a38edabb   submitted 17:08 (toolhang noreg at ep 11, val zstd 0.17 / 0.055)
    th_fx_mflsig192 (flow, pw_zp) job a08082cbb2c709a4  caption mf-000f1c4f   submitted 17:08
        TRAIN_OK mse-pw_zp 20:47: act 0.285 / val 0.271, state 0.0029, reg 1.04, val zstd 0.174 (noreg: 0.094).
        TRAIN_OK flow-pw_zp 20:47: act 0.343 / val 0.337, state 0.0198, reg 2.47, val zstd 0.068 (noreg: 0.024).
        No toolhang arm collapsed -> the plain-SIGReg fallback is not needed on toolhang. Evals (20:50-20:52):
        mse-pw_zp a 36ac9874e8ee30e1 mf-6afffc39, b 09a6a508cf6bd145 mf-df3311dc;
        flow-pw_zp a f483de8e850e5ebf mf-5ccaadc5, b 551e82a0e51ed5e1 mf-9e1a70eb.
    pu_fx_sig192 (mse, pw_zp)    job f2f74cfcf94bbf0b  caption mf-ae6704c0   submitted 17:42 (pusht noreg at ep 4/50, 21 H100 free)
    pu_fx_flsig192 (flow, pw_zp) job 0e140cc23b34c43d  caption mf-c6607e1b   submitted 17:42
  Wave 1 = 8/8 arms launched by 17:42. Pusht noreg pace 6.4 min/epoch (TRAIN_OK ~22:40); toolhang noreg 89 s/epoch
  (~19:45), toolhang pw_zp 111-117 s/epoch (~21:00). Early trend (epoch 4, pusht): mse-noreg act 0.741 / val 0.683,
  val zstd 0.241, no train/val gap; flow-noreg act 0.822 / val 1.349, state 0.044 / val 0.395, val zstd 0.056 --
  the SAME early pattern the old (pre-fix) pusht flow arm s5f192 showed (val act 1.3-1.6 vs train 0.8-1.0, val state
  0.5-2.0), i.e. the flow state head's known behaviour on pusht's small latent, not something the fix introduced.
  Ranking assumed (owner): noreg > pw_zp (guaranteed no collapse) > plain SIGReg. Collapse fallback = a noreg arm
  that trips the zstd floor (COLLAPSE_KILL) is resubmitted as plain SIGReg `--w_reg 0.04 --sigreg_proj_dim 0` (no
  projection, no policy view; identity at `train_jointflow.py:524`); the pw_zp arms run regardless.
  pusht START lines: fx_nm192 17:03 n124-136-221, fx_fl192 17:03 n124-139-220; TRAIN_BEGIN 17:05 both.
- Not in the wave-1 TRAINING tarball (sequenced after): the gradient-TR / SteerMPC / CEM-policy planner rows (they
  go into the standalone eval entries), goal-relative loss masking, reacher_policy, the other cells.

### Wave-1 board, snapshot 2026-09-06 23:00 (collect_board.py --arm_prefix fx_; mean +- sample std over eval seeds
42/0/1, 50 episodes each; the invalidated board's value in brackets, for reference only)
  pusht  fx_nm192 (mse, noreg)   reactive 72.0 +- 5.3 {70,78,68} [70.0] | best-of-K 78.0 +- 2.0 {76,80,78} [80.7]
                                 | gradient 84.0 +- 7.2 {78,92,82} [86.7] | random-candidates 38.0 +- 6.9 {34,46,34} [39.3]
  pusht  fx_fl192 (flow, noreg)  reactive 66.0 +- 8.0 {58,74,66} | best-of-K 62.0 +- 6.9 {54,66,66} | gradient: seed 42 = 54 (running)
  toolhang fx_mnm192 (mse, noreg)     reactive TC 84.7 +- 4.2 {86,80,88} [86.7]   timing: block 5.8 ms amortized, call 81 ms, episode 1.09 s
  toolhang fx_mfl192 (flow, noreg)    reactive TC 82.0 +- 0.0 {82,82,82}          timing: block 7.8 ms, call 117 ms, episode 1.31 s
  toolhang fx_msig192 (mse, pw_zp)    reactive TC 80.0 +- 8.5 {86,74} (seed 0 running)   timing: block 4.9 ms, call 80 ms, episode 1.07 s
  toolhang fx_mflsig192 (flow, pw_zp) reactive TC 90.0 +- 5.7 {86,94} (seed 0 running)   timing: block 14.8 ms, call 122 ms, episode 1.24 s
  Reads: the conditioning fix leaves the pusht mse-noreg planner ladder intact (72 -> 78 -> 84, random-candidates 38),
  every cell within one std of the invalidated row; pusht flow-noreg still inverts the ladder (best-of-K < reactive);
  on toolhang the pw_zp arms do NOT pay the SIGReg penalty the old board showed, and flow-pw_zp leads on two seeds.
  Pusht timing comes with the eval re-run wave (the chained evals ran on the training tarball 67e20e1).

### Protocol additions 3 and 5 (design verified by the owner 2026-09-06 "Agree"; implemented the same day)
- Planning time (item 5): `GetActionTimer` in `scripts/eval_gip.py`, installed on every policy right after
  `build_policy` (before the random-goal clip wrapper). Definitions: `t_call` = wall seconds of one `get_action`
  call, CUDA-synchronized on both sides; block time = mean over replan calls of `t_call / n_replanned` (the
  batch-amortized cost of one action block; the raw batch `t_call` and `n_replanned` are reported beside it);
  episode total = per env the sum over its alive calls of `t_call / n_alive`, mean +- std over envs. Replan
  detection reproduces the adapters' own rule BEFORE the call (deque empty and not terminated, or a harness
  flush). Output after `world.evaluate`: `results_path/timing_<policy>_seed<seed>.json` (pod-local) and the same
  json on one stdout line `[timing-json] {...}` plus a human line `[timing] block X ms amortized (call Y ms over
  N envs) | episode total M +- S s` -- the entries already copy every eval log to HDFS, so no entry changes; the
  dump is wrapped so a timing bug can never fail an eval. Fields: policy, mode, seed, num_envs, plan_mode, plan_k,
  grad_steps, grad_lr, grad_tr, pm_K, plan_random_candidates, exec_actions, action_block, horizon_blocks,
  num_actions_pred, eval_budget, full_traj, gpu, wall_total_s, success_rate, n_calls, n_replan_calls,
  t_call_mean/std_s, n_replanned_mean, t_block_amortized_mean/std_s, t_episode_mean/std_s, replans_per_env_mean.
- Mean +- std (item 3): `scripts/collect_board.py --ckpts <root> --arm_prefix fx_` parses the entries' heartbeat
  result lines (`[stamp] [who] <TAG> <arm|cell> evseed_<s> success_rate: <x>`), keeps arms with the prefix
  (fx_ = trained on 67e20e1 or later, so the invalidated board never leaks in), last line per seed wins, and
  reduces each (cell, arm, mode) to per-seed values, n, mean, SAMPLE std (ddof = 1); `[timing-json]` lines from
  the persisted `ev_*.log` files are averaged over seeds and joined. Writes board.json + board.md.
- Verification: timer unit test on a fake adapter (first-call replan, refills every `take` steps, a flushed env,
  a dead env; replan counts equal the adapter's own decisions on all 12 calls, attribution sums exactly, the
  amortized block time equals the fake's 2 ms/env); collector parser check on the old board's logs reproduces the
  known lines exactly (pusht nm192 reactive 62/78/70 -> 70.0 +- 8.0, best-of-K 80.7 +- 4.2, gradient 86.7 +- 6.1,
  random-candidates 39.3 +- 5.0; TC rows parsed; 0 rows for fx_ without crashing); end-to-end CPU run of
  eval_gip.py on the real pusht harness (2-episode u8 slice -> 2-epoch checkpoint at --H_max 10 -> reactive and
  best-of-K evals with 2 envs, budget 50): reactive 50 calls / 10 replans, call 82 ms over 2 envs -> block 41.1 ms
  amortized, episode total 0.44 s, 10 replans per env; best-of-K (K 4, horizon 5 blocks) call 386 ms -> block
  193 ms, episode total 1.97 s; both `[timing-json]` lines and json files present. The first attempt caught a real
  scope bug in the dump (`ge` undefined in main; the guard kept the evals alive and printed `[timing] unavailable`)
  -- fixed before commit; the json's per-replan action count is named `adapter_num_actions` (25 for the planner,
  10 reactive) beside `model_num_actions_pred` (the training value) so the two are never confused.
  Both SIGReg toolhang jobs verified running (START heartbeats) by 17:20.
- ETA: toolhang 3.1 h train + ~2.7 h eval; pusht 5.5 h train + 17 min chained evals.

## Dataset scale — trajectories per training set (read from ep_len 2026-09-06; paper-relevant)
The trainer's --train_split 0.9 partitions DECISION POINTS (randperm of n_starts), NOT trajectories, so every
episode is trained on; the 10% val holdout is start-points. "Trajectories trained" = the full episode count:
  pusht    18,685 traj  2,336,736 steps (~125/traj)  pusht_expert_train.h5 46GB
  tworoom  10,000 traj    920,809 steps (~92/traj)   tworoom.h5 13GB
  reacher  10,000 traj  2,010,000 steps (201/traj)   reacher.h5 99GB (random-policy file, being replaced)
  cube     10,000 traj  2,010,000 steps (201/traj)   cube_single_expert.h5 102GB (ogbench single-expert)
Files on the a2f SSD /mnt/hdfs/bi_algo_a2f/minghao.fu/lewam/data/ (devbox-only; training reads the derived
byte_ad_audit preload_cache, eval reads wf8/train/<cell>.h5). [[cell-protocol-map]]

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
- **TOOLHANG ENCODER LADDER (2026-08-25, board full-traj): resnet18dp 89.0/91.3 ·
  ViT-small fp32 (ep-100 ckpt) {56,60,76} = 64.0 · ViT-tiny 0.0 {0,0,0} (diverged ep
  <20 on bf16; per-loss floor).** ViT-small on bf16 diverged at ep 29 (loss explosion
  then nan); fp32 trained clean — both fp32 reruns cleared their bf16 failure epochs.
  Encoder verdict across cells: ResNet ≥ ViT everywhere in JOINTFLOW (toolhang 89 vs
  64, pusht 74 vs 62); scratch-ViT won only inside unified-on-cube. resnet18dp stays
  jf's default.
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
- **STEER-MPC ON FLOW NULL ON PUSHT (2026-08-26, 74a4fb1): plan_mode=steer (prompt-MPC
  port: delta on z_goal into the sampler, fixed per-replan noise, inpaint scoring under
  the TRUE goal, delta=0 floor, pm_steps 20 / lr 0.02 / rho 0.3) = {56,74,68} → 66.0;
  pm_random equal-norm control = {62,76,70} → 69.3; reactive 67.2 / BoK-inpaint 70.7.**
  Gradient ≈ random ≈ reactive: the z-space search is floor-safe (no CEM-style
  collapse — on-manifold actions + true-goal scoring held) but harvests nothing;
  consistent with pusht being policy-limited. The unified-era steer gains (+2-5 on
  TC cells) do not transfer. All flow noise eval-seeded; ckpt gr2_a10s1_s42.
  CEILING CELLS (same protocol, one job): tworoom grad {98,98,100}=98.7 / random
  {100,98,100}=99.3 (band 98.7); pml grad {100,96,100}=98.7 / random 100.0 (band
  100.0). Verdict across all three GR cells: steer never breaks the policy and never
  beats it; grad direction ≤ random direction everywhere (within noise). Steer is a
  safe-but-null test-time knob on jf GR — same conclusion family as BoK selection.
- **tau2 CLOSED (2026-08-27): --tau_alpha 2 (tied Beta(2,1)) plan 63.3 / reactive 64.7;
  probes == noreg (angle .49/.52, vel .57/.05, block_pos .89); state field
  machine-straight (cos 1.0000). Clean-end tau bias does not touch the state branch.
  SCALE-MISMATCH DIAGNOSIS (owner Q "how is state loss low but variance high?"): the
  unit-Gaussian source (||x0||~20) vs the ~0.03-per-dim latent target (||z||~1) makes the
  flow loss a denoising score (v ~ -x0, x1 invisible at Var~1e-3/dim) and the sampled
  state = a residual of a few % of ||x0|| ~ 1.2 latent units = the constant "imagined
  change" the probes measured, on any cell/action. Actions are z-scored so their flow is
  well-posed — "flow helps actions, hurts dynamics" = unnormalized state target. FIX
  ARM znorm (--state_target_norm, 0cbe4ad, mf-c408d295): state flow in standardized latent
  coords (EMA stats as buffers, online target keeps gradient, de-normalized sampling).
  RESULT: RUNAWAY COLLAPSE. zstd 0.0084 at ep 1 (anchor 0.067), 0.00015 at ep 2, 0.00046
  at ep 10; train state loss 0.84 -> 0.08 then rising (sd clamp 1e-3 binds); val state
  loss 1e2 -> 1e5 (clamped sd blows up the normalized val target); action loss ALSO
  worse (0.81 vs anchor 0.56 at ep 10): the action branch conditions on z and a dead z
  hurts it. Mechanism: standardizing the online target makes the state loss scale-free
  in z, so nothing penalizes the encoder shrinking z, and the gradient into the encoder
  through (z-mu)/sd carries a 1/sd factor, so shrinking strengthens the pull: positive
  feedback, finished in two epochs. The raw-scale noreg arm survives only because its
  state loss is tiny (~0.01) and the collapse pressure fades as z shrinks; the z-score
  removed that damping. Same lesson as EMA-detach on toolhang, from the other side: an
  online state target needs SOME collapse control (EMA / SIGReg) once its scale is made
  to matter. Formal job cannot be killed from CLI: console stop needed.
  OPEN TEST of the scale hypothesis WITHOUT training: sig04 already has a well-scaled
  target (zstd 0.86 vs unit noise), so probe_jf_cost on sig04 (+ sigvit) decides whether
  a well-posed state flow makes imagination informative (proposed, not launched).**
- **split CLOSED (2026-08-27, mf-76b3f6ed, --split_tau --tau_cond summed --fp32): plan
  {60,72,72} = 68.0 / reactive {62,80,66} = 69.3 vs anchor 70.7 / 66.0 -- in band. Train
  curve identical to the anchor (act 0.363 at ep 50 vs 0.363); the wild val (1.4-8.3,
  floor flat at 1.43 for 37 epochs) is a hot-weights artefact shared with anchor s42
  (val 4.7 at ep 40 -> 0.35 at ep 50 when lr -> 0), not a split symptom. Probes ~ noreg:
  agent_pos .972, block_pos .929, angle .542 (sincos .712), vel .568. State field
  straight at n8 (cos 0.9999) but BENT at n64 (ends 0.63, straight 0.988) like a5s1 --
  second bent state field with zero SR consequence: field geometry is not the lever.
  TAU FAMILY CLOSED (tau2, split); splits2 staged but recommended NOT launched.**
- **REACHER RESOLVED (2026-08-27, mf-baeb6d77 + eval rerun mf-970aadbd): jf + SIGReg 0.04, fp32,
  otherwise the GC recipe (fs 5, a10s1) = planning {96,96,90} = 94.0 / reactive {92,80,88} =
  86.7 (3 eval seeds x 50; planning >= reactive on every seed). Same eval: unified lewam_gc
  98.8, baseline_wam 8-16, so the cell is real and jf is within seed noise of unified. The
  action loss sat at 1.513 -> 1.503 for 50 epochs: on torque cells the flow loss is the entropy
  floor of torques given images and says NOTHING about control; the sampled torques' direction
  is what matters. "Reacher fails" was inferred from that curve because no earlier reacher jf
  arm reached eval (two bf16 NaNs, fs3 noreg collapse). Eval gotcha: pods ship MuJoCo 3.12.0
  where dm_control 1.0.43 dies ('MjData' has no 'qM'); pin mujoco==3.10.0 (memory: merlin-ops).
  CONSEQUENCE: the GR deficit is pusht only (74 sig / 66 noreg vs unified 87). CORRECTION
  (owner, same day): SIGReg on toolhang IS measured post-tau-fix -- round-2 a10s2 (SIGReg +
  per_modality) 72.2 +/- 8.7 vs noreg 89.0, round-1 a10s1 (SIGReg, summed tau) 76.3; the
  round-2 record already concluded "the unlock is SIGReg removal, not the tau fix" (residual
  confound: a10s2 layout). So SIGReg costs ~13-17 on toolhang and is NOT the one recipe:
  noreg wins TC + saturates tworoom/pml, SIGReg is needed for reacher and best on pusht.
  CLOSED 2026-08-27 (mf-0fceb53b train + mf-cfaecd12 board shard eval): a10s1 layout + SIGReg
  0.04 + per_modality tau + fp32, 120 ep = {74, 82, 74} = 76.7 (3 eval seeds x 50, full-traj)
  vs noreg 89.0 (n=12). No layout confound left: SIGReg costs ~12 on toolhang. Owner called ~70.
  Train curve: act 0.254 / zstd 0.89 at ep 120 (noreg ~0.26 / 0.027) -- the policy loss does
  not see the 12-point gap, same lesson as the crossattn WM arms.**
- **WORLD-MODEL ACTION DISCRIMINATION (2026-08-27, scripts/probe_wm_discrim.py, owner Q "is JEPA
  itself the issue?"; pusht, 200 dataset anchors x 32 wrong blocks per type, goal at +10
  anchors, each model in its own latent; outputs ckpts/wm_discrim/). Two pairwise scores per
  wrong-action type: TRUTH = P(prediction under the expert's block is closer to the REAL next
  latent than under the wrong block); GOAL = P(expert block imagined closer to the goal). 0.5 =
  coin flip. Wrong blocks: zero, -expert, another state's expert block, same-episode other time,
  expert + {0.25,0.5,1,2}sigma noise (z-scored units), uniform over the action box.
  TRUTH: LeWM-repro (see LEWM CKPT NOTE) 1.00/1.00/1.00/0.99 | 0.86/0.95/0.99/1.00 | 1.00; unified
  roll2-nogoal 1.00/1.00/1.00/0.99 | 0.87/0.96/0.99/1.00 | 1.00; unified idm05 0.95/1.00/0.97/
  0.97 | 0.82/0.92/0.98/1.00 | 1.00; jf noreg anchor 0.53/0.54/0.53/0.56 | 0.52/0.55/0.49/0.53
  | 0.59 (pred err 1.23 vs real progress 0.023: the noise residual); jf sig04 0.57/0.67/0.62/
  0.62 | 0.49/0.56/0.53/0.66 | 0.90 (pred err 3.58 vs progress 0.82).
  GOAL: every model ~0.45-0.6 on perturbed/shuffled blocks (LeWM 0.27 on uniform: a random block
  is imagined CLOSER to the goal 73% of the time; idm05 0.79).
  READING: JEPA/MSE latent dynamics know what an action does almost perfectly; latent
  distance-to-goal is not a progress measure for any of them (why CEM/BoK never beat reactive:
  the planner's cost, not the dynamics). jointflow's flow state branch is at chance in the joint
  model; SIGReg scale helps only for gross wrong actions (uniform 0.90). The twin (clean-action
  state flow) is measured with the same test. Note: the earlier "CEM never beats reactive" was
  wrong -- goal-free roll2 CEM 93.2 vs reactive 89.2 on-path / 81.2 vs 75.2 off-path (one run).**
- **TWINFLOW (owner design 2026-08-27; lewam/models/twinflow.py, --model twinflow): the
  jointflow action flow untouched (a JointFlow with no state slot) + a SEPARATE state flow
  trunk (same CrossAttnBlock stack, same [a..,z,a..] layout, cross-attn to the frame history)
  whose action tokens are the CLEAN ground-truth chunk (no noise, no tau) -- a plain conditional
  dynamics p(z_{t+fs} | history, a); planning scores a candidate by running the state flow on it
  (what inpainting faked by clamping). Switches: --state_detach (state branch trains on
  stop-gradded history + target: no gradient path into the encoder, so it cannot collapse or
  reshape it; the policy's representation is shaped by the action loss alone) and, only under
  detach, --state_target_norm (safe: the 1/sd runaway needed the encoder path). Same trainer,
  eval adapters and probes (interface = jointflow's); the curvature probe is jointflow-only.
  Devbox smoke: policy branch bitwise == JointFlow(num_states=0); detach leaves zero grad on
  history/target; inpaint(sampled chunk, shared noise) == sample state.
  OWNER CORRECTION (same day): the detach idea repeats the PARKED crossattn experiment. Record
  (toolhang, resnet18dp, DP-style head, 2026-08-19, one ckpt each): BC-only crossattn_bc 90
  (88 per-episode budget; 79.3 +/- 5.2 pooled over 6 eval seeds) -> + separate MSE dynamics
  head, online target + SIGReg (crossattn_wm) = 36 -> EMA target + stop-grad (crossattn_wmema)
  = 64. Stop-grad/EMA recovered half the damage and stayed ~25 under BC; SIGReg on the online
  target did not protect the POLICY. That parked separate-head WMs and motivated jointflow
  (89 without SIGReg; SIGReg costs ~15). Expectation for the twin: a separate dynamics
  objective on the shared encoder hurts the policy; the twin + SIGReg is that experiment with
  a flow head. Detach arms WITHDRAWN (flag kept, off by default); single arm staged: twinsig
  (--model twinflow --w_reg 0.04 --fp32, pusht). Owner default = SIGReg for anti-collapse.
  toolhang SIGReg (jf-tc-toolhang-sig04, mf-0fceb53b) LAUNCHED on the owner's word; owner
  expects ~70.
  TWIN RESULTS (2026-08-27, per-token tau c411c54, legacy 11-slot state trunk; SIGReg 0.04,
  fp32, otherwise the cell's board recipe): TOOLHANG twin (mf-4b094dad train + mf-11cb74e7
  board shard eval) = {58, 56, 64} = 59.3 vs SIGReg joint 76.7 vs noreg joint 89.0. Train
  curves: twin act 0.245 / state 0.053 / zstd 0.90 at ep 120 vs SIGReg joint 0.254 / 0.056 /
  0.89 -- the policy loss again does not see a 17-point gap. Ladder: the more the dynamics
  objective bites (noreg joint -> SIGReg joint -> SIGReg twin), the worse the toolhang policy,
  matching crossattn_bc 90 -> crossattn_wm 36 in direction. pusht twin (mf-c9fe379b) pending:
  SR + state probe + the action-discrimination test on its clean-action state flow.**
- **TWIN pusht (mf-c9fe379b, per-token tau, SIGReg 0.04, fp32): planning {58,74,72} = 68.0 /
  reactive {60,72,64} = 65.3 (sig04 67.3 / 74.0; anchor 70.7 / 66.0); state probe agent_pos .967
  block_pos .863 angle .584 (sincos .734) vel .609 (anchor .973/.890/.545/.568; unified .985/
  .935/.711/.748). ONE-STEP ACTION DISCRIMINATION of its clean-action state flow (200 anchors x 32):
  truth pairwise zero .975 / neg .995 / other-state .989 / other-time .973 / +0.25s .64 / +0.5s .79
  / +1s .91 / +2s .98 / uniform 1.00; top-1 (closest among 33) other-state .77, other-time .67,
  +1s .57, +2s .79, uniform .99 (LeWM .93/.74/.80/.96/1.00). Expert prediction error 3.39 vs
  real progress 0.94. VERDICT: with CLEAN action conditioning the flow state branch learns real
  action-conditioned dynamics, slightly coarser than the JEPA/MSE heads on fine perturbations;
  the joint model's chance-level dynamics came from noisy-action conditioning (+ scale), not from
  the flow objective. But the good dynamics did NOT transfer to the policy: pusht SR unchanged
  (65 vs sig04 74), readouts barely moved, toolhang hurt (59.3). Representation benefit and
  dynamics quality are separate axes.**
- **twinlr (2026-08-28, unified's split lr: encoder 1e-4 / policy 3e-4 / dynamics 3e-4, otherwise
  twinsig2 + the trimmed state trunk; code ab90b4b): TOOLHANG {58, 60, 38} = 52.0 vs single-lr twin
  59.3 (train act 0.234 vs 0.245 -- lower loss, no SR; zstd slid 0.86 -> 0.63 under the slow
  encoder). The lr split is not the twin's missing ingredient on TC. PUSHT: planning {66,80,72}
  = 72.7 / reactive {56,76,70} = 67.3 (single-lr twin 68.0 / 65.3; sig04 67.3 / 74.0); readouts
  agent_pos .971 block_pos .883 angle .512 (sincos .428) vel .622 -- inside seed noise, no
  readout gain. Split lr closed as a non-lever for the twin on both cells.**
- **ROLLOUT-TO-GOAL DISCRIMINATION (2026-08-28, probe_wm_discrim --horizon_blocks 5 = the GR
  eval's 25-env-step goal; 200 anchors x 16 wrong SEQUENCES; the expert's sequence rolled out
  through the model vs a wrong sequence: which ends closer to the goal = the real endpoint).
  Expert-first rate vs other-state / other-time / +0.5s / +1s / +2s / uniform: LeWM 1.00/.99/.94/
  .98/1.00/1.00; unified roll2 1.00/.99/.97/.99/1.00/1.00; idm05 .99/.97/.92/.97/.99/1.00;
  twinflow .985/.96/.78/.90/.97/.99. Endpoint error of the expert rollout vs start distance: LeWM
  2.1/14.8, unified 2.1/17.3, twin 9.8/22.3. So over the planning horizon latent goal-distance DOES
  rank the expert trajectory first for every model with real dynamics (the one-step "goal" test
  was an artifact, withdrawn); the twin drifts more but discriminates.
  Completed 2026-08-28: unified flow-head (pusht_uni_flowH5_full) one-step truth .99/.98/.83/
  .93/.98/.99/1.00, rollout expert-first 1.00/.98/.92/.97/.99/1.00 -- the flow policy head leaves
  the MSE dynamics intact. jointflow-noreg rollout: expert-first .95/.90/.69/.80/.91/.99 BUT
  hair-thin: endpoint error 0.56 vs start distance 0.53 (the rollout barely leaves z_t) and gaps
  of 0.01-0.5x that error; jointflow-sigreg .83/.79/.56/.64/.76/.93 with gaps 0.01-0.85x its 8.9
  error. Gap/error for the wrong-sequence types (how far a wrong plan lands vs the expert plan's
  own error): LeWM 2.9-6.6x, unified 2.1-8.7x, flow-head unified 0.6-3.5x, twin 0.3-2.1x, jf
  sigreg 0.01-0.85x, jf noreg 0.01-0.5x. Pairwise accuracy alone overstates the joint models:
  their preference is correctly signed but negligible in magnitude, which is what a planner
  scoring near-expert candidates actually meets (BoK score std 0.003).**
- **MOTFLOW (owner design 2026-08-28; lewam/models/motflow.py, --model motflow, commit see git):
  Mixture-of-Transformers joint model: separate state/action streams (own QKV/out/FFN/AdaLN), ONE
  global attention under a fixed mask. Tokens: z_hist, z* (S noisy next states or mse queries),
  z_g (goal, optional) | a (clean a_1..a_{S*fs}), a* (A noisy). Map: z_hist->z_hist; z*_q ->
  z_hist, z*_{<=q}, a_{<=q*fs}; z_g -> z_hist, z_g; a_j -> z_hist, a_{<=j}; a*_j -> z_hist, z_g,
  a*_{<=j}. tau_a/h on a* only, tau_s on z* only; goal enters the policy as a token (no readout
  injection). --mot_state_head flow|mse. Two-phase sampler (a* Euler, then z* under the sampled
  block); inpaint = phase 2. What it adds over twinflow: the dynamics loss reaches the ACTION
  stream's own weights (through the clean tokens) and the shared history processing, not only
  the encoder. Smoke: mask == table; a* bitwise blind to clean actions/z*; z* blind to a*/goal/h;
  state loss reaches the action stream; inpaint(sampled) == sample state; mse head; lr groups.**
- **MOT GRID (2026-08-28, single lr, fp32, code 697b1ec; toolhang board shard evals, 3 seeds x 50):
  TOOLHANG mot-flow-noreg {80, 92, 86} = 86.0 (jointflow-noreg 89.0; twinflow 59.3) -- the MoT
  carries a clean-action state flow WITHOUT the twin's policy loss on TC; train act 0.266 / zstd
  0.022 at ep 120 (the noreg signature). mot-flow-sig {72, 80, 60} = 70.7 (SIGReg joint 76.7; twin
  59.3): SIGReg still costs ~15 on toolhang.
  PUSHT (planning BoK-32 inpaint / reactive, 3 seeds x 50): mot-flow-noreg {62,78,54} = 64.7 /
  {66,82,72} = 73.3 (jointflow-noreg 70.7 / 66.0), readouts .975/.909/.540/.567 (= noreg's);
  mot-flow-sig {62,74,78} = 71.3 / {68,80,70} = 72.7 (sig04 67.3 / 74.0), readouts .842/.566/
  .463/.556 (SIGReg degrades linear readouts, SR unaffected). MoT-noreg = best noreg pusht so far
  (+7 reactive over jointflow), still 15 under unified.
  TOOLHANG MSE-HEAD ARMS: mot-mse-noreg {80,84,78} = 80.7; mot-mse-sig {60,42,62} = 54.7 -- despite
  the lowest toolhang policy losses on record (act 0.200 / 0.189 at ep 120 vs noreg joint 0.26):
  open-loop loss and closed-loop SR disagree again. TOOLHANG GRID: flow-noreg 86.0 > mse-noreg 80.7
  > flow-sig 70.7 > mse-sig 54.7 (jointflow 89.0 / 76.7; twin 59.3). SIGReg costs 15-26 in every
  architecture; MSE state head costs 5-16 vs flow.
  ONE-STEP ACTION DISCRIMINATION of the pusht MoT state flows (200 anchors x 32): mot-flow-noreg
  at CHANCE everywhere (truth .47-.58; pred err 1.31 vs progress 0.035 = the noise residual) -- with
  the 0.03-scale latent the state flow is a denoiser no matter how cleanly it is conditioned;
  mot-flow-sig truth .97/1.00/.98/.96 gross, .58/.77/.89/.98 at 0.25/0.5/1/2 sigma, 1.00 uniform;
  top-1 other-state .71 (twin .77, LeWM .93). So real dynamics need the well-scaled (SIGReg)
  latent, and SIGReg is what costs the TC policy: the trade-off is now measured consistently across
  joint, twin and MoT. Open knob: w_reg was never tuned (0.04; raw SIGReg 4-20 x 0.04 ~ act loss).
  PUSHT MSE-HEAD ARMS: mot-mse-noreg planning {58,78,72} = 69.3 / reactive {66,76,64} = 68.7,
  readouts agent_pos .973 block_pos .980 angle .734 (sincos .882) vel .586 -- block readouts ABOVE
  unified's (.935/.711); mot-mse-sig 71.3 / {66,80,70} = 72.0, readouts .973/.945/.729/.649 (=
  unified's). Neither collapsed (zstd 0.14 / 0.99). PUSHT GRID (plan / react): flow-noreg 64.7/73.3,
  flow-sig 71.3/72.7, mse-noreg 69.3/68.7, mse-sig 71.3/72.0 -- one band, 69-73, vs unified 87-89.
  READOUT HYPOTHESIS REFUTED: two MoT arms carry unified-level (or better) linear block/angle
  information and still score in the jointflow band, so the pusht gap is not the encoder's linear
  content. Remaining differences vs unified (roll2-nogoal 89.2): 5-frame aggregated context vs
  2-frame history; one-block MSE point policy vs 10-action flow policy (unified's own flow head
  reached 86.8, so the head alone is not it); ViT-tiny z192 vs resnet18dp (jf sigvit 62 < sig04 74
  says the ViT is not it); budget is NOT it (roll2-nogoal's 89 was 50 ep; the 200-ep idm05 run
  overtrained, owner). 
  MSE-HEAD DYNAMICS (one-step discrimination, 200 anchors x 32): mot-mse-noreg truth .995/1.00/
  .998/.988 gross, .81/.92/.98/1.00 at 0.25/0.5/1/2 sigma, uniform 1.00; top-1 other-state .95,
  other-time .70, +1s .69, +2s .94; gap/err 3.5-8.1x; expert pred err 0.265 vs progress 0.127.
  mot-mse-sig truth 1.00 gross, .885/.96/.99/1.00 fine; top-1 .99/.87/.85/.95; gap/err 7-11x --
  the best discrimination of any model measured (LeWM top-1 .93/.74/.80/.96). So the MSE state
  head in the MoT gives JEPA-level dynamics, and mot-mse-NOREG gets it without SIGReg and without
  collapse (zstd 0.14). MoT-mse-noreg = the first variant with real dynamics AND a TC policy in
  reach of the board (toolhang 80.7 vs 89.0; pusht 68.7 vs 73 flow / 87 unified). BoK planning
  with these dynamics still does not beat reactive on pusht (69.3 vs 68.7; 71.3 vs 72.0), as with
  unified: dynamics quality is not what limits BoK at this horizon/candidate set.
  ROLLOUT-TO-GOAL (H=5 blocks): mot-mse-noreg expert-first 1.00/.98/.93/.98/1.00/1.00 (other-state/
  other-time/+0.5s/+1s/+2s/uniform), gap/err 0.9-8x, endpoint error 0.48 vs start 2.96 (16% drift;
  LeWM 14%); mot-mse-sig 1.00/.99/.97/.99/1.00/1.00, gap/err 2.5-7.3x, 12% drift. Both at the
  LeWM/unified level over the planning horizon.**
  LEWM READOUT PROBE (2026-08-29, LeWM-repro encoder+projector, same ridge n=4000/1000):
  agent_pos .931 block_pos .962 angle .755 (sincos .916) vel .513 -- between the MoT-MSE arms and
  unified (mot-mse-noreg .973/.980/.734/.586; unified .985/.935/.711/.748) at SR 84.0 (measured,
  below) vs 69 (MoT): a model 15 SR better carries no more linear block/angle information.
  LEWM CKPT NOTE (owner caught 2026-08-29): every "LeWM" probe above (discrimination, rollout,
  readout) used ckpts/hf_official/pusht_lewm_base = OUR from-scratch LeWM repro
  (pusht_ours_lewm_weights.pt, L40S June; EXPERIMENTS.md "same-box CEM run-dir"), NOT the authors'
  release. Weights differ from code/lewm_main_eval/hf_release_native/pusht (the HF release,
  identical architecture) in all 303 tensors. MEASURED pusht SR on OUR eval (CEM, 3 seeds x 50):
  authors' release 94/90/82 = 88.7 (lewm_full_repro/A2, 2026-07-30); our repro 84/88/80 = 84.0
  (lewm_repro, 2026-07-03); our epoch-100 retrains 74-88 per seed, ~81 (B_ours/B_ours2). The "93"
  quoted on 2026-08-29 was never measured here (the paper's CEM number in EXPERIMENTS.md is 82.5);
  use 88.7 (release) / 84.0 (repro) for LeWM on pusht.
  AUTHORS' RELEASE PROBED (2026-08-29, hf_release_native/pusht, same settings; archived under
  ckpts/wm_discrim/*lewm_release*): readout R2 agent_pos .959 block_pos .974 angle .804 (sincos
  .915) vel .594 (repro .931/.962/.755/.513); one-step truth pairwise other-state .997 other-time
  .986 pert .25/.5/1/2 = .80/.91/.98/1.00 uniform 1.00, gap/err 3.7-8.2x, top-1 other-state .93
  other-time .71 (repro .93/.74); rollout-to-goal expert-first .994/.979/.918/.974/.993/.996
  (repro 1.00/.99/.94/.98/1.00/1.00), gap/err 1.6-5.1x, endpoint error 2.95 vs start 16.2 (18%
  drift; repro 14%). Release and repro are alike on every probe: the LeWM rows above stand with
  the corrected label, and the readout/discrimination conclusions do not depend on which one.
- **ROLLOUT PLANNING ON MoT (2026-08-29, job mf-de5752ce, code 503ce89; pusht, 3 eval seeds x 50,
  exec5, K=32). The owner's planner (`plan_rollout=5`): from the real 2-frame history the policy
  samples a chunk, its first block goes through the MSE state head, the imagined latent slides into
  the history, x5 to the goal time (25 env steps), cost = final imagined latent vs goal latent, execute
  the winner's first block, replan. mot-mse-noreg (mot_nm) BoK-roll5 {84,86,90} = 86.7 vs BoK-inpaint
  {60,78,72} = 70.0 (re-run) vs reactive 68.7; mot-mse-sig (mot_sm) BoK-roll5 {84,84,86} = 84.7 vs
  BoK-inpaint {62,84,70} = 72.0 vs reactive 72.0. +15-17 SR from the dynamics on a 69-72 reactive
  policy = the first jointflow-family planning result in unified's band (87-89), and the first
  time planning beats reactive by more than noise. INPAINT RETIRED FOR MoT (15659d0): the old
  planner scored each 10-action chunk by ONE dynamics step (5 env steps ahead vs a goal 25 steps
  away); MoT now plans by the rollout by construction (plan_rollout defaults to the goal horizon,
  exec one block; plan_score=inpaint / cem / steer refused). Every earlier "planning" number for
  jointflow/twin/MoT was the one-block verifier.
  ORACLE ROUND 1 (same job; expert chunk + K-1 wrong chunks, the dynamics picks): one-block inpaint
  scoring picks the expert 14-20% (uniform) / 5-8% (other demos) on mot_nm -> SR 28.7 / 36.7 (mot_sm
  8-11% / 4-6% -> 26.7 / 35.3); 2-block candidates + goal-conditioned policy continuation in
  imagination 9-20% -> SR 12.0 / 12.0 (mot_sm 14.7 / 12.7): the policy imagines a recovery after
  any prefix, so every candidate ends near the goal. Neither is a verifier; round 2 (mf-c5ba118c,
  staged) scores full 25-action candidate sequences through the dynamics alone, with the authors'
  LeWM release under the same test + its own CEM SR, and BoK-roll5 on the flow-head arms.**
- **ROUND 2 (2026-08-29, job mf-c5ba118c, code 15659d0; pusht, 3 eval seeds x 50, K=32).
  FLOW-HEAD ARMS UNDER THE ROLLOUT PLANNER: mot-flow-noreg (mot_nf) BoK-roll5 {66,78,74} = 72.7
  (reactive 73.3, inpaint 64.7); mot-flow-sig (mot_sf) {74,84,72} = 76.7 (reactive 72.7, inpaint
  71.3). The planning gain is the MSE head's: +13-17 for mse-noreg/mse-sig vs -1/+4 for the flow
  heads -- the flow state head (a denoiser at the noreg scale, coarser under SIGReg) does not carry
  the rollout. RECIPE: MoT + MSE state head + rollout planning; SIGReg optional (84.7 vs 86.7).
  LEWM RELEASE, OUR EVAL, FRESH: its own protocol (CEM 300x30, 5-block plan executed whole)
  {94,88,82} = 88.0 (2026-07-30 A2: 94/90/82 = 88.7 -- reproduced); replanning every block with
  the same fixed 5-block horizon (receding 1) {26,30,34} = 30.0 -- LeWM's plans are only good when
  executed whole (fixed-horizon replanning aims at the goal 25 steps out at every replan; the
  goal-conditioned MoT policy is time-aware through h_norm and does not suffer this).
  FULL-HORIZON ORACLE, MoT-MSE (expert's next 25 raw actions vs 31 wrong 25-action sequences, all
  5 blocks imagined by the dynamics alone, cost at the goal time, execute the first block):
  mot_nm uniform-box {32,38,26} = 32.0 with the expert picked 29-50% of replans, other-demo
  {18,18,10} = 15.3 (19-23%); mot_sm uniform 22.7 (26-34%), other-demo 14.7 (20-25%); chance 3%.
  Consistent with the offline rollout probe (expert-first PAIRWISE .93-1.00): 31 independent
  wrong candidates at ~.95 pairwise give ~.95^31 = 20% top-1. So the MoT dynamics is a real but
  imperfect verifier over arbitrary sequences, and the planner works because the policy proposes
  on-manifold candidates (BoK-roll5 86.7) -- cold search over random sequences would not (the
  owner's "you can't cold-CEM our dynamics"). LeWM's own oracle was not run (owner: MoT-MSE only).**
- **ORACLE DIAGNOSTIC (2026-08-29, devbox CPU, mot_nm, uniform rivals, K=32, eval seed 42; pick
  rate by replan index, "*" = on-demo, i.e. every earlier pick was the expert so candidate 0 IS
  the expert for the current state): r0* 47/50 = 94% -> r1* 36/47 = 77% -> r2* 26/36 = 72% ->
  r3* 12/25 = 48% -> r4* 1/9 = 11%; off-demo replans 2-20%; SR 30.0 (H100 run: 32.0). So the
  aggregate 29-50% mixed on- and off-demo replans (the retracted ".95^31" reading was wrong):
  on the demo state the MoT dynamics picks the expert 94% among 32 at the first replan,
  consistent with the offline pairwise 1.00. The decay over replans is the FIXED-HORIZON
  artifact: the candidate is always 25 actions and the cost is taken 25 imagined steps out
  while the goal is 20/15/10/5 steps away, so the demo's continuation PAST the goal leads
  away from it (LeWM's exec5 collapse to 30.0 is the same artifact). FIX (code, next
  commit): every rollout scorer takes the cost at the env's GOAL TIME -- the imagined block
  at H_i = clamp(round(steps_left), 1, plan_rollout) -- instead of after the last block
  (plan_goal_time, default on). BoK-roll5's 86.7 was measured with the last-block cost; the
  goal-time cost is re-measured in the planning job.**
- **ORACLE, CORRECTED (2026-08-29, devbox CPU, code c132fbc, mot_nm, expert's remaining demo
  actions vs 31 uniform-box sequences, K=32, cost at the goal time, exec5, eval seed 42 x 50):
  SR 96.0; expert picked 47/50, 47/47, 47/47, 46/46, 41/42 at on-demo replans 1-5 (98%/replan),
  235/251 = 93.6% overall (chance 3%). The owner's sanity check ("expert vs K-1 random -> dynamics
  for planning -> SR recovers to 95-100") PASSES: the MoT-MSE-noreg dynamics, used as the verifier
  with the correct receding-horizon cost, recovers the demo's success. The 30-32 measured before
  was the fixed-horizon scoring artifact, not the dynamics.**
  Same, rivals = 31 OTHER DEMOS' 25-action sequences (on-manifold, wrong state): SR 98.0; expert
  picked 48/50, 47/48, 44/47, 43/43, 36/39 at on-demo replans 1-5 (96%/replan), 224/242 = 92.6%
  overall. The dynamics separates the right expert sequence from other experts' sequences too.
- **GRADIENT PLANNING ON MoT-MSE (2026-08-29, job mf-d3633454, code c132fbc; pusht, 3 eval seeds
  x 50, K=32, exec5, cost at the goal time). plan_mode=grad = action-space gradient MPC: warm start
  = the best of 32 policy rollouts (the roll planner), then Adam (50 steps, lr 0.05, grad-norm clip
  10) on the z-scored 5-block plan (25 actions x 2) against the terminal latent cost differentiated
  through the MSE state head's autoregressive rollout (MoTFlow.imagine_step); |a| <= 3 sigma; warm
  start = floor; the best iterate by model cost executes its first block, replan.
  mot-mse-noreg (mot_nm): roll5 with the goal-time cost {80,84,86} = 83.3 (last-block cost 86.7);
  grad {92,88,86} = 88.7; grad + trust region 1.0*||U-U_warm||^2 {78,86,86} = 83.3.
  mot-mse-sig (mot_sm):  roll5 goal-time {84,86,86} = 85.3 (last-block 84.7); grad {92,100,94} =
  95.3; grad + trust region {90,88,86} = 88.0.
  Reactive 68.7 / 72.0 -> roll5 ~85 -> gradient 88.7 / 95.3: the SIGReg-MSE arm with the gradient
  planner is the best pusht number of the whole campaign, above LeWM (88.0, its own protocol) and
  unified (87-89), from a checkpoint whose reactive policy is 72. The refinement moved the plan by
  ||U-U0|| ~ 2.1 (noreg) / 1.2 (sig) z-units over 50 dims and improved the model cost on ~100% of
  replans; SR rose with it, so the dynamics is NOT being exploited at this step size -- the trust
  region (move ~0.2) gives up most of the gain. SIGReg's well-scaled latent (cost 0.18 -> 0.06)
  gives a better-conditioned landscape than the 0.03-scale noreg latent (0.0011 -> 0.0005).
  The goal-time cost itself was a wash for roll5 (-3.3 / +0.7, inside noise). CAVEAT: one training
  seed per arm; needs >= 3 training seeds before it is a headline. SteerMPC OOMed here (50 envs x
  32 draws x 5 blocks x 8 flow steps of autograd); re-run with env chunking = job mf-1f04f3f7.**
- **SteerMPC ON MoT-MSE (2026-08-29, job mf-1f04f3f7, code e7b5bfa; pusht, 3 eval seeds x 50,
  K=32, exec5, cost at the goal time). plan_mode=steer on the rollout: Adam (20 steps, lr 0.02) on a
  z_dim bias delta added to the goal latent fed to the POLICY, through the frozen flow sampler and
  the dynamics rollout; 32 noise draws per env share one delta (envs optimized 8 at a time);
  ||delta|| <= 0.3 ||z_goal||; cost vs the TRUE goal (the MoT state stream never sees the goal);
  iterate 0 = the roll planner (best of 32) stays in the set; the best (candidate, iterate) executes
  its first block. mot-mse-sig (mot_sm): {94,94,94} = 94.0 (roll5 85.3, action-space gradient 95.3);
  delta settled at 0.19 of the goal norm, model cost 0.14 -> 0.06, improved on 97% of replans.
  mot-mse-noreg (mot_nm): {84,90,80} = 84.7 = roll5's band; delta pinned at the 0.3 cap on every
  replan, cost 0.0012 -> 0.0009. So with the well-scaled SIGReg latent, steering the policy's goal
  input recovers nearly all of the gradient planner's gain while every executed action remains a
  policy sample; with the 0.03-scale noreg latent the cost surface is too flat for either search.
  FLOW-HEAD ARMS UNDER THE FOUR PLANNERS (job mf-75718ac0, code 9c594fe): mot-flow-noreg (mot_nf)
  roll5 goal-time {68,80,68} = 72.0 (last-block 72.7), gradient {70,74,66} = 70.0 with the
  refinement improving only 11-14 of ~300 replans (cost 0.0003 -> 0.0003: the noreg flow state
  head is a denoiser, no gradient signal), gradient+TR {72,82,74} = 76.0. mot-flow-sig (mot_sf):
  roll5 goal-time {78,84,78} = 80.0 (last-block 76.7), gradient {84,84,76} = 81.3 (improved ~50%
  of replans, cost 0.12 -> 0.08, ||U-U0|| 1.6), gradient+TR {72,82,76} = 76.7. SteerMPC OOMed on
  both flow arms even at 8 envs/chunk (the state flow doubles the passes in the graph; fixed:
  chunk/4 on flow heads, not re-run -- owner hold). PLANNING TABLE (pusht, 3x50, K=32, exec5):
    planner              mse-noreg  mse-sig  flow-noreg  flow-sig
    reactive               68.7      72.0      73.3        72.7
    roll5 (goal-time)      83.3      85.3      72.0        80.0
    gradient               88.7      95.3      70.0        81.3
    gradient + trust reg   83.3      88.0      76.0        76.7
    SteerMPC               84.7      94.0      OOM         OOM
  Only the MSE state head plans; SIGReg's unit-scale latent is what makes the gradient searches
  work (SteerMPC 94.0 on-manifold, gradient 95.3); the flow heads gain at most a few points.**
- **HISTORY LENGTH 5 (2026-08-29, job mf-b7f162e8, mot_nm recipe with --policy_history_len 5,
  i.e. 5 frames at fs spacing instead of 2; pusht, seed 42, 50 ep): reactive {26,56,42} = 41.3 vs
  68.7 with 2 frames; one-block planner (old scorer) {42,54,56} = 50.7 vs 70.0; readouts agent_pos .974 block_pos .977 angle .737 (sincos .872) vel .574 = the 2-frame arm's (.973/.980/.734/.586): same encoder content, worse policy.
  Training: val action loss rose to 1.34 at ep 30 while train fell (overfitting to the longer
  context), recovered to 0.30 by ep 50 under the lr decay; zstd 0.114, no collapse. More history
  hurts the flow policy on pusht by 20-27 on every seed -- the owner's "least likely" hypothesis
  (history aggregation explains unified's edge) is closed in the wrong direction for this recipe.
  roll5 / gradient evals of this ckpt staged (mf-d63642ca), not run (owner hold).**
- **WORLD-MODEL VISUALIZATION (2026-08-29, job mf-9f4139ed, experimental/viz_wm_dynamics.py
  f4c7068, adapted from the owner's viz_train_decoder.py; pusht, all four MoT arms, k=10 anchors,
  8 autoregressive imagined blocks, episodes 3/77/1234; per arm a z->pixel decoder trained 20
  epochs on 20k frames WITH the pred-decode term. Outputs ckpts/wm_viz/<arm>/: PNG grids
  (GT / decode(z_GT) / decode(z_imagined)), mp4s, decoders, metrics json).
  Latent MSE per imagined step vs the copy-last-latent baseline (ep77 // ep1234):
    mse-sig  (mot_sm): 0.005-0.11 vs 0.3-2.4   // 0.005-0.02 vs 0.3-2.4  -> 20-200x better than freeze
    mse-noreg(mot_nm): 0.017-0.068 vs 0.013-0.061 // ~2x better           -> barely beats freeze (tiny-scale latent)
    flow-sig (mot_sf): 0.03-0.55 vs 0.12-2.5                              -> 4-10x better, error compounds with depth
    flow-nore(mot_nf): 0.36-0.62 vs 0.002-0.02                            -> 30-200x WORSE than freeze
  Decoder recon PSNR: mot_sm 32.2-32.5 dB > mot_sf 30.9-31.1 > mot_nm 26.7-26.9 > mot_nf 25.1
  (SIGReg latents carry far more decodable content). Visually: mot_sm's imagined row tracks the
  scene through all 8 blocks (slight late blur); mot_nf's decode(z_GT) is already fog around the
  goal T and its imagined row fades to nothing. In pixels: SIGReg-MSE is a real world model,
  noreg-MSE's advantage over freezing depends on the episode, flow heads don't imagine.**
  TOOLHANG (2026-08-30, job mf-be4fe995, same protocol on tc_toolhang_mot_* s42; outputs
  ckpts/wm_viz/tc_<arm>/). Latent MSE per imagined step vs copy-last (ep77 // ep1234):
    mse-noreg: 0.0005-0.006 vs 0.005-0.058  -> ~10x better than freeze (unlike pusht, where it
               was ~freeze: toolhang's latent moves enough for the tiny scale to matter)
    mse-sig:   0.006-0.06  vs 0.45-2.15     -> 20-80x better
    flow-nore: 0.0004-0.021 vs 0.0001-0.0017 -> ~4-40x WORSE than freeze (its latents barely move)
    flow-sig:  0.02-0.47   vs 0.19-3.1      -> ~4-10x better, per-step spikes
  recon PSNR: mse arms 23.4-24.8 dB > flow arms 19.6-22.1 (the toolhang scene is much harder to
  decode than pusht; late-episode configurations blur in BOTH the recon and imagined rows =
  decoder limit, not dynamics). Visually: mse-sig's imagined row matches its recon row through
  all 8 blocks; flow-noreg's decoded GT is already a smeared robot and its imagination decays
  (per-step cos down to .43). Same ordering as pusht, with mse-noreg's advantage now clear.**
- **SUBGOAL PLANNING ON TOOLHANG (2026-08-30, job mf-ac781163, code 6d71d99; goal-blind TC
  checkpoints, cost target = the encoded demo frame 25 raw steps (5 blocks) ahead of the env's
  current step, replanned every block; 20 episodes x 1 seed, proof of concept. The model consumes
  no goal input; the subgoal enters only the planner's cost).
  mse-sig (tc_toolhang_mot_sm, reactive 54.7): SELECTION (best of 32 BC-policy rollouts through
  the dynamics vs the subgoal) = 80.0 -- +25, at mse-noreg's reactive 80.7 and near flow-noreg's
  86. The SIGReg reactive penalty is largely RECOVERABLE at plan time: the policy's latent is the
  bottleneck, and the dynamics + a subgoal supply what it lacks.
  Free GRADIENT refinement: mse-sig 10.0, mse-noreg 0.0 -- the optimizer moved plans ||U-U0|| ~8.9
  z-units (pusht: 1.2-2.2), "improved" the model cost 0.025 -> 0.004 on 100% of replans, and real
  success collapsed: on the contact-rich 7-dim cell the dynamics is confidently wrong far
  off-manifold and the unconstrained gradient finds exactly that region (on pusht the same
  planner gave 95.3 and the trust region only hurt; here a trust region / fewer steps is the
  missing guard -- untested). Consistent with the oracle picture: the model is a reliable
  verifier NEAR the manifold, exploitable far from it.**
  H1 (--state_mse, regression state slot inside the joint trunk) was implemented (e16c67f) and
  REMOVED the same day at the owner's request (800ddd3): a regression target under actions at
  random noise levels is not a coherent dynamics model.**
- **SNAPSHOT REACTIVE SR CURVES (2026-08-31, jobs mf-d19cf013/mf-cf755c52/mf-969cf385 =
  3 chunks of 165 min after the single job mf-0cbdb973 hit the 3h low-util kill at ~3h45m;
  entry jf_snapev2_6fb9727.sh, 20 episodes x eval seed 42 per point; snap_epN =
  jointflow_latest at epoch N, "final" = the best-val checkpoint, the board-eval artifact).
  All 32 points, toolhang probe arms s0 (epoch: nm / sm / nf / sf):
    ep15  65 / 10 / 10 /  5      ep75  60 / 75 / 75 / 60
    ep30  10 / 40 / 55 / 30      ep90  70 / 65 / 90 / 75
    ep45  35 / 50 / 55 / 45      ep105 65 / 50 / 70 / 80
    ep60  85 / 35 / 95 / 70      final 90 / 55 / 100 / 75
  Read with the two probes: mse-noreg starts FASTEST (65 at ep15, the cooperative phase --
  fdp gcos(P,Dtar)=+0.76, conflict 0.04), craters to 10 at ep30 = its one anti-aligned
  epoch AND the grad-probe's target-channel consolidation window, then swings 85/60/70/65
  while conflict mass climbs; best-val selection dodges the swings (final 90). flow-noreg
  is the calmest riser and finishes 20/20 = 100 (the zero-interaction arm; its final REACT
  hb line was lost to a concurrent HDFS append -- recovered from ev_snapev_mot_nf_final.log).
  mse-sig never exceeds 75 and sags to 50-55 late exactly while carrying the largest
  late conflict (3.1/1.2/1.2/2.5); flow-sig rises monotonically to 75-80 despite its 8
  crammed dims. n=20 x 1 seed per point: +-~10 binomial SE at p~0.5-0.7 -- shape claims
  only, no single-point claims.**
- **BOTTLENECK + PARAM-ABLATION RESULTS (2026-09-03, seed 42, evals on c299ee7).**
  BOTTLENECK (proj-sig + policy view + rank-32 P=UV) vs full-width pw_zp:
    toolhang reactive 73.3 {78,66,76} (pw_zp 76.7, noreg 80.7) / subgoal 70.0 (pw_zp 60.0) ·
    pusht reactive 75.3 {66,86,74} / roll 86.7 / grad 90.7 {90,90,92} (pw_zp 73.3/83.3/95.3).
  VERDICT: the bottleneck did NOT meet the owner's bar (keep ~95 pusht planning + close
  toolhang to -4). It TRADES: toolhang reactive -3.4 / subgoal +10; pusht reactive +2 / grad
  -4.6 (lost the 95.3 peak). A narrower policy view does not recover toolhang reactive; the
  hypothesis is not supported by the rank-32 arm (1 seed). pusht reactive 75.3 is the best of
  any pusht arm.
  PARAM ABLATION (d192/depth4/4heads, 17.3M total vs 55.0M) mse-noreg:
    toolhang reactive 82.0 {90,78,78} (55M 80.7) / subgoal 75.0 (55M 80.0) ·
    pusht reactive 72.7 {72,78,68} / roll 83.3 / grad 85.3 (55M noreg 68.7/86.7/88.7).
    The 3x-smaller model is EQUAL on reactive (toolhang +1.3, pusht +4) and ~3-5 lower on
    planning/selection; the ~0.11 toolhang loss gap did not cost success. Param count is
    nearly free -- a clean paper result.
  OPS SLIP (mine): reused the 911676c eval entry for c299ee7-trained bottleneck ckpts ->
    AssertionError missing policy_proj.weight / unexpected policy_proj.0.weight (old build_model
    makes single-Linear P, ckpt has rank-r Sequential). Violated the standing "eval tarball
    >= training commit" invariant. Fixed: jf_pvev_c299ee7.sh; all bottleneck/param evals pin
    c299ee7. Also: goal-state montage refutes "toolhang goals are wallpaper" -- goals vary
    MORE than starts (L1 17.4 vs 11.6); success predicate uses object xyz only (quats sliced
    off), so goal variation (orientation/pose) is orthogonal to reward = causal confusion,
    unlike cube (goal=position=reward). Failure videos at ckpts/jointflow_tc/toolhang_gc_fail_videos.
    Reacher bottleneck did NOT collapse (zstd ~0.20 through training vs mse-noreg 1e-4).
  DRAWER mse+noreg reactive (tc_drawer_mnm_s42, 120 ep, TC eval jf_tc_ev_c299ee7): 65.3
  {62,64,70} vs the jointflow-noreg drawer baseline 68.0 -- reproduced within noise.
  TRANSPORT mse+noreg reactive (tc_transport_mnm_s42, 120 ep): 86.7 {86,88,86} vs baseline
  86.7 -- reproduced exactly. (Eval job queued ~4 h on research with 18-42 H100 showing free;
  capacity was not the block.)
  CUBE mse+noreg (tc_cube_gcf_mnm_s42, goal-cond, 75 ep): trained 4 epochs healthy (val act
  0.585, zstd 0.26) at ~2400 s/epoch = 4.2x the jointflow cube's 574 s (transport/drawer show
  motflow at a uniform 1.3x), then killed at 3h17m with no error = the 3h low-util kill on a
  fuse-latency-bound loader (302 GB cache cannot fit the 120 GB pod; 0.29 s/step = 64 random
  fuse reads at ~7 ms with 6 workers, while 8 other jobs hammered the same fuse). Resume chain
  is built into the gcf2 entry (seeds jointflow_full.pt, --resume); ep-4 full ckpt is on HDFS.
  Proposed relaunch (not launched): --num_workers 12 + memory 200000, resume at ep 5.
  REACHER RESULT (bottleneck recipe): reactive 26.0 {26,26,26}; planning 99.3 roll
  {100,98,100} / 99.3 grad {100,98,100} vs the sigreg baseline 86.7/94.0. mse-noreg could not
  train here (collapsed), so SIGReg in the recipe is what makes reacher trainable.
  CORRECTION (2026-09-04, owner flagged the 26 as suspect; devbox re-eval, ~22 min/seed on
  CPU, same eval code): the reactive 26 is NOT "a weak policy rescued by the verifier" -- the
  policy is at the RANDOM-ACTION FLOOR. Fresh seed 7, same 50 configs: sig baseline 88.0
  (matches the pod 92/80/88, so the local eval reproduces the cluster); bottleneck 32.0;
  bottleneck with every executed action replaced by N(0,1) noise (+gip_eval.dyn_random_p=1.0)
  32.0 -- and the SAME 16 episodes succeed in both (identical index sets): those are the
  configs that succeed regardless of action. The three identical 13/50 on the pod were a
  ~1.6% coincidence (configs are resampled per seed; the count moved to 16 on seed 7). So
  JFROLL/JFGRAD 99 on this ckpt is best-of-32 shooting on a good dynamics model, not policy
  candidates + selection. Eval-side causes ruled out: config carries policy_proj_rank 32,
  the eval loader asserts on missing/unexpected keys (none), the reactive path uses the real
  horizon countdown for goal_terminal=False ckpts (911676c zeros apply only to goal_terminal).
  Owner's renderer-mismatch hypothesis (MuJoCo-EGL is machine-sensitive; an e2e encoder
  amplifies the pixel shift): the PIXEL shift reproduces exactly (dataset frame vs the same
  qpos/qvel re-rendered through the eval's own ReacherDMControlWrapper: MAE 2.581/255, 50% of
  pixels off by >2; owner measured 2.576/40%; the residual is anti-aliasing along the arm
  outline + floor texture filtering, pose identical) but NEITHER encoder amplifies it:
  latent cosine z(dataset) vs z(re-render) = 0.993 bottleneck / 0.998 sig, shift 8-11% of
  the between-state latent distance, and the bottleneck's mean action chunk is IDENTICAL
  under the two renders (per-state cosine 1.000). Render mismatch does not explain this
  arm; whatever is wrong is wrong on the training pixels. Open-loop at K=8 both arms' mean
  draw sits at the zero-action floor (rmse ~1.00 vs 0.987 for zero; single-draw spread
  0.87 z-scored), i.e. single draws are near the action marginal even for the 88% sig policy;
  K=48 MEAN-SIGNAL TEST (N=64 states, 24 min CPU; eval-consistent conditioning h_norm 0.1 --
  the eval countdown IS in anchors, horizon0 = goal_offset_steps/action_block = 5, so first
  replan = 5/50 = 0.1 = training convention; no protocol bug): cosine of the K-mean chunk
  with the dataset chunk, true goal: bottleneck 0.101+-0.028 (shuffled null 0.046) vs sig
  0.204+-0.023 (null 0.040); WRONG goal (another state's): both drop to 0.052 = null, so
  both use the goal, but the goal-dependent part is ~0.05 (bottleneck) vs ~0.15 (sig), 3x
  weaker; |mean|/single-draw spread 0.28 vs 0.45; re-rendered pixels give identical numbers
  (0.101 / 0.205). Off-distribution probe h_norm 0.5: sig 0.108, bottleneck null (0.060).
  POLICY-VIEW SURVIVAL: the goal-relative latent direction (z_goal - z_cur) keeps 81% of its
  norm through the trained rank-32 P (random rank-32 projector 28%, top-32 PCA 80%);
  velocity direction 75%. So the view does NOT discard the goal information -- the action
  branch under-learned a small goal-conditioned signal that the flow loss cannot see (both
  arms val act ~1.5: the loss is dominated by the action noise; on reacher closed-loop SR is
  the only readout of the policy). Not separated yet: policy view per se vs rank 32 (no
  full-rank pw_zp reacher run exists). Owner's renderer-mismatch mechanism: real pixel shift,
  but not amplified by either of these encoders (their 0.257 cosine is from another ckpt).
  REACHER WAVE LAUNCHED 2026-09-04 (owner go; manual mode): three MoT arms, MSE head + projected
  SIGReg + FULL-RANK policy view + zstd_floor 0.01, seed 42, one variable each; entry
  jf_gr_reacher_d9bbefd.sh (tarball d9bbefd = a2cf7cb + pcgrad match_s), evals gc/roll/grad +
  roll-with-random-candidates (3 seeds x 50), memory 200000, research H100:
    R1 rea_r1_base   mf-95d69b77 8079682924f46362  z384 d384 depth8 heads6 (the pusht/toolhang candidate)
    R2 rea_r2_small  mf-588ebbf3 e37f6ba8ec8a91cd  z192 proj_hidden384 d192 depth4 heads4 (192 throughout, ~17M)
    R3 rea_r3_sigbal mf-c39c3af8 04332abcdbae9359  R1 + --pcgrad match_s (S encoder grad rescaled to ||g_P|| each step)
  Readouts: R1 reactive >> 32 (random floor) -> the bottleneck was the problem; R2 ~ R1 -> 17M
  suffices; R3 > R1 -> magnitude story real; R3 collapsing -> SIGReg size load-bearing. NOTE the
  banked "192/4/4" arms (pusht 72.7/83.3/85.3, toolhang 82.0/75.0) are TRUNK-ONLY (z_dim 384),
  not the 192-throughout model the owner asked for; R2 is the first z192 run.
  R3 RESULT (2026-09-04): first launch crashed at the end of epoch 1 on a latent PCGrad-path bug
  (validation loop unpacked 3 of run_batch's 4 values; fixed 0cc5c38, relaunched 6528e75552900457
  resuming from the ep-1 checkpoint). TRAIN_COLLAPSED at epoch 6: val zstd 0.012 (ep1) -> 0.006
  -> 0.004 -> 0.003 -> 0.017 -> 0.0021 (< floor 0.01), reg 50 -> 105 (R1 at the same epochs:
  zstd ~0.5 -> 0.3). With the SIGReg encoder gradient rescaled to the policy's norm the mse
  dynamics collapses the latent inside the first epoch -- SIGReg's SIZE is load-bearing on
  reacher; gradient-magnitude balancing is not a route to the policy. Side facts: match_s costs
  2.07x per epoch (740 s vs 358 s: three backward passes); full-vector conflict rates PD ~0.5
  (orthogonal), DS ~1.0 (dynamics vs SIGReg oppose every step = the collapse tug-of-war).

## Final-experiments push (owner + Minghao, launched 2026-09-04; manual mode, owner go)
  Recipe: MoT + MSE head + noreg, 192 THROUGHOUT (--z_dim 192 --proj_hidden 384 --d_model 192 --depth 4
  --n_heads 4 = 16.9M: encoder 11.3M + trunk 5.6M), --w_reg 0, --zstd_floor 0.005, seed 42, tarball 0cc5c38.
  Minghao's list: (1) GR on 8 envs = 4 GR (pusht, tworoom, pointmaze_large, reacher) + the 4 TC cells converted
  to GR (fs5 strided caches, goal + horizon, H_max 50; toolhang/transport/drawer/cube), same ckpt CEM vs grad
  (CEM for the MoT rollout planner = code todo); (2) TC goal-blind toolhang/drawer/transport + cube goal-cond
  (gcf2, memory 400000, --num_workers 12, resume chain); (3) DP baselines: drawer 52.7 {44,60,54}, transport
  84.7 {84,86,84}, toolhang DP-T 71.0 done; cube DP must be goal-conditioned (todo, DP-side); (4) interpretability
  and (5) longer-horizon GR after training. Reacher HELD for R2 (192-throughout pw_zp) per owner.
  Jobs (entries jf_gr_0cc5c38.sh CELL ARM SEED EXTRA SKIP_EVAL / jf_tc_0cc5c38.sh / jf_tc_gcf2_0cc5c38.sh):
    gr_pusht_nm192    mf-f894e20a 9cd4f7aed939fd1b   gr_tworoom_nm192  mf-f210c74c fde3928fc48b274b
    gr_pointmaze_large_nm192 mf-4dc17ae3 efa41753747f11fb   (GR cells: chained gc/roll/grad/rand evals)
    grtc_toolhang_nm192 mf-a854f305 55858a25ae58dbf4   grtc_transport_nm192 mf-1470a067 5a793e87a86e0b76
    grtc_drawer_nm192 mf-150c9d9a 21bbc004c4dc87c9   grtc_cube_nm192 mf-0431adfe 30b7bb9071f875c9 (TC->GR, train-only; sim-stack GR eval entry todo)
    tc_toolhang_nm192 mf-823e5711 3c9274107615205a   tc_drawer_nm192 mf-6b5ef6ed 4ea65e747e07b2a0
    tc_transport_nm192 mf-ec77b56c bfc2a3002543110f  (goal-blind, train-only; standalone TC evals after)
    tc_cube_gcf_nm192 mf-4c9d91f1 60ad363ec0c03bd8   (goal-cond gcf2, in-entry gc eval, 400000 MB)
  Result dirs: ckpts/jointflow_gr_<cell>_nm192/ (GR + TC->GR), ckpts/jointflow_tc/tc_<cell>_nm192_s42/ (TC).

  REACHER WAVE RESULTS (2026-09-04): R1 base (z384/d384/depth8/heads6, mse + projected SIGReg + full-rank
  P(z) view): reactive 92.0 {94,92,90}, best-of-K 100 {100,100,100}, gradient 99.3 {100,98,100},
  best-of-K with RANDOM candidates 95.3 {96,96,94} -- a LIVE reacher policy, above the jointflow-class
  sig baseline (86.7/94.0); the policy adds ~5 over pure random shooting at K=32. R2 small (192
  THROUGHOUT: z192/proj384/d192/depth4/heads4): reactive 28.7 {28,28,30} = the random floor, best-of-K
  99.3, gradient 100, random-candidate 99.3 -- dead policy on a world model that plans anyway. R3
  (SIGReg gradient matched to the policy's) collapsed at epoch 6. So on reacher the rank-32 bottleneck
  was NOT the cause (R1 full-rank works at 384); MODEL SIZE is: the 192-throughout model kills the
  reacher policy. Unresolved: latent width (z 192) vs trunk (d192/depth4/heads4) -- the banked
  trunk-only 192 arms (z 384) kept live policies on pusht/toolhang; no z192 run exists elsewhere yet.
  The final wave is entirely 192-throughout; pusht_nm192's chained reactive eval is the first check.
  CEM planner for the MoT rollout added (b74c53d: plan_mode=cem, cem_init=policy|zero); unified GR
  eval entry jf_grev_b74c53d.sh CELL CKDIR MODES (gc,plan,grad,rand,cemp,cemz) for all 8 GR envs.
  FINAL-WAVE RESULTS (192 throughout, mse+noreg, seed 42, 3 eval seeds x 50):
  - toolhang AS GR (fs5 strided, goal+horizon, H_max 50; eval goal_offset 50 / budget 100, train h5):
    reactive 86.7 {84,90,86} | best-of-K (policy) 88.0 {90,86,88} | gradient 61.3 {62,62,60} |
    best-of-K random candidates 29.3 {32,26,30} | CEM policy-init 89.3 {88,94,86} | CEM zero-init 35.0
    {32,38,+1 pending}. The policy is the planner's prior here: pure-WM planners (random / CEM-zero) sit
    at ~30-35, policy-guided best-of-K/CEM ~88-89 = the reactive number; the GRADIENT planner HURTS
    (61). Note: reactive-with-goal 86.7 vs the goal-BLIND 384 toolhang 80.7 and trunk-only-192 82.0.
    Eval cost: 10-15 min per 50-episode eval on H100 (robosuite) -> the 18-eval job hit the 3h
    util kill at 17/18; continuation relaunched (idempotent). RULE: <=3 modes per grev job on TC cells.
  - pointmaze_large: reactive 100 {100,100,100} | best-of-K 100 | gradient 100 | random candidates
    80.0 {82,78,80} | CEM policy-init 100 {100,100,100} | CEM zero-init 76.0 {72,80,76}. Policy essential
    (pure WM 76-80 -> 100), nothing left for planning to add.
  - cube gcf2 (TC goal-cond): memory 400000 AND 256000 both died before the entrypoint ran (no
    heartbeat, no platform reason) -> relaunched at the proven 200000 (50310dd78ba952ab).
  - pusht (192): reactive 70.0 {62,78,70} | best-of-K 80.7 {76,84,82} | gradient 86.7 {80,92,88} | random
    candidates 39.3 {34,44,40} | CEM policy-init 86.0 {86,86,86} | CEM zero-init 40.0 {34,52,34}.
    vs the 384 pw_zp (73.3 / 83.3 / 95.3 / 44): the small model is -3 reactive,
    -9 on the gradient planner; the policy is alive (70 vs the 39 floor) -> the reacher policy death is
    reacher-specific to the small model. (val act 0.386 at ep 50 vs 0.32 at 384: slower learner.)
  - tworoom (192): reactive 98.7 {96,100,100} | best-of-K 100 | gradient 100 | random candidates 94.0 |
    CEM policy-init 99.3 {100,98,100} | CEM zero-init 98.0 {96,98,100} (the pure WM solves tworoom).
  - toolhang goal-blind TC (192, 120 ep; pvev board, jointflow_policy full_traj on the eval split):
    reactive 86.7 {82,82,96} | subgoal planner 80.0 (shards 70/90) -- same as the trunk-only
    192 (82.0/75.0) and the 384 noreg (80.7/80.0).
  - drawer goal-blind TC (192, 120 ep; jf_tc_ev reactive, dexmg): 68.7 {60,70,76} vs the 384 mse+noreg 65.3
    and DP 52.7 {44,60,54}.
  - transport goal-blind TC (192, 120 ep; jf_tc_ev reactive, dexmg; standalone eval job e85d1318d5d82a12 after
    TRAIN_OK 03:06): 84.7 {86,82,86} vs the 384 MoT mse+noreg (tc_transport_mnm) 86.7 {86,88,86}, the 08-22
    jointflow-class noreg arm 86.7 {84,92,84}, and DP 84.7 {84,86,84}. Same data
    goal-conditioned (transport AS GR below): 68.7 noreg, 80.0 pw_zp -- the goal pathway, not the model
    size, is what the small model loses on transport.
  - transport AS GR (192): reactive 68.7 {66,70,70} | best-of-K 69.3 {68,64,76} | gradient 42.0 {38,38,50}
    | random candidates 4.0 {4,4,4} | CEM policy-init 62.0 {60,54,72} | CEM zero-init 4.0 {4,4,4}.
  - drawer AS GR (192): reactive 42.7 {40,48,40} | best-of-K 48.0 {38,54,52} | gradient 14.0 {4,16,22}
    | random candidates 9.3 {6,10,12} | CEM policy-init 16.0 {12,18,18} | CEM zero-init 6.7 {6,8,6}.
    On the bimanual dexmg cells pure-WM planners are near zero and BOTH refining planners (gradient, CEM)
    fall well below the policy's best-of-K: refinement walks off the policy's proposals. Eval cost 24-36
    min per 50 episodes (dexmg) -> the four 3-mode jobs hit the 3h kill; continuations run one mode/job.
  DIAGNOSTIC (owner 2026-09-05): does the regularizer/policy view recover planning on the TC->GR
  cells? drawer-as-GR and transport-as-GR at 192 throughout with projected SIGReg + P(z) view
  (--w_reg 0.04 --sigreg_pertime --sep_policy_state, sigreg_proj_dim = z_dim 192, no rank), same
  entry/data/H_max as the noreg runs; train-only, evals as one-mode grev jobs on TRAIN_OK:
    grtc_drawer_pwzp192    mf-3bc46bc9 1bebdac1423906d8
    grtc_transport_pwzp192 mf-2819429c 145ba2f779d1df50
  Context: GCBC (frozen DINOv2-small 22.1M + 19.3M trained predictor = 41.4M) vs ours 16.9M; on the
  noreg 192 TC->GR ckpts planning falls off a cliff (drawer grad 14 / CEM 16 vs best-of-K 48;
  transport grad 42 / CEM 62 vs 69) -- the owner reads this as noreg's shortcoming.
  pw_zp@192 diagnostics TRAINED (drawer TRAIN_OK 01:49 val act 0.273 zstd 0.296; transport TRAIN_OK 02:24
  val act 0.496 zstd 0.290 -- latents ~2x wider than the noreg runs' 0.16); GR evals submitted one mode
  per job (gc/plan/grad/cemp): drawer 9765c5090d93fd75 47f311a2f13bdc9e 491e75c7e347d706 09996ff9c8e7a015;
  transport 458e3e581e2165a1 44e85d608d9b3631 82c4db186d68c524 76e2f6366eeb656a.
  RESULT (3 eval seeds x 50; noreg@192 row in brackets): reactive / best-of-K / grad / CEM-policy
    drawer-as-GR    pw_zp 45.3 {38,46,52} / 52.7 {46,58,54} / 15.3 {12,20,14} / 15.3 {6,20,20}
                    [noreg 42.7 / 48.0 / 14.0 / 16.0]
    transport-as-GR pw_zp 80.0 {78,78,84} / 77.3 {78,72,82} / 54.7 {58,52,54} / 67.3 {70,66,66}
                    [noreg 68.7 / 69.3 / 42.0 / 62.0]
  READS: (1) planning does NOT recover -- on both cells the gradient planner still lands far below
  reactive (drawer 15 vs 45, transport 55 vs 80) and its drift from the warm start is unchanged
  (drawer ||U-U0|| 12.5-13.0, transport 10.9-11.3 vs 13 / 13 noreg; every replan "improves" the model
  cost 3-4x), so projected SIGReg at 192 does not regularize the latent enough to stop the
  exploitation; the TR grid remains the only thing that removes it (and only back to warm start).
  (2) the regularizer + policy view lifts transport's goal-conditioned REACTIVE row by +11 (68.7 ->
  80.0, 3 seeds all >= 78; goal-blind TC on the same data 86.7) and best-of-K by +8; drawer moves
  within noise (+2.6 / +4.7; goal-blind TC 68.7). So on transport most of the goal-conditioning cost
  is recoverable on the policy side without touching the goal pathway; on drawer it is not.
  GOAL CONDITIONING AT THE HEAD (owner 2026-09-05, go on the proposal; code 96618d4): the goal-blind TC
  rows beat the goal-conditioned GR rows on the same data by 18 (transport) and 26 (drawer) points, the
  same loss the goal-image TC arms showed, so the MoT's goal pathway itself is suspect. jointflow never
  fed the goal to the trunk: goal + horizon conditioned ONLY the action readout (GCHeadMSE) and that
  worked. New --goal_cond head (config key goal_cond, default token = unchanged, old checkpoints load
  strictly): no goal token and no horizon AdaLN anywhere in the trunk (the action stream is goal- and
  horizon-free), the readout becomes GCHeadMSE(in 192, goal 192, hidden 512, cond 128, depth 3, dropout
  0.1) = +1.05M params (16.85M -> 17.90M at 192); the state stream is goal-invariant by construction
  (smoke-tested on CPU). Launched, mse + noreg, 192 throughout, seed 42, otherwise the final-wave nm192
  recipe (jf_gr_96618d4.sh, arm gh192):
    grtc_toolhang_gh192  mf-fc19e5be dcd311a1cb3328fc   (train-only; GR evals via jf_grev_96618d4.sh)
    grtc_transport_gh192 mf-2cdf5494 26634f2f5ea9aad2   (train-only; evals one mode per job, 3h kill)
    gr_pusht_gh192       mf-1338381d 8c8f76d09737cd77   (chained gc/plan/grad/rand)
  Rows to beat (reactive / best-of-K / grad): toolhang 86.7 / 88.0 / 61.3, transport 68.7 / 69.3 / 42.0
  (pw_zp 80.0 / 77.3 / 54.7; goal-blind 84.7), pusht 70.0 / 80.7 / 86.7.
  RESULT toolhang-as-GR gh192 (TRAIN_OK 05:35, 21 min, val zstd 0.14; 3 x 50): reactive 86.7 {80,92,88} |
  best-of-K 88.0 {86,88,90} | gradient 70.0 {68,70,72} (token 86.7 / 88.0 / 61.3; gradient drift 9.2 vs
  10.1). Identical to the token model on the reactive and best-of-K rows; the gradient planner is +8.7,
  consistent across seeds but inside the ~8-pt noise band. On the GR protocol the goal pathway's location
  does not matter for toolhang. NOTE (owner, same morning): the goal-image failure to test against is
  the goal-TERMINAL TC arm (toolhang g_nm 60.0 vs goal-blind 80.7 at 384), not GR -- transport never had
  a goal-conditioned TC arm; proposal for the TC-protocol head test pending the owner's go.
  RESULT transport-as-GR gh192 (TRAIN_OK 06:21, 68 min, val zstd 0.12; 3 x 50, one mode per job):
  reactive 76.7 {72,82,76} | best-of-K 73.3 {68,70,82} | gradient 45.3 {40,46,50}, drift 13.0-13.5
  (token 68.7 {66,70,70} / 69.3 / 42.0, drift 13; pw_zp 80.0 / 77.3 / 54.7; goal-blind TC 84.7).
  The head lifts transport's goal-conditioned reactive row by +8 (at the edge of the noise band, all
  three seeds above the token mean) and leaves best-of-K and the gradient planner where they were; the
  planner still walks 13 units off the warm start and collapses. So on GR the head recovers less than
  the regularizer + policy view did (80.0) and neither reaches goal-blind; the goal-image cost on the
  TC protocol (toolhang 60.0) remains the untested target.
  RESULT pusht GR gh192 (TRAIN_OK 10:45, 5 h 30 min = the token wall; val act 0.353 vs 0.365, zstd 0.15; chained
  3 x 50): reactive 66.0 {58,78,62} | best-of-K 72.7 {66,80,72} | gradient 78.7 {72,88,76} | random candidates
  39.3 {38,46,34}; gradient drift 2.4-2.6 = token's. Token: 70.0 / 80.7 / 86.7 / 39.3. The head costs pusht
  4 (noise) / 8 / 8 points -- the one GR cell where the goal is essential is the one the head hurts, while the
  random-candidate floor is untouched (the world model is the same). Across the three GR cells: toolhang =,
  transport +8 reactive only, pusht -8 on both planners; the head is not a GR improvement.
  GOAL-TERMINAL TC WITH THE HEAD (owner 2026-09-05: "only toolhang I wanted with goal-cond TC", "Don't need
  arm2, just 1"): toolhang on the TC protocol that produced the goal-image failure (raw consecutive-frame
  cache, goal = the demo's terminal frame, horizon 0; eval = jointflow_gc + full_traj on tool_hang_ev, 3 x 50,
  the g_nm 60.0 / goal-blind 80.7 protocol). Recipe = the goal-blind nm192 TC arm (jf_tc entry rebuilt on
  96618d4: 120 ep, warmup 10, batch 64, lr 1e-4, fp32, mse, noreg, 192 throughout, seed 42, zstd floor
  0.005) + --goal_conditioning --goal_terminal --goal_cond head; train-only, eval from jf_tcgc_ev_96618d4.sh
  (jf_tc_ev with the mode switched to goal-conditioned full-traj) on TRAIN_OK. No token control at 192
  (owner's call): the head number is read against goal-blind 86.7 (192) / 80.7 (384) and token 60.0 (384).
    tc_toolhang_gt192h  mf-60465876 36f7f3fec0dc3147  (memory 120000; goal-blind twin trained in 3 h 04 min)
    TRAIN_OK 12:12 (3 h 40 min, val act 0.253, val zstd 0.106); goal-conditioned full-traj eval submitted:
    tcgc_ev_tc_toolhang_gt192h_s42 mf-a68c4bc6 39909d72d589f03d.
    RESULT (2026-09-05, reactive JFTCGC, N=50): seed 42 = 70.0 (13:42), seed 0 = 80.0 (15:02); mean-of-2 = 75.0.
    SEED 1 NOT RUN: the job was KILLED ~15:20 by the 3 h low-util wall (started 12:20, each seed ~80 min, 3 seeds
    ~4 h > 3 h). The risk flagged at submit materialised. Per-seed logs persisted (ckpts/jointflow_tc/
    ev_tc_toolhang_gt192h_s42_e{42,0}.log), so 42/0 are safe; NO lock/claim guard survived the kill, so a re-run is
    not blocked. But jf_tcgc_ev_96618d4.sh hardcodes `for ES in 42 0 1` with no skip-if-done and no seed override,
    so a naive resubmit re-runs 42/0 and re-crosses the wall. Seed 1 needs a seed-only variant or a skip-if-log
    guard on the entry. NOT LAUNCHED, awaiting owner go.
    FINDING (2 seeds): the goal-terminal head (75.0 at 192) underperforms goal-blind 86.7 (192) by ~12, same
    direction as goal-token (60.0 < goal-blind 80.7 at 384) and as pusht GR (head < token). Goal conditioning via
    the head does not help on toolhang TC; the 2-seed gap already exceeds what a third seed could close.
    INFRA LESSON: the cube GR evals survived because sharded one-mode-per-job (6 jobs each < 3 h); this eval ran 3
    long seeds in ONE job (~4 h) and died at the wall. Eval entries with >2 long seeds need per-seed skip-if-done
    or seed-sharding to clear the 3 h kill. [[merlin-ops]]
    SEED 1 COMPLETED (2026-09-06 01:04; resubmit with EV_SEEDS=1 once the entry had the guard): 76.0, ALL_DONE,
    per-seed log persisted. FINAL (3 seeds, N=50): 70/80/76 = 75.3 vs goal-blind 86.7 (192) -> -11.4. The 2-seed
    read (75.0) held. Goal conditioning via the terminal head does not help on toolhang TC: it costs ~11 against
    the goal-blind trunk, the same direction as goal-token (60.0 < 80.7 at 384) and pusht GR (head < token). CLOSED.
  CUBE GOAL-CONDITIONED TC (gcf2 entry, 192, 75 ep, 200 GB raw cache mmap, resume chain): TRAIN_OK 2026-09-05
  08:23 (ep 75/75, val act 0.386, val zstd 0.184; 10.9 min/epoch measured). The entry's chained eval died in
  15 s on every seed: FileNotFoundError datasets/ogbench/cube_single_expert.h5 -- the cube eval config reads
  the dataset under the ogbench/ subdirectory and the gcf2 entry links it at datasets/ root (the 08-25 cube
  eval used jf_tc_eval_gc_2b7f7f5.sh, which mkdirs the subdirectory). Relaunched as a standalone
  goal-conditioned eval on the synced checkpoint through jf_tcgc_ev_96618d4.sh (jointflow_gc + full_traj on
  the cube eval split, 3 x 50 -- the 95.3 protocol; mkdir fix added): tcgc_ev_tc_cube_gcf_nm192_s42
  mf-39fc73be 7421b9461bb9facf. References: jointflow-class gcf 95.3 {96,96,94}; cube DP unmeasured (GC todo).
  RESULT: cube goal-conditioned TC, MoT mse+noreg 192 throughout (16.9M), 75 ep: 96.0 {94,98,96} (eval 08:50-09:08,
  ~9 min per 50 episodes). Matches the jointflow-class 95.3; the final wave's TC side is complete: toolhang 86.7,
  drawer 68.7, transport 84.7 (goal-blind), cube 96.0 (goal-conditioned).
  CUBE AS GR (final wave, grtc_cube_nm192 30b7bb9071f875c9): TRAIN_OK 2026-09-04 19:21 (50 ep, val act 0.545,
  val zstd 0.207) but its GR evals were never submitted -- the trigger fired while the wave watcher was pointed at
  the cube TC heartbeat (found 2026-09-05 12:20 when the owner asked what was left). Submitted as the scheduled
  continuation of the approved wave, one mode per job through jf_grev_96618d4.sh (cube case fixed: dataset link
  under datasets/ogbench + the eval split, full_traj protocol; memory 120000):
    gc 47e905a4b07dfd60  plan 4f927f60213eb8fc  grad 5bedb5226d7cf562  rand af1812c319f2fc78
    cemp 2699663f2310d8f1  cemz 617dec6593c16fbc
  RESULT (2026-09-05, all 6 jobs ALL_DONE; 3 seeds 42/0/1, N=50 each; cross-verified heartbeat vs 18 per-seed logs):
    JFGC 98/98/98 (98.0)   JFROLL 100/100/98 (99.3)   JFGRAD 100/98/96 (98.0)   JFCEM-pol 100/100/98 (99.3)
    JFROLLRAND 4/8/4 (5.3)   JFCEMZ 2/8/4 (4.7).
    The four informative planners saturate at 98-99.3 (cube goal-cond TC was 96.0), so cube-as-GR = cube-as-TC:
    a solved cell either way. The two UNINFORMED controls collapse to ~5 -- random candidates and zero-init CEM
    strip the learned prior, so the ~99 is the goal-conditioned policy prior, NOT the search machinery. This is the
    control that makes the planner rows interpretable. Naming (tarball 96618d4): rand job tags JFROLLRAND, zero-init
    CEM tags JFCEMZ. Results live ONLY in ckpts/jf_grev/{hb_cube_nm192.log, ev_cube_nm192_<mode>_e<seed>.log} (no json).
- **5-STATE / 25-ACTION ARMS (Minghao's ask via the owner, 2026-09-05; "Rather A and B. Both PushT GR and
  Toolhang TC"; "For consistency we should go as is").** Questions: does a flow head predicting 5 states
  learn meaningful dynamics without SIGReg; does 5-state mse bridge the planning gap; are 25 actions needed.
  CODE FACTS: with num_states_pred 5 and fs 5 the trainer forces num_actions_pred >= 25 (state token q
  attends to the CLEAN actions up to q*fs, the first 25 entries of the policy's target tensor) -- 25 actions
  are coupled to 5 states by construction, not by joint denoising (MoT denoises the action stream then the state stream in SEPARATE phases); but WITHIN the
  state pass each future latent attends causally to the earlier state tokens (allow[r, st[:q+1]], motflow.py:230), the
  mirror of action chunking's own causal self-attention among action tokens (allow[r, ny[:j+1]], :241) -- so state
  prediction is single-pass CHUNKED JOINT prediction with a causal horizon chain, NOT per-horizon feedforward and NOT
  commit-and-re-encode autoregressive rollout. token k = the latent fs*k action-steps ahead, so 5 tokens = horizons
  5/10/15/20/25 (one per action block). the reactive protocol
  executes one block (5 actions) per replan whatever the chunk length; the planner's _imagine uses only the
  FIRST state token per pass (multi-block imagination = a planner change, not written yet); both caches
  support 5 state targets. Smoke-tested on CPU (58 tokens, 16.86M params, state tokens 1-4 ignore block-5
  actions, token 5 responds; AdaLN-Zero makes every block the identity at init, so the wiring test needs
  randomised gates). Batch 16 / history 1 from the suggested command NOT carried over (owner: MoT predicts
  one anchor per sample, batch 16 is a mistake; history 2 keeps the velocity cue).
  BATCH-SIZE CONVENTION SURFACED (owner: "Was toolhang always bs64? First I'm hearing this"): every TC arm on
  record (28 toolhang + drawer/transport/cube) trains at batch 64, lr 1e-4, 120 ep, warmup 10 (the jf_tc entry
  inherited it from the toolhang r2 noreg recipe); every GR arm at batch 128, lr 1.5e-4, 50 ep, warmup 5.
  Kept as is for comparability (owner's call). DECISION (owner 2026-09-05, "not ... a way to replace autoregressive rollout ... a supervision signal like action
  chunking but for states"): the 5 states are a SUPERVISION signal, not a planning mechanism -- the multi-block
  imagination planner is NOT being written. the planner keeps consuming only the first predicted latent (horizon fs)
  per replan, so these arms eval by the SAME mechanism as 1-state and isolate whether the state-chunk target sharpens
  the shared trunk that also produces the actions. whether a trained model routes through the state-to-state edge vs
  leans on the action prefix is empirical (perturb one state token, watch downstream) -- not needed for the decision.
  Launched, 192 throughout, noreg, history 2, seed 42, memory
  120000, `--num_states_pred 5 --num_actions_pred 25`; B adds `--mot_state_head flow --state_prior gauss
  --state_param v`:
    gr_pusht_s5m192    mf-328e15fa f43ebc45c5f587ac   (GR recipe; chained gc/plan/grad/rand; ref 70.0/80.7/86.7/39.3)
    gr_pusht_s5f192    mf-b92508e8 98805bc4f2b9e1f9   (flow; the 384 flow head planned 72/70 at 1 state)
    tc_toolhang_s5m192 mf-442d0bb8 e89070d6f0fc7e39   (TC recipe, train-only; board reactive eval via jf_tc_ev_96618d4.sh; ref 86.7)
    tc_toolhang_s5f192 mf-dc94a415 a3c413eb7272a685   (flow; the 384 flow head was 86.0 vs mse 80.7 reactive)
  RESULT pusht GR (2026-09-05, 3 seeds/N=50; JFGC/JFROLL/JFROLLRAND = reactive/best-of-K/random-candidate):
    s5m192 (mse):  JFGC 70/78/66 = 71.3   JFROLL 72/86/72 = 76.7   JFGRAD <eval bug>   JFROLLRAND 40/50/38 = 42.7
    s5f192 (flow): JFGC 64/72/64 = 66.7   JFROLL 64/68/62 = 64.7   JFGRAD <eval bug>   JFROLLRAND 22/18/22 = 20.7
    CLEAN INTERNAL READ (both arms identical but the state head): flow < mse on EVERY working metric --
    reactive 66.7<71.3, best-of-K 64.7<76.7, random 20.7<42.7 (the low random = flow degraded the dynamics/action
    prior). vs the 1-state baseline (~70.0 reactive): mse 71.3 = NEUTRAL (state-chunk supervision neither helps nor
    hurts the mse trunk on pusht GR), flow 66.7 = HURTS. So on pusht GR the 5-state target does not improve the
    trunk, and mse is the head to keep. Matches the toolhang TC training proxy (flow val act 0.334 >> mse 0.277).
  BOTH TC TOOLHANG ARMS TRAIN_OK (mse val act 0.277 / state 0.00095, flow val act 0.334 / state 0.0127; 1-state
    goal-blind was ~0.253). Board eval PENDING owner go, wall-safe shard required (full_traj ~80min/seed x3 > 3h).
  JFGRAD FAILED all 6 (both arms x 3 seeds), NOT stochastic: RuntimeError at motflow.py:300 forward_tokens,
    'a (5) must match b (25) at dim 1'. The gradient planner _rollout_cost_grad -> imagine_step (gip.py:2015) hands
    imagine_step fs=5 clean actions, but num_states=5 makes n_clean=25 (num_states*fs). num_states>1 specific (the
    1-state planner matched because n_clean=fs=5). Reactive/best-of-K/random call the sampler with the full block,
    so they pass. Not a supervision-signal issue -- a planner path that assumes n_clean=fs. FIX = feed imagine_step
    the full 25-action block (or add a single-block imagine mode). NOT fixed; offered to owner.
  OWNER GO (2026-09-05, "Yes on all 3"): toolhang TC board evals, JFGRAD fix + grad re-run, goal-terminal seed 1.
  FIX 966463b (motflow.imagine_step): accepts a single block (fs actions), zero-padded to n_clean. Inert for state
    token 0 under the causal map (it attends only to its own block) = the token every planner reads; the gradient
    still flows through the block. CPU-verified mse+flow: single-block == full-block on token 0 (dz 0.00e+00),
    grad through the block 0.46/0.56. Tarball lewam_jointflow_966463b.tar.gz (mirrors 96618d4 minus the inert
    .gitignore); GR eval entry jf_grev_966463b.sh built on it.
  EVAL ENTRIES HARDENED (jf_tc_ev_96618d4.sh, jf_tcgc_ev_96618d4.sh): EV_SEEDS env overrides the seed list; a
    seed whose persisted ev_<arm>_e<seed>.log already carries a success_rate is skipped, so a kill is resumable by
    resubmit and long evals shard under the 3 h wall. Stager refusal guard: will not submit against an entry
    lacking EV_SEEDS, nor without the checkpoint. A fresh run of either entry is behaviourally unchanged.
  LAUNCHED (7 jobs, research-guarantee H100 [10 free at submit], memory 120000, Evaluation):
    toolhang board eval (full_traj ~80 min/seed) sharded (42,0)+(1) per arm:
      ev_s5m192_s42_0 mf-c1052bd8 4ece4613b950a271    ev_s5m192_s1 mf-a564b254 31381c5f18da314a
      ev_s5f192_s42_0 mf-caddc222 347a518110b42f53    ev_s5f192_s1 mf-a281f23f 27bb6d867bebc8f4
    goal-terminal seed 1 (42/0 persisted; the guard skips them): ev_gt192h_s1 mf-0e55d920 1b412f5dd0dee2b3
    pusht JFGRAD re-run on the fixed tarball (grev, 3 seeds per job; pusht grad is short, clears the wall):
      grad_s5m192 mf-540401d2 918320652dec9335    grad_s5f192 mf-9939914c e3b709833be36ff0
    Results: ckpts/jointflow_tc/hb_ev_tc_toolhang_{s5m192,s5f192,gt192h}_s42.log (both shards of an arm append
    to the same file) and ckpts/jf_grev/hb_pusht_{s5m192,s5f192}.log. Read the persisted per-seed logs, not a
    watcher, for the numbers.
  JFGRAD s5m192 RE-RUN (fixed code 966463b): 82/84/78 = 81.3, ALL_DONE, 97-229 s per seed. The fix holds on GPU
    (3/3 seeds succeed where 3/3 failed before). vs 1-state grad 86.7: -5.4, matching the best-of-K gap (76.7 vs
    80.7). Planner ordering preserved for the 5-state mse trunk: grad 81.3 > best-of-K 76.7 > reactive 71.3 >
    random 42.7. Full pusht GR 5-state mse row: 71.3 / 76.7 / 81.3 / 42.7 vs 1-state 70.0 / 80.7 / 86.7 / 39.3 --
    neutral on reactive, ~4-5 down on both planners. The state-chunk target improves the pusht GR trunk on no method.
  JFGRAD s5f192 RE-RUN (fixed code): 60/66/54 = 60.0, ALL_DONE, 248-351 s per seed.
  PUSHT GR 5-STATE COMPLETE (both arms, 4 methods, 3 seeds / N=50 each):
                   reactive  best-of-K   grad   random
    1-state ref      70.0      80.7      86.7    39.3
    s5 mse           71.3      76.7      81.3    42.7
    s5 flow          66.7      64.7      60.0    20.7
    mse: planner ordering NORMAL (grad > best-of-K > reactive) -- planning still pays; neutral on reactive, -4/-5
    on the planners vs 1-state. flow: planner ordering INVERTED (reactive 66.7 > best-of-K 64.7 > grad 60.0) --
    every planner lands BELOW acting reactively, i.e. the flow head's dynamics are not exploitable by search and
    planning through them degrades. With random-candidate 20.7 this is the cleanest evidence the flow state head
    does not learn useful dynamics (owner's question, does a flow head predicting 5 states learn meaningful
    dynamics without SIGReg: NO on pusht GR). flow < mse on all 4 metrics. VERDICT pusht GR: the 5-state target
    helps nothing; mse is the head to keep; flow is harmful. Toolhang TC (the other half) pending, 5 evals running.
  TOOLHANG TC 5-STATE FLOW COMPLETE (2026-09-06, reactive board protocol JFTC, 3 seeds / N=50, both shards
    ALL_DONE; the (42,0) shard finished 02:02 with seed 0 at 01:59 = 2 h 25 min after start, inside the wall):
    86/92/94 = 90.7 vs goal-blind 1-state 86.7 -> +4.0; also above the 384-width flow head (86.0). The best
    toolhang TC reactive number on record. The OPPOSITE of pusht GR (flow 66.7, harmful): the flow state head is
    CELL-DEPENDENT -- it helps the toolhang TC trunk and hurts the pusht GR trunk. mse (86/88, seed 0 pending)
    is tracking ~87 = neutral.
  TOOLHANG TC 5-STATE MSE COMPLETE (2026-09-06 02:09, both shards ALL_DONE, 6/6 per-seed logs persisted for both
    arms): 86/88/74 = 82.7 vs goal-blind 86.7 -> -4.0; one weak seed (74) drives it, the other two sit at/above
    baseline. Seed spread on this cell: baseline {84,92,84}, mse {86,88,74}, flow {86,92,94}; 4-point gaps sit at
    the edge of 3x50 resolution, the flow>mse direction is the robust part (flow wins 2 of 3 paired seeds, ties 1).
  === 5-STATE / 25-ACTION EXPERIMENT COMPLETE (all 4 arms, every eval, 3 seeds / N=50) ===
    PUSHT GR        reactive  best-of-K   grad   random      TOOLHANG TC    reactive (board)
    1-state          70.0      80.7      86.7    39.3        1-state        86.7 {84,92,84}
    s5 mse           71.3      76.7      81.3    42.7        s5 mse         82.7 {86,88,74}   -4.0
    s5 flow          66.7      64.7      60.0    20.7        s5 flow        90.7 {86,92,94}   +4.0
    ANSWERS. (1) flow head learning meaningful dynamics without SIGReg: CELL-DEPENDENT. pusht GR: no -- planner
    ordering inverted, random-candidate 20.7, search cannot exploit it. toolhang TC: it produced the best reactive
    number on record (90.7, +4) -- a trunk effect; the reactive protocol does not exercise the dynamics under
    search and no planner eval exists on toolhang. (2) 5-state mse bridging the planning gap: NO -- pusht planners
    went DOWN (-4/-5), toolhang reactive -4. (3) 25 actions needed: untested, structurally coupled to 5 states.
    VERDICT: the 5-state target is not a general improvement for the mse trunk on either cell. The flow head is
    harmful on pusht GR and the toolhang TC winner. RECOMMENDATION (owner's call, not decided): mse stays the
    default head (never catastrophic); flow as a toolhang-class TC option; the 5-state target is not a default.
    INFRA that made this land: imagine_step single-block fix 966463b (grad 6/6 where 6/6 failed), EV_SEEDS +
    skip-if-done eval entries (5 sharded toolhang evals + goal-terminal seed 1: zero re-runs, zero wall kills),
    job-status pollers replacing the START-timing watcher.
- **REACHER DATA = A PURE RANDOM POLICY (verified 2026-09-06 on our reacher.h5; Minghao's GitHub-issue lead).**
  10,000 episodes x 201 steps = 2,010,000; one NaN action row per episode (boundary marker, 0.50%). Both action
  dims on 100k rows: min -1 / max +1, mean 0.00, std 0.577 (= 1/sqrt(3), the std of U(-1,1)), 10-bin histogram
  exactly [0.1]x10 (uniformity 1.00); lag-1 autocorrelation within episodes +0.004 / -0.003; max |corr(action,
  any qpos/qvel component)| 0.003 / 0.002. The actions are i.i.d. uniform on [-1,1]^2 at every step, independent
  of state AND time: a random policy exactly, not "roughly". READS: (a) the marginal p(a) carries no policy to
  imitate, which is why random-candidate search already scores 95-99 on reacher and the policy adds ~5; (b) the
  mse-noreg collapse's "act flat at 1.52" is the flow velocity loss at its floor (the target a - eps has no
  conditional structure once the latent is dead, and its noise term dominates the gradient before that); (c)
  under hindsight goals E[a | s, g] IS state-dependent (the actions that happened to move toward g), a legitimate
  reaching action that an MSE head regresses directly with no noise term, while a flow SAMPLE from that broad
  conditional is mostly noise. Owner's hypothesis: this is flow matching's problem on reacher relative to a pure
  MSE policy. OWNER CORRECTIONS (2026-09-06, "Completely wrong"): (1) the mse ACTION head must MIRROR the mse
  state head -- a learned action_query in the noisy slots (as state_query fills the state slots), the loss
  branching only on the TARGET (the chunk itself vs the velocity), one pass at tau one at inference. My first
  version fed zeros through action_in and branched the tau/noise construction: wrong, replaced (action_query is
  registered only under the mse head so every flow ckpt keeps its layout; behaviour identical). (2) 192
  THROUGHOUT, not 384: a collapse is NOT a reference, and 384 broke the final wave's consistency. The reference
  is R2 (192, flow policy + SIGReg): reactive 28.7 {28,28,30} = the dead policy, best-of-K 99.3, grad 100,
  random 99.3; R2 pace 4 h 46 min / 50 ep. Proposed arm: the collapsed mse-noreg reacher recipe at 192 (SMALL
  flags) with --mot_action_head mse + --zstd_floor 0.005. Loss readout: the mse floor is the z-scored marginal
  variance 1.0, so act < 1.0 by ep 10 = learning E[a | s, g] (R2's flow act sat at 1.51-1.52 = its floor).
  LAUNCHED (owner "Go" 2026-09-06): gr_reacher_motnm_mse192  mf-21f8d882  a05ec4ca7a5208bb. Entry
    jf_gr_reacher_msehead_63d58f5.sh = the nm_c299ee7 entry verbatim + --mot_action_head mse, --zstd_floor 0.005,
    the 192 flags (z 192 / proj_hidden 384 / d 192 / depth 4 / heads 4), and the rand eval mode; tarball
    lewam_jointflow_63d58f5.tar.gz; YAML = the R1 reacher pod verbatim (memory 200000, Train, research-guarantee;
    20 H100 free at submit); seed 42; chained gc/plan/grad/rand x 3 eval seeds x 50 (dm_control 1.0.43). Results:
    ckpts/jointflow_gr_reacher/hb_gr_reacher_motnm_mse192_s42.log. Expected from R2's pace: ~72 min to ep 10 (the
    readout: act < 1.0 = learning; TRAIN_COLLAPSED = the latent died under noreg, via the trainer's zstd_floor
    marker), ~4 h 46 min to TRAIN_OK, evals after. A deterministic policy makes JFROLL == JFGC by construction;
    read reactive (vs R2 28.7), grad, rand.
  FAILED AT START (2026-09-06 08:20-10:28, a05ec4ca7a5208bb): ZERO epochs. Startup was line-for-line R2's (same
    prints, same 8-worker fork), then 32 tracebacks multiprocessing DupFd -> resource_sharer -> mkdtemp ->
    "OSError: [Errno 28] No space left on device: /tmp/pymp-*": the pod's /tmp had zero bytes when the DataLoader
    workers' fd-sharing sockets needed a temp dir. Our code stages nothing locally (the cache is read straight from
    the HDFS preload dir); both entries set TMPDIR=/tmp and num_workers 8; R2's identical path worked. The cause is
    on the pod side and not visible from the devbox (host n124-139-220 also ran a cube eval yesterday, which
    installs a sim stack under /tmp: host-persistent /tmp is a HYPOTHESIS, not verified). The main process HUNG on
    the dead workers (train.log kept being copied) at 0% GPU; killed via merlin-cli runs stop (status killed)
    instead of waiting for the 3 h util kill. The collapse question was never asked.
  v2 RELAUNCH (failed-job carve-out; training flags unchanged): gr_reacher_motnm_mse192v2  mf-fded5552
    f3f6ee996dc15f4b. Same entry edited in place (jf_gr_reacher_msehead_63d58f5.sh; own ARM name so its heartbeat
    and ckpt dir are separate from the killed run): DISK / TMP_TOP heartbeat lines at START and after env (df of
    /, /tmp, /opt/tiger, /dev/shm with fstype; top /tmp entries) so any next failure is explained within 3 min;
    TMPDIR, train/eval logs, pip temp and STABLEWM_HOME moved under $WORK on /opt/tiger (where the tarball
    extracted fine); pip --no-cache-dir + cache purge. Results:
    ckpts/jointflow_gr_reacher/hb_gr_reacher_motnm_mse192v2_s42.log.
  DIAGNOSIS CORRECTED by v2's START diagnostics (same host n124-139-220, fresh container): /, /tmp and /opt/tiger
    are ONE overlay, 984G with 126G free; /dev/shm 88G free; /tmp held a few hundred KB of platform files. So the
    host-persistent-/tmp theory was WRONG and the TMPDIR relocation moot. The loader (train_jointflow.py:201-206)
    is np.load of the whole fp16 .npy into RAM without --cache_mmap (R2 identical): our code writes no local copy.
    The cache is 410000 x 3x224x224 x fp16 = ~123G, read through the fuse mount; v1 exhausted a 126G-free overlay
    during exactly that read (a read-through block cache is the obvious mechanism, unproven from the devbox; R2's
    host simply had more room). v2 carried no disk sampling during the load, so it was killed rather than left to
    reproduce v1 blind. UNDETECTED-HANG FAILURE (owner, 2026-09-06): v1 sat hung 2 h at 0% GPU with 32 tracebacks
    in train.log (copied to HDFS every 60 s) while my pollers waited for ep 10 under a 4 h ceiling against a
    72 min ETA; the owner had to ask. Fix = progress_watch.sh: alarm within 5 min on any Traceback in train.log,
    on a missed ETA, or on death. [[merlin-ops]]
  v3 RELAUNCH: gr_reacher_motnm_mse192v3  mf-6f5ffb90  011f2974f8dc2204, same entry in place: MOUNTS (fuse
    options) at START; FAIL-FAST exit 4 with DISK_TOO_SMALL if the overlay has < 160G free; DISK_T free-disk
    sample every 5 min during the load; training flags unchanged. Last relaunch: if the guard trips twice, stop
    and bring the numbers to the owner. Results: ckpts/jointflow_gr_reacher/hb_gr_reacher_motnm_mse192v3_s42.log.
  CANCELLED (owner 2026-09-06 ~10:50, "Minghao says kill. He is fixing reacher data itself"): v3 killed via
    merlin-cli after passing the disk guard on n124-112-071 (344G free). The reacher.h5 on record is a pure random
    policy (see REACHER DATA above); Minghao is replacing the data, so no reacher arm runs on the current file.
    KEPT for the fixed data: the mse ACTION head (63d58f5, --mot_action_head mse, mirror of the mse state head),
    the entry jf_gr_reacher_msehead_63d58f5.sh (disk guard + DISK_T sampling + fail-fast), progress_watch.sh.
    When the new reacher.h5 lands: rebuild its preload cache first (the fp16 fs5 strided cache is dataset-
    specific), re-probe the action distribution, then re-propose the arm against a same-width reference.
  FIXED REACHER DATA LANDED (2026-09-06; Minghao's handoff commit 49079aa = swm/ReacherVisibleTargetDMControl-v0 +
  configs/eval/reacher_policy.yaml, registered in eval.py/eval_gip.py; data wf8/train/reacher_policy.h5 75.6 GB at
  03:46; cache preload_cache/reacher_policy/reacher_policy_fs5_i224.* built 11:48-12:27 by the workspace worker:
  506,632 frames / 489,128 samples / 17,504 episodes, 152.5 GB fp16; the trainer resolves --dataset_name
  reacher_policy.h5 to exactly that path). RE-PROBED (100k rows): 17,504 episodes of 64-350 steps (mean 143),
  success-terminated (n_eps_no_success 0); action std 0.47/0.46, center-peaked histogram (old: 0.577 = U(-1,1),
  flat); lag-1 autocorr +0.97 (old 0.00); max |corr(action, qpos/qvel)| 0.90/0.81 (old 0.00); target_pos = one fixed
  visible target per episode. A REAL policy: the random-policy problem is gone, and the flow-vs-mse policy-head
  question loses its original motivation. PROTOCOL: the new config keeps the SAME constants as reacher (num_eval
  50, goal_offset 25, eval_budget 50, action_block 5; goals drawn from the policy dataset itself); it changes only
  the env (the DMC target geom is kept visible and restored per episode from target_pos via a set_task_target_pos
  callable; the official qpos_match predicate is unchanged) -- Minghao's note: a CEM run on the new data through
  the old hidden-target env scored GR 16.7 because every live frame after step 1 left the dataset's visual
  domain. So the redone row IS protocol-comparable to the old reacher rows; the data and the env differ. Not yet
  on disk: an a2f copy (not needed by jobs) and a wf8/eval split (the reacher entries link the train file for
  eval, as before). Redo unblocked pending the owner's go.
- **FAILURE/SUCCESS VIDEOS, transport + drawer AS GR (owner 2026-09-06: "Just go reactive for now"; "give a couple
  samples of success too (they could be from degenerate goals) -- not only assess our policy, but whether GR makes
  sense even for these datasets").** Provenance of the table rows: grev jobs tagged nm192a (reactive / best-of-K /
  gradient) and nm192b (random / CEM), heartbeats ckpts/jf_grev/hb_{transport,drawer}_nm192{a,b}.log, weights
  ckpts/jointflow_gr_<cell>_nm192/nm192_s42; seed-42 reactive = transport 66.0, drawer 40.0. NOTE for the owner's
  table: transport gradient = 42.0 {38,38,50} on 3 seeds (the 38 in the row is seed 42 alone). Those grev runs
  already rendered the panel videos (agent | dataset | goal; env_{i}.mp4, one per episode; the 25-29 min per seed
  includes rendering) and never copied them off the pod; eval_gip prints the episode_successes array to the log, so
  failures are selectable with no code change. Entry jf_grvid_63d58f5.sh = the grev wiring verbatim (sim stack,
  DexMG vars, EVAL-split DSARG, staging) + copy all 50 env_*.mp4 and the eval log per cell to
  ckpts/jointflow_gr_<cell>_nm192/videos_gc_s42/, parse the printed array into failures.json, live-copy the eval
  log every 60 s (crash visibility), DISK line at START. Job grvid_transport_drawer_nm192 mf-5185dbd2
  7b1ad614466ec5c7 (Evaluation, memory 120000, research-guarantee, 25 H100 free). Delivery: failure clips + a few
  successes per cell, plus a per-episode goal-displacement proxy (|start frame - goal panel|) to flag degenerate
  goals.
  RESULT (job 7b1ad614466ec5c7, 11:57-12:54; transport 1517 s, drawer 1786 s; 50/50 clips per cell on HDFS under
  ckpts/jointflow_gr_<cell>_nm192/videos_gc_s42/ + failures.json + eval log): reproduction EXACT -- transport 66.0
  (17 failures), drawer 40.0 (30 failures) = the nm192a seed-42 values. Panel geometry (layout_check.png): 736x288,
  three 224 px panels at x=16+240k, y=16..240, labels below; order agent | dataset | goal. Every episode is exactly
  100 frames (eval_budget 100); success is LATCHED (ever reached), so a success's last frame need not match the
  goal. Pixel-proxy metrics per episode (mean |diff| on the panel; agent-vs-h5 comparisons carry a ~12-15 rendering
  baseline, demo-motion and agent-motion do not); analysis + sheets in
  /home/tiger/.claude/jobs/5f788414/tmp/vid/<cell>/ (episodes.json, contact_<cell>.png, analyze_vids.py).
  TRANSPORT: SR by demo-motion tercile = low(<=6.0) 29% | mid 75% | high(>9.3) 94% (n~17 each). 17/50 eval segments
  are NEAR-STATIC (demo motion < 6: the goal is the start within the tight positional threshold); SR within them
  29%, EXCLUDING them 85% (n=33). 4/33 successes (12%) are the agent standing still on a static segment (degenerate:
  ep 40/41/23 +1); 12 failures are static segments where a small precise change is required and the policy holds
  still (ep 48/28/32) or drifts (ep 24). Motion-segment failures (ep 33/38) are real: the object is carried to the
  wrong place. corr(demo motion, agent motion) 0.74. READ: the 68.7 row mixes real reaching (85-94 on motion
  segments) with a thresholded hold-still test on a static third -- GR on transport needs a minimum-displacement
  filter on the eval segments to measure reaching.
  DRAWER: SR by tercile = 47 | 31 | 41 (flat; only 5/50 segments near-static -- NOT the static artifact). Failures
  are real policy failures on multi-stage windows: grasp the mug and lift, never place it in the drawer (ep 21/8);
  a different configuration (ep 22); spurious motion on a static segment (ep 47: agent 21 vs demo 4). 3/20
  successes (15%) degenerate (agent static: ep 4/32/34); high-motion successes (ep 36/38) latched mid-episode =
  genuine. corr 0.57. READ: drawer's goals are diverse and non-trivial (demo motion 1.7-54); 42.7 is a real
  measure of a weak grasp-and-place policy on partial windows. Delivered: both contact sheets, 10 curated clips,
  episodes.json. One seed, 50 episodes, pixel proxies: indicative, not the criterion.
  TRUST-REGION GRADIENT PLANNING (owner 2026-09-05, "on manipulation the gradient planner becomes
  super exploitative"): the [grad] diagnostics confirm it -- on drawer/transport the refinement moves
  the plan by ||U-U0|| ~ 13 (z-scored, 25-step bimanual plan) and cuts the model cost 6x (0.012 ->
  0.002) on EVERY replan (pusht: move 2.6, cost 0.0020 -> 0.0011, and it helps). Grid launched on the
  noreg-192 ckpts, gradient planner otherwise unchanged (warm start best-of-K K32, 50 Adam steps, lr
  0.05, clip 10, |a|<=3), penalty grad_tr * ||U-U0||^2 with grad_tr in {1e-3, 1e-2, 1e-1} (the pusht-era
  value 1.0 would freeze the plan here); eval entry mode `gradtr` (env GRAD_TR/GRAD_STEPS/GRAD_LR):
    drawer    tr1e-3 31c037ae89816d4e  tr1e-2 caf8c7b4c7ef1388  tr1e-1 f56c28aca865818b
    transport tr1e-3 9bc778d6375bb77d  tr1e-2 e953c69981c8f427  tr1e-1 239769eced1399b3
    pusht     tr1e-3 e95906fdaeb3d690  tr1e-2 155d57e9e0be1a6d  tr1e-1 bb2cb00cabefaf21  (reference)
  Working theory: SIGReg regularizes the space so the pw_zp arms need no trust region (pw_zp@192
  drawer/transport diagnostic running).
  TR GRID RESULTS (3 seeds x 50; move = mean ||U-U0||):
    pusht:     warm 80.7 | free grad 86.7 (2.6) | tr1e-3 82.7 (0.36) | tr1e-2 84.7 (0.22) | tr1e-1 84.7 (0.21)
    transport: warm 69.3 | free grad 42.0 (13)  | tr1e-3 72.7 (1.4)  | tr1e-2 70.0 (1.3)  | tr1e-1 71.3 (1.3)
    drawer:    warm 48.0 | free grad 14.0 (13)  | tr1e-3 38.7 (1.7)  | tr1e-2 43.3 (1.4)  | tr1e-1 42.7 (1.3)
  The trust region removes the exploitation (transport 42 -> ~71, drawer 14 -> ~43) but lands at or
  slightly below the warm start: restrained refinement adds nothing on the manipulation cells; on pusht
  (where free refinement helped) it costs 2-4 pts. The penalty saturates by 1e-2 (moves ~equal 1e-2..1e-1).
  DYNAMICS ABLATION (owner 2026-09-05: "is the world model learned with the policy useful for grading
  policy rollouts, or could any-old world model do?"): plan_mode=extwm_bok (0d96e3d) keeps the policy's
  K=32 best-of-K proposals (still imagined block-by-block through OUR dynamics) but SELECTS by an
  external frozen LeWM's terminal latent cost (official stable_worldmodel LeWM instantiated from the
  checkpoint's config, predictor included; its encoder on the last 3 real frames + goal; rollout over
  the executed past blocks + candidate blocks; z-scored blocks = shared dataset stats). Logs the
  LeWM-vs-our-dynamics pick agreement. Graders: pusht = the authors' HF release
  (code/lewm_main_eval/hf_release_native/pusht, CEM 88.7 on our protocol); toolhang = the newest LeWM,
  wf8_uni/toolhang/toolhang_lewm_official50 (ep 50, 2026-09-03, vit-tiny scratch). NOTE the repo's
  build_frozen_lewm drops the predictor (GC-IDM encoder-only path) and the "cem_lewm" entry planned with
  a LeWAM-Unified ckpt -- neither is a LeWM world model; the probe_wm_discrim retarget loader is.
  RESULT pusht (nm192 ckpt, HF-release LeWM grader, 3 seeds x 50): JFEXTWM {80, 94, 90} = 88.0 vs
  our-dynamics best-of-K 80.7, reactive 70.0, random candidates 39.3, grad 86.7, CEM-policy 86.0.
  The picks are nearly independent: LeWM's argmin equals ours on 17/309, 21/266, 23/265 replans
  (6-9%, chance 3%), and the LeWM pick sits at mean rank 14.8/15.3/14.1 of 32 under our cost
  (uniform = 16.5). So on pusht the grader is interchangeable: a foreign world model that never saw
  the policy grades its proposals at least as well as the co-trained one (+7, within seed spread
  {80,94,90}); the +18 over reactive comes from selection among policy proposals, which any decent
  latent cost delivers. The co-trained dynamics is not what makes best-of-K work here.
  RESULT toolhang-as-GR (nm192 ckpt, official50 LeWM grader, 3 seeds x 50): JFEXTWM {86, 86, 84} = 85.3
  vs our-dynamics best-of-K 88.0, reactive 86.7, random candidates 29.3, grad 61.3, CEM-policy 89.3.
  Picks again nearly independent (agreement 28/403, 26/353, 21/405 = 5-7%; LeWM pick at mean rank
  14.6/13.6/14.7 of 32 under our cost). On toolhang selection buys nothing under either grader
  (both within noise of reactive), so the row says only that swapping the grader costs nothing.
  Across both cells: the co-trained dynamics is interchangeable with a foreign LeWM as the best-of-K
  scorer, and the two cost landscapes barely correlate -- the planner gain (pusht) or its absence
  (toolhang) is a property of the policy proposals, not of which world model ranks them.
- **BENCHMARK SWEEP + BOTTLENECK + PARAM ABLATION (2026-09-03, code c299ee7, seed 42).**
  Launched (owner go, waves 1+2 + p192): toolhang/pusht bottleneck (proj-sig + policy view,
  --policy_proj_rank 32), cube mse-noreg goal-terminal (gcf2 entry, 75 ep), transport and
  drawer mse-noreg (120 ep), toolhang/pusht mse-noreg at d192/depth4/4heads (17.3M total,
  MoT 5.3M vs 55.0M/42.6M). REACHER mse-noreg: first attempt OOM (rc 137: the 115 GB
  mmapped strided cache needs memory 200000, not the 120000 eval-template default);
  relaunched, then COLLAPSED SILENTLY by ep 10 (zstd 1.1e-4, state loss 1e-4, act flat at
  1.52 = marginal action variance; not a NaN, so the collapse guard did not fire) -- killed
  by owner. Reacher is the first cell where the default recipe fails outright; the SIGReg
  comparison exists (jointflow+sigreg 2026-08-27: reactive {92,80,88} = 86.7, planning
  {96,96,90} = 94.0). Open: reacher with proj-sig + policy view (candidate #2).
  OPS: collapse guard is NaN-only -- add a zstd floor kill for GR cells.**
- **THE POLICY-DYNAMICS TRADEOFF IS NOT A GRADIENT CONFLICT (2026-09-03, devbox diagnostics).**
  Four independent measurements on the toolhang mse-noreg arm (and others), all agreeing:
  (1) TRUE induced feature changes dz = z(theta - eta*g) - z(theta) on the encoder (owner's
  definition, linearity 2.00): at ep15 cos(E[dz_D],E[dz_P]) = +0.83, per-row +0.41, the
  mean-removed cross-cov cooperative 6x null vs fighting 2x null; converged +0.37 / +0.09.
  |dz_D| is 58x (ep15) to 135x (converged) SMALLER than |dz_P|: the dynamics is a passenger
  on the shared encoder. The earlier "opposed mean pulls -0.33 at ep15" used the dL/dz
  stand-in and is an artifact of ignoring the shared-encoder coupling. (2) Parameter-space
  cos(P,D) restricted to the MoT trunk: +0.005 (ep15 and converged), coin-flip signs, P/D
  norm ratio 26-103x; full-sig trunk -0.001, 11x. (3) Probe pairs over training: P vs D_in /
  D_tar / S all within +-0.05 with 15-45% negative steps at every timescale incl. ep<=1;
  the only large anti-alignment, D_in vs D_tar (-0.5, 100% of steps, all arms), is Siamese
  geometry (one residual through two similar Jacobians with opposite sign; net keeps ~90% of
  the larger half, target path 1.3-5x the input path) -- not conflict. (4) Round-1 fdp
  per-direction scores a_P_Dfull: mse arms ALIGNED early (+0.60 / +0.49, 1-3% negative dirs
  at ep15), mild late disagreement only under noreg (38% at final); flow arms = noise.
  DECISIONS: representation-space PCGrad (all variants) and parameter-space PCGrad NOT
  built; --pcgrad flag (sym / protect_p, {P,D[,S]} over the full param vector, per-epoch
  conflict-rate logging) exists in the trainer but is untested on the cluster.
  WHAT TRACKS SR INSTEAD: the width of z's live subspace and of what the policy reads.
  z dims90 (real frames): flow-noreg 9 (SR 86.0, per-dim std 0.003!), mse-noreg 17 (80.7),
  s2s 18 (69.3), pw_zp 30 (76.7; its P(z) 22), pw 31 (64.7), full-sig 34 (54.7). fdp
  usage_P dims90: 12 / 19 / 74 (sm) / 207 (sf). BUT the inference-side action<-z_t
  sensitivity has PR 3.5-5.8 (dims90 7-13) in EVERY arm -- the policy OUTPUT responds to
  ~10 directions everywhere; what differs is the gain per typical latent fluctuation
  (|dA/dz| x z-std: noreg 0.03, full-sig 0.06). Owner hypothesis (2) survives as "more
  action movement per unit latent motion, part of it nuisance", not as "reads more dims".
  FINAL DIAGNOSTICS (2026-09-03, owner: stop probing after these): (a) policy sensitive
  directions sit INSIDE z's live subspace in every arm (0.70-0.90; random ~0.05) carrying
  20-30x random variance -- no policy reads noise directions; the share of z variance inside
  the policy subspace falls with regularization: nf 59% / nm 57% / s2s 59% / pw_zp 33% /
  pw 38% / sm 22%. (b) Within-state action spread over 16 flow-noise draws: nf 0.147 (best
  SR, widest), all others 0.092-0.100; rmse(mean draw, dataset action) 0.16-0.17 for all but
  nf 0.23 -- the SIGReg policies are neither noisier nor worse-fit on demo states. (c) Paired
  live-vs-dataset render (same sim state, local robosuite 1.4.1 = lower bound): latent shift
  0.12-0.20 of z-std, isotropic w.r.t. policy dirs (0.10-0.18 std along policy dirs = along
  random dirs), action change 5-8% of between-state variation for EVERY arm -- no arm is
  more render-sensitive. CONCLUSION: on the demonstration distribution the policies are
  equivalent by fit, stochasticity, readout geometry and shift sensitivity; the 30-pt reactive
  spread exists only CLOSED-LOOP (on self-induced states). Untested (by owner decision): the
  off-manifold/recovery gap on the policies' own rollout states.
  WORKING HYPOTHESIS (owner, 2026-09-03): one mechanism explains both halves. A wider /
  higher-gain latent maps the same small physical deviation to a larger latent displacement:
  as the policy unrolls and drifts, states look MORE off-manifold than they physically are
  -> classic IL covariate-shift error (invisible to every on-distribution probe, which all
  came back flat); the SAME magnification is discriminability for the verifier -> expert vs
  random chunks separate better -> planning improves. Consistent with: nf/nm/sm ordering,
  s2s (noreg-like width but 1.7x std, anchored objective: lower reactive 69.3, planning ok),
  pw 64.7 vs pw_zp 76.7 (same wide latent; the shield gives the policy a low-gain view), the
  2x action-per-latent-fluctuation gain under full-sig. Evidence bar: keep 95 pusht planning
  while closing toolhang to -4 (pw_zp does this: -26 -> -4). Not yet measured: the encoder's
  gain on PHYSICAL state perturbations per arm (predicts the full ordering incl. s2s).
  Candidate follow-ups, all on the policy's view only: bottlenecked P (384 -> small,
  LoRA-style low-rank factorization so frame_in weight sharing is untouched), contraction/
  smoothness penalty on P, or noise/DART-style augmentation on P(z) for the policy branch.
  PHYSICAL-PERTURBATION TEST (2026-09-03, 48 demo states, 4 perturbed renders x 2 scales,
  arm joints 0.01/0.03 rad + object xyz 2/6 mm): the encoder's gain on physical deviation is
  the SAME in every arm in its own units -- |dz|/tot_std 0.24-0.30 (small) / 0.58-0.66
  (large); |dz| / demo nearest-neighbour spacing 2.6-3.4 / 6.1-7.5 (a 2 mm nudge moves every
  arm's latent ~3x beyond the demo spacing); along the policy's directions 0.21-0.25 / 0.48-
  0.56 std; action change 9-14% / 21-28% of between-state variation, no SR ordering (nf 13%,
  sm 14%, pw_zp 9%); the shield's P(z) gain (0.28 / 0.63) equals its raw-z gain. So the
  local-gain form of the hypothesis ("same deviation looks more off-manifold under SIGReg")
  is NOT supported: relative to each latent's own scale, all encoders resolve physical
  deviation identically and the policies compensate their gain. Whatever separates the arms
  closed-loop is non-local (larger/compounded deviations, or temporal structure of the latent
  along rollouts), not the local sensitivity at demo states.**
- **FEATURE-SPACE CONFLICT (M_s) IS NULL-CONSISTENT ON MSE-NOREG (2026-09-02, devbox, ~10 min).**
  Owner proposal: PCGrad in representation space via M = E[dz_D dz_P^T], project g_D away from
  h = E[J^T Q dz_P] with Q = the negative eigenspace of M_s. Measured with the free surrogate
  (feature gradients g_z = dL/dz at the encoder output, 2048 history rows, toolhang raw cache)
  on tc_toolhang_probe_mot_nm_s0 at ep15/45/105 + best, and on pw_zp best, against 3
  shuffled-pairing nulls each: the FIGHTING side of the cosine-normalized M_s spectrum never
  exceeds independent pairing (negative mass 0.26-0.45 vs null 0.47-0.59; top negative
  eigenvalue 0.7-1.1x the null's), while COOPERATION does (top positive 1.1-1.8x null,
  trace 4-7x null, mean cos(g_D,g_P) +0.014/+0.051/+0.020 at ep15/45/105). Policy and
  dynamics feature gradients are mildly aligned throughout training on the default recipe;
  Q = Pi_neg(M_s) would project onto noise. Also: with Q = I and the g_z surrogate, h is
  exactly the encoder part of g_P, so the scheme reduces to encoder-restricted PCGrad.
  DECISION: not built. Our earlier conflict evidence was regularizer-vs-dynamics-TARGET
  (cos(Dtar,S), r_S), not policy-vs-dynamics. Caveats: surrogate (not NTK-weighted JVP dz),
  excludes the dynamics target-path rows, noreg only. Script: jobs tmp ms_spectrum.py.
  Same day: --detach_goal_grad added (3cfd56b; z_goal.detach() before use, goal still
  conditions, encoder cannot be shaped by the goal pathway; smoke-verified) -- arms dg1 on
  toolhang goal-terminal (vs g_nm 60.0) and pusht GR (vs mot_nm 68.7/86.7/88.7), seed 42.
  RESULTS (2026-09-02 evening, 3x50): PUSHT detach reactive 69.3 {64,78,66} / roll 78.7
  {76,80,80} / grad 89.3 {86,96,86} vs mot_nm 68.7/86.7/88.7 -- reactive and gradient
  planning unchanged, rollout selection -8 (single seed, flag only). TOOLHANG goal+detach
  reactive 62/70 on seeds 0/1 (g_nm: 64/46 same seeds; seed 42 swept separately) -- the
  ~25-pt goal cost is NOT recovered (66 vs 60 vs goal-blind 80.7); subgoal 10.0 (the
  quarantined goal-cond selection anomaly again). VERDICT: the goal does not damage
  toolhang by reshaping the encoder; the cost lives in the conditioning itself (policy
  reading an uninformative, varying terminal image). Detach is harmless on GR.**
- **GOAL-BLIND ROUND + S2S (2026-09-02; trains 00c1d98, evals 911676c; toolhang seed 42,
  3x50 reactive, 20-ep 2-shard subgoal selection; pusht s2s 3x50 chained).**
  TOOLHANG GOAL-BLIND reactive / subgoal-selection:
    noreg ref 80.7 (rig control 85.0) / not measured · full-sig ref 54.7 / 80.0 ·
    proj+pertime sig ALONE (pw) 64.7 {66,64,64} / 80.0 {100,60 by 10-ep shard} ·
    proj sig + sep_policy_state (pw_zp) 76.7 {80,78,72} / 60.0 {60,60} ·
    s2s prev-prior flow sigma0.02 noreg 69.3 {70,64,74} / 75.0 {70,80}.
  READS: (1) the projection alone recovers only +10 of full-sig's -26 reactive damage; ADDING
  the policy view recovers +22 (76.7, 4 pts off noreg) -- the sep_policy_state shield is the
  load-bearing half on TC, causal evidence for the policy-dynamics conflict story. (2) the
  policy view COSTS plan-time selection (pw 80.0 -> pw_zp 60.0): the planner rolls the policy
  through P(z) of IMAGINED latents, which P never trained on -- same mechanism as the pusht
  nm_zp planning dip. No toolhang arm wins both columns; pw_zp is the best compromise.
  (3) s2s: usable selection (75) with zero regularizer, reactive -11 vs noreg on TC.
  PUSHT S2S: reactive 74.7 {74,78,72} = BEST pusht reactive of any arm · roll 82.0 · grad
  80.0 (BELOW its own roll: Adam through the displacement dynamics adds nothing, vs +7-12 on
  mse dynamics). s2s trades top-end planning for policy quality; direction FLIPS by cell
  (helps GR reactive, hurts TC reactive).
  GAP CLOSED (10:41): noreg toolhang subgoal-selection = 80.0 {80,80} (probe mot_nm ckpt,
  POC protocol). Full plan-time column: noreg 80.0 = full-sig 80.0 = pw 80.0 > s2s 75 >
  pw_zp 60. SELECTION NEVER DISCRIMINATED the dynamics on toolhang -- the POC's "+25 from
  plan-time support" was only full-sig recovering its own reactive damage (54.7 -> 80), not
  a regularizer-specific capability. On toolhang, plain noreg mse wins or ties EVERY
  measured column (reactive 80.7, selection 80.0); the sigreg family's real wins are pusht
  gradient planning (95.3 vs 88.7) -- the projected+shielded recipe's value is keeping that
  while not losing toolhang reactive (76.7 vs full-sig's 54.7). pw_zp is the ONLY arm where
  selection lands BELOW its own reactive (76.7 -> 60): the policy-view-on-imagined-latents
  OOD cost, now isolated against three 80s.
  OPS: startup lane-kills recurred at 3 lanes (1 lane lost in each job's first minutes;
  idempotent sweep-up jobs cover); pw train #1 died to a real H100 hardware fault
  (cudaErrorContained nvlink) -- plain relaunch reproduced its twin's curve exactly.**
- **SEP-POLICY / PROJECTED-SIGREG CAMPAIGN, DAY 1 (2026-09-01/02; 7 trains 00c1d98 + evals
  911676c; 1 training seed everywhere).**
  PUSHT (GR, 3x50 evals chained; refs mot grid): reactive/roll/grad --
    noreg ref 68.7/86.7/88.7 · full-sig ref 72.0/85.3/95.3 ·
    proj+pertime sig (mot_pw) 73.3/84.0/93.3 · sep_policy_state (mot_nm_zp) 69.3/81.3/81.3
    {78,94,72} · BOTH (mot_pw_zp) 73.3/83.3/95.3 {96,96,94}.
  Projected per-time SIGReg keeps the full-sig planning benefit at no reactive cost; the
  policy view alone mildly hurts planning on the collapsed noreg latent (P trained only on
  real z misreads imagined states there) and costs nothing on the SIGReg-shaped latent; the
  combination ties the campaign peak 95.3.
  TOOLHANG GOAL-TERMINAL (mode=jointflow_gc, 3x50): goal-cond arms all land ~52-60 vs the
  goal-blind board 80.7 -- g_nm 60.0 {70,64,46} · g_nm_zp 58.0 {64,58,52} · g_pw 52.7
  {52,60,46} · g_pw_zp 52.0 {44,56,56}. RIG CONTROL: the goal-blind probe mot_nm ckpt
  through the SAME rig = 85.0 {84,86} (board 80.7) -- the rig is clean, the ~25-pt GOAL COST
  ON TOOLHANG IS REAL (opposite of cube, where the same recipe+eval scored 95.3 vs 22
  goal-blind; toolhang demo terminals are near-identical scenes, so the goal token carries
  ~no task information but its incidentals can mislead -- hypothesis, untested). Within the
  goal batch: proj-sig costs only ~7 vs its noreg twin (full-sig cost 26 goal-blind), z_P
  costs ~0-2 reactive. SUBGOAL-SELECTION on the goal-cond pw arms = 15.0 both (20ep,
  2 shards, fixed planner cost/conditioning split) -- selection UNDERPERFORMS reactive by
  ~37 on goal-cond ckpts where the goal-blind POC gained +25; OPEN QUESTION, do not reuse
  goal-cond subgoal numbers.
  GRAD-PROBE VERDICT on proj+pertime SIGReg (r_S = applied SIGReg share of the encoder
  update, thirds): TH 0.51/0.48/0.43, PU 0.39/0.35/0.35 vs full-sig 0.44/0.31/0.32 -- the
  force does NOT retire (DxD W cannot make full-rank-Gaussian satisfiable); BUT
  cos(Dtar,S) ~ -0.01..-0.09 (collapse standoff neutralized, like full sig; noreg arms
  -0.19..-0.49) and r_D stays at noreg levels 0.03-0.07 (full sig amplified D's pull to
  0.14-0.30). Projected form = sigreg's geometry without conscripting the dynamics.
  OPS: eval-mode trap -- goal-cond TC ckpts MUST eval with +gip_eval.mode=jointflow_gc
  (jointflow_policy feeds null_goal to goal-always models: numbers 20-56, invalid); >=4
  parallel eval lanes draw startup SIGKILLs (2-3 lanes safe); idempotence checks must
  invalidate stale logs FIRST. FORMAL JOBS CAN BE KILLED:
  `yes | merlin-cli --control-plane i18n-tt job-v2 runs stop --json
  '{"sid":"<mlx job id>","stop_reason":"..."}'` (verified; the "web-UI only" claim was wrong).
  Planner fixes 911676c: steps=None => h_norm 0 (goal_terminal convention) everywhere;
  cost_goal separates the planner score target from the policy conditioning goal.**
- **FEATURE-DIMENSION PROBE ROUND 2 (2026-08-31, job mf-83396961, code bc75e97; [1] toolhang
  4x8 rerun with per-slot ablation zt/prev/tar -> fdp_slots/, [2] the pusht GR mot grid s42,
  goal-conditioned with real encoded goals, snaps ep25/ep40/final + goal slot -> fdp_pusht/).
  PUSHT REGIME TABLE (final): A_D mass (top-32+bands) mse-sig 272 >> mse-noreg 50 > flow-sig
  13 >> flow-noreg 0.4 -- the two arms where the pusht gradient planner works (95.3 / 88.7)
  are exactly the top two; eff-rank mse-sig 69 (vs 6.4 noreg: the SIGRegxMSE rank explosion
  transfers), BUT SIGReg RAISES flow's rank on pusht (4.0 -> 12.7) where it lowered it on
  toolhang (10.1 -> 7.9): the "SIGReg reduces flow ER" sub-claim is dataset-dependent, revise
  to inconsistent-and-small under flow. CLAIM-2 CONTRAPOSITIVE: pusht has almost no
  dynamics-dominated co-need (mse-noreg 0 co-needed anti-aligned dims; mse-sig 6 of ~69 live,
  asym 85-213x but confined) and pusht reactive is correspondingly FLAT (68.7-72 across arms,
  no craters) -- damage structure absent, damage absent. Same mse-sig recipe, opposite
  outcomes by ROOM: pusht = huge A_D mass + rank-69 space + fights confined -> policy fine +
  best planning; toolhang = fights across a 42-dim live space -> worst policy.
  SLOT LOCALIZATION (toolhang final): the policy reads BOTH history frames ~equally
  (A_P zt~=prev, e.g. nm 3.02/3.23); the dynamics ignores the earlier frame (A_D prev 5.2 /
  8.8 / 0.1 / 0.8) and splits between READING z_t (36.9 / 157 / 0.5 / 28.9) and SHAPING the
  target (62.8 / 239 / 0.8 / 35.7, ~1.7x the read side); on the top conflicted dims D's
  z_t-read dependence ALONE exceeds P's total (nm dim1: 6.23 vs 0.46+0.53) -- the fight is
  read-vs-read AND read-vs-write, not purely write-side. METHOD NOTE: all-slot ablation
  UNDERSTATES A_D (all < zt alone; ablating input+target together partially cancels in the
  MSE) -- consistent direction across arms so orderings stand, but single-slot is the sharper
  instrument. Pusht goal slot: A_D(goal)=0.0 in all arms (the state-stream goal mask verified
  empirically); mse-sig's policy barely reads the goal (A_P goal 0.14 vs 1.2-2.1 elsewhere).
  CAVEAT: the eps/2 linearity check degraded on the near-collapsed pusht arms (mse-noreg /
  flow-noreg cos 0.76-0.85) -- their a_k carry curvature error; ablation-based conclusions
  (A masses, co-need) don't differentiate and are unaffected.**
- **FEATURE-DIMENSION PROBE (2026-08-31, jobs mf-a3bd98f9 + ep15 backfill mf-13d17432, code
  5aeac4b, experimental/feature_dim_probe.py, owner design; toolhang, the 4 instrumented
  tc_toolhang_probe_* s0 arms x 8 snapshots ep15..ep105+final on ONE fixed probe set of 2048
  train windows; npz per combo in jointflow_tc/fdp/, figures jobs-tmp fdp_figs/). Method per
  (arm,snap): eigenbasis u_k/lam_k of Cov(z_t); NECESSITY A(k) = dL/L with direction k's
  variation replaced by its probe mean in every latent the trunk sees (top-32 dirs
  individually, identical tau/noise via re-seeded RNG); INDUCED REPRESENTATION UPDATES =
  virtual encoder step theta-eps*g_l per loss (P=action, D_full=state, D_in=target-detached
  exact split, D_tar=full-in, S=SIGReg, counterfactual on noreg arms), eps calibrated to
  rms(dz)/rms(z)=1e-3 (linearity at eps/2: cos .96-.99), dz on the same 1024 probe frames;
  a_k = uncentered per-direction corr of two updates (this measures interaction THROUGH the
  encoder Jacobian, unlike raw dL/dz); conflict mass = sum_k sqrt(A_P A_D) max(0,-a_k).
  FOUR REGIMES, one per arm:
    flow-noreg (best reactive 86): NO interaction -- every global cos(dz_P,dz_D) ~ 0.00-0.03,
      conflict ~0.01-0.09 (10x below every other arm), and A_D ~ 0.00-0.02 everywhere (the
      flow state loss needs no specific direction); the policy owns an eff-rank ~10 latent
      unopposed.
    mse-noreg: tug-of-war AT THE TOP of the spectrum -- final dir 1 (14.9% of variance) has
      A_P=1.55, A_D=2.66, a=-0.54; conflict mass grows 0.04 (ep15) -> 1.89 (final). The ep30
      SR crater (65 -> 10) coincides with the ONLY negative cos(dz_P,dz_Dtar) epoch (-0.10;
      +0.76 at ep15, +0.48 by ep45 as SR recovers) and a 15x conflict jump -- and with the
      grad-probe's cos(Dtar,S) consolidation window (ep20-45), two independent instruments.
    mse-sig: SIGReg SIDES WITH THE POLICY -- cos(dz_P,dz_S)=+.80/+.75/+.71 in the first half;
      the fight sits on dynamics-owned mid-spectrum dirs (e.g. dir 3: A_D=5.6 vs A_P=0.12,
      a=-0.55); the only arm whose eff-rank keeps GROWING (26->42).
    flow-sig: crammed AND fighting -- rep pins at exactly ~8 live dims (top-8 share 1.00),
      all eight needed by BOTH losses (A_P .6-1.1, A_D .9-2.0), top-2 anti-aligned (a=-0.89,
      -0.75), conflict mass highest anywhere (2.77 at final).
  RANK vs REACTIVE SR INVERTS THE NOISE HYPOTHESIS: the policy is best on the LOWEST-rank
  rep (flow-noreg ~10 -> 86) and worst on the highest (mse-sig ~42 -> ~55); SIGReg keeps
  dims alive that only the dynamics needs and the reactive policy pays for their presence --
  planning (subgoal 80.0) is where those dims pay off. Within mse-noreg, conflict mass and
  SR move inversely (min conflict 0.31 at the ep60 SR peak 85). Caveats: snapshot SR = 20
  episodes x 1 seed; a_k uncentered (owner formula, includes mean-push agreement); npz save
  to HDFS needs local-then-copy (Errno 95, first launch mf-a3bd98f9's predecessor died on it).**
- **COST-TO-GOAL PROBE (2026-08-27, 0f88f9d, owner spec: distribution over samples +
  cost vs unroll step; 200 dataset anchors, K=32, M=8 rollouts x H=8; pusht goal +10
  anchors, cube terminal goal). Latent-unit medians: pusht cost now 0.73 / real next
  0.69 / imagined next 1.37 (joint) 1.33 (inpaint); cube 0.79 / 0.69 / 1.42 / 1.41.**
  Imagination moves AWAY from the goal by ~0.63-0.65 while reality moves closer by
  0.04-0.10; only 7% of imagined samples are "closer" vs 62-66% of real next states.
  Spread over the 32 candidates: joint std 0.55 (CV 0.39, wide) but the planner's
  fixed-noise inpaint score std 0.003-0.004 (CV 0.002) — ALL CANDIDATES TIE under the
  score used for selection. Unroll: imagined cost flat at ~1.40-1.54 for 8 steps
  (pusht 1.45->1.40; cube 1.47->1.52) while the real expert curve falls 0.70->0.32 on
  pusht (cube flat ~0.65-0.71, terminal goal already near). The imagined trajectory
  never approaches the goal; it sits at a fixed offset ~1.4 from it. Owner reading
  confirmed: high-variance predicted state => every candidate equally bad under
  cost-to-goal => selection = random pick from the policy band. Plot
  jobs tmp/cost_probe.png; raw jointflow_tc/cost_raw_jf_ev_{pusht,cube}.npz.
- **DYNAMICS-INFORMATIVENESS PROBE (2026-08-27, f224e13, owner spec: real-world vs
  predicted next latents during GC eval; 30% executed random chunks). cube gcf / pusht
  noreg, eval seed 42 x 50 eps (cube 3286 transitions, pusht 367):**
  var(pred)/var(real) = 4.1 (cube) / 7-8 (pusht) — predictions are OVER-dispersed, not
  collapsed (sampler noise dominates; matches the straight noise-transport geometry).
  ||pred-real|| / ||real-prev|| median = 3.6-3.8 cube, 6.7-7.5 pusht on-policy (random
  chunks 5.1 / 8.1): a single-sample prediction is 4-8x WORSE than predicting "no
  change". corr(model goal-cost on pred, true cost on real) = 0.25 cube / 0.00 pusht
  (persistence baseline 0.87 / 0.81); corr of cost CHANGE 0.13-0.18 cube / -0.01..0.12
  pusht; sign(progress) accuracy 0.52 cube (chance) / 0.31-0.33 pusht (anti — the
  dispersed prediction sits farther from the goal than z_prev). Displacement direction
  cos(pred, real): cube on-policy 0.27, random 0.09; pusht 0.14-0.17 / 0.04 — weak
  directional knowledge on-policy, ~none off-policy. VERDICT: as an absolute predictor
  the jf state branch is uninformative on pusht and marginal on cube; the counterfactual
  ranking signal (top-1 .77) exists only under fixed noise. Planning-by-cost on pusht =
  random selection, measured. Owner framing confirmed: WAM SR rests on the policy; the
  imagination is not (yet) a usable verifier. Caveat: single-sample predictions; a
  K-averaged mean would reduce dispersion (the "best case" dynamics) but planning as
  practiced uses single/fixed-noise samples. Dumps: jointflow_tc/dyn_{cube,pusht}.npz.
- **LADDER RUNG 2 (2026-08-27): tau-biased sampling + independent schedules.** Owner
  parity check passed (same h5/caches/z-scoring, goal image on EVERY sample with
  h~U[1,H_max] tail-clamped = unified's RANDOM mode, shared eval harness). LDA-1B
  homework (lda/model/modules/action_model/UWM_ActionHeader.py, GR00T-style head):
  SEPARATE independent timesteps per modality, each ~ Beta(a,b) mapped t=(0.999-s)/0.999
  (biased toward the CLEAN end — same direction our curvature probe picked), BOTH
  timesteps fed to the trunk via separate embeddings, plain unweighted loss sum, Euler
  sampler; only policy mode implemented (obs denoised in tandem). Our equivalents:
  `--tau_alpha 2` (tau ~ Beta(2,1) = U^(1/2), clean-end bias, alpha=1 uniform) and
  `--split_tau --tau_cond summed` (independent taus; separate tau_action_in/tau_state_in
  projections summed into the global AdaLN cond = both taus visible to every token).
  Arms: tau2, split (pusht anchor, plan+gc evals + probes chained). Builder gained
  `--u8` (direct uint8 strided cache, npy memmapped straight to --out) + `--max_eps`;
  reacher fs3 rebuilt as fs3c on it (fp16 fs3 build died at 49 min: .bin+.npy on pod
  /tmp; 250G pod died pre-entry — proven shape is memory 200000).
- **BEST-BETS ROUND NULL (2026-08-26, jf_gr_arm 75595bc, one variable each on the
  pusht anchor): a10s2 (--num_states_pred 2) plan 64.7 / reactive 69.3, probes ==
  noreg (angle .52, vel .56/.06), state field still straight+non-contracting;
  resid (--state_residual, delta-target with z_t add-back) best fit ever (val act
  .332) but plan 68.0 / reactive 66.7, probes == idm-ish (angle .46/.58, vel
  .62/.12), state field unchanged; a5s1 (--num_actions_pred 5) FAILED both ways —
  bf16 exploded ep 23 (first-ever pusht bf16 NaN; healthy 0.44 -> blowup in one
  epoch) and the fp32 rerun trains but does not generalize (train act .46 vs val
  3.16 at ep 20): the 10-action chunk does real stabilizing/regularizing work.
  VERDICT: state-target restructuring does not move representation detail, SR, or
  flow geometry. Ladder rung 2 = tau-biased sampling toward tau=1 (the curvature
  probe's data end); rung 3 = honest-limitation write-up.
  a5s1-fp32 EPILOGUE: val oscillated 1.7-3.2 during training (lr-driven basin
  hopping) then SNAPPED to train (0.365) as lr annealed — settled ckpt evals plan
  64.7 / reactive 66.0 (band). Probes: block_pos 0.922 (BEST jf, near unified's
  .935) and the FIRST bent jf state field (n64 state cos_adj .981, turn ~58deg vs
  1.0000 machine-straight everywhere else) — the full-chunk-conditioned state token
  gives the state flow real conditional structure. But angle .50/.57 and vel .55
  unmoved, SR band. ROUND MORAL: individual representation metrics ARE movable
  (a5s1 blockpos+state-geometry, idm causality+margins, sig04 spectrum) — angle+vel
  never reach unified levels and SR never leaves 63-74 regardless.**
- **PUSHT STATE-FACTOR PROBES (2026-08-26, probe_state_factors.py f6ab7f9, ridge
  n=4000/1000, spec: pos/angle from z_t, vel from [z_{t-5},z_t]): test R²
  agent_pos / block_pos / block_angle / agent_vel(pair) — jf-noreg .973/.890/.545/.568
  · jf-idm .973/.872/.487/.613 · jf-sig04 .943/.719/.420/.601 · UNIFIED-idm05
  .985/.935/.711/.748.** Unified beats every jf variant on every factor, biggest on
  angle+vel (pusht's precision factors). Unified's ctx (aggregator) column ≈ its z1
  everywhere → the advantage is the per-frame ENCODER, not context. Unified reads vel
  0.61 from a SINGLE frame (expert position↔velocity correlation channel jf misses).
  SIGReg DEGRADES metric readout (block_pos .89→.72) while holding best jf SR (74) —
  its +8 is policy-shaping, not representation. FAILURE VIDEOS (noreg s42, SR 56,
  22 fails; OWNER trajectory review overrides the final-frame read): failures begin
  EARLY — the agent pushes too strong or from the wrong contact point at push onset
  and the trajectory diverges from expert quickly; the block ends near the target
  with the angle far off. Mechanism: the model does not learn contact physics (which
  push point yields the needed block rotation). ~2 escapes, 0 never-engaged. A
  POLICY-level limitation (reactive == planning fail identically). Ruled out for
  the jf-vs-unified gap: budget (owner receipt), context (flat), SIGReg, IDM.
  ViT-CLS+SIG jf (gr_sigvit_s42, 2026-08-27): .919/.714/.493/.604 (vel-single .03) ==
  ResNet+SIG (sig04 .943/.719/.420/.601) — encoder FAMILY is not the variable; unified's
  ViT (.985/.935/.711/.748) differs by OBJECTIVE. Same curvature signature (action turn
  cos .46, state field straight). Global-vs-local cue story does not explain the gap.
  Standing hypothesis: flow-head gradients starve the encoder of metric detail;
  discriminators = --split_tau arm, flow-on-aggregator hybrid. Owner constraints:
  ONE recipe for GR+TC, no cell hacks; no pretrained encoders; no MSE-aux (mode-mean
  risk). Videos: jointflow_gr2/pusht_fail_videos/.
- **PUSHT w_idm ARM (2026-08-26, jf anchor recipe + --w_idm 0.5, 50 ep, 36dd67e):
  HEADLINE (BoK-32 inpaint plan) {56,76,58} = 63.3; reactive diagnostic {60,76,62} =
  66.0 — both in the noreg band. BUT the substrate transformed: dyn probe top1 0.82
  (noreg 0.77, sig04 0.26), sensitivity 1.28, cost_cv 0.275 (noreg 0.069 — 4x wider
  margins); diversity probe out_div 4.17 (noreg 11.7) at unchanged act_div 0.36.
  zstd held 0.073 all run (noreg collapses to ~0.03). CONCLUSION: w_idm fixes the
  world-model half on every probe axis and moves SR zero — on pusht, policy quality
  and imagination quality are decoupled axes; selection planning neither exploits a
  good model nor is rescued by one. NOTE: idm05's 87.2 is pusht_uni_idm05 = a UNIFIED
  arm (z192+SIGReg+IDM+MSE head, 200 ep) — not a jf datapoint; this is the first
  clean jf+IDM one.**
  (gr_tworoom_s42) = 96.0/100.0/100.0 → 98.7**, identical to the reactive band
  (96/100/100, pre-seeding-fix canary eval). Planning neither helps nor hurts at the
  ceiling; the warm-CEM collapse remains pusht-specific until shown otherwise.
- **POINTMAZE_LARGE SOLVED (2026-08-26): reactive (jointflow_gc) 100/100/100 AND
  planning (BoK-32 inpaint exec5, the variant of record) 100/100/100 → 100.0** (GR
  anchor recipe: lr 1.5e-4, bs 128, 50 ep, warmup 5, fs-strided cache built in-entry,
  H_max 50; 3 eval seeds × 50, budget 100/offset 50). Final val act 0.870, state
  0.009, zstd 0.050. Headline number = the PLANNING one (owner rule: GR headlines are
  planning-mode). With tworoom 98.7 the pusht band (~67-74) is pusht-specific, not a
  general 2-D GR gap; reacher is the remaining GR datapoint.
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

**DP eval board (dp_ev_cell, protocol-identical to the jf rows, 3 eval seeds × 50):
toolhang 71.0 · drawer 52.7 (ep-120) · transport {84,86,84} = 84.7 (ep-120 snapshot;
jf goal-blind {84,92,84} = 86.7 — jf +2.0, both goal-blind so the row is fair).
Cube DP eval declined by owner (DP not goal-conditioned, honest protocol is GC).**

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
