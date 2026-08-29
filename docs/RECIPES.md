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
  H1 (--state_mse, regression state slot inside the joint trunk) was implemented (e16c67f) and
  REMOVED the same day at the owner's request (800ddd3): a regression target under actions at
  random noise levels is not a coherent dynamics model.**
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
