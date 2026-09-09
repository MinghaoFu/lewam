# Paper robustness plan (supervisor feedback, 2026-09-09; revision 3 after owner review)

Status: PROPOSAL, except E7-prep run A (done 2026-09-09, results in RECIPES). Each item lists what already exists in the repo,
the exact design, cost, and the deliverable. New code follows the design-before-code rule:
pseudocode here, review, then implement.

**Order (owner, 2026-09-09, confirmed):** E7 first (with E2's script built alongside, no
cards) -> B1 official baselines through our eval -> E3, E4, E5 -> E6 -> E1 lowest.

## 0. Story and audience

**Claim.** LeWAM is a world-action model trained end-to-end from pixels: one encoder, one
latent, one trunk, with a dynamics head (state prediction) and a policy head (flow action
prediction) trained jointly. The latent is therefore *action-aware* by construction, and the
same latent serves both the reactive policy and the planners (best-of-K, gradient, SteerMPC,
CEM). No prior world-action model trains encoder, dynamics and policy jointly from scratch on
pixels; DINO-WM and LeWM freeze or pretrain the encoder and bolt a planner on afterwards, and
policy-only methods (DP, BC-RNN) have no dynamics model.

**Who cares.**
- Robotics: one model gives a reactive policy *and* a plannable model; planning on top of the
  policy's own latent buys success where the policy has headroom (drawer +6 to +13, toolhang
  planner rows) and costs nothing where it is saturated.
- Representation learning: the joint objective learns a state that decodes actions and
  discriminates the expert's action from wrong ones (E4), uses more of its dimensions under
  SIGReg (E4), and stays useful under perturbation (E5).

| claim | evidence | status |
|---|---|---|
| differences are significant, not seed noise | E2 | script + inventory |
| baselines are trained the way their papers say | B1, B2 | audit needed |
| the flow head models the action distribution, not the mean | E3 | to run |
| the latent is action-aware; SIGReg uses more directions | E4 | probes exist |
| SIGReg improves robustness / generalization | E5 | to run |
| the equal loss weighting is justified | E6 | to run |
| success vs goal distance, vs LeWM | E7 | to run |
| end-to-end beats a frozen pretrained encoder | E1 | on hold |

## E2. Significance tests and baseline coverage (first, no GPU)

**What exists.** Every row has 3 eval seeds x 50 episodes; per-episode outcomes are in the
logs (`episode_successes` arrays), per-seed rates in the board json
(`docs/results/wave1/board_*.json`, `scripts/collect_board.py`). The sampler is deterministic
given the eval seed (`gip.py:sample_eval_episodes`), so two arms evaluated with the same seeds
face the **same 150 (episode, start) tuples**: the comparison is paired. Log roots: ours
`ckpts/jf_grev`, `ckpts/jointflow_tc`; DP `ckpts/wf8_dp/*_eval`; LeWM `ckpts/lewm_repro`,
`ckpts/lewm_ours*`; gcidm `ckpts/gcidm_repro*`; Minghao's suite `baselines/ckpts/<cell>/`
(dinowm, dinowm10, frozen_dinov2, frozen_dinov2_patch, gcbc, gcidm_gr, gcidm_tc, off50
variants; no dp, no bcrnn).

**Tests** (report all three; the paired one is the headline):
- episode level, paired: McNemar's exact test on the 150 paired outcomes;
- seed level: Welch t-test on the 3-vs-3 seed rates plus Cohen's d (low power at n=3; a
  non-significant result here is not evidence of no effect);
- episode level, unpaired fallback: two-proportion z-test / Fisher exact, for rows with
  per-seed rates only.

**Script** `scripts/sig_tests.py` (new, CPU). Built to take any checkpoint's log root and
its dataset, so it applies to future arms as they land; tested first on the current numbers.
```
--inventory: walk the log roots; print per (cell, protocol, mode) which arms have
             per-episode outcomes, under which seeds, and which have per-seed rates only
--compare A B: for each (cell, mode) with common seeds:
    concat outcomes over seeds -> a, b in {0,1}^150
    mcnemar_p = exact binomial test on the discordant pairs (b01, b10)
    welch_p, cohen_d from the 3 seed rates
    row: cell, mode, A mean+-std, B mean+-std, delta, mcnemar_p, welch_p
```
**Deliverable.** A p-value column in every results table; a footnote on the paired design;
the coverage matrix (the authoritative gap list).

## B1. DP audit and retrain (high priority, not an experiment)

**What we actually ran (verified 2026-09-09).** Entry `wf8_dp_cell_v6c_0d2dcfc.sh`: the
official diffusion_policy repo, `train_diffusion_transformer_hybrid_workspace.yaml` (DP-T),
task yamls `dp_task_yamls/<cell>_ours_image_noprop.yaml` (obs = one `[3,224,224]` agentview,
no proprio), `policy.crop_shape=[202,202]`, `training.num_epochs` = the cell's budget (toolhang
120), `checkpoint_every=2`, workspace defaults otherwise (EMA on). Runs die at the 3 h wall
(~15-17 epochs per link) and resume by chain. Board artifacts: toolhang
`wf8_uni/toolhang_dp_noprop/snap_ep120.ckpt` (task-only 68.0 +- 9.2, old protocol 71.0),
drawer `wf8_dp/drawer/latest.ckpt` = the ep-120 artifact (task-only 2.0), transport epoch
snapshots (old protocol 84.7; task-only never completed), cube partial (declined: not
goal-conditioned). No BC-RNN checkpoint exists anywhere.

**Deviations from the paper to audit, item by item, against the official configs and the
paper appendix:** (1) DP-T vs DP-C: the paper recommends the CNN variant for new tasks and
DP-C is at least as good on the image benchmarks; (2) observation: paper = task cameras
(toolhang sideview + wrist at 240, transport 4 cameras at 84) + eef pos/quat/gripper, ours = one
agentview at 224, no proprio; (3) crop: ours 202 on 224, theirs per task config; (4) budget:
paper trains image tasks for 3000 epochs, ours 120; (5) checkpoint selection: the paper's "max"
is the best of periodic evaluations during training (optimistic), "avg of last 10" is the
honest column, ours is one EMA snapshot at the budget with no selection; (6) n_obs_steps /
horizon / n_action_steps: workspace defaults (2 / 10 / 8), confirm; (7) eval protocol: paper
rollouts vs our task-only 3 x 50 on the re-rendered 200-demo data.

**Owner direction (2026-09-09): use the OFFICIAL released checkpoints and run them through
our eval, so no recipe is replicated.** What is released (verified):

| source | task | what | obs it expects | reported |
|---|---|---|---|---|
| robomimic model zoo v0.1 | tool_hang PH | BC-RNN image `tool_hang_ph_image_epoch_440_succ_74.pth` | sideview + wrist 240 + eef/gripper | ~50-74% |
| robomimic model zoo v0.1 | transport PH | BC-RNN image `transport_ph_image_epoch_580_succ_70.pth` | 2 shoulder + 2 wrist 84 + eef/gripper | ~70% |
| DP experiments site | tool_hang_ph | `diffusion_policy_cnn/train_{0,1,2}/checkpoints/epoch=2150-test_mean_score=0.955.ckpt` (+ transformer, lstm_gmm, ibc_dfo folders) | as above, 240 | 0.955 (train_0) |
| DP experiments site | transport_ph | `diffusion_policy_cnn/train_0/checkpoints/epoch=2750-test_mean_score=1.000.ckpt` | as above, 84 | 1.000 (train_0) |
| DexMimicGen | drawer, transport | **no policies released**, datasets only (HF `MimicGen/dexmimicgen_datasets`) | | |

(URLs: `downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/<task>/bc_rnn/`,
`diffusion-policy.cs.columbia.edu/data/experiments/image/<task>/<method>/train_N/checkpoints/`;
DP checkpoints are 4 GB each, three training seeds per method.)

**Observation contract of the official DP-C toolhang checkpoint (its `config.yaml`, verified):**
obs = `sideview_image` [3,240,240] + `robot0_eye_in_hand_image` [3,240,240] + `robot0_eef_pos`
[3] + `robot0_eef_quat` [4] + `robot0_gripper_qpos` [2]; crop 216; n_obs_steps 2, horizon 16,
n_action_steps 8; EMA; 3000 epochs; dataset `tool_hang/ph/image_abs.hdf5` with **absolute
actions** (action dim 10 = position 3 + rotation-6d 6 + gripper 1). So DP-C visual DOES use
proprio and the wrist camera, and it drives an absolute-pose OSC controller, not the delta
controller our data and env use. The BC-RNN model-zoo image checkpoints likewise take both
cameras plus eef pos/quat/gripper, with delta actions (v0.1 era).

**Can they run scene-only or without proprio? No.** The input contract is fixed: missing
keys do not load, and placeholder inputs (zeros for the wrist image or proprio) are far
out-of-distribution, so the result would be a failure that measures nothing. The official
checkpoints give one row only: "as published, own observations and controller". The
"same information as ours" row (one scene camera, no proprio) requires training their
method on our observation setup, which is what our DP-T rows already approximate at a low
budget. Both rows are worth reporting, and retraining is needed only for the second.

**Adapter additions for DP-C.** Build the env with `control_delta=False` for the OSC_POSE
controller and convert the 10-d (pos, rot6d, gripper) output to (pos, axis-angle, gripper) as
DP's `RobomimicImageRunner` does; the task predicate and start tuples are unchanged.

**Adapter to our eval (design, no training).** Our envs are the same robosuite tasks
(toolhang from the robomimic env_args, env_version 1.4.1; transport from the recorded
TwoArmTransport meta, all five cameras present), and the wrappers already expose eef
pos/quat/gripper proprio. The adapter renders the checkpoint's cameras at its resolution,
builds its obs dict, calls the policy (robomimic `RolloutPolicy` for BC-RNN; the DP workspace
policy as `eval_dp_toolhang.py` already does, with the robomimic-0.3 shims), and steps our
env on our start tuples with our task predicate:
```
for each (episode, start) tuple:  env.reset_to(state)
    loop: obs = {cam_i: env.render(camera_name=cam_i, h, w) for cam_i in ckpt.cameras}
                | {eef_pos, eef_quat, gripper_qpos (both arms for transport)}
          a = policy(obs)  (BC-RNN: per-step; DP: n_action_steps chunk)
          env.step(a); stop on task success or budget
```
Caveats to state in the paper: (a) their checkpoints were trained under robosuite 1.2
physics, ours runs 1.4/1.5, so their number under our sim can shift (cross-check by running
their own eval script in the `dp_env` on the same tuples); (b) our task-only protocol resets
to demo initial states, the papers reset with random inits; (c) transport: their policies
were trained on the 200 PH demos, our model on the 1029-episode set (data advantage on our
side; an apples-to-apples row would train ours on PH-200); (d) the DP "max" checkpoint is
selected on its test score, so also run `latest.ckpt`.

**Drawer (no official policy).** Train BC-RNN with DexMimicGen's own `generate_training_config.py`
on their hdf5 (their cameras at 84 px, their horizon 550): their script, their data, a few
card-hours; the closest thing to an official checkpoint. Same adapter for eval.

**Our own DP-T rows.** Keep as "budget-matched, our observation setup" with the audit table
beside them, or drop them once the official rows exist (owner's call).

**Deliverable.** Baseline rows from official checkpoints (toolhang BC-RNN, DP-C x3 seeds;
transport BC-RNN, DP-C), drawer BC-RNN from their script, all under our protocol with +- std
and E2 p-values, plus the one-page audit table.

## B2. BC-RNN

Covered by B1: official robomimic checkpoints for toolhang and transport, DexMimicGen's
script for drawer. Nothing to train on the robomimic cells.

## E3. Does the flow head recover the action distribution? (after B1)

**Why a toy.** On real data we observe one expert action per state, so there is no per-state
distribution to compare against; on a toy we know p(a | s) exactly. Follow the IBC / DP
papers' multimodal toy: a two-branch fork where the demonstrator goes left or right with
equal probability at a symmetric state. An MSE head predicts the mean (a mode that does not
exist); a distribution head puts mass on both.

**Toy.** State s in [-1,1]^2; expert action a | s from a 2-3 mode mixture whose weights
depend on s, plus the fork. No images: `MoTFlow.loss` and `MoTFlow.sample` take `z_history`
directly, so the toy feeds z = s (padded) through a tiny trunk and uses the real flow action
head. Arms: flow head, MSE action head (the mirror construction in the repo), truth. Metrics:
energy distance and 2-Wasserstein between K=256 samples and the true conditional at 50 test
states; mode coverage (true modes holding >= 5% of samples); a figure. Sweep `n_flow_steps`
in {1, 2, 4, 8, 16}.

**Real data (PushT): exact likelihood under the flow (for review before coding).**
The action head is a conditional flow: sampling integrates da/dtau = v(a, tau | c) from
a_0 ~ N(0, I) at tau = 0 to the action chunk at tau = 1 (Euler, `motflow.py:463-466`), with
c = (z_history, goal token, h_norm) and a in the model's z-scored action space (chunk of 10
steps x 2 dims = 20-d on PushT). For an ODE flow the density evolves by the instantaneous
change of variables:
```
d/dtau log p_tau(a_tau) = - tr( dv/da (a_tau, tau | c) )
log p_1(a*) = log N(a_0; 0, I) - integral_0^1 tr(dv/da)(a_tau, tau | c) dtau,
              where a_1 = a* and a_tau is obtained by integrating BACKWARD from a*.
```
Pseudocode (per anchor, per candidate chunk a*):
```
a = zscore(a*); logdet = 0; n = 64 (finer than the 8 sampling steps: the likelihood
belongs to the continuous flow)
for i in reversed(range(n)):
    tau = (i + 1) / n
    v = head(a, tau, c)
    tr = sum_j e_j^T (dv/da) e_j       # 20 exact JVPs (torch.func.jvp), or Hutchinson
    a = a - v / n ; logdet += tr / n
log_p = log N(a; 0, I) - logdet
```
Candidates per anchor, as in `probe_wm_discrim.py`: expert a*, zero, -a*, shuffled from
another anchor, shuffled in-episode, a* + sigma noise, uniform. Report mean and median log p
per candidate and the discrimination accuracy (expert scores highest). Caveat: the velocity
field is only trained near the interpolation path, so log p of far candidates (uniform) can
be numerically wild; report medians and clip. Note that the PushT expert is a scripted
controller with noise, so multimodality there is mild; the fork toy is where the
distribution claim is actually tested.

**Cost.** CPU for the toy; one short GPU job for PushT.
**Deliverable.** Fig: toy samples vs truth (flow vs MSE); PushT log-likelihood table.

## E4. Latent probes and pictures (after B1)

**What exists (all in `scripts/`).** `probe_wm_discrim.py` (expert action vs zero / negated
/ shuffled / perturbed / uniform, for LeWM, unified and jointflow), `probe_jf_dynamics.py`
(counterfactual dynamics: match_top1 / match_rank / sensitivity), `probe_jf_cost.py`
(cost-to-goal: imagined vs real progress), `probe_jointflow_latents.py` (spectrum, effective
rank, ridge probes for object poses, proprio, inverse dynamics), `probe_state_factors.py`,
`probe_jf_diversity.py`, `probe_jf_curvature.py`.

**Plan.** Run discrim, counterfactual dynamics, cost-to-goal and the latent probes (pose,
proprio, IDM) on noreg vs vanilla SIGReg on pusht, tworoom, reacher, toolhang, drawer,
transport (both arms exist on every cell). CPU on the devbox for encoder-only probes, one GPU
job for the sampling-based ones.

**SIGReg's contribution, unbiased first (owner).** Primary claim: SIGReg uses more directions.
Metrics on the same held-out frames, eval mode: participation ratio, effective rank
(exp of the spectral entropy), number of principal components for 90% of the variance, and the
eigenvalue spectrum plotted for both arms; report r_eff / action_dim and r_eff / state_dim as
the cheap size comparisons. Secondary, labeled as SIGReg's own objective (biased by
construction): the SIGReg statistic on held-out latents, per-direction kurtosis, QQ plots.
Downstream: the gradient-planner delta (toolhang 76.0 -> 91.3, drawer 12.7 -> 19.3) and the
cost-landscape smoothness from the curvature and cost-to-goal probes.

**Pictures.** t-SNE both unlabeled and labeled. Labels are ground-truth object state from the
h5 `state` column (pusht: block angle, block position; reacher: joint angles;
toolhang/drawer/transport: episode phase = step / length), not proprio, to avoid coloring in
a structure that the claim is not about. Alongside t-SNE, a 2-D PCA projection with equal
axes for the shape argument (t-SNE turns any Gaussian into a blob and any blob into clusters).

**Deliverable.** Table: discrim accuracy, match_top1, cost-to-goal gap, pose/proprio/IDM R^2,
participation ratio, r_eff, PCs-for-90% per cell and arm. Fig: spectrum, PCA, t-SNE grid.

## E5. Robustness and generalization with vs without SIGReg (after B1)

**What exists.** swm cells: the env exposes a `variation_space` set at reset
(`stable_worldmodel/envs/pusht/env.py:77-112, 209`): `background.color`, `agent.color / scale
/ shape`, `block.color / scale / shape / angle`, rendering flags; only start positions and
block angle vary by default. tworoom / reacher / cube: check the same way. robomimic and
dexmimicgen cells: nothing equivalent in our wrappers; pixel-level perturbations only for now
(owner: "for now, this is the best we can do").

**Design.** Same checkpoints, same 150 tuples, mean +- std, reactive and best-of-K:
- env-level (swm cells): background color shift, agent/block color shift, block scale x1.2;
- pixel-level (all cells): Gaussian noise sigma in {0.05, 0.10}, brightness +-20%, one random
  32 px occluding square;
- action noise: executed action + N(0, 0.1^2);
- goal distance: E7's offsets 50 and 100 on pusht and tworoom.

**Code change (eval only).** A frame transform in `scripts/eval_gip.py` before encoding,
selected by `EVAL_OBS_PERTURB=noise:0.1|bright:0.2|occl:32`; an action-noise hook
`EVAL_ACT_NOISE=0.1` after the policy/planner output; for swm cells
`EVAL_VARIATION=background.color=<rgb>,...` applied through `variation_space` at reset.
Verification: with everything unset the board numbers reproduce bit-for-bit.

**Cells and cost.** pusht, tworoom, reacher (env + pixel), toolhang (pixel). Eval-only,
~2-3 h per cell per perturbation set on one card.
**Deliverable.** Degradation curves noreg vs SIGReg per perturbation; a table of the drop.

## E6. Loss weighting between dynamics and policy (after E3-E5)

**What exists.** `loss = loss_action + loss_state` (equal weight); `--grad_probe_ema` logs
per-loss gradient norms on the encoder; `--pcgrad` in three modes (sym, protect_p, match_s).
match_s was run on reacher with no gain and is dropped.

**Design (owner: rescale, not project; a `w_state` sweep is too expensive).** New flag
`--grad_balance {off, d_to_p}` (not under `--pcgrad`, which is a projection): each step,
rescale g_D on the shared parameters (encoder + trunk) so ||g_D|| = ||g_P||; no projection.
Reuses the per-task-backward machinery in `lewam/models/pcgrad.py`; match_s can migrate to
this flag later as `s_to_p` for consistency. One cell (pusht or reacher), noreg recipe,
reactive + best-of-K + gradient, 3 seeds. Free companion: the logged ||g_D|| / ||g_P|| ratio
over training at equal weight, if the probe lines exist in the current run logs (to check).
**Cost.** 1-2 trainings + 6 evals.
**Deliverable.** One table row (equal weight vs d_to_p) and the ratio figure.

## E7-prep. Long-horizon planning protocol (agreed by the owners 2026-09-09; LeWM side = Minghao, LeWAM side = here)

**Agreed protocol.** Roll the dynamics the ENTIRE distance to the goal and score the final
imagined state against the goal (a direct test of dynamics drift); replan every 25 raw steps,
executing all 25 actions of each plan, exactly as LeWM does. LeWM's evaluator is NOT modified:
Minghao runs it as is (`plan_config.horizon = H/5`, `receding_horizon = 5`, last-step
`GoalMSE`). Fallback if the long rollouts fail: an intermediate cost every 25 steps (score the
rollout at each 25-step waypoint); designed only if needed.

**LeWAM mirror, flags only, no model or planner code.** `plan_rollout = H/5`,
`exec_actions = 25`, `plan_goal_time = false` (last-block cost, as LeWM), K = 32 proposals;
modes best-of-K and gradient-TR; arms fx_nm192 and fx_vsig192; seeds 42, 0, 1 x 50. Entry:
`jf_grev_42560ab_x.sh` = the GR eval entry with `${XARGS:-}` appended to the eval_gip line
(`++gip_eval.exec_actions=25 ++gip_eval.plan_goal_time=false`, plus
`eval.goal_offset_steps=H eval.eval_budget=2H` for the grid); tags get the suffix `ol25` so the
board reads them as their own arm. Optional secondary row (cheap): `plan_goal_time = true`,
i.e. the cost taken at the goal time at every replan, to show what the fixed-horizon choice
costs us.
- Run A, standard protocol (H = 25, budget 50): `plan_rollout` 5, exec 25. References: LeWM
  88.7 (A2, its own evaluator), LeWM through eval_gip 88.0, ours closed-loop exec 5: best-of-K
  80.7, gradient 86.7.
- Run B, grid H in {50, 75, 100}, budget 2H: `plan_rollout` 10 / 15 / 20, exec 25.
- Cost: per replan ~0.55 s x H/25 (best-of-K) and ~2.8 s x H/25 (gradient-TR), at most 8
  replans per seed; the whole grid is under one card-hour per arm. Two jobs (one per arm).

**Sampler finding (2026-09-09, `sampler_ab.py` on `wf8/train/pusht.h5`): there are TWO LeWM
evaluators and they draw different tuples.** le-wm's original `eval.py` (github
lucas-maes/le-wm) and our port `scripts/eval.py:150` draw `g.choice(len(valid) - 1, 50)`, the
same as `gip.py:sample_eval_episodes:131`; the A/B of 2026-08-07 (memory
"lewm-eval-equivalence") verified identical tuples between those, and A2's 88.7 was produced by
that path (`lewm_eval_full.sh` runs `eval.py` from the job dir), so A2 IS paired with every
eval_gip row. stable-worldmodel's newer `scripts/plan/eval_wm.py` (the copy under
`code/lewm_main_eval/stable-worldmodel`) draws `g.choice(len(valid), 50)` instead: same seed,
same valid mask, one different range, only 23/50 tuples in common at H = 25 and 50, 19/50 at
H = 100. The `- 1` is an off-by-one (the last valid row can never be picked) that le-wm's
eval.py and our ports share; the swm script dropped it.
**What this means for Minghao's runs:** if he runs le-wm's `eval.py` (or our
`scripts/eval.py`), nothing changes on our side and every row is paired. If he runs
stable-worldmodel's `eval_wm.py`, our comparison rows must drop the `- 1` in `gip.py` (one
line) to share his tuples, and the two closed-loop PushT reference rows should be re-run on
those tuples (minutes) so open-loop vs closed-loop stays paired. Ask him which script; it is
the only thing that decides pairing.

**Coordination with Minghao (both evaluators must agree on all of these):** which evaluator
script (above); seeds 42, 0, 1; H in {25, 50, 75, 100} with budget 2H; the default
random-start sampler on both sides (frame-0 starts only if both evaluators implement them);
`num_eval` 50; dataset = the train h5 as on the board; last-step cost.

**Launch gate.** Runs A and B wait for the evaluator answer. Compute is minutes; pairing is
the point.

## E7. Success vs distance to goal on PushT, vs LeWM (ON HOLD until the protocol is settled)

**What exists.** PushT checkpoints (fx_nm192, fx_vsig192) evaluated at offset 25 / budget 50
from random starts. Offsets are a hydra override (`eval.goal_offset_steps=N
eval.eval_budget=2N`). The authors' LeWM release (`code/lewm_main_eval/hf_release_native/pusht`)
was evaluated on our protocol with LeWM's own `stable-worldmodel/scripts/plan/eval_wm.py`
(`lewm_eval_full.sh`, run A2: 94/90/82 = 88.7 at offset 25). Its start sampler is the same
logic as ours (`max_start = len - offset - 1`, `rng(seed).choice` over valid rows), it sets
`max_episode_steps = 2 * eval_budget`, and asserts plan horizon x action block <= budget.

**Design (owner: start = frame 0, goal at H in {25, 50, 75, 100}, 50 episodes).** For each of
50 episodes per seed, start at frame 0 and place the goal at frame H; the same 50 episodes serve
all four H, so the grid is paired within episode. Every PushT episode is 109 steps, so frame 0
always has all four goals. Budget 2H (50, 100, 150, 200). Seeds 42, 0, 1. Modes, planning
only: ours = best-of-K and gradient-TR (fx_nm192 and fx_vsig192); LeWM = CEM 300 x 30, top-30.
No reactive row (the point is planning). Note the grid's H = 25 point is a new measurement
(frame-0 starts), not the board's random-start offset-25 row.

**Matched planning protocol (verified from both code paths, 2026-09-09).** As run so far the
two planners differ: LeWM (A2, `plan/config/pusht.yaml`) looks 5 blocks = 25 steps ahead with a
last-step latent MSE cost and `receding_horizon` 5, i.e. it executes the whole 25-step plan
open-loop and re-solves only if the goal is not reached (2 solves per 50-step budget in the A2
log); ours looks 5 blocks ahead with a terminal-latent cost, executes 1 block and replans every
5 steps, with the policy proposals conditioned on the remaining time (`_steps_left` -> h_norm)
while the imagination length stays fixed. Replanning LeWM every block with its fixed 5-block
horizon collapses it (88.7 -> 30.0, RECIPES:919-923) because every replan re-aims 25 steps out.
Neither planner imagines all the way to a goal beyond 25 steps. The grid therefore runs BOTH
planners under BOTH execution rules, with the lookahead tied to the goal distance:
- Row A, open-loop: horizon = H/5 blocks (5, 10, 15, 20), plan once, execute the whole plan,
  re-solve only if the goal is not reached. LeWM `plan_config.horizon=H/5
  receding_horizon=H/5`; ours `plan_rollout=H/5 exec_blocks=H/5`. This is LeWM's published
  rule generalised to the distance, and the pure test of long-range imagination.
- Row B, closed-loop: replan every block with the lookahead shrunk to the remaining distance,
  `horizon_t = min(H/5, ceil(steps_left/5))`, warm-started from the rest of the previous plan.
  LeWM `receding_horizon=1` plus a per-replan horizon (small change in our copy of
  `stable_worldmodel/policy.py`); ours `exec_blocks=1` plus the same shrink of `plan_rollout`
  (small change in `JointFlowPlanPolicy`). Same cost (terminal latent to the goal) on both.
Verification before launch: row A at H = 25 reproduces A2's 94/90/82 for LeWM (regression);
the shrink code prints horizon_t per replan on one dry episode; with the new flags unset the
board numbers reproduce.
The remaining difference is the candidate source, policy proposals (K = 32, or proposal +
gradient with trust region) vs CEM from scratch, which is the method itself and is stated as
such; our WM can also run CEM from random candidates as a like-for-like solver row if wanted.

**Sampler change, mirrored in both codebases (`gip.py:sample_eval_episodes` and LeWM's
`eval_wm.py`), design for review:**
```
start_zero = cfg.gip_eval.get("start_zero", False)          # LeWM: cfg.eval.start_zero
if start_zero:
    valid = np.nonzero((step_idx == 0) & (max_start_per_row >= 0))[0]   # frame 0 of every
    # episode long enough for this H; on PushT that is every episode for H <= 108, so
    # `valid` and hence the rng picks are identical across H -> paired grid
    picks = np.sort(valid[rng(cfg.seed).choice(len(valid), size=num_eval, replace=False)])
    return episodes[picks], starts = zeros, None            # goal at +goal_offset_steps as usual
```
Verification: (1) both samplers print the same 50 episode ids for seed 42 at every H;
(2) with the flag off, the current board tuples reproduce (regression); (3) one dry episode
per H shows goal frame index = H.

**Cost (measured per-REPLAN times on PushT, 50 envs batched; both sides replan per block,
not per step):** gradient-TR 2.8 s and best-of-K 0.55 s per replan at 5 blocks, scaling with
the lookahead; LeWM CEM ~17 s per solve at 5 blocks. Row A: a handful of solves per seed for
everyone, minutes. Row B with the shrinking horizon: ours ~10 min per seed (gradient-TR) and
~2 min (best-of-K) over all four H, about 1.5 h for both arms and three seeds; LeWM ~45 min per
seed, about 2.5 h for three seeds. Whole grid ~5 card-hours, 6-8 jobs.

**On "generalization" (owner question).** Yes. What H_max changes is only the distribution
of goal tokens seen in training (goals are sampled from the same trajectory, so the action
targets at a state are identical whatever the offset; the dynamics target is always the next
state). So offsets beyond 50 are out-of-distribution for the **goal conditioning**, not for
the action or dynamics heads, and retraining with a larger H_max is not expected to help.
Label the region beyond 50 as "beyond the trained goal horizon".

**What the PushT data is (verified from `wf8/train/pusht.h5`, 2026-09-09).** 18,685 episodes
of exactly 109 steps; the green target T is at the same fixed pose in every episode; block and
agent starts vary (with repeats); endpoints vary, and several episodes never reach the target
within 109 steps or push the block away from it. So the demonstrator's intent is
state-determined (push toward the fixed T) but the trajectories are noisy, and the eval goal
(the frame +H ahead of a random start, i.e. start = goal - H) tells the policy which
continuation this episode took. Both effects are in the curve: goal-conditioning OOD beyond
50 and a longer closed-loop horizon; LeWM faces both identically, so the comparison is fair.

**Deliverable.** Fig: SR vs goal distance H, ours (best-of-K, gradient-TR; noreg and SIGReg)
vs LeWM CEM, mean +- std over 3 seeds, paired McNemar p-values per H from E2.

## E1. End-to-end vs frozen pretrained encoder (on hold)

**Facts checked.** The `dinov3` backbone is frozen by construction (`module.py:483-485`);
its frozen path returns one pooled vector; patch tokens would come from `forward_features`.
Owner: patches are necessary (CLS alone lacks visual information), no trainable pooling, and
this is not a DINO-WM re-implementation: the model keeps the MoT and the policy and only
predicts frozen DINO patches as its state. Cost is the concern (196 tokens per frame in the
trunk and in planning). **Already available:** Minghao's baseline suite contains
`frozen_dinov2_patch` and `frozen_dinov2_patch_tc` rows on toolhang, drawer, pusht and others
(LeWAM-unified lineage: `encoder_backbone dinov2`, `encoder_token_mode patch`, z_dim 192,
w_reg 0.04, mse head). Those are the frozen-patch WAM baseline on the previous architecture
and can be cited now; the MoT version waits.

## Decisions still open

1. Which LeWM evaluator Minghao runs: le-wm `eval.py` (tuples already paired with ours) or
   stable-worldmodel `eval_wm.py` (then drop the `- 1` in `gip.py` for the comparison rows and
   re-run the two closed-loop reference rows). Runs A and B launch on that answer.
2. Start rule for the grid: default random starts on both sides, or frame-0 on both.
3. B1: which official checkpoints first (toolhang DP-C x3 seeds + BC-RNN, then transport),
   and whether to keep our budget-matched DP-T rows as the "same information" row.
4. E5: pixel-level only for robomimic/dexmimicgen (as agreed) or build the robosuite
   domain-randomization path later.

### E7-prep addendum: what the second plan sees (owner question 2026-09-09, verified in gip.py)

The time-to-goal counter is exact: each replan subtracts the executed raw actions in obs-steps
(`_steps_left -= take / action_block`, `gip.py:1206`), so after a 25-step plan at H = 25 it reads
1 block. Run A used `plan_goal_time=false`, so its second solve scored the LAST imagined block
(step 50) like LeWM, while the proposals were conditioned on "1 block left". Options going forward:
- Mirror (run A, done): exec 25 for every plan, last-block cost. LeWM's rules with our proposals.
- Actual distance on replan, no code: `plan_goal_time=true` scores the remaining-distance block
  (`_goal_idx`, `gip.py:1776-1782`). Consistent with exec 25 at every replan before the goal time
  when H >= 50 (at H = 50 the second solve scores block 5 = the 5 executed blocks). At H = 25 the
  second solve is past the goal time (counter clamps to 1) and exec 25 would run 20 unscored actions.
- DECISION (owner 2026-09-09): run A IS the LeWM match (same plan length, same execute-all,
  same last-block cost, same second try); no hybrid variant. The head-to-head reports our
  closed-loop replan-every-5 rows as the method and the run-A mirror row as the control, which
  already beats LeWM on its own scheme (SIGReg gradient-TR 91.3 vs 88.7).
