# Paper robustness plan

Revision 5 (2026-09-10). One section per experiment: motivation, experiment, result,
interpretation. Run-by-run history, job ids and per-seed values live in RECIPES; artifacts in
`docs/results/`. Order of work (owner): E7 done, E3 and E4 in progress, B1 and E2 with Minghao,
then E5, E6, E1 last.

## TODO

- [ ] **B1** Run the official DP-C (three seeds) and BC-RNN toolhang checkpoints through our
      eval with the observation/controller adapter; then transport; train drawer BC-RNN with
      DexMimicGen's script; write the one-page audit table. (owner + Minghao)
- [ ] **B2** Nothing separate: covered by B1.
- [ ] **E1** On hold, lowest priority (frozen-DINO patch baseline on the MoT).
- [ ] **E2** Move the paired McNemar test from scratch into `scripts/sig_tests.py` with the
      coverage inventory; add p-values to every results table. (Minghao helping)
- [x] **E3a** Two-door fork toy done (figure approved: `docs/results/e3/v2/rollouts_box1px.png`,
      caption in the section). Optional: a second training seed for the flow head's door lean.
- [x] **E3b** PushT chunk likelihood under the flow head done, shown as preference shares and
      ranks (`e3b_pusht_preference.png`, `e3b_pusht_ranks.png`). Optional: toolhang or cube
      (one GPU job each, minutes).
- [ ] **E4** Wave 1 done (pusht, toolhang). Optional: more seeds per arm; drawer, transport,
      reacher once their training files are on the devbox.
- [ ] **E5** Perturbation eval hook (pixel noise, brightness, occlusion, action noise, env
      variations on swm cells); not started.
- [ ] **E6** Gradient rescaling flag `--grad_balance d_to_p`, one cell, three seeds; not started.
- [ ] **E7** Done for LeWAM on PushT. In progress (ours; Minghao could not run LeWM): the LeWM
      row on the same episodes with per-plan timing records and separate batch-1 runs (five
      seeds); both smoke cells passed 2026-09-10, the grids are next. Open: other cells, GC-IDM row.

## Story

**Claim.** LeWAM trains one encoder, one latent and one trunk end-to-end from pixels with a
dynamics head and a flow action head; the same latent serves the reactive policy and the
planners. DINO-WM and LeWM freeze or pretrain the encoder and add a planner afterwards;
policy-only methods (DP, BC-RNN) have no dynamics model.

| claim | evidence | status |
|---|---|---|
| baselines are trained the way their papers say | B1, B2 | audit pending |
| end-to-end beats a frozen pretrained encoder | E1 | on hold |
| differences are significant, not seed noise | E2 | paired tests used for E7; repo script pending |
| the flow head models the action distribution, not the mean | E3 | done (toy + PushT likelihood ranks) |
| the latent is action-aware; SIGReg uses more directions | E4 | done on pusht and toolhang |
| SIGReg improves robustness | E5 | not run |
| the equal loss weighting is justified | E6 | not run |
| success vs goal distance | E7 | done (ours); LeWM row pending |

## B1. Baseline audit and official checkpoints

**Motivation.** Our DP rows were trained with a recipe that deviates from the paper in
several ways, and no BC-RNN row exists. The owner's decision: run the officially released
checkpoints through our eval so no recipe is replicated; retrain only for a row that matches
our observation setup.

**What we ran.** Official diffusion_policy repo, DP-T workspace, one 224 agentview image, no
proprio, crop 202, 120 epochs at the 3 h wall with resume chains, EMA snapshot at the budget.
Board artifacts: toolhang `wf8_uni/toolhang_dp_noprop/snap_ep120.ckpt` (task-only 68.0 +-
9.2), drawer `wf8_dp/drawer/latest.ckpt` (task-only 2.0), transport epoch snapshots (old
protocol 84.7; task-only never completed).

**Deviations to audit.** (1) DP-T instead of DP-C; (2) one agentview at 224 and no proprio
instead of the task cameras (toolhang sideview + wrist at 240, transport four cameras at 84)
plus eef pose and gripper; (3) crop 202 on 224 instead of the task config; (4) 120 epochs
instead of 3000; (5) one final EMA snapshot instead of the paper's "max" (best periodic
eval) or "avg of last 10"; (6) n_obs_steps / horizon / n_action_steps at workspace defaults;
(7) our task-only 3 x 50 protocol on re-rendered demos vs the paper's rollouts.

**Released checkpoints (verified).**

| source | task | checkpoint | observations | reported |
|---|---|---|---|---|
| robomimic model zoo v0.1 | tool_hang PH | BC-RNN image `tool_hang_ph_image_epoch_440_succ_74.pth` | sideview + wrist 240, eef, gripper | 50-74% |
| robomimic model zoo v0.1 | transport PH | BC-RNN image `transport_ph_image_epoch_580_succ_70.pth` | 2 shoulder + 2 wrist 84, eef, gripper | ~70% |
| DP experiments site | tool_hang_ph | `diffusion_policy_cnn/train_{0,1,2}/.../epoch=2150-test_mean_score=0.955.ckpt` | as above, 240 | 0.955 (train_0) |
| DP experiments site | transport_ph | `diffusion_policy_cnn/train_0/.../epoch=2750-test_mean_score=1.000.ckpt` | as above, 84 | 1.000 (train_0) |
| DexMimicGen | drawer, transport | no policies released, datasets only | | |

URLs: `downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/<task>/bc_rnn/`,
`diffusion-policy.cs.columbia.edu/data/experiments/image/<task>/<method>/train_N/checkpoints/`
(4 GB per DP checkpoint).

**Experiment.** The official DP-C toolhang checkpoint takes sideview and wrist images at
240, eef position, quaternion and gripper, crop 216, and outputs absolute-pose actions (10-d:
position, rotation-6d, gripper) for an OSC controller; BC-RNN takes both cameras plus proprio
with delta actions. Neither runs scene-only or without proprio (missing keys do not load,
placeholders are far out of distribution), so the official checkpoints give exactly one row,
"as published, own observations and controller". The adapter renders the checkpoint's cameras
from our robosuite envs (toolhang from the robomimic env_args, transport from the recorded
meta; all cameras present), builds its obs dict, and steps our env on our start tuples with
our task predicate:
```
for each (episode, start) tuple:  env.reset_to(state)
    loop: obs = {cam: env.render(cam, h, w) for cam in ckpt.cameras} | {eef_pos, eef_quat, gripper}
          a = policy(obs)          # BC-RNN per step; DP an n_action_steps chunk
          env.step(a)              # DP-C: control_delta=False, rot6d -> axis-angle as RobomimicImageRunner
          stop on task success or budget
```
Caveats for the paper: their checkpoints trained under robosuite 1.2 physics, we run
1.4/1.5 (cross-check with their own eval script on the same tuples); our protocol resets to
demo initial states, theirs to random inits; transport policies saw 200 PH demos, ours 1029
episodes; the DP "max" checkpoint is test-selected, so also run `latest.ckpt`. Drawer: train
BC-RNN with DexMimicGen's `generate_training_config.py` on their hdf5 (their cameras at 84,
horizon 550), a few card-hours, same adapter. Keep our DP-T rows as "budget-matched, our
observation setup" or drop them once the official rows exist (owner's call).

**Result.** Not run.

**Deliverable.** Official-checkpoint rows for toolhang and transport (BC-RNN, DP-C x 3 seeds),
drawer BC-RNN from their script, all under our protocol with +- std and E2 p-values, plus the
audit table.

## B2. BC-RNN

Covered by B1: official robomimic checkpoints for toolhang and transport, DexMimicGen's script
for drawer. Nothing to train on the robomimic cells.

## E1. End-to-end vs frozen pretrained encoder (on hold)

**Motivation.** Show that training the encoder jointly beats a frozen pretrained encoder.
**Experiment (owner's constraints).** Frozen DINO patches (not the CLS token) as the state,
no trainable pooling, the MoT and the policy unchanged; cost is the concern (196 tokens per
frame in the trunk and in planning). Minghao's suite already has `frozen_dinov2_patch` rows
on toolhang, drawer and pusht on the previous architecture, citable now.
**Result.** Not run. Lowest priority.

## E2. Significance tests

**Motivation.** Rows differ by a few points at 3 seeds x 50 episodes; the paper needs to say
which differences are real.

**Experiment.** The eval sampler is deterministic given the seed and flags, so two arms with
the same seeds face the same 150 (episode, start) tuples and the comparison is paired. Tests:
McNemar's exact test on the 150 paired outcomes (headline); Welch t-test and Cohen's d on the
3-vs-3 seed rates (low power); two-proportion test for rows with per-seed rates only. Script
`scripts/sig_tests.py`: an inventory of which arms have per-episode outcomes under which seeds,
and `--compare A B` producing cell, mode, means, delta, McNemar p, Welch p.

**Result.** The paired McNemar test exists as scratch scripts and produced the E7 and run A
p-values below; the repo script and inventory are pending. Measured noise floor: the GPU eval
is not bit-deterministic across nodes for the gradient planners and CEM (about 1 episode in 50
flips between runs of the same computation), so one-episode differences are noise.

**Deliverable.** A p-value column in every results table, a footnote on the paired design,
the coverage matrix.

## E3. The flow head models the action distribution, not the mean (closed 2026-09-10; figures approved by the owner)

### E3a. Two-door fork toy (done)

**Motivation.** On real data there is one expert action per state, so the claim "the flow
head recovers the distribution while an MSE head predicts the mean" needs a setting where
p(a | s) is known and bimodal. The IBC paper's 1-D multi-valued toy is the published analogue;
ours keeps the paper's pixels, encoder, trunk and heads.

**Experiment.** Two-room env from stable-worldmodel (224 px, vertical wall at x = 112) with
two doors at y = 72 and y = 152 (half-width 14), the target drawn as a green X at (165, 112)
in every frame, agent radius 7 and speed 5. Demonstrator: a coin picks the door per episode,
independent of the start (exactly 250 per door), unit steps to the door center then to the
target, Gaussian noise sigma 0.25 on every action; starts uniform in x 43.8 to 52.2, y 107 to
117. 500 episodes, 13,571 frames, 23 to 33 steps each, none failed; the two doors are
symmetric in the data to the third digit (episodes, frames, lengths, first actions). Training:
the toolhang TC recipe with SIGReg (z and d 192, depth 4, 10 actions and 1 state predicted,
history 2, lr 1e-4, batch 64, 50 epochs, fp32), flow action head vs mse action head, seed 42.
Eval: 100 closed-loop rollouts per head from the same 100 starts within 1 px of the box center
(48, 112), predicting 10 actions, executing 5, replanning; an episode ends as a crash on the
first wall contact outside a door (the env itself only clamps the blocked coordinate and lets
the agent slide along the wall). Also the first predicted chunk at the box center, 100 samples.
Earlier variants (wider start boxes, the env's own target dot) are in RECIPES.

**Result** (`docs/results/e3/v2/rollouts_box1px.png`, `first_chunks.png`).

| head | reached | via upper / lower door | crashed | timeout |
|---|---|---|---|---|
| flow | 82 | 54 / 28 | 5 | 13 |
| mse | 74 | 0 / 74 | 26 | 0 |

First chunk at the box center: the flow head's 100 samples fan toward both doors (66 up, 24
down, heading std 21 degrees); the mse head's chunk is one line with zero spread, level, into
the wall.

*Caption for the rollout figure.* Two-door fork. Left: the 500 expert demonstrations (250 per
door, door chosen by a coin independent of the start). Middle and right: 100 closed-loop
rollouts of the flow action head and the mse action head from the same 100 starts within 1 px
of the box center; the policy predicts 10 actions from the two most recent frames five steps
apart, executes 5, and replans; an episode ends as a wall collision (red) on the first contact
with the wall outside a door. Both heads share the encoder, trunk, dynamics head and training
recipe (SIGReg, 50 epochs, one seed).

**Interpretation.** From one start the mse head produces one trajectory with jitter: its first
chunk is the mean of the two expert headings, level into the wall, and the replanned chunks
then turn to one door, so the rollouts either crash (26) or all take the same door (74). The
flow head samples the door from the same start and reaches the target through both. Caveats:
the flow head's door weights are not the demonstrations' 50/50 (about two thirds upper for
this seed; the data are symmetric, so the lean is in the trained head, and both toy trainings
used seed 42), and 13 of its rollouts time out by wandering past the target.

### E3b. Which action chunks the flow head finds likely, on PushT (done)

**Motivation.** On real data the flow head can still be tested as a density: it should find
the expert's chunk more likely than plausible alternatives, and its own samples about as
likely as the expert.

**Experiment.** 200 random anchors on `pusht_expert_train.h5` (history frames t-5 and t,
goal frame t+25, goal horizon 5 of H_max 10); candidates per anchor in z-scored action units:
the expert's 10-step chunk, the expert's chunk from another time in the same episode, expert
plus noise at sigma 0.25 / 0.5 / 1, uniform in [-3, 3], and eight of the flow's own samples.
log p by backward Euler integration of the learned velocity from tau 1 to 0 in 64 steps with
the exact Jacobian trace from 20 reverse-mode products per step (verified against the full
Jacobian to 2e-6): a <- a - v / n, logdet += tr / n, log p = log N(a_0; 0, I) - logdet.
The likelihoods are only compared within a state, never reported as values: for each
alternative, the share of states where the head finds the expert chunk more likely, and for
each candidate its likelihood rank among the six candidates at a state. Both pusht arms; one
GPU job, 3.5 minutes.

**Result** (`docs/results/e3/e3b_pusht_preference.png`, `e3b_pusht_ranks.png`; 200 states).

| alternative chunk | states where the expert chunk is more likely, without / with SIGReg |
|---|---|
| expert + small noise (sigma 0.25 in z-scored units) | 95% / 96% |
| expert + medium noise (0.5) | 98% / 98% |
| expert + large noise (1.0) | 100% / 100% |
| the same episode's chunk from another time | 98% / 97% |
| a random chunk | 100% / 100% |

Mean rank among the six candidates (1 = most likely), without / with SIGReg: expert 1.1 / 1.1,
small noise 2.1 / 2.1, medium noise 3.1 / 3.1, large noise 4.5 / 4.4, same episode at another
time 4.4 / 4.5, random chunk 5.9 / 5.9. The expert chunk is rank 1 at 91.5% of states in both
arms; a random chunk is last at 87 to 89%. (The head's own samples rank above the expert chunk
at 83 to 86% of states, as expected for a chunk that carries the demonstrator's noise; they are
left out of the figures.)

*Caption for the preference figure.* For 200 random PushT states (history of two frames five
steps apart, goal frame 25 steps ahead), the flow action head's likelihood of the
demonstrator's 10-step action chunk is compared with that of an alternative chunk under the
same conditioning; bars give the share of states where the demonstrator's chunk is the more
likely one. Likelihoods are exact under the flow (backward integration of the learned
velocity with the Jacobian trace, 64 steps).

*Caption for the rank figure.* Same states and candidates; at each state the six candidate
chunks are ordered by their likelihood under the flow action head (1 = most likely), and each
bubble's area is the share of states at which that candidate takes that rank.

**Interpretation.** The flow head behaves as a density over action chunks: it prefers what the
demonstrator did to the same demonstrator's action at another moment, to noised versions of
it in proportion to the noise, and to random chunks, at nearly every state; its own samples
rank at the top, slightly above the expert chunk, which carries the demonstrator's noise. The
ranking is identical in both arms, so SIGReg changes the latent (E4), not the action head's
density.

## E4. Latent probes: the latent is action-aware and SIGReg uses more directions

**Motivation.** The representation claim needs unbiased evidence that SIGReg changes the
latent (how many directions carry variance) and that the change is useful (object state is
decodable, the dynamics respond to the action), separate from SIGReg's own Gaussianity
objective, which is biased by construction.

**Experiment.** Noreg vs vanilla SIGReg checkpoints on pusht (GR arms) and toolhang (TC arms),
one checkpoint per arm. Geometry and ridge probes on 5000 frames from 500 random episode
windows (train/test split by window): participation ratio, effective rank, principal
components for 90 and 99% of the variance, ridge R² for object pose, block angle, agent
position and inverse dynamics (actions between a frame and the frame five steps later).
Sampling probes on 200 anchors: counterfactual dynamics (imagined next state under the true
action chunk vs 15 alternatives, top-1 match to the real next state), and, on the goal-free
toolhang arms, discrimination (imagined next state under the expert chunk closer to the real
one than under an alternative). Pictures: PCA with equal axes and t-SNE, unlabeled and colored
by object state and by time in the episode, one frame per window.

**Result.**

| metric | pusht without SIGReg | pusht with SIGReg | toolhang without | toolhang with |
|---|---|---|---|---|
| participation ratio (z_dim 192) | 4.7 | 32.7 | 8.2 | 19.8 |
| effective rank | 7.8 | 35.3 | 13.8 | 21.8 |
| PCs for 90% / 99% of variance | 9 / 20 | 30 / 37 | 12 / 60 | 19 / 24 |
| object pose R² (block / tool) | 0.80 | 0.95 | 0.87 | 0.93 |
| block angle R² | 0.36 | 0.82 | | |
| agent position R² | 0.99 | 0.99 | | |
| inverse dynamics R² | 0.79 | 0.81 | 0.79 | 0.88 |
| counterfactual dynamics top-1 (chance 0.06) | 0.88 | 0.985 | 0.77 | 0.985 |
| discrimination vs expert + noise, sigma 0.25 / 0.5 / 1 | | | 0.66 / 0.77 / 0.89 | 0.84 / 0.93 / 0.985 |
| discrimination vs zero / negated / shuffled / uniform | | | 0.97 to 1.00 | 0.99 to 1.00 |
| variance in the first two PCs | 60% | 9% | 39% | 15% |

Figures: `docs/results/e4/latent_pictures_{pusht,toolhang}.png`; numbers in
`docs/results/e4/*.json`.

**Interpretation.** SIGReg spreads the latent over about four times more directions on pusht
and twice as many on toolhang, and the extra directions carry object state (block angle R²
0.36 to 0.82, tool and frame poses up) and make the dynamics respond to the action
(counterfactual top-1 0.77 to 0.985 on toolhang; discrimination against small action noise
0.66 to 0.84). What both arms share: the agent position is read out at 0.99, and the
cost-to-goal and sampling-diversity probes are the same. What not to claim: on these frames
the SIGReg latents are not more Gaussian per principal component (excess kurtosis 0.8 vs 0.5
on pusht, 1.0 vs 0.3 on toolhang; random directions alike), so the claim is "more directions
used", not "more Gaussian". In the PCA panels SIGReg looks stretched along one axis; that is
heavy tails (kurtosis 1 to 3.4) on nearly equal PC1 and PC2 variances, not a variance
imbalance. t-SNE is inconclusive and can go to an appendix.

## E5. Robustness with vs without SIGReg (not run)

**Motivation.** Show the regularized latent degrades more gracefully under perturbation.

**Experiment (design).** Same checkpoints, same 150 tuples, reactive and best-of-K, mean +-
std: env-level on the swm cells through their `variation_space` (background color, agent and
block color, block scale x1.2); pixel-level on all cells (Gaussian noise sigma 0.05 / 0.10,
brightness +-20%, one random 32 px occluding square); action noise N(0, 0.1²) on the executed
action. Code: a frame transform before encoding and an action-noise hook in
`scripts/eval_gip.py`, selected by `EVAL_OBS_PERTURB`, `EVAL_ACT_NOISE`, `EVAL_VARIATION`;
with everything unset the board numbers must reproduce. robomimic and dexmimicgen cells get
pixel-level perturbations only (agreed). Cells: pusht, tworoom, reacher (env + pixel),
toolhang (pixel); about 2-3 h per cell and perturbation set on one card.

**Result.** Not run.

## E6. Loss weighting between dynamics and policy (not run)

**Motivation.** The equal weighting of the two losses is a choice; show that balancing the
gradients on the shared parameters does not change the outcome (or does, and why).

**Experiment (design; owner: rescale, not project).** New flag `--grad_balance {off,
d_to_p}`: each step rescale the dynamics gradient on the shared encoder and trunk so its norm
equals the policy gradient's; reuses the per-task backward in `lewam/models/pcgrad.py`;
`match_s` (run on reacher, no gain) is dropped. One cell, noreg recipe, reactive + best-of-K +
gradient, 3 seeds; plus the logged gradient-norm ratio over training at equal weight.

**Result.** Not run.

## E7. Success vs distance to goal on PushT

**Motivation.** Planning on the latent should degrade gracefully with the goal distance,
and the regularized world model should hold up where the unregularized one fails; the paired
grid also tests the owner's theory that beyond about 75 steps the goal is the same task.

**Experiment (agreed with Minghao).** 50 episodes per seed drawn from the demos with at least
101 frames (13,837 of 18,685), the same 50 for every H, each started at frame 0 with the
goal at frame H, H in {25, 50, 75, 100}, budget 2H. The planner imagines the dynamics to the
goal horizon, scores the imagined state at the goal, executes 25 actions and replans with the
remaining distance; K = 32 proposals; best-of-K and gradient-TR; noreg and SIGReg arms; seeds
42, 0, 1. Flags: `++gip_eval.exec_actions=25 +gip_eval.random_start=false
+gip_eval.min_episode_len=101 eval.goal_offset_steps=H eval.eval_budget=2H`. Episode draw,
for reproduction: eligible episodes in h5 order, picks =
`sorted(rng(seed).choice(n_eligible, 50, replace=False))`.

**Result** (success %, mean +- std over three seeds).

| H | noreg best-of-K | noreg gradient-TR | SIGReg best-of-K | SIGReg gradient-TR |
|---|---|---|---|---|
| 25 | 96.0 +- 3.5 | 93.3 +- 4.6 | 99.3 +- 1.2 | 98.0 +- 2.0 |
| 50 | 22.0 +- 9.2 | 20.7 +- 11.0 | 39.3 +- 9.2 | 71.3 +- 3.1 |
| 75 | 10.0 +- 0.0 | 8.7 +- 1.2 | 26.0 +- 9.2 | 46.7 +- 6.4 |
| 100 | 9.3 +- 4.2 | 8.7 +- 2.3 | 23.3 +- 3.1 | 45.3 +- 6.4 |

Paired McNemar on the same 150 episodes: 50 vs 75 is a significant drop in every cell (p =
0.003, 0.004, 0.004, 0.000); 75 vs 100 is not different in any cell (p = 1.0, 1.0, 0.64,
0.89).

**Interpretation.** H = 25 from frame 0 is the approach phase and not comparable to the
board's random-start rows. The curve drops between 25 and 50 and again between 50 and 75,
then is flat: 75 and 100 are the same task (by frame 75 the block already sits at its
frame-100 pose; only the pusher's parking differs). The arms separate with distance: noreg is
at the floor from 75 with either planner, SIGReg holds 23-39 with best-of-K and 45-71 with
gradient refinement. The pusht arms were trained with goals at most 50 steps ahead, so H >= 75
is out of distribution for the goal conditioning only; label it "beyond the trained goal
horizon".

**Control: LeWAM under LeWM's open-loop scheme at the standard offset** (random starts,
H = 25, budget 50; plan 5 blocks with a last-step cost, execute all 25, one retry):

| arm, planner | closed-loop, replan every 5 | LeWM scheme, execute all 25 | delta | McNemar p |
|---|---|---|---|---|
| noreg best-of-K | 78.0 +- 2.0 | 73.3 +- 5.0 | -4.7 | 0.167 |
| noreg gradient-TR | 78.7 +- 2.3 | 74.7 +- 3.1 | -4.0 | 0.180 |
| SIGReg best-of-K | 89.3 +- 4.2 | 83.3 +- 3.1 | -6.0 | 0.012 |
| SIGReg gradient-TR | 94.7 +- 4.2 | 91.3 +- 2.3 | -3.3 | 0.227 |

LeWM under its own scheme: 88.7. The closed-loop rows are the method, this row the control;
SIGReg gradient-TR beats LeWM under LeWM's own rules. Scoring the retry at the goal time
instead moves at most 2 episodes per 150.

**Open.** The LeWM row on the same grid is ours now (Minghao could not run LeWM). Our port of
LeWM's evaluator (`scripts/eval.py`, the authors' released PushT checkpoint) draws the episodes
above (frame-0 starts, 101-frame filter; a guard calls our sampler and asserts the identical
draw), and both harnesses now record every plan cycle: the whole policy call, the plan phase
alone (proposal to finalized chunk), the envs planned in that call, the imagined length; plus
separate batch-1 runs (`eval.num_eval=1`, five seeds, the same single episode on both sides),
because a batched call's time divided by the batch is throughput, not the time one plan takes.
Smoke cells at H = 25, seed 42: LeWM 72.0
(CEM 300 x 30, 24 s for 50 plans batched), ours 100.0 (SIGReg best-of-K, 0.96 s for 50). The
LeWM grid (11 remaining cells) and the LeWAM grid rerun with records (47 cells) are next; the
timing report (`scripts/plan_timing_report.py`) then gives the time to complete a successful
episode, the time of one 25-step plan, and the per-replan curve at H = 100, in both regimes.
Also open: other cells (tworoom, reacher, pointmaze with their own ladders); GC-IDM as the one
goal-conditioned baseline that can run the ladder; the repo significance script (E2).

## Open decisions

1. E7: whether Minghao's evaluator gets the frame-0 start and the 101-frame filter; which
   other cells and whether GC-IDM joins.
2. B1: which official checkpoints first, and whether our DP-T rows stay as the
   "same-information" row.
3. E5: pixel-level only for robomimic and dexmimicgen (as agreed), or a robosuite
   domain-randomization path later.
