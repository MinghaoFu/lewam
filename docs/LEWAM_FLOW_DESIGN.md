# LeWAM-Flow (v0) — a flow-matching unified world-action model

> Working name **LeWAM-Flow**. Successor line to LeWAM-Seq (scrapped) and LeWAM-Unified
> (aggregator+heads, current). Design derived by reading two released codebases:
> **UWM** (`WEIRDLabUW/unified-world-model`, arXiv:2504.02792) and **DreamZero**
> (`dreamzero0/dreamzero`, arXiv:2602.15922). Status: DESIGN, not implemented. Gate: Minghao's
> fuse-vs-separate call. Author of this doc: design discussion 2026-07-28.

> **PIVOT (2026-07-28, owner):** the generative *state-transition* is DROPPED. Minghao considers the
> current Unified "unified enough," and in latent space (we reconstruct no video) a **deterministic**
> dynamics head is adequate — generative machinery clearly helps only for the **action** distribution.
> **v0 is now: a flow-matching POLICY head on the current deterministic Unified**, replacing the
> deterministic `gc_head` (keep encoder / aggregator / dynamics). NOT the generative DiT described
> below; the UWM / DreamZero material is retained as reference for the flow mechanism and the
> multimodal-action case. Why flow over the GMM head we already tried and failed: GMM failed from
> **ill-conditioning** (mixture weights + predicted variances under NLL → variance collapse / dead
> components), orthogonal to flow-matching's **regression** objective (velocity field, no mixture, no
> variance head) — so GMM's failure does not predict flow's. On PushT (a Diffusion Policy benchmark:
> multimodal but symmetry-solvable) a deterministic head can average the two symmetric pushes into an
> invalid middle action; flow samples one clean mode, so it has a path *above* the 90.8 reactive number
> and is the right head for harder / genuinely multimodal datasets regardless. **Bonus lever:
> flow-matching naturally emits an ACTION CHUNK** (generate `a_t:t+H` in one denoise, execute
> receding-horizon) — worth trying beyond the single frameskip block. Composes with the
> `cem_propose=policy` MPC already landed (sample K action chunks from the flow policy, WM-verify with
> the deterministic dynamics). Full rewrite of the sections below is pending the running r=0.05
> warm-CEM result (which also decides whether generative *dynamics* ever earns a revisit — only if CEM
> shows planning exploited at contact/branch points).

## v0 — LOCKED SPEC (2026-07-28): flow-matching policy head on the deterministic Unified

The sections below (§1–§9) are the reference/rationale trail (UWM/DreamZero mechanism, the pathology,
the simulator hinge). This is the actual plan. **Only the action head changes; everything else in
LeWAMUnified is untouched and deterministic** (ViT encoder → causal aggregator → `c_t = z_t + g·Aggr(z_≤t)`
→ deterministic `dynamics`). Replace the deterministic `gc_head` with a **FlowPolicyHead**.

**FlowPolicyHead.** Emits an action CHUNK `A ∈ R^{H×bd}`, `H=5` blocks (= the eval planning horizon;
frameskip units → 25 raw steps), `bd = raw_adim×frameskip`. **Rectified flow** (straight path,
velocity regression). A small stack of **AdaLN-Zero DiT blocks over the H chunk-tokens**. Reference
impls to lean on: **UWM** `models/common/adaln_attention.py` + its adaLN-Zero init
(`nn.init.constant_(block.adaLN_modulation[-1].weight/bias, 0)` and zeroed final layer), **DreamZero**
`FlowMatchScheduler` / `WANPolicyHead`, and π0-style flow action heads.
- **Conditioning = ADD of MLP-embedded signals + adaLN-Zero** (the proven DiT/SD3/FLUX pattern, NOT
  per-input LayerNorm): `cond = mlp_ctx(c_last) + mlp_goal(z_goal) + mlp_h(h_norm) + mlp_tau(τ)`, each
  MLP → `cond_dim`; the block's `adaLN_modulation` (SiLU→Linear→shift/scale/gate) is **zero-initialized**
  so every block boots as identity. That zero-init is the stabilizer.
- **Training loss (flow-matching, straight path):** `x0~N(0,I); τ~U(0,1); A_τ=(1−τ)x0+τA_gt;
  v_tgt=A_gt−x0; loss_act = MSE(v_θ(A_τ,τ,cond), v_tgt)`. **Mask beyond-goal chunk positions**
  (position `k` at remaining horizon `h_norm−k`; mask `k ≥ remaining_blocks`) — flow loss only, the
  dynamics head (per-transition, real steps) is untouched. `loss = w_act·loss_act + w_dyn·mse_dyn +
  w_reg·sigreg`. Keep **goal-dropout** (cheap regularizer + leaves CFG available). `A_gt` = next H
  action blocks per position (extend the window slice; `H_max`/`context_len` already cover ≥5).
- **Inference:** few-step Euler ODE, `n_steps=8–10` (tiny head → free; π0 ~10, DreamZero 16→distill),
  `shift≈1` (SD3 shift concentrates steps at high noise for high-dim data; our 25-dim actions don't
  need it). **Full-conditional by default (CFG w=1, single pass)**; CFG (`v=v_u+w(v_c−v_u)`, goal-null
  uncond) is an optional inference knob, off unless the goal is under-followed.

**Eval modes.**
- **Reactive receding-horizon (the policy IS the planner):** sample chunk, execute FIRST block,
  replan; optional **ACT temporal ensembling** (exp-weighted avg over overlapping chunks) — inference-only.
- **Policy-proposal WM-verify (replaces `unified_cem`; NO Gaussian CEM, proposals = the action
  distribution):** B1 whole-chunk — sample K chunks, roll the deterministic WM over each, pick min goal
  cost; B2 AR-resample — resample K per step, WM 1-step lookahead, greedy. Owner expectation: verify
  won't beat reactive (dynamics OOD off-policy), but stays in-distribution (no OOD exploitation).

**Knobs.** Train-time (fixed once): `H`, the flow objective, the beyond-goal mask. Inference-time
(free ablation on ONE trained model): `n_steps`, `K`, verify mode (reactive / B1 / B2), CFG `w`,
ensemble weight `m`, verify cost (terminal vs summed). Screen on pusht 8k first.

**Files.** New `lewam/models/flow_policy_head.py`; integrate via `--head_type flow` in
`lewam_unified.py`, `train_lewam_unified.py` (chunk targets + masked flow loss), and
`gip.py`/`eval_gip.py` (flow reactive + policy-proposal modes). NOTE: these four files currently hold
the owner's uncommitted WIP — coordinate before editing.

**Interpretability probe (parallel, Minghao):** decode-viz to see joint-training's effect on the
latent. Adapt `experimental/train_decoder.py`+`viz_decoder.py` for Merlin/pusht with a `--latent {z,c}`
switch: `z` = encoder CLS (single-frame); `c = z_t + Aggr(z_≤t)` (residual-inclusive head input, needs
a windowed aggregate). The recon delta `c−z` = the aggregator's contribution. Each latent needs its own
trained decoder. Compare across checkpoints (off-policy: op r=0.05 vs r=0 control; and/or joint-vs-less:
unified vs split/LeWM).

## 1. Motivation and the pathology we are designing against

Two prior LeWAM architectures independently died of the **same** failure:

- **LeWAM-Seq** (interleaved causal transformer): "gameable BC-shaped latent,
  representation-for-control ≠ representation-for-prediction"; seq_cem exploitable, precise tasks
  fell off (pusht 61 vs split 88, reacher 37 vs 98).
- **LeWAM-Unified** (shared aggregator `c` → gc_head + dynamics): the shared `c` is a
  *policy-shaped* representation the dynamics borrows, not physics. Proven by the coupled-vs-detach
  collapse experiment — you cannot improve the latent for planning without degrading the policy,
  because it is the same latent.

The lens for choosing any successor is therefore: **does it escape "policy-shaped latent →
un-plannable"?** The honest answer for a unified diffusion model is **not by construction**. Shared
weights + an action-denoising (BC) objective + expert-only data still shape the state-denoising path
toward expert-conditioned dynamics. Unification buys elegance, not plannability. The plannability
levers are unchanged from what we already found:

1. **Off-policy data in the state-denoising (dynamics) loss** — the WM finally sees non-expert
   actions. This is the primary, architecture-agnostic lever (our r=0.05 inverted-U result).
2. **MoT** (modality-expert weights, shared attention) — confines the BC gradient to the attention
   path and shields the state-expert's own MLPs. Mitigates, does not cure. Secondary / scale-up.

What the flow route DOES buy that the aggregator route did not: a **usable simulator** (feed a raw
action, read the resulting state) that the interpretability probe and planning both need, a strong
**multimodal action head** (flow-matching, which fixed nothing when we tried a GMM head but is the
right form here), and one model that is simultaneously policy / forward-dynamics / inverse-dynamics /
WM by choice of noise schedule.

## 2. The simulator hinge (the load-bearing mechanism)

A unified denoiser is a viable **simulator** (forward dynamics: clamp action clean, denoise state)
**iff training samples the two modalities' noise levels independently.** Independent sampling covers
the full grid of (action-noise, state-noise) pairs, so the "clean action, noisy state" slice is
explicitly supervised and in-distribution at inference. A shared/coupled timestep never trains that
off-diagonal, so the query returns nonsense.

Both codebases confirm this and bracket the choice:

- **UWM — independent by default.** `models/uwm/uwm.py`: `action_t = randint(0,T)` (line 289) and
  `next_obs_t = randint(0,T)` (line 298) are drawn separately. Forward dynamics
  (`sample_forward_dynamics`, 325–345) holds `action_t = timesteps[-1]` (=0, clean), passes the clean
  action, denoises only the observation. Works because the off-diagonal was trained.
- **DreamZero — coupled by default, decoupled by flag.** `.../action_head/wan_flow_matching_action_tf.py`:
  with `decouple_video_action_noise=False` (default, line 108) the action timestep is *derived from*
  the video timestep (lines 716–718, "action timestep derived from video timestep") — coupled, NOT a
  simulator. With the flag True, video draws a high-noise Beta (line 682) and the action timestep is
  sampled independently full-range (`randint`, 708–712) — simulator viable; a matching
  `decouple_inference_noise` (118–122) runs the asymmetric inference. The authors' comment says the
  flag exists "for training-inference alignment" — they hit exactly this.

**v0 uses independent per-modality flow-times.** Refinement from DreamZero: bias the *state* flow-time
toward high noise while keeping the *action* flow-time uniform, to emphasize the forward-dynamics
regime we query at inference rather than pure uniform.

**Separate tokens is the only architectural commit; the schedule is a downstream knob.** Coupled vs
independent sampling is a config choice *given* two separate action/state token streams — the DreamZero
flag exists only because there are two noise variables to decouple. Fuse them into one token (paper
Alg 1) and no flag recovers the simulator, because there is physically one noise variable. Moreover
independent training **subsumes** coupled: one independently-trained net serves policy,
forward-dynamics, inverse-dynamics AND joint generation (UWM's `sample_marginal_action` /
`sample_forward_dynamics` / `sample_joint` all run on the same network), so we do not choose
simulator-vs-not at train time — we train once and pick the mode at inference by which timesteps we
clamp. The real cost is that fitting the whole `(τ_a, τ_z)` grid is a harder problem than the diagonal
and can dilute the reactive policy on small data / limited epochs; the honest knob is therefore the
**sampling distribution over the grid** (how much off-diagonal mass), tuned on the pusht screen — not a
binary flag. That is why DreamZero (14B, no simulator need) defaults to coupled and biases even its
decoupled mode, whereas UWM (closest to our scale) trains fully independent and still reports a good
policy.

## 3. What we take from each reference

| | UWM | DreamZero |
|---|---|---|
| Role | **fork the mechanism** (small, released, clean) | **idea donor** (14B Wan2.2/GR00T, too heavy to fork) |
| Denoiser | DDIM eps-pred, 10 steps | **flow-matching** (`FlowMatchScheduler` shift=5), 16 steps, `dit_step_mask` 5/6/7/8-step flash |
| Unification | independent dual timestep, one shared-attention DiT, register tokens, global dual-timestep AdaLN | separate `WANPolicyHead` cross-attending Wan video DiT; coupled/decoupled flag |
| Horizon | single-step (action-chunk + 1 next-obs) | chunk-wise AR over frame blocks |
| Goal | none (multitask dataset) | text/instruction `q` via CFG (scale 5.0) |

We adopt **UWM's structure** (one DiT, shared attention, separate token positions + encoders/decoders,
register tokens, global timestep AdaLN, and its `sample_*` inference modes) and swap in
**DreamZero's flow-matching** (v-prediction, few-step ODE). Everything else is LeWAM-specific.

## 4. v0 architecture

Single-step-in-time, latent-space, separate action/state tokens, independent flow-times.

- **Targets in latent space, not pixels.** Denoise `z_{t+1}` (the LeWM ViT-tiny latent, 192-d) and an
  **action chunk** `(action_len, action_dim)`. UWM denoises VAE image patches; we denoise the LeWM
  latent — far cheaper and it is the LeWM-native WM space. `action_len` = our frameskip block to start
  (that is inherent chunking, distinct from ACT-style multi-vector chunking).
- **One DiT.** Tokens = `[action-chunk tokens, next-z token(s), registers]`, shared AdaLN self-attention
  (mirror UWM's `DualNoisePredictionNet.forward`, uwm.py:181–213). Action vs state distinguished by
  position + separate encoder/decoder MLPs.
- **Flow-matching.** v-prediction, `FlowMatchScheduler`. Independent per-modality flow-times
  `(τ_a, τ_z)` sampled per-sample; state biased high-noise, action uniform (§2 refinement).
- **Conditioning (global AdaLN cond).** Concatenate: history context + `z_goal` + `h_norm` +
  dual-flow-time embedding. History = **past-`z` tokens as context** to start (cleaner than an
  aggregator, avoids re-importing a single-`c` bottleneck; the bottleneck was never the real risk, but
  multi-token context costs nothing and sidesteps the argument). `z_goal` trained with goal-dropout →
  classifier-free guidance at inference (infra reused from Seq/Unified `--goal_dropout_act/dyn`).
  `h_norm` (normalized steps-to-goal) via the existing `ActionHead` horizon MLP → this is how a
  single-step model stays goal-directed within a planning budget without a trajectory diffuser.
- **Encoder.** Shared ViT-tiny (embed_dim 192), same backbone as current. DiT depth ~6–8, small heads.
  Target total params near the current 9.3M scale so comparisons stay fair.

## 5. Inference modes (UWM `sample_*` mapped to flow)

- **policy** — denoise action (`τ_a: T→0`), state uncared/noisy → `a | history, z_goal, h_norm`.
- **forward dynamics / simulator** — clamp action clean (`τ_a=0`), denoise state (`τ_z: T→0`) →
  `z_{t+1} | a, history`. The probe-and-verify tool the whole thesis needs.
- **inverse dynamics** — clamp next-state clean, denoise action.
- **joint / marginal next-obs** — both denoise.
- **few-step** — 4–8 ODE steps (DreamZero's flash mask), distill later if needed.

## 6. Planning

- **Reactive** — policy-sample the action chunk, execute first step, observe, re-condition, repeat
  (receding horizon).
- **Plan** — sample K action-chunk candidates from the policy-prior, **forward-simulate each** (clamp
  action clean, denoise `z'`), score reachability to `z_goal`, pick best. The WM-verify step, recovered
  precisely because we did NOT fuse the tokens.

## 7. Open decisions

- **fuse-vs-separate:** SEPARATE (keep the simulator + causal probe). Every reference that keeps a
  simulator keeps separate streams; DreamZero's default fuse is exactly why its base model is not one.
  This is the **only** architectural commit — coupled / independent / biased-independent scheduling is a
  tunable training-time sampling distribution downstream of it (§2), not a separate fork.
  **Gate = Minghao.**
- **training budget:** flow objectives train slower than MSE; 50ep (Unified's bar) is likely not the
  fair budget — set it by the val-loss plateau, expect > 50ep.
- **history representation:** past-`z` tokens vs aggregator context (start with tokens).
- **latent vs pixel target, `action_len`, state flow-time bias** — tune on the pusht screen.

## 8. Staging

- **v0** (this doc): single-step, latent, separate, independent flow-times, goal+horizon cond, built on
  UWM's structure + DreamZero's flow head. Success test: matches/beats LeWAM-Unified reactive AND yields
  a working `τ_a=0` forward simulator we can finally probe for causality.
- **v1**: multi-step trajectory diffusion (Diffuser-like) for genuine planning-in-budget; MoT if
  shared-weight collapse appears; off-policy data dosed into the state-denoising loss (the real
  plannability lever).

## 9. Reproduction pointers

- UWM: `models/uwm/uwm.py` (`UnifiedWorldModel`, `DualNoisePredictionNet`, `sample_forward_dynamics`),
  `experiments/uwm/train.py`. DDIM/eps, 10 inference steps.
- DreamZero: `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` (`WANPolicyHead`,
  training `forward` ~603, coupled/decoupled at 680–718), `base_vla.py` (VLA = Wan backbone + action
  head; `lazy_joint_video_action_causal_gt_cond` = the GT-conditioned causal mode).
