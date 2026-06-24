# LeWAM — slide outline for the chat with Prof. Randall Balestriero (2026-06-15)

Context: Randall endorsed "LeWAM" (the WAM sibling to his LeWM) — "perfect showcase, big impact, different needs than
pure LeWM." He asked for **background on the latest WAM setups** + our positioning, then a chat (next week). He knows
JEPA/LeWM deeply, so pitch technical; spend real time on the **WAM landscape** (he doesn't track the 2-month churn).
~12 slides. Lead with the WAM identity + the honest per-family/negative findings (rigor he'll respect), NOT an SR win.

---

## S1 — Title
**LeWAM: a Latent World-Action Model.** Joint next-(state, action) prediction on the LeWM latent — a JEPA world-action
model, not a VLA / pure BC. Minghao Fu · Randall Balestriero. (One line: "the WAM sibling to LeWM.")

## S2 — Why WAM, why now (the framing)
- VLA/BC predict actions from observations; **WAM models world dynamics explicitly** (predict next state AND action),
  giving a forward model you can plan/verify with — not just imitate.
- Community shift 2024→26: from BC-only to world-action modeling. LeWAM = LeWM's latent + an action pathway, unified.
- The pitch: same JEPA latent Randall built, now jointly predicting actions → "all-in, all-out" state-action model.

## S3 — The WAM landscape (Randall's ask — the "latest setups"). Map by HOW they get actions:
| family | examples (arXiv) | action mechanism |
|---|---|---|
| pure CEM/MPC over latent WM (no prior) | **LeWM** 2603.19312, DINO-WM 2411.04983, V-JEPA2-AC 2506.09985 | sample + score; no learned proposal |
| policy-prior **warm-starts** search | TD-MPC2 2310.16828, **PiJEPA** 2603.25981, **Newt** 2511.19584 | learned π seeds MPPI/CEM |
| **amortize / replace** search | **GC-IDM** 2605.08732 (on LeWM latent!), TD-JEPA 2510.00739 | (z, z_goal, horizon)→a, no search |
| diffusion / **hierarchical** | LDP 2504.16925, **WorldDP** 2606.08775 (LeCun), FF-JEPA 2606.09311 | WM picks subgoals, diffusion policy executes |
| "representation > planning" challenge | **MR.Q** 2606.05555 | model-free + aux predictive losses beats WM-planner |
- Two axes to draw: **search ↔ amortize**, and **flat ↔ hierarchical**. Note the churn (Newt/PiJEPA/WorldDP/FF-JEPA all post-cutoff, last ~6 mo).

## S4 — Where LeWAM sits (the wedge)
- Joint **(z_{t+1}, a_t)** on the LeWM latent: shared encoder → WM head + action/intention head, jointly trained.
- Distinct from the map: **history / past-action-conditioned** action pathway (rare — most priors are Markovian),
  **warm-start-then-refine** (between pure-CEM and pure-amortize), targeting **contact-rich manipulation** (robomimic,
  which WM-planning papers barely report).

## S5 — Method (concise architecture)
- Encoder (DINOv2/ViT) → latent z; WM predictor (z dynamics, SIGReg-regularized → plannable); action/intention head
  (autoregressive on z-history + past actions) → act-embedding → decoder → raw action. Frozen-CLIP task token (multitask).
- Eval modes: BC (history) / guided / CEM-planning, all on the same latent.

## S6 — Finding 1: **planning is per-family** (the core analysis)
- WM-planning (CEM) WORKS on geometry/goal-reaching: OGBench-Cube 68→78(planning)→88(intuition-guided), Push-T, Reacher,
  Two-Room. FAILS (≈0) on contact-rich robomimic Lift/Can/Square.
- Mechanism (verified): both are grasp tasks + identical eval, so NOT "push vs grasp." On cube the goal-image shows the
  object **displaced** → CEM must grasp+carry to lower latent-cost. On robomimic-lift the goal is the cube +few cm,
  **swamped by arm pose** → CEM matches pose, skips grasp; grasp state is invisible in a single-frame latent. (+ data:
  cube ~1M transitions vs robomimic 200 demos.)

## S7 — Finding 2: **history-BC ≫ CEM on contact**
- On robomimic, history-conditioned BC recovers grasp state (post-grasp P(success|grasp) 76→99) → Lift ~95, Can ~50–72,
  Square ~60s, while online CEM = 0. The action pathway, conditioned on past actions, carries contact tasks where the
  WM's single-frame planning can't. (Closest published latent planner LDP: 0.69/0.70/0.46 — we beat it on Square.)

## S8 — Finding 3 (honest negative): **the action-head A/B is within training noise**
- detach=TRUE / FALSE / solveact(raw-action) / anti-collapse(SIGReg on act-emb): all **tied ~72±5 on Can** at N=50,
  3 TRAIN-seeds. The ~10-pt **training-run variance swamps** the 4–7-pt effects single runs showed.
- Lesson (Randall will like): multi-TRAIN-seed is mandatory; single-run A/Bs (and eval-seed-only "3-seed") are
  misleading. We caught a false "solveact wins / intent_loss hurts" headline before publishing it.

## S9 — Contribution / positioning
- LeWAM = unified JEPA world-action model. Contributions = **(i)** the per-family "when does WM-planning help?" analysis,
  **(ii)** history/past-action-conditioned action pathway recovering contact state, **(iii)** the WAM reframe + honest
  variance accounting. **NOT** a fine action-head SR win (that washed out).

## S10 — Differentiation (related work, head-on)
- vs Newt (multitask JEPA + BC-prior + MPPI, Markovian), PiJEPA (instantaneous JEPA prior), GC-IDM (amortize on LeWM —
  must-beat baseline), WorldDP/LDP (hierarchical + diffusion execution), MR.Q (representation>planning). Our wedge:
  history-conditioned prior + per-family analysis + warm-start-refine on contact.

## S11 — Next directions
- **Hierarchical subgoal-LeWAM** (WorldDP/LDP-style): WM plans subgoal latents, the action pathway executes each →
  attack robomimic planning=0 (the per-family limit). Unified "z,a interleaved, all-in-all-out" tokenization.
- Open: does subgoaling beat flat history-BC where the WM is contact-blind? (the test).

## S12 — Discussion / asks for Randall
- Framing & venue (LeWAM as the showcase). Which direction to push first (hierarchical subgoal vs the analysis paper).
- His read on the WAM-landscape positioning (is the per-family planning story the right headline?).

---
**Build notes:** sources = proposal/literature_survey.md (the landscape + arXiv IDs) and EXPERIMENTS.md (the numbers,
all verified this session). Keep S6–S8 the heart. Numbers to re-pull fresh before presenting: robomimic histbc 3-seed
(lift ~94.7 / can ~48–72 / square ~64.7), cube 68/78/88, the can A/B (74/72/76 detT ≈ 70/70 solve = tied).
