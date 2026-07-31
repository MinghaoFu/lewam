# LeWAM — Positioning & Novelty Notes (2026-06-15)

Strategic notes for the paper + the Randall chat. Companion to `literature_survey.md` (full landscape),
`related_work.tex` (the polished section), and `EXPERIMENTS.md` (the verified numbers). Honest, blunt — the
WAM field is crowded and LeWAM's novelty is narrow; this file pins down exactly where it is.

## TL;DR (REVISED 2026-06-16 — planning thesis, user-steered)
- **Thesis: LeWAM = intuition-guided PLANNING on CONTACT-RICH manipulation.** NOT an analysis paper, NOT "another WAM."
  Stay a *planner* (vs LeWM/DINO-WM pure-CEM, vs the search-free WAMs); the hard target is *contact*.
- **The contribution = hierarchical, probe-grounded, intuition-guided subgoal planning** that conquers contact, where flat
  planning (pure OR guided) = 0. The per-family analysis + multi-task probe are the *grounding/evidence*, not the headline.
- **Key mechanism insight (TD-MPC2-grounded):** flat planning dies on contact because we lack TD-MPC2's *value bootstrap*
  (they plan only horizon-3 because Q carries the rest); our reward-free fix = **subgoals as a structured value**, and the
  **probe proves the latent can score grasp subgoals (0.96/0.999)** where the flat goal can't. See "Intuition-guided planning
  on CONTACT" section.
- **Fork ✅ RESOLVED → (b) the planning method.** Next experiment = the subgoal-cost diagnostic (running).
- *(The WAM field exploded in 2026 — we still can't claim "WAM"/"latent WAM"/"is search worth it"; that's why the
  contribution is the contact-PLANNING method + its probe grounding, not a WAM or a bare analysis.)*

## The 2026 WAM ecosystem (the crowd we're entering)
| paper | arXiv | what it is | search at test time? |
|---|---|---|---|
| **DreamZero** "World Action Models are Zero-shot Policies" | 2602.15922 (Ye/Ge/Zheng+, NVIDIA-scale) | video-diffusion WAM, $p(\text{video},a)$, 14B | **NO** (no rollout) |
| **Being-H0.7** "A Latent World-Action Model from Egocentric Videos" | 2605.00078 (Luo/.../Lu, BeingBeyond/PKU) | **latent** WAM (our exact term!), 200k h video, prior/posterior future-aware latent workspace | **NO** (discards posterior) |
| **Fast-WAM** "Do World Action Models Need Test-time Future Imagination?" | 2603.16666 (Yuan/Dong/Liu/Zhao) | **asks our meta-question; answers NO** (video co-training > rollout, 4× faster) | **NO** |
| Efficient-WAM "1B-param low-cost future imagination" | 2606.10040 | small WAM | partial |
| OA-WAM "Object-Addressable WAM" | 2605.06481 | object-addressable manipulation | — |
| WAM survey / OpenMOSS Awesome-WAM | 2605.00080 / GitHub | survey + reading list | — (evidence the field is hot) |
- **The trend: the field is going SEARCH-FREE** (amortize planning into one forward pass). Fast-WAM makes it explicit
  and concludes test-time imagination is unnecessary (uniformly).

## What is TAKEN — do NOT headline these
1. "World-action model" / joint state-action / WM+policy — the whole field.
2. "Latent WAM" — Being-H0.7, at scale.
3. "Is test-time planning/imagination worth it?" — Fast-WAM (no, for video) + MR.Q (representation > planning).
4. Hierarchical subgoal + policy execution — WorldDP, LDP.
5. Action-history for BC — RNN-BC, ACT, MBOP.
6. The action-head micro-choices (detach / solveact / anti-collapse) — and we PROVED they're within training noise
   (negative result, EXPERIMENTS.md §1c). The original "embedding-intuition" + "warm-start-CEM" wedges are both weak now
   (intuition doesn't matter; CEM is being abandoned by the field).

## What is OURS — the un-scooped core (the only thing to stake the paper on)
A **controlled, mechanistic** result the scale-WAM crowd structurally can't produce:
1. **Per-family decomposition** — not Fast-WAM's blanket "no," but **planning helps on geometry/goal-reaching,
   collapses on contact**, with the boundary characterized.
2. **The mechanism (falsifiable, ⚠️ now TWO-PART — probe-corrected 2026-06-16):** CEM≈0 on robomimic for *two distinct*
   reasons. **(a) Lift:** the goal-image latent cost only rewards a *large visible* change, and grasp of the small cube is
   *visually subdominant in a single frame* (probe 0.82 vs 0.99 from history), so CEM matches arm pose and skips the grasp —
   a **perception** failure. **(b) Can/Square:** the single-frame latent *already* decodes grasp (probe 0.96/0.999), so it's
   NOT perception; CEM fails because the task is **multi-stage** (approach→grasp→transport→place) and a flat horizon-$H$
   goal-image cost can't decompose the sequence — a **horizon/decomposition** failure (= WorldDP's diagnosis). (Plus data:
   cube ~1M vs robomimic 200 demos.)
3. **Contact-state recovery (probe, multi-task DONE 2026-06-16) — LIFT-SPECIFIC, honest counter-evidence on Can/Square:**
   decoding grasp state from the latent, 5 demo-split seeds, detachT100: **Lift single-frame 0.82 → latent-history 0.99 →
   full $[z,a]$ 0.999** (big gap ✅, the headline case); but **Can 0.96/0.97/0.97 and Square 0.999/0.999/0.999** (single-frame
   already decodes grasp → history adds ~0 ❌). So "history ≫ single-frame" is **true only for the small Lift cube**, NOT a
   universal contact law — Can/Square objects lift visibly. History-BC still beats CEM on all three (it executes the
   sequential demo behavior; on Lift it *also* supplies contact-state memory, post-grasp $P(\text{succ}|\text{grasp})$ 76→99).
   **Do NOT headline a universal single-frame contact-blindness.** The richer (and more defensible) claim = "planning fails
   for two diagnosable reasons across the contact family, and we tell which is which per task." [[project_gip_intention_format_eval_convention]]

## The defensible framing
**LeWAM = the controlled, mechanistic study in a field of scale plays.** DreamZero/Being-H0.7/Fast-WAM are
1–14B-param, 200k-hr-video, LIBERO/CALVIN/real-robot efforts answering "does search help?" *empirically and uniformly*.
LeWAM, on a clean LeWM latent with *explicit* CEM, isolates **when and why, per task family**, with probes + multi-seed
rigor. Contribution = **understanding, not SR**. **For Randall:** it's a scientific extension of *his* latent (LeWM) —
"when does LeWM-style latent planning help, and mechanistically why not on contact" — a natural, appealing angle for him.

## To make it a STRONG paper (not just "an analysis") — pick one
- **(a) Routing rule** — from the goal-image-cost geometry, *predict* plan-vs-BC and route per task/state; beat both
  always-plan and always-BC. Makes the analysis **actionable**. (lower risk; plays to the strength)
- **(b) Hierarchical-subgoal-LeWAM** — use the analysis to *fix* contact (WM plans subgoal latents, history-BC executes).
  Higher ceiling, but **real scoop risk from WorldDP**, and must beat flat history-BC. (EMA-target + scaffolding already
  implemented; the open test is whether subgoaling beats flat history-BC when the WM is contact-blind.)
- **(c) Definitive probe study** — make the contact-state probe + goal-image-cost mechanism airtight (controlled,
  multi-task) and own "why latent WM-planning fails on contact" as *the* reference result.

~~**Recommendation: (c) + (a)** — the mechanistic study + the plan-vs-BC routing rule.~~ **⇒ REVISED 2026-06-16 (user
steer, two explicit messages): the thesis is (b) — "LeWAM = intuition-guided PLANNING on CONTACT-RICH manipulation."**
The user does NOT want an analysis paper or "another WAM"; they want LeWAM to stay a *planner* (vs LeWM/DINO-WM pure-CEM
and vs the search-free WAMs) and to *conquer contact* (the hard, valuable target). So (b) the hierarchical planning method
is the paper; (c) the probe study is its empirical *grounding* (the probe is the license for subgoal costs, see below);
(a) routing is a fallback only. **Key = CONTACT. Stay on PLANNING.** See the new "Intuition-guided planning on contact"
section below for the TD-MPC2-grounded design.

## Intuition-guided planning on CONTACT (resolved direction + TD-MPC2-grounded design, 2026-06-16)
**Identity:** LeWAM = LeWM/DINO-WM's plannable latent + a learned, **history-conditioned intuition that guides the search**.
Three-way contrast: (i) vs LeWM/DINO-WM/V-JEPA2-AC = pure CEM, *no prior*, blind cold-start; (ii) vs DreamZero/Being-H0.7/
**Fast-WAM** = search-*free* (Fast-WAM literally says "test-time imagination isn't worth it") — we are the contrarian "planning
still wins, *when guided*, on the right family"; (iii) "WAM" is our model class, the *contribution* is intuition-guided planning.

**The hard fact (verified):** flat planning on robomimic = **0** whether pure-CEM or intuition-guided ("guided" also ≈0,
§7b); even aligned-goal CEM (#47) = 0. Only history-BC works (93/71/61). So "intuition-guided planning on contact" CANNOT
come from the flat planner — needs a new mechanism.

**TD-MPC2 evidence (read the code 2026-06-16, github nicklashansen/tdmpc2 `tdmpc2.py`+`config.yaml`):** their policy prior
is used **two ways**, and the 2nd is the one we lack.
- *Role 1 — seed the search:* roll `pi` through the model for `num_pi_trajs=24` trajs, concat into the `num_samples=512`
  MPPI pool (`actions[:, :num_pi_trajs] = pi_actions`), regenerated each of `iterations=6`.
- *Role 2 — bootstrap terminal value:* `_estimate_value` = Σ discounted rewards `+ discount*(1-term)*Q(z, pi(z))` at the
  last latent. `pi` trained SAC-style to max Q.
- **Punchline: `horizon=3`.** TD-MPC2 only plans 3 steps because **the learned value Q bootstraps everything beyond** — the
  prior both seeds the short search AND supplies the terminal action for the Q-bootstrap.
- **Our LeWM planner has NO value bootstrap:** long horizon to a distant goal-image-distance cost, no reward/Q. Our "guided"
  mode is **only Role 1 (seeding), missing Role 2 (the bootstrap)** — which in TD-MPC2 is *the* thing that makes short-horizon
  planning work. **This is the smoking gun for why flat planning dies on multi-stage contact:** forced to plan the whole
  horizon against an undecomposable goal-image the WM can't roll through contact.

**Two levers to give planning a bootstrap (reward-free), both keep planning central:**
1. **Subgoal decomposition = structured value.** Plan over *subgoal latents* (pre-grasp→grasped→aligned→placed), each a
   near-term, WM-evaluable target → turns one distant goal into 3-step-reachable legs (TD-MPC2's working regime).
2. **Learned goal-conditioned value `V(z, z_goal)` / steps-to-go** = TD-MPC2's Q-bootstrap made reward-free; plan short-horizon
   + bootstrap. (TD-JEPA-flavored; heavier.)

**Why subgoals are "grounded in the probe":** (a) the probe (Can/Square latent decodes grasp **0.96/0.999**) is the *license* —
proof the subgoal cost has signal (distance-to-grasped-latent is informative, unlike the flat goal that buries grasp); (b) the
subgoals are *defined* from the same grasp/object-z signal (cluster demo latents into stages). "Intuition-warm-started subgoal
search (not blind)" = roll the history policy forward → latents along its rollout = candidate subgoals (TD-MPC2 Role-1 lifted to
subgoal level), vs sampling subgoals blindly in 192-D (hopeless).

**Differentiation vs TD-MPC2 / WorldDP:** TD-MPC2 prior = **Markovian** `pi(z_t)`, raw-action, **online-RL/Q**. Ours = **history/
past-action-conditioned** (contact-state memory — a Markovian prior literally can't remember "I'm holding it"), **offline-demo**,
reward-free, subgoal-bootstrapped. WorldDP = object-slots + diffusion-policy; ours = holistic JEPA latent + history-intuition +
probe-grounded subgoals + warm-start-refine.

**Experiment ladder:**
- **Cheap diagnostic (run FIRST):** does CEM-to-z(grasped-subgoal) succeed where CEM-to-z(final-goal)=0, on Can/Square? Tests
  whether "the latent sees grasp" (probe) ⇒ "a planner can reach the grasp subgoal." Green-lights or kills the approach for ~1 script.
- **Make-or-break:** intuition-warm-started **hierarchical** subgoal planning **> flat history-BC** on Can/Square (70.7/61.3).
  Square is the target (most multi-stage, most drift; planning's value = closed-loop subgoal correction vs open-loop BC drift).
- Fallback if subgoal-cost too weak: the learned goal-value bootstrap (lever 2).

## OPEN FORK — ✅ RESOLVED 2026-06-16 → (b) the planning method
~~analysis+routing vs hierarchical method~~ → **user picked the planning thesis (contact-rich, hierarchical).** (a) routing
demoted to fallback. Next experiment = the cheap subgoal-cost diagnostic above.

## For the Randall chat — talking points (REVISED 2026-06-16 → planning thesis)
1. Acknowledge the crowd (DreamZero / Being-H0.7 / Fast-WAM) up front — but stake the *contrarian* line: the field went
   search-free; **LeWAM keeps planning and makes it smart** (intuition-guided), and shows it can be pushed onto *contact*.
2. The pitch: **LeWAM = intuition-guided PLANNING on contact-rich manipulation** — building on *his* LeWM latent + a
   history-conditioned intuition, made to work on contact via **hierarchical, probe-grounded subgoal planning**.
3. The hook (TD-MPC2-grounded): flat planning fails on contact because we lack TD-MPC2's *value bootstrap* (they plan only
   horizon-3 because Q carries the rest); our fix is reward-free — subgoals as a structured value, **and the probe proves the
   latent can score grasp subgoals (0.96/0.999) where the flat goal can't**.
4. Show him: the per-family contrast + the multi-task probe + (if green) the subgoal-cost diagnostic = the empirical anchors.

## In-flight evidence
- **Contact-state probe** (multi-task DONE 2026-06-16): grasp-state decoding, 5 demo-split seeds, detachT100. **Lift**
  single-frame **0.82** → latent-history **0.99** → full $[z,a]$ **0.999** (big gap ✅); **Can 0.96/0.97/0.97**, **Square
  0.999/0.999/0.999** (single-frame already decodes grasp ❌). ⇒ "history ≫ single-frame" is **Lift-specific**; Can/Square
  fail from multi-stage structure, not perception. Two-part mechanism — see §"What is OURS" #2/#3. Honest counter-evidence,
  NOT a universal law.
- **Goal-reaching numbers** (3-eval-seed DONE; 3-train-seed running): guided/planning ≫ bc on tworoom (90/95 vs 46) and
  reacher (87/85 vs 3); pusht bc≈guided (71≈72, reactive). All ≫ robomimic planning 0. ⚠️ eval-seed variance LARGE on
  goal-reaching (pusht planning range 28) → need eval×train averaging (EXPERIMENTS.md §1b/§CONCLUSIONS #4).
