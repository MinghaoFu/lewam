# LeWAM / MoT-flow: formulation, architecture, and the current puzzle

Self-contained summary for external discussion (2026-08-30). All numbers are measured (3 eval
seeds × 50 episodes unless noted); no speculation is mixed into the results sections.

## Goal

Learn, from pixel demonstrations only (no pretrained encoders, no state access), a single recipe
producing jointly: an encoder, a latent world model (dynamics), and a policy — such that (a) the
same recipe works on both goal-reaching cells (GR: pushT, two-room, reacher; goal given as an
image, success = reaching it) and task-conditioned manipulation (TC: robomimic tool-hang etc.,
no goal input, success = task completion), and (b) the world model is genuinely usable — it must
support planning and pass dynamics probes. A model with good success rate but a bad world model
is considered a failure ("this is not a policy paper").

Reference points on pushT (all our evals, same protocol): LeWM (JEPA world model + CEM, no
learned policy) 88.0; LeWAM-unified (separate aggregator/dynamics/policy heads on one encoder)
87–89; behavior-cloning-only variants of our joint models ~66–73.

## Data and problem setup

Expert demonstrations as frame sequences; models operate at frameskip fs = 5 (one "block" = 5
raw actions; raw action dim 2 on pushT, 7 on tool-hang). The encoder E maps a 224×224 frame to
z ∈ R^384 (ResNet-18-style, trained from scratch). History = L = 2 latents at fs spacing. The
policy predicts a chunk of 10 raw actions (2 blocks); the dynamics predicts z one block ahead
(z_{t+5}) from the history and that block's 5 actions. GR adds a goal latent z_g = E(goal frame)
and a scalar h = normalized remaining horizon. Evaluation: GR headline = planning success (goal
25 env steps ahead, execute 5 actions per replan); TC = reactive rollout success.

## MoT-flow architecture ("Mixture-of-Transformers")

One transformer stack, two parameter streams (state, action): every token is processed by
stream-specific QKV/MLP/AdaLN parameters, but all tokens interact through a single global masked
self-attention. Tokens and the attention mask (rows attend to listed keys only):

- z_hist (2 tokens, encoder latents): attend to z_hist only.
- z* (1 state token — noisy state for the flow head, a learned query for the MSE head):
  attends to z_hist, itself, and the CLEAN action tokens a_1..a_5.
- z_g (goal token, GR only): attends to z_hist and itself.
- clean actions a_1..a_5 (teacher-forced first block): attend to z_hist and a_{≤j} (causal).
- noisy actions a*_1..a*_10 (the policy flow's sample): attend to z_hist, z_g, a*_{≤j} (causal).

Two decoupling properties by construction: the policy tokens never attend to any state token
(the policy cannot read the world model), and the state stream never sees the goal (the dynamics
is goal-blind; on GR the goal reaches only the policy). The two heads share nothing but the
encoder and the trunk's attention pattern.

Losses. Policy: rectified flow on the 10-action chunk (velocity target a − ε, τ_a ~ U[0,1],
per-modality AdaLN conditioning; goal token dropped/kept per config). Dynamics, two variants:
(i) "MSE head": the state token is a learned query regressing z_{t+5} directly; (ii) "flow
head": rectified flow on the state token (Gaussian source). Targets are the online encoder
latents (the encoder learns from being the target; no EMA). Optional SIGReg (from LeJEPA) on the
encoder latent: a regularizer pushing the batch latent distribution toward an isotropic Gaussian
(random 1-D slices, characteristic-function statistic), weight 0.04. Without it the latent is
collapse-adjacent (per-dim std ~0.02–0.15 and shrinking); with it, ~unit scale. Training: AdamW
1.5e-4, one learning rate, fp32, 50 epochs (GR) / 120 (tool-hang).

## Planning (all with the frozen model)

- Reactive: sample the policy, execute 5 actions.
- roll5 (best-of-K rollout): K = 32 policy samples; each rolled autoregressively through the
  dynamics (imagined latent slides into the history) to the goal time; cost = ‖ẑ_goal-time −
  z_g‖²; execute the winner's first block, replan.
- Gradient planner: warm start = roll5's best plan, then 50 Adam steps on the 25 action values
  through the differentiable rollout (warm start kept as a floor).
- SteerMPC: 20 Adam steps on an offset δ added to the goal latent fed to the policy, through the
  frozen sampler + rollout; every candidate remains a policy sample; cost uses the true goal.

## Results

PushT (2×2 arms: state head ∈ {MSE, flow} × SIGReg ∈ {0.04, none}; one training seed each):

| planner            | MSE+SIGReg | MSE noreg | flow+SIGReg | flow noreg |
|--------------------|-----------|-----------|-------------|-----------|
| reactive           | 72.0      | 68.7      | 72.7        | 73.3      |
| roll5              | 85.3      | 83.3      | 80.0        | 72.0      |
| gradient           | **95.3**  | 88.7      | 81.3        | 70.0      |
| gradient + trust-region | 88.0 | 83.3      | 76.7        | 76.0      |
| SteerMPC           | **94.0**  | 84.7      | (OOM, unrun)| (unrun)   |

Tool-hang (reactive only; TC has no goal input): flow-noreg 86.0 > MSE-noreg 80.7 > flow-SIGReg
70.7 > MSE-SIGReg 54.7 — and MSE-SIGReg has the LOWEST behavior-cloning loss of all four
(0.189 vs 0.266): open-loop loss and closed-loop success disagree.

Dynamics quality, independent of planning: an oracle test (expert's remaining demo actions vs 31
wrong sequences, dynamics alone picks by imagined distance-to-goal at the goal time) recovers
the demo: pushT SR 96 (random rivals) / 98 (other demos' sequences), expert picked 96–98% per
on-demo replan for the MSE heads. Pixel-space visualization (a small decoder trained per
checkpoint, also on dynamics-rolled latents; 8 autoregressive imagined blocks vs the copy-last-
latent baseline): MSE+SIGReg imagines 20–200× better than freezing the latent on both cells and
visually tracks the scene; MSE-noreg ~10× better on tool-hang but episode-dependent on pushT;
flow+SIGReg 4–10× with per-step noise spikes; flow-noreg is WORSE than freezing (its imagination
is fog — on pushT its decoded ground-truth latents are already blurry). Linear readout probes of
the encoders (agent/block position, angle) are roughly equal across arms and roughly equal to
LeWM's and unified's — the differences above are not explained by linear state content.

## The puzzle

1. The architecture decouples the heads, yet SIGReg — which is what makes the world model real —
   costs the *reactive policy* 9–26 points everywhere (worst on precision cells), while the
   collapse-adjacent no-reg latent is the policy's best and the world model's worst. The only
   shared object is the encoder: the working hypothesis is gradient conflict there (a Pareto
   frontier between BC-shaped and dynamics-shaped features; "isotropy forces nuisance variance
   into the latent" is one story, unproven). Open: why can't the encoder satisfy both — in
   principle a latent supporting good dynamics contains what BC needs?
2. Flow-head dynamics look acceptable on probes yet plan poorly: their imagination is a sample
   (per-step stochasticity makes the planning cost noisy) and gradients through the 8-step ODE
   are ill-conditioned. Candidate fix: state-to-state flow (source = z_t instead of Gaussian,
   optionally z_t + σε), implemented but unrun; known risk: velocity ≡ 0 becomes a trivial
   minimum (collapse), possibly held off only because the state gradient reaches the encoder
   through attention alone.
3. With the goal entering only the planner's cost (not the model), verifier-style planning needs
   no goal-conditioned training — so subgoal planning on TC (cost = encoded demo frame 6–8
   blocks ahead) is runnable on existing checkpoints and would test whether the MSE+SIGReg
   tool-hang model (reactive 54.7, best dynamics) recovers toward 86 when the planner supplies
   what its policy lacks.

## Questions for discussion

- Optimization view: is the BC-vs-dynamics conflict on the shared encoder fundamental (Pareto)
  or an artifact of the isotropy constraint / loss scales? What would resolve a saddle the two
  objectives cannot agree on (gradient surgery, loss-scale scheduling, constraint on a
  projection rather than the trunk output, two-timescale training)?
- Anti-collapse that rewards planning: alternatives to isotropy that keep the latent
  discriminative for imagined-vs-real comparison (e.g., InfoNCE next-state prediction —
  discrimination is literally the planning operation) without reshaping the policy's input.
- Is stochastic dynamics ever wanted here (environments are deterministic), or should the state
  head be deterministic by design and flow reserved for the policy?
