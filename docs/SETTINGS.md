# Settings & Equations

Every variant in this repo is selected by config flags on **one** training config
(`configs/train/lewm.yaml`) — there is a single master switch, `action_pred.enabled`,
plus a handful of flags that each add one term to the loss or one input to the head.
This file states, for each flag, **what config controls it** and **the equation it adds**,
so the difference between settings is visible at a glance.

Run any setting with Hydra overrides, e.g.:
```bash
python scripts/train.py data=pusht action_pred.enabled=true action_pred.goal_conditioned=true
```

---

## Notation

| symbol | meaning |
|---|---|
| $x_t$ | pixel observation at obs-step $t$ |
| $z_t = E_\theta(x_t)$ | latent from the ViT encoder $E_\theta$ |
| $\hat z_{t+1} = P_\phi(z_{\le t}, e_{<t})$ | forward dynamics (FDM / world-model predictor $P_\phi$) rollout |
| $x_g,\; z_g = E_\theta(x_g)$ | hindsight **goal** frame and its latent |
| $a_t$ | ground-truth action block (`frameskip` raw actions) |
| $e_t = A_\psi(a_t)$ | action embedding (action encoder $A_\psi$) |
| $h,\; \bar h = \min(h, H_{\max})/H_{\max}$ | remaining horizon to the goal, and its normalization |
| $\Pi$ | intention/action head; $\iota$ its intention embedding, $\hat a$ its decoded action |
| $I_\omega$ | inverse-dynamics head |
| $\mathrm{sg}[\cdot]$ | stop-gradient |

---

## Base world model — `action_pred.enabled: false` (default)

Plain LeWM: next-embedding prediction + a Gaussian regularizer on the latent.

$$\mathcal{L}_{\text{LeWM}} \;=\; \underbrace{\lVert \hat z_{t+1} - \mathrm{sg}[z_{t+1}] \rVert^2}_{\mathcal{L}_{\text{pred}}} \;+\; \lambda_{\text{reg}}\,\mathcal{L}_{\text{SIGReg}}(z)$$

- $\lambda_{\text{reg}}$ = `loss.sigreg.weight` (default **0.09**).
- No policy is trained; the model is used with CEM planning.

Everything below is **added on top** of $\mathcal{L}_{\text{LeWM}}$ when `action_pred.enabled: true`.

---

## The action head — `action_pred.enabled: true`

Predict the action from the latent history (+ optional goal / horizon):

$$(\iota,\; \hat a) \;=\; \Pi\big(z_{\le t},\, e_{<t};\; z_g,\, \bar h\big)$$

$$\mathcal{L} \mathrel{+}= \; w_{\text{act}}\underbrace{\lVert \hat a - a_t \rVert^2}_{\mathcal{L}_{\text{act}}} \; + \; w_{\text{int}}\underbrace{\lVert \iota - \mathrm{sg}[e_t] \rVert^2}_{\mathcal{L}_{\text{int}}}$$

| flag | default | effect |
|---|---|---|
| `w_act` | 1.0 | weight on decoded-action MSE $\mathcal{L}_{\text{act}}$ |
| `w_intent` | 0.0 | weight on intention↔action-embedding MSE $\mathcal{L}_{\text{int}}$ (JEPA-style) |
| `detach_target` | true | $\mathrm{sg}[\cdot]$ on the intention target (anti-collapse) |
| `ema_target` | false | target $e_t \leftarrow A_\psi^{\text{EMA}}(a_t)$ (BYOL/I-JEPA momentum encoder) |
| `head` | mse | decoder: `mse` (MLP point est.) · `gmm` (mixture NLL) · `diffusion` (DDPM) |

For `head=gmm`/`diffusion`, $\mathcal{L}_{\text{act}}$ is replaced by the decoder's own NLL/DDPM loss.

### Goal conditioning — `action_pred.goal_conditioned: true`
Adds the hindsight goal latent $z_g$ into the head (additive on the past-action stream). The goal is sampled by `GoalSamplingDataset`: a frame $h$ obs-steps ahead, $h\sim\mathcal{U}[1,\texttt{hindsight\_max\_k}]$ (default 50).

$$\Pi(\dots;\, z_g,\,\cdot)\quad\text{with}\quad z_g = E_\theta(x_g)$$

- `goal_dropout` $=p$: zero $z_g$ per-sample w.p. $p$, so **one** head learns both the goal-conditioned and goal-agnostic policy (classifier-free guidance). Default 0.
- Off (default): $z_g$ absent → history-conditioned BC policy.

### Horizon conditioning (OURS GC head) — `action_pred.horizon_conditioned: true`
AdaLN-Zero modulates the intention embedding by the normalized remaining horizon:

$$\iota \;\leftarrow\; \mathrm{AdaLN\text{-}Zero}(\iota,\, \bar h),\qquad \bar h = \min(h, H_{\max})/H_{\max}$$

- `horizon_H_max` (default 50) sets $H_{\max}$ (must match across compared arms).
- Off (default): identity (horizon-agnostic).

---

## Inverse dynamics (SMWM anti-collapse) — `action_pred.w_inv > 0`

Recover the action from a latent pair; forces action info into the latent (can replace SIGReg by also setting `loss.sigreg.weight=0`).

$$\mathcal{L} \mathrel{+}= \; w_{\text{inv}}\underbrace{\big\lVert I_\omega\big([z_\tau \,\Vert\, z_{\tau+1}]\big) - a_\tau \big\rVert^2}_{\mathcal{L}_{\text{inv}}}$$

| flag | default | effect |
|---|---|---|
| `w_inv` | 0.0 | weight; $>0$ builds $I_\omega$ and adds $\mathcal{L}_{\text{inv}}$ |
| `inv_target` | encoded | `encoded`: $z_{\tau+1}=E_\theta(x_{\tau+1})$ · `predicted`: $z_{\tau+1}=\hat z_{\tau+1}$ (FDM rollout — the **A8 / cycle** variant) |
| `inv_mode` | dense | `dense`: mean over all consecutive pairs in the window · `last`: final pair only |

---

## FDM↔IDM consistency — `action_pred.w_cyc > 0`

Roll the FDM one step with the head's **predicted** action $\hat a$ and require it to reach the true next latent — couples forward and inverse heads end-to-end.

$$\mathcal{L} \mathrel{+}= \; w_{\text{cyc}}\underbrace{\big\lVert P_\phi(z_t, \hat a) - \mathrm{sg}[z_{t+1}] \big\rVert^2}_{\mathcal{L}_{\text{cyc}}}$$

Default `w_cyc=0`.

---

## End-to-end vs frozen — `freeze_wm`

- `freeze_wm: false` (default): encoder $E_\theta$ + predictor $P_\phi$ train **jointly** with the head (end-to-end LeWAM).
- `freeze_wm: true`: $E_\theta, P_\phi$ frozen; only the head (and $I_\omega$) train — the frozen-latent protocol. The cached-latent fast trainers are `scripts/train_ours_gc.py` (OURS head) and `scripts/train_gcidm.py` (GC-IDM baseline).

### GC-IDM baseline (Markov, frozen) — `scripts/train_gcidm.py` (`lewam/models/gcidm.py`)
No history, no FDM: a goal+horizon-conditioned head on a single frozen latent.

$$\hat a = G_\eta\big([z_t \,\Vert\, z_g],\, \bar h\big),\qquad \mathcal{L} = \lVert \hat a - a \rVert^2,\quad z=E_\theta(x)\ \text{frozen}$$

The wedge vs OURS: OURS conditions on the latent **history** $z_{\le t}$ + past actions; GC-IDM uses only the current $z_t$.

---

## Named settings → config → total loss

| Setting | Config override (on `lewm.yaml`) | Total loss |
|---|---|---|
| **LeWM** (world model only) | *defaults* (`action_pred.enabled=false`) | $\mathcal{L}_{\text{pred}} + \lambda\mathcal{L}_{\text{SIGReg}}$ |
| **BC policy** (history, no goal) | `action_pred.enabled=true` | $\;+\; w_{\text{act}}\mathcal{L}_{\text{act}}\,(+\,w_{\text{int}}\mathcal{L}_{\text{int}})$ |
| **GC policy** (goal-conditioned) | `+ goal_conditioned=true` | …with $z_g$ in $\Pi$ |
| **OURS GC** (goal + horizon) | `+ horizon_conditioned=true horizon_H_max=50` | …with $\mathrm{AdaLN}(\iota,\bar h)$ |
| **+ inverse dynamics** | `+ w_inv=1.0 inv_target=encoded` | $\;+\; w_{\text{inv}}\mathcal{L}_{\text{inv}}$ |
| **+ cycle (A8)** | `+ w_inv=1.0 inv_target=predicted`  *or*  `+ w_cyc=1.0` | $\mathcal{L}_{\text{inv}}$ on FDM target  /  $\;+\; w_{\text{cyc}}\mathcal{L}_{\text{cyc}}$ |
| **GC-IDM baseline** (Markov, frozen) | `scripts/train_gcidm.py` | $\lVert G_\eta([z_t\Vert z_g],\bar h) - a\rVert^2$ |

All flags are byte-identical no-ops when left at default, so each row is exactly the row above **plus one term/input** — that is the whole design.

### Merged backbone (architecture variant)

`model=lewm_merged` (or `model._target_=lewam.models.jepa_merged.MergedJEPA`) ties the action
predictor to the forward-dynamics predictor: **one shared `ARPredictor` backbone** feeds both
readout heads (`pred_proj` → $z_{t+1}$, `action_decoder` → $a_t$), instead of two separate
transformers. The two AdaLN conditioning modes ($a_{\le t}$ for dynamics, $a_{<t}+z_g+\bar h$ for
the action) are unchanged. Same losses; ~one `ARPredictor` fewer parameters (~11 M at vit-tiny).
It is an ablation vs the default separate-backbone runs, and every setting above composes with it.
