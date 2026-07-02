# LeWAM — New-Machine Onboarding

Get a fresh machine from **zero → running LeWAM training/eval**. Read top-to-bottom: env setup is ~20 min, plus the data pull. The deeper context lives in `docs/` (pointers at the bottom).

---

## 0. What this is (60-second context)

**LeWAM** is a latent **J**oint-**E**mbedding **P**redictive **A**rchitecture **W**orld-**A**ction **M**odel, built on **LeWM** (LeWorldModel, arXiv 2603.19312 — the encoder-*learnable*, SIGReg-regularized JEPA world model from pixels). On top of LeWM's plannable latent we add a **planning-free, history/goal/horizon-conditioned action head**, so **one trained model** serves three control settings: behavior cloning, goal-conditioned policy, and CEM planning. The contribution is the **efficiency Pareto** (planning-free ≈ sampling-based search at ~1104× fewer FLOPs/action) + a per-task-family analysis — *not* a SOTA success rate.

**The operative code is three things:**
- `jepa.py` — the model, ONE class `JEPA` (`encode`→latent z; `predict`=forward dynamics/FDM; `predict_intention`=action head/IDM; `intention_rollout`=the actor).
- `train.py` — the loss, `lejepa_forward` (pred + λ·SIGReg + intent + act + optional w_cyc / goal-cond / horizon / **w_inv**).
- `configs/train/lewm.yaml` — **every setting is a flag under `action_pred.*`** (defaults = vanilla LeWM, byte-identical).

⚠️ It is built on **LeWM, NOT DINO-WM and NOT the old "tcwm/MT-JEPA" method** — see `docs/CLAUDE.md`.

---

## 1. Clone

```bash
git clone git@github.com:MinghaoFu/lewam.git && cd lewam
```

## 2. Recreate the environment

```bash
python3 -m venv .venv && source .venv/bin/activate
# (a) install torch matching YOUR CUDA first, e.g. cu121:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# (b) the rest (stable-pretraining 0.1.7, hydra, omegaconf, einops, lance, huggingface_hub, mujoco, ...):
pip install -r requirements.txt
# NOTE: requirements.txt has two `-e /var/lib/docker/.../robomimic|robosuite` lines pinned to the OLD machine —
#       remove/ignore them here; install robomimic/robosuite separately only if you run robomimic tasks (§5).
# verify:
python -c "import torch; print('cuda', torch.cuda.is_available()); import stable_pretraining, hydra, omegaconf; print('ok')"
```

## 3. Get the data + checkpoints (NOT in this repo — 425 GB data + 111 GB ckpts)

Pick a disk with space and call it `<DATA>`. Two ways:

- **HuggingFace** (once uploaded — *pending*): 
  ```bash
  huggingface-cli download MinghaoFu/lewam-data  --repo-type dataset --local-dir <DATA>/.stable-wm/datasets
  huggingface-cli download MinghaoFu/lewam-ckpts --repo-type model   --local-dir <DATA>/.stable-wm
  ```
  (private repos → set the HF token first: `huggingface-cli login`; token is in the personal vault.)
- **rsync from the source box** (current home) — exact commands + sizes in `docs/MACHINE_TRANSFER.md §2`.

**Minimum to run the 4-env LeWM campaign (~260 GB):** `ogbench`(cube) + `reacher.h5` + `pusht_expert_train.h5` + `tworoom.h5` under `.stable-wm/datasets/`, plus the SIGReg bases `.stable-wm/decoders/{cube,pusht,reacher,tworoom}_ours_lewm_weights.pt`. Skip `xinyue_mix` (146 GB) + robomimic unless you need those tasks.

## 4. Point the code at the data — CRITICAL (the codebase hardcodes paths)

The code assumes the layout `/mnt/data_nvme1/minghao.fu/.stable-wm`. Fix it ONE of two ways:
- **env var (cleanest):** `export STABLEWM_HOME=<DATA>/.stable-wm` (governs `get_cache_dir`), or
- **symlink tree** (zero-copy) — `docs/MACHINE_TRANSFER.md §3a`.

And ALWAYS export the **disk-safe redirects** every run (a missing one = a disk-full crash mid-training):
```bash
export STABLEWM_HOME=<DATA>/.stable-wm  XDG_CACHE_HOME=<DATA>/cache  TMPDIR=<DATA>/tmp \
       MPLCONFIGDIR=<DATA>/mpl  HF_HOME=<DATA>/hf  SPT_CACHE_DIR=<DATA>/spt \
       HF_HUB_OFFLINE=0  MUJOCO_GL=egl
```
⚠️ **`SPT_CACHE_DIR` is the #1 gotcha** — stable-pretraining writes its ~855 MB Lightning `.ckpt` + `metrics.csv` there, and the *other* vars do NOT govern it. Miss it and training crashes at epoch 1 with `FileNotFoundError: metrics.csv`.

## 5. (robomimic tasks only) the sim env

`robomimic` + `robosuite` are editable installs. Get their source (`docs/MACHINE_TRANSFER.md`) and `pip install -e robomimic robosuite`. Eval must use the `eval_*_robomimic.py` entrypoints (they register the robosuite env). **Skip this entirely if you only run the 4 LeWM envs.**

## 6. Smoke test (proves it's runnable)

```bash
# tiny 1-epoch train on the fastest env; should print pred/sigreg losses + save a checkpoint:
python scripts/train.py data=tworoom trainer.max_epochs=1 +trainer.limit_train_batches=20
```
If it dies on `metrics.csv` → `SPT_CACHE_DIR` isn't set (§4). If on a missing dataset → §3/§4 path.

## 7. Real runs — the settings ARE the config (`configs/train/lewm.yaml` → `action_pred.*`)

One model, three settings, all config-driven (defaults off → vanilla LeWM):
```bash
# behavior-cloning / base (raw-direct: w_intent=0, detach_decoder=false):
python scripts/train.py data=<task> action_pred.enabled=true
# planning-free goal-conditioned policy:
python scripts/train.py data=<task> action_pred.enabled=true action_pred.goal_conditioned=true \
       action_pred.goal_dropout=0.5 action_pred.horizon_conditioned=true
# eval any trained model in planning / GC / BC:
python scripts/eval_gip.py --config-name <task> +gip_eval.mode=planning            # CEM
python scripts/eval_gip.py --config-name <task> +gip_eval.mode=policy +gip_eval.goal_conditioned=true   # GC
python scripts/eval_gip.py --config-name <task> +gip_eval.mode=policy +gip_eval.goal_conditioned=false  # BC
```
Tasks: `pusht`, `tworoom`, `ogb`(cube), `dmc`(reacher). **Ablation flags** (all config-gated, default-off):
`w_intent`/`detach_decoder` (predict-embedding vs raw-direct), `use_action_history` (drop past-action stream), `history_size` (Markovian=1), `loss.sigreg.weight` (anti-collapse on/off), `w_cyc` (FDM↔IDM consistency), **`w_inv`** (SMWM-style inverse-dynamics anti-collapse — can replace SIGReg via `loss.sigreg.weight=0`), `head` (mse|gmm|diffusion), encoder (vit-tiny scratch vs `model=lewm_dinov2`).

## 8. Code map

| path | what |
|---|---|
| `jepa.py` | the model (`JEPA`): encode / predict (FDM) / predict_intention (action head) / intention_rollout / inverse head |
| `train.py` | `lejepa_forward` — the full loss (pred+sigreg+intent+act+w_cyc+goal+horizon+w_inv) |
| `configs/train/lewm.yaml` | all settings via `action_pred.*`; `configs/train/{model,data,launcher}/` |
| `eval_gip.py` | planning / GC / BC eval for the LeWM envs |
| `eval_*_robomimic.py` | robomimic entrypoints (register the sim env) |
| `gcidm.py` | the GC-IDM baseline (Markovian) |
| `bench_speed.py` | the FLOPs/latency benchmark (the efficiency headline) |
| `docs/` | the context (see Pointers) |

## 9. Gotchas (learned the hard way — full list in `docs/CLAUDE.md`)
- **Path hardcoding + `SPT_CACHE_DIR`** (§4) — the most common crash.
- **Disk:** redirect everything to a disk with space; prune ckpts to newest-3; `+ckpt_every=10`.
- **GPU:** co-locate several runs per GPU (models are ~16–25 GB), BUT the real bottleneck is **CPU/disk** — over-concurrency makes **evals OOM-crash**. Run evals with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and low eval concurrency (≤4).
- **Same box for train + eval** (robomimic re-renders on the eval GPU → cross-GPU mismatch).
- **N=50** for any headline eval; 3 seeds for headline numbers.


---

## 10. Open experiment queue — run these on the new box

Everything below was **stopped mid-flight** on L40S/174 (the OOM episode); trained checkpoints for some arms still exist on those boxes, but on a fresh multi-GPU box treat them all as **to-run**. Co-locate 2–3 arms per GPU (models are ~16–25 GB); run arms in parallel, not serially. Use the **§4 env block** for every launch (add `hydra.run.dir=$DATA/hydra/<name>`).

**Eval protocol for every headline number: `eval.num_eval=50`, 3 seeds `{42,0,1}`, train + eval on the SAME box** (robomimic re-renders on the eval GPU → cross-GPU SR mismatch). The fast-iter budgets get an arm *near* convergence to compare cheaply; **the winning arm of each ablation is then re-run to full convergence (~100ep, verify the val-plateau) + 3-seed N=50 for the paper (rule 8).**

**Task ↔ config mapping** (train `data=` differs from eval `--config-name` for cube/reacher):

| task | train `data=` | eval `--config-name` | max_epochs | limit_train_batches | ckpt_every |
|---|---|---|---|---|---|
| pusht | `pusht` | `pusht` | 15 | 2000 | 5 |
| tworoom | `tworoom` | `tworoom` | 12 | 4000 | 10 |
| cube | `ogb_lance` | `cube` | 15 | 2000 | 5 |
| reacher | `dmc` | `reacher` | 15 | 1000 | 10 |
| can / lift / square | `robomimic_<task>` | (use `eval_histbc_robomimic.py`) | 30 | 1000 | 5 |
| tool_hang / transport | `robomimic_<task>` | (use `eval_histbc_robomimic.py`) | 40 | 1000 | 5 |

⚠️ **Within one dataset, every arm of an ablation MUST share `max_epochs` AND `limit_train_batches`** (rule 12) — a budget that varies between arms confounds the ablation with training compute and is invalid.

---

### A. SIGReg-redundancy / anti-collapse — the core scientific question
**Q: does the action-prediction head make latent SIGReg redundant — can we drop SIGReg without collapse or SR loss?** Per dataset, 3 arms. Shared base: vit-tiny-192 from scratch, `action_pred.enabled=true w_act=1 w_intent=1 detach_decoder=false head=mse use_action_history=true history_size=3`, `freeze_wm=false`.

| arm | override vs base | tests |
|---|---|---|
| A `base` | `loss.sigreg.weight=0.09` | reference |
| B `sigreg_off` | `loss.sigreg.weight=0` | does z stay full-rank with NO SIGReg? |
| C `sigreg_act` | `loss.sigreg.weight=0.09 action_pred.sigreg_act=true action_pred.detach_target=false` | anti-collapse on the action embeddings instead |

Base command (fill `<data>`/`<ep>`/`<batches>` from the table; arm sets `<sigw>`/extras):
```bash
python scripts/train_sigreg.py data=<data> \
  action_pred.enabled=true action_pred.w_act=1.0 action_pred.w_intent=1.0 \
  action_pred.detach_decoder=false action_pred.head=mse \
  model.use_action_history=true history_size=3 \
  loss.sigreg.weight=<sigw> init_from=null freeze_wm=false embed_dim=192 \
  output_model_name=sigreg_<arm>_<ds>_scratch subdir=sigreg_<arm>_<ds>_scratch \
  trainer.max_epochs=<ep> trainer.devices=[0] \
  +trainer.limit_train_batches=<batches> +trainer.limit_val_batches=20 \
  +ckpt_every=10 loader.batch_size=64 num_workers=4 hydra.run.dir=$DATA/hydra/sigreg_<arm>_<ds>
```
**Datasets:** pusht, tworoom, cube, reacher (4 × 3 arms = 12 trainings). **Eval:** `eval_gip.py --config-name <cfg> policy=<name> +gip_eval.mode=<planning|bc> seed=<42|0|1> eval.num_eval=50`. For arm B also `python measure_collapse_generic.py <name>` (z_std, erank, per-dim std).

**Clean variant (§31-clean — removes the history-copyability confound):** rerun arms A & B with `model.use_action_history=false` on **cube + reacher**. Decision: if sigreg-OFF stays full-rank AND SR ≈ sigreg-ON *even with history zeroed*, the action-prediction loss anchors the latent and **SIGReg is redundant** — the headline claim.

### B. Inverse-dynamics term (SMWM-inspired) — the "replace SIGReg" path — **NOT STARTED**
**Q: can a dense inverse-dynamics term `‖h(z_τ,z_{τ+1}) − a_τ‖²` over the context window REPLACE SIGReg (anti-collapse) and/or IMPROVE SR?** (cf. SMWM arXiv 2606.20104. The code is already in `train.py`/`jepa.py`, gated by `action_pred.w_inv`; default 0 = byte-identical.) Shared base: end-to-end GC (`action_pred.enabled=true w_act=1 w_intent=0 goal_conditioned=true goal_dropout=0.5 horizon_conditioned=true`), vit-tiny-192 scratch.

| arm | override vs base | tests |
|---|---|---|
| `idm_dense_on` | `loss.sigreg.weight=0.09 action_pred.w_inv=0.5 action_pred.inv_mode=dense action_pred.inv_target=encoded` | does inverse IMPROVE on top of SIGReg? |
| `idm_dense_sigregoff` | `loss.sigreg.weight=0 action_pred.w_inv=0.5 action_pred.inv_mode=dense action_pred.inv_target=encoded` | does inverse REPLACE SIGReg? |
| `idm_dense_A8` | `loss.sigreg.weight=0 action_pred.w_inv=0.5 action_pred.inv_mode=dense action_pred.inv_target=predicted` | cycle-consistent (inverse reads action off the FDM's own rollout ẑ) |

```bash
python scripts/train_sigreg.py data=<data> \
  action_pred.enabled=true action_pred.w_act=1.0 action_pred.w_intent=0.0 \
  action_pred.detach_decoder=false action_pred.head=mse \
  model.use_action_history=true history_size=3 \
  loss.sigreg.weight=<0.09|0> \
  action_pred.w_inv=0.5 action_pred.inv_mode=dense action_pred.inv_target=<encoded|predicted> \
  action_pred.goal_conditioned=true action_pred.goal_dropout=0.5 action_pred.horizon_conditioned=true \
  init_from=null freeze_wm=false embed_dim=192 \
  output_model_name=ov_<arm>_<ds> subdir=ov_<arm>_<ds> \
  trainer.max_epochs=<ep> trainer.devices=[0] \
  +trainer.limit_train_batches=<batches> +trainer.limit_val_batches=20 \
  +ckpt_every=5 loader.batch_size=64 num_workers=2 hydra.run.dir=$DATA/hydra/ov_<arm>_<ds>
```
**Datasets:** pusht, cube, tworoom, reacher (4 × 3 = 12). **Baselines (reuse, don't rerun):** the §A `base` (SIGReg-on) and `sigreg_off`. **Eval:** `eval_gip.py` planning + bc + `policy +gip_eval.goal_conditioned=true`, N=50 ×3; collapse measurement on the `sigregoff` arm.

### C. End-to-end ablation matrix (rule 13) — one knob off the end-to-end baseline
The §32 7-arm × 4-dataset sweep — `freeze_wm=false` throughout, each arm flips exactly ONE knob off `base`. Shared base: `action_pred.enabled=true w_intent=0 w_act=1 detach_decoder=false head=mse use_action_history=true history_size=3 loss.sigreg.weight=0.09 goal_conditioned=true goal_dropout=0.5 horizon_conditioned=true`, vit-tiny-192 scratch.

| arm | override vs base | axis |
|---|---|---|
| `base` | — | reference (end-to-end) |
| `enc_dino` | `model=lewm_dinov2 model.encoder.pretrained=true embed_dim=384` | pretrained DINOv2-small encoder |
| `input_stateonly` | `model.use_action_history=false` | drop the past-action stream |
| `sigreg_off` | `loss.sigreg.weight=0` | anti-collapse on/off |
| `predict_emb` | `action_pred.w_intent=1 action_pred.detach_decoder=true` | predict-embedding vs raw-direct |
| `input_markov` | `history_size=1` | Markovian vs history-conditioned |
| `head_diffusion` | `action_pred.head=diffusion` | action-head architecture |

Train with `train.py` (same base flags as §B minus w_inv) + the arm override; **datasets** pusht/cube/tworoom/reacher (7 × 4 = 28 trainings). **Eval:** `eval_gip.py --config-name <cfg> policy=<arm>_<ds> +gip_eval.mode=<bc|guided|planning> eval.num_eval=50 seed=<42|0|1>`.

### D. Encoder capacity — DINOv2-large (§29)
**Q: is the robomimic can-GC floor (~0.26) an encoder-capacity bottleneck?** DINOv2-large (1024-d, 5.3× wider than vit-tiny CLS), GC head, end-to-end finetune:
```bash
python scripts/train.py data=robomimic_can model=lewm_dinov2 \
  model.encoder.pretrained=true model.encoder.model_name=facebook/dinov2-large \
  embed_dim=1024 loader.batch_size=24 freeze_wm=false \
  action_pred.enabled=true action_pred.goal_conditioned=true \
  action_pred.horizon_conditioned=true action_pred.goal_dropout=0.5 action_pred.hindsight_max_k=50 \
  output_model_name=can_gc_dinov2L subdir=can_gc_dinov2L \
  trainer.max_epochs=100 trainer.devices=[0] +trainer.limit_train_batches=300 +ckpt_every=10
```
Eval bc + clean-GC (`eval_histbc_robomimic.py ... world.task=PickPlaceCan eval.dataset_name=can +gip_eval.mode=policy +gip_eval.goal_conditioned=<false|true> +eval.goal_mode=terminal eval.num_eval=50`). **Extend to lift/square** if it lifts the floor. Decision: DINOv2-large bc > 0.71 → capacity *is* the bottleneck; GC > 0.26 → encoder helps contact-state localization; both flat → it's the GC formulation / occluded-state difficulty, not encoder size.

### E. Scalability size-curve — the **SCALABLE** headline (from the positioning)
Model-size curve **embed_dim ∈ {192, 384, 768}** (vit width), ≥3 seeds × N=50 × same-box, **plus a DOWN baseline contrast** (a variant whose SR does NOT improve — or degrades — with size, so monotone-up is shown to be non-trivial). Run per-task AND on the **80-task pad+mask multi-task agent**. This is the headline evidence for "one recipe scales with size" (TD-MPC2 sense). Set `embed_dim=<192|384|768>` (and `model=lewm_dinov2` for the pretrained variants); keep all else fixed across sizes.

### F. Efficiency / FLOPs benchmark — the **FAST** headline (`bench_speed.py`)
Run `python bench_speed.py` to measure **wall-clock/action + FLOPs/action** for the planning-free GC policy (1 forward pass) vs CEM planning (population × horizon × iters). The ~1104× cheaper claim needs the measured numbers — report both and the ratio. This is the efficiency-Pareto headline (no current JEPA-WM baseline reports latency, so it's uncontested — measure it cleanly).

### G. Robomimic heterogeneous-data axis — LDP (#59)
xinyue abs-action robomimic: DP baseline + LDP suboptimal-data recipe + more-data re-run (the "action-free + suboptimal data" story where latent-planning's gap over DP opens up, 0.65→0.95). Entry point `eval_histbc_robomimic.py` for the GC/bc evals; see `docs/EXPERIMENTS.md` §LDP and the `reference_ldp` memory for the exact data recipe.

---

**Suggested order on a fresh box:** B (inverse — not started, freshest question) and A-clean (the SIGReg-redundancy headline) first; then C (the rule-13 matrix) in parallel across GPUs; D/E/F/G as the box frees up. Update `docs/EXPERIMENTS.md` (per-seed numbers + verdict) and the unified table (`docs/results/master_results.html` via `results/gen_master_table.py`) as each lands (rules 7 + 11).
## Pointers (read in `docs/`)
- `docs/CLAUDE.md` — project rules + the lineage (LeWM, not DINO-WM, not tcwm).
- `docs/EXPERIMENTS.md` — every experiment: motivation, exact config, per-seed results, verdict.
- `docs/results/lewam_equations.html` — the model equations ↔ code, per setting (open in a browser).
- `docs/MACHINE_TRANSFER.md` — full asset inventory + rsync/HF transfer commands.
- `docs/proposal/` — paper positioning notes + the literature survey (incl. the SMWM relationship).
