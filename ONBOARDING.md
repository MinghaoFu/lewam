# LeWAM — New-Machine Onboarding

Get a fresh machine from **zero → running LeWAM training/eval**. Read top-to-bottom: env setup is ~20 min, plus the data pull. The deeper context lives in `docs/` (pointers at the bottom).

---

## 0. What this is (60-second context)

**LeWAM** is a latent **J**oint-**E**mbedding **P**redictive **A**rchitecture **W**orld-**A**ction **M**odel, built on **LeWM** (LeWorldModel, arXiv 2603.19312 — the encoder-*learnable*, SIGReg-regularized JEPA world model from pixels). On top of LeWM's plannable latent we add a **planning-free, history/goal/horizon-conditioned action head**, so **one trained model** serves three control settings: behavior cloning, goal-conditioned policy, and CEM planning. The contribution is the **efficiency Pareto** (planning-free ≈ sampling-based search at ~1104× fewer FLOPs/action) + a per-task-family analysis — *not* a SOTA success rate.

**The operative code is three things:**
- `jepa.py` — the model, ONE class `JEPA` (`encode`→latent z; `predict`=forward dynamics/FDM; `predict_intention`=action head/IDM; `intention_rollout`=the actor).
- `train.py` — the loss, `lejepa_forward` (pred + λ·SIGReg + intent + act + optional w_cyc / goal-cond / horizon / **w_inv**).
- `config/train/lewm.yaml` — **every setting is a flag under `action_pred.*`** (defaults = vanilla LeWM, byte-identical).

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
python train.py data=tworoom trainer.max_epochs=1 +trainer.limit_train_batches=20
```
If it dies on `metrics.csv` → `SPT_CACHE_DIR` isn't set (§4). If on a missing dataset → §3/§4 path.

## 7. Real runs — the settings ARE the config (`config/train/lewm.yaml` → `action_pred.*`)

One model, three settings, all config-driven (defaults off → vanilla LeWM):
```bash
# behavior-cloning / base (raw-direct: w_intent=0, detach_decoder=false):
python train.py data=<task> action_pred.enabled=true
# planning-free goal-conditioned policy:
python train.py data=<task> action_pred.enabled=true action_pred.goal_conditioned=true \
       action_pred.goal_dropout=0.5 action_pred.horizon_conditioned=true
# eval any trained model in planning / GC / BC:
python eval_gip.py --config-name <task> +gip_eval.mode=planning            # CEM
python eval_gip.py --config-name <task> +gip_eval.mode=policy +gip_eval.goal_conditioned=true   # GC
python eval_gip.py --config-name <task> +gip_eval.mode=policy +gip_eval.goal_conditioned=false  # BC
```
Tasks: `pusht`, `tworoom`, `ogb`(cube), `dmc`(reacher). **Ablation flags** (all config-gated, default-off):
`w_intent`/`detach_decoder` (predict-embedding vs raw-direct), `use_action_history` (drop past-action stream), `history_size` (Markovian=1), `loss.sigreg.weight` (anti-collapse on/off), `w_cyc` (FDM↔IDM consistency), **`w_inv`** (SMWM-style inverse-dynamics anti-collapse — can replace SIGReg via `loss.sigreg.weight=0`), `head` (mse|gmm|diffusion), encoder (vit-tiny scratch vs `model=lewm_dinov2`).

## 8. Code map

| path | what |
|---|---|
| `jepa.py` | the model (`JEPA`): encode / predict (FDM) / predict_intention (action head) / intention_rollout / inverse head |
| `train.py` | `lejepa_forward` — the full loss (pred+sigreg+intent+act+w_cyc+goal+horizon+w_inv) |
| `config/train/lewm.yaml` | all settings via `action_pred.*`; `config/train/{model,data,launcher}/` |
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

## Pointers (read in `docs/`)
- `docs/CLAUDE.md` — project rules + the lineage (LeWM, not DINO-WM, not tcwm).
- `docs/EXPERIMENTS.md` — every experiment: motivation, exact config, per-seed results, verdict.
- `docs/results/lewam_equations.html` — the model equations ↔ code, per setting (open in a browser).
- `docs/MACHINE_TRANSFER.md` — full asset inventory + rsync/HF transfer commands.
- `docs/proposal/` — paper positioning notes + the literature survey (incl. the SMWM relationship).
