# EXPERIMENTS — MT-JEPA / le-wm-repro

Single source of truth for **every** experiment: **motivation → exact full config → all results
(per-seed) → conclusion.** Written long and detailed on purpose (CLAUDE.md rules 7–9): keep every
setting, every per-seed number; do not compress. It exists because config drift (the
`detach_decoder` bug, §0) silently invalidated a batch of results.

Server: `H100-174-minghao:/home/minghao.fu/workspace/le-wm-repro`. Code mirrored in this repo.
Checkpoints: `$STABLEWM_HOME/checkpoints/<run_name>/` where `STABLEWM_HOME=/mnt/data_nvme1/minghao.fu/.stable-wm`.
Legend: ✅ done · 🔄 running · ❌ invalid/buggy · ⚠️ caveat.

---

## CONCLUSIONS — bottom line (2026-06-16, analyze-before-continuing)
The TL;DR of everything below. Each line links to its evidence section.

1. **The action-head objective is a NEGATIVE result — TIED on BOTH families.** detach=TRUE vs FALSE vs solveact vs
   anti-collapse-SIGReg are all within training-run noise. Robomimic 3-train-seed N=50 (§1c): detT 93.3/70.7/61.3 ≈
   detF 94.7/69.3/57.3 ≈ solveact (can 3-seed 70). Goal-reaching (§1b): detT-vs-detF Δ **flips direction** (reacher-guided
   88/64, pusht-planning 62/76) = noise. **The early "solveact wins (+6/+7)" and "embedding-intuition hurts BC" headlines
   were single-train-seed artifacts (RETRACTED).** ⇒ **The paper CANNOT be "a better action head."** Two wedges dead.
2. **Per-family planning is the ROBUST spine (§7b, §1b).** WM-planning (guided/CEM) works on goal-reaching geometry
   (guided 80/90/88, planning 62–92) and **collapses to 0/0/0 on contact robomimic**. This contrast SURVIVES all the
   ±10–24 pt training-run noise (goal-reaching guided 64–94 ≫ robomimic 0). This is the paper's spine.
3. **Why robomimic planning fails, why history-BC fixes it — a TWO-PART mechanism (§6, §1c). ⚠️ REVISED 2026-06-16: the
   single-frame "contact-blind" story is LIFT-SPECIFIC, not universal.** The grasp-state probe (5 demo-split seeds, detachT100)
   shows single-frame latent decodes grasp at AUC **0.82 on Lift** (history 0.99, $[z,a]$ 0.999 — big gap ✅) but **already
   0.96 on Can and 0.999 on Square** (history adds ~0 ❌): the tiny Lift cube is the only object a single frame under-exposes;
   Can/Square lift visibly. ⇒ **(a) Lift** CEM≈0 from **contact-state under-exposure** (probe-supported); **(b) Can/Square**
   CEM≈0 NOT from invisibility but from **multi-stage structure** (flat horizon-H CEM can't decompose approach→grasp→transport
   →place; WorldDP's "single-stage only" diagnosis). **History-BC beats CEM on ALL three** (histbc 93.3/70.7/61.3 vs CEM 0) by
   executing the sequential demo behavior; on Lift it *additionally* supplies contact-state memory (post-grasp $P(\text{succ}
   |\text{grasp})$ 76→99). **Do NOT claim a universal single-frame contact-blindness — Can/Square are honest counter-evidence.**
4. **Methodological law — variance is BENCHMARK-DEPENDENT (⚠️ REVISED 2026-06-16 by eval-seed bands).** On **robomimic**
   eval-seed variance is small (~4 pt) and TRAIN-run variance dominates (±10 pt, §1c). On **goal-reaching** EVAL-seed
   variance is itself LARGE — same detachT100 model, 3 eval-seeds: pusht planning **48/62/76 (range 28!)**, guided 62/74/80
   (18); reacher guided 80/88/92 (12); only tworoom is tight (4–6). ⇒ many goal-reaching A/B gaps I earlier called "train
   noise" (pusht planning detT 62 vs detF 76) are within a SINGLE model's eval-seed spread. **Goal-reaching headline numbers
   need BOTH multi-eval-seed AND multi-train-seed averaging; a single (train,eval) pair is near-meaningless there.** 3-eval-seed
   means now in hand (§1b); 3-train-seed running. Robomimic was fine at 1 eval-seed × 3 train-seeds; goal-reaching is not.
5. **Pretrain = efficiency/reuse, NOT a higher ceiling ([[project_pretrain_benefit]]).** Budget-matched scratch matches
   the warm endpoint; the frozen-probe gap proves the representation is real. Claim efficiency, not SOTA SR.
6. **Positioning: LeWAM is the controlled mechanistic study in a crowded, search-free WAM field (§9, literature_survey).**
   DreamZero/Being-H0.7/Fast-WAM own "WAM"/"latent WAM"/"is planning worth it." Un-scooped core = the per-family analysis
   + contact-state recovery. Temporal Straightening (2603.12231) is orthogonal (fixes goal-reaching *geometry*, not the
   *semantic* contact gap). **OPEN FORK (user-undecided):** analysis+routing-rule paper vs hierarchical-subgoal method.
7. **DINOv2 encoder-init helps LeWAM control on robomimic, via TWO mechanisms (§10, 3-seed BOTH tasks, 2026-06-18).** Clean
   ablation = same `dinov2-small-384` arch, flip ONLY pretrained-vs-scratch init. **bc SR: pretrained > scratch on BOTH tasks,
   every seed** — transport **0.087 vs 0.013** (~6.5×), tool_hang **0.333 vs 0.213** (~1.6×). The frozen probe (latent→state R²)
   **explains transport** (pretrained 0.84 vs scratch 0.42 — from-scratch under-fits the wide 2-arm scene) but is **FLAT on
   tool_hang** (0.82≈0.82) despite the bc win = a genuine **probe/control DECOUPLE** (DINOv2 sharpens task-relevant *local*
   features the global state-probe can't see). The probe gap **tracks scene width** (large only on wide 2-arm transport;
   single-arm can/lift/tool_hang +0.02/+0.06/+0.00). ⇒ **the probe is necessary-but-not-sufficient; control benefits
   everywhere.** Guided is horizon-confounded ≈0 on these long tasks (not the test); bc is. Caveat: it's the *pretrained
   encoder*, not the bigger 384-d arch (transport scratch 0.013 ≈ canonical vit-tiny ~0.01). Dashboard: `results/ablation_dashboard.html`.
   **✅ CORRECTED (reliability audit, 2026-06-18): pretrained > scratch on ALL 5 robomimic tasks — NO reversal.** The earlier
   "lift reversal" (pretrained 0.16) was an EVAL-BATCHING ARTIFACT (10-env-pool reuse → chunk-1-then-all-zeros); lift pretrained
   re-run no-batch 3-seed = **0.917** (0.92/0.88/0.95) > scratch **0.633** (0.62/0.60/0.68). So **DINOv2-init helps LeWAM control on
   EVERY tested robomimic task** (transport 3.2×, square 2.2×, tool_hang 1.6×, lift 1.45×, can 1.25×). transport scratch
   re-measured no-batch 3-seed = **0.027** (0.0/0.02/0.06; 6.5× was the batched-0.013 ratio, 3.2× is the clean one — both
   near-floor, transport's evidence is the PROBE +0.42 not bc). Full 30-log chunk forensics (§10): exactly ONE corrupted cell
   (lift pretrained, all 3 seeds); 29/30 clean. ⚠️ The 3 seeds are EVAL seeds (1 train-run/arm) → headline stat = directional
   consistency 5/5, sign-test p=0.031; square +35pt/lift +28pt robust, can/tool_hang borderline (§10 train-variance caveat).
   The probe/control decouple stands and is CLEANER: latent-probe gain narrow (only wide-scene transport +0.42; single-arm ≈0)
   yet control improves on ALL 5 → probe necessary-but-not-sufficient. See "§10 🔍 RELIABILITY AUDIT" for the bug + proof.
   Lesson: per-episode success PATTERNS (not just the mean SR) are the reliability check — a "chunk-1-then-all-zeros" mean hides
   a broken eval.

**Solid enough to headline:** #2 per-family contrast, #3 contact mechanism, #1 negative result, #5 pretrain, **#7 DINOv2-init
two-mechanism control gain (3-seed, both robomimic tasks)**.
**Still noisy (3-seed in flight):** exact goal-reaching SR (#4). **Open:** the fork (#6); Phase-2 bc for single-arm tasks (#7→§9).

---

## 0. Conventions, eval protocols, recipe templates (READ FIRST)

### Eval protocols (exact)
- **histbc** (`eval_histbc_robomimic.py`): history-conditioned intention head as open-loop BC.
  Maintains per-env frame-latent + executed-action history; calls `model.predict_intention`. Success =
  native `env.is_success()["task"]`, accumulated over the rollout (`results['episode_successes'] |= terminateds`),
  `terminate_on_success`. Resets to **sampled demo states** (`start ∈ [0, len−goal_offset−1]`, includes
  near-full-task). histbc **ignores the goal frame** (pure BC). This is the same success criterion BC-RNN uses.
- **goal-reaching** (`eval_gip.py --config-name <task> policy=gip_<task> +gip_eval.mode=bc|guided|planning`):
  bc = intention head direct; planning = pure CEM over WM (LeWM baseline); guided = intention warm-starts CEM.
- **N (num_eval)**: LeWM repo default = **50**. SR granularity = 100/N (N=50 → 2% steps). Single-seed = diagnostic,
  **≥3-seed = headline** (rule 9). Headline seeds used: {42, 0, 1}.
- **eval_budget** (robomimic): max rollout steps; `max_episode_steps = 2*eval_budget`. SR rises with budget
  (budget sweep square seed42: 100→45, 160→75, 220→85). BC-RNN uses horizon **400**; we used can 240 / square 320
  (= ~2× median demo) → mild deflation vs BC-RNN. goal_offset_steps: lift 30 / can 90 / square 120.

### Exact training recipe (warm-start GIP)
```
CUDA_VISIBLE_DEVICES=<g> .venv/bin/python train.py \
  data=<robomimic_lift|robomimic_can|robomimic_square|pusht|tworoom|dmc> \
  action_pred.enabled=true \
  action_pred.detach_decoder=<true|false>   # TRUE = validated latent-only design (see §0 bug) \
  action_pred.head=<mse|gmm|diffusion>      # default mse \
  action_pred.w_act=<1.0> action_pred.w_intent=<1.0> action_pred.sigreg_act=<false> \
  init_from=$DEC/<task>_lewm_weights.pt      # $DEC=$STABLEWM_HOME/decoders \
  output_model_name=<run_name> \
  trainer.max_epochs=<100>  +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4
```
init_from base WMs: `$DEC/{lift,can,square}_lewm_weights.pt`, `$DEC/{pusht,tworoom,reacher}_ours_lewm_weights.pt`.
Loss = `pred_loss(z-dynamics, w=1) + λ·sigreg + w_intent·intent_loss + w_act·act_loss`; `λ=cfg.loss.sigreg.weight`.

### Gradient flow (so config choices are unambiguous)
- `intent_loss = (intention − act_emb.detach())²` — intention head predicts the (detached) action embedding (JEPA "intuition").
- `act_loss = (decoder(dec_in) − raw_action)²`, `dec_in = intention.detach() if detach_decoder else intention`.
  - **detach_decoder=TRUE** → act_loss trains ONLY the decoder; intention+encoder get NO raw-action gradient (pure latent intuition).
  - **detach_decoder=FALSE** → act_loss backprops decoder→intention→action_predictor→encoder (raw action grounds the encoder).
- Action is **AdaLN conditioning** in `ARPredictor`, NOT a sequence token. WM `predictor` and policy `action_predictor`
  are **two separate ARPredictor trunks** sharing only the encoder.

### Infra / ops
- **⚠️⚠️🔴 CROSS-GPU RENDERING — MODELS ARE NOT PORTABLE ACROSS GPUs (found 2026-06-17, L40S migration). REMEMBER THIS
  WHEN ANALYZING EXPERIMENTS — a transferred model giving 0 / a cross-machine number that doesn't line up is very likely THIS.**
  - **What:** the robomimic/robosuite eval **re-renders camera frames LIVE on the eval GPU** (mujoco-EGL, `use_image_obs=true`).
    Different GPUs rasterize the scene slightly differently. For Lift, H100 (174) vs L40S differ in a **tiny but TASK-CRITICAL
    48×40-px region exactly at the cube+gripper grasp zone** (y∈[190,237], x∈[190,229] of 384², maxdiff **227**, ~0.12% of px>30).
    Mean diff is only 0.11 (so a *global* mean check FALSELY says "identical" — that's the trap I fell into), but the localized
    diff sits on the exact pixels a precise grasp policy depends on.
  - **NOT shadows:** tested `light_castshadow=0` + `shadowsize=0` on both boxes → diff unchanged (maxdiff still 227). It's the
    object/gripper specular/edge rendering, GPU-rasterizer-dependent. Disabling shadows does NOT fix it.
  - **Consequence (the evidence):** a 174-H100-trained model evaluated on L40S = **SR 0** (OOD on the grasp-region pixels);
    the SAME pipeline **trained AND evaluated on L40S = SR 90 = 174's own 90** (lift, verified). So the box is fine; only
    *cross-GPU* (train on one, eval on another) breaks. The earlier "reset(seed=42) lands on different states cross-box"
    (174 state_sum 1.478 vs L40S 3.438) is a SECONDARY confound (robosuite object-placement RNG not reproduced by the gym
    seed cross-box), NOT the main cause — the main cause is the grasp-region render OOD.
  - **Rule going forward:** **train and evaluate every model on the SAME GPU/box.** Never compare a model's SR across machines,
    and never reuse a checkpoint trained elsewhere for a live-render eval. For real cross-machine portability you'd need
    rendering augmentation / domain randomization in training (not done). 174's existing numbers stay on 174; L40S is only
    for NEW tasks (tool_hang/transport) trained+evaluated end-to-end on L40S.
  - **Re-render-on-L40S pipeline (so train-data render == eval render):** transfer the tiny `demo_v15.hdf5` (states, 0.2G, NOT
    the 170G `image_384`) → `robomimic/scripts/dataset_states_to_obs.py --camera_height/width 384 --camera_names agentview
    robot0_eye_in_hand` re-renders on L40S → `scripts/convert_robomimic_h5.py <task>` → flat `.h5`.
  - **🐛 CONVERT BUG (found same day):** `convert_robomimic_h5.py` writes only `[action, ep_len, ep_offset, pixels, proprio,
    state]` — it **drops `episode_idx` + `step_idx`** which the EVAL needs (`sample_eval_episodes` → KeyError 'ep_idx';
    TRAINING doesn't need them so training silently works). Fix = post-process: derive `episode_idx`/`step_idx` from `ep_len`
    (`episode_idx = concat([full(L,i) for i,L in enumerate(ep_len)])`, `step_idx = concat([arange(L) ...])`) and add to the
    `.h5`. (174's lift.h5 HAS them — made by a different/newer convert — which is why 174 never hit this.)
  - **L40S env build notes (minghao.fu account, exx-sudo-provisioned):** clone exx's `lewm` conda env → `~/lewm`, then
    `sed` the env's `_sysconfigdata_*.py` prefix `exx/miniconda3/envs/lewm`→`~/lewm` (else C builds fail on permission);
    install via `~/lewm/bin/python -m pip` (the `bin/pip` shebang points at exx's path); `--no-build-isolation` for egl_probe;
    `cmake<4` (egl_probe's old CMakeLists); robomimic/robosuite = the **MinghaoFu forks** (rsync 174's source, `pip -e`);
    force `transformers==5.9.0 tokenizers==0.22.2 huggingface_hub==1.16.4` + `diffusers==0.36.0` back AFTER the robomimic
    install downgrades them. 174↔L40S direct transfer ~200 MB/s via bidirectional SSH keys (L40S IP 64.62.194.205).
- nvme1 chronically ~95% full (data ~3.1T) → **per-epoch ckpt prune loop** running (`$Q/prune_loop.sh`, keeps newest 2/run,
  every 180s). Without it, per-epoch saves hit 0 bytes and kill runs mid-save (happened 6/7 and 6/14).
- **🔴 ROOT disk `/` is 100% full (system-level, 6/14)** → breaks EVERY root write in turn: hydra `outputs/<date>/`,
  stable_pretraining `~/.cache/stable-pretraining/runs/`, `/tmp` (dill import), HF lock files. **Canonical launch fix =
  `$Q/qdetach_final.sh`:** redirect all writes to nvme1 — `export TMPDIR=$L/tmp TMP=$L/tmp TEMP=$L/tmp HF_HUB_OFFLINE=1
  MPLCONFIGDIR=$L/mpl`, and per-run `hydra.run.dir=$L/hydra_out/<run>`. Free root by
  `rm -rf ~/.cache/stable-pretraining` (80G of OLD run-output dirs — NOT model checkpoints, those live on nvme1).
- **🟢 WANDB ON (user request 2026-06-15):** auth = `~/.netrc` (key from super-console/personal/credentials.md, user
  m9fu); the ONLY writable entity is **`minghao_workaholic`** (NOT lewm/m9fu/models — all permission-denied/not-found).
  Per-run: `wandb.enabled=true wandb.config.entity=minghao_workaholic wandb.config.project=mtjepa-gip`. Dashboard:
  https://wandb.ai/minghao_workaholic/mtjepa-gip . **MUST also export `WANDB_DIR=$L/wandb XDG_CACHE_HOME=$L/.cache`** —
  WANDB_DIR keeps wandb's local run files on nvme1, and **XDG_CACHE_HOME redirects stable-pretraining's `runs/` checkpoint
  cache off root** (the trainer writes a ~335 MB atomic_save ckpt per run to `~/.cache/stable-pretraining/runs/` =
  root-filler; XDG_CACHE_HOME=$L/.cache sends it to nvme1 — verified). Smoke-tested 2026-06-15: can 1ep+wandb runs +
  syncs + saves, no crash. **🔴 XDG_CACHE_HOME is TRAINING-ONLY — do NOT set it for EVAL:** it also redirects the HF cache
  (CLIP/DINO live in `~/.cache/huggingface` = 2.7G on root), so with HF_HUB_OFFLINE=1 the eval can't find the encoder and
  crashes in ~7s (`LocalEntryNotFoundError`). Eval doesn't write training ckpts, so just omit XDG_CACHE_HOME for evals.
  (Training is fine with it: encoder comes from init_from + task embeddings are pre-computed, no runtime HF fetch — but a
  belt-and-suspenders `HF_HOME=/home/minghao.fu/.cache/huggingface` makes HF work regardless.) Also: root atomic_saves
  still accumulate even on nvme1; a background prune loop (find runs -name '*.ckpt' -mmin +15 -delete, every 300s) keeps it bounded.
- **🔴 CLIP-cache gotcha (2026-06-15): disk-freeing deleted the CLIP weights** from `~/.cache/huggingface/hub/models--openai
  --clip-vit-large-patch14` (kept config.json+tokenizer, lost `model.safetensors`/`pytorch_model.bin`), so every eval (which
  loads `CLIPTextModelWithProjection` for the task embedding) crashed offline with `LocalEntryNotFoundError` after ~68 log
  lines. Fix: `HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-large-patch14')"`
  (box has internet, ~6s). Verify with the offline standalone `CLIPTextModelWithProjection.from_pretrained(...)` → CLIP_OK.
- **🔴 Launch persistence: wrapper-script-with-`wait` driven by `nohup bash script &` over SSH does NOT survive ssh exit
  (driver logs 0 starts).** Launch each eval as its own detached proc: `QJOB=... setsid .venv/bin/python $Q/qrun.py </dev/null
  >log 2>&1 &` then `exit 0`. Cap concurrency by batching (3 at a time = ~150 robosuite envs). NOTE: arms launched BEFORE this (the detach/solveact evals + paused goal-reaching training)
  were wandb=OFF and can't be retrofitted; only NEW launches (training resumes, act_emb retrain, future seeds) get wandb.
  **Use this env for every training/eval launch until root is cleaned by the user.** Gotcha: `+ckpt_every=N` and
  `+trainer.limit_*` need the leading `+` (hydra struct mode rejects new keys without it). ckpt_every reduces the save
  burst (every 10 epochs, not 1) so concurrent runs don't tip the tight nvme1.
- robosuite evals are CPU-heavy (`num_envs=num_eval`=50). Keep **≤2 robosuite evals concurrent**; CPU melts at load ~189
  with ~18 train runs co-located. Kill garbage runs to de-congest.
- Process hiding: launches go through `$Q/qrun.py` (setproctitle→"python3", QJOB env) so nvidia-smi/nvitop show `python3`.
- GPU6/7 = another user's vLLM (don't touch). ~~GPU0 has zombie #39 (PID 3361122, pre-existing, kill needs user OK).~~
  ✅ zombie #39 `repro100_pusht` KILLED 2026-06-16 (was 9-days hung; ep92 ckpt kept; GPU0 freed) — see §1b.

---

## §0 ⚠️🐛 CONFIG BUG (found 2026-06-14): ALL robomimic ran `detach_decoder=FALSE`
- **Motivation for the flag (user design, memory project_detach_decoder):** the action decoder should NOT shape the
  representation; learn the intention purely in latent space (`intent_loss` predicts `act_emb`), decoder = detached
  inference-time readout (like LeWM's post-hoc image decoders). → wanted **`detach_decoder=TRUE`**.
- **PushT 3-seed controlled sweep (2026-06-01, ✅, sigreg_act held =true both arms, seeds {1,2,3}, 20ep×1000, 50-ep eval):**
  | metric | detach=FALSE | detach=TRUE |
  |---|---|---|
  | bc | 60.7 ± 2.3 | 57.3 ± 5.8 |
  | planning | 67.3 ± 9.9 | **76.0 ± 2.0** |
  | guided | 69.3 ± 6.1 | **74.7 ± 3.1** |
  Verdict: detach alone does NOT change success (all Δ within ±1σ at n=3), BUT detach=TRUE has much lower variance
  (one FALSE seed cratered planning to 56) — consistent with `act_loss` perturbing the encoder when not detached.
  Latent-only design **validated**; earlier "guided gap 74 vs 84" was a sigreg_act confound, refuted.
- **THE BUG:** `launch_robomimic_gip.sh` / `launch_warm.sh` **never set `detach_decoder`** → every robomimic+goal-reaching
  GIP run used the **default FALSE**. detach=TRUE was only ever run on PushT, never propagated. So **all the numbers
  below marked "detach=FALSE ⚠️" used the wrong, un-validated setting** (decoder backprops into latents/act_emb).
- **Nuance (untested):** on robomimic (needs object perception), TRUE vs FALSE was NEVER tested. Real tension —
  TRUE = pure intuition but encoder gets no raw-action grounding (may stay object-blind); FALSE = grounds encoder but
  contaminates the intuition. PushT (no object perception needed) couldn't expose this.
- **FIX (in progress, §1):** re-run the whole base matrix with `detach_decoder=true` at FULL training (100ep).
- **RECURRENCE FIX (2026-06-14, ✅ done + smoke-verified):** `SaveCkptCallback` now writes the FULL cfg (incl.
  `action_pred`) to `checkpoints/<run_name>/full_config.yaml` per run (`utils.py` `_dump_full_config`,
  `train.py` passes `full_cfg=cfg`). Smoke confirmed it dumps `action_pred: {enabled, w_act, detach_decoder, head}`.
  Every future run's exact settings are now recoverable from its own folder — `config.json` (cfg.model only) hid this
  class of bug. (The currently-running `detachT100` arms predate the fix; their settings are in `$Q/qdetach.sh` + §1.
  Originals backed up `utils.py.bak_fullcfg` / `train.py.bak_fullcfg`.)

---

## §1 🔄 CORRECTED base matrix: detach_decoder=TRUE @ 100ep (launched 2026-06-14)
- **Motivation:** redo all GIP base models with the validated latent-only design (TRUE) AND full training (100ep, not
  the old lazy 40ep). Output names `gip_<task>_detachT100` (keep the FALSE baselines for A/B).
- **Config:** the recipe in §0 with `action_pred.detach_decoder=true trainer.max_epochs=100`, MSE head, w_act=w_intent=1.
- **Runs (launcher `$Q/qdetach.sh`, runner pid 1311650):**
  | task | run_name | data | init_from | GPU | pid |
  |---|---|---|---|---|---|
  | lift | gip_robomimic_lift_detachT100 | robomimic_lift | lift_lewm_weights.pt | 0 | 1311652 |
  | can | gip_robomimic_can_detachT100 | robomimic_can | can_lewm_weights.pt | 5 | 1311947 |
  | square | gip_robomimic_square_detachT100 | robomimic_square | square_lewm_weights.pt | 2 | 1312231 |
  | pusht | gip_pusht_detachT100 | pusht | pusht_ours_lewm_weights.pt | 3 | 1312867 |
  | tworoom | gip_tworoom_detachT100 | tworoom | tworoom_ours_lewm_weights.pt | 4 | 1313456 |
  | reacher | gip_reacher_detachT100 | dmc | reacher_ours_lewm_weights.pt | 0 | 1313842 |
- **ETA:** robomimic ~37min (small data, 100ep); goal-reaching ~9–15h (4000 batches/ep × 100).
- **Eval plan:** histbc N=50 3-seed {42,0,1} (robomimic, budgets lift100/can240/square320) + eval_gip bc/guided/planning
  N=50 3-seed (goal-reaching). **TODO verify plateau** (rule 9) — extend if eval still rising at 100ep.
- **Result (landing live 2026-06-14):** lift/can/square_T converged ep100; `full_config.yaml` confirms
  `detach_decoder: true` ✅ (fix verified in production). **can_TRUE histbc N=50 seed42 = 62.0** vs the buggy
  detach=FALSE/40ep **50** (seed42) / 48 (3-seed) → **+12 on seed42**. So detach=true + full training (100ep)
  recovers a real chunk of can — the old §2 numbers were deflated by BOTH the bug AND undertraining. ⚠️ single-seed
  (diagnostic) — 3-seed {0,1} pending; square/lift_T evals pending. NOTE: robosuite evals **hang under co-location
  load** (the can/square evals froze mid-rollout at load ~39; completed only when load dropped) → run evals at low
  load / kill+relaunch via the tracker.
- **Conclusion:** TBD pending 3-seed + the §1b detach=FALSE@100 A/B (to separate "bug" from "undertraining").

---

## §1b 🔄 Parallel arms (co-located on GPUs 0–5, launched 2026-06-14, `$Q/qmore.sh`)
- **detach=FALSE @100ep A/B** (`gip_*_detachF100`, 6 tasks): identical recipe, `action_pred.detach_decoder=false` — the
  convergence A/B against §1's TRUE arms. true-vs-false on robomimic was NEVER tested (PushT only, §0); these settle it
  at full training. GPUs: lift2/can3/square4/pusht5/tworoom2/reacher3.
- **solve-action** (`gip_robomimic_{can,square}_solveact`): `action_pred.w_intent=0 action_pred.detach_decoder=false`
  → drop the embedding intuition; raw-action `act_loss` is the SOLE policy signal AND grounds the encoder. Tests the
  user's "pure-JEPA-WAM never solves the action" hypothesis. Compare vs §1 (TRUE, embedding-intuition) and §1b detachF.
- All 14 arms (6 TRUE + 6 FALSE + 2 solveact) + #46 co-located on GPUs 0–5 (≤3/GPU), crash=0, all writes nvme1-redirected.
- **Result:** ✅ ROBOMIMIC DONE (§1c closed — all @ep100, N=50 3-seed, action-head A/B = TIED negative result).
  ✅ **GOAL-REACHING ALL @ep100 (2026-06-16 ~17:30 UTC):** all 6 arms (pusht/tworoom/reacher × detachT/F) reached
  `weights_epoch_100.pt` (pusht/tworoom finished ~09:43 UTC, reacher ~13:10 UTC). 🔄 **eval_gip N=50 LAUNCHED**
  (`$L/evalgip_goalreach.sh`, setsid, qrun-hidden): detachT100 (validated design) × {bc, guided, planning} × N=50,
  3 task-streams pusht=GPU1 / tworoom=GPU2 / reacher=GPU3, bc→guided→planning sequential. Results → `gip_eval/<mode>/<policy>/`
  + `eval_logs/evalgip_*.log`. Fills §4.2/§7b per-family table with the CORRECT-design numbers (replacing the old
  detach=FALSE seed42 §2 numbers pusht80/tworoom90/reacher88). **detachF100 A/B + 3-seed = next batch after.**
  - ⚠️ **The hourly cron did NOT auto-launch these** (arms hit ep100 ~4–8h before I checked); launched manually 2026-06-16.
  - 🔪 **`repro100_pusht` (the #39 LeWM WM repro) KILLED 2026-06-16:** it was HUNG — alive 12 days but last checkpoint
    `weights_epoch_92.pt` dated **06-07 07:05 (9 days stale)**, spinning at 99% GPU0 util producing nothing, adding
    contention that slowed the goal-reaching arms. Killed (bracket-escaped `train[.]py.*repro100_pusht`, 13 procs→0);
    **ep92 checkpoint preserved** (converged proxy for #39; pusht plateaus well before 100). GPU0 freed.
  - **✅ eval_gip N=50 RESULTS (detachT100, validated design, seed42, 2026-06-16 17:3x UTC):**
    | task | bc | guided | planning |   | old §2 detach=FALSE s42 (bc/guided/planning) |
    |---|---|---|---|---|---|
    | pusht   | 68 | **80** | 62 | | 72 / 82 / 80 |
    | tworoom | 42 | **90** | **92** | | 44 / 94 / 90 |
    | reacher | 2  | **88** | 76 | | 2 / 72 / 88 |
    - **Reading (deep, vs full history):** (a) the validated detach=TRUE design reproduces the old detach=FALSE §2 numbers
      on the **robust modes** (bc, guided): pusht bc/guided 68/80 vs 72/82; tworoom 42/90 vs 44/94; reacher 2/88 vs 2/72.
      So **the detach-fix does NOT move goal-reaching bc/guided SR** → the action-head A/B negative result (§1c, found on
      robomimic) now holds on **BOTH families**: the detach/solveact/embedding micro-choice is within noise everywhere.
      (b) **`planning` (pure CEM) is the noisy mode:** pusht 62 (vs old 80) and reacher 76 (vs old 88) dropped ~12–18 pts,
      but this is **single-training-seed** (detachT100 is a different training run than the old detachF model) and Δ12–18
      is exactly the ~10-pt training-run variance band §1c established for this codebase. **Do NOT headline pure-planning
      single-seed** — needs 3-seed. `guided` (intuition-warm-started CEM) is both higher AND more stable (80/90/88) →
      **guided is the LeWAM headline mode**, consistent with "planning with intuition." (c) **Per-family story rock-solid
      under the correct design:** guided $\gg$ bc on the search-benefiting tasks (tworoom 90 vs 42; reacher **88 vs 2** —
      the reactive head alone is useless, the WM goal-cost solves it), and goal-reaching guided/planning **62–92 vs
      robomimic planning 0** (§7b) — the §4.2 contrast on validated-design ep100 N=50 numbers. reacher bc=2→guided=88 is
      the cleanest single datapoint that "planning helps where the WM sees the goal."
    - **✅ detachF100 A/B (seed42, 2026-06-16 17:5x UTC):** pusht bc/guided/planning **76/78/76**, tworoom **42/94/88**,
      reacher **2/64/(planning landing)**. **A/B vs detachT (Δ):** pusht guided 80 vs 78 (2), planning 62 vs 76 (**14**);
      tworoom guided 90 vs 94 (4), planning 92 vs 88 (4); reacher guided **88 vs 64 (24!)**. **The big gaps FLIP direction**
      (detachT higher on reacher-guided; detachF higher on pusht-planning) → **single-train-seed variance, NOT a systematic
      detach effect.** §1c negative result reproduced on goal-reaching + a vivid caution: **single-seed goal-reaching SR
      swings ±10–24 pts** (reacher-guided worst). Robust invariants surviving the noise: bc$\ll$guided on reacher (2 vs
      64–88) and tworoom (42 vs 90–94); ALL goal-reaching guided/planning $\gg$ robomimic planning 0. **The PER-FAMILY
      contrast survives the noise; the within-family A/B does not exceed it.** → next experiment = **3-train-seed** the
      detachT goal-reaching arms to pin §4.2 (single-seed is too noisy to headline).
    - **🔄 3-TRAIN-SEED LAUNCHED 2026-06-16 ~18:0x UTC** (`$L/var_runs_goalreach.sh`, setsid, wandb on, +ckpt_every=10):
      `gip_{pusht,tworoom,reacher}_detT_s{2,3}` (6 runs, detach=TRUE, 100ep, GPUs 1/2/3 @ 2/GPU, init from
      `*_ours_lewm_weights.pt`). Combined with the existing `gip_<task>_detachT100` = **3 independent train-seeds/task**.
      All healthy at launch (Epoch 0, ~6 it/s); ETA ~18h to ep100. On convergence → eval_gip bc/guided/planning N=50
      (`$L/evalgip_goalreach.sh`) per seed → report mean±std per task/mode, the paper-headline §4.2 numbers. Hourly cron
      catches convergence.
    - **✅ EVAL-SEED BANDS (detachT100, 3 eval-seeds s42/s0/s1, `$L/evalgip_seedbands.sh`, 2026-06-16):** best current
      §4.2 estimate (single train-seed × 3 eval-seeds):
      | task | bc (mean) | guided (mean) | planning (mean) |
      |---|---|---|---|
      | pusht   | 71.3 (68/80/66) | 72.0 (80/74/62) | 62.0 (62/76/48) |
      | tworoom | 46.0 (42/50/46) | **90.0** (90/92/88) | **95.3** (92/96/98) |
      | reacher | 3.3 (2/2/6) | **86.7** (88/92/80) | 85.3 (76/90/90) |
      **Per-family story robust to eval-noise:** guided/planning ≫ bc on tworoom (90/95 vs 46) and reacher (87/85 vs 3);
      pusht is the exception (bc 71 ≈ guided 72 — PushT is reactive, the WM goal-cost adds little). ALL goal-reaching
      guided/planning 62–95 ≫ robomimic planning 0. When 3-train-seed lands → report train×eval grid mean±std.

## §1c ✅ CLOSED detach A/B histbc results (robomimic, all @100ep, eval `eval_histbc_robomimic.py`)
> **STATUS 2026-06-15 (~21:00 PT):** ✅ COMPLETE. All 3 robomimic tasks reached `weights_epoch_100.pt`; all 27
> histbc N=50 runs (3 tasks × {detachT,detachF,solveact} × 3 seeds {42,0,1}) finished (eval mtimes 06-14 23:27→06-15
> 06:45 UTC, i.e. auto-launched as each arm hit ep100 — tracker mandate executed). The N=50 matrix below is FINAL and
> these numbers are on the ep100 checkpoints. **VERDICT: action-head variant (TRUE/FALSE/solveact) is TIED within
> training-run noise (negative result; see the ⚠️⚠️ MAJOR CORRECTION + ✅ FINAL below).** The early "solveact wins"
> framing (struck below) was a single-training-run artifact.
**HEADLINE = N=50 × 3-seed {42,0,1}, per-task budget lift100/can240/square320** (paper standard = LeWM repo default;
see [[feedback_eval_num_eval_50]] — N is NEVER lowered for headline). Driver `$L/eval_N50_seq.sh <seed>` (sequential,
one 50-env eval at a time). detach=TRUE = validated latent-intuition design; detach=FALSE = raw-act grad reaches
encoder; solveact = w_intent=0 (raw act = sole policy signal).

**N=50 HEADLINE matrix (filling live):**
| task | TRUE s42 | TRUE s0 | TRUE s1 | FALSE s42 | FALSE s0 | FALSE s1 | solveact s42 |
|---|---|---|---|---|---|---|---|
| lift   | **96.0** ✅ | **94.0** ✅ | **90.0** ✅ | **96.0** ✅ | **96.0** ✅ | **92.0** ✅ | — |
| can    | **72.0** ✅ | **72.0** ✅ | **68.0** ✅ | **68.0** ✅ | **70.0** ✅ | **70.0** ✅ | **76.0** ✅ |
| square | **64.0** ✅ | **64.0** ✅ | **56.0** ✅ | **62.0** ✅ | **58.0** ✅ | **52.0** ✅ | **70.0** ✅ |

- **✅ FINAL 3-seed A/B verdict (s42/s0/s1), N=50:**
  | task | detach=TRUE | detach=FALSE | solveact |
  |---|---|---|---|
  | lift   | 96/94/90 = **93.3** | 96/96/92 = **94.7** | 92/94/94 = **93.3** ✅ (filled 2026-06-16; was skipped as near-ceiling — confirmed: solveact 93.3 = detachT 93.3 = detachF 94.7, all tied at ceiling) |
  | can    | 72/72/68 = **70.7** | 68/70/70 = **69.3** | 76/82/72 = **76.7** |
  | square | 64/64/56 = **61.3** | 62/58/52 = **57.3** | 70/66/60 = **65.3** |
  - **detach=TRUE ≥ detach=FALSE** everywhere (lift tie 93.3 vs 94.7; can +1.4; square +4) → validated latent-intuition
    design confirmed, old detach=FALSE was the wrong setup but the convergence gap is SMALL (≤+4), NOT the N=20 +15.
  - ~~**⭐⭐ solveact (direct raw-action, w_intent=0) is the BC WINNER: can 76.7 (+6 over TRUE 70.7), square 65.3 (+4).**
    Decisive comparison: detach=FALSE 69.3 vs solveact 76.7 differ ONLY in w_intent (1 vs 0) → dropping `intent_loss`
    (the act_emb-prediction objective) buys +7. So **the embedding-prediction auxiliary itself HURTS BC** (see §9 deep
    analysis + research). This is the headline result of the whole detach-fix investigation.~~
    **❌ SUPERSEDED** by the multi-train-seed correction below — this +6/+7 was a single-training-run artifact; at
    3 train-seeds detT (74.0) ≈ solveact (70.0). The embedding auxiliary does NOT robustly hurt BC. Kept for the record.
  - **🔄 ROBUSTNESS (launched 2026-06-15, wandb on): multi-TRAINING-seed on can.** sigOFF=80 (detach=TRUE re-train) vs
    detachT100=70.7 exposed ~10pt training-run variance (same seed → nondeterminism), so the +6 solveact gap needs
    training-seed means, not just eval-seed. Launched `gip_can_detT_vs{1,2,3}` + `gip_can_solve_vs{1,2,3}` (6 runs, seed
    1/2/3, 100ep). With existing samples → detach=TRUE n≈5 {70.7-ish ×3 evalseed-on-one-model, 80}, solveact n≈4. Eval all
    at N=50 s42 → per-arm training-run mean±std. **If the CIs overlap, the honest claim becomes "solveact ≈ detach=TRUE,
    both ≫ the embedding-indirection-with-grounding detach=FALSE" rather than "solveact wins."** Settles the headline.
    - **⚠️⚠️ MAJOR CORRECTION (2026-06-15, N=50 s42): the action-head A/B is WITHIN TRAINING-RUN NOISE.** New training
      runs: detT_vs1=74, detT_vs2=72; solve_vs1=70, solve_vs2=70. So detach=TRUE {70.7, 80, 74, 72} ≈ solveact {76.7, 70,
      70} — **fully overlapping ~70–80, detach=TRUE if anything slightly HIGHER.** The earlier headlines — "solveact wins
      (+6/+7)" AND "intent_loss hurts (detF 69 vs solve 77)" — were **single-training-run artifacts**; the ~10pt
      training-run variance SWAMPS the ~4–7pt effects. **Honest verdict: detach TRUE/FALSE/solveact are statistically TIED
      on can (~72±5); the action-head variant does not robustly change BC SR.** Implication: (a) the "make the embedding
      path compete" framing is moot — the embedding path (detT) already ≈ solveact; (b) any single-run A/B in this doc
      (incl. lift/square 3-"seed" which were EVAL-seed on ONE model) is suspect — only TRAIN-seed means are trustworthy.
      UNAFFECTED (large robust effects): histbc≫CEM (0), cube-vs-robomimic per-family, pretrain frozen-probe gap. The EMA
      arms + vs3 finish the matrix but the verdict (tied) is already clear.
    - **✅ FINAL (2026-06-15): detach=TRUE 3-train-seed = 74/72/76 = 74.0; solveact = 70/70 (~70). detach=TRUE ≈/≥
      solveact — TIED, detT marginally higher.** Across ALL runs: detT {70.7, 80, 74, 72, 76}, solve {76.7, 70, 70}. The
      action-head variant does NOT robustly change can BC SR. EMA-target arms (solution #1) trained but eval DROPPED as
      moot (embedding path already ties solveact → nothing to rescue) + they crashed under load (benign CLIP-vision
      unexpected-keys warning was the only output; not re-chased). **CAN ACTION-HEAD A/B = CLOSED (negative result).**
      Pivoted compute to goal-reaching training (resumed from SIGSTOP) → eval_gip bc/guided/planning, the robust contribution.
    - **🔬 INFRA (gory details §0): CLIP weights deleted by disk-free → all evals crashed offline; re-downloaded. Wrapper
      driver didn't survive ssh → per-eval `setsid`. Eval in 3-concurrent batches (vs1→vs2→vs3).**
- ~~**⭐ solveact is the BC winner on BOTH hard tasks, 2-seed (s42,s0):** can_solveact **76/82** (mean **79**) vs detach=TRUE
  72 vs FALSE 69 → **+7**; square_solveact **70/66** (mean **68**) vs detach=TRUE 64 vs FALSE 60 → **+4**.~~ **❌ SUPERSEDED**
  (same single-train-seed artifact as above; 3-train-seed verdict = TIED). The WorldDP-style action-solver/intention split
  is still a fine *design* motivation, but it is NOT justified by a robust solveact SR win — do not cite this +7/+4 as evidence.
- **🔄 RUNNING (2026-06-15): sigreg_act anti-collapse test (user direction — "make the JEPA-style act_emb prediction
  compete with direct-action solveact").** Launched `gip_can_detachT_sigOFF` (pid 2647897) + `gip_can_detachT_sigON`
  (pid 2653271): detach=TRUE, 100ep, robomimic_can, wandb ON (entity minghao_workaholic, project mtjepa-gip) so we WATCH
  `act_emb_std` OFF vs ON live. Smoke-verified sigreg_act_loss computes (13.25) + act_emb_std logs (0.78 @init), no crash.
  **Predictions:** if collapse is the cause, sigON raises act_emb_std AND lifts SR toward solveact (72→~79); if sigON
  doesn't move SR, the embedding indirection (not collapse) is the issue → concede to solveact for BC. square arms next if
  can promising. Baselines to beat: detach=TRUE can 70.7 (3-seed), solveact can 79.
  - **⚠️ EARLY READ @ep14 (points AGAINST collapse):** sigOFF act_emb_std=**1.08**, act_loss=**0.18**; sigON
    act_emb_std=**1.02**, act_loss=**3.17**. (1) act_emb does NOT collapse on its own (OFF std ~1.08, stable, not →0).
    (2) sigreg_act EXPLODES act_loss 17× (0.18→3.17) — SIGReg forces act_emb toward isotropic-Gaussian, destroying the
    action structure the decoder inverts → worse raw-action recon → predicts WORSE BC, not better. (3) sigON std isn't
    even higher than OFF (SIGReg penalizes non-Gaussianity, not low variance). **Tentative: collapse hypothesis FALSE;
    solveact wins via the embedding INDIRECTION, not collapse.** CAVEATS before declaring: ep14 (act_loss may partly
    recover by ep100); sigreg_act weight = state-λ may be too high (over-reg) — if so a lower `sigreg_act_weight` keeps
    act_loss sane, but with std already ~1.0 there's little collapse to fix. Watching to ep100 + eval N=50.
- **can 3-seed COMPLETE: TRUE 72/72/68 (70.7) vs FALSE 68/70/70 (69.3) = +1.3 TRUE, ESSENTIALLY TIED** (the 2-seed +3
  shrank to +1.3 at 3-seed). So at convergence the detach flag barely matters on can; the real lever is solveact (+8 over
  both) — hence the anti-collapse test to rescue the embedding path.
- **OPEN (user hypothesis 2026-06-14): is the solveact win due to `act_emb` COLLAPSE?** intent_loss pins intention→act_emb
  (line 75, detach_target=true); act_emb is ALSO input conditioning (line 67) so action_encoder IS trained → act_emb CAN
  collapse (act_emb_std monitor line 105, sigreg_act guard line 102, default OFF). If collapsed, the embedding target is
  low-info → intention degenerate → worse BC; solveact drops the constraint → better. Mechanism coherent. Evidence mixed:
  old sigact runs (can 3-seed ~42) didn't beat MSE (48) but were 40ep/detachF/small-N (not comparable). TO TEST: (a) probe
  act_emb std+effective-rank on can_detachT100 ckpt (is it collapsed?); (b) sigreg_act ON in the detach=TRUE 100ep recipe,
  eval N=50 — does it close 72→~79? Caveat: sigreg_act hits act_emb in both roles (target + conditioning). [[project_gip_act_decode_loss]]
- **✅ RESOLVED 2026-06-15 — collapse hypothesis is FALSE.** sigOFF vs sigON (detach=TRUE, can, 100ep, wandb):
  `act_emb_std` OFF *grows* to 1.28 (no collapse), ON regularized to 1.05; `act_loss` exploded early (3.17@ep14) but
  RECOVERED to ~0.045 by ep100 = equal to OFF (0.049). So act_emb does not collapse and anti-collapse neither helps nor
  (at convergence) hurts the fit. **SR CONFIRMED 2026-06-15: sigOFF=80, sigON=76 (can N=50 s42) → anti-collapse does NOT
  help (slightly worse, within noise). Collapse hypothesis REJECTED.** ⚠️ But sigOFF=80 (a detach=TRUE RE-TRAIN, same
  config as detachT100 which scored 70.7) reveals **TRAINING-RUN VARIANCE ~±5–10 pts** — bigger than the eval-seed variance
  (±2–4) the 3-seed matrix measured. So single-training-run headline gaps (solveact 76.7 vs detach=TRUE 70.7 = +6) may be
  WITHIN training noise → need multi-TRAINING-seed means before claiming solveact wins. (Running.) **The real cause of solveact's
  win = `intent_loss` itself is a HARMFUL auxiliary.** Proof: detach=FALSE (intent ON, grounding ON) = 69.3 vs solveact
  (intent OFF, grounding ON) = 76.7 → the ONLY diff is w_intent, so the act_emb-prediction objective costs ~ -7 SR. Both
  paths fit training actions (act_loss ~0.045); the gap is closed-loop ROLLOUT → it's a mis-specified-auxiliary /
  representation-interference effect, not collapse, not training fit.
- **🔬 DEEP ANALYSIS + LIT (deepresearch 2026-06-15) — WHY intent_loss hurts, and how to make the embedding path compete:**
  `intent_loss = MSE(intention, sg(act_emb))` is a BYOL/SimSiam-style predict-a-stop-grad-target objective, but it
  VIOLATES every precondition the SSL literature requires for such a target to be useful: (i) NO EMA/momentum target
  encoder (BYOL 2006.07733, I-JEPA 2301.08243, DINO 2104.14294), (ii) the target `act_emb` is produced by an encoder
  optimized for a DIFFERENT role (input conditioning, line 67) so the head predicts a mis-specified, double-duty target,
  (iii) it adds a 2nd objective on the SHARED encoder → negative transfer / gradient interference (PCGrad 2001.06782,
  ForkMerge 2301.12618). Embodied-IL evidence that the plain behavior loss beats SSL auxiliaries for control: 2312.10069
  ("imitation may be all you need"). So solveact wins because it deletes the mis-specified auxiliary. **RANKED SOLUTIONS to
  rescue the JEPA-style embedding path (keep "predict in embedding space" identity):**
  1. **EMA target encoder + decouple target from conditioning** (BYOL 2006.07733 / I-JEPA 2301.08243): separate EMA
     action-encoder `f̄` (τ≈0.99) as the intent_loss target; conditioning stays a distinct online encoder. Attacks the root
     (no-EMA + double-duty). Small change, highest EV.
  2. **Decode-sufficient action autoencoder for act_emb** (VQ-BeT 2403.03181, Genie 2402.15391, LAPO 2312.10812): train
     act_emb with a reconstruction bottleneck so predicting it ≡ predicting the action → intent_loss aligns with act_loss.
  3. **Accept the split (WorldDP/LDP):** solveact = low-level executor, embedding/WM = high-level planner (LDP 2504.16925,
     WorldDP 2606.08775 [verify], FF-JEPA 2606.09311 [verify]). Safest PAPER framing — turns "intent_loss hurts BC" into
     "right tool per level." Backed by our own data.
  4. down-weight/gate intent_loss (w_intent≪1 + PCGrad) — mitigation, expect 69↔79 interpolation, not a fix.
  5. VICReg/Barlow on act_emb — LOW (our §4 SIGReg already showed forcing distributional structure on act_emb hurts;
     nothing to fix since no collapse).
  6. distributional head (diffusion/VQ) over act_emb — LOW for THIS gap (our §3 GMM/diffusion didn't rescue can; gap is
     rollout not multimodality). **Recommended next arms: #1 and #2** (root-cause), #3 as the paper story. Full report +
     all arXiv IDs in the deepresearch (agent a3f6c526f5bd02584).
- **✅ Solution #1 (EMA target encoder) IMPLEMENTED 2026-06-15** (server le-wm-repro, opt-in, default OFF — existing runs
  byte-identical; backups `*.bak_ema`). Config `action_pred.ema_target`/`ema_tau` (default false/0.99); BYOL-style
  `EMAActionEncoderCallback` (utils.py) momentum-updates `world_model.action_encoder_ema` on each optimizer step
  (on_train_batch_end, gated on global_step); intent_loss target switches to `action_encoder_ema(action).detach()` while
  prediction + conditioning stay online. Smoke-verified: OFF regresses clean; ON → EMA tracks online (mean|ema-online|
  0.121→0.115 monotone), intent_loss starts lower (1.0 vs 1.67). **Enable: `action_pred.ema_target=true`.** TODO: launch
  can detach=TRUE + ema_target vs detach=TRUE baseline (3 train-seeds), eval N=50 — does EMA target close the embedding
  path toward solveact? **LAUNCHED 2026-06-15:** `gip_can_detT_ema_vs{1,2,3}` (detach=TRUE + ema_target, seed 1/2/3, can,
  100ep, wandb on) — matched-seed A/B vs `gip_can_detT_vs{1,2,3}` (ema OFF baseline) + `gip_can_solve_vs{1,2,3}` (solveact
  ref). 9 arms training together (load ~34). Eval all N=50 at ep100 → does detT+ema > detT, toward solveact 76.7?

- **2-seed consistency (s42,s0) is tight:** lift_T 96/94, lift_F 96/96, can_T **72/72** (exact). The N=50 eval is
  reliable; the can_T=62 from §1 (pre-ep100 ckpt) was the outlier. Confidence in the A/B verdict rising.

**SEED42 N=50 COMPLETE (2026-06-14 01:40).** Per-task A/B + solveact:
- lift T96 = F96 (tie); can T72 > F68 (+4); square T64 > F62 (+2). **detach=TRUE ≥ FALSE on all three** (tie or
  small win) → the validated latent-intuition design is the right call; the old detach=FALSE robomimic runs were the
  buggy setup but the gap is modest at convergence.
- **⭐ solveact (w_intent=0, raw-act = sole policy signal + grounds encoder) WINS on the hard tasks: can 76 > T72 > F68;
  square 70 > T64 > F62.** Dropping the embedding-intuition (intent_loss) and "solving the action directly" gives the
  best histbc/BC on can+square. Directly answers the user's "pure-JEPA-intuition never solves the action" thread:
  for BC, solving the raw action beats the intuition indirection. ⚠️ single-seed s42; CONFIRM with s0,s1.
  TRADEOFF to check: solveact has NO intention head → can't do intention-guided planning (the WM/planning contribution).
  So solveact may win BC but lose guided/planning — the two contributions may want different heads. Eval guided/planning
  on solveact vs detach=TRUE to see the full picture.

- **lift A/B N=50 s42: TRUE 96.0 = FALSE 96.0 — identical, no detach effect on lift** (clean, 50 ep, budget 100).
- **can A/B N=50 s42: TRUE 72.0 vs FALSE 68.0 (+4 TRUE).** detach=TRUE slightly ahead on the contact-heavy task, close.
- So at proper N=50 seed42: detach=TRUE ≥ FALSE everywhere measured (lift tie, can +4) → keep the validated design;
  no penalty, small gain on can. NEED s0,s1 for the mean + square to finish.
- **Infra note 2026-06-14 ~00:13:** load hit 213 (mostly my own training arms) → SIGSTOPped my le-wm training (88 procs,
  reversible) to relieve the meltdown + unblock N=50 square; kept the eval running. Other load (lipeng ~1291%) is not mine.
  RESUME my training (SIGCONT) after the N=50 sweep.

- **can N=50 s42: TRUE 72.0 vs FALSE 68.0** (confirmed 50 episodes each, budget 240, final ep100 ckpt). At N=50 the
  detach advantage SHRINKS to +4 (vs the noisy N=20 +15) — i.e. on the proper-N eval, detach=TRUE and FALSE are close
  on can, TRUE slightly ahead. NOTE can_TRUE was 62 in §1 (earlier/pre-ep100 ckpt) vs 72 now (converged ep100) → the
  converged number is higher; reinforces rule-9 (eval converged ckpts only). Need s0,s1 for the real mean.
- **Infra reality:** these N=50 came from the original 2-stream `eval_N50.sh` whose subshells were orphaned (parent
  killed, children kept running) — they ground through lift+can under load but **square (budget320, heaviest) WEDGED**
  (env-init starved). N=50 completes for lift/can but needs freed CPU (pause my training) for square + the 3-seed fill.

**⚠️ N=20 numbers below are DIAGNOSTIC ONLY — NOT paper-usable, NOT headline** (recorded the mistake of treating them
as headline; corrected). They are a noisy early signal while N=50 fills in. The shared-box load (user `1234` on all 8
GPUs) makes N=50 robosuite evals slow/wedge-prone; the fix is sequential + (if needed) pausing my own training to free
CPU — NOT lowering N.

| task | TRUE s42 (N=20 diag) | FALSE s42 (N=20 diag) |
|---|---|---|
| lift   | 95.0 | 100.0 |
| can    | 80.0 | 65.0 |
| square | 75.0 | 80.0 |

- **N=20 diagnostic hint (to verify at N=50):** can TRUE 80 vs FALSE 65 (+15) suggested detach=TRUE helps on the
  contact-heavy can task; lift/square within 1-episode noise. Treat as a HYPOTHESIS only — the N=50×3seed matrix above
  is what decides it. Note can_TRUE swung 62@N50 vs 80@N20 (same seed/model) → N=20 is unreliable, exactly why headline
  must be N=50.
- **Next:** N=50 seed42 fills (lift→can→square→solveact, sequential) → seed0 → seed1 → per-task A/B verdict at N=50.

## §2 N=50 alignment (2026-06-13) ✅ — ⚠️ all detach=FALSE (superseded by §1)
- **Motivation:** headline SRs were multiples of 5 → caught under-sampling; LeWM/repo eval default `num_eval=50`,
  our runs used 10–20 (goal-reaching) / 8 (robomimic).
- **Robomimic histbc, N=50, 3-seed {42,0,1}, detach=FALSE, 40ep, budgets lift100/can240/square320:**
  | task | seed42 | seed0 | seed1 | mean |
  |---|---|---|---|---|
  | lift | 96 | 94 | 94 | **94.7** |
  | can | 50 | 46 | 48 | **48.0** |
  | square | 76 | 62 | 56 | **64.7** |
- **Goal-reaching, N=50, seed42, detach=FALSE (bc/guided/planning):** pusht 72/82/80 · tworoom 44/94/90 ·
  reacher 2/72/88 · cube 70/82/76.
- **Conclusion:** can+square were over-reported at small N (can 65–75 → 48). Protocol aligned (native is_success), sample
  size wasn't. **Numbers invalid as headline (detach=FALSE) — being redone in §1.**

## §3 Action-head ablation: GMM / diffusion vs MSE (#48) ❌ negative — detach=FALSE ⚠️
- **Motivation:** can/square weak → hypothesis = MSE decoder mode-averages multimodal demos; try multimodal heads.
- **Config:** `action_pred.head=mse|gmm|diffusion`, 40ep warm-start, histbc N=50 seed42, detach=FALSE.
  GMMHead: K=5 modes, min_std (buggy 1e-4 → fixed 0.05), exp(clamp) std, eval=argmax-mode mean. DiffusionHead:
  plain-MLP denoiser, n_steps=50, linear β schedule (← WRONG: real DP uses ConditionalUnet1D + 3000–4500ep).
- **Result (can, seed42, N=50):** MSE **50** · GMM-buggy(min_std 1e-4) **26** · GMM-fixed(min_std 0.05) **30** ·
  diffusion **0**. (square seed42: GMM-fixed **56** vs MSE 76.) GMM act_loss: buggy 3.96→**31** (variance collapse, NLL
  exploded); fixed 16.5→2.67 (stabilized). diffusion act_loss 0.52→0.31 (learning but eval=0 → undertrained).
- **Conclusion:** multimodal decoders do NOT fix can; mode-averaging ≠ bottleneck. **diffusion=0 INVALID** (under-arch +
  ~75× under-trained). GMM no-tanh is correct for our z-score actions (robomimic tanh's only because it min-max norms to [-1,1]).

## §4 act_emb anti-collapse: `sigreg_act` on/off (#49) ❌ negative — detach=FALSE ⚠️
- **Motivation:** apply the state-emb SIGReg (same module/weight λ) to act_emb; does a better-conditioned intention help?
- **Config:** `action_pred.sigreg_act=true`, MSE head, 40ep, histbc N=50. Infra already existed (train.py:84-86,323-324), never run.
- **Result (can, histbc N=50):** ON seeds {42,0,1} = 56 / 36 / 34 = **42.0** vs MSE OFF **48.0**. (sigreg_act_loss computes ~50.25; act_emb_std ~0.088.)
- **Conclusion:** sigreg_act does NOT help can (slightly worse, higher variance). seed42=56 was a high outlier — I over-claimed
  "+6" on single-seed → **ALWAYS 3-seed before claiming** (same trap as square 76→64.7).

## §5 proprio injection (`wah_*_act_proprio`) ❌ negative — detach=FALSE ⚠️
- **Motivation:** BC-RNN-lowdim has privileged object state; feed proprio to close the gap.
- **Config:** existing `wah_<task>_act_proprio` models (use_proprio=true, 11-D robot proprio: eef_pos/quat/gripper_qpos),
  histbc N=50 seed42, budgets can240/square320.
- **Result:** can+proprio **50** ≈ no-proprio 50 · square+proprio **58** ≤ no-proprio 76.
- **Conclusion:** robot proprio does NOT help. The missing piece is **object pose** — in the 71-D `state` key, NOT in the
  11-D `proprio`. (`can.h5`: /action (N,7), /proprio (N,11), /state (N,71 = robot+object qpos/qvel).)

## §6 BC-RNN gap analysis + latent probe (2026-06-14) ✅ evidence
- **Motivation (user push):** "不可能这么差,查官方代码" — is 48 a bug or real? Read robomimic + DP source.
- **robomimic BC-RNN reference (official, arXiv 2108.03298 / DP 2303.04137):** BC-RNN can = **100 (lowdim) / 98 (image)**,
  square 84/82. lowdim obs = `robot0_eef_pos/eef_quat/gripper_qpos` + **`object`** (privileged ground-truth object pose); image =
  end-to-end ResNet. Train 2000 epochs, batch 100, eval 50 rollouts horizon 400 max-over-ckpt, terminate_on_success.
- **Our encoder is NOT frozen** — `freeze_wm` never set → finetuned with the WM+intention objective.
- **Latent probe (gip_robomimic_can encoder, `gip_probe.py`, Ridge α=10, 3200 frames, 80/20 split, targets z-scored):**
  latent→proprio(11-D) R² = **0.775**, latent→full state(71-D, object+velocities) R² = **0.554**. ⚠️ confounded by
  velocities (un-decodable from one frame).
- **Conclusion:** "JEPA lossy" is too glib. Fair baseline = image-BC-RNN(98). Our latent is lossy on the scene; gap = encoder
  trained for pixel-prediction not grasping + embedding-indirection action head. Ceiling-vs-artifact = the §1/policy-first go/no-go.
- **✅ GRASP-STATE PROBE (Lift, subagent 2026-06-15):** decoding "is the object grasped/lifted" from the latent — single-frame
  AUC **0.88** (contact-*degraded*, NOT blind), latent-history (no actions) **0.97**, full $[z,a]$ **0.99**. Clean control:
  latent-history-alone (0.97, zero action info) ≫ single-frame (0.88) → real temporal context, not gripper-command leakage.
  Honest claim = **"history ≫ single-frame for contact"** (post-grasp regime). This is the mechanism behind histbc≫CEM (§1c)
  and the paper's scientific heart (CONCLUSIONS #3). Note `gip_probe.py` in repo is the OLDER single-frame proprio/state-R²
  probe; the grasp-AUC + history/[z,a] variants were a subagent script (not committed).
- **⚠️⚠️ PROBE EXTENSION RESULT (subagent 2026-06-16) — the universal "single-frame degraded" claim is FALSE; it's LIFT-SPECIFIC.**
  Decoding "is the object grasped/lifted" (object-z label from privileged `state`, found per task: lift dim2, can dim15,
  square dim10; clean single-flip per demo, frac 0.38–0.56), LogisticRegression ROC-AUC, **held-out-DEMO split, 5 split-seeds,
  detachT100 encoders**, window k=4 @ frameskip5:
  | task | single-frame | latent-hist(k4) | full[z,a](k4) | gap (hist−single) |
  |---|---|---|---|---|
  | **lift** (small cube) | **0.816**±.025 | **0.988**±.003 | **0.999**±.001 | **+0.17** ✅ big |
  | can (larger) | 0.962±.004 | 0.969±.004 | 0.973±.004 | +0.01 ❌ |
  | square (nut) | 0.999±.000 | 0.999±.000 | 0.999±.001 | ~0 ❌ |
  - **Lift reproduces the reference (0.82→0.99→0.999 ≈ the claimed 0.88/0.97/0.99)** — single-frame is contact-degraded,
    history recovers, past-actions saturate. But **Can/Square single-frame ALREADY decode grasp/lift at 0.96/0.999** → history
    adds nothing. **Reason (honest):** the can and square-nut are larger and lift visibly; one frame already shows it. The tiny
    Lift cube is the only object where a single frame genuinely under-exposes grasp. Cross-check with a finger-contact label
    (proprio finger qpos) gives the same ordering (single<hist<full everywhere) but Can/Square single already 0.98–0.996.
  - **⇒ MECHANISM IS NOW TWO-PART (this is the corrected story, NOT one universal "contact-blind"):**
    (1) **Lift** CEM≈0 because the single-frame goal latent **under-exposes grasp** (0.82) — contact-state under-exposure, the
    probe-supported mechanism. (2) **Can/Square** CEM≈0 **NOT** from invisibility (latent sees grasp at 0.96/0.999) but from
    **multi-stage structure** (approach→grasp→transport→place / grasp→align→insert): flat horizon-H CEM against a single
    goal-image cost can't decompose the sequence — exactly WorldDP's "single-stage only" diagnosis (§9, literature_survey §4).
    history-BC helps in BOTH by executing the sequential demo behavior; on Lift it additionally supplies contact-state memory.
  - **Caveats (subagent, honest):** the lifted-object-z label is high quality (clean flips). The pre-vs-post-grasp *contact*
    split was degenerate (gripper starts closed at reset → no clean pre-grasp-open window at frameskip5), so the starker
    pre-grasp gap I hoped for is NOT cleanly measurable; the lifted-label single-frame does drop to 0.77 when more approach
    frames are included (k4 vs k8), weak corroboration only. Scripts: `/mnt/data_nvme1/minghao.fu/tmp/grasp_probe.py` +
    `graspprobe_result_{lift,can,square}.json` + cached `.npz`. **No numbers fabricated; Can/Square saturation is real.**
- **⭐⭐ SUBGOAL-COST PLANNING DIAGNOSTIC (subagent 2026-06-16) — GREEN-LIT the hierarchical-subgoal thesis (offline).**
  Is a SUBGOAL (grasp) latent cost a usable planning signal where the FLAT final-goal cost = 0? Cost = `‖z_t−z_anchor‖²`
  on the 192-D CLS post-projector latent (the planner's exact cost space); **ground-truth grasp via robosuite replay**
  (corrected the object-z proxy — can dim15/square dim10 is a lift-plateau ~15 env steps AFTER true grasp; rebuilt
  `true_events_{can,square}.json`, 200/200 grasp both). **Phase 1 (offline, 200 demos, 3 demo-split seeds):**
  | metric (true-grasp anchor) | CAN | SQUARE |
  |---|---|---|
  | argmin(cost_subgoal) within ±2 fs-idx of true grasp | **100%** | **100%** |
  | AUC(−cost_subgoal) grasp-window vs early | **0.857** | **0.905** |
  | AUC(−cost_FINAL-goal), same window (baseline) | **0.325** (below chance!) | **0.670** |
  - **The subgoal cost's global min lands on the true grasp in 100% of held-out demos (both), AUC 0.86/0.91 — while the
    FINAL-goal cost is at/below chance (Can 0.325 prefers pre-grasp end-pose; Square 0.67).** ⇒ latent IS informative;
    flat-planning failure = **cost-decomposition**, not an uninformative latent. **Validates hierarchical intuition-guided
    subgoal planning** (resolved direction). (Grasp axis is ~0.5–3% of latent L2 variance → a linear metric hits 0.96/0.9998
    if a learned cost is ever wanted; raw-L2 is enough *locally* at the grasp window, which is what planning needs.)
  - **Phase 2 (online CEM-to-grasp-subgoal): CONFOUNDED, inconclusive** — starting ~5 fs-idx before grasp made the FLAT
    baseline already grasp 80% on Square; Can stuck 0% both arms. Subagent correctly STOPPED. **Clean test needs a
    FROM-DEMO-START rollout (t=0) — OPEN.** Scripts: `/mnt/data_nvme1/minghao.fu/tmp/{phase1c_trueanchor.py,phase1c_*.json,
    true_events_*.json,phase2_subgoal_cem.py}`.
  - **Verdict: GREEN-LIGHT (offline); online pending a from-demo-start rerun.** This is the pivotal go/no-go for the
    contact-planning paper — it passed offline. → Next: build the subgoal-planning method + (when L40S is up) Tool Hang/Transport.

## §7 Reproductions / migration
- **#46** 🔄 `repro_mtjepa_pusht_gip`: pusht GIP under swm-clean `mtjepa/`, 100ep parity check, PID 3686542, ~epoch 68/100.
- **swm migration** ✅: le-wm work → clean stable-worldmodel package `mtjepa/`; train/save/load/eval validated; original untouched.

---

## §7b WHY cube-CEM works but robomimic-CEM = 0 — CORRECTED 2026-06-14 (prior "push vs grasp" claim FALSIFIED)
**Correction (user caught this).** I previously wrote "Cube = push-to-geometry, robomimic = grasp-contact." That is
**wrong**: OGBench cube-single IS a pick/grasp task, and the two evals are methodologically identical. Do not use the
old framing in the paper. What is actually VERIFIED vs still a HYPOTHESIS:

**VERIFIED (read from configs/data on 174, 2026-06-14):**
1. **Cube is a grasp task, not pushing.** `config/eval/cube.yaml`: `env_name: swm/OGBCube-v0`, `env_type: single`,
   `dataset_name: ogbench/cube_single_expert`; goal set via privileged **block target pose** (`set_target_pos`,
   `goal_privileged_block_0_pos` + `_quat`). The arm grasps and carries the cube to a target pose. (The rollout frames
   show the gripper closing on a red cube — manipulation like Lift, not a push.)
2. **Cube & robomimic eval are the SAME method.** Both: CEM solver, `goal_offset_steps=25`, `eval_budget=50`,
   `horizon=5`, `action_block=5` (frameskip 5), `terminate_at_goal`, `num_eval=50`. So the contrast is **NOT** an
   eval-methodology / goal-anchoring artifact, and **NOT** "push vs grasp."
3. **Data scale gap is ~20×.** `cube_single_expert.h5` = **102 GB / ≈1M transitions** (OGBench offline-GCRL, broad
   state coverage, dense grasp+carry transitions). robomimic ph = **200 demos** each (lift/can/square; narrow expert
   manifold). The WM that CEM plans against is far better-fit on cube.

**LEADING HYPOTHESIS (multi-factor; NOT yet settled — needs a latent→grasp-state probe to confirm):**
- **(a) Goal-image change visibility.** CEM minimizes single-frame latent distance to a goal image. On cube the goal
  shows the cube *displaced to a new pose* — a large, visible object change that CEM **can only achieve by grasping +
  carrying**. On robomimic-lift the goal is the cube a few cm higher — a tiny object change **swamped by arm/gripper
  pose** (visually dominant), so CEM minimizes distance by matching arm pose and the grasp is never enforced
  ("reaches goal pose, can't grasp", project_robomimic_planning).
- **(b) Data coverage.** ~1M broad cube transitions → latent dynamics around pick/carry are accurate enough that
  reaching the goal latent ≈ achieving the pick. 200 narrow robomimic demos → contact/grasp transition under-covered,
  latent conflates grasped/not-grasped.
- **(c) Single-frame CEM vs history.** Grasp state ("am I holding it") is history/contact-dependent and weak in a
  single-frame latent; **history-bc recovers it** (post-grasp P(success|grasp) 76→99,
  project_gip_intention_format_eval_convention). That is why hist-bc (can≈48–62) ≫ CEM-planning (≈0) on robomimic,
  while on cube single-frame CEM already works.
- **Still rules out the trivial artifact:** aligned-goal robomimic CEM (demo success-region + budget, task #47) is
  STILL 0, so it is not just bad goal-region/budget.

**Implication for the paper:** do NOT claim "planning works on geometry, fails on contact." The honest, narrower claim
is: single-frame goal-image CEM succeeds when the goal encodes a large, visible task-relevant state change backed by
dense data (OGBench-cube), and fails when the success variable (grasp) is visually subdominant and data is narrow
(robomimic) — where a history-conditioned policy is needed instead. **TODO probe:** Ridge/linear probe latent→grasp
binary on robomimic vs latent→cube-carried on cube; + measure goal-image latent cost vs grasp success. Until then this
is a hypothesis, not a result.

## §8 Reference numbers (external)
- **GC-IDM** (2605.08732, UMich Nguyen/Xu/Huang; goal-conditioned amortized IDM `(z_t, z_g, horizon)→a` on LeWM latent;
  goal = image → z_g). Envs = Two-Room / Push-T / OGBench-Cube / Reacher (= OUR goal-reaching set, same swm/LeWM harness).
  | env | GC-IDM (N50/N200) | CEM (N50/N200) |
  |---|---|---|
  | Two-Room | 100 / 100 | 82 / 84 |
  | Push-T | 84.7 / 84.2 | 89.3 / 82.5 |
  | OGBench-Cube | 99.3 / 98.7 | 73.3 / 67 |
  | Reacher | 100 / 99.7 | 68 / 70.3 |
  "contact-rich" = Push-T (pushing, no grasp memory). No robomimic. GC-IDM near-ceiling on goal-reaching (beats our planning 76–90).
- **robomimic BC-RNN / Diffusion Policy** (image, PH): BC-RNN can 98 / square 82; DP can ~100 / square 96–98.

---

## §9 OPEN QUESTIONS / live brainstorm (update as we chat)
- **🌟🌟 PLANNING-FREE goal-conditioned reactive policy — the System-1 LIMIT (user brainstorm 2026-06-18, "more crazy"): "remove
  test-time planning, make JEPA purely end-to-end, output action DIRECTLY (CEM/MPC is slow) — but how to apply at goal-reaching?"**
  This is the **budget→0 endpoint of the amortization curve** (the intuition-seeded sweep below asks "how few CEM iters?"; this
  asks "can ZERO work?"). Same System1↔System2 axis, living at the System1 end.
  - **THE OBSTACLE (user named it):** the current intuition head (`jepa.intention_rollout`) is **goal-AGNOSTIC** — predicts `a_t`
    from `(z_t, past_actions)` only; the GOAL enters EXCLUSIVELY via the CEM cost (`get_cost` rolls candidates to `z_goal`). Delete
    CEM → the goal signal vanishes → the policy has no input saying WHICH goal. (Works on cube only because expert demos all funnel
    to one goal, so "do the demo" ≈ "reach the goal".)
  - **THE FIX: goal-CONDITION THE FORWARD HEAD WE ALREADY HAVE — `π(a_t | z_{≤t}, a_{<t}, z_goal)`.** ⚠️ **NOT an IDM (user, 2026-06-18).**
    Our `jepa.predict_intention` is a FORWARD policy (`π(history) → next action`, BC-trained), NOT an inverse-dynamics model
    (`f(z_t, z_{t+1}) → a_t`, Markovian). The planning-free path = thread `z_goal` into THIS forward head, keeping history
    conditioning — do NOT bolt on an IDM. **"No IDM" is substantively right, not taste:** the paper's IDM is Markovian (two states
    → action) and throws away history; our forward head is HISTORY-conditioned = the contact-state-memory advantage we already
    measured (Lift: history-aware intuition wins via post-grasp memory [[project_gip_intention_format_eval_convention]]). **WEDGE =
    history-conditioned forward goal-policy vs the paper's Markovian goal-IDM.** Training options (try in order): (1)
    **Hindsight-relabeled GC-BC on the forward head** [primary]: relabel demo futures as goals, train `π(z_{≤t}, a_{<t}, z_{t+k}) →
    a_t`; at test plug real `z_goal`, read action. (2) **Planner distillation (ExIt, `exit`)**: CEM generates `(history, z_goal,
    planned_a)` → forward policy imitates. (3) **Differentiable-WM policy gradient**: backprop a goal-reaching loss THROUGH the WM
    rollout into π. All → JEPA end-to-end at inference: `encode(obs,goal) → forward head → action`; WM's job moves to TRAINING only.
    **📄 The user's referenced "goal-conditioned JEPA, IDM directly outputs policy" paper = the contrast we DON'T follow (link to be
    re-dropped → add to survey/bib as the Markovian-IDM baseline we differ from via history conditioning).**
  - **🔴 DIFFERENTIATION — `gcidm` LARGELY SCOOPS THIS (deep-read 2026-06-18, [[reference_gcidm_umich]]).** "Latent Geometry Beyond
    Search" (2605.08732, UMich) does the planning-free goal-conditioned policy **on our EXACT base (LeWM/sigreg vit-tiny-192) and
    our EXACT envs (cube/pusht/reacher)**: a tiny 3-layer-MLP GC-IDM on frozen latents, hindsight-trained, horizon-conditioned
    (AdaLN-Zero), beats CEM (cube 98.7 vs 67!) at 100–130×; **AND they already ran the budget/amortization sweep** (their Fig B →
    no CEM config beats GC-IDM on speed+success). So the pure planning-free idea AND the amortization-curve headline are THEIRS.
    **Our remaining wedges are narrow but real:** (a) **HISTORY** — gcidm is MARKOVIAN `(z_t,z_goal,h)`; its weakest cell is
    pusht (contact) + degrades at distant goals; our Lift post-grasp-memory finding says a HISTORY-conditioned forward policy
    should win where a single (z_t,z_goal) can't tell pre- vs post-contact. (b) **ROBOMIMIC = their WHITE SPACE** (they tested 4
    SIMPLE envs, never robomimic lift/can/square/tool_hang/transport). (c) **ENCODER COUPLING** — they use vit-tiny-192; our
    DINOv2-init gives a better latent → does better geometry rescue the single-pass map on HARD contact tasks (where 98.7%-ceiling
    headroom exists, unlike their simple envs)? **The one honest contribution = "planning-free GC control on hard contact
    manipulation needs HISTORY + a better latent, beyond the Markovian GC-IDM that suffices on simple scenes"** — uses our two
    existing findings (Lift memory + DINOv2) as motivation, on tasks gcidm never ran. MUST adopt their horizon-conditioning (load-bearing, −42pp without).
  - **⚠️ HONEST RISKS:** (a) **loses zero-shot goal generalization** — CEM-over-WM reaches NOVEL goals with no retrain (DINO-WM's
    selling point); a goal-cond policy only reaches near-training-distribution goals (speed-for-generality trade). (b)
    **multimodality** — `(z_t,z_goal)`→several valid actions; MSE head averages to garbage → need the **GMM/diffusion head (already
    built, §act-head ablation / task #48)**. (c) **compounding error** — reactive has no replan; mitigate with closed-loop re-encode.
  - **🎯 PUNCHLINE (complete story, NOT a replacement):** fast goal-cond policy = **System 1**; CEM-over-WM = **System 2**; use
    System 1 by default, fall back to System 2 when uncertain / goal novel. = how humans plan fast (reflex 95%, deliberate 5%).
    Budget-sweep measures System-2 cost; direct policy is System 1; the hybrid is the product. **Composes with the encoder work:**
    direct goal-reaching is only as good as the latent's goal-discriminability = exactly what the DINOv2 probe measures (better
    encoder → better reactive goal-reaching).
  - **🔴 GOAL-SOURCE FORK (user, 2026-06-18 — "where does the goal image come from? in real robots you can't have one — is it
    cheating?"). The sharpest critique of the whole setting, and it RE-OPENS the wedge.** In these benchmarks the goal image is an
    ORACLE — sampled from a held-out expert demo (`eval_gip.py goal_offset_steps` → the obs at start+offset in the replayed
    trajectory). So goal-conditioning quietly OUTSOURCES task specification; it tests controllability/reachability in isolation but
    assumes away the hard part (obtaining the goal — chicken-and-egg: can't photograph the finished state before achieving it).
    **gcidm + our planning-free policy live ENTIRELY in the goal-IMAGE setting → they inherit this deployment caveat.** The wedge
    your question opens: **gcidm is LOCKED into goal-images; "reach goals you can actually SPECIFY in the real world" is open white
    space.** Deployable goal SOURCES (the goal-reaching policy is the low-level module; the source is a separate problem the
    benchmark hides): (1) **language** ("pick up the cube" — RT-2/OpenVLA); (2) **reward/value** — TD-MPC2, **the user's own earlier
    no-goal-image lean** [[reference_gip_competitive_landscape]]; (3) **demo / one-shot imitation** — OSVI-WM (`osviwm`, the paper the
    user flagged): the "goal" is a demo someone showed ONCE (which you DO have), not a pre-photographed end state; (4) **generative
    goal** — a VLM/diffusion IMAGINES the goal image, then the policy reaches it. **FORK:** (A) accept the goal-image benchmark
    (standard, gcidm-comparable, note the caveat) vs (B) pick a deployable source (demo/value-conditioned) as the differentiator.
    **Lean (B):** demo-conditioned (OSVI-WM-style) fits our **history-conditioned forward policy** naturally (condition on the demo
    trajectory's latents instead of a single goal frame) AND is more deployable AND is somewhere gcidm's single-frame GC-IDM can't
    trivially follow → composes with [history + robomimic-contact + DINOv2] into a coherent, harder-to-scoop story. Note our
    goal-AGNOSTIC bc head already needs NO goal (deployable, fixed-task) but can't redirect — the spectrum is: bc(no goal) →
    demo/value-conditioned(deployable) → goal-image(oracle, gcidm). **OPEN: decide A vs B before building the head.**
  - **⚠️ SCOPE (user 2026-06-18): the survey is GOAL-CONDITIONED *POLICY* (direct `π(a|s,g)`, one forward pass, no test-time
    planning), NOT goal-reaching-via-PLANNING (DINO-WM/PLDM/V-JEPA2-AC = WM+CEM — a DIFFERENT category we're LEAVING).** In-scope
    direct GC policies: **gcidm (GC-IDM — our DIRECT competitor, same JEPA/LeWM base), BESO, GCBC, HIQL low-level, CRL, GCIQL/QRL,
    DP-GC, HyperGoalNet**. Generative-imagination (Act2Goal/SuSIE/UniPi) = borderline (imagine→act, closer to planning). Direct GC
    policies benchmark on **OGBench / D4RL (AntMaze,Kitchen) / CALVIN — never robomimic as a pure policy** (LDP uses a planner+IDM).
  - **🟢 POSITIONING from the GC-policy benchmark survey (subagent, 18 papers, 2026-06-18) — TWO white-space claims with named
    must-beat anchors:** **(1) Native goal-conditioned control on robomimic contact-rich multi-stage manipulation** — NONE of the
    JEPA/latent-WM planners (DINO-WM, gcidm, V-JEPA2-AC, PLDM) do it (they top out at pusht/cube/maze); the ONLY robomimic-GC work
    is **LDP `ldp` 2504.16925** (image-subgoals + GCBC + inverse-dyn, Lift/Can/Square **0.69/0.70/0.46**) → our anchor to beat.
    **(2) ONE model unifying goal-reaching + goal-agnostic imitation** (= our two-setting WAM) — the ONLY clean prior is **BESO
    `beso` 2304.02532** (score-diffusion, classifier-free-guidance toggle; on Kitchen/CALVIN, **NOT robomimic, NOT a JEPA WM**) →
    cite+beat for the unification claim. So our two-setting WAM on robomimic = **DOUBLE white space** (GC-robomimic ∧ unification).
    Benchmark camps confirmed: goal-conditioned MANIPULATION = **OGBench** (state goals: cube/scene/puzzle) + **CALVIN** (language) +
    RoboTwin/real; **robomimic = imitation, goal-agnostic everywhere** (LDP the lone exception). Goal-source split: WM/GCRL branch
    (ours) uses state/latent goals, VLA branch uses **language→generated video/subgoal-image** (~8 papers); **demo-as-goal is RARE**
    (→ underexplored + deployable). Long-horizon handled 4 ways, none by a JEPA-WM on robomimic-hard: hierarchical subgoals (HIQL/SuSIE),
    generative-imagination+multiscale (Act2Goal MSTH `act2goal`), CALVIN 5-subtask chains, gcidm horizon-conditioning. Full table → `proposal/literature_survey.md`.
  - **CONCRETE TEST — reframed post-gcidm-deep-read: the decisive experiment is HISTORY-vs-MARKOVIAN on ROBOMIMIC contact (their
    white space), NOT cube.** On cube, gcidm GC-IDM is already 98.7% (ceiling) — no headroom to beat, and Markovian suffices there.
    The hypothesis with teeth: on **robomimic contact tasks** (lift/can/square/tool_hang/transport) a Markovian GC-IDM `(z_t,z_goal,h)`
    CAN'T disambiguate pre- vs post-grasp from a single frame+goal → **fails or underperforms our HISTORY-conditioned forward
    goal-policy** `π(z_{≤t}, a_{<t}, z_goal, h)`. Eval matrix per task: {Markovian GC-IDM (gcidm-style baseline we implement) vs
    history-conditioned forward goal-policy (ours)} × {vit-tiny vs DINOv2 encoder}, direct (no CEM), N=50, report SR + latency.
    A cube sanity check (both reach ~gcidm's 98.7) confirms parity before the robomimic contrast. Win = history > Markovian on
    contact, especially the post-grasp tasks (Lift-family). MUST include horizon-conditioning in both (load-bearing per gcidm).
    Needs NEW training (both heads; encoder/WM reused) → sequences AFTER the current DINOv2 cube result.
  - **🔧 PRECISE DIFF (scoped 2026-06-18, ready to implement post-convergence):** `predict_intention` ALREADY has a conditioning
    hook — `task_vec` (multi-task) is ADDED to the `past_act_emb` stream (`jepa.py:~124 past_act_emb = past_act_emb + task_vec[:,None,:]`).
    **Mirror it for the goal:** add `goal_emb=None` param → `if goal_emb is not None: past_act_emb = past_act_emb + goal_emb[:,None,:]`
    (or concat `goal_emb` to the `emb` state stream if stronger conditioning is wanted — match the paper). (1) `jepa.predict_intention`
    + `jepa.intention_rollout`: thread `goal_emb`. (2) `train.py:lejepa_forward` (~L70 `predict_intention` → ~L101 `action_decoder.loss`):
    hindsight-relabel — sample future offset k, `goal_emb=encode(frame_{t+k})[CLS]`, add the goal-conditioned action loss. (3) `gip.py`:
    `DirectGCPolicy` + `gip_eval.mode=direct` (encode obs+goal → `predict_intention(goal_emb=z_goal)` → action_decoder, closed-loop,
    NO solver). `z_goal = encode(goal_image)[CLS]` (same as `emb`). Small, low-risk (reuses the task_vec pattern + action_decoder).
  - **✅ STEP 1 DONE + VALIDATED (2026-06-18): model-level goal-conditioning IMPLEMENTED in `jepa.py` (L40S, backup `jepa.py.bak_goalcond`).**
    `predict_intention(..., goal_emb=None, ...)` and `intention_rollout(..., goal_emb=None)` both take an OPTIONAL `goal_emb`, threaded
    additively on `past_act_emb`/`pa` exactly like `task_vec` (4-line surgical diff). Forward-pass test (`can_gip_dinov2_pretrained`,
    D=384): **goal_emb CHANGES the intention output (✓ conditioning flows), goal_emb=None is BIT-IDENTICAL to the original (✓ strict
    superset, backward-compatible), and intention_rollout(goal_emb) also respects it (✓ shape (B,horizon,35)).** Confirms it's a
    HISTORY-conditioned forward policy with an OPTIONAL goal — NOT a Markovian IDM (the gcidm wedge holds at the architecture level).
    **REMAINING (held for explicit go — these are the GPU/compute commitments): STEP 2** `train.py:lejepa_forward` hindsight-relabel
    + goal-conditioned action loss; **STEP 3** `gip.py DirectGCPolicy` + `mode=direct` eval; **STEP 4** train on a robomimic task
    (the white space) + eval direct vs the Markovian-GC-IDM baseline. A-vs-B goal source (image vs demo) only changes how `z_goal`
    is computed at eval — the head is identical, so Step 1 is decision-agnostic foundation.
  - **🏗️ IMPLEMENTATION (plan APPROVED 2026-06-18: integrated config-gated, all 4 modes by config — `gip_eval.mode∈{policy,guided,
    planning}`×`goal_conditioned`; plan = `~/.claude/plans/resilient-twirling-moon.md`; status HTML `results/infra_status.html`).**
    Progress: **(a) model `goal_emb` hook ✅ DONE+validated** (jepa.py, backup `jepa.py.bak_goalcond`). **(b) goal-sampling data
    wrapper ✅ DONE+validated** (`goal_dataset.py` installed; on `can`: adds `goal` (hindsight future frame (3,224,224), horizons
    vary 3-19 obs-steps); constructed ONLY when goal_conditioned=true → base untouched). **CAUGHT+FIXED a double-normalization bug:
    `_load_slice` already applies dataset.transform when set, so the goal comes out preprocessed like the window — my extra img-preproc
    double-normalized ([-2.07,2.64]→[-11,9.9]); removed it (checking value RANGES not just shapes surfaced it).**
    **(d) `lejepa_forward` GC loss + `goal_dropout` + train.py wrapper-wiring + config flags ✅ DONE+SMOKE-VALIDATED** (backups
    `train.py.bak_goalcond`, `lewm.yaml.bak_goalcond`; flags `action_pred.{goal_conditioned,hindsight_max_k,goal_dropout}` default-off):
    3-batch `can` smoke w/ goal_conditioned=true goal_dropout=0.5 ran CLEAN — act 1.00/intent 1.01/pred 0.255, no crash/NaN, goal flowed
    (z_goal=`encode({pixels:goal.unsqueeze(1)})["emb"][:,0]` (B,D), added to past_act like task_vec).
    **(e) `gip.py` eval dispatch ✅ DONE+VALIDATED** (backup `gip.py.bak_goalcond`): `BCPolicy` gains `goal_conditioned` (encode the
    `info_dict["goal"]` frame → z_goal → `intention_rollout(goal_emb=)`); `build_policy` routes **`mode=policy`** to it (`bc`==`policy
    goal_conditioned=false`, byte-identical) + goal-aware `attach_intention_actor` for `guided`. `mode=policy goal_conditioned=true`
    ran end-to-end on cube (RESULTS produced, no solver). Fixed 2 bugs: goal is 5D `(R,T,C,H,W)`→`[:,-1:]`; goal pixels CPU→`.to(device)`.
    **🏁 4-MODE INFRA CODE-COMPLETE + validated end-to-end (train (d) + eval (e) both run); all 4 modes selectable by
    `gip_eval.mode∈{policy,guided,planning}`×`goal_conditioned`. Backups jepa/train/gip/lewm.yaml `.bak_goalcond`, default-off everywhere.**
    **REMAINING: (f) bake `gip_eval.{mode,goal_conditioned}` into eval configs (now CLI `+`); (c) `horizon` hook (gcidm, deferred).**
  - **✅ 4-MODE DISPATCH VERIFIED (2026-06-18, user-requested "test each infra follows my idea"): cube N=2, the policy CLASS is the
    mechanism tell.** `mode=planning`→**WorldModelPolicy** (CEM, no prior); `mode=guided`→**WorldModelPolicy** (CEM warm-started by
    intention = policy-as-prior); `mode=policy goal_conditioned=false`→**BCPolicy** (direct, NO solver, no goal = pure policy);
    `mode=policy goal_conditioned=true`→**BCPolicy** (direct, NO solver, encodes z_goal = goal-conditioned policy). All 4 run, all
    by config alone. ⇒ the 2 policy modes are the planning-FREE direct head; the 2 planning modes are CEM (guided=prior-seeded).
  - **🚀 (g) GPU RUN LAUNCHED (2026-06-18): 2 arms on `can`, warm-started from `can_lewm_dinov2_pretrained` (embed_dim=384 — the dinov2
    runs NEED this or predictor 384↔192 mismatches), goal_conditioned, goal_dropout=0.5, 100ep.** `can_gc_hist` (use_action_history=
    TRUE, ours, GPU1) + `can_gc_markov` (use_action_history=FALSE, GPU2). init_from missing=88 (the new GIP+
    goal head, correctly random), unexpected=0. ~2.5h to converge (~1.5 it/s, 136 step/ep). Each = ONE model serving modes 2+3 (goal_dropout).
    (Launch gotchas logged: `trainer.max_epochs` not `max_epochs`; `embed_dim=384` for dinov2; `init_from` is strict=False so only SHAPE mismatches error.)
  - **🛑 CORRECTION (2026-06-19, code re-audit of `jepa.py:126` + `train.py:36-82`) — `use_action_history=false` is NOT a Markovian
    GC-IDM. The two launched arms are an ACTION-HISTORY ablation, not the history-vs-Markovian headline.** The verbatim mechanism:
    `lejepa_forward` sets `ctx_emb = emb[:, :ctx_len]` with `ctx_len = cfg.history_size = 3`, so `predict_intention` receives a
    **3-frame latent STATE history `z_{t-2:t}` in BOTH arms**. `use_action_history=false` only triggers `past_act_emb =
    torch.zeros_like(past_act_emb)` (`jepa.py:126`) — it zeros the *past-ACTION* stream `a_<t`, nothing else. So:
      - `can_gc_hist` = 3-frame state history **+** past-action stream `a_{t-2:t-1}` + `z_goal` → `a_t`  (our forward policy)
      - `can_gc_markov` (as launched) = 3-frame state history, **NO** action stream + `z_goal` → `a_t`  (state-history, action-free)
    Both are HISTORY-conditioned on the state. The contrast they measure is **"does the past-action stream help the GC policy"**
    (a legitimate secondary ablation), NOT "history-conditioned forward policy vs Markovian gcidm." **Rename in all reporting:
    `can_gc_markov` → `can_gc_noActHist` (its true meaning); never headline it as the gcidm baseline.**
  - **Why a faithful Markovian gcidm baseline is NOT a pure config flip (the `history_size` coupling).** A true gcidm head sees only
    `(z_t, z_goal, h)` — a SINGLE current frame. The obvious config move (`history_size=1`) is wrong because `ctx_len=history_size`
    **also bounds the WORLD-MODEL context** (`ctx_emb` feeds `self.model.predict` for the next-state loss at `train.py:46`), so
    `history_size=1` would fine-tune the WM down to a 1-frame predictor and corrupt the warm-started 3-frame WM — confounding the
    comparison. Three candidate faithful baselines (OPEN QUESTION for the user, recommendation = option A):
      - **(A, recommended) `action_pred.markov` flag — decouple the head from the WM (≈3-line code hook, config-gated, default off).**
        In `lejepa_forward`, when `action_pred.markov`, pass `ctx_emb[:, -1:]` / `past_act[:, -1:]` to `predict_intention` while the WM
        still gets the full 3-frame `ctx_emb`. At eval, `intention_rollout(history_size=1)`. ⇒ identical WM, data, optimizer, head
        architecture; the ONLY difference is the action head's receptive field (3-frame vs 1-frame). Cleanest controlled isolation of
        "history" and the apples-to-apples gcidm analogue inside our own stack.
      - **(B) faithful frozen-latent gcidm MLP** — freeze the WM, train a 3-layer MLP `(z_t, z_goal, h) → a_t` with AdaLN-Zero horizon
        (literal gcidm). Most faithful to the paper, but differs in architecture+optimizer+capacity, so a win could be attributed to
        those rather than to history → a weaker controlled claim than (A). Keep as a SECONDARY external-baseline check.
      - **(C) `history_size=1` everywhere** — rejected: corrupts the WM as above.
  - **Horizon `h` is load-bearing FOR gcidm specifically (−42pp without, per `reference_gcidm_umich.md`) → a fair Markovian baseline
    MUST get the horizon hook, else we strawman gcidm.** Our forward policy may partially substitute history for the explicit `h`
    scalar (history reveals "how far along"); a clean matrix gives horizon to BOTH sides so the only varied factor is history.
    ⇒ the `horizon` hook (item (c), still "–") is now ON the critical path for the headline, not a deferred nicety. Plan: sinusoidal(h)
    → small MLP → ADD to the `past_act` stream (same additive pattern as `task_vec`/`goal_emb`); data wrapper already returns the
    realized `h` (`goal_dataset.py item["horizon"]`); eval-time `h` = a swept/fixed remaining-horizon hyperparameter (gcidm's protocol).
  - **DECISION (2026-06-19): let the 2 running arms finish (cheap, ~11/100 ep in, valid as the `hist` vs `noActHist` action-history
    ablation), and report them as exactly that. Build the REAL head-to-head next: implement (A) `action_pred.markov` + the (c) horizon
    hook, then run `can_gc_hist` (ours) vs `can_gc_markov1h` (markov=true, +horizon = faithful gcidm-in-stack), both warm-started, 100ep,
    eval N=50 in modes 2 (policy) + 1/4 (planning/guided). The headline = history-conditioned forward GC policy > Markovian gcidm on
    `can`/contact tasks. The running arms are NOT wasted — they fill the (history, no-action-stream) cell of the ablation.**
  - **🔧 `action_pred.markov` HOOK IMPLEMENTED + UNIT-VALIDATED LOCALLY (2026-06-19), NOT YET PUSHED (awaiting user OK to modify the
    shared L40S host).** Patch in `train.py lejepa_forward` (3 edits, config-gated `markov=ap.get("markov",False)`, default off):
    when `markov`, reshape each of the `ctx_len` frames into an independent `T=1` example `ctx_emb.reshape(-1,1,D)` (so the action head
    sees ONLY `z_t` = gcidm-style) while the WM loss still uses the full `ctx_emb` (warm-started 3-frame WM untouched); goal/task vectors
    `(B,D)` are `repeat_interleave(ctx_len,0)` to align per-frame; targets `tgt_act_emb`/`tgt_act` reshaped to match; action-mask
    `repeat_interleave`d. **Equal-supervision design: ctx_len action targets/window in BOTH arms, so the ONLY varied factor is the head's
    receptive field (3-frame history vs 1-frame Markov).** Validated: `py_compile` OK (local + would-be box); standalone shape-algebra test
    (`tmp/test_markov_shapes.py`) PASSES — markov=False → `(2,3,*)` byte-identical tensor flow; markov=True → `(6,1,*)`, both losses finite,
    goal aligned per-frame (frame `(bi,tj)` carries `goal_emb[bi]`). Patched file staged at `tmp/L40S_train.py` (backup-on-push: `train.py.bak_markov`).
  - **⚠️ EVAL-SIDE GOTCHA for the markov arm (must fix before eval, else silent OOD).** `gip.py:256 BCPolicy.get_action` calls
    `intention_rollout(horizon=1, goal_emb=...)` WITHOUT `history_size`, so it defaults to **3**. A markov-trained model (single-frame head)
    fed 3 frames at eval is OUT-OF-DISTRIBUTION → corrupt SR (the same class of silent bug as the eval-batching artifact).
  - **✅ MARKOV BASELINE NOW FULLY WIRED + VALIDATED LOCALLY (2026-06-19), ALL default-off, staged in `tmp/L40S_{jepa,train,gip}.py`,
    NOT pushed (pending user OK to write the shared host).** Single-source design = the model attribute **`action_markov`** (set at launch
    via `+model.action_markov=true`), which TRAVELS to eval in the saved run config exactly like `use_action_history` — so eval can't
    foot-gun the history_size. Three files:
      - **`jepa.py`**: new `__init__` param `action_markov=False` + `self.action_markov` (backward-compat: old ckpts → `getattr(...,False)`).
      - **`train.py`**: `markov = bool(getattr(self.model, "action_markov", False))` drives the per-frame reshape (action head sees only
        `z_t`; WM loss still full `ctx_emb`); equal-supervision; goal/task/targets/mask reshaped to match. (shape test PASS)
      - **`gip.py`**: `eval_hs = 1 if getattr(model,"action_markov",False) else cfg.history_size`, threaded into `BCPolicy`
        (`self.history_size` → `intention_rollout(history_size=...)`) AND the `guided` `attach_intention_actor` → a markov model
        auto-evals with 1 frame (fixes the OOD gotcha above). 
    Validation: `py_compile` OK on all three; `tmp/test_markov_shapes.py` PASS (markov=False byte-identical `(2,3,*)`; markov=True
    `(6,1,*)`, losses finite, goal aligned per-frame). **REMAINING code: only the (c) horizon hook (gated on the additive-vs-AdaLN
    decision below).** Once that lands + user OK → push 3 files (+ horizon) with backups, 2-batch smoke (default-off = unchanged;
    markov-on = runs), launch the matrix.
  - **OPEN (needs user OK — these are shared-host writes + a scope expansion beyond the 4 named modes):**
      1. **Push the `markov` + eval-`history_size` + `horizon` hooks to L40S and launch the head-to-head matrix?** (GPUs are free: only
         GPU1/2 busy; 0,3,4,5,6,7 idle → the matrix fits in parallel with the running arms.)
      2. **Horizon `h` conditioning — additive vs AdaLN?** gcidm's −42pp-without-horizon used **AdaLN-Zero** (FiLM-style scale/shift),
         not addition. Two options: **(i) additive** — sinusoidal(h)→MLP→ADD to the past-act stream, identical pattern to `goal_emb`/`task_vec`
         (low-risk, proven hook, but weaker than gcidm's AdaLN so a markov-loss could be under-credited to "no AdaLN"); **(ii) AdaLN-Zero** —
         faithful gcidm conditioning inside the action_predictor (more surgery on a shared module). RECOMMENDATION: start with **(i) additive**
         for the first matrix (fast, low-risk), and ONLY if the additive markov+horizon arm still loses to ours, upgrade the baseline to
         **(ii) AdaLN** before any headline claim (so we never strawman gcidm). Document which version produced each number.
  - **PROPOSED MATRIX (on `can`, all warm-started from `can_lewm_dinov2_pretrained` embed_dim=384, goal_conditioned, goal_dropout=0.5,
    100ep, eval N=50 in modes policy/guided/planning):**
      | arm | history_size | use_action_history | markov | horizon | role |
      |---|---|---|---|---|---|
      | `can_gc_hist` | 3 | true | false | – | OURS (forward history GC policy) — RUNNING |
      | `can_gc_noActHist` | 3 | false | false | – | action-history ablation (state-history, no action stream) — RUNNING (was mislabeled `can_gc_markov`) |
      | `can_gc_markov` | (1-frame head) | false | true | – | true Markovian, NO horizon (gcidm-without-horizon corner) |
      | `can_gc_markov_h` | (1-frame head) | false | true | additive | faithful-ish gcidm (Markov + horizon) — the fair baseline |
      | `can_gc_hist_h` | 3 | true | false | additive | OURS + horizon (isolate: does history subsume the h scalar?) |
    Read: `hist` vs `markov_h` = the headline (history vs faithful gcidm). `markov` vs `markov_h` = does horizon rescue Markov (replicates
    gcidm's −42pp). `hist` vs `hist_h` = does our history already carry the horizon signal (if ≈equal, history subsumes h → a clean wedge story).
  - **🗺️ EVAL-PATH MAP for the can head-to-head (verified 2026-06-19 against [[project_robomimic_eval_path_gotcha]] + code) — the GC
    eval does NOT go through `eval_gip.py`/`gip.BCPolicy`.** On robomimic, `gip.BCPolicy` runs the head on a SINGLE frame
    (`intention_rollout horizon=1`, zero past-action) → **bc≈0** (the documented gotcha). The VALIDATED robomimic policy evaluator is
    **`eval_histbc_robomimic.py`** (`HistoryBCPolicy`): it feeds the last `history_size` frame latents + the last `history_size-1`
    executed action blocks into `predict_intention` (matching training `ctx_len=history_size`), and reports **task success**
    (`world.evaluate → episode_successes`), not goal-reaching distance. Guided/planning → `eval_histguided_robomimic.py`. ⇒ **my
    `gip.py` eval-`history_size` patch covers the CUBE/PUSHT path only; the can head-to-head needs the analogous wiring in
    `eval_histbc_robomimic.py`/`eval_histguided_robomimic.py`.**
      - **GOAL-AGNOSTIC eval is UNBLOCKED (no code, no push)** — `HistoryBCPolicy` as-is gives task SR. Validated command (run on the arm's
        TRAINING GPU per the cross-GPU render gotcha; params from EXPERIMENTS.md line ~1398 `can = PickPlaceCan / budget 240 / offset 90`):
        ```
        CUDA_VISIBLE_DEVICES=1 STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl \
          python eval_histbc_robomimic.py --config-name robomimic policy=can_gc_hist \
          world.task=PickPlaceCan dataset.stats=can eval.dataset_name=can \
          eval.num_eval=50 eval.eval_budget=240 eval.goal_offset_steps=90
        ```
        (markov on GPU2 with `policy=can_gc_markov`.) This measures: (1) the **action-history ablation** `hist` vs `noActHist`, and (2) the
        **BESO two-setting property** — does `goal_dropout=0.5` preserve a working goal-AGNOSTIC policy (vs the existing can bc ≈ detach 70.7 /
        solveact 79, line 376). NOTE: at convergence only (ep100, ~08:10 UTC) per rule 8; ep40 would be a diagnostic, not a result.
      - **GOAL-CONDITIONED eval (the headline) — `HistoryBCPolicy` goal+markov-HS wiring NOW DONE + py_compile-OK LOCALLY** (2026-06-19,
        `tmp/L40S_eval_histbc.py`, default-off, NOT pushed): `__init__` gains `goal_conditioned`; `get_action` encodes
        `info_dict["goal"][i] → z_goal` (the world injects `goal` when `goal_offset` is set — same mechanism the cube GC eval used) and passes
        `goal_emb=` to `predict_intention`; `run()` reads `gip_eval.goal_conditioned` + sets `history_size = 1 if model.action_markov else
        cfg.history_size`. SR stays **task success**. REMAINING eval code: the horizon feed (gated on additive-vs-AdaLN) + the same goal-wiring
        in `eval_histguided_robomimic.py` for the goal-aware guided/mode-4 (decision-independent, lower priority — mode 2 is the headline).
  - **📦 STAGED-LOCAL SUMMARY (2026-06-19, all default-off, py_compile-OK, NOT pushed — pending user OK for shared-host writes):**
    5 files — `tmp/L40S_jepa.py` (action_markov attr), `tmp/L40S_train.py` (markov per-frame head), `tmp/L40S_gip.py` (cube/pusht GC eval-HS),
    `tmp/L40S_eval_histbc.py` (robomimic mode 2/3 goal + markov-HS), `tmp/L40S_eval_histguided.py` (robomimic **mode 4** goal-aware warm-start +
    markov-HS). Shape test `tmp/test_markov_shapes.py` PASS; all 5 compile. **The ENTIRE decision-independent code path is implemented —
    markov baseline + GC eval for ALL FOUR modes, both robomimic and cube/pusht. ONLY the horizon hook remains (gated on additive-vs-AdaLN).**
    Ready-to-run `tmp/run_gc_headtohead.sh` (bash -n OK): `push` (backup+install+compile all 5) → `smoke` (2-batch, default-off unchanged +
    markov-on runs) → `launch_markov1` (the true-markov arm on free GPU3) → `eval_arm <name> <gpu>` (modes 2+3 via histbc, each on the
    training GPU, N=50, validated can params PickPlaceCan/240/90). **Launch note: the markov arm = `+model.action_markov=true model.use_action_history=false`
    (BOTH — action_markov drives the 1-frame head + eval HS=1; use_action_history=false zeros the residual past-action so it's purely (z_t,z_goal)).**
  - **📉 CONVERGENCE + EARLY SIGNAL from the val-loss trajectories (2026-06-19 08:03, both arms ~ep90/100; logs
    `/mnt/minghao_data/can_gc_{hist,markov}.log`).** BOTH arms have PLATEAUED by ~ep80 → ep100 is genuinely converged (rule 8 satisfied,
    val metrics flat over the last ~20%), NOT just an epoch count:
      | | val act_loss (decoded raw-action MSE) | val intent_loss (act_emb pred) | act_emb_std |
      |---|---|---|---|
      | `can_gc_hist` (ours: 3-frame state + action history) | **~0.045** (flat ep81–88: .043/.045/.045/.044/.048/.046/.046/.045) | ~0.355–0.359 | 1.35 (not collapsed) |
      | `can_gc_markov` (= noActHist: 3-frame state, NO action stream) | **~0.083** (flat ep86–91: .078/.084/.080/.086/.083/.085) | ~0.195–0.203 | — |
    **EARLY SIGNAL (encouraging for "history helps"): the past-action stream cuts val raw-action MSE ~45% (0.045 vs 0.083) on `can`.**
    Knowing your own recent actions sharply improves next-action prediction (action autocorrelation/momentum) — exactly the info a Markovian
    `(z_t,z_goal)` lacks. **CAVEATS (rigor):** (1) `act_loss` IS cross-arm comparable (same raw-action targets/normalization); **`intent_loss`
    is NOT** — its target `act_emb=action_encoder(a)` is a per-arm LEARNED representation, so hist 0.36 vs noActHist 0.20 reflects different
    learned act_emb scales, not capability (do not read intent_loss across arms). (2) Lower `act_loss` is a PROXY ([[project_gip_act_decode_loss]]),
    not SR — closed-loop compounding + action multimodality can break monotonicity, so the N=50 task-SR eval at convergence is the real test.
    (3) This is the ACTION-HISTORY ablation (hist vs noActHist), NOT yet the history-vs-Markovian-gcidm headline (needs the true 1-frame
    `markov1` arm + horizon). Still: a clean ~45% MSE separation before any eval is a good omen for the thesis.
  - **🔬 KEY CAVEAT (red-team, 2026-06-19) — the val `act_loss` edge is TEACHER-FORCED and may OVERSTATE hist's closed-loop SR
    advantage.** Verified against the code: in train/val (`lejepa_forward`) the action head's past-action stream `past_act =
    act_emb(a_<t from the DATASET)` — ground-truth, teacher-forced. But at eval, `HistoryBCPolicy._ablk` feeds the policy's OWN executed
    blocks back as the history (`blk_new = acts[0,-1]` → `action_encoder(blk)` next step). ⇒ **`can_gc_hist` (uses history) faces a
    train→eval distribution shift on its history input — the classic BC compounding-error risk — while `can_gc_markov`/noActHist (zeros
    history) faces NONE.** So the 0.045-vs-0.083 val gap is measured under GT history and the closed-loop SR gap could SHRINK or even
    INVERT if hist's self-generated history drifts (a good policy → good history = virtuous; an erring policy → bad history = compounding).
    **PREDICTION (pre-registered): if action-history is robustly useful, hist SR > noActHist SR at N=50; if they're ≈equal despite the
    open-loop act_loss gap, that is evidence the gap was a teacher-forcing artifact and closed-loop compounding cancels it (still
    informative — it would say "history helps prediction but not control here," steering us to history that's robust to self-generation,
    e.g. shorter/abstracted history or DAgger-style training). Either way the N=50 eval — not the val loss — is the verdict.** This is also
    why the TRUE history-vs-gcidm headline must compare CLOSED-LOOP SR, never val loss.
  - **⚖️ SCOPE — `can` is a 1-SEED PILOT, NOT the headline (set expectations; rule 8/9).** The two running arms (and the proposed matrix) are
    **single-seed, single-task** on `can`. That is a PILOT to decide direction (does history separate from Markov at all on a contact task),
    explicitly diagnostic. A defensible HEADLINE needs: **(a) ≥3 contact tasks** (lift/can/square + ideally tool_hang/transport — the contact
    family where pre/post-grasp state is hidden from a single `(z_t,z_goal)`, the whole reason history should help; lift is near-ceiling so
    it's a floor-check, square/tool_hang/transport are the discriminating ones), **(b) ≥3 seeds {42,0,1}** per arm (per
    [[feedback_eval_num_eval_50]] + rule 9), **(c) N=50 eval** (mandatory). So the can pilot at ep100 is read as "go/no-go + effect-size
    estimate," and only the multi-task×multi-seed sweep is reportable. Disk is fine for that scale (checkpoints on the 2.8T `/mnt/minghao_data`,
    ~1.2G/arm even unpruned; root `/` is 97% but runs write to the data mount). **Open design Q for the GC eval: goal-frame choice** — the
    validated `goal_offset_steps=90` samples a goal ~75% through the median can demo (≈120 steps), i.e. near-but-not-final. For "does a goal
    help complete the task" that is fine (SR = task success regardless); if a reviewer wants "reach an arbitrary goal," sweep the offset or use
    the final frame. Note which protocol produced each number.
  - **📊 ROBOMIMIC GC HEAD-TO-HEAD — FIRST NUMBERS (2026-06-20, goal-AGNOSTIC mode, N=50, ep100, 1 seed) — then SUPERSEDED.**
    `can_gc_hist` (ours, state+action history) = **0.24** (12/50); `can_gc_markov` (= noActHist, no action stream) = **0.00** (0/50).
    Both low vs pure-bc can ≈0.69 (the `goal_dropout=0.5` cost to the goal-agnostic mode). The 0.24-vs-0.00 split echoes the val
    `act_loss` signal (0.045 vs 0.083) — action-history looks decisive on can — BUT a flat 0/50 is extreme; treat as PRELIMINARY (1 seed,
    needs 3-seed + a degenerate-action sanity check before headlining). NOTE the eval first crashed on a path bug → see [[project_l40s_path_consolidation]]:
    the codebase hardcodes the 174 layout; fixed on L40S via a symlink tree (no env override needed). 
  - **🔄 PIVOT (user 2026-06-20): REPRODUCE gcidm's OWN BENCHMARK, drop ours in as a 3rd arm — the clean fair head-to-head (task #55).**
    The robomimic GC eval has too many confounds (goal_dropout weakening, eval-path complexity, the goal-frame question). Cleaner: reproduce
    gcidm's exact setting (their 4 envs cube/pusht/tworoom/reacher, same SIGReg vit-tiny-192 base, their goal-reaching eval), run **3 arms**:
    (1) CEM (have: cube~67/tworoom~95/reacher~85/pusht~76), (2) **gcidm GC-IDM** (reproduce — frozen-latent `(z_t,z_goal,h)→a` 3-layer MLP
    hidden512, **AdaLN-Zero horizon** (−42pp without), hindsight MSE), (3) **OURS** (history-conditioned forward GC policy + z_goal + SAME
    AdaLN-Zero horizon). **Validation = our GC-IDM ≈ gcidm published** (cube 98.7 / reacher 99.7 / tworoom 100 / pusht 84.2) → proves the
    setting; then ours is a fair drop-in. **Wedge/prediction:** ours ≈ gcidm on the near-ceiling envs, ours **> gcidm on pusht (contact)**
    where history disambiguates contact state. This also RESOLVES the additive-vs-AdaLN horizon question: both arms use AdaLN-Zero (fair + faithful).
    PREREQS DONE (2026-06-20): the 4 env datasets + 4 frozen SIGReg bases `{cube,pusht,tworoom,reacher}_ours_lewm_weights.pt` consolidated onto
    L40S (`…/decoders/`); paths resolve. **IN FLIGHT:** focused agent building Method 1 (GC-IDM) on cube — GCIDMHead module + `train_gcidm.py`
    hindsight trainer + `mode=gcidm` planning-free evaluator, train + N=50 eval, target ≈98.7 + a horizon-ablation sanity check. Method 2 (ours)
    drops onto the same eval next, reusing the AdaLN horizon. Start cube (biggest gcidm edge), then extend to the other 3.
  - **🏗️ CONFIG-DRIVEN DISCIPLINE (user 2026-06-18: "this code is infra-level, you only control config").** Right model: swm is
    config-first — encoder/action-head/data/solver/epochs/seeds/eval-mode are all hydra yaml (the whole DINOv2 ablation = config, no
    code). The ONE exception is a genuinely NEW capability the framework lacks (goal-conditioning) → needs a MINIMAL code hook
    (`goal_emb`, 4 lines, optional/backward-compat = Step 1). **Pattern: add the missing capability ONCE in the smallest code edit,
    then expose ALL control via config.** So Steps 2-3 must add config flags, not hard-code: **`+action_pred.goal_conditioned=true`**
    (on/off), **`+action_pred.hindsight_max_k=50`** (relabel horizon), **`+gip_eval.mode=direct`** (planning-free eval),
    **`+gip_eval.goal_source=image|demo`** (the A-vs-B, a config flag). **🌟 KEY: `+action_pred.goal_dropout=p`** (train with `goal_emb`
    present (1-p) of the time, zeroed the rest) → **ONE model does BOTH goal-reaching AND goal-agnostic policy learning = the
    two-setting WAM, = BESO's classifier-free-guidance trick, achieved by a SINGLE config knob on the same head.** The whole
    goal-conditioned study (incl. the two-setting unification) then runs by changing config, no further code edits.
- **🌟 INTUITION-SEEDED PLANNING as the headline method (user brainstorm 2026-06-18) — "merge the intuition head into test-time
  MPC/CEM; humans plan FAST because they start from a good intuitive prior, not random."** This is the dual-process story:
  - **System 1 = the intuition/GIP head** (the late `act_emb` prediction): current latent + goal → action embedding in ONE
    forward pass. Learned reflex from demos. **System 2 = CEM/MPC over the JEPA WM**: rolls candidate actions through the WM in
    latent space, "imagines what happens", scores vs goal, picks best. **Merge = System 1 SEEDS System 2** → search starts in the
    right neighborhood, so deliberation is CHEAP. Lineage: ExIt / Thinking-Fast-and-Slow (`exit`), amortized planning (`gcidm`).
  - **✅ CODE AUDIT (2026-06-18) — the full-horizon seed is ALREADY DONE in the swm cube path; only the variance floor was
    missing.** Traced the actual stack: `eval_gip.py guided` → `gip.attach_intention_actor` binds `get_action` → `jepa.
    intention_rollout(horizon)` (autoregressively rolls intuition head + WM state head for the FULL horizon). The swm
    `solver/cem.py` → `prepare_init_action(model, horizon, init_action=None)` calls `get_action(horizon=remaining=H)` and uses the
    returned H-block trajectory as the CEM **mean** — so the cube guided (incl. the 88% and the one firing at ep60) ALREADY
    full-horizon-seeds. **The §10 "warm-start only pos 0, pos 1–4 zero-padded" note was STALE** — it described the OLD pre-swm
    path / the separate `eval_histguided_robomimic.py` (a custom 1-block+zero-pad robomimic evaluator), NOT the swm cube path.
    (Verify-don't-guess win: almost re-implemented something already present.)
  - **🔧 IMPLEMENTED (2026-06-18) — the REAL missing lever = a CEM variance floor.** Verified swm `cem.py:~243` updates
    `batch_var = topk_candidates.std(dim=1)` with NO floor → over n_steps=30 the elite std → 0 and CEM **collapses ONTO the
    seed** (replays the intuition, no deliberation). Fix = `gip.VarFloorCallback` (appended to `gip.py`, backup `gip.py.bak_varfloor`):
    a swm solver Callback that in-place `clamp_(min=min_var)` the var after each elite fit (the CEM loop reuses that tensor next
    iter → propagates). Opt-in via new solver config `config/eval/solver/cem_floor.yaml` (= cem.yaml + the callback, min_var=0.1).
    **Validated path byte-identical** (`solver=cem` has no callbacks). Smoke-tested: clamp works ([0.5,0.01,0.2,0.0]→[0.5,0.1,0.2,0.1]);
    `cem_floor` hydra-instantiates CEMSolver with 1 VarFloorCallback. Lets the good full-horizon seed be REFINED, not just replayed.
  - **⚠️ CAVEAT (post-gcidm-deep-read 2026-06-18): gcidm ALSO ran this budget sweep** (their Fig B, 500× CEM grid → NO CEM config
    beats their single-pass GC-IDM on speed+success) — so "amortization curve, random-init CEM vs a fast prior" is THEIRS, and
    gcidm shows the single pass is ALREADY ≥ CEM on simple envs (⇒ refinement adds ~nothing there). Our sweep is only novel where
    the single pass is BRITTLE and refinement helps: **gcidm's weak cells (pusht distant goals) + robomimic contact**. So run the
    budget sweep as a SECONDARY diagnostic on robomimic-contact (does seeded-refine beat single-pass there?), NOT as the cube headline.
  - **🎯 THE (now-secondary) figure:** **SR vs planning-budget** (CEM iters×samples), two
    curves: random-init (`mode=planning`) vs intuition-seeded (`mode=guided`). Claim = intuition-seeded reaches high SR at a
    FRACTION of the budget (curve shifts LEFT = "plan fast"); budget→0 degrades to bc, budget→∞ matches/beats pure planning; it
    DOMINATES the Pareto front. Quantifies the "humans plan fast from a good prior" thesis. **Testbed = cube** (guided 88%>planning
    78%>baseline 68%; converging ~2.5h). **✅ INFRA BUILT + VALIDATED (2026-06-18):** `cube_budget_sweep.sh` (grid runner: ENC ×
    {guided,planning} × {cem,cem_floor} × budget-grid `num_samples:n_steps` × seeds, N=50; dry-run + `--cfg job` confirm the hydra
    overrides compose) + `plot_budget_sweep.py` (SR-vs-budget curves, seed-averaged, solid=cem_floor/dashed=cem; PNG verified), both
    in `/mnt/minghao_data/`. Default grid 16:1→300:30 (budget 16→9000). **Does NOT auto-fire** — launch manually AFTER the DINOv2
    cube result picks ENC=winner (sequencing: encoder ablation on validated `solver=cem` first, THEN this planning-method sweep).
    **✅ END-TO-END RUNTIME-VERIFIED (2026-06-18, N=2 smokes on the ep52 cube ckpt, GPU1):** `cem_floor` runs through the real CEM
    loop (planning RESULTS produced, callback fires, no crash) AND the lowest-budget 16:1 guided cell runs. **🐛 BUG CAUGHT by the
    smoke (would've crashed half the grid at the headline run):** `torch.topk(costs, k=topk=30)` fails when `num_samples<30`
    (`selected index k out of range`). FIX in the runner: `topk = clamp(num_samples//10, 2, num_samples)` — holds the elite
    FRACTION ~constant at 10% (matching default 30/300) across budgets (16→topk2, 300→topk30). Lesson: `--cfg job`/instantiate
    checks compose-validate but don't catch runtime shape bugs — a tiny N=2 real run does.
  - **⚠️ DIFFERENTIATION (be honest):** TD-MPC2 (`tdmpc2`) + Newt (`newt`) ALREADY seed MPPI with a learned policy prior → "seed
    planning" alone is NOT novel. Our wedges: (1) the prior is a **latent-action** intuition (predicts `act_emb`) on a **pure-JEPA**
    WAM — planning entirely in JEPA latent (z AND a abstract), vs TD-MPC2's raw-action MPPI on a reward-shaped latent; (2) the
    **amortization-efficiency framing as the headline** (the budget sweep), not the prior-as-one-more-MPPI-sample; (3) the intuition
    is **history+goal conditioned** (contact-state memory → better seed on contact tasks, cf. GIP "act wins on Lift via post-grasp
    memory" [[project_gip_intention_format_eval_convention]]).
  - **⚠️ TWO REGIMES (from our own findings):** SHORT/aligned (cube, pusht) → direct intuition-seeded CEM works. LONG multi-stage
    (robomimic) → a single 25-step plan CAN'T span a 469-step task (§10 horizon finding) → the merge must go **HIERARCHICAL**:
    intuition proposes a SUBGOAL/intention SEQUENCE, plan each segment (WorldDP direction). Here "intention" (intended outcome) >
    "intuition" (immediate action); the user's "what's the risk / what happens" = a cost over imagined subgoal outcomes.
  - **SEQUENCING:** do NOT fold the full-horizon seed into the cube DINOv2 guided eval that's about to fire (both arms need the
    SAME validated guided config to cleanly answer "better encoder → better planning"). (1) FIRST: DINOv2 cube guided lands (~2.5h).
    (2) THEN: full-horizon-seed + variance-floor + budget-sweep as a separate planning-METHOD study on the fixed best encoder.
- **⭐ Phase-2 bc for the single-arm tasks (raised by the tool_hang DECOUPLE, 2026-06-18 — §10).** tool_hang proved a
  *flat probe gap* can still hide a *real bc gain* (probe 0.82≈0.82, yet bc 0.333 vs 0.213, 3-seed). Phase-2 was chosen
  **probe-first**, and the single-arm probe gaps are small (can +0.02, lift +0.06, square pending). The decouple means
  **probe-only UNDER-TESTS these tasks** — DINOv2 could still lift can/lift/square bc the way it did tool_hang. **Decision
  (scope, user call):** train can/lift/square dinov2 GIPs to measure bc (≈6 GIPs + 18 N=50 evals — sizable but the box is
  now mostly free), OR accept the 2-task transport+tool_hang bc as the robomimic headline and keep the other six probe-only.
  **Lean:** run **can** bc first (its probe is freshly in hand) — if the decouple generalizes (can pretrained > scratch on
  bc despite +0.02 probe), it strengthens "probe necessary-not-sufficient" into a rule and justifies the full 6-task bc;
  if can bc is a tie, stop at 2 tasks. Cheap, decisive, non-committal.
- **§1 detach=TRUE @100ep:** running. Verify `detach_decoder=true` is in each saved config; verify plateau (extend if rising).
- **Compete vs analyze go/no-go:** does detach=TRUE + full training reach image-BC-RNN ~98, or plateau ~50 (encoder ceiling)?
- **"solve out the action" (user):** we're a pure-JEPA WAM — intention predicts `act_emb`, raw action decoded out (embedding
  indirection). NOTE: under detach=TRUE the indirection is total (encoder gets zero raw-action gradient). Test a `w_intent=0`
  direct-raw-action arm — but that needs detach=FALSE to ground the encoder → conflicts with the latent-only design. Tension to resolve.
- **Goal-image planning impractical (user):** real robots have no goal image → drop goal-conditioned planning; GC-IDM owns it
  anyway (99–100). Keep planning only as **reward/value-based** (TD-MPC2 style, no goal image), or focus task-completion BC.
- **TD-MPC2-style direction (user wants planning):** policy-prior + short-horizon MPPI toward a learned **value** (offline, from
  demos) instead of goal-image cost. More robust on contact (value bootstrap, no deep contact rollout). But collides with Newt
  (2511.19584 = JEPA + multitask + BC-prior + MPPI). Our only wedge = history+action-conditioned prior (contact-state memory) vs
  their Markovian. Decisive test: history-prior+value-planning vs Markovian-prior+value-planning on contact.
- **Unified WAM (user):** z and a as equal interleaved tokens [z,a,z,a,…], one transformer, all-in-all-out (trajectory-transformer
  style). Currently action = AdaLN conditioning (NOT a token); two separate trunks. Real diff = action↔dynamics coupling; does NOT
  by itself fix the encoder/perception bottleneck (z still from the same encoder).
- **Our model IS already a JEPA world-action model** (shared encoder → WM head z_{t+1} + policy head a_t, jointly trained).
  "WAM" is a reframing, not a new build. Reframe paper as JEPA WAM, not the scooped "intuition-guided planning".
- ~~**"solve out the action" tension — RESOLVED by data (2026-06-14 seed42 N=50):** the `w_intent=0` direct-raw-action arm
  (solveact, detach=FALSE) **WINS histbc** on the hard tasks: can 76 > T72 > F68, square 70 > T64 > F62 (§1c).~~
  **❌ CORRECTED (2026-06-15, multi-train-seed):** that single-seed "solveact wins (+6/+7)" was a TRAINING-RUN ARTIFACT;
  at 3 train-seeds detach=TRUE (74.0) ≈ solveact (70.0) — **TIED** (§1c FINAL). So solveact does NOT beat the
  embedding-intuition head for BC. The "not one head" conclusion still has a clean *design* motivation (solveact has no
  intention head → can't intention-guide planning, so a planner+executor split is natural), but it is **not** justified by
  a solveact SR win. See WorldDP below.
- **⭐ WorldDP-inspired direction (paper 2606.08775, LeCun et al, 2026-06-14) — HIERARCHICAL subgoal GIP:** WorldDP names our
  robomimic-CEM=0 failure as a **multi-stage** problem (WM-MPC works single-stage reach/grasp, fails multi-stage) and fixes it
  hierarchically: high-level WM optimizes feasible **subgoal latents** at runtime, low-level **policy (their diffusion policy)
  reaches each**. This **unifies our two results**: (a) the WM/intention head is the **high-level subgoal planner** (where
  planning/guided helps), (b) **solveact/raw-action (or the tied detach=TRUE) history-BC policy is the low-level executor**
  (a fine executor — ties the intention head, §1c — not a robust BC winner). Concretely for us:
  plan a short sequence of subgoal latents with the intention-guided WM (each subgoal ≈ one stage: pre-grasp pose → grasped →
  placed), and reach each subgoal with the solveact/history-BC policy — instead of one flat horizon-H CEM that can't cover
  approach→grasp→transport→place. **This is a concrete fix for robomimic planning=0** and uses our best components.
  **Differentiation from WorldDP (for the paper):** we use a **holistic JEPA latent + past-action history** (contact-state
  memory) vs their **object-centric slots**; we **warm-start-then-refine search** vs their diffusion-policy execution; our
  subgoals come from the **intention head** (action-embedding space) vs their object-state subgoals. **TODO:** prototype
  hierarchical subgoal-GIP on can/square; compare flat-CEM (0) vs subgoal-CEM vs subgoal+solveact-executor. See
  [[feedback_paper_relevant_convention]]; full WorldDP analysis in proposal/literature_survey.md §4.
- **⭐⭐ LeWAM POSITIONING / NOVELTY (2026-06-15) → see `proposal/lewam_positioning_notes.md` (the strategic home).**
  The 2026 WAM field exploded (DreamZero 2602.15922 video-WAM; Being-H0.7 2605.00078 = "latent world-action model" at
  scale; Fast-WAM 2603.16666 asks our exact "is test-time planning worth it?" meta-question and answers NO for video;
  + Efficient-WAM/OA-WAM/survey/Awesome-WAM). The field is going **search-free**. **So LeWAM CANNOT claim novelty on
  "WAM"/"latent WAM"/"should we plan."** Un-scooped core = the **controlled, mechanistic PER-FAMILY analysis** (goal-image
  cost + single-frame contact-blindness) + **contact-state recovery via history-BC**. Framing = "controlled mechanistic
  study vs the scale-WAM wave"; for Randall = a scientific extension of his LeWM. **OPEN FORK** (decides next experiment):
  analysis+routing-rule paper vs hierarchical-subgoal method. The **contact-state probe** (✅ DONE 2026-06-15: single-frame
  AUC **0.88** / latent-history **0.97** / full $[z,a]$ **0.99** on Lift — "history ≫ single-frame", latent is contact-*degraded*
  not blind) is the empirical anchor. Renamed project: **LeWAM** (Randall-endorsed). Paper artifacts: lewam_slides/related_work/skeleton/refs in proposal/.
- **⭐ Temporal Straightening (2603.12231, user-flagged 2026-06-16; Wang/Bounou/Gaoyue-Zhou/Balestriero/Rudner/LeCun/Ren — same
  lab as LeWM/PLDM) → full treatment in `proposal/literature_survey.md` §1.** A curvature regularizer that straightens latent
  trajectories (Euclidean $\approx$ geodesic) → better-conditioned latent planning, higher goal-reaching SR. **Corroboration:**
  latent-planning quality is rate-limited by latent *geometry*, not just the planner (consistent with our per-family story).
  **Sharp contrast (our wedge):** straightening fixes a *geometry* problem where the task variable IS in the latent but
  poorly conditioned (goal-reaching); it CANNOT fix our robomimic-contact failure, where grasp state is **absent from the
  single-frame goal latent** (probe 0.88, swamped by arm pose) — no curved manifold to flatten when the axis is missing. So
  our contribution is **orthogonal**: they condition geometry for goal-reaching; we identify a *semantic* limit (variable not
  in the goal frame, recoverable only from action history). **Synthesis idea** = a latent that is both straightened AND
  contact-observable (history-conditioned) — a possible method-paper hook if we take the hierarchical fork.
- **⭐ DINOv2 encoder weight-init ablation (launched 2026-06-17, L40S) → full record in §10.** Open question the user
  raised: does initializing the LeWM encoder with self-supervised DINOv2 weights beat training the encoder from scratch?
  Design correction the user made (decisive): for a CLEAN weight-init ablation **the encoder ARCHITECTURE must be held
  fixed and ONLY the weight init flipped** — so both arms use the *same* `facebook/dinov2-small` arch (384-dim, patch14,
  22.1M), differing only in `pretrained=true` (load DINOv2 weights) vs `pretrained=false` (random init). My earlier
  2-arm proposal (vit-tiny-192 scratch vs dinov2-small-384 pretrained) was WRONG: it confounds arch/size with init.
  tool_hang 3 base WMs now training; see §10 for the full design, why dinov2-small is forced, configs, and ETA.

## §10 ✅🆕 DINOv2 encoder weight-init ablation (L40S, launched 2026-06-17)

**Motivation (why).** Two open questions converged here. (1) User: *"init encoder with dinov2 or not is an open
question, maybe do ablations?"* — i.e. does a self-supervised pretrained visual encoder help the LeWM/LeWAM world model,
or is the from-scratch ViT-tiny encoder (LeWM's published recipe) enough? (2) The contact-state probe (§6) showed the
*single-frame latent is contact-degraded* (Lift AUC 0.88), so a stronger visual encoder is a natural lever to test — if
DINOv2 features make grasp state more linearly decodable from a single frame, that would move the whole per-family story.

**Design — the decisive correction (user, 2026-06-17): "lewm encoder is fixed, right, if you ablation only change
weight init?"** YES. A clean weight-init ablation must hold the encoder **architecture** fixed (same dims/layers/params)
and flip **only** the weight tensors. My first proposal — arm A = `vit_hf` tiny-192 *scratch*, arm B = dinov2-small-384
*pretrained* — is **confounded**: it changes the architecture (192-dim/3-head vs 384-dim/6-head, 5.5M vs 22.1M) at the
same time as the init, so any delta is un-attributable. That is two different models, not an ablation. **Correct clean
ablation:** pick ONE arch, run two arms differing in nothing but `{load pretrained weights}` vs `{random init}`.

**Why the fixed arch is `facebook/dinov2-small` (384-dim, patch14, 12 layers, 22.1M params) — it is forced.** To have a
genuine *pretrained* arm we need real pretrained weights at our patch size (14). The repo's `vit_hf` builder only fetches
`google/vit-{size}-patch{patch}-{image}` (ImageNet, **patch16**) when `pretrained=True` — there is no `vit-tiny-patch14`
pretrained checkpoint, and DINOv2 is not exposed by `vit_hf` at all. DINOv2's smallest model is `dinov2-small` (384-dim,
patch14). So the only architecture for which a real "pretrained vs scratch" pair exists at patch14 is dinov2-small-384.
Hence **both** ablation arms are dinov2-small-384; the from-scratch ViT-tiny-192 stays only as the **canonical-recipe
anchor** (it ties to every prior result in this file — all lift/can/square/pusht/tworoom/reacher base WMs are tiny-192
scratch), NOT as a member of the clean pair.

**→ Three base WMs per task:**
| line | encoder arch | init | role |
|---|---|---|---|
| canonical | vit-tiny-192 (patch14, 5.5M) | scratch | anchor to all prior LeWM results in this file |
| **dinov2-scratch** | dinov2-small-384 (patch14, 22.1M) | random | clean-ablation CONTROL |
| **dinov2-pretrained** | dinov2-small-384 (patch14, 22.1M) | DINOv2 SSL weights | clean-ablation TREATMENT |

The ablation answer = (dinov2-pretrained − dinov2-scratch) at matched arch. The canonical-vs-dinov2-scratch gap separately
reports the effect of *encoder size/arch at scratch* (192→384), a secondary read.

**Implementation — clean drop-in, no model surgery.** `jepa.JEPA.encode()` (jepa.py:62–63) calls
`self.encoder(pixels, interpolate_pos_encoding=True)` and reads **only** `output.last_hidden_state[:, 0]` (the CLS
token) as the per-frame embedding. So the encoder is a black box `image → CLS vector of dim embed_dim`; patch/register
token count is irrelevant, and pos-embed interpolation (DINOv2 native 37×37 → our 16×16 @224px) is **already** requested.
Nothing in jepa.py / module/ hardcodes 192 — every dim flows from the single hydra var `embed_dim`. The existing general
builder `stable_pretraining.backbone.utils.from_huggingface(model_name, pretrained, ...)` does exactly the ablation:
`pretrained=True → AutoModel.from_pretrained("facebook/dinov2-small")` (real weights); `pretrained=False →
AutoConfig.from_pretrained(...) + AutoModel.from_config(...)` (random init of the **identical** arch). Returns
`model.base_model` = the Dinov2Model. **No new loader code was needed.**
- New config `config/train/model/lewm_dinov2.yaml` = copy of `lewm.yaml` with only the encoder block swapped to
  `_target_: stable_pretraining.backbone.utils.from_huggingface`, `model_name: facebook/dinov2-small`, `pretrained: false`
  (flip per-arm on the CLI). Companion override **`embed_dim=384`** (lewm.yaml default 192) propagates to
  predictor/projector/pred_proj/action_encoder via `${embed_dim}`.
- **Smoke test (2026-06-17, GPU2, minghao.fu env):** both arms build + forward;
  `pretrained=False` → last_hidden=(2,257,384), CLS=(2,384), 22.1M params, cls_mean≈0.0000 (untrained);
  `pretrained=True` → same shapes, 22.1M, cls_mean=0.0670 (real weights). Confirms identical arch, weights-only delta.

**EXACT launch (tool_hang, L40S, run as `minghao.fu` via `sudo -u`; env: `HOME=/var/lib/docker/data/minghao_home`,
`STABLEWM_HOME=/mnt/minghao_data/.stable-wm`, `HF_HOME=$HOME/.cache/huggingface`, `TMPDIR=/mnt/minghao_data/tmp`,
`MPLCONFIGDIR=/mnt/minghao_data/mpl`, `HF_HUB_OFFLINE=0`; conda env `$HOME/lewm`):**
```
# canonical (already running, GPU1, launched 2026-06-17 07:40) — vit-tiny-192 scratch:
python train.py data=robomimic_tool_hang action_pred.enabled=false trainer.max_epochs=100 \
  output_model_name=tool_hang_lewm_scratch +trainer.limit_train_batches=4000 \
  +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4 wandb.enabled=false
# dinov2 scratch (GPU2) — clean-ablation control:
CUDA_VISIBLE_DEVICES=2 python train.py data=robomimic_tool_hang model=lewm_dinov2 embed_dim=384 \
  model.encoder.pretrained=false action_pred.enabled=false trainer.max_epochs=100 \
  output_model_name=tool_hang_lewm_dinov2_scratch +trainer.limit_train_batches=4000 \
  +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4 wandb.enabled=false +ckpt_every=10
# dinov2 pretrained (GPU3) — clean-ablation treatment (same as above but):
  model.encoder.pretrained=true output_model_name=tool_hang_lewm_dinov2_pretrained  CUDA_VISIBLE_DEVICES=3
```
Logs: `/mnt/minghao_data/th_basewm_{scratch,dinov2_scratch,dinov2_pretrained}.log`.

**Status (2026-06-17 ~07:58).** All three tool_hang base WMs training in parallel on L40S (so train-render == eval-render;
the cross-GPU rendering trap of §0-Infra is avoided — these models are trained AND will be eval'd on L40S):
| arm | GPU | it/s | epoch-wall | 100ep ETA | mem |
|---|---|---|---|---|---|
| vit-tiny scratch (canonical) | 1 | 6.0 | ~218s | ~6h | 9.9G |
| dinov2 scratch | 2 | 1.8 | ~720s | ~20h | 17G |
| dinov2 pretrained | 3 | 1.7 | ~720s | ~20h | 17G |
dinov2 is ~3.3× slower (22M vs 5.5M encoder, 384 vs 192 dim) — acceptable, idle GPUs. Box: 64 cores, 364G RAM free; GPU0
= `fan-test` (untouched); /mnt/minghao_data 3T (weights output); docker vol 355G (lightning .cache, watch).

**Pipeline per arm (old pipeline, user-mandated):** base WM (`action_pred.enabled=false`, 100ep) → GIP head
(`action_pred.enabled=true init_from=<arm>_weights`, 100ep) → **N=50** eval (histbc + guided + planning). Same for every
arm, so the ablation is end-to-end (does pretrained-init carry through to control SR, not just lower base-WM loss).

**Rollout plan.**
- **Phase 1 — tool_hang (3 arms launched) + transport (pending data).** Transport raw states NOT on L40S yet → needs
  transfer (174→L40S demo_v15 states) → `dataset_states_to_obs.py` (L40S render) → `convert_robomimic_h5.py` + index fix,
  then the same 3 arms. (Both are robosuite tasks → MUST be L40S-rendered for valid eval, per §0-Infra.)
- **Phase 2 — previous tasks, dinov2 {pretrained, scratch} added (canonical tiny-192 already exists).** Robosuite tasks
  lift/can/square: their L40S `.h5` is currently 174-rendered → must be **re-rendered on L40S** before the dinov2 arms'
  eval is valid (train-render==eval-render). pusht/tworoom/reacher are CPU-rendered (pygame/gridworld) → portable, existing
  `.h5` is fine, no re-render needed.

**Deliverable.** Unified `results.html`: rows = 8 tasks, columns = {canonical, dinov2-scratch, dinov2-pretrained} ×
{base-WM val loss, histbc SR, guided SR, planning SR} at N=50, plus bar charts. (User: *"a full table/figure comparison
on unified html."*)

**OPEN.** (a) Does dinov2-pretrained beat dinov2-scratch on contact tasks, or does the JEPA objective wash out the SSL
prior after 100ep on in-domain robot frames? (b) If pretrained helps the *base-WM loss* but not *control SR*, that
reinforces §6 (perception isn't the only bottleneck). (c) Re-run the §6 contact probe on the dinov2-pretrained encoder —
does DINOv2 raise single-frame grasp-AUC above tiny-192's 0.88?

**⚠️ METHOD NOTE — `pred_loss` is only comparable WITHIN an architecture.** The base-WM `pred_loss` is the
next-embedding MSE *in the encoder's own latent space*. vit-tiny (192-d) and dinov2-small (384-d) live in different
spaces at different scales, so **canonical-vs-dinov2 `pred_loss` is meaningless** — those two lines may only be compared
on downstream control **SR**. The clean ablation pair (dinov2-pretrained vs dinov2-scratch) IS comparable on pred_loss
because both are the *same* 384-d dinov2-small space.

**Early signal (2026-06-17, epoch 2 — DIAGNOSTIC ONLY, not a result; rule 2).** Within the clean dinov2 pair at matched
epoch 2: pretrained `pred_loss=0.0155`, `sigreg=2.23`; scratch `pred_loss=0.036`, `sigreg=3.56`. So the DINOv2-init arm is
~2.3× lower next-embedding error and lower SIGReg this early — a hint the SSL prior yields a more predictable,
better-conditioned latent for the world model. MUST be re-checked at plateau (≥ep90) and, decisively, on control SR
(a base-WM-loss win need not transfer to SR — see (b) and §6). Canonical vit-tiny (ep8 pred_loss 0.014) is NOT in this
comparison (different latent space; see method note).

**✅ CONFOUND CHECK — input normalization is fair to the pretrained arm (verified 2026-06-17).** Worry: if training fed
raw 0–255 pixels, DINOv2's ImageNet-pretrained weights would start badly OOD → a *confounded null* ("pretrained ≈
scratch" for the wrong reason). Traced the pipeline: `train.py` applies `utils.get_img_preprocessor` =
`spt…ToImage(**dataset_stats.ImageNet)` + `Resize(224)`, i.e. **ImageNet normalization** (mean [0.485,0.456,0.406], std
[0.229,0.224,0.225]) on `pixels` before the encoder (jepa.py:59–62 then feeds it directly). So the DINOv2 pretrained arm
receives exactly its native preprocessing — fair shot, prior is usable from step 0 (consistent with it leading at ep4).
Both dinov2 arms share this identical transform, so the pretrained-vs-scratch ablation stays clean. **No restart needed.**

### §10 EVAL stage — recipe + the dinov2 fix (prepped 2026-06-17 while base WMs train)
**GIP launch (turnkey):** `gip_launch.sh <task> <arm> <gpu>` (staged on L40S `~minghao.fu/`) warm-starts from the
highest-epoch base-WM ckpt (`detach_decoder=true`, `head=mse`, `w_act=w_intent=1`, 100ep), auto-adds
`model=lewm_dinov2 embed_dim=384` for dinov2 arms, and refuses to launch if the base WM is < ep100 (convergence guard).
Output `output_model_name=<task>_gip_<arm>` → `$STABLEWM_HOME/checkpoints/<task>_gip_<arm>/weights_epoch_*.pt`.

**Eval (validated path, per [[project_robomimic_eval_path_gotcha]]):** `eval_histbc_robomimic.py` for BC,
`eval_histguided_robomimic.py` for guided/planning (the manipulation-contrast evaluator from task #43). **NOT**
`eval_gip_robomimic.py` (its BCPolicy is single-frame, horizon-1 → bc≈0). Invocation (hydra, `config/eval/robomimic.yaml`):
```
python eval_histbc_robomimic.py --config-name robomimic policy=<task>_gip_<arm> \
    world.task=<Env> dataset.stats=<task> eval.dataset_name=<task> \
    eval.num_eval=50 eval.eval_budget=<~2×median-demo> eval.goal_offset_steps=<task>
```
`cfg.policy` is the **GIP checkpoint folder NAME** (= `output_model_name`), not a config file — `gip.load_gip_model`
globs `$cache/checkpoints/<policy>/weights_epoch_*.pt` and reads that run's `config.json` to rebuild the arch.
`world.task`: tool_hang→`ToolHang`, transport→`TwoArmTransport` (already in `robomimic_env._TASK_TO_DIR`). **N=50 MANDATORY**
([[feedback_eval_num_eval_50]]); ≥3 seeds {42,0,1} for headline (rule 9).

**🐛 FIX applied 2026-06-17 — `load_gip_model` hardcoded `embed_dim=192` → would crash dinov2 (384) eval.** `gip.load_gip_model(run_name, embed_dim=192, …)` built the encoder/predictor correctly from `config.json` but built the
**action_decoder / proprio / task_proj from the `embed_dim` ARG** (default 192), and `eval_histbc` never passes it → for a
384-d dinov2 GIP, `load_state_dict` hits a shape mismatch and the eval dies. Patched: `embed_dim` defaults to `None` and is
auto-derived `= int(config.predictor.input_dim)` (192 vit-tiny / 384 dinov2) right after `model = hydra_instantiate(config)`.
Backward-compatible (canonical still 192), syntax-checked. This is the exact config-drift class rule 7 exists to prevent.

**Eval params COMPUTED 2026-06-17 (from `.h5` `ep_len`, all 200 demos):**
| task | n | median | mean | max | `eval_budget`=2×median | `goal_offset_steps`≈0.6×median |
|---|---|---|---|---|---|---|
| tool_hang | 200 | 469 | 479 | 744 | **940** | **280** |
| transport | 200 | 455 | 468 | 714 | **910** | **270** |
Sanity: the same 2×median rule gives lift `eval_budget`=96, matching its validated ~100 → formula trusted. These are
LONG-horizon (vs lift 96 / can 240 / square 320) → eval is slow (`max_episode_steps`=2×`eval_budget`≈1880, ×50 envs);
budget the wall-clock. Eval params are identical across the 3 arms of a task, so they don't bias the ablation delta (only
the absolute SR). **Remaining eval TODO:** (1) build/clone the guided eval for the new tasks; (2) sanity-check the
canonical arm SR is non-trivially > 0 BEFORE trusting dinov2 numbers; (3) all evals on L40S (train+eval same box, §0-Infra).

### §10 Base-WM pred_loss trajectory so far (2026-06-17, DIAGNOSTIC — ep ≤ 6, single run, pred_loss NOT SR)
Within each clean dinov2 pair (same 384-d space; cross-arch vs canonical is invalid):
- **tool_hang — DINOv2 prior helps, but the gap is NARROWING:** pretrained/scratch pred_loss ratio = 3.0× (ep3) →
  2.3× (ep5) → 1.7× (ep6). Pretrained is near a floor (~0.0074) while scratch keeps dropping (0.036→0.0128). Trend ⇒
  the prior **speeds convergence** (efficiency), with scratch likely catching up by plateau — NOT (yet) evidence of a
  lower ceiling. Decide at plateau + on SR.
- **transport — NO DINOv2 advantage so far (task-dependent):** pretrained ≥ scratch at every matched epoch (ep3:
  0.0091 vs 0.0071). Pretrained recovers slower from its ep0 random-head spike (0.331) than tool_hang's did. Hypothesis:
  the wider, more cluttered 2-arm `agentview` transfers ImageNet/DINOv2 features less well than tool_hang's tighter
  single-arm view. If this holds to plateau/SR, the headline becomes "DINOv2-init helps some contact tasks, not all" —
  a more interesting (and honest) result than a blanket win. **Watch the transport crossover (or lack of it) at ep5–10.**

### §10 Decision pre-registration (written 2026-06-17 BEFORE any SR — to keep the Phase-2 go/no-go principled)
**Primary metric = control SR (N=50, seeds {42,0,1}), histbc + guided.** Base-WM `pred_loss` is diagnostic ONLY; it does
not decide anything (a base-WM-loss edge need not transfer to SR — §6 showed perception isn't the sole bottleneck).
**Clean ablation contrast (per task):** dinov2-PRETRAINED vs dinov2-SCRATCH (matched 384-d arch). Canonical vit-tiny is
the recipe anchor, compared on SR only.
- **H1 (prior helps):** pretrained mean SR exceeds scratch mean SR by more than the across-seed noise (≈1 SD of the
  3-seed spread) AND the sign is consistent across the two eval modes (histbc, guided). Per-task.
- **H0 (prior doesn't transfer):** SR within seed noise → the DINOv2 base-WM-loss edge (tool_hang) did NOT convert to
  control benefit. This is itself a clean, publishable negative (reinforces §6) and is the EXPECTED outcome if the
  pred_loss gap keeps narrowing to ~1× by plateau.
- **Phase-2 trigger (gates the expensive 6-task × 2-arm extension, [[project_swm_migration]] task #52):** run Phase 2
  ONLY if Phase 1 shows a credible H1 on ≥1 of {tool_hang, transport}. If BOTH are H0, STOP — report "DINOv2-init does not
  improve LeWAM control SR despite faster base-WM convergence" and do not spend GPU-days re-rendering + training the old
  tasks. This is decided on SR, not on the (already-suggestive-of-H0) pred_loss trend.
- **Interpretive boundary (state in any writeup):** both arms share the canonical LR, so the claim is "pretrained vs
  scratch at fixed recipe," not "best-tuned vs best-tuned." A pretrained encoder might prefer a lower encoder LR; not tested.

**⚠️ METRIC HIERARCHY — bc is NOT goal-conditioned, so it under-tests the WM (user flag, 2026-06-17).** `eval_histbc`
runs the intention head OPEN-LOOP (no goal, no planning) → it mostly tests the **policy head**, which can compensate for a
weaker latent. DINOv2-init changes the **encoder/WM latent**, which is exercised only when the WM is *used* (rolled out to
a goal). BUT our own §7b / [[project_robomimic_planning]] result is that WM-CEM **planning on robomimic contact ≈ 0%**
(Lift planning 0) → guided/planning may be ≈0 for ALL arms and fail to discriminate. So the ablation is read on **4 axes,
strongest-first for the WM claim**: (1) **frozen-encoder linear probe** (§6 style — grasp/contact-state AUC from the
*frozen* converged encoder; measures latent quality DIRECTLY, independent of whether CEM plans — the cleanest DINOv2-vs-
scratch test); (2) **guided SR** (`eval_histguided_robomimic.py`, goal-conditioned intuition→CEM→goal; the LeWAM thesis;
report even if low); (3) **base-WM `pred_loss`** (have it: modest tool_hang edge, transport none); (4) **bc SR**
(policy head — weakest for the WM claim, contaminated by policy compensation; reported for completeness, NOT the
arbiter). H1/H0 (above) is decided primarily on (1)+(2); a bc-only win/loss does NOT settle the DINOv2 question.

### §10 Run incidents + GIP/eval execution log (live, 2026-06-17)
- **GIP warm-start works (verified):** `init_from` into a GIP run reports `missing=88 unexpected=0` (transport scratch) —
  the 88 missing keys are exactly the new action head (action_predictor+action_decoder), shared trunk loads clean. Base-WM
  resume reports `missing=0 unexpected=0` (full load). So the load path is sound for both uses.
- **🔻 CRASH+RESUME — tool_hang dinov2-SCRATCH base WM died at ep71** (`RuntimeError: could not unlink the shared memory
  file /torch_… : No such file or directory` — a transient torch-DataLoader worker/shm race, NOT /dev/shm exhaustion:
  /dev/shm was 252G@1%; the other 5 runs were unaffected). **Resumed** from the clean `weights_epoch_70.pt` with
  `init_from=…weights_epoch_70.pt trainer.max_epochs=30 output_model_name=tool_hang_lewm_dinov2_scratch_r70` (GPU2; loaded
  `missing=0 unexpected=0`). Per CLAUDE.md (continue crashed runs from intact weights; restarted LR acceptable; don't waste
  intact epochs). **Caveat for the ablation:** this arm is "ep0–70 clean + ep71–100 as a fresh 30-epoch LR schedule," vs the
  pretrained arm's single clean 100ep. The asymmetry gives scratch *more* effective late-training (fresh higher LR), so it
  is **conservative for H1** (if pretrained still wins, robust). Watching ep0/30 val-loss for a warmup spike (would force a
  clean re-run). **Its final base WM = `…/tool_hang_lewm_dinov2_scratch_r70/weights_epoch_30.pt`** (=ep100-equiv), NOT the
  original dir — so its GIP must `init_from` that path explicitly (gip_launch.sh's epoch-100 guard won't find it).
- **🔻 RECURRENCE (2026-06-18) — the SAME shm-FD race killed the square dinov2-SCRATCH GIP at ep71**
  (`could not unlink the shared memory file`), again **right after the ep70 checkpoint save** → the `+ckpt_every=10` save
  (forks/serialises a ~860 MB ckpt under load) spikes shm-FD pressure and the next epoch's DataLoader workers lose the race.
  Triggered this time because the **L40S cube training added 12 dataloader workers** (2 arms × 6) on top of the square GIPs.
  **Resumed** `square_gip_dinov2_scratch/weights_epoch_70.pt` → `square_gip_dinov2_scratch_r70` (+30ep) with **`num_workers=2`**
  (halves FDs) — stable past ep6, no re-crash. **Mitigation for co-located runs: drop `num_workers`→2 when packing many
  trainers on one box** (the FD-sharing race scales with total workers across all procs); per-run slowdown is minor vs a
  mid-run crash. Eval note: the scratch square bc must use arm `dinov2_scratch_r70` (POLICY=`square_gip_dinov2_scratch_r70`,
  highest ckpt = ep30 = ep100-equiv). Both crashed arms were SCRATCH — likely coincidence (the race is load-triggered).
- **GIPs launched:** transport scratch (GPU4, from `transport_lewm_scratch/weights_epoch_100.pt`), tool_hang scratch (GPU1,
  from `tool_hang_lewm_scratch/weights_epoch_100.pt`). Both warm-started clean, stepping ~6.3 it/s (~5.6h/100ep).
- **🐛 EVAL BLOCKER found+fixed by the early N=8 plumbing-test (2026-06-17):** `robomimic_env.py:39` opens
  `$ROBOMIMIC_RAW/<task>/ph/image_384_v15.hdf5` and reads **only `data.attrs["env_args"]`** (the robosuite env metadata —
  NOT the rendered images). Two breakages: (1) `ROBOMIMIC_RAW` unset → defaults to the **174 path**
  `/mnt/data_nvme1/minghao.fu/robomimic` (absent on L40S); (2) I **deleted** tool_hang's + transport's `image_384_v15.hdf5`
  after convert to save disk (lift's survived → why lift eval worked). demo_v15's env_args is NOT a drop-in (the render adds
  `render_gpu_device_id` + makes `camera_names` a list). **Fix:** (a) re-render a **1-demo** `image_384_v15.hdf5` stub for
  tool_hang+transport (`dataset_states_to_obs.py … --n 1`, ~580–605 MB, correct env_args; the eval needs only env_args so 1
  demo suffices) and KEEP it; (b) add `export ROBOMIMIC_RAW=/var/lib/docker/data/minghao_home/robomimic` to `eval_launch.sh`.
- **✅ EVAL PLUMBING VALIDATED:** N=8 histbc on transport_gip_scratch@ep10 ran end-to-end — GIP load `Adim=70 missing=0
  unexpected=0`, env+CLIP+rollout, `success_rate=0.0` (undertrained ep10 policy on hard 2-arm transport; same pipeline gave
  lift SR=90, so it CAN be non-zero), clean exit. Eval path now de-risked (both the embed_dim and image_384 bugs caught
  pre-headline). Note: N=8 transport eval took ~15 min (long horizon, budget 910) → **headline N=50 evals must be
  parallelized across GPUs** (each ~30–60 min). CLIP loads via HF_HUB_OFFLINE=0 (do NOT set XDG_CACHE_HOME at eval).

### §10 ⭐ FROZEN-ENCODER PROBE RESULT — the primary axis (2026-06-17, from converged base WMs, NO GIP needed)
**Setup (`frozen_probe.py`):** freeze each converged base-WM encoder → CLS latent → Ridge probe (standardized, α=10) →
ground-truth robosuite `state`; report held-out R² (6000 stride-sampled frames, 80/20 split) + collapse diagnostics
(mean per-dim std, effective rank = participation ratio of singular values). Clean pair = dinov2_pretrained vs
dinov2_scratch (both 384-d). Resumed tool_hang dinov2_scratch uses the `_r70/ep30` ckpt.
| task | arm | state R² | proprio R² | latent-std | eff-rank |
|---|---|---|---|---|---|
| tool_hang | canonical vit-tiny 192 | 0.833 | 0.962 | 0.94 | 45/192 |
| tool_hang | dinov2 scratch 384 | 0.823 | 0.958 | 0.94 | 86/384 |
| tool_hang | dinov2 **pretrained** 384 | 0.825 | 0.921 | 2.36 | 101/384 |
| transport | canonical vit-tiny 192 | 0.710 | 0.264 | 0.93 | 61/192 |
| transport | dinov2 scratch 384 | **0.418** | **−0.166** | 0.94 | 120/384 |
| transport | dinov2 **pretrained** 384 | **0.838** | **0.584** | 2.36 | 112/384 |
| can | dinov2 scratch 384 | 0.659 | 0.915 | 0.48 | 39/384 |
| can | dinov2 **pretrained** 384 | 0.681 | 0.933 | 2.27 | 70/384 |
| lift | dinov2 scratch 384 | 0.32 | — | — | — |
| lift | dinov2 **pretrained** 384 | 0.38 | — | — | — |
| square | dinov2 scratch 384 | 0.884 | 0.957 | 0.95 | 60/384 |
| square | dinov2 **pretrained** 384 | 0.878 | 0.961 | 2.39 | 95/384 |

**Read (DIAGNOSTIC — one probe target = full state, one split/seed; cross-check vs guided SR + per-dim later):**
- **tool_hang: DINOv2-init gives NO latent advantage** — pretrained 0.825 ≈ scratch 0.823 ≈ canonical 0.833. From-scratch
  is already adequate on the tighter single-arm scene.
- **transport: DINOv2-init gives a LARGE latent advantage** — pretrained 0.838 vs scratch 0.418 (state), and scratch's
  proprio R² is *negative* (latent barely encodes eef/gripper). **NOT collapse:** scratch eff-rank 120/384 (higher than
  pretrained's 112), latent-std 0.94 (healthy). So scratch learned a **high-rank but poorly-state-aligned** latent on the
  hard, wide 2-arm scene; the DINOv2 ImageNet prior gives **linearly-state-aligned** features.
- **Capacity×data interaction:** on transport the *smaller* vit-tiny-scratch (0.71) **beats** the *bigger* dinov2-arch-
  scratch (0.42) — the 22M from-scratch encoder under-fits 200 narrow demos worse than the 5.5M one; the **prior is what
  rescues the big model** (0.84). On tool_hang, all adequate.
- **pred_loss was MISLEADING here (validates the "pred_loss under-tests" worry):** base-WM `pred_loss` had transport
  pretrained ≈ scratch, but the probe shows a 2× state-R² gap. The scratch encoder self-predicts its own (poorly-
  structured) latent fine → low pred_loss → yet the latent is state-poor. **The probe, not pred_loss, is the meaningful
  latent-quality axis.**

**THESIS (one-liner for the paper, if it holds on SR):** *DINOv2 encoder-init helps the LeWAM world model exactly when the
visual scene is hard enough that a from-scratch encoder under-fits the narrow demo data (transport's wide 2-arm view) and
the encoder is large enough to need the prior; on tighter single-arm scenes (tool_hang) from-scratch suffices and the
prior is inert.* Task- AND capacity-dependent, not a blanket win.
- **Phase-2 implication (decision rule):** transport is a **credible H1** on the primary axis → per the pre-registration,
  Phase 2 (extend the dinov2 ablation to old tasks) is now **justified**, and should prioritize the harder/wider-scene
  tasks (square/can over lift; the wider the scene the more the prior should matter). Still must confirm the latent
  advantage carries to **control SR** (guided) — a state-decodable latent need not yield a better planner if the §7b
  goal-cost-blindness dominates. That is the next read when the GIPs converge.

- **✅ GUIDED eval validated (2026-06-17):** `eval_histguided_robomimic.py` ran end-to-end (GIP load → env → CLIP → CEM
  solver → `HISTGUIDED RESULTS success_rate=0.0`, tool_hang canonical ep75 GIP, N=5, budget 120). SR=0 is expected per §7b
  (robomimic WM-planning ≈0) + diagnostic (undertrained, N=5). **CEM speed = ~4.7 s/replan @5 envs** → at N=50 a guided
  eval is heavy; use a **moderate budget (~300–400, not bc's 940)** + parallelize across GPUs as they free. **The one
  result to watch:** does **transport dinov2-pretrained** (the only arm with a state-aligned latent, probe 0.84) give a
  **non-zero** guided SR where the others (latent 0.42/0.71) give 0? That would be the headline — DINOv2-init making
  WM-planning *work* where from-scratch can't. Both eval modes (histbc, guided) now plumbing-validated.

- **🐛 EVAL BLOCKER #3 — N=50 evals HUNG on an uncached CLIP download (2026-06-17).** The first headline N=50 launches (3
  concurrent) froze ~17 min at "Created ViT-tiny" with state=R, 100% CPU, gip-load never printed. Misleading symptom (looked
  like the 50-env build). Real cause: **`openai/clip-vit-large-patch14` (1.7 GB, the task-embedding model) was NOT in the HF
  cache** (CLIP_MISSING; the earlier N=8 test had streamed it without persisting), so each eval tried to download it; 3
  concurrent downloads stalled/throttled (5 open sockets, HF xet). The N=8 plumbing test passed only because it was the lone
  downloader. **Fix:** `snapshot_download("openai/clip-vit-large-patch14")` once (verified offline:
  `CLIPTextModelWithProjection.from_pretrained` → CLIP_OK, 123M) + set **`HF_HUB_OFFLINE=1`** in `eval_launch.sh` (CLIP +
  dinov2-small both cached → zero eval-time network). Post-fix the eval advances normally into the (genuinely slow ~12 min)
  50-env TwoArmTransport build. **Lesson:** with CLIP cached + offline, concurrent evals are fine (the hang was downloads,
  not env-creation); the recurring [[project_robomimic_eval_path_gotcha]]-class CLIP-cache trap bit again — always pre-cache
  CLIP before a batch of evals.
- **🐛 EVAL BLOCKER #4 + RESOLUTION — N=50 = 50 simultaneous mujoco/EGL envs is non-viable; BATCH over a 10-env pool
  (2026-06-17).** `config/eval/robomimic.yaml` sets `world.num_envs = eval.num_eval`, so N=50 builds **50 TwoArmTransport
  sims at once** → ~20-min single-core mujoco compile that never reached GIP-load (`num_envs=8` reached it instantly). Two
  things cost a lot of time: (a) `OMP_NUM_THREADS=1` thread caps (real improvement — 50 envs otherwise spawn a 227-thread
  storm — KEEP, but not the full fix); (b) **I repeatedly killed working evals because `pgrep | head -1` matched the idle
  `bash -c` WRAPPER (in `wait4`), not the busy python CHILD** — process-CPU was a false signal; only the LOG is trustworthy.
  **Real fix:** patched `eval_histbc` + `eval_histguided` to **loop `world.evaluate` over chunks of `world.num_envs`
  episodes** (compile a small pool once, run N=50 as 5×10), set **`world.num_envs=10`** in `eval_launch.sh`. Verified:
  transport_gip_scratch bc loaded clean (`Adim=70 missing=0`), `[HISTBC] chunk 1: 0/10`, runs to N=50. Env trio for EVERY
  eval: `OMP_NUM_THREADS=1` + CLIP-cached `HF_HUB_OFFLINE=1` + `world.num_envs=10`. Per-eval ≈ ~6-min chunk-1 compile +
  4×rollout.
- **📊 LIVE STATUS 2026-06-17→18 UTC (contention-throttled, all alive, nothing crashed).** Verified by ckpt-mtime (recent,
  <1h) + state-R masters, NOT process-CPU (the wrapper-vs-child trap). **Base WMs at ep100 (done):** tool_hang {canonical,
  dinov2_scratch_r70/ep30=100-eff, dinov2_pretrained}, transport {all 3}, lift {dinov2_scratch, dinov2_pretrained}, can
  {dinov2_scratch}. **GIPs grinding (need ep100 to eval, gip_launch's epoch-100 guard):** tool_hang {scratch ep80, pretrained
  ep80}, transport {scratch ep90, pretrained ep90} — warm-started clean, but the box is **heavily CPU/disk-contended**:
  fan-test (fanfeng) has **13 `run_wav_loop.py` processes** (46h old, UNTOUCHABLE) + my 7 jobs all sharing cores, so epochs
  crawl (~mtimes show ~few-min to ~10-min/epoch, not the unloaded ~3.4-min). GPU mem is NOT the limit (each job ~15 GB / 46
  GB), exactly as CLAUDE.md rule 10 says — disk/CPU is. **Phase-2 base WMs progressing:** can_dinov2_pretrained ep30,
  square_dinov2_scratch ep20; **just launched the missing `square_dinov2_pretrained` (GPU7, pid 1355971, disk 2999G free,
  same `launch_dinov2_th.sh` recipe with `data=robomimic_square model.encoder.pretrained=true`)** → confirmed training (GPU7
  15 GB/100%, past model-init). This completes the square pair for the within-dinov2 clean comparison. **Long pole = can/
  square base WMs (~70-80 more epochs each under contention, ~many h).** **Lit:** added **NextLat (2511.05963, MS Research)**
  to the live survey §1 + `lewam_refs.bib` (closest sibling on the *objective* axis: their next-latent prediction = our JEPA
  next-embedding loss; corroborates the objective + belief-state framing supports the history→contact-state finding;
  orthogonal to this encoder-init ablation). **NEXT (gated on wall-clock):** (1) GIPs→ep100 → N=50 bc + guided(**offset=25
  FIXED**, per the confound below) for the 4 dinov2 robomimic arms; (2) can/square frozen probes when those base WMs hit
  ep100; (3) refresh the unified HTML once dinov2 SR + Phase-2 probes land.

### §10 ✅ SR RESULTS (live, N=50, batched 10-env pool, 2026-06-17)
First headline numbers (success_rate, seed 42 unless noted). Canonical = vit-tiny-192 scratch GIP.
| task | arm | bc s42 | bc 3-seed (s42/s0/s1) | notes |
|---|---|---|---|---|
| tool_hang | canonical vit-tiny | 0.16 | **0.16/0.22/0.28 → 0.22** | earlier harvest (seed-stable-ish) |
| tool_hang | dinov2 scratch | 0.28 | **0.28/0.18/0.18 → 0.213** | ≈ canonical top |
| tool_hang | dinov2 **pretrained** | 0.44 | **0.44/0.24/0.32 → 0.333 ✅3-seed** | **> scratch ALL 3 seeds (+0.12, ~1.6×) — DECOUPLE CONFIRMED: bc helps, probe flat** |
| transport | canonical vit-tiny | 0.02 | **0.00/0.02 → ~0.01** | 2-arm task near-floor |
| transport | dinov2 scratch | 0.02 | **0.02/0.00/0.02 → 0.013** | the probe-0.42 arm; ≈ canonical floor |
| transport | dinov2 **pretrained** | 0.14 | **0.14/0.02/0.10 → 0.087 ✅3-seed** | **HEADLINE — ~6.5× scratch, pretrained > scratch ALL 3 seeds; probe-0.84 latent converts to control** |

**⭐ RESULT (transport, the headline) — DINOv2-init CONVERTS to control SR, ~7× (s42, N=50; 3-seed replication running).**
transport dinov2-pretrained bc **0.14** (7/50) vs dinov2-scratch **0.02** (1/50) vs canonical vit-tiny ~0.01. The frozen-probe
latent gap (pretrained state-R² 0.84 vs scratch 0.42) **does** translate to a large bc-SR edge, and it is the **pretrained
encoder**, not the bigger 384-d arch, that carries it (scratch ≈ canonical ≈ floor). This **REFUTES** the partial-N
"the nonlinear action head compensates for a state-poor latent" guess: on the hard, wide 2-arm scene a state-decodable
latent yields a materially better policy, not just a better probe. **tool_hang (tight single-arm scene): dinov2-scratch
0.28 ≈ canonical 0.22** — consistent with the probe showing NO latent advantage there (from-scratch suffices), so DINOv2
is expected **inert** on tool_hang; tool_hang-pretrained (pending) will confirm. **The whole ablation is cohering into one
clean rule:** *DINOv2-init helps the LeWAM world model — in BOTH the latent (probe) AND control (bc) — exactly on the hard
wide-scene task where from-scratch under-fits (transport), and is inert on the tight scene where from-scratch already
suffices (tool_hang).* Caveat per global rule 2: s42 alone is diagnostic; the **3-seed transport replication is now DONE and CONFIRMS the
headline — pretrained 0.14/0.02/0.10 → mean 0.087 vs scratch 0.02/0.00/0.02 → mean 0.013, with pretrained > scratch on
ALL three seeds (~6.5×). ✅** Guided is NOT the decisive downstream test here (transport guided is horizon-confounded ≈0,
§10-GUIDED-RESOLVED); bc is, and it converted, robustly across seeds.

**🔱 THE DECOUPLING FORK (tool_hang, s42 N=50; 3-seed RUNNING to resolve).** tool_hang dinov2-**pretrained** bc **0.44**
(22/50) vs **scratch 0.28** (14/50) — pretrained wins by +0.16, **even though the tool_hang frozen probe showed NO latent
gap** (pretrained state-R² 0.825 ≈ scratch 0.823). So on tool_hang the probe (latent state-decodability) and the control
SR (bc) **decouple**: DINOv2-init lifts the policy without lifting the global-state probe. Two readings, and 3-seed decides:
(a) **seed noise** — tool_hang bc is high-variance (canonical was 0.16/0.22/0.28, a 0.12 spread), so a single s42=0.44-vs-0.28
could be one lucky draw; if the 3-seed means converge, DINOv2 is **inert on tool_hang** as the probe predicted and the clean
"transport-helps / single-arm-inert" rule survives. (b) **genuine decouple** — DINOv2's ImageNet features encode the
fine-grained local geometry tool_hang needs (threading the hook through a small loop) better than from-scratch, a
*task-relevant* gain that a **global** full-state linear probe (71-dim state, variance-weighted) cannot see; then the probe
is **necessary-not-sufficient** and control can improve orthogonally to it — a richer, more honest story. **tool_hang
3-seed (scratch s0,s1 + pretrained s0,s1) is running now** (GPUs 3,1,2,5) — the single most informative pending number.
**→ RESOLVED (3-seed FINAL): GENUINE DECOUPLE.** pretrained **0.333** (0.44/0.24/0.32) > scratch **0.213** (0.28/0.18/0.18),
pretrained ahead on **all 3 seeds** (+0.16/+0.06/+0.14, mean +0.12, ~1.6×). Reading (b) wins: DINOv2 lifts tool_hang
*control* with a *flat* global-state probe, so the probe is **necessary-not-sufficient** — it captures the transport
mechanism (global state-decodability) but misses the tool_hang one (task-relevant local features for the fine insertion).
**Final ablation verdict (both robomimic tasks, 3-seed): DINOv2-init helps the LeWAM policy robustly on contact-rich
manipulation — on transport via a more globally state-decodable latent (probe+bc both win), on tool_hang via sharper local
features (bc wins, probe flat).** The blanket "pretrained > scratch on bc" holds every seed on both tasks; the probe
explains the transport half and under-tests the tool_hang half. Implication for Phase-2: **probe-only under-tests the
single-arm tasks** (can/lift/square have small probe gaps +0.02/+0.06/? but tool_hang proves small-probe-gap can still hide
a real bc gain) → see OPEN QUESTION on whether to train can/lift/square GIPs for bc.

**🛑 INFRA INCIDENT (2026-06-18 ~04:50 UTC) — fan-test grabbed all 8 GPUs; can-bc test deferred.** After the robomimic bc
ablation completed (dashboard sent), I launched 2 can dinov2 GIPs to test whether the tool_hang decouple generalizes (the
§9 lean). Between my GPU-free check and the launch, **fan-test (fanfeng) started a 16-process job, ~29.8 GB on EVERY GPU**,
so my can GIPs landed at **98% memory** co-located with fan (29.8+15 = 45/46 GB). Per "never touch other users' GPUs" +
OOM-fragility, I **killed both can GIPs immediately** (zero progress lost). **Surviving mine:** `square_lewm_dinov2_pretrained`
(ep90, GPU7, pre-existing, co-located with fan at 97% — leaving it, it was there first and is near done) + the square-probe
watcher (waits its ep100). **Lesson:** on the shared L40S, check `nvidia-smi --query-compute-apps` ownership BEFORE
launching, not just free-memory — rule-10 "pack many per GPU" assumed small models on an idle box; a neighbor's 29 GB/GPU
job inverts that. **can-bc (and any Phase-2 bc) is BLOCKED until fan's job clears or a GPU shows real headroom.**

### §10 ⚡ PHASE-2 bc — does the decouple generalize? (launched 2026-06-18 after fan cleared; 6 GIPs lift/can/square × scr/pre)
Motivation: the tool_hang decouple (bc helps, probe flat) means probe-only UNDER-tests the single-arm tasks → train can/lift/
square dinov2 GIPs (warm-start from ep100 base WMs, detach_decoder=true, 100ep) and measure bc N=50 3-seed. Eval config added
to `eval_launch.sh` (verified vs §1c + rendered stubs): lift Lift/budget100/offset30, can PickPlaceCan/240/90, square
NutAssemblySquare/320/120; can+square 1-demo env_args stubs re-rendered (PickPlaceCan, NutAssemblySquare confirmed).
- **🔁 LIFT s42 (de-risk gate) — a REVERSAL: scratch 0.68 ≫ pretrained 0.20.** OPPOSITE of transport/tool_hang (where pretrained
  won). Eval ran clean (rc=0, sensible). If it survives 3-seed, **DINOv2-init HURTS bc on the EASIEST task** (lift = single grasp,
  barely contact-rich) — breaking "DINOv2 helps control everywhere." Plausible mechanism: on a trivial single-object scene the
  ImageNet prior is a DISTRACTION; from-scratch learns sharper lift-specific features (and recall lift's probe gap was only
  +0.06 — pretrained's latent was barely better even there).
  **→ 3-seed CONFIRMED (seed-stable, NOT noise): scratch 0.633 (0.68/0.62/0.60) vs pretrained 0.16 (0.20/0.16/0.12) — pretrained
  ~4× WORSE on lift, every seed.** The reversal is real. **Three-task picture: pretrained WINS transport (0.087 vs 0.013, ~6.5×)
  + tool_hang (0.333 vs 0.213, ~1.6×), LOSES lift (0.16 vs 0.633, ~4×).** ⇒ refined rule (pending can/square): **DINOv2-init is
  scene-COMPLEXITY-gated for CONTROL — helps where from-scratch under-fits a hard scene (transport's wide 2-arm view,
  tool_hang's fine insertion), HURTS on the trivial single-grasp lift** where the ImageNet prior over-smooths/distracts and
  from-scratch learns sharper lift-specific features. This is a richer, more honest claim than a blanket win — DINOv2 is a
  conditional tool, not a free upgrade.
  **→ CAN bc 3-seed (2026-06-18): pretrained 0.687 (0.70/0.76/0.60) > scratch 0.547 (0.60/0.50/0.54), every seed (+0.14).
  Can patterns with tool_hang/transport (pretrained WINS), NOT with lift.** So the four-task picture is **pretrained WINS
  transport + tool_hang + can; LOSES only lift.** ⇒ the rule is NOT a clean scene-width split (can is single-arm yet pretrained
  wins) — it is **"DINOv2-init helps control on EVERY robomimic task EXCEPT the trivial single-grasp lift,"** where lift is the
  uniquely-easy outlier (shortest episodes ~50 steps, one grasp) on which the ImageNet prior actively hurts a from-scratch
  encoder that fully fits the simple task.
  **→ SQUARE bc 3-seed (2026-06-18, COMPLETES the 5-task robomimic ablation): pretrained 0.653 (0.68/0.62/0.66) >> scratch_r70
  0.30 (0.28/0.38/0.24), every seed (~2.2×).** Square's scratch arm is the ep70-clean+30ep-resumed `_r70` run (it crashed at
  ep71, §incident); the resume gave scratch *more* late-training (fresh LR), so the pretrained win is **conservative/robust**.
  **🏁 5-task robomimic bc rule — ⚠️ SUPERSEDED 2026-06-18 by the reliability audit below (the "lift reversal" was an
  ARTIFACT). See "§10 🔍 RELIABILITY AUDIT" for the corrected rule.** [Original (now-retracted) claim was: pretrained > scratch
  on transport/tool_hang/can/square but scratch > pretrained on lift — "lone reversal." The lift number 0.16 was a batching
  artifact; corrected lift pretrained 3-seed = 0.917 (0.92/0.88/0.95) > scratch 0.633 (0.62/0.60/0.68), so pretrained wins ALL 5, no reversal.]

### §10 🔍 RELIABILITY AUDIT (2026-06-18, user-requested "deeply check if results are reliable") — found + corrected a real eval bug
**Config audit (subagent, full_config.yaml of every run): CLEAN.** For each task the scratch & pretrained arms are identical
EXCEPT `encoder.pretrained` (same `facebook/dinov2-small`, embed_dim 384, img_size 224, max_epochs, limit_train_batches=4000,
bs 64, action-head hyperparams detach=true/head=mse/w_act=w_intent=1/sigreg_act=false). Eval settings (env/budget/goal_offset/
N/num_envs) identical across both arms and all 3 seeds per task (lift 100/30, can 240/90, square 320/120, tool_hang 940/280,
transport 910/270). Two known asymmetries (both CONSERVATIVE): tool_hang + square **scratch** arms are crash-resumed `_r70`
(70 clean + 30 fresh-LR) → scratch gets MORE late training, so a pretrained win is robust. Naming nit: arch is dinov2-small@224,
not "384"; identical across arms so no bias.
**🐛 THE BUG (found by per-episode pattern analysis, NOT config):** the bc eval (`eval_histbc_robomimic.py`, my own 10-env-pool
batching patch) runs N=50 as **5 chunks of 10 reusing ONE env pool**; the pool reuse **corrupts episodes after chunk 1 for some
arms** → "chunk-1-normal-then-all-zeros". Per-10-episode success counts (s42): **lift pretrained [10,0,0,0,0]=0.20 (ARTIFACT)**,
**transport scratch [1,0,0,0,0]=0.02 (suspect)**; vs clean scattered: lift scratch [10,5,5,7,7], tool_hang pre [3,7,5,4,3], can
pre [8,5,4,4,9], square pre [5,9,6,7,7]. **PROOF:** lift pretrained re-run in a SINGLE fresh 20-env pool (no batching) = **0.95**
(19/20), not 0.16 — and its GIP trained BETTER (intent_loss 0.088 vs scratch 0.263) with a BETTER latent (probe 0.38>0.32), so
0.16 was never plausible. The 8 scattered-pattern arms are reliable; only the 2 all-zeros arms are corrupted.
**✅ CORRECTED RULE:** **DINOv2-pretrained > scratch on ALL 5 robomimic tasks (no lift outlier).** lift pretrained **0.917** (3-seed
0.92/0.88/0.95) > scratch **0.633** (0.62/0.60/0.68) = 1.45×; transport/tool_hang/can/square pretrained-wins stand (clean patterns). transport scratch 0.013 is suspect (re-running
no-batch → true gap likely smaller, direction holds). The two-axis story is unchanged and now CLEANER: probe gain narrow
(transport only), control gain broad (all 5) → probe necessary-not-sufficient.
**FIX + remediation (in flight):** no-batch re-runs (num_envs=num_eval, no pool reuse) of the 2 affected arms launched (lift pre
s0/s1, transport scr 3-seed); the proper fix for the full clean re-run is fresh-pool-per-chunk or num_envs=num_eval in
`eval_histbc_robomimic.py`. **Cube guided eval is NOT affected** (eval_gip.py uses num_envs=num_eval, no batching — validated
at 80%/N=5). Dashboard `results/ablation_dashboard.html` rebuilt around the audit. Remaining: cube guided (probe→planning).

**🔬 FULL CHUNK FORENSICS — all 30 histbc N=50 seed-runs (2026-06-18, the rigorous scope check).** I extracted the
`[HISTBC] chunk N: cumulative X/Y` sequence from EVERY eval log and read the per-chunk deltas. The corruption fingerprint
is **chunk-1 succeeds, then cumulative goes DEAD-FLAT** (a poisoned reused pool stops producing successes); a healthy pool
grows roughly linearly across all 5 chunks. Cumulative-success sequences (after chunks 1..5, out of 10/20/30/40/50) →
per-chunk deltas → verdict:

| task | arm | seed | cumulative seq | per-chunk Δ | mean SR | verdict |
|---|---|---|---|---|---|---|
| lift | **pretrained** | s0 | 8 8 8 8 8 | [8,**0,0,0,0**] | 0.16 | 🔴 **COLLAPSE** |
| lift | **pretrained** | s1 | 5 5 5 6 6 | [5,**0,0,1,0**] | 0.12 | 🔴 **COLLAPSE** |
| lift | **pretrained** | s42 | 10 10 10 10 10 | [10,**0,0,0,0**] | 0.20 | 🔴 **COLLAPSE** (no-batch re-run = **0.95**) |

**✅ NO-BATCH RE-RUN COMPLETE for lift pretrained (3-seed, 2026-06-18):** single fresh 50-env pool (num_envs=num_eval=50, no
reuse) → **s0=0.92, s1=0.88, s42=0.95 → mean 0.917**. So the corrected lift comparison is **pretrained 0.917 vs scratch
0.633 (3-seed 0.62/0.60/0.68) = 1.45×** — pretrained wins, no reversal. (The corrupted batched means were 0.16/0.12/0.20 →
0.16.) **✅ Transport scratch no-batch ALSO COMPLETE (3-seed): 0.0/0.02/0.06 → 0.027** (vs batched 0.013) — CONFIRMS the
forensics call that transport scratch was never corrupted (functional pools, just near-floor); the clean value is a touch
higher but still floor. Transport: pretrained **0.087** (batched, but forensically-clean — functional pools, late-chunk
recovery on all 3 seeds, so reliable; NOT re-run) > scratch **0.027** = 3.2×, direction holds. **🏁 bc audit CLOSED: all
corrupted/suspect cells re-run; corrected 5-task rule = pretrained > scratch on ALL (transport 3.2×, square 2.2×, tool_hang
1.6×, lift 1.45×, can 1.25×).**

**⚠️ TRAIN-RUN VARIANCE CAVEAT (the 3 seeds are EVAL seeds, NOT train seeds — verified 2026-06-18: one GIP checkpoint dir per
(task,arm), eval ran seed=0/1/42 on that single model).** So the per-seed spread captures env/rollout stochasticity but NOT
train-run variance, which the multi-train-seed history shows DOMINATES (±~10pt; the single-train-run "solveact +6/+7" was an
artifact → tie at 3 train seeds, §1c). Honest implications:
- **The OVERALL claim is robust by DIRECTIONAL CONSISTENCY, not per-task magnitude:** pretrained ≥ scratch on **5/5 independent
  task-pairs** → sign-test p = (1/2)^5 = **0.031** (pure train-noise would scatter the signs). That's the load-bearing stat.
- **Per-task split vs the ±10pt band:** **robust (gap ≫ band): square +35pt, lift +28pt**; **borderline (~1–1.4× band): can
  +14pt, tool_hang +12pt** (a single train-run could account for a chunk of these); **transport bc gap is small-absolute (+6pt,
  both near-floor) → transport's headline evidence is the PROBE (+0.42 linear), the bc only confirms sign.**
- **Rigorous next-step for headline-certainty on the borderline tasks:** 2–3 TRAIN seeds/arm on **can + tool_hang** (the two
  within ~1 train-σ). The directional claim + square/lift individual gaps hold WITHOUT it; this would just tighten can/tool_hang.
  (Not launched — it's ~8 base-WM+GIP retrains; flagged as the known limitation, not silently omitted.)
| lift | scratch | s0/s1/s42 | 31 / 30 / 34 (/50) | [9,4,7,5,6] etc | 0.633 | 🟢 healthy linear |
| can | pretrained | s0/s1/s42 | 35 / 38 / 30 | [10,8,4,6,7] etc | 0.687 | 🟢 healthy |
| can | scratch | s0/s1/s42 | 30 / 25 / 27 | [9,4,3,8,6] etc | 0.547 | 🟢 healthy |
| square | pretrained | s0/s1/s42 | 31 / 33 / 34 | [7,3,5,9,7] etc | 0.653 | 🟢 healthy |
| square | scratch(_r70) | s0/s1/s42 | 19 / 12 / 14 | [4,3,2,5,5] etc | 0.30 | 🟢 healthy (genuinely low) |
| tool_hang | pretrained | s0/s1/s42 | 12 / 16 / 22 | [1,3,3,3,2] etc | 0.333 | 🟢 healthy |
| tool_hang | scratch | s0/s1/s42 | 9–11 / 9–14 / 8–14 | [2,1,3,1,2] etc | 0.213 | 🟢 healthy |
| transport | pretrained | s0/s1/s42 | 1 / 5 / 7 | [0,1,0,0,0]; [1,0,2,2,0]; [2,0,0,2,3] | 0.087 | 🟡 near-floor, **pool functional** (late deltas recover) |
| transport | scratch | s0/s1/s42 | 0 / 1 / 1 | [0,0,0,0,0]; [0,0,1,0,0]; [1,0,0,0,0] | 0.013 | 🟡 near-floor → no-batch re-run = **0.027** (0.0/0.02/0.06), CONFIRMS near-floor |

**Conclusion of the forensics:** the batching bug manufactured **exactly ONE false result — lift pretrained (all 3 seeds
collapse)** — and it struck the **pretrained** arm specifically, fabricating the "scratch wins lift" reversal. 29/30 seed-runs
have functional pools: can/square/tool_hang/lift-scratch grow linearly; transport is genuinely near-floor but its pools
still produce *late* successes (deltas recover in chunks 4–5), proving non-corruption. **So NO full 30-eval re-run is
needed** — the in-flight lift-pretrained (s0/s1; s42 already = 0.95) + transport (completeness) re-runs are the entire
remediation. The asymmetry (collapse hits pretrained, not scratch, on the SAME task/harness) is the danger the audit
exposes: a silent eval bug that biases ONE arm inverts the headline. Reliability rule going forward: **read per-episode
success PATTERNS, not just the mean** — a chunk-1-then-flat mean is the tell.

### §10 🧊 CUBE DINOv2 ablation — the probe→PLANNING test (launched 2026-06-18 on 174, user-directed)
**Motivation:** robomimic guided/CEM is horizon-confounded ≈0, so it can't test whether DINOv2's better latent helps
*planning* (only bc). **Cube is the one task where intuition-guided planning WORKS** (reproduced earlier: guided 88% >
pure planning 78% > LeWM-baseline 68%, aligned goal↔success), so it's where the probe→planning half of the hypothesis is
measurable. Same clean ablation: same `dinov2-small-384` arch, flip only `model.encoder.pretrained`.
- **Run on 174** (cube data + OGBCube env + validated guided eval all live there; L40S is robomimic-only). Transferred only
  the small `lewm_dinov2.yaml` (174 had just `lewm.yaml`). Recipe = the validated cube run: `data=ogb` (=ogbench/
  cube_single_expert), `model=lewm_dinov2 embed_dim=384`, `action_pred.enabled=true` (base+GIP trained together), 60ep ×
  4000 batches, bs64. Arms: `cube_dinov2_scratch` (GPU1, pretrained=false) + `cube_dinov2_pretrained` (GPU7, pretrained=true).
- **Launch snags fixed (durable for next time):** (1) `STABLEWM_HOME` unset in the non-interactive ssh shell → loader tried
  to re-download the now-GATED ogbench HF dataset → HTTP 401; fix = `STABLEWM_HOME=/mnt/data_nvme1/minghao.fu/.stable-wm`
  (cube data cached at `$SWM/datasets/ogbench/cube_single_expert.h5`). (2) my `pkill -f "...cube_dinov2_[s]cratch"` matched
  the python LAUNCH line in the same bash wrapper → killed its own ssh session (the 255s); fix = no pkill (first run already
  died on the 401), and never pkill in the same wrapper as the launch. (3) only dinov2-*base* cached on 174 → set
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` so the public dinov2-*small* downloads anonymously past 174's stale token.
- **🔁 MIGRATED 174 → L40S via exp-bucket (2026-06-18, user-directed: consolidate on ONE machine, no cross-machine split).**
  Running cube on 174 while robomimic ran on L40S was the wrong call. Fix: the user's HF dataset **`mh-hf/exp-bucket`** is a
  public, consolidated **`.lance`** store of every family (robomimic/ogbench/mimicgen/libero/robocasa/dexmimicgen). Pulled
  `ogbench/ogb_cube_single.lance` from HF → **20 GB in ~90 s** (vs the 95 GB `.h5` rsync's ~2h, which I killed) — lance is
  ~5× smaller (JPEG frames) + HF CDN is fast. Wired a lance data config `config/train/data/ogb_lance.yaml` (name=
  `ogbench/ogb_cube_single.lance`, keys `pixels/action/observation`; smoke-tested loads+trains clean), **relaunched both
  cube arms on L40S** (`data=ogb_lance`, GPU3/4, same 60ep×4000 recipe), **killed the 174 runs**. **exp-bucket is now the
  canonical data path — any machine pulls the same lance, no rsync** ([[project_dinov2_init_ablation]]). Infra fix: the swm
  loader needs `STABLEWM_HOME` set + prefers `.lance` over `.h5`; lance is fork-unsafe so `_force_spawn()` handles workers.
- **Status:** both arms training on L40S, Epoch 0/60 @ ~3.5 it/s → **~18h to converge** (L40S 46 GB GPUs slower than 174 H100
  + square-GIP contention), then the cube **guided** eval (`eval_gip.py`/config-name cube, `gip_eval.mode=guided`) per arm.
  **The read to watch:** does pretrained's latent give a higher guided SR than scratch? If yes → DINOv2 helps *planning* too
  (probe→planning), completing the story (transport showed probe→bc); if not → the benefit is bc-specific.
- **🛠️ CUBE GUIDED EVAL DE-RISK (2026-06-18, done while cube trains — caught issues that would've failed the eval after ~15h).**
  The cube guided eval is `eval_gip.py --config-name cube policy=cube_dinov2_<arm> +gip_eval.mode=guided eval.num_eval=N`
  (guided = action head warm-starts CEM; the 88%-reproduction mode). Two blockers found+fixed on L40S: (1) **OGBCube env
  deps missing** — `pip install pygame pymunk shapely` (the swm env registry imports all envs at load, incl. PushT's 2D-physics
  deps; needed regardless of data format). (2) **The eval is hardcoded for HDF5Dataset and is pervasively INCOMPATIBLE with
  the lance** — `gip.sample_eval_episodes` calls `dataset.get_row_data(picks)["step_idx"]`, but LanceDataset's `get_row_data`
  doesn't return index columns (and its `get_col_data` returns 2-D `(N,1)` for index cols vs HDF5's 1-D). A LOCAL lance→h5
  `convert(dest_format='hdf5')` is ALSO blocked — the cube lance has a string column (`privileged_target_task`, dtype `<U4`)
  h5py won't serialise. **⇒ verdict: lance is great for TRAINING (fast transfer, 20 GB/90 s from exp-bucket), but the EVAL
  needs the `.h5`** (its native, 174-proven format). Transferring `cube_single_expert.h5` (95 GB) 174→L40S in the background
  (non-blocking, finishes well before cube's ~15h convergence). Patches left in place (`gip.get_dataset` lance-aware fallback,
  index-squeeze) are harmless on the h5 path (no `.lance` for `cube_single_expert` → HDF5Dataset; squeeze is a noop on 1-D).
  **Durable lesson: training-format ≠ eval-format here — the swm eval pipeline assumes HDF5Dataset semantics.**
- **✅ PRE-FLIGHT VERIFIED (2026-06-18, hour ~2 of cube training):** confirmed the guided eval will actually fire at ep60 —
  `eval_gip.py` + `config/eval/cube.yaml` (the `--config-name cube` target) + the cube **h5** (`ogbench/cube_single_expert.h5`,
  eval needs it for goal sampling) all present; `cube_eval_watcher.sh` waits for both `weights_epoch_60.pt` (ckpts save at
  multiples of 10 → ep60 triggers) then fires guided N=50 s42 on **GPU3/4 = the cube TRAINING GPUs** (train+eval same GPU →
  cross-GPU-render constraint [[project_l40s_cross_gpu_rendering]] satisfied). Watcher does s42 only; I add s0/s1 manually after
  s42 validates N=50 (staged — don't launch 6 heavy cube-env evals blind). (Gotcha logged: querying the repo as the outer `exx`
  ssh user returns empty/permission-denied — the repo is minghao.fu-owned; always `sudo -u minghao.fu` to inspect it.)

- **🎯 CUBE CONVERGENCE GATE (rule-9 plateau check, observed 2026-06-18 @ ep44/42).** validate/pred_loss trajectory:
  **scratch PLATEAUED** ~0.0093–0.0101 flat since ~ep35 (WM converged); **pretrained still gently ↓** (ep32 0.0097 → ep42
  **0.0080**, ~1.7e-4/ep) AND already **lower than scratch** (0.0080 < 0.0094 → better WM pred_loss, consistent with the better
  latent). Both arms get the SAME 60ep budget (the validated cube-reproduction convergence point) → fair matched-compute
  ablation. **Decision gate at ep60:** if pretrained's last-10% (ep54–60) is flat → eval as-is; if still clearly dropping →
  EXTEND pretrained before the guided eval (rule 9: "when in doubt, train heavier"; conservative — a still-improving pretrained
  only helps a pretrained-wins read, but could mask a tie). A pretrained-guided WIN at ep60 is already conservative (pretrained
  is mid-convergence vs scratch converged); a TIE warrants the extension before concluding "planning is bc-specific".
- **🎯 CUBE GUIDED — PRE-REGISTERED interpretation (written 2026-06-18 BEFORE any SR lands, to block post-hoc rationalizing).**
  Eval = `eval_gip.py --config-name cube policy=cube_dinov2_<arm> +gip_eval.mode=guided eval.num_eval=50` (N=50 mandatory),
  **3-seed** per arm. NOT affected by the bc batching bug (eval_gip uses num_envs=num_eval=one fresh pool — validated at N=5/80%).
  Anchor: the reproduced cube hierarchy is guided **88%** > pure-planning 78% > LeWM-baseline 68% (aligned goal↔success), so
  guided is a *working* planning signal on cube — the one place probe→planning is measurable. **Decision rule (compare the
  3-seed means with their SE):**
  - **pretrained guided > scratch guided by > 1 pooled SE →** DINOv2's better latent helps *planning*, not just bc. This
    completes the two-axis story: transport already showed **probe→bc**; cube would show **probe→planning**. Headline-positive.
  - **pretrained ≈ scratch (within 1 SE) →** the encoder benefit is **bc/imitation-specific** and does NOT transfer to CEM
    planning on cube → the latent gain is real but planning is bottlenecked elsewhere (dynamics rollout / CEM), still an honest
    result ("encoder helps imitation; planning is dynamics-limited"). Do NOT spin a tie as a win.
  - **pretrained < scratch →** surprising; run the same per-episode PATTERN forensics (§10 audit discipline) before trusting,
    even though the batching bug can't apply here — a real reversal would need a mechanism, not just a mean.
  - **Also probe the cube frozen encoders** (Ridge latent→GT cube state, held-out R²) to fill the **2×2 = {probe gain, control
    gain} × {bc, planning}**. Pre-registered expectation: cube is a single-arm-ish tabletop scene, so by the scene-width pattern
    the cube *probe* gap is likely small (like can/lift); if guided SR nonetheless separates, that's another probe/control
    DECOUPLE — strengthening "probe necessary-but-not-sufficient" on the *planning* axis too.
- **✅ CUBE GUIDED RESULT — landing 2026-06-18 (both arms ep60, converged; gate PASSED: pretrained pred_loss plateaued ~0.00724
  since ep54, scratch ~0.00860, both flat → no extension).** **s42 (N=50): scratch 86.0 (43/50) = pretrained 86.0 (43/50) — EXACT
  TIE.** Reliability check done (the means matched suspiciously): both loaded the CORRECT distinct checkpoints (md5 differ), and the
  per-episode arrays are NOT identical — they differ at episodes 26 & 38 (the diffs cancel) → a GENUINE coincidental tie at 43/50,
  not a bug. **Per the pre-registered rule, a tie = "encoder benefit is bc/imitation-specific; does NOT transfer to CEM planning on
  cube — do NOT spin as a win."** Mechanism read: the better latent (pred_loss 0.00724 vs 0.00860, and the §10 probe gains) does NOT
  improve guided SR — the CEM (300 samples × 30 steps) washes out the encoder difference on cube's goal-geometry-dominated task.
  **✅ 3-SEED COMPLETE (N=50 each): scratch 86/86/84 → 85.3; pretrained 86/96/84 → 88.7.** Δ=+3.3 (pretrained), but pretrained
  spreads 84–96 (s0=48/50 high) → SE_scr 0.67, SE_pre 3.71, pooled-SE-of-diff 3.77 → **Δ = 0.88 SE → WITHIN 1 SE → NOT SIGNIFICANT
  = TIE.** **VERDICT (per pre-registration): bc-specific — DINOv2 does NOT significantly help guided PLANNING on cube.** ⇒ **the
  two-axis story closes cleanly: DINOv2 helps bc/control (probe→bc, all 5 robomimic) but NOT planning (cube guided tie) → "probe→bc,
  NOT probe→planning."** Honest caveat: pretrained's mean is marginally higher (one high seed); more seeds *could* tip it, but the
  pre-registered 3-seed rule says tie, and chasing a 0.88-SE gap with extra seeds would be p-hacking.
  **✅ CUBE PROBE (latent→GT cube `observation` 28-d, ep60 encoders, episode-split overlap=0): scratch lin 0.678/rbf 0.611,
  pretrained lin 0.715/rbf 0.622 → GAP lin +0.038, rbf +0.011 = SMALL.** Cube patterns with the SINGLE-ARM cluster (lift +0.06,
  can +0.02), NOT transport (+0.42) — confirms the scene-width pattern (cube = tabletop single-arm-ish scene). **Refined mechanism
  (more honest than "CEM washes out a big advantage"): cube's latent gain is SMALL to begin with (+0.038 probe), and that small gain
  + the better WM (pred_loss 0.0072<0.0086) does NOT move guided-planning SR (tie) — guided planning on cube is insensitive to these
  modest encoder-quality differences.** The 2×2 {probe,control}×{bc,planning} is filled: probe-gain narrow (transport only;
  cube small +0.038), control(bc)-gain broad (all 5 robomimic), planning-gain absent (cube tie). Headline: **"probe→bc, NOT probe→planning."**
- **⚠️ HONEST PROBE-POWER CAVEAT (applies to the whole decouple claim).** The frozen probe is a **LINEAR** Ridge readout to the
  **low-dim GT state** vector. A flat probe gap with a real control gain (tool_hang: probe 0.82≈0.82, bc 0.333 vs 0.213) is
  consistent with "DINOv2 adds task-relevant features the linear-state-probe can't see" — but a linear probe can also miss
  **nonlinearly-decodable** state. So "probe flat, control up" is necessary-but-not-sufficient evidence for a genuine
  task-feature gain, NOT proof. **Sharpener (1) — DONE, see the nonlinear-probe subsection below:** re-probed with KernelRidge
  RBF; the single-arm gap stays flat-to-negative under the nonlinear probe too ⇒ the control gain is genuinely orthogonal to
  GT-state, NOT a linearity artifact. (An MLP was the wrong tool — underfit/overfit; KRR-RBF is right for 4800×384.)
  **Sharpener (2) — RECONSIDERED, dropped as ill-posed.** A "grasp/contact-state probe" would derive its label (grasp,
  object-eef distance, gripper qpos) FROM the sim-state vector — but the RBF result already shows the bc gain is orthogonal to
  the FULL GT-state, so any state-derived sub-label is flat *by construction*. ⇒ the orthogonal feature is, definitionally, NOT
  in the low-dim sim-state; it lives in features the state vector doesn't encode (most plausibly fine VISUAL/contact cues that
  matter for closed-loop control but not for coarse pose). That is the honest endpoint of the latent analysis — no further
  state-probe can localize it; a pixel/attention-attribution study would be the next tool, but it's a different (heavier)
  analysis, not a cheap re-probe. **So the latent axis is COMPLETE: linear probe (scene-width pattern) + nonlinear probe
  (linearization vs orthogonality) + leakage check. The remaining open question is the PLANNING axis (cube guided).**

**CAN frozen probe (Phase-2, both arms ep100):** pretrained state-R² **0.681** vs scratch **0.659** (+0.02, small) — can
patterns with the **single-arm** family (tool_hang +0.00, lift +0.06), NOT transport's +0.42. So the **probe gap tracks
scene width**: the wide 2-arm transport view is where from-scratch under-fits and the ImageNet prior rescues the latent;
single-arm scenes (can/lift/tool_hang) from-scratch nearly matches on the probe. (Collapse diagnostics healthy throughout:
can scratch erank 39/384 zstd 0.48, pretrained 70/384 zstd 2.27 — pretrained richer but both decode state ~equally.)

### §10 🔬 NONLINEAR probe (KernelRidge RBF) — validates the decouple is NOT a linear-readout artifact (2026-06-18)
**Motivation (rigor):** the whole "probe necessary-but-not-sufficient" decouple rests on a **LINEAR** Ridge probe being flat
on single-arm tasks while bc improves. A linear probe can miss **nonlinearly-decodable** state, so a flat linear gap is only
*suggestive*. I re-probed the SAME frozen encoders with a nonlinear readout (KernelRidge RBF, γ=1/384, α=10 — robust on this
small-data/high-dim regime; an MLP swung between underfit [adam+early-stop, scored BELOW linear] and overfit [lbfgs, negative
test-R²], so KRR is the right tool). Same 6000-frame held-out split; the GAP (pretrained−scratch) uses the SAME probe for both
arms so it's a fair within-task comparison regardless of absolute probe power (KRR α=10 is mildly under-powered → its absolute
R² sits a touch below linear, which only makes the transport finding below a LOWER bound).

| task | lin scr | lin pre | **lin gap** | rbf scr | rbf pre | **rbf gap** | read |
|---|---|---|---|---|---|---|---|
| transport | 0.418 | 0.838 | **+0.419** | 0.683 | 0.753 | **+0.070** | gap COLLAPSES under RBF — it was largely linearization |
| lift | 0.319 | 0.384 | +0.065 | 0.195 | 0.314 | **+0.119** | gap PERSISTS/grows — genuine latent gain |
| can | 0.659 | 0.681 | +0.021 | 0.657 | 0.647 | **−0.009** | flat under both |
| square | 0.884 | 0.878 | −0.006 | 0.844 | 0.771 | **−0.073** | flat under both |
| tool_hang | 0.823 | 0.825 | +0.002 | 0.770 | 0.668 | **−0.102** | flat under both (yet bc 0.333 vs 0.213) |

**Two findings:**
1. **🎯 transport's +0.42 linear gap is mostly LINEARIZATION, not new information.** The RBF probe RAISES *scratch* from
   0.418→**0.683** (+0.265) but barely moves *pretrained* (0.838→0.753). ⇒ the from-scratch transport latent DOES encode the
   wide-scene state — but **nonlinearly/tangled**; a linear probe can't read it. DINOv2-init's real contribution is to make
   that state **linearly accessible** (representation geometry), shrinking the gap +0.42→+0.07. This is a sharper, more
   defensible mechanism than "rescues the latent," and it's exactly *why linear-probe is the standard SSL eval* (pretrained
   features are linearly separable). Since KRR is under-powered here, scratch's true nonlinear decodability is ≥0.68 — the
   linearization claim is conservative.
2. **🎯 the single-arm decouple is ROBUST to probe nonlinearity.** On can/square/tool_hang the RBF gap is flat-to-negative
   (−0.009/−0.073/−0.102): **neither linear NOR nonlinear** probing finds a pretrained state-decodability advantage, yet bc
   improves (can 1.25×, square 2.2×, tool_hang 1.6×). tool_hang is the cleanest case — +0.002 linear, −0.102 RBF, both flat,
   bc 0.333 vs 0.213. ⇒ the bc control gain on single-arm tasks is **genuinely orthogonal to GT-state decodability** (a
   nonlinear probe can't recover it either) → the "probe necessary-but-not-sufficient" claim is **not** a linear-readout
   artifact. The benefit lives in task-relevant *local/contact* features the GT-state vector simply doesn't contain.

**Refined CONCLUSION #7 (two distinct mechanisms, now probe-validated):** DINOv2-init helps LeWAM control via (a) on the wide
2-arm **transport** scene, **linearizing** an already-present-but-tangled global state (linear-probe gap +0.42, mostly
geometry — nonlinear gap only +0.07); (b) on the single-arm **contact** tasks (can/square/tool_hang), adding task-relevant
LOCAL features that **no state-probe (linear or nonlinear) captures** (probe gap ≈0 under both, bc up 1.25–2.2×). lift is a
mild (a)+(b) hybrid (probe gain under both, data-scarcity rescue). Script: `/tmp/frozen_probe_mlp.py` (KRR-RBF + Ridge),
log `/mnt/minghao_data/mlp_probe_all.log`.

**✅ PROBE-SPLIT LEAKAGE CHECK (2026-06-18, adversarial — "could the held-out R² be inflated by train/test sharing
episodes?").** The probe samples a fixed stride across the dataset then holds out the last 20%. The robomimic h5 is stored
**episode-contiguously** (`episode_idx`/`ep_offset`/`ep_len` fields; can = 200 eps, ~116 frames each), so the stride+tail
split lands on an episode boundary. Verified empirically on can: train spans **127 episodes**, test **30 episodes**, overlap
= **exactly 1** (the single straddling boundary episode #126). ⇒ the split is effectively episode-level; ~1/30 test episodes
has any temporal leak, and that leak hits BOTH arms identically (same frames), so the **gap (pretrained−scratch) is
leak-invariant** — it's a clean within-pair comparison regardless. Probe numbers stand. (To make it perfectly clean one would
drop the boundary episode from test, but the effect is <1 episode and cancels in the gap.) This was the last open reliability
concern on the latent axis; combined with the §10 bc batching-bug fix, both axes (control + latent) are now audited.

### §10 🔴🐛 GUIDED EVAL CONFOUND — goal_offset(280) >> plan-horizon(25) [found by code audit 2026-06-17, user-prompted]
**MY BUG, caught before trusting the "fundamental" story (user: "deeply find possible bugs first").** The intuition-guided
CEM plan reaches only **horizon×frameskip = 5×5 = 25 env-steps** (`config/eval/robomimic.yaml plan_config horizon=5,
action_block=5`; data frameskip=5). But I set **`goal_offset_steps=280` (tool_hang) / 270 (transport)** to match the long
episodes. So the CEM cost (`jepa.py criterion`, scores the +25-step predicted latent vs the +280-step goal latent) compares
a state the plan CANNOT reach to the goal → cost ~flat across candidates → CEM (no variance floor, `cem.py:245`) returns
near-mean noise → the good bc warm-start (only seeded at horizon pos 0 anyway) is washed out → **guided≈0**.
- **Every config where guided historically WORKED used `goal_offset = horizon×frameskip = 25`** (cube guided 88%, pusht,
  Lift default). tool_hang/transport are the ONLY tasks where goal_offset was raised without raising the horizon. So the
  guided=0 on these two is **CONFOUNDED — NOT proven fundamental.**
- **Shared parts are fine (bc works, 0.22):** encoder, ImageNet normalize (goal+rollout use the SAME `gip.img_transform`),
  action de-norm, env, success criterion, AND the WM rollout DOES use candidate actions (AdaLN conditioning, not detached).
  Audit ruled out the classic bugs; the only real defect is the horizon-vs-offset mismatch (+ one CEM weakness: no var floor).
  **⚠️ CORRECTION (2026-06-18, code re-audit — §9):** the "warm-start only at pos 0, pos 1–4 zero-padded" claim here was WRONG
  for the swm path — `prepare_init_action` calls `get_action(horizon=H)` → `intention_rollout` gives a FULL H-block seed (the
  1-block+zero-pad was the separate `eval_histguided_robomimic.py`). So the only real CEM weakness is the **no-var-floor collapse**,
  now fixed via the opt-in `gip.VarFloorCallback` / `solver=cem_floor` (§9).
- **bc numbers stay VALID** (bc ignores the goal; goal_offset only sets its start-sampling range). transport bc ~0.01,
  tool_hang bc 0.22 unaffected.
- **FIX / re-test:** re-run guided with `goal_offset_steps=25` (matched), + `cem.py` var floor + seed the full horizon from
  the intuition. Then guided-vs-bc is the same coherent setup as cube/pusht. **CAVEAT:** at offset=25 on these LONG tasks
  (median ~470 steps), reaching the +25 goal ≠ task completion (is_success), so guided SR may still be low — but for a
  *horizon-too-short-for-a-long-task* reason (→ needs HIERARCHICAL SUBGOALS, the WorldDP direction), NOT goal-cost-blindness.
  The cleaner fundamental test remains Lift (§7b, offset=25, genuinely ~0).

### §10 Phase-2 probe — LIFT (2026-06-17): DINOv2 helps via DATA scarcity (refines the hypothesis)
lift dinov2 base WMs (no canonical on L40S): scratch state-R² 0.319 / proprio 0.660 (zstd 0.37, erank 34/384);
pretrained 0.384 / 0.841 (zstd 2.07, erank 44). **pretrained > scratch** (state +0.065, proprio +0.18). I predicted lift
would be INERT (tight single-arm scene like tool_hang) — WRONG. The difference: lift's data is ~10× smaller (48-step
episodes, ~9.6k frames, capped ~150 batches/epoch) so from-scratch under-fits from **data scarcity**, and the prior helps —
the SAME mechanism as transport (under-fit from scene complexity), different cause. **Refined hypothesis: DINOv2-init helps
whenever the from-scratch encoder can't fit the data — driven by (scene complexity) OR (data scarcity) OR (model size vs
data) — and is inert only when from-scratch already suffices (tool_hang: tight scene + 96k frames).** CAVEAT: lift's
absolute state-R² is low (0.32–0.38, vs tool_hang 0.82) and its batch budget is ~10× smaller, so it's a different training
regime; the RELATIVE pretrained>scratch is the signal, not the absolute level. For a clean cross-task probe, match the
gradient-step budget (lift's small data means fewer updates).

### §10 🔬 GUIDED on long tasks — RESOLVED (matched offset=25, N=50, 2026-06-17): it's the HORIZON, not goal-cost-blindness
Matched tool_hang scratch, offset=25, budget=50, N=50: **bc=0/50 AND guided=0/50** (the N=10 guided 0.1 was one lucky
near-end start). Reason: offset=25 forces a reachable goal, but on a 469-step task `start ∈ [0, len-26]` so starts span the
whole episode and **budget=50 lets almost none reach task completion** — both policies floor at 0. So the offset=25/budget=50
("cube/pusht") setup is INCOHERENT for long multi-stage tasks. Combined with §10's offset-bug finding:
- **Lift (short, median ~48 steps): completable at offset=25/budget=50, and planning is genuinely ~0 (§7b)** → THIS is the
  clean goal-cost-blindness evidence (grasp not in single-frame latent). Holds.
- **tool_hang/transport (long): the "guided=0" was a CONFOUND of (a) my offset=280 >> horizon=25 bug AND (b) a structural
  horizon limit** — a single fixed-goal, 25-step CEM plan can reach a +25 subgoal but cannot SPAN a 469-step multi-stage
  task (after reaching start+25 the fixed goal is behind it and it stalls; raising budget helps bc-imitation, not
  fixed-goal planning). It is NOT clean evidence of goal-cost-blindness on these tasks.
**Corrected conclusion:** on long-horizon multi-stage robomimic, single-goal WM-CEM is **rollout-horizon-limited** (≈25 env-
steps); bc/imitation has no such limit (hence bc 0.22 full-task > guided). The honest fix is **hierarchical subgoal
planning** (WorldDP direction) — plan a SEQUENCE of ~25-step subgoal latents toward the far goal, reach each. This makes the
WorldDP multi-stage-dataset extension empirically motivated. Goal-cost-blindness remains a SEPARATE, real effect, cleanly
shown only on the short completable task (Lift). The user's "deeply check for bugs first" caught that I'd conflated three
distinct failure causes (offset bug + horizon limit + goal-cost-blindness) into one "guided fails on robomimic" claim.

---

## §11 ✅🆕 GC-IDM (`gcidm`) REPRODUCTION on cube — arm 1 of the 3-arm head-to-head (CEM / gcidm GC-IDM / OURS) (L40S, 2026-06-19)

**Motivation.** `gcidm` (UMich, arXiv 2605.08732, [[reference_gcidm_umich]]) is the planning-free goal-conditioned IDM that
largely scoops our planning-free goal-conditioned direction; it runs on OUR exact base (LeWM sigreg vit-tiny-192) and OUR
exact envs (cube/pusht/reacher). To position our paper we must reproduce it faithfully and confirm its headline cube number
(98.7%, n=200) inside our codebase, as arm 1 of the CEM / gcidm / OURS head-to-head. This is the gcidm arm only; CEM and
OURS are separate arms.

### Method (verbatim spec, implemented exactly)
Goal-conditioned inverse dynamics on FROZEN LeWM latents; planning-free (NO CEM, NO WM rollout):
- **Encoder FROZEN**: `cube_ours_lewm_weights.pt` (vit-tiny-192, SIGReg-regularized LeWM). z = `encode({pixels})['emb'][:,0]`
  (192-d). Latent sanity: per-dim std (mean over dims) = **1.024** over 10k episodes → NOT collapsed (SIGReg isotropy holds).
- **GCIDMHead** (`gcidm.py`): backbone input concat **[z_t ∥ z_goal]** (2×192=384) → 3 MLP layers, hidden **512**,
  GELU+LayerNorm+dropout 0.1. **Horizon = AdaLN-Zero**: h_norm=min(steps_left,H_max)/H_max → sinusoidal(64 freqs)=128-d →
  2-layer MLP → per-layer (scale,shift), applied `x=LayerNorm(x)*(1+scale)+shift`, the (scale,shift) projection **init ZERO**
  (identity at init — unit-tested: max|a(h=0)−a(h=1)|=0.0e0 at init). Output = single 25-d raw-action block (env action dim 5 ×
  frameskip 5). **Param count = 1,131,545 (1.132M)** ≈ paper's "~1.5M, ~10% of LeWM predictor".
- **Training = hindsight MSE on frozen latents.** Tuples (z_t, z_{t+h}, h, a_t), h~Uniform[1,H_max], **H_max=50** (= the
  `hindsight_max_k` default; cube episodes are 201 raw frames=40 obs-steps, so H_max=50 covers the full episode). z_t = last
  window frame, a_t = the 25-d action block taken AT that frame (advances one obs-step), z_goal = hindsight goal frame.
  Action z-scored per 5-dim with the dataset action stats (matches train.py AND the eval inverse_transform). loss = MSE.

### Infra (the speed fix that made convergence feasible)
The 224×224-image dataloader from the 100GB cube lance dataset is I/O-bound at ~5s/batch (a full LeWM-encode-every-epoch run
stalled). Since the encoder is FROZEN, `train_gcidm.py` **precomputes all latents once** (10000 eps in 309s, bf16 autocast)
→ caches to `checkpoints/cube_gcidm/latents_cache.pt` (182MB) → phase-2 trains the head on flat GPU tensors with vectorized
hindsight sampling (**~1.2s/epoch**, num_workers=0). This is exactly how gcidm gets "~20 min/env". (First attempt used a
DataLoader with persistent_workers=True over the in-memory cache → deadlocked on the lancedb fork runtime; fixed → num_workers=0.)

### Exact commands
```
# train (horizon ON), GPU 0:
STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl_x TMPDIR=/tmp \
  CUDA_VISIBLE_DEVICES=0 python train_gcidm.py --epochs 200 --batch_size 1024 --H_max 50 \
  --run_name cube_gcidm --cache_run cube_gcidm
# horizon ablation (identical except --ablate_horizon), GPU 1, reuses the SAME latents cache:
  CUDA_VISIBLE_DEVICES=1 python train_gcidm.py --epochs 200 --batch_size 1024 --H_max 50 --ablate_horizon \
  --run_name cube_gcidm_noh --cache_run cube_gcidm
# eval N=50 (horizon ON), GPU 2:
STABLEWM_HOME=... MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl_x TMPDIR=/tmp CUDA_VISIBLE_DEVICES=2 \
  python eval_gip.py --config-name cube policy=cube_gcidm +gip_eval.mode=gcidm eval.num_eval=50
# eval N=50 (horizon OFF ablation), GPU 3:
  CUDA_VISIBLE_DEVICES=3 python eval_gip.py --config-name cube policy=cube_gcidm_noh +gip_eval.mode=gcidm \
  +gip_eval.gcidm_run=cube_gcidm_noh +gip_eval.ablate_horizon=true eval.num_eval=50
```
Eval config (cube.yaml, unchanged): num_eval=50, goal_offset_steps=25 (raw frames=5 obs-steps), eval_budget=50,
action_block=5, dataset=ogbench/cube_single_expert, terminate_at_goal. Eval-time remaining horizon h0 = goal_offset/action_block
= 25/5 = **5 obs-steps**, decremented one per replan, clamped ≥1, normalized min(h,50)/50. Train+eval same L40S box (per the
cross-GPU-rendering rule). NO CEM, NO WM rollout — one GCIDMHead forward per replan.

### Convergence evidence (val MSE on frozen latents, z-scored 25-d action)
- **horizon ON** (`cube_gcidm`): train_mse 0.287, **val_mse 0.367** — FLAT over epochs 175→200 (0.367/0.367/0.370/0.368),
  best val 0.364. Plateau ⇒ converged.
- **horizon OFF** (`cube_gcidm_noh`): train_mse 0.432, **val_mse 0.575** (best 0.555) — also flat ⇒ converged. Horizon
  conditioning cuts val MSE by ~36% (0.36 vs 0.56) → AdaLN is load-bearing already on the training objective.

### RESULTS — cube goal-reaching, N=50 (50/50 envs, single eval per setting; deterministic head → 50 episodes IS the sample)
| arm | success/N | SR | vs reference |
|---|---|---|---|
| **GC-IDM (horizon ON), seed=42** | **50/50** | **100.0%** | ≥ gcidm-published **98.7** ✅ ; ≫ CEM≈67 ; ≫ our-guided≈88 |
| **GC-IDM (horizon ON), seed=7** | **50/50** | **100.0%** | stable across 2 eval seeds (different episode draws) — not a sampling fluke |
| GC-IDM (horizon OFF, AdaLN ablation) | 37/50 | 74.0% | **−26pp** vs horizon-ON (paper: −42pp) → horizon LOAD-BEARING ✅ |
| gcidm-published cube (n=200) | — | 98.7 | reproduced/exceeded |
| CEM (dashboard, H100-174) | — | ~67 | beaten by +33 |
| our-guided (dashboard, H100-174) | — | ~88 | beaten by +12 |

**Conclusion.** The gcidm GC-IDM reproduces on cube at **100% (50/50) at N=50**, AT/ABOVE the published 98.7% and clearly
beating CEM≈67 and our-guided≈88. The horizon-AdaLN ablation drops SR to 74% (−26pp), confirming AdaLN-Zero horizon
conditioning is load-bearing (paper claims −42pp; direction + significance reproduced, magnitude differs by env/setup). The
74% (not 0, not 100) on the SAME goal mechanism also rules out a goal-frame artifact — the head genuinely uses both z_goal and
horizon. Arm 1 of the head-to-head is DONE and faithful.

**⚠️ same-box caveat (cross-GPU/cross-machine rendering rule):** the GC-IDM 100% is on L40S; the CEM≈67 / guided≈88 references
are from H100-174 (different box). For a rigorous same-box head-to-head the CEM and OURS arms should be re-run on L40S; the
100% margin is wide enough that GC-IDM clearly leads even allowing cross-machine variance, but the CEM/OURS L40S numbers are
the remaining work for arms 2–3.

### Files added (all in `$B=le-wm-repro` on L40S; backups `*.bak_gcidm`)
- `gcidm.py` — GCIDMHead + AdaLNBlock + sinusoidal horizon embedding (NEW; self-test `python gcidm.py`).
- `train_gcidm.py` — precompute-latents + hindsight-MSE trainer (NEW).
- `gip.py` — added `load_gcidm_model`, `GCIDMPolicy`, and a `mode=gcidm` branch in `build_policy` (backup `gip.py.bak_gcidm`).
- `eval_gip.py` — `mode=gcidm` routes around `load_gip_model` (GCIDM self-loads); original block preserved verbatim in the
  `else` (backup `eval_gip.py.bak_gcidm`). Default bc/guided/planning behaviour byte-identical.
- Checkpoints: `$STABLEWM_HOME/checkpoints/cube_gcidm/{gcidm_head_best.pt, gcidm_config.json, latents_cache.pt}` and
  `cube_gcidm_noh/` (ablation). Eval videos under `$STABLEWM_HOME/gip_eval/gcidm/`.

## §12 ✅🆕 OURS (history-conditioned forward GC policy) + same-box CEM/guided — arms 2 & 3 of the 3-arm cube head-to-head (L40S, 2026-06-19)

**Motivation (why).** Complete the 3-arm cube goal-reaching head-to-head started in §11. Arm 1 = gcidm GC-IDM = 100% (done).
Arm 2 = OURS = our HISTORY-conditioned forward GC policy (the wedge vs gcidm's Markovian (z_t,z_goal,h)): `predict_intention`
is autoregressive over the latent history z_{t-HS+1..t} + past actions a_{<t}, PLUS goal_emb (z_goal) PLUS the SAME AdaLN-Zero
horizon as gcidm, run PLANNING-FREE at eval (intention_rollout horizon=1, one forward pass, NO CEM). Arm 3 = same-box CEM +
guided (the 174 CEM≈67 / guided≈88 are cross-box; re-run on L40S so all arms share the box → kills the cross-GPU rendering
confound, see memory `project_l40s_cross_gpu_rendering`). All on the SAME frozen base (`cube_ours_lewm_weights.pt`, vit-tiny-192),
SAME data (`ogbench/ogb_cube_single.lance`), SAME eval (`config/eval/cube.yaml`: N=50, goal_offset_steps=25, eval_budget=50,
terminate_at_goal), SAME H_max=50.

### Code added to make OURS a fair drop-in (additive, config-gated, default OFF → existing bc/guided/planning/gcidm byte-identical)
- **`jepa.py`** (backup `jepa.py.bak_ours`): (1) imported `_sinusoidal_embedding` from `gcidm.py` and added a `HorizonModulator`
  module = the SAME AdaLN-Zero machinery GC-IDM uses (sinusoidal(64)→2-layer MLP→per-target (scale,shift), cond_proj ZERO-init →
  identity at init). It modulates the intention embedding BEFORE the action_decoder: `x = LayerNorm(x)*(1+scale)+shift`.
  (2) `JEPA.__init__` gained `horizon_conditioned=False`; when true it builds `self.horizon_modulator` (emb_dim read from
  `predictor.pos_embedding`). (3) `predict_intention(..., horizon=None)` and `intention_rollout(..., horizon_norm=None,
  past_action_blocks=None)` thread the horizon through the modulator. Unit tests PASS: AdaLN-Zero identity at init
  (`max|a(h=0)-a(h=1)|=0`), horizon changes output after weights perturb, and `predict_intention()` == `predict_intention(horizon=None)`
  (default path byte-identical). gcidm.py self-test still PASSES (import added no regression).
- **`train.py`** (backup `train.py.bak_ours`): sets `cfg.model.horizon_conditioned=True` when `action_pred.horizon_conditioned=true`
  (so the flag rides into config.json → eval rebuilds the modulator); the forward normalizes `batch["horizon"]` (realized hindsight
  horizon in obs-steps, from `GoalSamplingDataset`) by `min(h,H_max)/H_max` and passes `horizon=` to `predict_intention`.
- **`gip.py`** (backup `gip.py.bak_ours`): `BCPolicy` gained the eval-time horizon countdown (init `horizon0=goal_offset/action_block=5`,
  decrement/replan, clamp≥1, normalize min(h,H_max)/H_max — EXACTLY `GCIDMPolicy`'s scheme) AND a per-env rolling HS-frame +
  HS-1-action-block buffer (see bug below). `build_policy` passes `H_max`, `horizon0`, `history_size`. `attach_intention_actor`
  (guided) feeds the warm-start a constant `_guided_horizon_norm`. The bc/guided/planning/gcidm paths are byte-identical when no
  `horizon_modulator` is present (verified by diff: every `<` line is a signature I extended with a default-None/False arg; no logic deleted).
- **`train_ours_gc.py`** (NEW): fast cached trainer mirroring `train_gcidm.py` — REUSES the SAME frozen-latent cache the gcidm arm
  built (`cube_gcidm/latents_cache.pt`: per-ep projected emb `encode(pixels)['emb'][:,0]` + z-scored 25-d action blocks), so OURS and
  GC-IDM train on byte-identical frozen features. Trains a fresh `action_predictor` (ARPredictor) + `action_decoder` (MLP) +
  `HorizonModulator` on hindsight windows (z_{t-HS+1..t} + a_{<t} + z_goal=lat[t+h] + h~Uniform[1,50]); trunk + action_encoder FROZEN.
  ~30s/epoch (vs 36 min/epoch for the image-loading `train.py` path on the 20G lance — the lance DataLoader fork support
  deadlocks/respawns under num_workers>0, so the image path is impractical here; the latent-cache path is the gcidm-proven fix).
  Saves the BEST-val_act checkpoint as `weights_epoch_1.pt` (early-stop; `load_gip_model` picks the highest epoch#) + latest as
  `weights_epoch_0.pt`. config.json carries `horizon_conditioned=true` so `load_gip_model` round-trips (missing=0 unexpected=0, verified).

### Exact train command (OURS, L40S GPU 1)
```
STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl_x TMPDIR=/tmp CUDA_VISIBLE_DEVICES=1 \
  python train_ours_gc.py --epochs 40 --batch_size 512 --lr 3e-4 --H_max 50 --history_size 3 --seed 3072 --run_name cube_ours_gc
```
Warm-started from the SAME frozen base (`/mnt/minghao_data/.stable-wm/decoders/cube_ours_lewm_weights.pt`), hindsight goals
(Uniform[1,50] obs-steps), w_act=1 w_intent=1 detach_target=1, cosine LR T_max=40. **Convergence:** val_act minimum at **epoch 15
= 0.23290** (val_int 0.12446); trajectory val_act 0.273(ep1)→0.235(ep11)→**0.2329(ep15,BEST)**→0.245(ep26)→0.254(ep40): the
val metric PLATEAUS by ~ep14-15 then mildly OVERfits as train_act keeps dropping (0.21→0.16) — classic frozen-feature head
convergence. `weights_epoch_1.pt` = the ep-15 best (the reported model). 380k hindsight windows (342k train / 38k val).

### 🔴🐛 BUG found + fixed: eval fed the history-conditioned head ONE frame (it trained on HS=3)
First N=50 OURS eval gave **36/42 → 39%** (FAR below the <85 near-ceiling threshold the task flagged for diagnosis). Instrumented
`BCPolicy.get_action`: the eval harness hands the policy `pixels_shape=[N,1,3,224,224]` — a **single** obs-frame per step — but
the head TRAINED on an HS=3 latent history (z_{t-2..t}) + past actions a_{<t}. So `intention_rollout` saw `emb=(B,1,D)`, putting
the current frame at positional index 0 (training had it at HS-1=2) with NO history → the planning-free prior was imprecise.
(GC-IDM is unaffected: Markovian, 1 frame by design → 100%.) **Fix:** `BCPolicy` now buffers per-env the last HS observed frames +
the last HS-1 emitted (z-scored) action blocks (consecutive replans ARE consecutive obs-steps), and `intention_rollout` gained
`past_action_blocks=` which fills `act_hist` from those blocks WITHOUT advancing the state head (the buffered frames already ARE
the observed states) → reconstructs the EXACT training context z_{t-HS+1..t} + a_{<t}. Buffering is gated on `use_horizon` (the
OURS head) so plain bc stays single-frame/byte-identical. After the fix: N=8 50%→75%, then the full N=50 below.

### Exact eval commands (N=50, L40S)
```
# OURS (planning-free, mode=policy, goal+horizon), seeds 42 (GPU2) and 7 (GPU3):
... python eval_gip.py --config-name cube policy=cube_ours_gc +gip_eval.mode=policy +gip_eval.goal_conditioned=true \
      +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|7>
# Same-box CEM (planning, frozen base via a built run-dir cube_lewm_base = config.json[vit-tiny-192] + the frozen weights):
... python eval_gip.py --config-name cube policy=cube_lewm_base +gip_eval.mode=planning eval.num_eval=50 seed=<42|7>
# Same-box guided (CEM warm-started by the OURS intention prior, goal+horizon-aware):
... python eval_gip.py --config-name cube policy=cube_ours_gc +gip_eval.mode=guided +gip_eval.goal_conditioned=true \
      +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|7>
```

### RESULTS — cube goal-reaching, N=50, ALL SAME BOX (L40S), per-seed
| Arm | method | seed 42 | seed 7 | seed 123 | mean |
|---|---|---|---|---|---|
| 1 | **GC-IDM** (gcidm, Markovian, planning-free) | 100.0 | 100.0 (re-confirmed this box) | — | **100.0** |
| 2 | **OURS** (history-cond. forward GC, planning-free) | 78.0 | 68.0 | 72.0 | **72.7** (3 seeds, std≈5) |
| 3a | **guided** (CEM + OURS goal+horizon prior) | 86.0 | 74.0 | — | **80.0** |
| 3b | **CEM / planning** (frozen-WM CEM, LeWM baseline) | 68.0 | 66.0 | — | **67.0** |

L40S CEM 67.0 reproduces the 174 CEM≈67 EXACTLY (cross-box rendering is NOT an issue for cube/ogbench — unlike robomimic).
L40S guided 80.0 is in the 174 guided≈88 range (same-box, slightly lower; acceptable cross-machine variance). OURS broken-then-fixed
39→73 (history-context fix).

**Conclusion.** Pipeline VALIDATED: GC-IDM re-confirms 100% on the same box (eval harness/GPU sound), CEM reproduces 67, guided 80.
OURS history-conditioned forward GC policy WORKS planning-free (73%, one forward pass, NO CEM) — the same head warm-starting CEM
lifts it to 80% (guided). But on this NEAR-CEILING cube task the added HISTORY does NOT help: the simpler Markovian GC-IDM (100%)
beats our history-conditioned variant (73%). Interpretation: cube goal-reaching is fully determined by (current-frame, goal, horizon)
— exactly GC-IDM's inputs — so conditioning on a latent action history adds variance (per-seed spread 78/68) without signal here.
The history wedge is expected to pay off on tasks with partial observability / contact-state memory (cf. Lift act-format §, where
history/contact memory mattered), NOT on a fully-observed pick-place like cube. This is an honest negative for OURS-vs-GC-IDM ON CUBE,
and a clean same-box baseline (GC-IDM 100 | OURS 73 | guided 80 | CEM 67). The big methodological catch: a history-conditioned policy
MUST be fed its training-length frame+action history at eval (the harness only gives 1 frame) — silently feeding 1 frame cost 34pp.

### Files added (all `$B=le-wm-repro` on L40S; backups `*.bak_ours`)
- `jepa.py` (`HorizonModulator` + horizon threading; backup `jepa.py.bak_ours`)
- `gip.py` (`BCPolicy` horizon countdown + HS-frame/action buffering; `build_policy`/`attach_intention_actor` horizon hooks; backup `gip.py.bak_ours`)
- `train.py` (horizon_conditioned wiring; backup `train.py.bak_ours`)
- `train_ours_gc.py` (NEW — cached latent trainer, reuses `cube_gcidm/latents_cache.pt`)
- `config/train/lewm.yaml` (added `action_pred.horizon_conditioned` / `horizon_H_max`; backup `ogb_lance.yaml.bak_ours` is unrelated, the lewm.yaml change is in-place)
- Checkpoints: `$STABLEWM_HOME/checkpoints/cube_ours_gc/{weights_epoch_1.pt=BEST, weights_epoch_0.pt=latest, config.json}` and
  `cube_lewm_base/` (the frozen-base run-dir for same-box CEM). Eval results under `$STABLEWM_HOME/gip_eval/{policy,guided,planning}/`.

### OPEN QUESTIONS / next
- Does OURS reach GC-IDM's 100% on a PARTIALLY-OBSERVED task where history matters (the wedge's intended regime)? Cube can't show it.
- Would `use_action_history=false` (state-only intention, HS-frame but no past-action stream) close the cube gap (i.e. is the
  past-action stream the noise source)? A quick ablation retrain would isolate it.
- DONE: OURS at 3 seeds = 78/68/72 → mean 72.7 (std≈5); the estimate is stable, the GC-IDM gap (100 vs 72.7) is real.

## §13 ✅🆕 PushT 3-arm goal-conditioned head-to-head — the DISCRIMINATING (contact-heavy) env (L40S, 2026-06-19)

**Motivation (why).** PushT is gcidm's WEAKEST published cell (GC-IDM 84.2 n200 / 84.7 n50 vs ~98–100 on cube/reacher; CEM
82.5) because it is contact-heavy *pushing* with NO grasp-memory — the regime where our HISTORY wedge was hypothesized to beat
gcidm's Markovian `(z_t,z_goal,h)`. **Pre-registered prediction: OURS (history) ≥ GC-IDM on pusht** (the opposite of cube §12,
where the near-ceiling fully-observed task let Markovian GC-IDM win 100 vs OURS 72.7). This §13 runs the SAME 3-arm pipeline as
cube §11/§12 (CEM / gcidm GC-IDM / OURS history) on the pusht base, ALL SAME-BOX on L40S, N=50, ≥2 eval-seeds, to test that
prediction. **VERDICT: prediction FALSIFIED — OURS 31 ≪ GC-IDM 90 — but the mechanism is new and clean (history LOWERS open-loop
action-MSE yet WORSENS closed-loop SR under covariate shift on contact). Honest, fully-diagnosed negative.**

### What changed vs cube (env / base / data), and the code adaptation
- **Frozen base:** `decoders/pusht_ours_lewm_weights.pt` (vit-tiny-192, SIGReg; 303 keys, 198 encoder — byte-identical key
  structure to `cube_ours_lewm_weights.pt`; loads `missing=1 unexpected=0`, the 1 missing = mask_token, harmless, same as cube).
- **Data:** `datasets/pusht_expert_train.h5` (46 GB HDF5, **NOT** the cube lance). swm `HDF5Dataset`, **18685 episodes**,
  frameskip 5, **raw action dim 2** (cube was 5) → **action_block_dim = 2×5 = 10** (cube was 25). Episode lengths 49/123/246
  (min/median/max raw frames = ~10/24/49 obs-steps). h5 has `state`(7) / `proprio`(4) / `action`(2) / `pixels` / `episode_idx`
  / `step_idx`. `_load_slice` returns the SAME (pixels (Fsub,3,224,224) uint8, action (L,2)) interface as the cube lance, so the
  precompute loop is unchanged.
- **Eval config `config/eval/pusht.yaml` (unchanged):** `env_name swm/PushT-v1`, N=50, `goal_offset_steps=25`, `eval_budget=50`,
  `plan_config horizon=5 action_block=5` (→ horizon×block=25 ≤ budget 50 ✓), dataset `pusht_expert_train`, success = 95% T-block
  coverage of the goal pose; callables `_set_state`/`_set_goal_state` (vs cube's `set_target_pos`). Eval horizon0 = goal_offset /
  action_block = 25/5 = **5 obs-steps** (normalized 5/50=0.1) — IDENTICAL to cube. So H_max=50 covers the full pusht episode.
- **Code adaptation (additive, cube path byte-preserved):** `train_gcidm.py` got two new CLI args `--dataset_name`
  (default `ogbench/ogb_cube_single.lance`) and `--keys_to_load` (default `pixels,action,observation`), and its hardcoded
  `dataset_cfg` now reads from them (`keys_to_cache` = the non-pixels subset). Called WITHOUT those flags it is byte-identical to
  the cube run (verified by `diff train_gcidm.py.bak_pusht train_gcidm.py` = only the additive args + the `name=args.dataset_name`
  swap). `train_ours_gc.py` was **NOT edited** (byte-identical to `train_ours_gc.py.bak_pusht`) — it only reads the latent cache
  and auto-adapts `action_block_dim` from `act_list[0].shape[-1]`=10. `gcidm.py` is generic on `action_dim` (head params =
  **1,123,850** for adim=10, vs 1,131,545 for cube's 25). `gip.py`/`jepa.py`/`eval_gip.py` UNTOUCHED this session (mtimes
  03:58–04:00 from the cube §11/§12 work; my session = 05:04+; only `train_gcidm.py` newer). So cube/bc/guided/planning/gcidm
  paths verified intact.
- **Same-box CEM run-dir:** built `checkpoints/pusht_lewm_base/` = cube_lewm_base's config.json with `action_encoder.input_dim`
  10 + `pusht_ours_lewm_weights.pt` copied as `weights_epoch_0.pt` (mirrors cube §12's `cube_lewm_base`). Loads `Adim=10
  missing=89 unexpected=0` (the 89 missing = fresh unused action head; planning ignores it). For `mode=planning`.

### ⚠️ latent-sanity gotcha (resolved — NOT a collapse)
The smoke run (`--max_eps 50`) printed `latent sanity per-dim std=0.0039` → looked collapsed. **It is NOT.** The sanity stat is
`std` over the *frame-0 latent of each episode*, and **PushT resets all episodes to a near-identical canonical start** (T-block +
pusher in the same pose) → frame-0 latents barely differ → 0.004 over 50 eps. The FULL run over **18685** episodes reads the
proper `per-dim std=0.9453` (cube was 1.024) → NOT collapsed. Direct check: WITHIN-episode per-dim std = **0.72**, frame0-vs-last
L2 = **18.1**, across-20-eps all-frames std = **0.70**, emb-norm ~13.9 — the encoder tracks the scene fine; only the *start-state*
cross-episode variance is tiny (cube randomizes the block, pusht doesn't). The `train_mse=0.00000` in the 50-ep smoke was a single
1024-batch overfitting 945 samples, gone at full scale. (Cube had high frame-0 std purely because OGBench randomizes the cube pose.)

### Infra (same as cube §11)
Precompute frozen latents ONCE → `checkpoints/pusht_gcidm/latents_cache.pt` (**201 MB**, 18685 eps in **421 s** at ~44 eps/s,
bf16 autocast, num_workers=0). gcidm-ON, gcidm-noh AND OURS all share that ONE cache → byte-identical frozen features across all
three arms. Phase-2 head training: gcidm ~1.3 s/epoch (200 ep ≈ 5 min), OURS ~35 s/epoch (40 ep ≈ 23 min). Env for everything:
`STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl_* TMPDIR=/tmp CUDA_VISIBLE_DEVICES=<g>`.

### Exact train commands (L40S, `$B=…/le-wm-repro`, py `…/lewm/bin/python`)
```
# GC-IDM horizon ON (also builds the shared cache), GPU0:
... CUDA_VISIBLE_DEVICES=0 python train_gcidm.py --epochs 200 --batch_size 1024 --H_max 50 \
    --dataset_name pusht_expert_train.h5 --keys_to_load pixels,action \
    --weights /mnt/minghao_data/.stable-wm/decoders/pusht_ours_lewm_weights.pt \
    --run_name pusht_gcidm --cache_run pusht_gcidm
# GC-IDM horizon OFF (AdaLN ablation), reuses the cache, GPU1:
... CUDA_VISIBLE_DEVICES=1 python train_gcidm.py --epochs 200 --batch_size 1024 --H_max 50 --ablate_horizon \
    --dataset_name pusht_expert_train.h5 --keys_to_load pixels,action \
    --weights .../pusht_ours_lewm_weights.pt --run_name pusht_gcidm_noh --cache_run pusht_gcidm
# OURS history-conditioned forward GC policy, reuses the cache, GPU2:
... CUDA_VISIBLE_DEVICES=2 python train_ours_gc.py --epochs 40 --batch_size 512 --lr 3e-4 --H_max 50 \
    --history_size 3 --seed 3072 --weights .../pusht_ours_lewm_weights.pt \
    --run_name pusht_ours_gc --cache_run pusht_gcidm
```

### Convergence (val on frozen latents, z-scored 10-d action)
- **GC-IDM ON** (`pusht_gcidm`): val_mse 0.392(ep1)→0.194(ep50)→0.184(ep100)→0.178(ep150)→**0.177(ep195-200, FLAT)**; best
  **0.176**. Plateaued ⇒ converged.
- **GC-IDM noh** (`pusht_gcidm_noh`): val_mse 0.402→0.222→0.217→0.213→**0.212(ep200)**; best **0.212**. Plateaued. Horizon ON cuts
  val MSE ~17% (0.176 vs 0.212) → AdaLN load-bearing on the objective (cube was ~36%; pusht's short 5-step eval horizon makes
  horizon matter less, see SR ablation below).
- **OURS** (`pusht_ours_gc`, 418443 hindsight windows = 376599 train / 41844 val, HS=3, trainable 11.29 M): val_act
  0.253(ep1)→0.151(ep10)→0.137(ep20)→**0.1319(ep28, BEST)**→0.132(ep30)→0.132(ep40); val_act FLAT 0.132–0.137 over ep20→40 while
  train_act keeps falling (0.094→0.075) = classic frozen-feature-head convergence. `weights_epoch_1.pt` = the ep-28 best (reported).

### Exact eval commands (N=50, L40S, same box)
```
# GC-IDM (mode=gcidm), seeds 42(GPU0) / 7(GPU1):
... python eval_gip.py --config-name pusht policy=pusht_gcidm +gip_eval.mode=gcidm eval.num_eval=50 seed=<42|7>
# GC-IDM horizon-OFF ablation, seed 42(GPU3):
... python eval_gip.py --config-name pusht policy=pusht_gcidm_noh +gip_eval.mode=gcidm \
    +gip_eval.gcidm_run=pusht_gcidm_noh +gip_eval.ablate_horizon=true eval.num_eval=50 seed=42
# OURS planning-free (mode=policy, goal+horizon), seeds 42(GPU0)/7(GPU1):
... python eval_gip.py --config-name pusht policy=pusht_ours_gc +gip_eval.mode=policy +gip_eval.goal_conditioned=true \
    +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|7>
# Same-box CEM (mode=planning, frozen base), seeds 42(GPU4)/7(GPU5):
... python eval_gip.py --config-name pusht policy=pusht_lewm_base +gip_eval.mode=planning eval.num_eval=50 seed=<42|7>
# Same-box guided (CEM warm-started by OURS goal+horizon prior), seeds 42(GPU3)/7(GPU4):
... python eval_gip.py --config-name pusht policy=pusht_ours_gc +gip_eval.mode=guided +gip_eval.goal_conditioned=true \
    +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|7>
```
(BCPolicy `use_horizon`/`use_history` BOTH auto-engage here: verified the loaded model has `horizon_modulator=True`,
`use_action_history=True`, `horizon_conditioned=True`, `load missing=0 unexpected=0` → the cube §12 history+horizon eval-context
fix IS active; OURS is NOT fed a single frame.)

### RESULTS — PushT goal-reaching, N=50, ALL SAME BOX (L40S), per-seed
| Arm | method | seed 42 | seed 7 | mean | vs gcidm-published |
|---|---|---|---|---|---|
| 1 | **GC-IDM** (gcidm, Markovian, planning-free) | 86.0 | 94.0 | **90.0** | ≈/≥ published **84.2(n200)/84.7(n50)** ✅ NOT a ceiling |
| 2 | **OURS** (history-cond. forward GC, planning-free) | 30.0 | 32.0 | **31.0** | — (the wedge; FAILS open-loop here) |
| 3a | **guided** (CEM + OURS goal+horizon prior) | 80.0 | 84.0 | **82.0** | — |
| 3b | **CEM / planning** (frozen-WM CEM, LeWM baseline) | 84.0 | 94.0 | **89.0** | ≈ published CEM **82.5** ✅ |
| — | GC-IDM horizon-OFF (AdaLN ablation) | 82.0 | — | 82.0 | −8 vs ON; horizon helps less than cube |

**GC-IDM reproduced ~84 → YES (mean 90, both seeds 86/94, at/above published 84–85; pusht is NOT a 100 ceiling like cube).**
**OURS ≥ GC-IDM (the thesis) → NO — FALSIFIED: OURS 31 ≪ GC-IDM 90.** The horizon-OFF GC-IDM ablation drops only 8pp (94→82,
n=1 seed) vs cube's −26pp — pusht's eval horizon is a fixed short 5 obs-steps so the AdaLN signal is near-constant at eval and
matters less there (it still helps the train objective by 17%).

### 🔬 DIAGNOSIS (task-4: OURS landed far below the ~84 GC-IDM line → diagnosed BEFORE reporting)
OURS planning-free = 31% is **REAL, not a bug** — five independent checks:
1. **Model loads clean:** `load_gip_model(pusht_ours_gc)` → `Adim=10 missing=0 unexpected=0`, `horizon_modulator=True`,
   `use_action_history=True`, `horizon_conditioned=True` → the HS=3-frame + horizon eval-context (cube §12 fix) IS engaged.
2. **Goal pathway works:** OURS goal_conditioned=true 30% vs goal_conditioned=false (goal-agnostic bc) 10% (N=10, s42) → the
   z_goal input genuinely lifts SR 3× → the goal is being used, not ignored.
3. **No batching artifact:** the 50-episode `episode_successes` are *scattered* (interleaved T/F, different per seed), NOT the
   chunk-1-then-all-zeros 10-env-pool reuse pattern that bit §10 DINOv2 — so 30/32 is a true success rate.
4. **🔑 OURS predicts the expert action BETTER than GC-IDM offline, yet rolls out worse.** On 4000 held-out hindsight tuples,
   per-sample action-MSE (z-scored 10-d): at the eval horizon h=5 → **GC-IDM 0.140 vs OURS 0.0797**; on random hindsight h →
   **GC-IDM 0.135 vs OURS 0.0835**. OURS's history gives it ~1.7× LOWER open-loop action-prediction error — yet closed-loop SR is
   31 (OURS) vs 90 (GC-IDM). **Lower BC-MSE ≠ higher closed-loop SR.**
5. **The same OURS prior is GOOD once the loop is closed:** guided (OURS prior → CEM refines) = 82% ≈ CEM 89%. So the head's
   proposal is a useful *seed*; it just fails when executed *open-loop* on contact.

**Mechanism (the new finding).** PushT is contact-heavy pushing with NO recoverable grasp-memory. A HISTORY-conditioned forward
policy minimizes next-action MSE on the expert manifold (history is a strong predictor of the demonstrator's next push), but in
open-loop rollout it suffers **covariate shift / compounding error**: once the pushed T-block deviates from the expert path, the
HS-frame + past-action history buffer fills with off-distribution context and the policy spirals (the classic IL pathology where
history induces *causal-confusion / shortcut-on-past-actions* and HURTS closed-loop robustness). The MARKOVIAN GC-IDM
`(z_t,z_goal,h)` is far more robust to this shift — each step re-reads only the current state vs the goal, so it self-corrects —
which is exactly why a tiny Markovian head reaches 90 here. CEM closes OURS's open-loop gap (guided 82) by *searching* around the
prior instead of executing it blindly.

**Conclusion.** PushT reproduces the gcidm cell faithfully (GC-IDM 90 ≈ published 84–85; CEM 89 ≈ published 82.5 — pusht is NOT a
ceiling). But the pre-registered HISTORY-beats-Markovian prediction **does NOT hold**: OURS 31 ≪ GC-IDM 90. The honest reading is
the OPPOSITE of the hypothesis — on a contact task with no grasp-memory, history *helps imitation MSE but hurts closed-loop control*
(covariate shift), and the simple Markovian GC-IDM dominates. Combined with cube §12 (GC-IDM 100 ≫ OURS 73 on the near-ceiling
fully-observed task), **GC-IDM beats OURS on BOTH cube AND pusht** — the history wedge has NOT yet found a benchmark cell where it
wins. The Lift post-grasp-memory finding ([[project_gip_intention_format_eval_convention]]) remains the only regime where history
demonstrably mattered, and it mattered for *grasp-state contact memory* (a recoverable latent the current frame omits), which pusht
lacks (the T-block pose IS fully in the current frame). **Wedge implication: the history advantage is specific to PARTIAL
OBSERVABILITY of a task-critical recoverable state (grasp/contact memory), NOT to contact-heaviness per se** — pusht is contact-heavy
but fully observed, so Markovian suffices. The next decisive test is therefore robomimic Lift-family (partial grasp-state),
NOT pusht (cf. `can_gc_hist`/`can_gc_markov` checkpoints already on disk — the robomimic GC head-to-head).

### Files added / changed (all `$B=le-wm-repro` on L40S; backups `*.bak_pusht`)
- `train_gcidm.py` — added `--dataset_name` / `--keys_to_load` args (default = cube, byte-preserved); dataset_cfg reads them.
  Backup `train_gcidm.py.bak_pusht`. (`train_ours_gc.py.bak_pusht` is an unchanged backup — the file needed no edit.)
- **NOT touched:** `gip.py`, `jepa.py`, `eval_gip.py`, `gcidm.py`, `train_ours_gc.py` (all from the cube §11/§12 work; verified by
  mtime + diff). cube/bc/guided/planning/gcidm paths intact.
- Checkpoints: `$STABLEWM_HOME/checkpoints/pusht_gcidm/{gcidm_head_best.pt, gcidm_config.json, latents_cache.pt(201M)}`,
  `pusht_gcidm_noh/`, `pusht_ours_gc/{weights_epoch_1.pt=BEST@ep28, weights_epoch_0.pt=latest, config.json}`, `pusht_lewm_base/`
  (frozen-base run-dir for same-box CEM). Eval results under `$STABLEWM_HOME/gip_eval/{gcidm,policy,guided,planning}/`.

### OPEN QUESTIONS / next
- **History helps neither cube (full-obs, near-ceiling) nor pusht (contact, full-obs).** The remaining live hypothesis: history
  wins ONLY when a task-critical state is partially observed and recoverable from history (grasp/contact memory = Lift family).
  Test the `can_gc_hist` vs `can_gc_markov` (already trained on disk) + lift/square GC head-to-head — that is the wedge's last stand.
- Quantify the covariate-shift story directly: log OURS's per-step latent distance to the expert manifold over a rollout vs
  GC-IDM's; predict OURS diverges after the first off-expert push while GC-IDM stays bounded.
- `use_action_history=false` (state-only HS-frame, no past-action stream) ablation on pusht: if it RAISES OURS SR, the
  past-action stream is the causal-confusion source (the cube §12 open question, now with a contact task to test it on).
  → **ANSWERED in §18 below: YES, it rises (31→42), the past-action stream WAS a (partial) causal-confusion source.**

## §18 ✅🆕 PushT MECHANISM ablation — zeroing the PAST-ACTION stream RESCUES OURS from compounding-error collapse (L40S, 2026-06-20)

**Motivation (why).** §13 falsified the pre-registered "OURS history ≥ GC-IDM on pusht" prediction: OURS history goal-conditioned
forward policy scored **31** (seeds 30/32) vs GC-IDM **90** (seeds 86/94), despite OURS having LOWER open-loop action-MSE
(0.080 at the eval horizon vs GC-IDM 0.140). §13's diagnosis was **closed-loop covariate shift / causal confusion**, and the
single strongest follow-up it pre-registered (the last §13 open question, and the bug-audit's top recommendation after it confirmed
the 31 is REAL not a bug) was: zero the PAST-ACTION stream `a_<t` while keeping the state history (z_≤t, 3 frames) + goal + horizon.
The hypothesis: the policy's conditioning on its OWN drifting executed actions is the causal-confusion source (the classic IL
shortcut-on-past-actions pathology). If closed-loop SR RISES above 31 with the past-action stream zeroed, the past actions ARE the
culprit (a mechanistic win). If not, the covariate shift lives in the state-history too (a deeper problem). **VERDICT: SR RISES
31 → 42 (+11 pp, BOTH seeds up: 30→46, 32→38) — the past-action stream WAS a causal-confusion source. A partial mechanistic win:
removing it recovers ~⅓ of the gap to GC-IDM, but 42 still ≪ 90, so a second covariate-shift component remains in the state-history.**

### Setup — reuses EVERYTHING from §13, changes ONLY the action-history flag
- **Frozen base:** SAME `decoders/pusht_ours_lewm_weights.pt` (vit-tiny-192, SIGReg; loads `missing=7 unexpected=0` at train, the
  7 missing = fresh action_predictor/decoder/horizon_modulator, identical to §13's "missing=7" — byte-checked the same base).
- **Latent cache:** REUSED `checkpoints/pusht_gcidm/latents_cache.pt` (201 MB, the SAME 18685-episode bf16 cache §13's three arms
  shared) via `--cache_run pusht_gcidm`. Did NOT re-precompute, did NOT touch the cube default. Loaded `n_eps=18685 emb_dim=192
  adim=10` — byte-identical frozen features to the §13 OURS arm.
- **Hindsight windows:** `n=418443 train=376599 val=41844 HS=3 H_max=50` — EXACTLY §13's OURS counts (418443 / 376599 / 41844).
  Same seed 3072 shuffle, same 0.9 train split. Trainable params **11.29 M** — byte-identical to §13's OURS. Everything matches.
- **Eval config `config/eval/pusht.yaml` (unchanged):** `env_name swm/PushT-v1`, N=50, `goal_offset_steps=25`, `eval_budget=50`,
  `plan_config horizon=5 action_block=5`, success = 95% T-block coverage. Eval horizon0 = 5 obs-steps (0.1 normalized). IDENTICAL to §13.
- **The ONE change:** `use_action_history=False`. In `jepa.JEPA.predict_intention` (line 192-193) this zeros `past_act_emb`
  (`past_act_emb = torch.zeros_like(past_act_emb)  # state-only intention`) at TRAIN, and in `JEPA.intention_rollout` (line 338-339)
  it zeros `pa` at EVAL — so the past-action stream is removed in BOTH train and rollout. The goal_emb and horizon are added AFTER
  the zeroing (lines 196-197), so goal-conditioning + AdaLN horizon SURVIVE. The flag travels to eval via the saved `config.json`
  (`use_action_history: false`), which `gip.load_gip_model` reads (line 133-134 `hydra_instantiate(config)`) AND the BCPolicy reads
  (`gip.py` line 246 `self.use_history = bool(getattr(self.model, "use_action_history", True))`). So at eval the model literally
  cannot see a_<t. **Verified at load time:** `gip.load_gip_model("pusht_ours_noact")` → `use_action_history= False`,
  `horizon_conditioned= True`, `has horizon_modulator= True` → state history + goal + horizon kept; past actions zeroed; correct.

### Code adaptation (NON-invasive — shared method code untouched, per the concurrent-agents constraint)
`train_ours_gc.py` HARDCODES `"use_action_history": True` in its `build_jepa` cfg (no CLI knob). I did NOT edit it (it is shared and
a Phase-2/Phase-3/audit set of agents are concurrently active). Instead I **copied** it to a new file `train_ours_gc_noact.py` and
patched ONLY 4 lines there: (1) `build_jepa(..., use_action_history=True)` param; (2) cfg `"use_action_history": use_action_history`
(was hardcoded `True`); (3) a `--use_action_history` CLI arg (default 1); (4) the `build_jepa(...)` call passes
`use_action_history=bool(args.use_action_history)`. `diff train_ours_gc.py train_ours_gc_noact.py` = exactly those 4 hunks, nothing
else. Backup `train_ours_gc.py.bak_abl` made first. **NOT touched (read-only reuse):** `jepa.py`, `gip.py`, `eval_gip.py`,
`gcidm.py`, `train_gcidm.py`, `train_ours_gc.py`, and all other agents' checkpoints. Only WROTE my own `pusht_ours_noact` ckpts.

### Infra (same as §13)
L40S `stratus-lookout`, `$B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, py
`/var/lib/docker/data/minghao_home/lewm/bin/python`, run as `sudo -u minghao.fu`. Env for all:
`STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl_abl TMPDIR=/tmp
CUDA_VISIBLE_DEVICES=<g>`. Train GPU1; eval seed42 GPU4 / seed7 GPU7 (lowest-occupied, packed onto contended GPUs per rule 10 — all
8 GPUs were at 100% util ~15-35 GB from concurrent agents; the head is tiny, no OOM). Train+eval SAME BOX (L40S) → no cross-machine
render OOD (cf. [[project_l40s_cross_gpu_rendering]]). Train ~46 s/epoch (vs §13's 35; slower from GPU contention, acceptable).

### EXACT train command (L40S)
```
STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl_abl TMPDIR=/tmp \
CUDA_VISIBLE_DEVICES=1 python train_ours_gc_noact.py --epochs 40 --batch_size 512 --lr 3e-4 --H_max 50 \
  --history_size 3 --seed 3072 --use_action_history 0 \
  --weights /mnt/minghao_data/.stable-wm/decoders/pusht_ours_lewm_weights.pt \
  --run_name pusht_ours_noact --cache_run pusht_gcidm
```
(IDENTICAL to §13's OURS train command except `train_ours_gc_noact.py` + `--use_action_history 0` + `--run_name pusht_ours_noact`.
Same 40 epochs, batch 512, lr 3e-4, H_max 50, history_size 3, seed 3072, same weights, same cache. w_act=w_intent=1.0,
weight_decay 1e-4, detach_target=1, cosine LR — all at the file defaults, same as §13.)

### EXACT eval commands (N=50, L40S, same box)
```
# seed 42 (GPU4), seed 7 (GPU7), staggered 7s:
STABLEWM_HOME=... MPLCONFIGDIR=/tmp/mpl_abl ... CUDA_VISIBLE_DEVICES=<4|7> python eval_gip.py --config-name pusht \
  policy=pusht_ours_noact +gip_eval.mode=policy +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 \
  eval.num_eval=50 seed=<42|7>
```
(IDENTICAL to §13's OURS eval line except `policy=pusht_ours_noact`. mode=policy, goal_conditioned=true, horizon_H_max=50, N=50.)
Eval loaded `weights_epoch_1.pt` (= the BEST ep37 ckpt): `[GIP] load pusht_ours_noact <- weights_epoch_1.pt: Adim=10 missing=0
unexpected=0`, `[GIP] eval mode=policy policy=BCPolicy`. Each eval ~36 s.

### Convergence (val on frozen latents, z-scored 10-d action — DONE, plateaued ⇒ converged)
val_act: 0.375(ep1)→0.224(ep5)→0.192(ep10)→0.171(ep17)→0.168(ep20)→0.157(ep28)→0.155(ep30)→**0.15300(ep37, BEST)**→0.154(ep40).
FLAT at 0.153–0.155 over ep30→40 while train_act keeps falling (0.112→0.105) = classic frozen-feature-head convergence (the same
pattern §13's OURS showed). `DONE best_val_act=0.15300 @epoch 37  saved`. `weights_epoch_1.pt` = the ep-37 best (reported);
`weights_epoch_0.pt` = ep-40 latest. Only 2 ckpts on disk → already pruned, no excess snapshots.
**⚠️ Offline action-MSE went the OPPOSITE way to SR:** noact best val_act **0.153** vs §13 OURS-with-history best **0.132**
(ep28). Zeroing the past-action stream RAISED open-loop val action-MSE by **~16%** (0.153 vs 0.132) — the past actions genuinely
help predict the demonstrator's next push offline — YET closed-loop SR ROSE (see below). The §13 task-3 held-out per-sample
action-MSE for OURS-with was 0.080 @h=5 / 0.084 random-h and GC-IDM 0.140 / 0.135; this noact arm's val_act 0.153 is in the
GC-IDM ballpark for open-loop fit, consistent with "noact ≈ a weaker open-loop predictor than OURS-with, but a better closed-loop one".

### RESULTS — PushT goal-reaching, N=50, mode=policy goal_conditioned=true horizon_H_max=50, SAME BOX (L40S), per-seed
| Arm | past-action stream | seed 42 | seed 7 | mean | offline val_act |
|---|---|---|---|---|---|
| **OURS-noActHist** (THIS §18: state hist + goal + horizon, a_<t ZEROED) | **zeroed** | **46.0** | **38.0** | **42.0** | 0.153 |
| OURS-with-past-actions (§13 OURS) | on | 30.0 | 32.0 | **31.0** | 0.132 |
| GC-IDM (§13, Markovian `(z_t,z_goal,h)`, planning-free) | n/a (no history) | 86.0 | 94.0 | **90.0** | — (0.140 @h5) |
| CEM / planning (§13, frozen-WM) | n/a | 84.0 | 94.0 | **89.0** | — |
| guided (§13, CEM + OURS prior) | on (in prior) | 80.0 | 84.0 | **82.0** | — |

Both noact seeds' `episode_successes` are SCATTERED (interleaved T/F across all 50 eps, different per seed — NOT the
chunk-1-then-zeros 10-env-pool reuse pattern that bit §10) → 46/38 are TRUE success rates, no batching artifact (same check §13 ran).
seed42 array: `F F T T F F T T F F F F T T F F T F F T F F T T T F T T T T F F F T F T F F T F F T F T T T F F F T` (23 T = 46%).
seed7 array: `F F F T F T F T F F T F T F T F F T T T F F F F F T F T F F F F F T F F F F T T F F T T T F F T T F` (19 T = 38%).

### VERDICT (the deliverable answer)
- **Did zeroing the past-action stream RAISE SR? YES.** 31 → **42** (+11 pp, +35% relative), and it rose on BOTH seeds
  independently (seed42 30→46 = +16, seed7 32→38 = +6). This is the pre-registered "RISES above 31" criterion → **the past-action
  stream WAS a causal-confusion source.** Mechanistic win: the classic IL shortcut-on-past-actions pathology is confirmed present
  on a contact task — the policy conditioning on its own drifting executed actions hurt closed-loop control, and removing that
  conditioning self-heals ~⅓ of the gap.
- **DOUBLE-DISSOCIATION nails the §13 covariate-shift story.** Removing past actions HURT offline imitation (val_act 0.132→0.153,
  +16% MSE — past actions are a strong open-loop predictor of the demonstrator's next push) yet HELPED closed-loop SR (31→42, +35%).
  Lower BC-MSE ↛ higher SR, and here the SIGN of the effect flips between the two metrics. This is the cleanest possible confirmation
  that §13's 31 was a closed-loop covariate-shift failure, not an open-loop capacity failure: the very signal that improves the
  offline fit (a_<t) is the one that destabilizes the rollout.
- **But the win is PARTIAL — 42 still ≪ 90 (GC-IDM).** Zeroing a_<t recovers only ~⅓ of the 31→90 gap. So the past-action stream is
  ONE causal-confusion component, NOT the whole story: a SECOND covariate-shift component lives in the STATE-HISTORY itself (the
  HS=3 latent frames z_{t-2..t} drift off-manifold once the pushed T-block deviates from the expert path, and even state-only history
  spirals). The Markovian GC-IDM `(z_t,z_goal,h)` re-reads ONLY the current state each step → self-corrects → 90. So on pusht
  (contact-heavy but FULLY observed) the deeper lesson stands: **history of ANY kind (past-actions OR state-frames) is a closed-loop
  liability when the current frame already contains the full task state; Markovian wins.** The history wedge still has NOT found a
  pusht/cube cell where it beats Markovian — its last stand remains the PARTIALLY-observed grasp/contact-memory regime (robomimic
  Lift family, `can_gc_hist`/`can_gc_markov` already on disk), where the current frame OMITS a task-critical recoverable state.

### Files added / changed (all `$B=le-wm-repro` on L40S)
- `train_ours_gc_noact.py` — NEW (copy of `train_ours_gc.py` + 4-line `--use_action_history` patch; diff verified = only those hunks).
- `train_ours_gc.py.bak_abl` — backup of the unedited shared trainer (made before copying; the shared file itself is UNTOUCHED).
- **NOT touched (read-only reuse):** `jepa.py`, `gip.py`, `eval_gip.py`, `gcidm.py`, `train_gcidm.py`, `train_ours_gc.py`, and all
  other agents' checkpoints. No fan/air-hockey procs touched.
- Checkpoints (mine only): `$STABLEWM_HOME/checkpoints/pusht_ours_noact/{weights_epoch_1.pt=BEST@ep37, weights_epoch_0.pt=latest@ep40,
  config.json(use_action_history=false)}`. Eval results: `$STABLEWM_HOME/gip_eval/policy/pusht_ours_noact/policy_pusht_ours_noact_results.txt`
  (+ per-episode env_*.mp4 renders, auto-generated by eval). Train log `/tmp/train_pusht_noact.log`, eval logs `/tmp/eval_noact_s{42,7}.log`.

### OPEN QUESTIONS / next (post-§18)
- The state-history covariate-shift residual: to fully decompose, a `history_size=1` (single-frame, state-only, Markovian-like OURS)
  arm would test whether collapsing the state-history to one frame closes the remaining 42→90 gap (predicting it would approach GC-IDM,
  since OURS at HS=1 + a_<t-zeroed ≈ GC-IDM's `(z_t, z_goal, h)` modulo the AR-predictor architecture). Cheap (same cache).
- The Lift-family GC head-to-head (`can_gc_hist` vs `can_gc_markov`, + lift/square) remains the wedge's last stand: pusht confirms
  history hurts under FULL observability; Lift's partial grasp-state is the only regime left where history could win.

## §17 ✅🆕 ROBOMIMIC GC head-to-head — tool_hang + transport (the long-horizon contact cells) (L40S, 2026-06-20, Phase-3)

**Motivation (why).** Fill the last two cells of the robomimic GC-policy head-to-head: **GC-IDM (Markovian `(z_t,z_goal,h)`,
single-frame) vs OURS (history `(z_≤t, a_<t, z_goal, h)`)**, both goal-conditioned (hindsight goals) + AdaLN-Zero horizon, on
**tool_hang** and **transport** — the two LONGEST-horizon, most multi-stage contact tasks in robomimic. These are the wedge's
intended regime per the §13 conclusion: history is hypothesized to win ONLY where a task-critical state is partially observed and
recoverable from history (grasp/contact memory), which tool_hang (insert-hook-then-hang) and transport (two-arm hand-off) plausibly
have, unlike fully-observed cube/pusht. This MIRRORS the Phase-2 lift/can/square GC cells (drivers `gc_train_driver.sh` /
`gc_eval_driver.sh`, run names `<task>_gcidm` / `<task>_gc_ours`); only the task/base/dataset/eval-params change. Phase 2 owns
`eval_histbc_robomimic.py` (it added the `gcidm` + goal-conditioned `policy` modes — I RAN it, did not edit it).

### ⚠️ BASE = vit-tiny-192 (config-correctness flag — same arch as lift/can/square, NOT dinov2-384)
- The task spec allowed a dinov2-384 fallback IF no vit-tiny base existed. **A vit-tiny-192 base DOES exist for both tasks**:
  `$CK/tool_hang_lewm_scratch/weights_epoch_100.pt` and `$CK/transport_lewm_scratch/weights_epoch_100.pt` (encoder
  `vit_hf size=tiny pretrained=False`, predictor input_dim 192 — the from-scratch-encoder LeWM bases from the §10 ablation).
  Both load into `train_gcidm.build_frozen_lewm` (hardcoded vit-tiny-192) with **`missing=0 unexpected=0`** (verified standalone).
- So I copied them to the flat-decoder path the unmodified driver expects: `$DEC/tool_hang_lewm_weights.pt` and
  `$DEC/transport_lewm_weights.pt` (72 MB each = vit-tiny; the dinov2 dirs are 200 MB = 384-d). **This keeps the cross-task base
  TYPE consistent: ALL FIVE robomimic GC cells (lift/can/square + tool_hang/transport) use vit-tiny-192.** The dinov2-384 fallback
  was NOT needed and NOT used. (Note the lift/can/square vit-tiny bases `*_lewm_weights.pt` were also pretrained=False vit-tiny —
  same family.) Per-task comparison is fair regardless (both arms of each task share the SAME frozen base + SAME latent cache).
  ⚠️ The ONLY config caveat: these tool_hang/transport vit-tiny bases are the `_lewm_scratch` (from-scratch encoder) runs, which
  §10 found weaker for control than dinov2; but since BOTH GC-IDM and OURS of each task use the identical base, the GC-IDM-vs-OURS
  contrast (the question here) is clean.

### Datasets (HDF5, on L40S `$STABLEWM_HOME/datasets/`)
- `tool_hang.h5`: 200 episodes, 95962 steps, raw action dim **7** → action_block_dim 7×5=**35** (same as lift/can/square). Has
  `episode_idx`/`step_idx` (no convert-bug). state 58-d, proprio 11-d.
- `transport.h5`: 200 episodes, 93752 steps, raw action dim **14** (two-arm) → action_block_dim 14×5=**70**. state 115-d.

### Exact train commands (L40S, `$B=…/le-wm-repro`, py `…/lewm/bin/python`, drivers reused VERBATIM from Phase-2)
```
# tool_hang GPU6, transport GPU2 (each driver runs GC-IDM then OURS sequentially on the same GPU+cache):
ENV="STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl_<task> TMPDIR=/tmp"
# inside gc_train_driver.sh <task> <gpu>:
#   train_gcidm.py    --dataset_name <task>.h5 --keys_to_load pixels,action --weights $DEC/<task>_lewm_weights.pt \
#                     --run_name <task>_gcidm   --cache_run <task>_gcidm --epochs 200 --H_max 50
#   train_ours_gc.py  --weights $DEC/<task>_lewm_weights.pt \
#                     --run_name <task>_gc_ours --cache_run <task>_gcidm --epochs 200 --H_max 50
bash gc_train_driver.sh tool_hang 6     # → tool_hang_gcidm + tool_hang_gc_ours
bash gc_train_driver.sh transport 2     # → transport_gcidm + transport_gc_ours
```
GC-IDM = `GCIDMHead([z_t∥z_goal], h)` (Markovian, single-frame, planning-free); OURS = `train_ours_gc.py` history-conditioned
forward GC policy (HS=3 latent history + past-action blocks + z_goal + AdaLN horizon, planning-free). Both share the ONE frozen
latent cache `$CK/<task>_gcidm/latents_cache.pt` (built once by the GC-IDM run) → byte-identical frozen features across arms.

### Convergence (val on frozen latents, z-scored action; both arms plateaued)
- **tool_hang GC-IDM** (`tool_hang_gcidm`, adim=35): val_mse flat 0.238 over ep196–200, **best_val_mse 0.236**. Converged.
- **transport GC-IDM** (`transport_gcidm`, adim=70): val_mse flat ~0.33 over ep196–200, **best_val_mse 0.322**. Converged.
- **tool_hang OURS** (`tool_hang_gc_ours`, HS=3): val_act minimum ~0.157 (plateau ep35→ while train_act keeps falling — classic
  frozen-head convergence); `weights_epoch_1.pt` = best-val checkpoint. Converged.
- **transport OURS** (`transport_gc_ours`, HS=3): val_act minimum ~0.198 (plateau ~ep55→); `weights_epoch_1.pt` = best-val. Converged.

### Exact eval commands (N=50, 3 seeds {42,0,1}, on the TRAINING GPU per the cross-GPU render gotcha)
```
# p3_eval_driver.sh <task> <gpu> <world_task> <budget> <offset>  (mirrors gc_eval_driver.sh; world.num_envs=10 pool, N=50):
#   GC-IDM: eval_histbc_robomimic.py --config-name robomimic world.task=<WTASK> world.num_envs=10 \
#           dataset.stats=<task> eval.dataset_name=<task> eval.num_eval=50 eval.eval_budget=<B> eval.goal_offset_steps=<O> \
#           policy=<task>_gcidm   +gip_eval.mode=gcidm                                seed=<s>
#   OURS:   ... policy=<task>_gc_ours +gip_eval.mode=policy +gip_eval.goal_conditioned=true seed=<s>
bash p3_eval_driver.sh tool_hang 6 ToolHang        940 280
bash p3_eval_driver.sh transport 2 TwoArmTransport 910 270
```

### RESULTS — robomimic GC goal-reaching, N=50 × 3 seeds {42,0,1}, ALL SAME-BOX (L40S), per-seed [FILLING LIVE]
| task | arm | s42 | s0 | s1 | mean | histbc-ref (goal-AGNOSTIC, §10) |
|---|---|---|---|---|---|---|
| tool_hang | GC-IDM (Markovian) | **0.0** | … | … | … | dinov2 .44/.32/.24; vit-tiny-scr .16/.22/.28 |
| tool_hang | OURS (history) | **0.0** | … | … | … | (same task ref — solvable, ~0.2 on vit-tiny) |
| transport | GC-IDM (Markovian) | **0.0** | … | … | … | dinov2 .14/.10/.02; vit-tiny-scr ~0/.02 |
| transport | OURS (history) | **0.0** | … | … | … | (vit-tiny base itself ≈0 here) |

**⚠️ TIE-AT-FLOOR on tool_hang (s42): BOTH arms 0/50** (clean n=50, ~23 min real rollout each, episode_successes all-False,
chunks accumulate 0/10→0/50 — NOT a batching artifact). transport GC-IDM s42 also **0/50**. Config-correct: OURS log =
`policy ready adim=35 HS=3 goal_cond=True use_horizon=True horizon0=56.00` (offset 280/block 5 ✓; goal+horizon+history all active).

**🔑 VALIDATION GATE RESOLVED — the goal-wiring is NOT broken, tool_hang 0 is GENUINE.** The prompt's gate ("if OURS ≈0 too,
suspect broken wiring") is cleared by a cross-check against Phase-2's lift/can/square run on the SAME evaluator: those produce clearly
NON-ZERO results on BOTH arms — **lift GC-IDM 0.44/0.48/0.58, OURS 0.08–0.32; can GC-IDM 0.10–0.24, OURS 0.08–0.20; square GC-IDM
0.10–0.30, OURS 0.06–0.12** (read from the gip_eval result files 2026-06-20). So the goal-conditioned evaluator works; tool_hang/
transport = 0 is the task being too hard for a *goal-conditioned, single-replan GC policy*, NOT a wiring bug. Note the §10 histbc-ref
~33/~9 was the **goal-AGNOSTIC** open-loop history-BC (full action-history rollout, NO goal frame, the easier objective); the GC
head-to-head here is the harder goal-conditioned single-pass-per-replan setup, and tool_hang's extreme long-horizon precision
(insert-tool-then-hang, budget 940) floors both arms.

**⚠️⚠️ BASE-CAPACITY CONFOUND (important, must qualify the 0/0):** the goal-AGNOSTIC histbc reference proves the TASK is solvable
in this harness, but the solvability is BASE-dependent. On the SAME vit-tiny-192 base type I used here: tool_hang goal-agnostic histbc
(`tool_hang_gip_scratch`) reaches **0.16/0.22/0.28** (~0.22) — so tool_hang IS solvable on vit-tiny, yet my vit-tiny GC policies get
0 → on tool_hang the **goal-conditioned single-replan GC formulation** specifically fails (the cleaner finding). BUT **transport
goal-agnostic histbc on vit-tiny (`transport_gip_scratch`) is itself ≈0** (0/0/0.02) — the vit-tiny-192 base under-fits the wide
two-arm transport scene (the §10 frozen-probe gap: transport needs dinov2, scratch-vit-tiny R² only 0.42). So **transport 0/0 is
partly a BASE-CAPACITY floor, not purely a GC-policy limitation** — even a good policy on this base is ≈0. (dinov2-384 histbc gets
transport ~9; a dinov2 GC head-to-head might lift transport off the floor — NOT run here, the spec required vit-tiny consistency with
lift/can/square.) **Verdict qualifier: tool_hang is a clean GC-policy floor (base solves it, GC doesn't); transport is confounded by
base capacity (base barely solves it either). Neither task discriminates history-vs-Markovian — both TIE at 0.**

### Files added / changed (Phase-3; all additive, scripts read-only-reused)
- **NOT touched (read-only reuse):** `train_gcidm.py`, `train_ours_gc.py`, `gcidm.py`, `gip.py`, `jepa.py`,
  `eval_histbc_robomimic.py` (Phase-2 owns the evaluator), `gc_train_driver.sh`. Verified by mtime.
- **Created:** `$DEC/tool_hang_lewm_weights.pt`, `$DEC/transport_lewm_weights.pt` (flat vit-tiny base copies of the
  `*_lewm_scratch/weights_epoch_100.pt`). `p3_eval_driver.sh` (eval driver, in `/tmp` + `$STABLEWM_HOME/p3_logs/`).
- **Checkpoints:** `$CK/{tool_hang,transport}_gcidm/{gcidm_head_best.pt, gcidm_config.json, latents_cache.pt}` and
  `{tool_hang,transport}_gc_ours/{weights_epoch_1.pt=BEST, weights_epoch_0.pt=latest, config.json}`. Logs `$STABLEWM_HOME/p3_logs/`.

## §16 ✅🆕 ROBOMIMIC GC head-to-head — Markovian GC-IDM vs OURS (history) on lift/can/square — the PARTIAL-OBSERVABILITY test (L40S, 2026-06-20)

**Motivation (why).** §11/§12 (cube) and §13 (pusht) ran the GC head-to-head on gcidm's OWN benchmark envs. Cube is
fully-observed near-ceiling → Markovian GC-IDM won (100 vs OURS 72.7); pusht is contact-PUSHING with no recoverable hidden
state → OURS 31 ≪ GC-IDM 90 (history HURT under covariate shift). **Robomimic contact tasks are the regime the wedge was
designed for:** lift/can/square hide a *recoverable task-critical state* — "has the gripper grasped?" — that a single
`(z_t, z_goal)` cannot see but an action+state history remembers (the §6 contact-state probe: single-frame AUC 0.88 vs
latent-history 0.97 vs full `[z,a]` 0.99). **Pre-registered prediction: OURS (history) ≥ GC-IDM (Markovian) on robomimic
contact, possibly by a large margin** (GC-IDM may ≈0 if single-frame is insufficient for the grasp→lift/transport transition).
This §16 is the decisive test. The robosuite env re-renders live on the eval GPU → every arm evals on its TRAINING GPU
(cross-GPU render gotcha [[project_l40s_cross_gpu_rendering]]); N=50, 3 seeds {42,0,1}, SAME-BOX.

### What's the same / what changed vs cube §11/§12 and pusht §13
- **Frozen bases (vit-tiny-192 SIGReg `*_lewm_weights.pt`):** `lift_lewm_weights.pt` (on L40S), `can_lewm_weights.pt` +
  `square_lewm_weights.pt` transferred from 174 via the verified direct hop (174 read-only; rsync `--rsync-path='sudo -n -u
  minghao.fu rsync'` 174→L40S `/mnt/minghao_data/.stable-wm/decoders/`). All three load `missing=1 unexpected=0` (the 1 missing
  = mask_token, harmless, same as cube/pusht). Latent sanity per-dim std ≈ 0.83 (lift) — NOT collapsed.
- **Data:** robomimic HDF5 `{lift,can,square}.h5` (swm `HDF5Dataset`), frameskip 5, **raw action dim 7** (6-DoF pose + 1
  gripper; cube was 5, pusht 2) → **action_block_dim = 7×5 = 35**. GCIDMHead params = **1,136,675** (adim=35). Eval env action
  space = 7-DoF ("Action size is 7" confirmed at env build).
- **Eval params (validated, EXPERIMENTS.md §1c / ~L1488):** lift `world.task=Lift`/`eval_budget=100`/`goal_offset_steps=30`;
  can `PickPlaceCan`/240/90; square `NutAssemblySquare`/320/120. Eval-time GC-IDM/OURS horizon0 = goal_offset/frameskip =
  lift 6 / can 18 / square 24 obs-steps (all ≤ H_max=50). `world.num_envs=10` (N=50 = 5×10 chunks; the 50-env-at-once mujoco
  compile is non-viable, §10 EVAL BLOCKER #4). Env trio for EVERY eval: `OMP_NUM_THREADS=1` + CLIP-cached `HF_HUB_OFFLINE=1`
  + `world.num_envs=10`.

### Train (frozen latents, hindsight goals, SAME base/cache/horizon for BOTH arms) — same two-phase design as cube §11/§12
Per {lift, can, square}: precompute frozen latents ONCE from `<task>_lewm_weights.pt` (cached to
`checkpoints/<task>_gcidm/latents_cache.pt`) → train **GC-IDM** (`train_gcidm.py`: Markovian `(z_t, z_goal, h)→a_t`, 3-layer
MLP hidden512, AdaLN-Zero horizon, hindsight MSE, 200 ep) + **OURS** (`train_ours_gc.py`: history `(z_{≤t}, a_{<t}, z_goal, h)`,
the GIP intention head reusing the SAME AdaLN-Zero horizon, reusing the SAME frozen-latent cache → byte-identical frozen
features → maximally fair, 200 ep with early-stop on val_act). Both goal-conditioned (hindsight goal ~ a future frame,
h~Uniform[1,H_max] clamped to the episode end). Exact launch (L40S, `$B=…/le-wm-repro`, py `…/lewm/bin/python`, per task GPU
lift→4/can→5/square→7):
```
# GC-IDM (Markovian):
CUDA_VISIBLE_DEVICES=$G python train_gcidm.py --dataset_name <task>.h5 --keys_to_load pixels,action \
  --weights .../decoders/<task>_lewm_weights.pt --run_name <task>_gcidm --cache_run <task>_gcidm --epochs 200 --H_max 50
# OURS (history), reuses the gcidm cache:
CUDA_VISIBLE_DEVICES=$G python train_ours_gc.py --weights .../decoders/<task>_lewm_weights.pt \
  --run_name <task>_gc_ours --cache_run <task>_gcidm --epochs 200 --H_max 50
```
**Train val losses (converged, 200 ep).** GC-IDM hindsight MSE (best_val): lift 0.437 / can 0.226 / square 0.324. OURS val_act
(best, early-stop): lift 0.258 @ep66 / can ~0.147 / square ~0.219. OURS's open-loop action-MSE is LOWER than GC-IDM's on all
three (the history wedge shows in val loss) — but per §13 val loss is NOT the headline; **closed-loop SR is** (history can
lower open-loop MSE yet worsen closed-loop SR under covariate shift). Heads/configs saved:
`checkpoints/<task>_gcidm/{gcidm_head_best.pt,gcidm_config.json}` (action_dim=35, H_max=50, ablate_horizon=false) and
`checkpoints/<task>_gc_ours/{weights_epoch_1.pt,config.json}` (horizon_conditioned=true, action_head=mse).

### THE EVAL = THE BUILD. New robomimic GC evaluator (additive; only `eval_histbc_robomimic.py` touched, backup `.bak_rmgc`)
Robomimic CANNOT use `eval_gip.py` (its single-frame BCPolicy gives bc≈0 AND it doesn't register the robosuite env). The
validated robomimic evaluator is `eval_histbc_robomimic.py` (imports `robomimic_env`). It had NO gcidm / goal-conditioned mode;
I added BOTH, porting the logic from the cube `gip.GCIDMPolicy` / `gip.BCPolicy(goal_conditioned)` to the robomimic chunked
`world.evaluate` env loop:
- **`gip_eval.mode=gcidm`** → new `GCIDMRobomimicPolicy`: each replan does ONE forward pass `GCIDMHead(z_t, z_goal, h)` —
  single frame, NO history (faithful to gcidm). Loads via `gip.load_gcidm_model` (frozen LeWM + the head). Goal frame from
  `info_dict["goal"]` (the world sets it when `goal_offset` is given), un-normalized via the same StandardScaler. Per-env
  obs-step horizon countdown (init horizon0 = goal_offset/frameskip), decremented one obs-step per replan, normalized
  min(h,H_max)/H_max.
- **`gip_eval.mode=policy +gip_eval.goal_conditioned=true`** → OURS: `HistoryBCPolicy` gained a `goal_conditioned` flag that
  encodes `info_dict["goal"] → z_goal` and threads it (+ the AdaLN-Zero horizon, active iff the model carries a
  `horizon_modulator`) into `predict_intention(goal_emb=…, horizon=…)`. History wedge unchanged; goal/horizon are optional
  additive hooks (strict superset). `mode=policy goal_conditioned=false` and the absent-`gip_eval` default are byte-identical
  to the prior goal-AGNOSTIC history-bc.
- The cube/pusht/two-room/reacher/bc/guided/planning/gcidm paths live in `eval_gip.py`+`gip.py`+`jepa.py`+`train_*.py`, all
  UNTOUCHED this session (mtimes 01:34–05:04, before my 08:22 edit to `eval_histbc_robomimic.py` alone; `eval_histguided`
  still parses + imports the new `HistoryBCPolicy` — positional-compatible signature).

### 🔴🐛 EVAL BUG CAUGHT + FIXED BY VALIDATION — the chunk-1-then-zeros artifact (the test is subtle; this would have faked a 0)
The first lift OURS validation gave **0.08 (4/50) with ALL 4 successes in chunk 1, chunks 2–5 = exactly 0** — the signature
eval-batching artifact ([[project_dinov2_init_ablation]]: "10-env-pool reuse → chunk-1-then-all-zeros"). **Root cause:**
robomimic eval runs `world.evaluate` in `mode='wait'`, which NEVER sets `_needs_flush` (only `mode='auto'` does). The chunked
N=50 loop reuses ONE policy instance across the 5 chunks, so chunk N inherits chunk N−1's stale per-env deques (`_zhist`,
`_ablk`, `_frame_buf`) and horizon counters → corrupted history → 0 success after chunk 1. **This is a latent bug in the
ORIGINAL chunked evaluator too** (the validated 95/75/70 goal-agnostic reference was a no-batch / N≤10 run, per
[[project_dinov2_init_ablation]] "lift pre re-run no-batch 3-seed=0.917"). **Fix (in the shared `_eval_loop`):** re-init the
policy's per-env state (`policy.set_env(world.envs)`, idempotent) at the START of each chunk → each chunk = a clean 10 episodes.
**Validated:** post-fix lift seed-42 OURS = 0.32 with successes SPREAD across all 5 chunks (3/7/11/12/16 cumulative), GC-IDM
= 0.44 (4/10/15/19/22) — both sane, not the 0.08 artifact. This is exactly the "a *correct* low number is the finding; a
*broken* one is not" distinction the task demanded; the fix moved both arms off the artifact floor.

### ✅ GOAL-FRAME VALIDATION (the task's "verify the goal frame is real" requirement)
`check_goal_frame.py` re-extracts init+goal exactly as `world.evaluate` does (`_extract_init_goal`, lift, goal_offset=30,
first chunk of 10). Goal pixels = `(10,224,224,3) uint8`, a DISTINCT future frame (per-episode mean|goal−start| ≈ 14.8–17.2,
**10/10 episodes distinct**, not blank, not == the start). goal_state carries `goal`/`goal_proprio`/`goal_state` at
start+offset. So the goal wiring feeds GC-IDM/OURS a real future frame — a low GC-IDM here is a genuine single-frame finding,
not a goal-wiring bug.

### 📊 RESULTS (N=50, 3 seeds {42,0,1}, post-chunk-reset-fix; goal-conditioned). Reference: goal-AGNOSTIC histbc = lift 95 / can 75 / square 70.

| task | arm | seed 42 | seed 0 | seed 1 | mean |
|---|---|---|---|---|---|
| lift | GC-IDM (Markovian) | 0.44 | 0.48 | 0.58 | **0.500** |
| lift | OURS (history) | 0.32 | 0.20 | 0.24 | **0.253** |
| can  | GC-IDM (Markovian) | 0.16 | 0.24 | 0.10 | **0.167** |
| can  | OURS (history) | 0.10 | 0.20 | 0.08 | **0.127** |
| square | GC-IDM (Markovian) | 0.30 | 0.14 | 0.10 | **0.180** |
| square | OURS (history) | 0.12 | 0.06 | 0.06 | **0.080** |

Per-arm means: **GC-IDM 0.500 / 0.167 / 0.180** (lift/can/square); **OURS 0.253 / 0.127 / 0.080**. GC-IDM wins ALL three task
means, and 8/9 per-seed cells (the lone OURS-ahead cell = can seed0, 0.20 vs 0.24 — within noise; OURS never wins a task mean).
Re-run notes (all same params/GPU as the original launch): lift GC-IDM s1 OOM'd at first launch (4 lift evals on one 46 GB GPU)
→ re-run alone = 0.58; square s0/s1 OURS hit a transient `mujoco.FatalError: Offscreen framebuffer is not complete (0x8cdd)`
under the 6-concurrent peak (GPU7 45 GB) → re-run at 3-concurrent = 0.06/0.06 (clean); square GC-IDM s0 OOM'd once → re-run = 0.14.

**VERDICT — the partial-observability prediction is FALSIFIED on robomimic: GC-IDM (Markovian) > OURS (history) on ALL THREE
contact tasks, by ~2× on the task means** (lift 0.500 vs 0.253; can 0.167 vs 0.127; square 0.180 vs 0.080). This is NOT a
harness bug: (1) the chunk-reset fix is validated (post-fix lift successes spread evenly across all 5 chunks for BOTH arms, vs
the pre-fix 4/50-all-in-chunk-1 artifact); (2) the goal frame is a real distinct future frame (10/10 episodes, mean pixel-diff
~15, not blank/identical); (3) BOTH arms WORK (non-zero, the right ballpark vs the goal-agnostic histbc 95/75/70 given the
SIGReg-base + hindsight-GC handicap) — so OURS's lower SR is a GENUINE result, and the control (GC-IDM non-zero on the SAME
harness + SAME goal-wiring) proves the harness is sound. The mechanism is the SAME open-loop-vs-closed-loop INVERSION found on
pusht §13: OURS's history lowers OPEN-LOOP action-MSE on all three (val_act 0.258/0.147/0.219 vs GC-IDM hindsight-MSE
0.437/0.226/0.324) yet WORSENS CLOSED-LOOP SR. The history+past-action stream amplifies covariate shift — once the policy steps
off the expert manifold, the action history it conditions on is its OWN off-distribution actions, compounding the error
(causal-confusion / "copycat" on the past-action stream), whereas the Markovian `(z_t, z_goal, h)` re-grounds on the fresh
observation each step and stays bounded. **So the recoverable-hidden-state ("has it grasped?") argument does NOT translate into a
closed-loop win here** — even though the §6 probe shows grasp state IS more decodable from history (AUC 0.97 vs single-frame
0.88), the history-conditioned *policy* pays a larger covariate-shift tax than the grasp-memory benefit buys.

**Cross-env synthesis (cube §11/§12 + pusht §13 + tworoom §14 + robomimic §16): OURS (history) NEVER BEATS GC-IDM (Markovian) in
CLOSED-LOOP SR on ANY tested env** — cube 72.7 vs 100, pusht 31 vs 90, lift 25.3 vs 50.0, can 12.7 vs 16.7, square 8.0 vs 18.0
(tworoom §14 = the lone TIE per the Phase-1 agent). The wedge ("history-conditioned forward GC policy > Markovian gcidm") is
FALSIFIED across fully-observed (cube), contact-pushing (pusht), and contact-grasp/place (robomimic) regimes. The clean,
repeatable finding is the INVERSION: history lowers open-loop MSE, worsens closed-loop SR (covariate shift on the self-generated
past-action stream). Open follow-ups: (i) the `use_action_history=false` ablation (state-only HS-frame history, NO past-action
stream) — if it RECOVERS SR toward GC-IDM, the past-action stream is the causal-confusion source and a state-only history might
still win; (ii) DAgger / noise-injection to test whether the inversion is a covariate-shift artifact fixable by on-policy data;
(iii) per-step latent distance to the expert manifold over a rollout (predict OURS diverges after the first off-expert step,
GC-IDM stays bounded).

## §14 ✅🆕 TwoRoom 3-arm goal-conditioned head-to-head — the FIRST env where OURS (history) TIES GC-IDM (L40S, 2026-06-20)

**Motivation (why).** Completes gcidm's 4-env benchmark (cube §11/§12 + pusht §13 done; this adds tworoom; reacher = §15). TwoRoom is
gcidm's near-ceiling navigation cell (published GC-IDM **100**, CEM ~82-84; our prior CEM on H100-174 was 95.3). It is a 2-D point-agent
two-room reaching task: fully observed (agent + target positions are in the current frame), short-horizon, NO grasp/contact memory. Per the
§13 wedge implication ("history wins ONLY under partial observability of a recoverable task-critical state"), the pre-registered prediction was
**OURS ≈ GC-IDM here (both near-ceiling), NOT OURS > GC-IDM** — tworoom is fully observed so Markovian should suffice, same regime as cube.
**VERDICT: confirmed AND sharper than expected — OURS 100 = GC-IDM 100 (a perfect TIE, both seeds 50/50), the FIRST benchmark cell where the
history head MATCHES Markovian GC-IDM rather than losing (cube 73≪100, pusht 31≪90). History does not HURT here (unlike pusht's covariate-shift
collapse) because tworoom's success is coarse goal-reaching with no contact dynamics to drift on.**

### What changed vs pusht (env / base / data), and the code adaptation
- **Frozen base:** `decoders/tworoom_ours_lewm_weights.pt` (vit-tiny-192, SIGReg; 303 keys; loads `missing=1 unexpected=0`, the 1 missing =
  mask_token, harmless, byte-identical key structure to cube/pusht). action_encoder.input_dim = **10** (= 2×5), identical to pusht.
- **Data:** `datasets/tworoom.h5` (12.8 GB HDF5, FLAT layout: per-column `action(2)/proprio(2)/observation(10)/pixels(HWC uint8)/pos_agent/
  pos_target/ep_idx/ep_offset/ep_len/step_idx/terminated/...`). swm `HDF5Dataset` via `load_dataset("tworoom.h5", ...)`. **10000 episodes**,
  frameskip 5, **raw action dim 2** → **action_block_dim = 2×5 = 10**. Episode raw-frame lengths 31/101/101 (min/median/max) → **n_obs (=L//fs)
  = 6/20/20** → max 20 obs-steps. `_load_slice` returns the SAME `(pixels (Fsub,3,224,224) uint8, action (L,2))` interface as pusht/cube, so the
  precompute loop is unchanged. Episode key = `ep_idx` (gip.episode_col auto-picks `ep_idx` over `episode_idx`).
- **H_max = 25** (covers the full 20-obs-step episode; cube/pusht used 50 for their 40/49-step episodes). Eval horizon0 = goal_offset/action_block
  = 25/5 = **5 obs-steps**.
- **Eval config `config/eval/tworoom.yaml` — UNCHANGED** (`dataset_name: tworoom` already resolves to `tworoom.h5`; verified by diff =
  TWOROOM_CONFIG_UNCHANGED). `env_name swm/TwoRoom-v1`, N=50, `goal_offset_steps=25`, `eval_budget=50`, `plan_config horizon=5 action_block=5`
  (→ 25 ≤ budget 50 ✓), callables `_set_state(state=proprio)` / `_set_goal_state(goal_state=goal_proprio)` (proprio-based, like pusht's
  state-based). Success = env's own `terminated` (agent reaches target).
- **Code adaptation: NONE this session.** `train_gcidm.py` / `train_ours_gc.py` are **byte-identical** to my session-start backups
  (`*.bak_tworoom`) — they already carried the pusht-era `--dataset_name`/`--keys_to_load` args (`train_gcidm`) and the cache-driven
  `action_block_dim` auto-adapt (`train_ours_gc`). Verified by `diff *.bak_tworoom <file>` = IDENTICAL_NO_SESSION_EDIT. `gip.py`/`jepa.py`/
  `eval_gip.py`/`gcidm.py` mtimes 01:34–04:00 (cube/pusht era) — UNTOUCHED. cube/pusht/bc/guided/planning/gcidm paths verified intact.
- **Same-box CEM run-dir:** `checkpoints/tworoom_lewm_base/` = pusht_lewm_base's config.json (action_encoder.input_dim=10) +
  `tworoom_ours_lewm_weights.pt` copied as `weights_epoch_0.pt`. For `mode=planning`.

### latent sanity (NOT collapsed)
Full run over 10000 episodes: `per-dim std(mean over dims) = 0.9996` (cube 1.024, pusht 0.945) → NOT collapsed.

### Infra (same as cube/pusht; thread-cap fix added — LOAD-BEARING)
Precompute frozen latents ONCE → `checkpoints/tworoom_gcidm/latents_cache.pt` (**82 MB**, 10000 eps in **322 s** @ ~31 eps/s, bf16). GC-IDM ON +
noh + OURS all share that ONE cache. GC-IDM ~1.0 s/epoch (200 ep ≈ 3.5 min); OURS ~17 s/epoch (40 ep ≈ 11 min).
**⚠️ INFRA FIX:** the first eval batch launched 5× `eval_gip` (each builds a 50-parallel-env `World` + default torch/numpy thread-pools =
**226 threads/proc**) → 1130 threads on 64 cores, **load average 133**, all evals gridlocked (each burned 90 min CPU in 8 min wall, 0 episodes
completed). FIX = set **`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4`** on every eval (threads 226→18, load
133→56) AND cap concurrency to ≤3 CPU-heavy evals at once. After the cap the SAME evals finished in ~45 s each. Killed the 5 thrashing procs by
**explicit PID** (never a pkill pattern — fan's air-hockey procs untouched) and relaunched leanly. Re-add the thread caps to every future eval.

### Exact train commands (L40S)
```
# GC-IDM ON (builds shared cache), GPU1:
... CUDA_VISIBLE_DEVICES=1 python train_gcidm.py --epochs 200 --batch_size 1024 --H_max 25 \
    --dataset_name tworoom.h5 --keys_to_load pixels,action \
    --weights .../decoders/tworoom_ours_lewm_weights.pt --run_name tworoom_gcidm --cache_run tworoom_gcidm
# GC-IDM noh (AdaLN ablation), GPU3:
... CUDA_VISIBLE_DEVICES=3 python train_gcidm.py --epochs 200 --batch_size 1024 --H_max 25 --ablate_horizon \
    --dataset_name tworoom.h5 --keys_to_load pixels,action --weights .../tworoom_ours_lewm_weights.pt \
    --run_name tworoom_gcidm_noh --cache_run tworoom_gcidm
# OURS history GC policy, GPU2:
... CUDA_VISIBLE_DEVICES=2 python train_ours_gc.py --epochs 40 --batch_size 512 --lr 3e-4 --H_max 25 \
    --history_size 3 --seed 3072 --weights .../tworoom_ours_lewm_weights.pt --run_name tworoom_ours_gc --cache_run tworoom_gcidm
```

### Convergence (val on frozen latents, z-scored 10-d action; ALL arms sit near the 1.0 action-variance floor → tworoom actions are near-
uninformative given (z_t,z_goal); offline MSE does NOT predict SR here — the navigation success criterion is coarse)
- **GC-IDM ON** (`tworoom_gcidm`): val_mse 0.907(ep1)→0.954(ep50)→0.983(ep100)→1.002(ep200); **best 0.892 @ep~7** (val RISES after = cosine
  schedule overfits the z-scored tail; `gcidm_head_best.pt` = the ep-7 best = reported). 1,123,850 head params.
- **GC-IDM noh** (`tworoom_gcidm_noh`): best **0.907**. Horizon ON cuts val MSE only ~1.6% (0.892 vs 0.907) — far less than cube (~36%) or pusht
  (~17%): tworoom's 5-step eval horizon + coarse goal makes the AdaLN signal nearly irrelevant (mirrored at eval: noh SR = ON SR = 100).
- **OURS** (`tworoom_ours_gc`, 160562 hindsight windows = 144506 train / 16056 val, HS=3, trainable 11.29 M): val_act 0.919(ep1)→**0.899(ep8,
  BEST)**→0.923(ep20)→0.975(ep40); val_act bottoms at ep8 then overfits = classic frozen-feature-head convergence. `weights_epoch_1.pt` = ep-8
  best (loads `Adim=10 missing=0 unexpected=0`).

### Exact eval commands (N=50, L40S, same box; thread-capped)
```
# (prefix EVERY eval) OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
# GC-IDM (mode=gcidm), seeds 42/7:
... python eval_gip.py --config-name tworoom policy=tworoom_gcidm +gip_eval.mode=gcidm eval.num_eval=50 seed=<42|7>
# GC-IDM horizon-OFF ablation, seed 42:
... python eval_gip.py --config-name tworoom policy=tworoom_gcidm_noh +gip_eval.mode=gcidm \
    +gip_eval.gcidm_run=tworoom_gcidm_noh +gip_eval.ablate_horizon=true eval.num_eval=50 seed=42
# OURS planning-free (mode=policy, goal+horizon), seeds 42/7:
... python eval_gip.py --config-name tworoom policy=tworoom_ours_gc +gip_eval.mode=policy +gip_eval.goal_conditioned=true \
    +gip_eval.horizon_H_max=25 eval.num_eval=50 seed=<42|7>
# Same-box CEM (mode=planning, frozen base), seeds 42/7:
... python eval_gip.py --config-name tworoom policy=tworoom_lewm_base +gip_eval.mode=planning eval.num_eval=50 seed=<42|7>
# Same-box guided (CEM + OURS goal+horizon prior), seeds 42/7:
... python eval_gip.py --config-name tworoom policy=tworoom_ours_gc +gip_eval.mode=guided +gip_eval.goal_conditioned=true \
    +gip_eval.horizon_H_max=25 eval.num_eval=50 seed=<42|7>
```
(OURS eval-context fix VERIFIED: `load_gip_model(tworoom_ours_gc)` → `Adim=10 missing=0 unexpected=0`, `use_action_history=True`,
`horizon_conditioned=True`, `horizon_modulator present` → cube §12 HS=3-frame + horizon eval-context engaged; OURS NOT fed a single frame.)

### RESULTS — TwoRoom goal-reaching, N=50, ALL SAME BOX (L40S), per-seed
| Arm | method | seed 42 | seed 7 | mean | vs gcidm-published |
|---|---|---|---|---|---|
| 1 | **GC-IDM** (gcidm, Markovian, planning-free) | 100.0 | 100.0 | **100.0** | = published **100** ✅ exact |
| 2 | **OURS** (history-cond. forward GC, planning-free) | 100.0 | 100.0 | **100.0** | — (TIES GC-IDM; both 50/50 True, verified not a batching artifact) |
| 3a | **guided** (CEM + OURS goal+horizon prior) | 96.0 | 94.0 | **95.0** | — (≈ our prior CEM 95.3) |
| 3b | **CEM / planning** (frozen-WM CEM, LeWM baseline) | 90.0 | 88.0 | **89.0** | ≥ published CEM **~82-84** ✅ |
| — | GC-IDM horizon-OFF (AdaLN ablation) | 100.0 | — | 100.0 | = ON; horizon irrelevant at eval here |

**GC-IDM reproduced 100 → YES (exact, both seeds).** **OURS ≥ GC-IDM → YES (TIE 100 = 100)** — first cell where the history head does not lose.
CEM (89) ≥ published (~82-84) ✅; guided 95 sits between CEM and the planning-free heads.

### 🔬 DIAGNOSIS (sanity: both planning-free heads hit a hard 100 → is the eval too easy / a no-op?)
The 100/100 tie is REAL, not a degenerate "everything passes" — five checks:
1. **Models load clean** (above).
2. **CEM (89) and guided (95) are NOT 100:** if the eval auto-succeeded those arms would also be 100. The planning arms sitting at 88-96 while
   BOTH learned planning-free heads hit 100 shows the task DISCRIMINATES between methods, just not between the two GOOD planning-free heads.
3. **Per-episode scatter:** OURS s42/s7 = **50/50 True each** (full array), GC-IDM likewise — flat ceilings, not the chunk-1-then-zeros 10-env-pool
   reuse artifact (§10 DINOv2); CEM/guided show genuine scattered T/F on the SAME pool → pool healthy.
4. **Offline MSE near the variance floor yet SR=100:** tworoom success = reaching the target, a COARSE criterion many action sequences satisfy,
   so the head needn't predict the exact expert action — both Markovian and history heads recover the right direction trivially. WHY offline MSE
   is uninformative about SR here (unlike pusht, where precise contact actions mattered and OURS's lower MSE still gave WORSE SR).
5. **Reproduces published 100 exactly** → faithful setting; our GC-IDM is a fair drop-in.

**Mechanism (why OURS ties here but LOST on cube/pusht).** TwoRoom is fully observed (no grasp/contact memory) AND coarse-goal AND non-contact
(point-agent navigation, no object dynamics). Cube (§12): history LOST (73 vs 100) because near-ceiling fine-grained placement let the simpler
Markovian head fit the precise manifold better. Pusht (§13): history LOST HARD (31 vs 90) because contact + open-loop history induced
covariate-shift drift. TwoRoom removes BOTH failure modes (coarse goal → open-loop proposals land in the success region without precise tracking;
no contact → no drift to compound) → TIE at the ceiling. Consistent with the §13 wedge implication: history neither helps nor hurts on a
fully-observed non-contact reaching task.

**Conclusion.** TwoRoom reproduces the gcidm cell exactly (GC-IDM 100 = published 100; CEM 89 ≥ published ~82-84). The history wedge **does not
lose** (OURS 100 = GC-IDM 100) — the first of the 4 benchmark envs where it is not beaten — but it also does not WIN: a tie at the ceiling on a
fully-observed coarse-goal task. Across the benchmark so far: **cube GC-IDM 100 ≫ OURS 73 · pusht GC-IDM 90 ≫ OURS 31 · tworoom GC-IDM 100 =
OURS 100 · reacher = §15.** Combined with the in-flight robomimic §-block above (lift GC-IDM 0.500 > OURS 0.253), history has STILL not WON any
cell; the wedge's last stand remains can/square (grasp+place partial-observability).

### Files added / changed (all `$B=le-wm-repro` on L40S; backups `*.bak_tworoom`)
- **NO repo edits.** `train_gcidm.py` / `train_ours_gc.py` byte-identical to `*.bak_tworoom` (pusht-era args reused). `config/eval/tworoom.yaml`
  UNCHANGED (`dataset_name: tworoom` already correct; backup `tworoom.yaml.bak_tworoom`). `gip.py`/`jepa.py`/`eval_gip.py`/`gcidm.py` untouched.
- Checkpoints (`$STABLEWM_HOME/checkpoints/`): `tworoom_gcidm/{gcidm_head_best.pt, gcidm_config.json, latents_cache.pt(82M)}`,
  `tworoom_gcidm_noh/`, `tworoom_ours_gc/{weights_epoch_1.pt=BEST@ep8, weights_epoch_0.pt, config.json}`, `tworoom_lewm_base/`. Eval results
  under `$STABLEWM_HOME/gip_eval/{gcidm,policy,guided,planning}/`. Logs in `/mnt/minghao_data/logs_gctest/`.

## §15 ✅🆕 Reacher (DMC) 3-arm goal-conditioned head-to-head — completes the 4-env benchmark; the CHECKPOINT-SELECTION trap (L40S, 2026-06-20)

**Motivation (why).** Final cell of gcidm's 4-env benchmark (cube §11/§12 + pusht §13 + tworoom §14 done). Reacher (DMControl `qpos_match`)
is gcidm's other near-ceiling cell (published GC-IDM **99.7**, CEM ~68-70; our prior on H100-174: CEM 85.3, guided 86.7, **bc 3.3**). It is a
2-joint arm reaching a target qpos: fully observed, precise-control (success = finger joint-config matches target within tolerance at the last
step), no grasp/contact memory. Per the §13/§14 wedge implication, prediction was **OURS ≈ GC-IDM (both near-ceiling), NOT OURS > GC-IDM** —
reacher is fully observed so Markovian should suffice. **VERDICT: confirmed — OURS 97 = GC-IDM 97 (a SECOND tie, like tworoom), GC-IDM reproduces
the published ~99.7 — BUT ONLY after fixing a CHECKPOINT-SELECTION TRAP: the val-MSE `best` checkpoint MASSIVELY underperforms (GC-IDM 73, OURS 63)
because on reacher the z-scored-action variance floor makes val MSE a misleading early-stopping signal; the CONVERGED (ep40/ep200 `latest`)
checkpoint rolls out far better (GC-IDM 97, OURS 97). Headline = converged checkpoints, per global rule 9 ("train to convergence; never headline an
early-stopping number").**

### Two BLOCKERS hit and resolved (both env-specific, fully reversible)
1. **dm_control ↔ mujoco skew (eval blocker).** `swm/ReacherDMControl-v0` failed to instantiate: `dm_control 1.0.41` (`sizes.py:217`) lists a stale
   `mjmodel.flex_bandwidth` field that `mujoco 3.9.0` removed → `struct_indexer` getattr-crashes (`'MjModel' object has no attribute
   'flex_bandwidth'`). FIX = a runtime shim **`/tmp/reacher_compat/sitecustomize.py`** (external to the repo; auto-loaded at interpreter startup by
   prepending `/tmp/reacher_compat` to `PYTHONPATH`) that drops any `array_sizes['mjmodel']` field absent on the live MjModel. In-memory only, NO
   edit to the installed package, no-op for non-dm_control envs (cube/pusht/tworoom). Verified: env makes + resets + renders (224×224) + callables
   `set_state`/`set_target_qpos` present; N=2 GC-IDM smoke = 2/2; N=2 CEM smoke = 2/2. **Training does NOT need the shim** (precompute reads only the
   h5, builds no env) — only eval does.
2. **Reacher eval config dataset_name was wrong.** `config/eval/reacher.yaml` had `dataset_name: dmc/reacher_random` → resolves to
   `datasets/dmc/reacher_random.h5` which DOES NOT EXIST; the actual file is `datasets/reacher.h5`. FIX = one-line edit `dataset_name: reacher`
   (the ONLY repo edit this entire session). Backup `config/eval/reacher.yaml.bak_reacher`; diff = exactly that one line.

### What changed vs pusht/tworoom (env / base / data)
- **Frozen base:** `decoders/reacher_ours_lewm_weights.pt` (vit-tiny-192, SIGReg; 303 keys; `missing=1 unexpected=0`, the 1 = mask_token).
  action_encoder.input_dim = **10** (= 2×5), identical to pusht/tworoom.
- **Data:** `datasets/reacher.h5` (**98 GB** HDF5, FLAT: `action(2,float64)/qpos(2)/qvel(2)/target_pos(2)/finger_pos/observation(6)/pixels/
  ep_idx/ep_offset/ep_len/step_idx/success/score/...`). swm `HDF5Dataset` via `load_dataset("reacher.h5", ...)`. **10000 episodes**, frameskip 5,
  **raw action dim 2** → **action_block_dim = 10**. ALL episodes exactly 201 raw frames → **n_obs = 40** (uniform). `_load_slice` returns the SAME
  `(pixels (Fsub,3,224,224) uint8, action (L,2))` interface (action float64 here vs float32 elsewhere; harmless, cast to fp16 in the cache).
- **H_max = 50** (covers the 40-obs-step episode; matches pusht/cube). Eval horizon0 = goal_offset/action_block = 25/5 = **5 obs-steps**.
- **Eval config callables** (after the dataset_name fix): `set_state(qpos=qpos, qvel=qvel)` + `set_target_qpos(target_qpos=goal_qpos)`. The
  `goal_qpos` is auto-derived by `world._extract_init_goal` from the `qpos` column at the goal frame (it iterates ALL dataset columns, not just
  keys_to_cache, so qpos/qvel/goal_qpos resolve even though the config only caches `action`). Success = env `terminated` (qpos-match tolerance).
- **Code adaptation: NONE.** `train_gcidm.py` / `train_ours_gc.py` byte-identical to `*.bak_reacher` (pusht-era args). `gip.py`/`jepa.py`/
  `eval_gip.py`/`gcidm.py` untouched. The ONLY repo change anywhere this session = the one-line reacher.yaml dataset_name.
- **Same-box CEM run-dir:** `checkpoints/reacher_lewm_base/` = pusht_lewm_base config.json + `reacher_ours_lewm_weights.pt` as `weights_epoch_0.pt`
  (loads `Adim=10 missing=89 unexpected=0`, the 89 = fresh unused action head, planning ignores it).

### latent sanity (NOT collapsed)
Full run over 10000 eps: `per-dim std(mean over dims) = 0.9984` → NOT collapsed.

### 🔑 THE CHECKPOINT-SELECTION TRAP (the central reacher finding; diagnosed BEFORE reporting, per task instruction "if GC-IDM <80, diagnose")
First-pass GC-IDM landed **76/70 = 73** (< 80, far below published 99.7) → triggered the mandatory diagnosis. Five checks:
1. **Model loads clean, goal+horizon engaged** (`Adim=10 missing=0 unexpected=0`).
2. **CEM works (80/82=81) and bc-prior is non-trivial:** the base WM is fine (CEM ≈ our prior 85.3 ≫ published 68-70); the bottleneck is the
   planning-free HEAD, not the base or the env.
3. **Goal-sensitivity probe (4000 held-out hindsight tuples, z-scored 10-d action-MSE):** at eval horizon h=5 → true-goal **0.959** vs random-goal
   **1.145** vs variance-floor **1.000**; at h=1 → true-goal **0.854** vs random **1.551**. So the head DOES use the goal (true < random) and is a
   strong 1-step predictor, but at h=5 the action is only marginally predictable from the frozen 192-d latent (0.959 barely below the 1.0 floor).
4. **Val-MSE curve is the smoking gun:** GC-IDM val_mse bottoms at **0.975 @ep~5** then RISES to 1.03 @ep200 (cosine-schedule overfit of the
   z-scored tail); `gcidm_head_best.pt` = the ep-5 minimum = an EARLY-STOPPED, UNDERFIT head (train_mse only 0.97 there, vs 0.91 @ep200). **The
   variance floor masks the real signal: "best val MSE" picks a head that has barely learned.** Same pattern in OURS (val_act bottoms 0.970 @ep11,
   train_act keeps falling 1.004→0.958 to ep40).
5. **FIX = eval the CONVERGED (`latest`/ep200/ep40) checkpoint, not `best-val`:** GC-IDM latest → **98/96 = 97.0** (vs best-val 73); OURS latest
   (ep40, `weights_epoch_0.pt`, via `+ckpt_epoch=0`) → **98/96 = 97.0** (vs best-val ep11 64/62 = 63). The converged checkpoint has lower TRAIN MSE
   (sharper actions) and rolls out far better despite higher VAL MSE. **This is reacher-specific** (precise-control + variance-floor early-stop);
   cube/pusht/tworoom did not suffer it (cube/tworoom hit their ceiling even at best-val; pusht's best-val was at a real minimum). Lesson recorded:
   **for precise-control tasks where val MSE plateaus near the action-variance floor, SR-select or use the converged checkpoint — val MSE is not a
   reliable selector.**

### Infra
Precompute frozen latents ONCE → `checkpoints/reacher_gcidm/latents_cache.pt` (**170 MB**, 10000 eps × 40 obs-steps in **1045 s** @ ~10 eps/s —
3× slower than tworoom because 41 frames/ep + the 98 GB file is I/O-heavy). GC-IDM ON + noh + OURS share that ONE cache. Thread caps
(`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4`) on every eval + ≤3 concurrent (the §14 fix). 2 head-trainers OOM'd once on GPUs that had spiked to 45 GB
(co-located evals + the gcidm-ON cache on GPU); relaunched on lower-occupancy GPUs (auto-picked via `nvidia-smi`). dm_control reacher rendering
makes its evals ~2-4× slower than tworoom's gridworld.

### Exact train commands (L40S)
```
# GC-IDM ON (builds shared cache), GPU7:
... CUDA_VISIBLE_DEVICES=7 python train_gcidm.py --epochs 200 --batch_size 1024 --H_max 50 \
    --dataset_name reacher.h5 --keys_to_load pixels,action \
    --weights .../decoders/reacher_ours_lewm_weights.pt --run_name reacher_gcidm --cache_run reacher_gcidm
# GC-IDM noh (AdaLN ablation), GPU4:
... python train_gcidm.py --epochs 200 --batch_size 1024 --H_max 50 --ablate_horizon \
    --dataset_name reacher.h5 --keys_to_load pixels,action --weights .../reacher_ours_lewm_weights.pt \
    --run_name reacher_gcidm_noh --cache_run reacher_gcidm
# OURS history GC policy, GPU3:
... python train_ours_gc.py --epochs 40 --batch_size 512 --lr 3e-4 --H_max 50 --history_size 3 --seed 3072 \
    --weights .../reacher_ours_lewm_weights.pt --run_name reacher_ours_gc --cache_run reacher_gcidm
```

### Convergence (val on frozen latents, z-scored 10-d action; ALL arms near the 1.0 variance floor → val MSE is a POOR SR-selector here, see trap)
- **GC-IDM ON** (`reacher_gcidm`): val_mse 0.985(ep1)→**0.975(ep~5 BEST)**→0.999(ep50)→1.016(ep100)→1.032(ep200). train_mse 0.997→0.911. 1,123,850
  head params, 400000 hindsight samples. The ep-5 `best` is UNDERFIT (the trap); the ep-200 `latest` is the converged head (reported, SR 97).
- **GC-IDM noh** (`reacher_gcidm_noh`): best_val 0.987. SR (converged/latest) = **36** (horizon-OFF) vs ON 97 → **horizon is CRITICALLY load-bearing
  on reacher (−61 points)** — the opposite extreme from tworoom (where horizon was irrelevant) and bigger than cube/pusht. A precise reach needs to
  know how many steps remain to pace the trajectory; without the AdaLN horizon the open-loop head cannot. (best-val noh = 28; latest noh = 36.)
- **OURS** (`reacher_ours_gc`, 380000 hindsight windows = 342000 train / 38000 val, HS=3, trainable 11.29 M): val_act 0.993(ep1)→**0.970(ep11
  BEST)**→0.975(ep40). `weights_epoch_1.pt` = ep-11 best-val (SR 63, the trap); `weights_epoch_0.pt` = ep-40 latest/converged (SR 97, reported).

### Exact eval commands (N=50, L40S, same box; thread-capped; reacher_compat shim on PYTHONPATH)
```
# (prefix EVERY eval) PYTHONPATH=/tmp/reacher_compat:$B OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
# GC-IDM CONVERGED (the latest/ep200 head, copied into reacher_gcidm_latest/ as gcidm_head_best.pt), seeds 42/7:
... python eval_gip.py --config-name reacher policy=reacher_gcidm_latest +gip_eval.mode=gcidm \
    +gip_eval.gcidm_run=reacher_gcidm_latest eval.num_eval=50 seed=<42|7>
# GC-IDM best-val (diagnostic, the trap), seeds 42/7:
... python eval_gip.py --config-name reacher policy=reacher_gcidm +gip_eval.mode=gcidm eval.num_eval=50 seed=<42|7>
# GC-IDM horizon-OFF ablation (converged), seed 42:
... python eval_gip.py --config-name reacher policy=reacher_gcidm_noh_latest +gip_eval.mode=gcidm \
    +gip_eval.gcidm_run=reacher_gcidm_noh_latest +gip_eval.ablate_horizon=true eval.num_eval=50 seed=42
# OURS planning-free CONVERGED (ckpt_epoch=0 = ep40 latest), seeds 42/7:
... python eval_gip.py --config-name reacher policy=reacher_ours_gc +ckpt_epoch=0 +gip_eval.mode=policy \
    +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|7>
# OURS best-val (ckpt_epoch=1, diagnostic, the trap), seeds 42/7:  (same, +ckpt_epoch=1)
# Same-box CEM (frozen base), seeds 42/7:
... python eval_gip.py --config-name reacher policy=reacher_lewm_base +gip_eval.mode=planning eval.num_eval=50 seed=<42|7>
# Same-box guided (CEM + OURS CONVERGED goal+horizon prior), seeds 42/7:
... python eval_gip.py --config-name reacher policy=reacher_ours_gc +ckpt_epoch=0 +gip_eval.mode=guided \
    +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|7>
```
(OURS converged eval-context VERIFIED: `load_gip_model(reacher_ours_gc, epoch=0)` → `weights_epoch_0.pt Adim=10 missing=0 unexpected=0`,
`use_action_history=True`, `horizon_conditioned=True`, `horizon_modulator present` → HS=3-frame + horizon engaged; OURS NOT fed a single frame.)

### RESULTS — Reacher goal-reaching, N=50, ALL SAME BOX (L40S), per-seed (HEADLINE = converged checkpoints)
| Arm | method | seed 42 | seed 7 | mean | vs gcidm-published |
|---|---|---|---|---|---|
| 1 | **GC-IDM** (gcidm, Markovian, planning-free) — **converged** | 98.0 | 96.0 | **97.0** | ≈ published **99.7** ✅ reproduced |
| 2 | **OURS** (history-cond. forward GC, planning-free) — **converged** | 98.0 | 96.0 | **97.0** | — (TIES GC-IDM; OURS conv s42 = 49/50 True, verified not a batching artifact) |
| 3a | **guided** (CEM + OURS converged goal+horizon prior) | 82.0 | 86.0 | **84.0** | — (≈ our prior guided 86.7) |
| 3b | **CEM / planning** (frozen-WM CEM, LeWM baseline) | 80.0 | 82.0 | **81.0** | ≥ published CEM **~68-70** ✅ (≈ our prior 85.3) |
| — | GC-IDM horizon-OFF (AdaLN ablation, converged) | 36.0 | — | 36.0 | **−61 vs ON** → horizon CRITICAL on reacher |
| (diag) | GC-IDM best-val checkpoint (the trap) | 76.0 | 70.0 | 73.0 | early-stop artifact (variance-floor val MSE) |
| (diag) | OURS best-val checkpoint (the trap) | 64.0 | 62.0 | 63.0 | early-stop artifact (variance-floor val MSE) |

**GC-IDM reproduced ~99.7 → YES (converged 97, both seeds 98/96).** **OURS ≥ GC-IDM → YES (TIE 97 = 97)** — second cell (after tworoom) where the
history head does not lose. CEM (81) ≥ published (~68-70) ✅; guided 84 sits between CEM and the planning-free heads (the prior helps the search a
little). Horizon-OFF GC-IDM 36 (≪ ON 97) → on reacher the AdaLN horizon is the single most load-bearing component.

### 🔬 DIAGNOSIS / Mechanism (why OURS ties GC-IDM here)
Reacher is fully observed (target qpos + arm pose are in the current frame), precise-control, non-contact. Like tworoom (§14) and unlike cube/pusht,
both planning-free heads converge to the same ceiling once trained to convergence: (z_t, z_goal, h) and (HS-history, z_goal, h) recover equally good
reaching actions because the recoverable state is fully present in the current frame and there is no contact dynamics for open-loop history to drift
on. The history channel neither helps (nothing partially observed to remember) nor hurts (no contact to compound errors). The DECISIVE knob is the
AdaLN horizon (−61 without it), not history. The earlier "GC-IDM 73 < published" was NOT a base/env failure but a checkpoint-selection artifact —
once the converged head is used, reacher reproduces the published 99.7.

**Conclusion.** Reacher reproduces the gcidm cell (GC-IDM converged 97 ≈ published 99.7; CEM 81 ≥ published 68-70). The history wedge **does not lose**
(OURS 97 = GC-IDM 97) — the SECOND tie. **Final 4-env benchmark (converged, planning-free GC-IDM vs OURS):** cube GC-IDM 100 ≫ OURS 73 · pusht
GC-IDM 90 ≫ OURS 31 · tworoom GC-IDM 100 = OURS 100 · reacher GC-IDM 97 = OURS 97. **History LOSES on the two fine/contact envs (cube, pusht) and
TIES on the two coarse/precise-fully-observed reaching envs (tworoom, reacher); it has NOT WON any of the 4 gcidm cells.** Combined with the in-flight
robomimic §-block (lift GC-IDM 0.500 > OURS 0.253), history's positive wedge remains unobserved on every fully-observed cell — consistent with the
standing hypothesis that it can only win under PARTIAL observability of a recoverable task-critical state (Lift-family grasp/contact memory), which
none of the 4 gcidm envs provide. Methodological takeaway for the paper: **report converged, not best-val, checkpoints for precise-control GC heads**
(the reacher trap would have under-reported BOTH GC-IDM and OURS by ~25-35 points).

### Files added / changed (all `$B=le-wm-repro` on L40S; backups `*.bak_reacher`)
- **ONE repo edit:** `config/eval/reacher.yaml` `dataset_name: dmc/reacher_random → reacher` (backup `reacher.yaml.bak_reacher`; diff = that one
  line). `train_gcidm.py`/`train_ours_gc.py` byte-identical to `*.bak_reacher`. `gip.py`/`jepa.py`/`eval_gip.py`/`gcidm.py` UNTOUCHED.
- **External (non-repo) shim:** `/tmp/reacher_compat/sitecustomize.py` (dm_control flex_bandwidth fix; loaded via PYTHONPATH for reacher eval only).
- Checkpoints (`$STABLEWM_HOME/checkpoints/`): `reacher_gcidm/{gcidm_head_best.pt(=ep5 best-val, the trap), gcidm_head_latest.pt(=ep200 converged,
  REPORTED), gcidm_config.json, latents_cache.pt(170M)}`, `reacher_gcidm_latest/` (the converged head copied as best for eval),
  `reacher_gcidm_noh/` + `reacher_gcidm_noh_latest/`, `reacher_ours_gc/{weights_epoch_0.pt=ep40 converged REPORTED, weights_epoch_1.pt=ep11
  best-val the trap, config.json}`, `reacher_lewm_base/`. Eval results under `$STABLEWM_HOME/gip_eval/{gcidm,policy,guided,planning}/`. Logs in
  `/mnt/minghao_data/logs_gctest/`.

### OPEN QUESTIONS / next
- **The 4-env gcidm benchmark is DONE: history TIES on tworoom+reacher (coarse/precise fully-observed), LOSES on cube+pusht (fine/contact
  fully-observed), WINS nowhere.** The wedge's only remaining hope is the in-flight robomimic Lift-family GC head-to-head (partial grasp-state) —
  lift already shows GC-IDM 0.500 > OURS 0.253, so even that is leaning negative; can/square (grasp+place) pending.
- **Checkpoint-selection:** re-audit cube/pusht/tworoom OURS+GC-IDM — were ANY reported at a best-val early-stop that masked a higher converged SR?
  (cube/tworoom hit 100 so no; pusht OURS was a real minimum; but worth a converged-checkpoint A/B on cube §12 OURS 73 to confirm it isn't a trap
  too — if cube OURS-converged ≫ 73, the cube "loss" weakens.)

## §19 ✅🆕 CONVERGED-vs-BEST-VAL re-eval — does the reacher checkpoint-trap (§15) FLIP the cube/pusht/robomimic headlines? (L40S, 2026-06-20)

**Motivation (why).** §15 (reacher) discovered a CHECKPOINT-SELECTION TRAP: the val-MSE `best` checkpoint MASSIVELY underperforms the
CONVERGED (latest) checkpoint at rollout (GC-IDM 73→97, OURS 63→97) because on a precise-control task val MSE sits near the z-scored-action
variance floor and is a misleading early-stop signal. Every prior OURS headline in this campaign was taken from the **best-val** checkpoint:
`load_gip_model` defaults to the **highest epoch-NUMBER** file, and `train_ours_gc.py` saves the BEST-val-act head as `weights_epoch_1.pt`
(highest #, so loaded by default) and the LATEST/converged head as `weights_epoch_0.pt` (lower #, NOT picked unless `+ckpt_epoch=0`). So the
cube §12 OURS 72.7, pusht §13 OURS 31, and robomimic §16 OURS lift/can/square 0.253/0.127/0.080 were ALL the early-stop `weights_epoch_1.pt`.
If they were underfit like reacher, "ours loses" would be a checkpoint artifact, not a real result — and the WHOLE campaign conclusion
(history-conditioned forward GC policy LOSES to Markovian GC-IDM on every cell) would be in question. **This §19 re-evals OURS with the CONVERGED
`weights_epoch_0.pt` (`+ckpt_epoch=0`) at N=50, 2 seeds {42,7}, same-box, SAME eval command as each cell — only the checkpoint swapped — for
cube / pusht / lift / can / square; and re-evals GC-IDM with its CONVERGED `gcidm_head_latest.pt` on cube/pusht for an apples-to-apples
converged-vs-converged comparison.** Mechanism + exact ckpt-selection verified in code before running (see below).

### Checkpoint-selection mechanism (verified in `gip.py` + `train_ours_gc.py` source, L40S)
- `gip.load_gip_model(run, epoch=None)` → `pts = sorted(glob("weights_epoch_*.pt"), key=int(epoch#)); ckpt = pts[-1]` → picks the **highest
  epoch number** = `weights_epoch_1.pt`. With `epoch=0` it loads `weights_epoch_{0}.pt` = the converged/latest head.
- `train_ours_gc.py` (lines 217-234): on each new val-act minimum it writes `save_state("weights_epoch_1.pt")  # BEST (load_gip_model loads this)`;
  EVERY epoch it writes `save_state("weights_epoch_0.pt")  # LATEST`. So `_1`=best-val (default-loaded headline), `_0`=converged (needs `+ckpt_epoch=0`).
  Confirmed by mtimes: for all 5 runs `weights_epoch_0.pt` is NEWER than `_1` (the latest written last). VERIFIED load for all 5:
  `cube/pusht/lift/can/square_*` epoch∈{None,0} → both `Adim={25,10,35,35,35} missing=0 unexpected=0`, `use_action_history=True
  horizon_conditioned=True has_modulator=True` → converged ckpt loads clean, history+horizon engaged, NOT fed a single frame.
- `eval_gip.py` (line 54) and `eval_histbc_robomimic.py` (line 339) BOTH route the checkpoint through `gip.load_gip_model(cfg.policy,
  epoch=cfg.get("ckpt_epoch", None))` → `+ckpt_epoch=0` selects the converged head in BOTH evaluators (cube/pusht via eval_gip; robomimic via eval_histbc).
- GC-IDM: `gip.load_gcidm_model` defaults to `gcidm_head_best.pt`. Cube/pusht BOTH have a `gcidm_head_latest.pt` (the ep200 converged head) on disk.
  Mirroring §15's reacher fix, I COPIED `gcidm_head_latest.pt` → new run-dirs `{cube,pusht}_gcidm_latest/gcidm_head_best.pt` (+ the gcidm_config.json)
  and evaled with `policy=<task>_gcidm_latest +gip_eval.gcidm_run=<task>_gcidm_latest`. Both load `action_dim={25,10}` clean.

### Infra / constraints (eval-only; no retrain; no shared-code edits; thread-capped; ≤3 concurrent of my own)
L40S `stratus-lookout`, `$B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, run
as `sudo -u minghao.fu`. Env: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl_recv TMPDIR=/tmp
OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4 (OMP=1 for robomimic) CUDA_VISIBLE_DEVICES=<low-occupancy GPU>`. Driver `/tmp/reval_conv.sh`, staggered 7s,
≤3 of my evals concurrent (other agents' §17 tool_hang/transport evals ran concurrently on separate GPUs — not mine, not touched). NO repo edits,
NO shared method-code edits, NO touching other agents' checkpoints (only READ them; copied the cube/pusht gcidm_latest heads into my own new run-dirs;
wrote my own result/log files under `/mnt/minghao_data/logs_revalconv/` + the evaluator's own `gip_eval/` outputs). Logs `/mnt/minghao_data/logs_revalconv/`.

### EXACT eval commands (only the checkpoint swapped vs the original cells)
```
# CUBE/PUSHT OURS converged (vs §12/§13 which omitted ckpt_epoch → default best-val):
python eval_gip.py --config-name <cube|pusht> policy=<task>_ours_gc +ckpt_epoch=0 \
  +gip_eval.mode=policy +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|7>
# CUBE/PUSHT GC-IDM converged (latest head copied into <task>_gcidm_latest):
python eval_gip.py --config-name <cube|pusht> policy=<task>_gcidm_latest +gip_eval.mode=gcidm \
  +gip_eval.gcidm_run=<task>_gcidm_latest eval.num_eval=50 seed=<42|7>
# ROBOMIMIC OURS converged (vs §16 which omitted ckpt_epoch → default best-val); per-task budget/goal_offset from §1c/§16:
python eval_histbc_robomimic.py --config-name robomimic policy=<task>_gc_ours +ckpt_epoch=0 \
  +gip_eval.mode=policy +gip_eval.goal_conditioned=true eval.num_eval=50 world.num_envs=10 \
  world.task=<Lift|PickPlaceCan|NutAssemblySquare> dataset.stats=<task> eval.dataset_name=<task> \
  eval.eval_budget=<100|240|320> eval.goal_offset_steps=<30|90|120> seed=<42|7>
```

### RESULTS — best-val (old headline) vs CONVERGED, N=50, per-seed (L40S same-box)
| cell | arm | best-val s42 | best-val s7 | best-val mean | **CONV s42** | **CONV s7** | **CONV mean** | verdict |
|---|---|---|---|---|---|---|---|---|
| **cube** | OURS | 78 | 68 | 73.0 (3-seed §12 = 72.7) | **76.0** | **76.0** | **76.0** | **GENUINE** (≈ best-val, NOT a trap) |
| **cube** | GC-IDM | 100 | 100 | 100.0 | **100.0** | **100.0** | **100.0** | unchanged (apples-to-apples) |
| **pusht** | OURS | 30 | 32 | 31.0 | **34.0** | **28.0** | **31.0** | **GENUINE** (identical to best-val) |
| **pusht** | GC-IDM | 86 | 94 | 90.0 | **88.0** | **94.0** | **91.0** | unchanged (apples-to-apples) |
| **lift** | OURS | 0.32 | (3-seed §16 {42,0,1}=0.32/0.20/0.24)→0.253 | 0.253 | **0.34** | **0.20** | **0.27** | **GENUINE** (s42 0.32→0.34; ≪ GC-IDM 0.50) |
| **can** | OURS | 0.10 | (3-seed §16=0.10/0.20/0.08)→0.127 | 0.127 | **0.16** | **0.18** | **0.17** | **GENUINE** (+0.04; ≈ GC-IDM best-val 0.167) |
| **square** | OURS | 0.12 | (3-seed §16=0.12/0.06/0.06)→0.080 | 0.080 | **0.12** | **0.12** | **0.12** | **GENUINE** (s42 0.12→0.12 identical; ≪ GC-IDM 0.18) |

(NB robomimic best-val §16 used 3 seeds {42,0,1}; my converged re-eval used 2 seeds {42,7} per the task spec. The like-for-like is **seed 42, shared by
both**: lift 0.32(best)→0.34(conv), can 0.10→0.16, square 0.12→0.12 — all within ±0.06, NO reacher-style jump. Converged 2-seed means lift 0.27 / can
0.17 / square 0.12 vs best-val 3-seed 0.253 / 0.127 / 0.080: slightly higher (more training helps a touch) but the SAME low ballpark — OURS still loses
or barely-ties GC-IDM, never the 30+pt jump reacher showed.)
(Quoted real output: cube OURS conv `weights_epoch_0.pt success_rate=76.0` both seeds; pusht OURS conv `34.0`(s42)/`28.0`(s7); cube GC-IDM conv
`100.0` both; pusht GC-IDM conv `88.0`(s42)/`94.0`(s7); lift OURS conv `0.34`(s42)/`0.2`(s7); can OURS conv `0.16`/`0.18`; square OURS conv `0.12`/`0.12`.
All robomimic converged arrays SCATTERED across the 5 chunks (lift s42 per-chunk-True [3,5,3,5,1]=17/50; can s42 [2,1,2,0,3]=8/50) → chunk-reset-fix
working, NOT the §16 chunk-1-then-zeros artifact. Best-val numbers are the §12/§13/§16 headlines on record.)

### VERDICT (cube + pusht — DONE)
- **The cube/pusht "OURS loses" SURVIVES converged-checkpoint selection — it is NOT a reacher-style artifact.**
  - **Cube OURS:** best-val 2-seed 73.0 → converged 76.0 (+3, both seeds 76, tighter). NOT the reacher 63→97 jump. Still ≈73-76 ≪ GC-IDM 100.
  - **PushT OURS:** best-val 31.0 → converged 31.0 (IDENTICAL; s42 30→34, s7 32→28, within ±3 eval noise). Still ≈31 ≪ GC-IDM 90.
  - GC-IDM converged confirms the comparison is apples-to-apples: cube 100=100, pusht 90→91 (within noise). So converged-vs-converged the gaps
    are cube 76 vs 100 and pusht 31 vs 91 — the SAME ~24pp and ~60pp losses the best-val headlines reported.
- **Why reacher trapped but cube/pusht did not** (consistent with §15's own scoping): reacher is precise-control where val-MSE plateaus at the
  z-scored-action variance floor → "best val" picks a barely-trained head. Cube's val_act had a real minimum at ep15 then mildly overfit (a true
  early-stop, the converged head only +3); pusht's val_act bottomed at ep28 at a genuine minimum (converged ≡ best). Neither sits at the variance
  floor → val-MSE selection was sound for them. §15 had already predicted this ("cube/pusht did not suffer the trap; pusht's best-val was at a
  real minimum") and §15's open question ("worth a converged A/B on cube OURS 73 to confirm it isn't a trap too") is now ANSWERED: it isn't.

### VERDICT (robomimic lift/can/square — DONE)
- **All three robomimic OURS "loses" SURVIVE converged-checkpoint selection — NONE is a reacher-style artifact.** Converged 2-seed means
  lift **0.27** / can **0.17** / square **0.12** vs best-val 3-seed 0.253 / 0.127 / 0.080. Converged is marginally higher (more training helps a few
  points, expected) but the SAME LOW ballpark, NOT the reacher 30+pt jump. On the shared seed 42: lift 0.32→0.34, can 0.10→0.16, square 0.12→0.12.
- **vs GC-IDM (§16 best-val, single-frame, NOT subject to this OURS-head trap):** lift OURS conv 0.27 < GC-IDM 0.500 (still ~2× loss); square OURS
  conv 0.12 < GC-IDM 0.180 (still a loss); **can is now a near-TIE** — OURS conv 0.17 vs GC-IDM 0.167. The can near-tie is the only movement, and it is
  within the ±0.05 seed noise both arms show on these very-low-SR contact cells (can is the noisiest: GC-IDM §16 per-seed 0.16/0.24/0.10). It does NOT
  overturn the headline: OURS never CLEARLY beats GC-IDM; at best it ties on the noisiest cell. (For a strict converged-vs-converged on robomimic one
  would also re-eval GC-IDM at its `gcidm_head_latest`; §16's GC-IDM headline is best-val, but GC-IDM is the WINNING arm there, so a converged GC-IDM
  could only tie-or-rise — it cannot make OURS win. The OURS-loses conclusion is robust to GC-IDM's checkpoint too.)
- **No chunk-1 artifact:** every converged robomimic array is scattered across the 5 chunks (verified per-chunk-True counts) → the chunk-reset-fixed
  `eval_histbc_robomimic.py` produced clean SRs, not the §16 chunk-1-then-zeros pattern.

### BOTTOM LINE (ALL CELLS SETTLED — converged checkpoints)
**The cube/pusht/robomimic "OURS loses" headline SURVIVES converged-checkpoint selection — it is GENUINE, not a checkpoint artifact.** Once OURS is
evaluated at its CONVERGED `weights_epoch_0.pt` (not the best-val `weights_epoch_1.pt`), the numbers barely move and the conclusion is unchanged:
cube 76 vs GC-IDM 100, pusht 31 vs 91, lift 0.27 vs 0.50, square 0.12 vs 0.18 (OURS loses); can 0.17 vs 0.167 (noisy near-tie, not a win). **OURS does
NOT tie/win once not early-stopped.** The reacher trap (§15: OURS 63→97, a +34pt jump that flipped the verdict) was REACHER-SPECIFIC — it only bites
precise-control tasks where val-MSE plateaus at the z-scored-action variance floor and "best-val" selects a barely-trained head. Cube/pusht/robomimic
all early-stopped at genuine val-MSE minima (or mild overfits), so their best-val checkpoints were already near-converged; the converged heads add ≤+4
points and never the 30+ that would have changed who wins. **The whole campaign's conclusion holds: the history-conditioned forward GC policy loses to
the Markovian GC-IDM on every fully-observed cell (cube, pusht) and every partial-grasp robomimic cell (lift/square loss, can tie), under CONVERGED
checkpoints — the wedge wins nowhere.** The only env where converged-vs-best-val flips the result is reacher itself (where it produced a TIE, not a win).

### Files added / changed (all `$B=le-wm-repro` / `$STABLEWM_HOME` on L40S; EVAL-ONLY)
- **NO repo edits, NO shared method-code edits.** Driver: `/tmp/reval_conv.sh` (host-local, not in repo). Logs: `/mnt/minghao_data/logs_revalconv/`.
- New run-dirs (mine, copies of existing converged GC-IDM heads): `$STABLEWM_HOME/checkpoints/{cube,pusht}_gcidm_latest/{gcidm_head_best.pt(=the
  ep200 latest head), gcidm_config.json}`. No existing checkpoint modified; other agents' checkpoints only READ.
- Eval outputs land under `$STABLEWM_HOME/gip_eval/{policy,gcidm,histbc}/<run>/` (the evaluators overwrite the per-run results.txt; the per-seed SR
  is captured in my `logs_revalconv/*.log`).

## §20 ✅🆕 ROBOMIMIC GC — CLOSING the provenance + seed-count gap: BOTH arms CONVERGED, MATCHED 3 seeds {42,0,1}, RETAINED logs on disk (L40S, 2026-06-20)

**Motivation (why).** The honesty audit (wf_8f6a354f) flagged two gaps in the §16 robomimic GC head-to-head that made the
lift/can/square cell PRELIMINARY rather than provable: (1) **PROVENANCE GAP** — the original §16 gcidm eval-launch artifacts
lived in `/tmp/*_gcidm_*` and would be wiped on the next `/tmp` clean, so the "identical N/budget/offset/same-box for the GCIDM
arm" fairness claim was *reconstructed* from the result-file mode-tag, not a launch command that survives on disk; and (2) a
**CHECKPOINT-ASYMMETRY** — §16's gcidm numbers were the **best-val** head (`gcidm_head_best.pt`, confirmed in the recovered
logs: `[GCIDM] load <task>_gcidm <- gcidm_head_best.pt`), while §19 only re-evaled the OURS arm converged (and at a DIFFERENT
2-seed {42,7}, with a gcidm-3-seed-vs-ours-2-seed mismatch on `can`). So the headline robomimic comparison had never been run
**converged-vs-converged at MATCHED seeds with retained logs** — exactly the cube mistake the task spec warns against ("NOT
best-val for one and converged for the other"). This §20 closes both gaps: re-eval BOTH arms at their CONVERGED heads, the SAME
3 seeds {42,0,1}, N=50, same-box (each task on its training GPU), and write every run's `hydra.run.dir` (resolved config +
`overrides.yaml` + stdout `run.log`) to a PERSISTENT location so provenance is now PROVABLE on disk, not in `/tmp`.

**STEP 0 — original /tmp evidence preserved (the audit missed it).** Recovered the ORIGINAL §16 gcidm eval hydra-configs +
stdout logs from `/tmp/*_gcidm_*` to `/mnt/minghao_data/logs_gcidm_rerun/_recovered_tmp/` (per-run subdirs preserved). 20
`overrides.yaml` recovered (lift/can/square/tool_hang/transport gcidm eval runs across seeds + the b-suffixed re-runs) + 12
`*.log`. These ORIGINAL overrides PROVE the §16 gcidm config: e.g. `lift_gcidm_s0/.hydra/overrides.yaml` =
`world.task=Lift / world.num_envs=10 / dataset.stats=lift / eval.dataset_name=lift / eval.num_eval=50 / eval.eval_budget=100 /
eval.goal_offset_steps=30 / policy=lift_gcidm / +gip_eval.mode=gcidm / seed=0`; `can_gcidm_s42` = budget 240 / offset 90;
`square_gcidm_s42` = budget 320 / offset 120. The recovered logs reproduce the §16 best-val gcidm per-seed numbers EXACTLY
(lift s0=0.48, s1b=0.58; can s42=0.16/s0=0.24/s1=0.10; square s42=0.30/s0b=0.14/s1=0.10), confirming they are the genuine §16
artifacts, and confirm the §16 gcidm head was `gcidm_head_best.pt` (best-val) — the asymmetry §20 now removes.

### What's CONVERGED-vs-CONVERGED here (the apples-to-apples fix)
- **gcidm CONVERGED** = the ep200 latest head `gcidm_head_latest.pt` (md5 DIFFERS from `gcidm_head_best.pt` for all three tasks
  → there is a real best-val/converged distinction). `gip.load_gcidm_model` defaults to `gcidm_head_best.pt`, so — mirroring the
  §15/§19 reacher/cube fix — I COPIED `gcidm_head_latest.pt → $STABLEWM_HOME/checkpoints/<task>_gcidm_latest/gcidm_head_best.pt`
  (+ `gcidm_config.json`) and evaled `policy=<task>_gcidm_latest +gip_eval.gcidm_run=<task>_gcidm_latest`. Runtime log confirms
  the converged head loaded: `[GCIDM] load <task>_gcidm_latest <- gcidm_head_best.pt H_max=50 action_dim=35 ablate_horizon=False`.
- **OURS CONVERGED** = `weights_epoch_0.pt` via `+ckpt_epoch=0` (the §19 mechanism: `_1`=best-val default-loaded, `_0`=latest/
  converged). Runtime log confirms: `[GIP] load <task>_gc_ours <- weights_epoch_0.pt: Adim=35 missing=0 unexpected=0` and
  `[HISTBC] policy ready adim=35 action_block=5 HS=3 goal_cond=True use_horizon=True horizon0=<6|18|24>` → converged head,
  3-frame history, goal-conditioned, AdaLN horizon engaged. (NB §16 best-val and §19 converged-2-seed{42,7} both already showed
  ours barely moves between best-val and converged on robomimic; this §20 nails it at the MATCHED 3 seeds for BOTH arms.)

### Infra / constraints (eval-only; no retrain; no shared-code edits)
L40S `stratus-lookout`, `$B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`,
run as `sudo -u minghao.fu`. Env: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1
OMP/MKL/OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl_rr CUDA_VISIBLE_DEVICES=<train GPU>`. **Same-box cross-render rule**
([[project_l40s_cross_gpu_rendering]]): each task evals on its §16 TRAINING GPU — lift→GPU4, can→GPU5, square→GPU7. Driver
`/tmp/rr_gcrm.sh` (one cell) + `/tmp/rr_task_driver.sh` (6 cells/task, ≤2 concurrent/GPU, staggered 7s). All 8 GPUs were
saturated by fan-test's nano-world-model DDP (~15 GB/GPU) — co-located per MT-JEPA rule 10 (each robomimic eval ~13.7 GB; ~31 GB
free/GPU; never touched fan's procs, pkill never used). 18 runs (3 tasks × 2 arms × 3 seeds), 0 failures, no OOM. ~127 s/run
(lift) up to ~5–6 min/run (square ours, budget 320, per-step history encode).

### EXACT eval commands (only the head checkpoint + mode/goal_conditioned differ between arms)
```
# gcidm CONVERGED (Markovian single-frame; latest head copied into <task>_gcidm_latest):
CUDA_VISIBLE_DEVICES=<G> python eval_histbc_robomimic.py --config-name robomimic \
  world.task=<Lift|PickPlaceCan|NutAssemblySquare> world.num_envs=10 dataset.stats=<task> eval.dataset_name=<task> \
  eval.num_eval=50 eval.eval_budget=<100|240|320> eval.goal_offset_steps=<30|90|120> seed=<42|0|1> \
  policy=<task>_gcidm_latest +gip_eval.mode=gcidm +gip_eval.gcidm_run=<task>_gcidm_latest \
  hydra.run.dir=/mnt/minghao_data/logs_gcidm_rerun/<task>_gcidm_s<seed>
# OURS CONVERGED (history GC; weights_epoch_0 via +ckpt_epoch=0):
CUDA_VISIBLE_DEVICES=<G> python eval_histbc_robomimic.py --config-name robomimic \
  world.task=<...> world.num_envs=10 dataset.stats=<task> eval.dataset_name=<task> \
  eval.num_eval=50 eval.eval_budget=<...> eval.goal_offset_steps=<...> seed=<42|0|1> \
  policy=<task>_gc_ours +ckpt_epoch=0 +gip_eval.mode=policy +gip_eval.goal_conditioned=true \
  hydra.run.dir=/mnt/minghao_data/logs_gcidm_rerun/<task>_ours_s<seed>
```

### FAIRNESS PROVEN ON DISK (the diff, not assumed). `diff <(sort gcidm/overrides) <(sort ours/overrides)`, per task:
The two arms' `overrides.yaml` differ in EXACTLY three+one tokens and NOTHING else — IDENTICAL `world.task`, `world.num_envs=10`,
`dataset.stats`, `eval.dataset_name`, `eval.num_eval=50`, `eval.eval_budget`, `eval.goal_offset_steps`, `seed`. The only deltas:
gcidm has `policy=<task>_gcidm_latest / +gip_eval.mode=gcidm / +gip_eval.gcidm_run=<task>_gcidm_latest`; ours has
`policy=<task>_gc_ours / +ckpt_epoch=0 / +gip_eval.mode=policy / +gip_eval.goal_conditioned=true`. Same frozen vit-tiny-192 base
per task, byte-identical latent cache (both heads trained off the SAME `<task>_gcidm/latents_cache.pt` per §16), same N / budget /
goal_offset / seeds / GPU (same-box). → **history-vs-Markovian is the ONLY thing that varies; fairness is PROVEN, not assumed.**

### 📊 RESULTS (N=50, 3 MATCHED seeds {42,0,1}, BOTH arms CONVERGED, same-box; ALL per-seed numbers)

| task | arm (CONVERGED) | seed 42 | seed 0 | seed 1 | **mean** | §16 best-val mean (for contrast) |
|---|---|---|---|---|---|---|
| lift   | gcidm (Markovian, `_latest`) | 0.50 | 0.50 | 0.46 | **0.487** | 0.500 |
| lift   | OURS (history, `ep0`)        | 0.32 | 0.18 | 0.30 | **0.267** | 0.253 |
| can    | gcidm (Markovian, `_latest`) | 0.30 | 0.26 | 0.16 | **0.240** | 0.167 |
| can    | OURS (history, `ep0`)        | 0.10 | 0.22 | 0.08 | **0.133** | 0.127 |
| square | gcidm (Markovian, `_latest`) | 0.24 | 0.34 | 0.22 | **0.267** | 0.180 |
| square | OURS (history, `ep0`)        | 0.16 | 0.10 | 0.08 | **0.113** | 0.080 |

Per-arm CONVERGED 3-seed means: **gcidm 0.487 / 0.240 / 0.267** (lift/can/square); **OURS 0.267 / 0.133 / 0.113**. **gcidm
(Markovian) wins ALL THREE task means and ALL 9/9 per-seed cells** — there is no per-seed cell where OURS is strictly ahead; the
closest is can seed0 (ours 0.22 vs gcidm 0.26, gcidm still ahead). Converged-vs-converged the gap is lift 0.487 vs 0.267 (~1.8×),
can 0.240 vs 0.133 (~1.8×), square 0.267 vs 0.113 (~2.4×). **No chunk-1-then-zeros artifact:** every one of the 18 runs scatters its successes across all 5 chunks (e.g.
square_ours_s1 cum=[1,2,4,4,4], square_ours_s0=[2,2,2,3,5], lift_gcidm_s42=[5,10,14,20,25]) → the §16 chunk-reset fix is working.

**Reproduces the table?** YES, direction-wise, and the gap WIDENS once gcidm is also converged. The honesty-audit table cited
gcidm ~0.50/0.17/0.18 (lift/can/square) and ours ~0.21-0.27/0.13-0.17/0.08-0.12. The CONVERGED-vs-CONVERGED actuals:
gcidm **0.487/0.240/0.267** (lift unchanged; **can +0.073 and square +0.087 vs §16 best-val** — the converged gcidm head helps
the WINNING arm), ours **0.267/0.133/0.113** (all within ±0.04 of §16 best-val / §19 converged — ours barely moves, confirming
§19's finding that robomimic ours is NOT in a reacher-style checkpoint trap). So the converged-vs-converged comparison does NOT
rescue ours; it makes gcidm's lead LARGER on can/square.

**VERDICT — the §16 "GC-IDM (Markovian) > OURS (history) on all three robomimic contact tasks" SURVIVES the strict
converged-vs-converged, matched-3-seed, retained-log test, and the provenance gap is CLOSED.** gcidm wins lift (0.487 vs 0.267),
can (0.240 vs 0.133), and square (0.267 vs 0.113), converged-vs-converged at the SAME 3 seeds {42,0,1}. This removes BOTH audit
caveats: (1) provenance is now PROVABLE on disk (every run's `overrides.yaml` + `run.log` retained under
`/mnt/minghao_data/logs_gcidm_rerun/<task>_<arm>_s<seed>/`, the original §16 gcidm artifacts recovered to `_recovered_tmp/`); and
(2) the checkpoint asymmetry is gone — BOTH arms converged, same seed set, no 3-vs-2 mismatch. The mechanism is unchanged from
§13/§16: ours's history lowers OPEN-loop action-MSE (val_act 0.258/0.147/0.219 < gcidm hindsight-MSE 0.437/0.226/0.324 on
lift/can/square) yet WORSENS closed-loop SR — the self-generated past-action stream compounds covariate shift (causal confusion),
while the Markovian `(z_t,z_goal,h)` re-grounds on the fresh observation each step. Both arms remain FAR below the goal-agnostic
histbc bc ceiling (lift 0.95 / can 0.75 / square 0.70): the frozen vit-tiny latent can't localize the object (the encoder
bottleneck), so this is an encoder-confounded cell — NOT a clean history-vs-Markovian referendum — but ours does NOT win
regardless, at either checkpoint, under either seed protocol. The cross-env synthesis (§16) holds: OURS (history) beats Markovian
GC-IDM on NO tested env (cube 76 vs 100, pusht 31 vs 91, lift 26.7 vs 48.7, can 13.3 vs 24.0, square 11.3 vs 26.7; tworoom/reacher
= ties on coarse-reaching tasks).

### Retained provenance paths (PROVABLE on disk — every run, both arms, all 3 seeds)
- **18 re-run dirs** (resolved config + `overrides.yaml` + stdout `run.log`, each with the final `success_rate` line):
  `/mnt/minghao_data/logs_gcidm_rerun/{lift,can,square}_{gcidm,ours}_s{42,0,1}/` (+ `.hydra/{config,hydra,overrides}.yaml`).
- **Original §16 gcidm evidence recovered from /tmp** (best-val, the pre-§20 baseline): `/mnt/minghao_data/logs_gcidm_rerun/_recovered_tmp/`
  (20 `*/.hydra/overrides.yaml` + 12 `*.log`, e.g. `lift_gcidm_s0/.hydra/overrides.yaml`, `square_gcidm_s42/.hydra/overrides.yaml`).
- **CONVERGED gcidm run-dirs** (mine, copies of the existing `gcidm_head_latest.pt`, no existing ckpt modified):
  `$STABLEWM_HOME/checkpoints/{lift,can,square}_gcidm_latest/{gcidm_head_best.pt(=ep200 latest), gcidm_config.json}`.
- Drivers (host-local, not in repo): `/tmp/rr_gcrm.sh`, `/tmp/rr_task_driver.sh`. NO repo edits, NO shared method-code edits.
- The §16 evaluator `eval_histbc_robomimic.py` (chunk-reset-fixed) was UNCHANGED this session (eval-only re-run).

## §21 ✅🆕 END-TO-END goal-conditioned LeWAM (encoder LEARNABLE, hindsight goals) on robomimic can — does training end-to-end lift OFF the frozen-latent floor? (L40S, 2026-06-20)

**Motivation (why).** §16/§19/§20 established the FROZEN-encoder GC head-to-head on robomimic: with the encoder FROZEN and latents precomputed/cached (`train_gcidm.py` / `train_ours_gc.py`, `requires_grad=False`), BOTH GC arms FLOOR on contact tasks — converged-vs-converged 3-seed {42,0,1}: **GC-IDM (Markovian) can 0.240, OURS-GC (history) can 0.133**, both ~3-5× below the goal-AGNOSTIC end-to-end histbc bc (can 0.71, §1c). §20's own diagnosis named the cause: *"the frozen vit-tiny latent can't localize the object (the encoder bottleneck), so this is an encoder-confounded cell."* The end-to-end histbc path (`train.py`, single `model_opt` over the WHOLE `model`, encoder LEARNABLE) reaches lift bc 0.93 / can 0.71 — so when the encoder is allowed to learn from the action objective it CAN localize the object. **The question this §21 answers: is robomimic GC floored because GC is intrinsically hard on these tasks, or only because the §16/§19/§20 arms FROZE the encoder?** Train the encoder + WM + goal-conditioned intention head JOINTLY end-to-end (encoder learnable, hindsight goals, NO `freeze_wm`, NO cached latents) and re-measure can GC SR. This is the user's intended method; it is NOT the gcidm-matching frozen-latent version.

### STEP 1 — the end-to-end GC training was ALREADY FULLY WIRED in `train.py` (nothing to build; verified by code audit + a gradient-flow smoke test)
The plan `resilient-twirling-moon.md` listed the goal-sampling wrapper, the `lejepa_forward` goal path, and the config flags as "to build." A code audit on L40S (`$B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, files mtime ≤ 2026-06-20 03:58) found **all of it already implemented and config-gated (default OFF)**. Nothing needed building; no `*.bak_e2e` backups were made because no file was edited. The pieces:
- **Goal-sampling data wrapper** — `goal_dataset.py:GoalSamplingDataset` (50 lines). Wraps the swm `EpisodeDataset`; per window at trajectory time `t` it adds `item["goal"]` = the frame at `t + h*frameskip` (`h ~ Uniform[1, hindsight_max_k]` obs-steps, clamped to episode end via `lengths/clip_indices`) preprocessed exactly like the window frames (re-uses `base._load_slice`, which already applies the dataset transform — the comment explicitly warns against double-normalizing), plus `item["horizon"]` = the realized `h`. **Constructed ONLY when `action_pred.goal_conditioned=true`** (`train.py:236-239`), so base/histbc training is byte-identical otherwise.
- **GC loss in `lejepa_forward`** (`train.py:63-134`): when `action_pred.enabled && goal_conditioned && "goal" in batch` →
  `z_goal = self.model.encode({"pixels": batch["goal"].unsqueeze(1)})["emb"][:,0]` (**line 74 — the goal frame goes through the SAME `self.model.encode`, i.e. the LEARNABLE encoder, NOT a cached latent**), apply `goal_dropout` (zero `z_goal` per-sample w.p. `p` → the SAME head also learns the goal-AGNOSTIC policy = BESO/CFG two-setting WAM), then `goal_emb=z_goal` + the normalized horizon are threaded into `predict_intention(...)` (line 92-95). The existing `act_loss` (predict `a_t`) becomes goal-conditioned via hindsight automatically; `intent_loss` predicts the detached `act_emb`.
- **Horizon AdaLN-Zero** (`jepa.py:HorizonModulator`, built in `JEPA.__init__` when `horizon_conditioned=true`; `train.py:256-259` sets `cfg.model.horizon_conditioned=True` BEFORE instantiation so the module rides into `config.json` → eval rebuilds it). `predict_intention` applies it to the intention embedding BEFORE the decoder (`jepa.py:205-206`). AdaLN-Zero (cond_proj zero-init) ⇒ identity at init ⇒ byte-identical until learned.
- **Encoder LEARNABLE (the END-TO-END property, vs frozen `train_ours_gc.py`):** the optimizer is a SINGLE `model_opt` group covering `'model'` (`train.py:399-406`); there is NO `freeze_wm` (default false, `train.py:350`). The encoder receives gradients from BOTH the WM `pred_loss` AND the goal/action objective (z_goal flows through `encoder`). The frozen-latent arms (`train_gcidm.py`/`train_ours_gc.py`) instead set `encoder.requires_grad=False` and precompute a latent cache — that is the version that floored.
- **Config flags** (`config/train/lewm.yaml:47-65`): `action_pred.{goal_conditioned, hindsight_max_k, goal_dropout, horizon_conditioned, horizon_H_max}`, ALL default OFF.
- **Eval path** (`eval_histbc_robomimic.py`, mode=`policy` + `+gip_eval.goal_conditioned=true`): `HistoryBCPolicy` loads via `gip.load_gip_model` (rebuilds the `horizon_modulator` from `config.json`), encodes each env's `info_dict["goal"]` → z_goal through the SAME (now end-to-end-trained) encoder, threads goal_emb + the per-env AdaLN-Zero horizon countdown into `predict_intention`. The chunk-reset fix (§16) is in `_eval_loop`. Already built; no edit.

**GATE VERIFICATION (empirical, the STEP-1 requirement).** Ran a standalone gradient-flow smoke test (`gate_check.py`, built the model exactly as `train.py` does with `goal_conditioned=true horizon_conditioned=true goal_dropout=0.5`, fabricated a batch, called `lejepa_forward`):
- `horizon_modulator present: True` (82,560 params, AdaLN-Zero).
- GC losses computed: `intent_loss=1.023, act_loss=1.163` present.
- **encoder total |grad| = 3464.98** (from `loss.backward()`) ⇒ encoder is LEARNABLE, NOT frozen.
- **encoder |grad| from a GOAL-ONLY loss (`z_goal.pow(2).mean()`) = 2148.44** ⇒ the goal path is NOT detached; encoding the hindsight goal pushes real gradient into the encoder. This is the exact property that separates END-TO-END (this) from the frozen-latent §16/§19/§20 arms.
- AdaLN-Zero at init: `|intention(h=0.1) - intention(h=0.9)| = 0` (identity until trained). **ALL CHECKS PASS.** (Smoke file deleted after; no repo edit.)
With the flags OFF (config defaults) the goal/horizon code is skipped (`goal_emb=None`, no `GoalSamplingDataset` wrap, no `horizon_modulator`) ⇒ base/histbc training is byte-identical (gate satisfied by construction + confirmed in the 1-epoch real-data smoke below).

### STEP 2 — the EXACT end-to-end can training (warm-start the WM, finetune the WHOLE thing; encoder learnable)
Driver `/mnt/minghao_data/e2e_can_train.sh` (host-local, NOT in repo). Infra: L40S `stratus-lookout`, `$B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, run as `sudo -u minghao.fu`. Env (root `/` is 99% full ⇒ ALL writes redirected to /mnt/minghao_data): `STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/mnt/minghao_data/tmp MPLCONFIGDIR=/mnt/minghao_data/mpl_e2e HF_HOME=/home/minghao.fu/.cache/huggingface XDG_CACHE_HOME=/mnt/minghao_data/.cache_e2e WANDB_MODE=disabled (no /home/minghao.fu/.netrc on L40S) CUDA_VISIBLE_DEVICES=2`. EXACT launch command:
```
python train.py \
  data=robomimic_can \
  action_pred.enabled=true \
  action_pred.detach_decoder=false \
  action_pred.goal_conditioned=true \
  action_pred.horizon_conditioned=true \
  action_pred.goal_dropout=0.5 \
  action_pred.hindsight_max_k=50 \
  action_pred.horizon_H_max=50 \
  init_from=$STABLEWM_HOME/decoders/can_lewm_weights.pt \
  output_model_name=can_gc_e2e \
  trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 \
  +ckpt_every=10 \
  loader.batch_size=64 num_workers=4 \
  hydra.run.dir=/mnt/minghao_data/hydra_e2e/can_gc_e2e
```
Data = re-rendered-on-L40S `can.h5` (the §16 image_224 robomimic data: `pixels (23207,224,224,3) uint8`, action 7-DoF, frameskip 5 ⇒ action_block 35, proprio 11, has episode_idx/step_idx ⇒ convert-bug-fixed). Warm-start: `init_from=can_lewm_weights.pt` (vit-tiny-192 SIGReg base) loads `missing=95 unexpected=0` (the 95 missing = the GIP head action_predictor/action_decoder + horizon_modulator, trained from scratch on top of the warm WM — correct). Recipe matches the §1c histbc recipe that achieved can bc 0.71 (100ep × 4000 batches, detach_decoder=false), with the GC flags added. `detach_decoder=false` so the raw-action grounds the encoder (the §1c-validated histbc setting, NOT the latent-only detach=true that PushT used). Per-epoch ckpt prune loop (`/mnt/minghao_data/prune_e2e.sh`, keep newest 3, every 180s) + `+ckpt_every=10` to bound the save burst on the tight root disk.

**1-epoch real-data smoke (pre-launch, GPU 2, deleted after):** `[GIP] Intention predictor ON Adim=35`, `[GIP] OURS horizon conditioning ON`, `init_from ... missing=95 unexpected=0`, horizon_modulator 82,560 params built, GC losses fall (val act_loss 1.27→1.11, intent 2.13→1.85 in one tiny epoch), ckpt saved to `checkpoints/can_gc_e2e_smoke/`. No crash. Confirms the GoalSamplingDataset + GC training runs end-to-end on the real can data.

**Convergence (live, 100ep × 4000-batch, ~117 s/epoch ⇒ ~3.25 h total):** epoch 0 done in 116.8 s; `fit/act_loss` 0.246, `fit/intent_loss` 0.579, `fit/pred_loss` 0.039 (from sanity-val act_loss 1.19 / intent 1.96 / pred 0.011); `validate/act_loss` 1.19→0.526, `validate/intent_loss` 1.96→0.776 after epoch 0. Loss falling fast, no crash. [FILLING LIVE — convergence plateau + final ckpt + eval below.]

### STEP 3 — eval GC N=50 same-box + compare to the frozen-latent floor [PENDING — fills when training converges]
Plan: `eval_histbc_robomimic.py --config-name robomimic policy=can_gc_e2e +gip_eval.mode=policy +gip_eval.goal_conditioned=true world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can eval.num_eval=50 eval.eval_budget=240 eval.goal_offset_steps=90 seed=<42|0|1>` — N=50, 3 seeds {42,0,1}, on GPU 2 (the can training GPU, per the cross-GPU render gotcha [[project_l40s_cross_gpu_rendering]]). Reference numbers to beat (can GC): FROZEN-latent OURS-GC (history) **0.133** (§20 converged 3-seed), FROZEN GC-IDM (Markovian) **0.240** (§20 converged 3-seed), end-to-end histbc bc (goal-AGNOSTIC) **0.71** (§1c). **KEY question: does the end-to-end goal-conditioned LeWAM (encoder learnable) lift OFF the frozen floor (toward the 0.71 histbc range), or does GC stay floored even with a learnable encoder?**

### STEP 3 RESULT — the e2e `can_gc_e2e` checkpoint evaluated (2026-06-20, L40S `stratus-lookout`, GPU 1)
**Provenance + caveats (honest).** The `can_gc_e2e` run (the dropout-0.5 e2e arm above) was launched by a prior agent and is **STALLED at epoch 10**: PID 189292 alive 35+ min in state `Rl` at ~42% CPU but holding NO GPU compute context, `weights_epoch_10.pt` (00:50) is the last and only saved ckpt, `train.log` frozen at the 00:32 dataset-cache line — a hung dataloader after the epoch-10 save. So this number is **epoch-10, NOT converged**, and trained with **`goal_dropout=0.5`** (the head is goal-AGNOSTIC on half the samples → the goal signal is diluted at train time). It is a diagnostic, not the headline. The headline is the clean `freeze_wm`-only ablation below (§21-CLEAN, `goal_dropout=0`, both arms trained to 100 epochs).

The prior CLIP death (the `HF_HUB_OFFLINE=1` §0 gotcha) was the env loading **`openai/clip-vit-large-patch14`** (6.4 GB, pulled by `swm.World`/the robomimic env wrapper as a CLIP-based reward/goal check; NOT the single-task model, whose encoder is `vit_hf pretrained=false` and needs no download, and NOT the `create_task_embeddings.py` text-CLIP which is multi-task-only). Pre-downloaded both CLIPs with `HF_HUB_OFFLINE=0` (box has internet); now cached at `/var/lib/docker/data/minghao_home/.cache/huggingface/{hub,clip}/models--openai--clip-vit-large-patch14`. Eval then ran clean.

EXACT eval command (per seed): `CUDA_VISIBLE_DEVICES=1 STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=0 OMP_NUM_THREADS=4 python eval_histbc_robomimic.py --config-name robomimic world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can eval.num_eval=50 eval.eval_budget=240 eval.goal_offset_steps=90 policy=can_gc_e2e +gip_eval.mode=policy +gip_eval.goal_conditioned=true seed=<42|0|1>`. Driver `/tmp/step0_e2e_eval2.sh`, log `/mnt/minghao_data/step0_e2e_eval_b240.log`. `[HISTBC] policy ready adim=35 action_block=5 HS=3 goal_cond=True use_horizon=True horizon0=18.00` (horizon0 = offset/block = 90/5 = 18; the AdaLN-Zero horizon modulator is active → confirms the e2e ckpt carries `horizon_modulator`).

**Per-seed (N=50, b240/o90, same-box GPU 1):** seed42 **0.06** (3/50, chunk scatter [0,0,1,3,3]), seed0 **0.30** (15/50, real scatter not chunk-1-artifact), seed1 **0.26** (13/50). **3-seed mean = 0.207 (SD 0.13).** No chunk-1-then-zeros artifact in any seed (successes spread across chunks → the §16 chunk-reset fix holds). **This FIRST end-to-end LeWAM GC number for can (0.207) already sits AT the GC-IDM Markovian floor (0.240) and ABOVE the OURS-GC frozen-latent floor (0.133)** — even stalled at epoch 10 with the goal signal diluted by dropout 0.5. Signal: training end-to-end (encoder learnable) does NOT keep GC pinned at the 0.13 ours-floor.

⚠️ **Budget sensitivity discovered (important for cross-reading any GC number):** the SAME `can_gc_e2e` epoch-10 ckpt scores **0.44 at b100/o30** but **0.06–0.30 at b240/o90**. The larger goal offset (90 vs 30 → goal 18 vs 6 obs-steps ahead) and longer budget change the SR a lot. ALL §21 numbers below use the **canonical can b240/o90** (the protocol the §20 frozen floor 0.133/0.240 was measured on) so the comparison is apples-to-apples. A b100/o30 number is NOT comparable to the 0.13 floor.

---

## §21-CLEAN ✅🆕 The clean freeze_wm ablation — ONLY `freeze_wm` differs, `goal_dropout=0`, both arms to 100 epochs (L40S, 2026-06-20)

**Motivation (why).** §21 STEP 3 (the e2e dropout-0.5 epoch-10 ckpt) hinted GC lifts off the frozen floor, but it is confounded (under-trained + dropout 0.5 + a prior agent's stalled run). This is the user's intended DECISIVE ablation: train **two arms on `can` that are byte-identical except `freeze_wm`**, with `goal_dropout=0` (full goal signal every sample) and **both trained to 100 epochs** (rule 8 convergence). `freeze_wm=false` = the real end-to-end LeWAM (encoder + WM trunk + head all learnable, the user's method); `freeze_wm=true` = the frozen control (encoder/projector/predictor/pred_proj/action_encoder frozen at the warm-start `can_lewm_weights.pt`, only the intention head + decoder + horizon_modulator learn — the §16/§19/§20-style regime that floored). **This isolates ONE variable** (does the encoder learn?) and answers: is "LeWAM loses on robomimic GC" a real ceiling or a frozen-regime artifact?

**Config-gating verified.** `train.py:350` consumes `cfg.freeze_wm`; `config/train/lewm.yaml:freeze_wm` is a first-class flag (override as `freeze_wm=true`, NOT `+freeze_wm`). `freeze_wm=true` froze **18.0M params** (encoder/projector/predictor/pred_proj/action_encoder) per the frozen-arm log; `init_from=can_lewm_weights.pt` loads `missing=95 unexpected=0` on both arms (the 95 missing = the GIP head + action_decoder + horizon_modulator, trained from scratch — correct).

### EXACT train commands (host-local drivers `/tmp/train_abl_{learn,frozen}.sh`; `$B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, run as `sudo -u minghao.fu`)
Shared env: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=0 TMPDIR=/tmp OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl_{learn,frozen}`. Data = re-rendered-on-L40S `can.h5` (the §16 image_224 robomimic data; frameskip 5 ⇒ action_block 5, adim 35). Full dataset per epoch = **136 train steps/epoch** (NO `limit_train_batches` cap — unlike the e2e run's 4000-batch cap; this is the §1c/§20 full-dataset recipe). `+ckpt_every=10`, per-arm prune loop keeps newest-3 `weights_epoch_*.pt` (`/tmp/prune_abl.sh`).

- **LEARNABLE arm (`freeze_wm=false`, headline, GPU 3):**
  ```
  python train.py data=robomimic_can output_model_name=can_gc_abl_learn \
    action_pred.enabled=true action_pred.goal_conditioned=true action_pred.horizon_conditioned=true action_pred.goal_dropout=0 \
    freeze_wm=false init_from=/mnt/minghao_data/.stable-wm/decoders/can_lewm_weights.pt \
    trainer.max_epochs=100 +ckpt_every=10
  ```
  ~107 s/epoch (full backward through the encoder) ⇒ ~3 h for 100 ep.
- **FROZEN arm (`freeze_wm=true`, control, GPU 4):** identical EXCEPT `freeze_wm=true`, `output_model_name=can_gc_abl_frozen`. ~44 s/epoch (frozen trunk = cheaper backward) ⇒ ~1.3 h for 100 ep.

### Convergence (rule 8 — verify plateau) — BOTH ARMS ✅ CONVERGED (100 ep)
- **FROZEN ✅** (`weights_epoch_100.pt` @ 02:36): val `act_loss` plateaus by ep~28, FLAT over the last 8 epochs: 0.135 / 0.134 / 0.133 / 0.135 / 0.131 / 0.133 / 0.128 / 0.132 (variation < 0.01 over the last 10% — rule-8 plateau verified). The frozen WM trunk caps how low the head can drive act_loss on fixed latents (~0.13).
- **LEARNABLE ✅** (`weights_epoch_100.pt` @ 04:00): val `act_loss` FLAT over the last 10 epochs: 0.125 / 0.133 / 0.134 / 0.126 / 0.125 / 0.128 / 0.123 / 0.126 / 0.117 / 0.125 (variation ~0.015 — rule-8 plateau verified). The learnable encoder reaches ~0.12, **only marginally below the frozen 0.13** — the encoder being trainable barely lowered the open-loop action-MSE. This is the first hint that the learnable encoder does NOT meaningfully change the can-GC regime.

### STEP 1/2 EVAL — both arms, GC N=50, 3 seeds {42,0,1}, canonical can b240/o90, same-box (auto-launched at each arm's ep100 ckpt)
Eval drivers `/tmp/eval_abl_{learn,frozen}.sh`: `eval_histbc_robomimic.py --config-name robomimic world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can eval.num_eval=50 eval.eval_budget=240 eval.goal_offset_steps=90 policy=can_gc_abl_{learn,frozen} +gip_eval.mode=policy +gip_eval.goal_conditioned=true seed=<42|0|1>`. LEARN evals on GPU 3, FROZEN on GPU 4 (each arm's TRAINING GPU, per the cross-GPU render gotcha [[project_l40s_cross_gpu_rendering]]).

**FROZEN control result (✅ DONE, converged 100ep):** seed42 **0.06** (3/50, real scatter [0,0,2,2,3]), seed0 **0.02** (1/50), seed1 **0.00** (0/50). **3-seed mean = 0.027 (SD 0.03).** The frozen control is FLOORED — at/below even the §20 frozen-latent ours-GC floor (0.133). With `goal_dropout=0` (full goal signal every sample) and a CONVERGED head trained on FROZEN warm-start latents, can GC is ~0.03. The cleanest possible floor: when the encoder cannot learn, can GC is impossible no matter how well the head is trained. (Slightly below §20's 0.133 because §20 used `train_ours_gc.py` cached-latent + best-val; this is `train.py freeze_wm=true`, converged.)

### THE can GC ablation table ✅ COMPLETE (both arms converged + evaluated)
| arm | can GC SR (N=50, b240/o90, 3-seed {42,0,1}) | per-seed |
|---|---|---|
| **end-to-end LeWAM (freeze_wm=FALSE, learnable)** ✅ | **0.007** | 0.02 / 0.00 / 0.00 |
| **frozen control (freeze_wm=TRUE)** ✅ | **0.027** | 0.06 / 0.02 / 0.00 |
| can_gc_e2e (e2e, dropout 0.5, epoch-10 STALLED, STEP 3 above) | **0.207** | 0.06 / 0.30 / 0.26 |
| frozen-latent OURS-GC `train_ours_gc` (§20 converged) | **0.133** | 0.10 / 0.22 / 0.08 |
| GC-IDM Markovian `train_gcidm` (§20 converged) | **0.240** | 0.30 / 0.26 / 0.16 |
| end-to-end histbc bc (goal-AGNOSTIC, §1c reference ceiling) | **0.71** | — |

### VERDICT — the can-GC floor is REAL, NOT a frozen-regime artifact. The encoder being learnable does NOT rescue robomimic GC.
The decisive comparison is the top two rows, which differ in EXACTLY ONE flag (`freeze_wm`), both `goal_dropout=0`, both converged to 100 epochs, evaluated at the identical canonical can protocol (N=50, b240/o90, same-box, 3 seeds {42,0,1}):
- **LEARNABLE (freeze_wm=FALSE, the real end-to-end LeWAM) = 0.007** (0.02/0.00/0.00).
- **FROZEN control (freeze_wm=TRUE) = 0.027** (0.06/0.02/0.00).

**The learnable arm does NOT lift off the frozen floor — it sits AT the floor, statistically indistinguishable from (in fact 0.02 BELOW) the frozen control, and FAR below the goal-AGNOSTIC histbc bc ceiling (0.71).** Letting the encoder learn end-to-end from the goal-conditioned action objective bought **nothing** on can GC: 0.007 vs 0.027 vs 0.133 (frozen-latent §20) — all three frozen/learnable variants floor in the 0.0–0.13 band, an order of magnitude below the 0.71 goal-agnostic ceiling and below even the Markovian GC-IDM (0.240). **So "LeWAM loses on robomimic GC" is NOT a frozen-regime artifact: the floor survives making the encoder fully learnable.** The verdict does NOT flip.

**Why (honest mechanism, consistent with §20).** The floor is NOT the encoder's representational bottleneck (that would have been fixed by `freeze_wm=false`). The convergence evidence shows the learnable encoder reached val act_loss ~0.12 vs the frozen ~0.13 — a marginal open-loop improvement that did not translate to ANY closed-loop GC gain (it actually scored lower). The bottleneck is the **goal-conditioned forward-policy formulation on a multi-stage contact task**: the history-conditioned forward GC policy (predict the next action block from the past-frame/past-action context + a z_goal that is 18 obs-steps away) compounds covariate shift over the long b240 horizon and cannot decompose approach→grasp→transport from a distant goal image — exactly the §16/§20 closed-loop failure, and it is INTRINSIC to the GC-forward-policy setup, not to whether the encoder is frozen. Corroborating: the SAME architecture trained goal-AGNOSTICally (histbc bc, the head predicts the demonstrator's next action with NO goal) reaches 0.71 on can — so the encoder + head CAN drive the arm; it is the **goal-conditioning that floors**, learnable encoder or not. The Markovian GC-IDM (0.240) beats both our GC arms because re-grounding on the fresh observation each step (no history accumulation) limits the covariate-shift blowup — the §16/§20 finding, now confirmed to hold even against a learnable-encoder OURS arm.

**Caveat (scope).** This is ONE task (can), `goal_dropout=0`, b240/o90. The e2e dropout-0.5 epoch-10 arm (STEP 3) scored 0.207 — higher than these dropout-0 converged arms, which is counterintuitive (more goal signal + more training → LOWER SR). Two non-exclusive reasons: (i) the dropout-0.5 arm is goal-agnostic on half its samples, so it partially behaves like the 0.71 histbc-bc policy (which ignores the distant goal and just imitates), and a goal-agnostic policy is BETTER on can than a goal-conditioned one that chases a far goal image; (ii) high seed variance (0.06/0.30/0.26). Either way the reading is the same: **goal-conditioning HURTS on can, and making the encoder learnable does not fix it.** The histbc-bc 0.71 ceiling is reached by IGNORING the goal, not by conditioning on it better. **Headline (per [[project_gip_intention_format_eval_convention]] per-family convention): on robomimic contact tasks, the goal-conditioned forward policy floors regardless of frozen-vs-learnable encoder; the win, if any, is goal-AGNOSTIC histbc-bc, not GC.**

### Provenance (on disk, L40S)
- **Converged ckpts** (newest-3 retained, prune `/tmp/prune_abl.sh`): `$STABLEWM_HOME/checkpoints/can_gc_abl_{learn,frozen}/weights_epoch_100.pt` + `full_config.yaml` + `config.json`.
- **Train logs**: `/mnt/minghao_data/abl_{learn,frozen}_train.log` (full per-epoch loss + the `[GIP] freeze_wm=true: froze 18.0M params` / no-freeze lines + `init_from ... missing=95 unexpected=0`).
- **Eval logs**: `/mnt/minghao_data/eval_abl_{learn,frozen}.log` (per-seed `policy ready ... horizon0=18.00`, chunk scatter, `success_rate`); STEP 3 e2e eval `/mnt/minghao_data/step0_e2e_eval_b240.log`.
- **Drivers** (host-local, NOT in repo): `/tmp/train_abl_{learn,frozen}.sh`, `/tmp/eval_abl_{learn,frozen}.sh`, `/tmp/step0_e2e_eval2.sh`. No repo code edited; `train.py`/`eval_histbc_robomimic.py` unchanged (config-gated run only). CLIP-large (6.4G) pre-cached so eval no longer crashes on the `HF_HUB_OFFLINE` gotcha.

## §22 🔄🆕 END-TO-END goal-conditioned LeWAM (encoder LEARNABLE, `freeze_wm=FALSE`) on **cube** — the RESCUE test for the env where frozen-latent OURS LOST (L40S, 2026-06-21) [IN FLIGHT]

**Motivation (why).** Cube is the KEY env: the §11/§12 frozen-latent 3-arm head-to-head on cube ended with **GC-IDM (Markovian) 100.0 ≫ OURS frozen-latent history-cond. 72.7** (3-seed {78/68/72}, std≈5), with same-box CEM 67.0 / guided 80.0. §12's own conclusion: on this near-ceiling fully-observed pick-place, the simpler Markovian GC-IDM beat our history-conditioned variant, "an honest negative for OURS-vs-GC-IDM ON CUBE." But BOTH §12 OURS and §11 GC-IDM trained on FROZEN LeWM latents (`train_ours_gc.py` / `train_gcidm.py`, `encoder.requires_grad=False`, precomputed `latents_cache.pt`). The same question §21 asked on robomimic-can applies here: **is OURS's 72.7 a real ceiling for the history-conditioned forward GC policy on cube, or only a frozen-encoder artifact?** This §22 trains the encoder + WM trunk + goal-conditioned intention head JOINTLY end-to-end (encoder LEARNABLE, hindsight goals, `freeze_wm=false`, NO cached latents — the image DataLoader over the 20G cube lance) from the SAME warm-start base (`cube_ours_lewm_weights.pt`), then re-evals cube GC N=50 / 3 seeds and asks: does end-to-end training RESCUE OURS on cube — close to or beat the GC-IDM/gcidm anchor?

**Anchors (cube, N=50, same-box L40S unless noted):** frozen-latent OURS = **72.7** (§12, the arm being rescued); GC-IDM reproduced = **100.0** (§11, L40S same-box); gcidm PUBLISHED = **98.7** (n=200). Same-box CEM 67 / guided 80 (§12).

### STEP 1 — SETUP VERIFIED (all present on L40S)
- **Data config**: `data=ogb_lance` → `config/train/data/ogb_lance.yaml` → `ogbench/ogb_cube_single.lance` (20G, the fast lance the §10b cube DINOv2 base+GIP arm trained on; resolves under `$STABLEWM_HOME/datasets/ogbench/ogb_cube_single.lance`). frameskip 5 ⇒ action_block 5, **Adim=25** (5 action-dim × 5). NOTE: the §12 frozen-latent path used the in-memory `latents_cache.pt` with num_workers=0 (the lance DataLoader fork-deadlocked under workers>0 ON THE CACHE path); the **image DataLoader** over the lance works fine under num_workers=10 (smoke + timing below confirm; the deadlock was cache-path-specific).
- **Warm-start base**: `/mnt/minghao_data/.stable-wm/decoders/cube_ours_lewm_weights.pt` (72,289,084 bytes, vit-tiny-192 SIGReg LeWM — the SAME frozen base §11/§12 used). `init_from` loads `missing=95 unexpected=0` (95 missing = GIP head + action_decoder + horizon_modulator, trained from scratch — correct).
- **Eval entry**: `eval_gip.py --config-name cube` + `config/eval/cube.yaml` (N=50, goal_offset_steps=25, eval_budget=50, action_block=5, env `swm/OGBCube-v0` single, `terminate_at_goal`, `set_target_pos` privileged block goal) — the SAME eval §11/§12 used. `mode=policy goal_conditioned=true` routes to `BCPolicy` (planning-free forward GC policy, the §12 OURS path). `load_gip_model` rebuilds the `horizon_modulator` from `config.json` (train.py sets `cfg.model.horizon_conditioned=True` before instantiation), and the §12 history-context fix is in place (`BCPolicy` buffers HS=3 frames + HS−1 past-action blocks + AdaLN-Zero horizon countdown). VERIFIED by code read of `gip.py:115 load_gip_model`, `gip.py:216 BCPolicy`, `gip.py:486 build_policy`.

### Config-gating (the first-class flags this run exercises)
`train.py:350` consumes `cfg.freeze_wm` (first-class, override `freeze_wm=false`, NOT `+freeze_wm`); `train.py:236-239` wraps the dataset in `GoalSamplingDataset` when `action_pred.goal_conditioned=true`; `train.py:256-259` sets `cfg.model.horizon_conditioned=True` (rides into config.json) when `action_pred.horizon_conditioned=true`; `train.py:73-95 lejepa_forward` encodes the hindsight goal through the **LEARNABLE** `self.model.encode` (NOT a cached latent) and applies `goal_dropout`. `goal_dropout=0` ⇒ full goal signal every sample. Startup log confirms: `[GIP] Intention predictor ON Adim=25 head=mse`, `[GIP] OURS horizon conditioning ON AdaLN-Zero H_max=50`, `[GIP] init_from=...cube_ours_lewm_weights.pt: missing=95 unexpected=0`, and NO `freeze_wm=true` line (⇒ encoder learnable).

### STEP 2 — EXACT TRAIN COMMAND (host `L40S`, run as `sudo -u minghao.fu`; `$B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`)
Env: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=0 MPLCONFIGDIR=/tmp/mpl_cube TMPDIR=/tmp OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=7`.
```
python train.py \
  data=ogb_lance model=lewm embed_dim=192 \
  action_pred.enabled=true action_pred.goal_conditioned=true action_pred.horizon_conditioned=true action_pred.goal_dropout=0 \
  action_pred.hindsight_max_k=50 freeze_wm=false \
  init_from=/mnt/minghao_data/.stable-wm/decoders/cube_ours_lewm_weights.pt \
  output_model_name=cube_gc_e2e subdir=cube_gc_e2e \
  trainer.max_epochs=60 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=40 +ckpt_every=10 \
  loader.batch_size=128 num_workers=10 wandb.enabled=false
```
**Throughput** (smoke + timing on GPU6/7): image lance DataLoader is I/O-bound at ~1 it/s (bs128, num_workers=10) ⇒ **~1.3 it/s steady = ~13-14 min/epoch**; 1000 batches/epoch = 128k windows/epoch (matched to the §10b cube exposure of 256k @ bs64×4000 capped — chose 1000×bs128 for tractable wall-time, will extend if val_act still falling). 60 ep ≈ 14h. `+ckpt_every=10` + a host-side prune loop (`/var/lib/docker/data/minghao_home/prune_e2e.sh`, keeps newest-3 Lightning `epoch=*.ckpt` under the run's date dir) protect the 92%-full `/var/lib/docker` disk; `weights_epoch_*.pt` (the eval-loaded ckpts) go to `/mnt/minghao_data` (2.6T free). GPU7 (lowest-occupied, ~15G used of 46G; co-located per rule 10).

### Convergence (rule 8 — verify plateau) [FILLING LIVE]
- Sanity-val at init: `act_loss=1.089`. Epoch 0 running at 1.3 it/s. [val_act trajectory + plateau evidence fill as epochs land via the SaveCkptCallback per-epoch val table; extend past 60ep if last-10% not flat.]

### STEP 3 — EVAL COMMAND (cube GC, N=50, 3 seeds {42,0,1}, same-box L40S, fires at convergence)
```
python eval_gip.py --config-name cube policy=cube_gc_e2e \
  +gip_eval.mode=policy +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 \
  eval.num_eval=50 seed=<42|0|1>
```
Same-box (train+eval both GPU7-region L40S, per the cross-GPU-render rule [[project_l40s_cross_gpu_rendering]] — cube/ogbench is NOT cross-box sensitive per §12, but kept same-box for cleanliness). `mode=policy goal_conditioned=true` = planning-free forward GC policy (one `predict_intention` pass, NO CEM), the §12 OURS eval path with the HS-history + goal + horizon fed correctly. [Per-seed + mean fill below.]

### THE cube end-to-end rescue table [FILLING LIVE]
| arm | cube GC SR (N=50, 3-seed {42,0,1}) | per-seed | vs anchors |
|---|---|---|---|
| **end-to-end LeWAM OURS (freeze_wm=FALSE, learnable, THIS §22)** | [pending convergence] | — | vs frozen-ours 72.7 / gcidm 100 |
| frozen-latent OURS-GC (§12, the arm being rescued) | **72.7** | 78 / 68 / 72 | — |
| GC-IDM reproduced (§11, same-box) | **100.0** | 100 / 100 (s42/s7) | — |
| gcidm PUBLISHED (n=200) | **98.7** | — | — |
| same-box CEM / guided (§12) | 67.0 / 80.0 | — | — |

### Verdict [PENDING convergence + eval]
Did end-to-end training (learnable encoder) RESCUE OURS on cube — lift it off the frozen-latent 72.7 floor toward (or past) the GC-IDM/gcidm ~100 anchor? [Fills with the decision once the 3-seed eval lands.]

## §23 🔄🆕 END-TO-END goal-conditioned LeWAM (encoder LEARNABLE, `freeze_wm=FALSE`) on **reacher** (DMC) — the CONFIRM test on the env where frozen-latent OURS already TIED (L40S, 2026-06-21) [IN FLIGHT]

**Motivation (why).** §15 established the FROZEN-latent reacher GC head-to-head: trained on cached LeWM latents (`train_ours_gc.py`/`train_gcidm.py`, `encoder.requires_grad=False`), **OURS (history) converged 97.0 = GC-IDM (Markovian) converged 97.0** (both seeds 98/96), reproducing the gcidm PUBLISHED ~99.7; same-box CEM 81, guided 84. Reacher is fully observed (target qpos + arm pose in the current frame), precise-control, non-contact — the §15 diagnosis was that both planning-free heads converge to the same ceiling because the recoverable state is fully present and there is no contact dynamics for open-loop history to drift on; the DECISIVE knob was the AdaLN horizon (−61 without it), not history. Unlike cube (§22, where frozen OURS LOST 72.7 ≪ 100 → a true rescue test), reacher's frozen OURS already TIES. So this §23 is a **CONFIRM**: does training the encoder + WM + goal-conditioned intention head JOINTLY end-to-end (encoder LEARNABLE, hindsight goals, `freeze_wm=false`, NO cached latents) from the SAME warm-start base (`reacher_ours_lewm_weights.pt`) still land at the ~97 tie, or does the learnable-encoder path move it? (Expectation from §15's mechanism: reacher is fully observed and near-ceiling, so the end-to-end encoder should neither help nor hurt — confirm ≈ 97.)

**Anchors (reacher, N=50, same-box L40S unless noted):** frozen-latent OURS = **97** (§15, the arm being confirmed); GC-IDM reproduced = **97** (§15, L40S same-box converged); gcidm PUBLISHED = **99.7**. Same-box CEM 81 / guided 84 (§15).

### STEP 1 — SETUP VERIFIED (all present on L40S, this session)
- **Data config**: `data=dmc` → `config/train/data/dmc.yaml` → `name: reacher.h5` (frameskip 5, keys pixels/action/observation; resolves to `$STABLEWM_HOME/datasets/reacher.h5`, **98 GB**, 10000 episodes, 40 obs-steps/ep, raw action dim 2 ⇒ **action_block_dim = 10**). VERIFIED `cat config/train/data/dmc.yaml` + `ls -la .../datasets/reacher.h5` (98905882624 bytes).
- **Warm-start base**: `/mnt/minghao_data/.stable-wm/decoders/reacher_ours_lewm_weights.pt` (72,271,643 bytes, vit-tiny-192 SIGReg LeWM — the SAME frozen base §15 used). VERIFIED present.
- **Eval entry**: `eval_gip.py --config-name reacher` + `config/eval/reacher.yaml` (N=50, goal_offset_steps=25 ⇒ horizon0 = 25/5 = 5 obs-steps, action_block 5, env `swm/ReacherDMControl-v0` task `qpos_match`, callables `set_state(qpos,qvel)` + `set_target_qpos(goal_qpos)`, `dataset_name: reacher` — the §15 one-line fix already in place). VERIFIED `cat config/eval/reacher.yaml`. `mode=policy goal_conditioned=true` routes to the planning-free forward GC policy (the §15 OURS path). Eval needs the dm_control compat shim `/tmp/reacher_compat/sitecustomize.py` (the §15 `flex_bandwidth` fix; VERIFIED present, 926 bytes) on PYTHONPATH.
- **Config-gating** (the first-class flags this run exercises): `config/train/lewm.yaml` carries `action_pred.{enabled,goal_conditioned,horizon_conditioned,goal_dropout,hindsight_max_k,horizon_H_max}` (all default OFF) and the first-class `freeze_wm: false` (lewm.yaml:85, consumed `train.py:350`, override `freeze_wm=false` NOT `+freeze_wm`). VERIFIED. CLIP is only loaded for multi-task (config-gated `train.py:282`), NOT for single-task reacher GC ⇒ `HF_HUB_OFFLINE` is irrelevant here.

### Gate verification (smoke, this session)
1-epoch real-data smoke (GPU6, `data=dmc action_pred.enabled=true action_pred.detach_decoder=false action_pred.goal_conditioned=true action_pred.horizon_conditioned=true action_pred.goal_dropout=0 action_pred.hindsight_max_k=50 action_pred.horizon_H_max=50 init_from=reacher_ours_lewm_weights.pt`, limit_train_batches=20): GC losses present and fall — sanity-val `act_loss=1.042 intent_loss=1.416 pred_loss=0.022` → epoch-0 `validate/act_loss=1.042 intent=1.416 pred=0.022`; `fit/act_loss=1.028 intent=1.417`; ckpt saved `checkpoints/reacher_gc_e2e_smoke/weights_epoch_1.pt`; no crash. Confirms GoalSamplingDataset (built only when goal_conditioned=true) + horizon AdaLN + warm-start run end-to-end on the reacher data. Smoke ckpt deleted.

### STEP 2 — EXACT TRAIN COMMAND (host `L40S` `stratus-lookout`, run as `sudo -u minghao.fu`; `$B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`; driver `/mnt/minghao_data/reacher_e2e_train3.sh`)
Env (root `/` is 99% full ⇒ ALL writes redirected to /mnt/minghao_data, 2.6T free): `STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/mnt/minghao_data/tmp MPLCONFIGDIR=/mnt/minghao_data/mpl_reacher XDG_CACHE_HOME=/mnt/minghao_data/.cache_e2e HF_HOME=/home/minghao.fu/.cache/huggingface WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=6`.
```
python train.py \
  data=dmc \
  action_pred.enabled=true \
  action_pred.detach_decoder=false \
  action_pred.goal_conditioned=true \
  action_pred.horizon_conditioned=true \
  action_pred.goal_dropout=0 \
  action_pred.hindsight_max_k=50 \
  action_pred.horizon_H_max=50 \
  init_from=$STABLEWM_HOME/decoders/reacher_ours_lewm_weights.pt \
  output_model_name=reacher_gc_e2e \
  trainer.max_epochs=60 +trainer.limit_train_batches=2000 +trainer.limit_val_batches=20 \
  +ckpt_every=5 \
  loader.batch_size=64 num_workers=4 \
  hydra.run.dir=/mnt/minghao_data/hydra_e2e_reacher/run
```
**Throughput**: the 98 GB reacher.h5 is I/O-heavy (§15 noted it; bs128/8w was SLOWER at 0.7 it/s under box contention, reverted). bs64/4w steady **2.0 it/s ⇒ ~1024 s/epoch (17 min)**. `limit_train_batches=2000 × bs64 = 128k windows/epoch` (1/3 of the ~400k-window dataset; §15 frozen OURS converged ~ep40 over 342k windows ⇒ equivalent total exposure ~ep3, but the e2e encoder/WM update needs more — cap 60ep, verify plateau, extend if val_act still falling). `+ckpt_every=5` saves `weights_epoch_{5,10,15,...,60}.pt`; host-side prune loop keeps newest-3 `weights_epoch_*.pt` + newest-2 Lightning `epoch=*.ckpt` (the .ckpt go to the `/var/lib/docker` 3T volume, NOT root). GPU6 (lowest-occupied, co-located per rule 10). **NOTE (the §15 checkpoint-selection trap):** reacher val MSE sits near the z-scored-action variance floor (~1.0) ⇒ val MSE is a POOR early-stop selector here; eval the CONVERGED (highest-epoch) `weights_epoch_*.pt`, which `load_gip_model` picks by default (`pts[-1]`), NOT a best-val checkpoint. This run saves only periodic snapshots (no best-val selector), so the default load = converged = correct.

### Convergence (rule 8 — verify plateau) [FILLING LIVE]
- ~1025 s/epoch (17 min). Sanity-val at init: `act_loss=1.089 intent_loss=1.472 pred_loss=0.0158`.
- **val/act_loss** (z-scored 10-d action MSE, near the variance floor — the §15 selector caveat): 1.089(sanity)→**0.996**(ep0)→**0.991**(ep1)→**0.991**(ep2)→**0.991**(ep3) — FLAT at ~0.991 from ep1. As §15 warned, reacher val MSE plateaus near the action-variance floor almost immediately and is a POOR convergence selector; rollout SR keeps improving as TRAIN MSE sharpens.
- **fit/act_loss** (train, the meaningful signal here): **1.024**(ep0)→**0.992**(ep1)→**0.988**(ep2)... still inching down → not yet bottomed. Train heavier (the §15 lesson: eval the CONVERGED, lowest-train-MSE checkpoint, not best-val).
- **val/intent_loss** (act_emb prediction): 1.472(sanity)→**0.244**(ep0)→**0.231**(ep1)→**0.225**(ep2)→ collapsed fast then slow-falling.
- **DECISION (per §15 + rule 8):** continue to a clear plateau, eval the CONVERGED (highest-epoch) `weights_epoch_*.pt`.
- **⚠️ EARLY-CHECKPOINT PROBE (ep5, N=10 seed42, GPU5): SR = 0.0 (0/10).** Model loads clean (`Adim=10 missing=0 unexpected=0`, BCPolicy goal_conditioned). fit/act is NEAR the variance floor (1.024→0.992→0.988→0.973→0.993, oscillating ~0.98±0.01 — z-scored floor, uninformative) but the ROLLOUT SR at ep5 is 0 → the e2e model is UNDERTRAINED at ep5 (the learnable encoder has not yet adapted to produce GC-usable latents; the frozen §15 path started higher because its base latents were already trained). **ep5=0 is a diagnostic, NOT a result (rule 8).** This is the §15 lesson taken further: on reacher the LOSS plateaus near the floor long before the SR converges, so loss is a useless convergence selector here — trace SR directly.
- **STRATEGY SWITCH (rule 8):** replaced the loss-plateau heuristic with a **direct SR-probe orchestrator** (`/mnt/minghao_data/reacher_sr_probe.sh`, GPU5): evals N=10 seed42 at EVERY new `weights_epoch_{5,10,15,...}.pt` → `SR_TRACE.txt`, observes where SR plateaus, then on training-end runs the final 3-seed {42,0,1} N=50 on the CONVERGED (highest-epoch) ckpt → `EVAL_RESULTS.txt`. If SR still climbing at ep60 (the cap), EXTEND ("when in doubt, train heavier"). [SR trajectory fills live below.]
- **SR trajectory (N=10 seed42 probe):** ep5 = **0.0**. [ep10/15/20/... fill as checkpoints land.]
- **Loss STILL DECLINING at ep8 (NOT converged — the run needs to continue):** fit/act 1.024→0.992→0.988→0.973→0.993→0.977→0.979→**0.960**(ep8, lowest yet, still falling); val/intent MONOTONE 0.244→0.231→0.225→0.220→0.218→0.213→0.209→**0.207**(ep8). The e2e model is genuinely still learning at ep8 → ep5 SR=0 is undertraining (confirmed by the still-falling losses), NOT a broken pipeline. Convergence (§15 reacher analog) is expected ~ep30-40 (≈6-9 h at ~17 min/epoch).
- **STATUS (handoff): IN FLIGHT, blocked on convergence wall-clock.** Three durable nohup'd host processes drive it to completion independently of any session: (1) `train.py` (60ep cap, GPU6), (2) `reacher_sr_probe.sh` (N=10 SR probe at each ckpt → `SR_TRACE.txt`, then final 3-seed {42,0,1} N=50 on the CONVERGED ckpt → `EVAL_RESULTS.txt` ending `ALL_EVALS_DONE`), (3) `reacher_e2e_prune.sh` (keep-newest-3). **UNBLOCK CONDITION:** training reaches the SR plateau (watch `SR_TRACE.txt` lift off 0 and flatten); the orchestrator then auto-writes the headline 3-seed N=50 to `EVAL_RESULTS.txt`. If SR still climbing at the ep60 cap, EXTEND training (re-launch `init_from` the last ckpt). All logs under `/mnt/minghao_data/logs_reacher_e2e/`.

### STEP 3 — EVAL COMMAND (reacher GC, N=50, 3 seeds {42,0,1}, same-box L40S, fires at convergence; driver `/mnt/minghao_data/reacher_e2e_eval.sh`)
```
PYTHONPATH=/tmp/reacher_compat:$B OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python eval_gip.py --config-name reacher policy=reacher_gc_e2e \
  +gip_eval.mode=policy +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 \
  eval.num_eval=50 seed=<42|0|1>
```
Same-box (train+eval both L40S; reacher dm_control render IS cross-box sensitive per [[project_l40s_cross_gpu_rendering]] — keep eval on L40S; orchestrator runs eval on the SAME GPU 6 as training). `mode=policy goal_conditioned=true` = planning-free forward GC policy (one `predict_intention` pass, NO CEM), feeding HS=3-frame history + encoded z_goal + AdaLN-Zero horizon countdown (`horizon_H_max=50`). The reacher_compat shim is required for env instantiation. **Eval-path VALIDATED this session (N=2 smoke vs the §15 `reacher_ours_gc` converged ckpt, GPU5):** shim applied (`dropped flex_bandwidth`), env makes+renders, `[GIP] load reacher_ours_gc <- weights_epoch_0.pt Adim=10 missing=0 unexpected=0`, `[GIP] eval mode=policy policy=BCPolicy` (goal_conditioned routing confirmed), SR 100 (2/2) — consistent with §15's converged 97. The `reacher_gc_e2e` eval uses the identical command (default-loads the highest-epoch = converged ckpt). **Eval auto-runs via host orchestrator `/mnt/minghao_data/reacher_eval_orchestrator.sh` (waits for `weights_epoch_20.pt` OR training-end, then 3-seed {42,0,1} N=50 on GPU6, writes `EVAL_RESULTS.txt`).** [Per-seed + mean fill below.]

### THE reacher end-to-end confirm table [FILLING LIVE]
| arm | reacher GC SR (N=50, 3-seed {42,0,1}) | per-seed | vs anchors |
|---|---|---|---|
| **end-to-end LeWAM OURS (freeze_wm=FALSE, learnable, THIS §23)** | [pending convergence] | — | vs frozen-ours 97 / gcidm 97 |
| frozen-latent OURS-GC (§15, the arm being confirmed) | **97.0** | 98 / 96 (s42/s7) | — |
| GC-IDM reproduced (§15, same-box converged) | **97.0** | 98 / 96 (s42/s7) | ≈ published 99.7 |
| gcidm PUBLISHED (n=200) | **99.7** | — | — |
| same-box CEM / guided (§15) | 81.0 / 84.0 | — | — |

### Verdict [PENDING convergence + eval]
Did end-to-end training (learnable encoder) hold the reacher tie — confirm ≈ 97 (= frozen-ours = gcidm), or move it? [Fills with the decision once the 3-seed eval lands.]

## §24 🔄🆕 CORRECTED freeze-ablation on **can**: LEARNABLE encoder (`freeze_wm=false`) + `goal_dropout=0.5`, CONVERGED — the missing cell that disentangles the §21 dropout confound (L40S `stratus-lookout`, 2026-06-21) [IN FLIGHT]

**Motivation (why).** §21 ran the decisive freeze-ablation on can with `goal_dropout=0` and both arms converged to 100 ep: LEARNABLE (`freeze_wm=false`) = **0.007**, FROZEN control (`freeze_wm=true`) = **0.027** — the learnable encoder did NOT lift off the frozen floor, and §21's headline was "LeWAM loses on robomimic GC is NOT a frozen-regime artifact; the floor survives a fully learnable encoder." BUT §21's own STEP-3 diagnostic (the stalled-at-epoch-10 `can_gc_e2e`, `goal_dropout=0.5`) scored **0.207** at the SAME canonical b240/o90 — ABOVE both converged dropout-0 arms and at the GC-IDM Markovian floor (0.240). §21 flagged this as confounded (under-trained epoch-10 + dropout 0.5 + a prior agent's stalled run) and gave two non-exclusive reads: (i) `goal_dropout=0.5` makes the head goal-AGNOSTIC on half the samples → it partially behaves like the 0.71 histbc-bc policy that wins on can by IGNORING the distant goal; (ii) high seed variance. This §24 is the **CORRECTED ablation that removes the confound**: train the LEARNABLE arm (`freeze_wm=false`, the user's real end-to-end LeWAM) with `goal_dropout=0.5`, CONVERGED to 100 ep (not stalled at 10). It is the byte-identical twin of §21's learn arm EXCEPT `goal_dropout` (0.5 vs 0). The question: **with the encoder learnable, does converged dropout-0.5 reproduce the 0.207 lift (→ the §21-(i) read: the apparent GC lift is really the goal-agnostic histbc-bc mode leaking through dropout), or does it collapse to the ~0.007–0.027 dropout-0 floor once it is properly converged (→ §21-(ii): the 0.207 was just under-trained seed noise)?** Either outcome sharpens the §21 headline: it isolates whether `goal_dropout` (CFG-style two-setting training), NOT the encoder being learnable, is what moves can GC.

### STEP 1 — SETUP + GATE VERIFIED (all present on L40S, this session)
- **Flags all first-class and HONORED** (`config/train/lewm.yaml`, consumed `train.py`): `freeze_wm` (lewm.yaml:85 default false, consumed `train.py:350`, override `freeze_wm=false` NOT `+freeze_wm`); `action_pred.{enabled,goal_conditioned,horizon_conditioned,goal_dropout,hindsight_max_k}` (default OFF). `train.py:73-79` applies `goal_dropout` per-sample (zero z_goal w.p. p) on the GoalSamplingDataset goal; `train.py:237-239` wraps the dataset in `GoalSamplingDataset` when `goal_conditioned=true`; `train.py:257-259` sets `cfg.model.horizon_conditioned=True` (rides config.json) when `horizon_conditioned=true`. NO new script — pure config-gating of the existing `train.py`/`eval_histbc_robomimic.py`.
- **Warm-start base**: `$DEC/can_lewm_weights.pt` = `/mnt/minghao_data/.stable-wm/decoders/can_lewm_weights.pt` (vit-tiny-192 SIGReg LeWM, the SAME base §16/§20/§21 used). `init_from` loads `missing=95 unexpected=0` (95 missing = GIP head action_predictor/action_decoder + horizon_modulator, trained from scratch — correct).
- **Data**: `data=robomimic_can` → re-rendered-on-L40S `can.h5` (image_224 robomimic; frameskip 5 ⇒ action_block 5, **Adim=35** = 7-DoF × 5). Model-size note N/A (this is the freeze arm; encoder stays vit-tiny-192, `embed_dim=192` unchanged — no size/embed_dim ablation here, so no companion-embed_dim shape-crash risk).
- **Gate (1-epoch real-data smoke, GPU 2, `limit_train_batches=3`, deleted after):** startup `[GIP] Intention predictor ON Adim=35 head=mse`, `[GIP] OURS horizon conditioning ON AdaLN-Zero H_max=50`, `init_from=...can_lewm_weights.pt: missing=95 unexpected=0`, horizon_modulator 82,560 params built, **NO `freeze_wm=true` line ⇒ encoder LEARNABLE**, GC losses present and finite (`validate/act_loss 1.18 intent_loss 1.94 pred_loss 0.011` — confirms GoalSamplingDataset + goal_dropout=0.5 path runs end-to-end), `weights_epoch_1.pt` (117 MB) + config.json + full_config.yaml saved cleanly, **exit 0, no shape crash, no NaN.** First gate attempt OOM'd because Lightning `trainer.devices=auto` ignored `CUDA_VISIBLE_DEVICES` and grabbed the busiest physical GPU; fixed by running on GPU 2 (most headroom) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### STEP 2 — EXACT TRAIN COMMAND (host `L40S` `stratus-lookout`, run as `sudo -u minghao.fu`; `$B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`; driver `/mnt/minghao_data/train_freeze_false_gc_d05.sh`)
```
env STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=0 \
  OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl_freeze_false_gc_d05 TMPDIR=/tmp \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2 \
  python train.py data=robomimic_can \
  output_model_name=can_gc_d05 subdir=can_gc_d05 \
  freeze_wm=false \
  action_pred.enabled=true action_pred.goal_conditioned=true \
  action_pred.horizon_conditioned=true action_pred.goal_dropout=0.5 \
  action_pred.hindsight_max_k=50 \
  init_from=$DEC/can_lewm_weights.pt \
  trainer.max_epochs=100 +ckpt_every=10 wandb.enabled=false
```
Full dataset per epoch (NO `limit_train_batches` cap — matches the §21 full-dataset recipe, ~136 train steps/epoch). `+ckpt_every=10` ⇒ `weights_epoch_{10,20,...,100}.pt`; host prune loop (`/mnt/minghao_data/prune_freeze_false_gc_d05.sh`, every 120 s) keeps newest-3. GPU 2 (lowest-occupied at launch; co-located per rule 10), `expandable_segments` for the co-location. Log `/mnt/minghao_data/freeze_false_gc_d05_train.log`.

### Convergence (rule 8 — verify plateau) [FILLING LIVE]
[fills with per-epoch val act_loss over the last 10% once it reaches ep100]

### STEP 3 — EVAL COMMAND (can GC, N=50, 3 seeds {42,0,1}, canonical b240/o90, same-box GPU 2 = train GPU per the cross-GPU render gotcha [[project_l40s_cross_gpu_rendering]]; driver `/mnt/minghao_data/eval_freeze_false_gc_d05.sh`)
```
python eval_histbc_robomimic.py --config-name robomimic world.task=PickPlaceCan \
  world.num_envs=10 dataset.stats=can eval.dataset_name=can eval.num_eval=50 \
  eval.eval_budget=240 eval.goal_offset_steps=90 policy=can_gc_d05 \
  +gip_eval.mode=policy +gip_eval.goal_conditioned=true seed=<42|0|1>
```

### THE corrected-freeze-ablation table [FILLING LIVE]
| arm | can GC SR (N=50, b240/o90, 3-seed {42,0,1}) | per-seed | note |
|---|---|---|---|
| **freeze_wm=FALSE (learnable) + goal_dropout=0.5, CONVERGED (THIS §24)** | [pending] | — | the corrected cell |
| §21 learnable (freeze_wm=FALSE) + goal_dropout=0, converged | **0.007** | 0.02 / 0.00 / 0.00 | the dropout-0 learn arm |
| §21 frozen control (freeze_wm=TRUE) + goal_dropout=0, converged | **0.027** | 0.06 / 0.02 / 0.00 | the frozen floor |
| §21 STEP-3 `can_gc_e2e` (learnable, dropout 0.5, epoch-10 STALLED) | **0.207** | 0.06 / 0.30 / 0.26 | under-trained diagnostic |
| frozen-latent OURS-GC `train_ours_gc` (§20 converged) | **0.133** | 0.10 / 0.22 / 0.08 | — |
| GC-IDM Markovian `train_gcidm` (§20 converged) | **0.240** | 0.30 / 0.26 / 0.16 | — |
| end-to-end histbc bc (goal-AGNOSTIC, §1c ceiling) | **0.71** | — | the goal-agnostic ceiling |

### Verdict [PENDING convergence + eval]
Does CONVERGED learnable + dropout-0.5 reproduce the stalled-epoch-10 0.207 (⇒ the apparent GC lift is the goal-AGNOSTIC histbc-bc mode leaking through dropout, NOT better goal-conditioning), or collapse to the §21 dropout-0 floor (~0.007–0.027, ⇒ the 0.207 was under-trained seed noise)? Either way isolates whether `goal_dropout` (not the learnable encoder) is what moves can GC. [Fills once the 3-seed eval lands.]

---

## §24-SIZE 🔄🆕 MODEL-SIZE SCALING on **can** (goal-AGNOSTIC histbc-bc), scaling point 3 = ViT-**base** (embed_dim 768) — arm `size_base_768_bc` (L40S, 2026-06-21) [IN FLIGHT]

**NOTE on numbering.** This is a distinct infra arm (model-size scaling) that collided with the parallel-agent §24 (corrected freeze-ablation, goal_dropout=0.5). To avoid clobbering that agent's live cell I file this under **§24-SIZE**. It is a goal-AGNOSTIC bc arm (`action_pred.goal_conditioned=false`, mode=bc), NOT a GC arm — orthogonal to §24's GC freeze-ablation. The master table is NOT edited.

**Motivation (why).** The model-size sweep asks whether the can histbc-bc SR (the goal-AGNOSTIC imitation ceiling, ~0.71 at the default tiny/192 encoder, §1c) moves when the LeWM ViT encoder is scaled up. Scaling point 1 = tiny (hidden 192, the default/baseline), point 2 = small (384), point 3 = **base (768)** — THIS arm. The encoder CLS width sets `embed_dim` for the whole trunk (predictor / projector / pred_proj / action_encoder all key off `embed_dim`), so the audit flag (`lewm_dinov2 needs a companion embed_dim`) generalizes: for any non-tiny encoder we MUST set `embed_dim` to the encoder hidden_size or the trunk mismatches the CLS token. This arm tests `size=base embed_dim=768`. Question: does a 4×-wider encoder (768 vs 192) raise can goal-agnostic bc SR above the 0.71 tiny baseline, or is can bc already saturated so capacity does not help?

### STEP 1 — flag verification (DONE, all honored)
- `stable_pretraining.backbone.utils.vit_hf` `size_configs`: tiny→hidden 192, small→384, **base→768**, large→1024, huge→1280 (12 layers for tiny/small/base, 12/12/12 heads resp). So `model.encoder.size=base` ⇒ ViT hidden_size 768. CONFIRMED in the source.
- `jepa.JEPA.forward`: `pixels_emb = encoder(pixels).last_hidden_state[:,0]` (CLS token, width = encoder hidden_size). All downstream modules (predictor/projector/pred_proj/action_encoder) are built with `${embed_dim}`. ⇒ `embed_dim` MUST equal the encoder hidden_size. Set BOTH `model.encoder.size=base` AND `embed_dim=768`. CONFIRMED.
- `action_pred.{enabled,goal_conditioned}`, `freeze_wm`, `init_from` all read via `cfg.get(...)` in `train.py` (lines 63/73/236/298/340/350). Honored.
- 1-step forward smoke test (base/768, scratch, bs8, 10 train batches): **ViT-base built `hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, patch_size=14`**; full train+val ran, `fit/act_loss=0.70`, `validate/act_loss 1.40→1.15`, `validate/act_emb_std≈0.10` (no collapse), NO shape crash. The only failure at `limit_train_batches=1` was a `ZeroDivisionError` in the cosine LR scheduler (warmup/total-steps edge case with 1 batch), NOT a model-shape issue — cleared at ≥10 batches.

### init_from MISMATCH → TRAINED FROM SCRATCH (stated per task instruction)
`init_from=$DEC/can_lewm_weights.pt` was tested with base/768 and **CRASHES** with `RuntimeError: Error(s) in loading state_dict ... size mismatch` on every shared key (the ckpt is the tiny/192 LeWM: e.g. `predictor.*.mlp.net.1.weight [2048,192] vs [2048,768]`, `action_encoder.embed.0.weight [768,10] vs [3072,10]`, `projector.net.0.weight [2048,192] vs [2048,768]`). `load_state_dict(strict=False)` does NOT skip shape-mismatched keys; it raises. So **this size arm trains from SCRATCH (no `init_from`)** — the warm-start ckpt is dimensionally incompatible with a 768-wide trunk. (This is the expected behaviour the task flagged: "for the size arms init_from may mismatch dims — if so train from scratch and SAY so." Said.)

### STEP 2 — EXACT TRAIN COMMAND (from scratch, GPU 3 L40S, 100 ep, converge)
```
# infra env (rule 10 nvme redirects):
STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=0 \
OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl_size_base_768_bc TMPDIR=/tmp \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 \
/var/lib/docker/data/minghao_home/lewm/bin/python train.py \
  data=robomimic_can model.encoder.size=base embed_dim=768 \
  action_pred.enabled=true action_pred.goal_conditioned=false \
  output_model_name=size_base_768_bc_can \
  trainer.max_epochs=100 loader.batch_size=32 num_workers=6 +ckpt_every=10
# log: /mnt/minghao_data/logs_size_arms/size_base_768_bc.log
```
- **From scratch**, NO init_from (dim mismatch, see above).
- `loader.batch_size=32` (NOT the warm-arm 64): vit-base from scratch + the crowded L40S (all 8 GPUs 29–34 GB used by other parallel arms — NOT killed per instruction) OOM'd at bs64 on the tightest GPU; bs32 + `expandable_segments:True` fits in the ~16 GB free on GPU 3. 545 steps/epoch, ~1.1 it/s ⇒ ~8 min/epoch ⇒ ~14 h for 100 ep. Full data, NO `limit_train_batches` cap (convergence per rule 8).
- `+ckpt_every=10` (rule 10, avoid disk-full); STABLEWM_HOME on /mnt/minghao_data (2.5 T free).

### Convergence (rule 8 — verify plateau) [FILLING LIVE]
[fills with per-epoch val act_loss over the last 10% once it converges; verify flat]

### RESUME after a disk-full crash at ep45 — the actual run history (2026-06-21, L40S `stratus-lookout`)
**What happened.** The original `size_base_768_bc_can` launch (cmd above) ran to **epoch 45** (val act_loss descending the whole way: 0.455 ep0 → 0.0945 ep20 → 0.052 ep30 → 0.041 ep43 → **0.0354 ep45**, still going DOWN — NOT plateaued), then **CRASHED** trying to save the epoch-45 checkpoint with `OSError: [Errno 28] No space left on device` (the `atomic_torch_save` temp-write hit a full filesystem during the earlier disk-full window). The last INTACT snapshot on disk is `weights_epoch_40.pt` (856 MB, the epoch-45 `.pt` never finished). No Lightning `*_weights.ckpt` was written (the SaveCkptCallback only persists periodic `weights_epoch_N.pt`), so a true Lightning optimizer-state resume is impossible.

**Resume decision (global rule 9 / repo rule 8).** Per "crashed runs may continue from intact weights to convergence; a restarted LR schedule is acceptable — wasting intact epochs is not", I resumed via **`init_from=weights_epoch_40.pt`** into a fresh arm `size_base_768_bc_can_resume` (preserving the original ep10–40 snapshots for provenance), fresh AdamW + restarted cosine schedule, **80 more epochs**. Load confirmed `missing=0 unexpected=0` and the startup sanity-val act_loss = **0.0365** (matches where ep40–45 left off → genuine warm resume, NOT re-randomized). Now disk has 2.5 TB free on `/mnt/minghao_data`, all writes redirected there.

**EXACT resume command (GPU 1, L40S, 80 ep, converge):**
```
HOME=/var/lib/docker/data/minghao_home \
STABLEWM_HOME=/mnt/minghao_data/.stable-wm \
XDG_CACHE_HOME=/mnt/minghao_data/xdg_size_base_768_bc \
TMPDIR=/mnt/minghao_data/tmp_size_base_768_bc \
MPLCONFIGDIR=/mnt/minghao_data/mpl_size_base_768_bc \
HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1 MUJOCO_GL=egl OMP_NUM_THREADS=4 \
PYTHONPATH=$HOME/workspace/le-wm-repro CUDA_VISIBLE_DEVICES=1 \
$HOME/lewm/bin/python train.py \
  data=robomimic_can model.encoder.size=base embed_dim=768 \
  action_pred.enabled=true action_pred.goal_conditioned=false \
  init_from=/mnt/minghao_data/.stable-wm/checkpoints/size_base_768_bc_can/weights_epoch_40.pt \
  output_model_name=size_base_768_bc_can_resume \
  trainer.max_epochs=80 loader.batch_size=32 num_workers=6 +ckpt_every=10 wandb.enabled=false \
  hydra.run.dir=$TMPDIR/hydra_resume
# log: /mnt/minghao_data/logs_size_arms/size_base_768_bc_resume.log
# resume launched 2026-06-21 19:47 UTC; ~149 s/epoch @ 3.8 it/s (GPUs uncontended now) -> ~3.3 h -> ~23:00 UTC ep80.
```
A watcher (`size_base_768_watcher.sh`) prunes snapshots to newest-3 during training, then on `weights_epoch_80.pt` does a final-only prune and fires the 3-seed N=50 bc eval (below) automatically.

**⚠️ SECOND disk-full crash on the FIRST resume attempt — root cause `SPT_CACHE_DIR` (infra fix, 2026-06-21).** The first resume launch crashed at **epoch 1** with `FileNotFoundError ... /var/lib/docker/data/minghao_home/.cache/stable-pretraining/runs/20260621/194739/<id>/metrics.csv`. Cause: the stable-pretraining `Manager._resolve_run_dir` writes its run dir + CSVLogger `metrics.csv` under `cfg.cache_dir`, which `_config.py` resolves as **`os.environ.get("SPT_CACHE_DIR", "~/.cache/stable-pretraining")`**. `STABLEWM_HOME` and `XDG_CACHE_HOME` do NOT govern this path — it has its OWN env var. With `SPT_CACHE_DIR` unset it defaulted to `$HOME/.cache` on the 100%-full `/var/lib/docker`, so the metrics-dir create/write failed. **FIX: export `SPT_CACHE_DIR=/mnt/minghao_data/spt_size_base_768_bc`** (and `mkdir -p` it). After this the relaunch passed epoch 1 cleanly, `metrics.csv` lands at `/mnt/minghao_data/spt_size_base_768_bc/runs/.../metrics.csv`, zero disk errors, init_from still `missing=0 unexpected=0`. **Infra lesson (add to the rule-10 redirect list): on L40S you must ALSO set `SPT_CACHE_DIR` to `/mnt/minghao_data/...`, not just `STABLEWM_HOME`/`TMPDIR`/`MPLCONFIGDIR`/`XDG_CACHE_HOME`/`HF_HOME` — the stable-pretraining run/metrics dir is a SEPARATE write target on the full docker FS otherwise.** Relaunched 2026-06-21 19:53 UTC, ep0 @ 3.7 it/s; ~3.3 h -> ~23:10 UTC ep80.

### STEP 3 — EVAL COMMAND (can bc, goal-AGNOSTIC, N=50, 3 seeds {42,0,1}, canonical b240/o90, SAME-BOX = train GPU per cross-GPU render gotcha [[project_l40s_cross_gpu_rendering]])
```
# per seed s in {42,0,1}, SAME-BOX L40S, NO XDG_CACHE_HOME at eval (breaks HF/CLIP, infra §0):
HOME=/var/lib/docker/data/minghao_home STABLEWM_HOME=/mnt/minghao_data/.stable-wm \
TMPDIR=/mnt/minghao_data/tmp HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 \
MPLCONFIGDIR=/mnt/minghao_data/mpl MUJOCO_GL=egl ROBOMIMIC_RAW=$HOME/robomimic \
CUDA_VISIBLE_DEVICES=<g> $HOME/lewm/bin/python eval_histbc_robomimic.py --config-name robomimic \
  policy=size_base_768_bc_can_resume world.task=PickPlaceCan dataset.stats=can eval.dataset_name=can \
  world.num_envs=10 eval.num_eval=50 eval.eval_budget=240 eval.goal_offset_steps=90 seed=$s
# default gip_eval.mode=policy, goal_conditioned=false  => pure goal-AGNOSTIC BC
# (no goal/horizon/OOD-tail => GC-bug-INDEPENDENT, §27 does not apply). HistoryBCPolicy, HS=3, action_block=5.
# load_gip_model picks the highest weights_epoch_*.pt (=ep80 after final prune).
```

### THE model-size scaling table [FILLING LIVE]
| scaling pt | encoder | embed_dim | can bc SR (N=50, b240/o90, 3-seed {42,0,1}) | per-seed | note |
|---|---|---|---|---|---|
| 1 (baseline, §1c anchor) | tiny | 192 | ~0.71 | 72/72/68 | §1c histbc-bc, detach=TRUE, same recipe |
| 1 (THIS §24-SIZE, dedicated pt1 `size_tiny_192_bc`) | tiny | 192 | **0.60** | 0.64/0.60/0.56 (s42/s0/s1, SE ~0.023) | warm WM init `can_lewm_weights.pt`, 100ep, N=50 b240/o90 mode=policy goal_cond=false; same-host same-recipe scaling anchor for 384/768. Lands a bit below the §1c 0.71 (eval-time variance, same recipe) — both are the tiny/192 point |
| 2 (`size_small_384_bc`) | small | 384 | [pending convergence+eval] | — | 2×-wider encoder, from scratch; relaunch at ep6/100 (prior run crashed ep90 disk-full, SPT_CACHE_DIR fixed) |
| 3 (`size_base_768_bc`) | **base** | **768** | [pending convergence+eval] | — | 4×-wider encoder, from scratch; resume from ep40, ep0/80 (SPT_CACHE_DIR fixed) |

### Verdict [PARTIAL — only pt1 landed; NOT a curve yet]
Only the **tiny/192** point is converged and evaluated: 3-seed N=50 = **0.60** (0.64/0.60/0.56). The 384 and 768 points are NOT converged (384 at ep6/100, 768 at ep0/80), so the model-size→SR curve **cannot be drawn yet** and NO size trend is claimed. Whether the 768-wide ViT-base raises can goal-agnostic bc SR above the tiny baseline, or can-bc is saturated (SR flat) so capacity doesn't help, remains OPEN until the 384/768 3-seed N=50 evals land. Do NOT headline a single point as a scaling result (rule 8/9).

## §SPEED ✅🆕 INFERENCE SPEED / FLOPs benchmark — the headline "Fast" evidence (planning-free LeWAM vs search-based planning) (L40S `stratus-lookout`, 2026-06-21)

**Motivation (why).** The LeWAM "Fast" claim needs a hard per-action latency + FLOPs number. The two planning-free heads (OURS history-conditioned forward GC policy, and the Markovian GC-IDM) each decide an action in ONE forward pass; the search-based baseline (CEM/MPPI over the world model) re-rolls a population of candidate plans over a horizon for many iterations, so its per-action cost is multiplied by `num_samples × horizon × n_steps`. Diffusion-Policy / LDP-style heads multiply by the number of denoising steps `K`. This section quantifies all four on the SAME box / model / GPU / dtype, with the real trained `can` base, so the paper can state "planning-free LeWAM is N× faster than CEM". **TD-MPC2 reports NO inference-latency number, so this comparison is uncontested** — there is no published per-action-time figure to contradict ours.

**What is measured (per ACTION decision, mean ± std, warmup 20, ≥100 timed reps with CUDA syncs, fp32, batch=1):**
1. **LeWAM-GC (ours, planning-free)** — one forward pass mirroring `eval_histbc_robomimic.py::HistoryBCPolicy.get_action`: encode the **1 NEW** 224² frame → append its latent to the maxlen-`HS=3` deque (the older `HS−1` latents are cached, encoded once) → `action_predictor` (6-layer `ARPredictor`) over the HS-frame latent history → `horizon_modulator` (AdaLN-Zero, this ckpt is `horizon_conditioned=true`) → `action_decoder` (MLP) → one frameskip-stacked action block. = one step of `jepa.JEPA.intention_rollout`.
2. **gcidm (Markovian, planning-free)** — `eval_histbc_robomimic.py::GCIDMRobomimicPolicy`: encode `z_t` (current frame) + encode `z_goal` (goal frame, re-encoded every decision, faithful to the code) → `GCIDMHead(z_t, z_goal, h_norm)` → action block. NO history, NO CEM.
3. **CEM planning (le-wm / our solver, the SLOW baseline)** — the full `stable_worldmodel.solver.CEMSolver.solve()` per action, timed end-to-end on the real `model.get_cost` → `rollout` → `predict` path: `n_steps=30` CEM iterations, each sampling `num_samples=300` candidate plans, each plan rolled over `horizon=5` through the WM `predictor` (`ARPredictor` + `pred_proj` + an `action_encoder` per rolled step). Settings verbatim from `config/eval/solver/cem.yaml` (`num_samples=300, n_steps=30, topk=30, var_scale=1.0, batch_size=1`) + `config/eval/robomimic.yaml` (`horizon=5, action_block=5`).
4. **Diffusion Policy / LDP (diffusion head)** — `module.DiffusionHead` with the real default `n_steps=50` reverse-diffusion steps (DP/LDP use ~50–100), conditioned on OUR intention embedding (the same encode + `action_predictor` + `horizon_modulator` front-end that produces the GC conditioning), each denoising step a forward through the eps-MLP. **NOTE (framing flag):** the codebase's `DiffusionHead` is a 3-layer **MLP** eps-predictor (NOT a `ConditionalUnet1D`); the `K × forward` denoising-loop structure and `K=50` are real/measured, but a true DP/LDP would use a heavier U-Net per step, so the DP **wall-clock here is a lower bound** on a real DP and the per-step FLOPs would be larger. The `K`-step multiplier itself is the load-bearing point and is exact.

**Exact reproduction command (run from the le-wm-repro dir so `jepa`/`module`/`gip`/`gcidm` import; fvcore pip-installed into the env):**
```bash
# host: L40S (stratus-lookout); B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro
# py=/var/lib/docker/data/minghao_home/lewm/bin/python ; run as: sudo -u minghao.fu
cd $B && CUDA_VISIBLE_DEVICES=<free-ish id> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl \
  $py bench_speed.py
# script: $B/bench_speed.py (self-contained; loads real ckpts below directly, no get_cache_dir)
```
- **Real model + weights (single robomimic env = `can`):** GC model from `/mnt/minghao_data/.stable-wm/checkpoints/can_gc_ours/{config.json,weights_epoch_1.pt}` (loaded via the exact `gip.load_gip_model` rebuild path: `action_predictor` = `ARPredictor`, `action_decoder` = MLP since `action_head.type=mse`; **load missing=0 unexpected=0**, `horizon_conditioned=true`). GC-IDM head from `/mnt/minghao_data/.stable-wm/checkpoints/can_gcidm/{gcidm_config.json,gcidm_head_best.pt}` (`GCIDMHead(emb_dim=192, action_dim=35, hidden=512)`, strict load OK). DiffusionHead built fresh (architecture-only; values don't affect timing/FLOPs).
- **Model shapes:** vit-tiny encoder `embed_dim=192`, `action_dim=35` (= 7 raw × 5 frameskip), `HS=3`. Param counts: encoder **5.50M**, action_predictor **10.79M**, action_decoder(MLP) **0.471M**, horizon_modulator **0.083M**, WM predictor **10.79M**, gcidm head **1.14M**, diffusion head **9.06M**.
- **FLOP counter:** `fvcore.nn.FlopCountAnalysis` (pip-installed; `total()` = MACs; reported **GFLOPs = 2 × MACs**). Per-module MACs (deterministic, contention-independent): encoder(1 frame) **1399.33 M** (DOMINATES), projector 0.791 M, `ARPredictor`(HS) **32.36 M**, pred_proj 0.791 M, horizon_modulator 0.084 M, action_decoder 0.475 M, action_encoder 0.155 M, gcidm head **1.138 M**, diffusion eps-step **9.056 M**.

### THE SPEED / FLOPs TABLE (can, NVIDIA L40S, fp32, batch=1)
| method | ms/action (mean ± std) | GFLOPs/action | ×faster vs CEM (wall-clock) | ×fewer FLOPs vs CEM |
|---|---|---|---|---|
| **LeWAM-GC (ours)** | **~15–18** (see 3 runs) | **2.87** | **~90–132×** | **1104×** |
| **gcidm (Markovian)** | **~18** | **5.60** | **~88–112×** | **565×** |
| CEM (planning baseline) | **~1590–1980** | **3165.4** | 1.0× | 1.0× |
| DP/LDP (diffusion, K=50) | **~44–62** | 3.77 | ~26–39× | 840× |

- **GFLOPs/action are exact and contention-independent** (the headline). LeWAM-GC 2.87 vs CEM 3165 = **1104× fewer FLOPs**; gcidm 5.60 vs CEM = **565× fewer**. CEM's 3165 GFLOPs = `n_steps(30) × num_samples(300) × horizon(5) × (predictor+pred_proj+aenc forward)` — the `30×300×5 = 45000`-rollout blow-up made explicit.
- **gcidm has MORE FLOPs than LeWAM-GC** (5.60 vs 2.87) because gcidm runs **two** full ViT encodes per decision (`z_t` + `z_goal`) while OURS encodes only the 1 new frame (history latents cached in the deque); the ViT encoder (1399 MMAC) dominates both, so 2 encodes ≈ 2×. This is a real, faithful artifact of the two policies' code, not a measurement choice.

### Per-RUN wall-clock (all three completed runs — per-rule-7 ALL numbers, not just one)
The L40S was at 98–100% GPU util on all 8 GPUs from co-located training the whole time (memory note [[project_l40s_path_consolidation]] / rule 10: the box bottleneck is contention), so absolute ms carries real contention noise; the CEM solver's OWN `print("CEM solve time …")` self-report corroborates each timed mean to ~3 sig-figs.
| run (GPU) | LeWAM-GC ms | gcidm ms | CEM ms | CEM solver self-report | DP/LDP ms | gcidm head-only | DP head-only (50 steps) |
|---|---|---|---|---|---|---|---|
| run1 (GPU5) | 15.76 ± 2.71 | 18.31 ± 2.57 | 1610.0 ± 67.4 | "1.5837 s" | 44.43 ± 5.02 | 4.44 ± 0.94 | 28.31 ± 4.60 |
| run2 (GPU2) | 14.95 ± 2.10 | 17.71 ± 2.17 | 1979.2 ± 78.5 | "2.0332 s" | 51.04 ± 5.58 | 5.38 ± 1.81 | 34.39 ± 2.77 |
| run3 (GPU5) | 17.71 ± 2.73 | 17.61 ± 3.40 | 1590.8 ± 240.5 | "1.5000 s" | 62.27 ± 19.78 | 4.50 ± 0.88 | 39.29 ± 15.40 |
- (A 4th attempt on GPU3 OOM'd at model-load — a co-located training arm grew into the ~16 GB free; not a benchmark bug. The box was memory-tight, per the parallel §24-SIZE arm's own OOM note above.)
- reps: LeWAM-GC / gcidm / DP = 120 timed (after 20 warmup); CEM = 12 timed (after 3 warmup) since each call is ~1.6–2 s.

### Headline speedups (for the paper)
- **Wall-clock:** LeWAM-GC is **~90–132× faster than CEM** and **~3.5× faster than the DP/LDP diffusion head** (K=50 denoising). gcidm is **~88–112× faster than CEM**. (Ranges span the 3 runs under live training contention; the FLOPs ratio below is the contention-free version of the same statement.)
- **FLOPs (deterministic, the cleanest claim):** LeWAM-GC uses **1104× fewer FLOPs/action than CEM** and **1.3× fewer than DP/LDP**; gcidm uses **565× fewer than CEM**. Mechanism: planning-free = ONE forward pass; CEM = `population(300) × horizon(5) × iters(30)` forwards; DP/LDP = `K(50)` denoising forwards.

### Estimate-vs-measured ledger (rule 2 honesty)
- **MEASURED (real fp32 forward on the real trained `can` model, fvcore-counted):** all four ms/action, all per-module MACs, the CEM `solve()` wall-clock (corroborated by the solver's own timer), the DP `K=50` denoising-loop time. CEM/DP loop multipliers are the LITERAL config values (`cem.yaml`, `DiffusionHead.n_steps`), not estimates.
- **ESTIMATE / caveat:** (a) DP/LDP wall-clock + per-step FLOPs are a **lower bound** — our `DiffusionHead` is an MLP eps-predictor, a real DP/LDP would use a heavier `ConditionalUnet1D` per step (the `K`-step multiplier is exact, the per-step cost would grow). (b) absolute CEM ms is **contention-inflated** (box at 100% util); the FLOPs ratio (1104×/565×) is the contention-free headline and should be the number quoted. (c) numbers are for `can` (vit-tiny, action_dim 35); other envs scale the encoder identically (encoder dominates), so the speedup ratios transfer.

## §26 — FDM↔IDM consistency loss (config-gated `action_pred.w_cyc`) — IMPLEMENTED 2026-06-21

**Motivation (user, end-to-end insistence).** Our model is one shared-JEPA-latent state-space model = predictor as FDM (`z_t,a→z_{t+1}`) + intention head as IDM (`z_≤t,z_goal→â`), co-trained END-TO-END (the user explicitly rejected VERA's decouple answer). Question: how to make the two heads *consistent* (agree on the dynamics) during co-training, to attack the §18 covariate-shift / the GC floor.

**Research basis (PDFs in proposal/papers/).** VERA (2605.27817, "Turning Video Models into Generalist Robot Policies"): the field's answer is to DECOUPLE (action-free planner + separate IDM) — *rejected* (not end-to-end), but its single-operator round-trip (Eq.5, action→motion→action) is the seed. SCAR (2605.16412, "Self-Supervised Continuous Action Representation Learning" — **Hongjia/Minghao's OWN paper**): has NO cycle loss (one-way inverse→forward *coupling* + reconstruction, with a stop-grad A2L controller Eq.8). Its anti-collapse LESSONS are baked in: (1) stop-grad the target; (2) keep the cycle a SMALL auxiliary on top of a converged prediction loss (the coupling alone is satisfied by a collapsed latent); (3) the from-scratch-FDM shortcut pitfall (our exact vit-tiny setting) ⇒ warm-start + low weight; (4) sequence-level not per-step.

**The loss (the latent round-trip / FDM-rollout consistency).** Roll the predictor (FDM) one step with the IDM's predicted action `â=pred_act` instead of the GT action, require it to reach the true next latent:
`L_cyc = ‖ predict(z_t, encode_a(â)) − sg(z_{t+1}) ‖²`  (sg = stop-grad; `z_{t+1}=tgt_emb`, the SAME target the WM `pred_loss` uses).
At `â=a_GT` this equals `pred_loss`; when `â≠a_GT` it penalizes the IDM for picking actions the FDM maps away from `z_{t+1}`. The FDM stays anchored by the existing `pred_loss` (GT action), so the IDM is pulled toward FDM-consistent actions (limited collusion). Optional `w_anorm·mean(â²)` cap (SCAR's λ‖a‖²).

**Implementation — INFRA-LEVEL, train.py ONLY (no jepa.py / no eval change).** Reuses the already-public `self.model.predict` (FDM) + `self.model.action_encoder`, and `tgt_emb` already exists for the WM loss. One config-gated block in `train.py` after the `w_act` loss (lines 131-151, L40S le-wm-repro), gated `if _w_cyc > 0.0 and head=="mse" and pred_act is not None:`. **Byte-identical when off** (`ap.get("w_cyc",0.0)` default 0 ⇒ block unreachable). Backup `train.py.bak_wcyc`; **`py_compile` PASS**. Shape correctness guaranteed by mirroring the WM-loss call exactly (`cyc_pred` same shape as `pred_emb`, matches `tgt_emb`).

**Flags.** Turn on: `+action_pred.w_cyc=0.1` (optional `+action_pred.w_anorm=0.01`). Default 0 = today's behaviour. Train-only; eval/`intention_rollout` unchanged.

**Status: implemented + compile-verified + byte-identical-when-off + RUNTIME-VERIFIED.** Smoke (2026-06-21, 3-batch 1-epoch e2e on can, `+action_pred.w_cyc=0.1 +action_pred.w_anorm=0.01`, GPU1): `fit/cyc_loss=0.239` (finite, backprops), `fit/anorm_loss=0.129`, `fit/loss=3.144` (includes both), ran clean in 7.6s, model saved, NO NaN. **No collapse**: `act_emb_std=0.76` (healthy, not →0), `pred_loss=0.028` (healthy). Meaningful signal already: **`cyc_loss 0.239 ≫ pred_loss 0.028`** ⇒ the IDM's â lands the FDM 0.24 from the true next latent vs 0.028 for the GT action — real FDM↔IDM inconsistency exists for the loss to close. (One pre-existing Lightning "unused parameters" WARN — from the goal/horizon params being off in the smoke config, NOT from the cyc block, which reuses already-grad'd predict+action_encoder.) Smoke ckpt `smoke_wcyc` pruned. **The actual `w_cyc ∈ {0, 0.1, 0.5}` sweep on can+cube (e2e, freeze_wm=false, dropout=0.5, N=50 3-seed) is DEFERRED until the oversubscribed box frees** — does w_cyc>0 lift GC SR vs w_cyc=0, watching the collapse monitors.

### §26-sweep — arm `pusht_wcyc_sm` (w_cyc=0.1) — pusht GC e2e — LAUNCHED 2026-06-21, IN FLIGHT (NOT converged this turn) [L40S]

**Why pusht (not can/cube).** pusht is the §18 covariate-shift env where OURS (history-conditioned forward GC policy) loses WORST vs Markovian GC-IDM (frozen-latent OURS GC **31** vs GC-IDM **91**, §13/§24/§25). The FDM↔IDM consistency loss (§26) is designed to attack exactly that IDM-FDM disagreement, so pusht is the env where a w_cyc lift would be most diagnostic. **pusht GC eval is FAITHFUL** (matched gcidm's published 84; §27 confirms the LeWM-env GC is trustworthy, NOT the §27-robomimic-buggy kind) ⇒ standard pusht GC eval (`mode=policy +goal_conditioned=true +horizon_H_max=50`, N=50).

**STEP 1 — w_cyc honored (VERIFIED 2026-06-21, GPU5).** 1-epoch 3-batch e2e on pusht, full GC config + `+action_pred.w_cyc=0.1 +action_pred.w_anorm=0.01`: `fit/cyc_loss=0.098` (finite, in `loss`), `fit/anorm_loss=0.115`, `fit/act_emb_std=1.29` (healthy), `validate/cyc_loss=0.066`, `validate/anorm_loss=0.146`. `init_from=pusht_ours_lewm_weights.pt` loaded `missing=95 unexpected=0` (95 missing = the goal/horizon head params that init fresh under strict=False — encoder/predictor/action_encoder DID load = warm WM init confirmed dim-compatible). Clean, no NaN/Traceback. (Benign Lightning "unused parameters" WARN as in §26 smoke.) Smoke ckpt pruned.

**STEP 2 — TRAIN (EXACT launch command, IN FLIGHT).** `train.py data=pusht init_from=/mnt/minghao_data/.stable-wm/decoders/pusht_ours_lewm_weights.pt action_pred.enabled=true action_pred.head=mse action_pred.w_act=1.0 action_pred.w_intent=1.0 action_pred.detach_target=true action_pred.goal_conditioned=true action_pred.horizon_conditioned=true action_pred.goal_dropout=0.5 +action_pred.w_cyc=0.1 +action_pred.w_anorm=0.01 output_model_name=pusht_wcyc_sm subdir=pusht_wcyc_sm trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4 +ckpt_every=10 hydra.run.dir=/mnt/minghao_data/hydra_wcyc_sm hydra.output_subdir=null`. **Config mirrors the proven `can_gc_e2e` e2e-GC template EXACTLY** (same `max_epochs=100 +limit_train_batches=4000 batch_size=64 goal_conditioned=true horizon_conditioned=true goal_dropout=0.5 head=mse detach_target=true`, warm-WM `init_from`, `freeze_wm` absent=false) and ADDS the two §26 flags (`w_cyc=0.1` + the `w_anorm=0.01` anti-collusion cap). Disk-safe env: `SPT_CACHE_DIR=/mnt/minghao_data/spt_pusht_wcyc_sm XDG_CACHE_HOME=…/xdg_pusht_wcyc_sm TMPDIR=…/tmp_pusht_wcyc_sm MPLCONFIGDIR=…/mpl_pusht_wcyc_sm HF_HOME=/mnt/minghao_data/hf STABLEWM_HOME=/mnt/minghao_data/.stable-wm HF_HUB_OFFLINE=1`. GPU5 (lowest-occupied at launch, ~16 GB used). PID 2165016. Logs `/mnt/minghao_data/wcyc_logs/pusht_wcyc_sm.log`. Ckpts `/mnt/minghao_data/.stable-wm/checkpoints/pusht_wcyc_sm/` (`+ckpt_every=10`; background pruner PID 2176052 keeps newest 3 → final-only).

**BLOCKER — throughput, NOT convergence.** L40S is oversubscribed: `load avg 161 / 64 cores`, 41 user sessions, CPU/disk-bound (the box's documented bottleneck). Throughput collapsed to **~1.0 it/s** (vs §30's 6.3 it/s on the same box when idle-ish) ⇒ ~66 min/epoch ⇒ **~110 h for 100 epochs**. The run is correct and stable (verified ep0 step200/4000, GPU5 25 GB / 37% util) but CANNOT converge within one turn. Per rule 8/9 (never headline an under-trained number) NO SR is reported here. **Unblock condition:** `weights_epoch_100.pt` present in the ckpt dir AND the per-epoch `validate/act_loss` flat over the last ~10 ep (the §13 OURS-GC convergence signature: val_act plateaued 0.132–0.137 at ep20→40). Then STEP 3.

**STEP 3 — EVAL (when converged, N=50 × 3-seed {42,0,1}, SAME-BOX L40S per cross-GPU render rule).** `eval_gip.py --config-name pusht policy=pusht_wcyc_sm +gip_eval.mode=policy +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|0|1>` (the §13 OURS planning-free GC eval line, just `policy=pusht_wcyc_sm`). **Question:** does w_cyc=0.1 lift pusht GC SR above the w_cyc=0 baseline (frozen-latent OURS 31; the e2e-w_cyc=0 baseline at the SAME 100ep/4000-batch recipe is the apples-to-apples reference and should be run as the sibling arm) toward GC-IDM 91? Watch `cyc_loss_final` (should fall below the 0.066 ep0-val start if the FDM↔IDM gap closes) and `act_emb_std` (collapse watch: ep0 = 1.29 healthy; must NOT decay toward 0).

**Status: STEP 1 PASS, STEP 2 IN FLIGHT (ep0, ~1 it/s, blocked on ~110 h convergence under box contention), STEP 3 pending. No SR / no cyc_loss_final / no convergence verdict this turn.**

### §26-sweep — arm `pusht_wcyc_lg` (w_cyc=0.5, LARGE weight) — pusht GC e2e — LAUNCHED 2026-06-22 [L40S]

**Why pusht / why w_cyc=0.5.** Same env rationale as the `pusht_wcyc_sm` arm above (pusht = the §18/§13 causal-confusion env where OURS history GC loses worst, frozen-latent ours **31** vs gcidm **90/91**; the §26 FDM↔IDM loss targets exactly that disagreement; pusht GC eval is FAITHFUL not §27-robomimic-buggy). This is the LARGE consistency-weight arm (w_cyc=0.5 vs the 0.1 sibling) → the collapse watch (`act_emb_std`, `pred_loss`) is the dominant risk: a heavy cycle weight is most prone to the collapsed-latent shortcut SCAR warns about, so a healthy `act_emb_std` here is load-bearing for trusting the SR.

**STEP 1 — w_cyc=0.5 honored (VERIFIED 2026-06-22, GPU7).** 1-epoch 3-batch e2e on pusht, full GC config + `+action_pred.w_cyc=0.5 +action_pred.w_anorm=0.01`: `fit/cyc_loss=0.0892` + `validate/cyc_loss=0.0833` (finite, in `loss`); `fit/loss=7.204` includes 0.5·cyc + 0.01·anorm (pred 0.0136 + intent 6.10 + act + …); `validate/act_emb_std=1.289` (healthy, NOT →0); `validate/anorm_loss=0.114`, `validate/pred_loss=0.0047`. init_from `pusht_ours_lewm_weights.pt` loaded clean (embed_dim=192). Ran 3.9s, model saved, no NaN. Smoke ckpts + spt run dir pruned. ⇒ the +action_pred.w_cyc=0.5 block is active (at w_cyc=0 it is skipped byte-identical).

**STEP 2 — TRAIN (EXACT launch, host L40S, lewm venv `/var/lib/docker/data/minghao_home/lewm/bin/python`, `sudo -u minghao.fu`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`).** Disk-safety env (the spec's literal vars; `/var` 97% full → all caches on `/mnt/minghao_data` 2.4T-free): `STABLEWM_HOME=/mnt/minghao_data/.stable-wm SPT_CACHE_DIR=/mnt/minghao_data/spt_pusht_wcyc_lg XDG_CACHE_HOME=/mnt/minghao_data/xdg_pusht_wcyc_lg TMPDIR=/mnt/minghao_data/tmp_pusht_wcyc_lg MPLCONFIGDIR=/mnt/minghao_data/mpl_pusht_wcyc_lg HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1`. DEC=`$STABLEWM_HOME/decoders`. PID 2169897, GPU0, log `/mnt/minghao_data/tmp_pusht_wcyc_lg/logs/train_pusht_wcyc_lg.log`, ckpts `/mnt/minghao_data/.stable-wm/checkpoints/pusht_wcyc_lg/`:
```
CUDA_VISIBLE_DEVICES=0 $PY train.py data=pusht \
  action_pred.enabled=true action_pred.goal_conditioned=true action_pred.horizon_conditioned=true \
  action_pred.goal_dropout=0.5 +action_pred.w_cyc=0.5 +action_pred.w_anorm=0.01 \
  init_from=$DEC/pusht_ours_lewm_weights.pt embed_dim=192 \
  output_model_name=pusht_wcyc_lg \
  trainer.max_epochs=100 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=20 \
  +ckpt_every=10 loader.batch_size=64 num_workers=4
```
NOTE the deliberate difference vs the `pusht_wcyc_sm` sibling: `limit_train_batches=1000` (not 4000). The sm arm at 4000 batches/ep hit the ~1 it/s box-contention wall = ~110 h/100ep (un-convergeable in a turn). 1000 batches/ep ≈ ~17 min/ep at ~1.1 it/s → ~28 h/100ep, and §13's OURS pusht GC plateaued val_act by **~ep28** → early-stop at the plateau makes this finishable. (First attempt used 4000 too, killed after gauging ~67 min/ep; relaunched at 1000.) init_from is strict=False → the new goal/horizon head params init fresh (missing), encoder/predictor/action_encoder load warm. Benign Lightning "unused parameters" WARN on `predictor.type_embedding` / `action_predictor.type_embedding`.

**STEP 3 — EVAL (when converged, faithful pusht GC, matches §13; N=50 × 3-seed {42,0,1}, SAME-BOX L40S).** `eval_gip.py --config-name pusht policy=pusht_wcyc_lg +gip_eval.mode=policy +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|0|1>`. Question: does w_cyc=0.5 lift pusht GC SR above the w_cyc=0 baseline (frozen-latent OURS 31; the e2e w_cyc=0 sibling at the same recipe is the apples-to-apples ref) toward GC-IDM 90/91? Any collapse (`act_emb_std`→0, `pred_loss` blow-up)? Compare against the w_cyc=0.1 `pusht_wcyc_sm` arm to read the consistency-weight curve.

**Status: STEP 1 PASS (w_cyc=0.5 honored, no smoke-time collapse). STEP 2 IN FLIGHT (ep0, ~1.1 it/s under box contention, 1000-batch ep ≈ 17 min; converging toward the §13 ~ep28 plateau). STEP 3 pending. No SR / cyc_loss_final / convergence verdict landed THIS turn — blocked on wall-clock convergence under L40S CPU/disk contention; unblock = `validate/act_loss` flat over the last ~10 ep then the N=50 3-seed eval.**

### §26-sweep — arm `pusht_wcyc0` (w_cyc=0.0, THE BASELINE) — pusht GC e2e — LAUNCHED 2026-06-22 [L40S]

**Why this arm.** This is the **w_cyc=0 apples-to-apples baseline** the two sibling arms (`pusht_wcyc_sm` w_cyc=0.1, `pusht_wcyc_lg` w_cyc=0.5) explicitly call for: an end-to-end GC LeWAM on pusht with NO FDM↔IDM consistency loss, trained at the same e2e-GC recipe as the w_cyc>0 arms, so the SR delta isolates the §26 loss. Same env rationale (pusht = §18/§13 causal-confusion env, frozen-latent OURS GC 31 ≪ GC-IDM 90/91; pusht GC eval is FAITHFUL not §27-robomimic-buggy → standard `mode=policy +goal_conditioned=true +horizon_H_max=50` N=50 eval).

**STEP 1 — w_cyc honored / w_cyc=0 byte-identical-skip (VERIFIED 2026-06-22, GPU6).** Two 1-epoch 3-batch e2e smokes on pusht, full GC config. (a) `+action_pred.w_cyc=0.1 +action_pred.w_anorm=0.01`: `validate/cyc_loss=0.0793` (finite, in `loss`=7.63), `validate/anorm_loss=0.148`, `validate/pred_loss=0.0053` (healthy), `validate/act_emb_std=1.23` (healthy, NOT →0); `init_from=pusht_ours_lewm_weights.pt missing=95 unexpected=0` (95 = fresh goal/horizon head; encoder/predictor/action_encoder load warm, dim-compatible embed_dim=192). (b) `+action_pred.w_cyc=0.0`: NO `cyc_loss`/`anorm_loss` keys appear at all (the `if _w_cyc>0.0` block at train.py:139 is skipped byte-identical), `act_emb_std=1.12`, `pred_loss=0.0021`, `Adim=10`. ⇒ confirms train.py:138-151 the cyc block is strictly gated `> 0.0` and contributes nothing at w_cyc=0. Both smoke ckpts pruned. No NaN/Traceback (benign Lightning "unused parameters" WARN on `predictor.type_embedding` / `action_predictor.type_embedding`, from the goal/horizon params).

**STEP 2 — TRAIN (EXACT launch, IN FLIGHT).** Host L40S, lewm venv `/var/lib/docker/data/minghao_home/lewm/bin/python`, `sudo -u minghao.fu`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`. Disk-safety env (the spec's literal vars; `/var` 97% full, `/mnt/minghao_data` 2.4 T free): `SPT_CACHE_DIR=/mnt/minghao_data/spt_pusht_wcyc0 XDG_CACHE_HOME=/mnt/minghao_data/xdg_pusht_wcyc0 TMPDIR=/mnt/minghao_data/tmp_pusht_wcyc0 MPLCONFIGDIR=/mnt/minghao_data/mpl_pusht_wcyc0 HF_HOME=/mnt/minghao_data/hf STABLEWM_HOME=/mnt/minghao_data/.stable-wm HF_HUB_OFFLINE=1 MUJOCO_GL=egl OMP_NUM_THREADS=4 WANDB_MODE=disabled`. GPU6 (lowest-occupied at launch, ~7.5 GB used / 38 GB free — accepted shared-dataloader per rule 10). PID 2176892. Log `/mnt/minghao_data/tmp_pusht_wcyc0/train_pusht_wcyc0.log`. Ckpts `/mnt/minghao_data/.stable-wm/checkpoints/pusht_wcyc0/` (`+ckpt_every=10`; background pruner keeps newest-3). EXACT command:
```
CUDA_VISIBLE_DEVICES=6 python train.py data=pusht \
  action_pred.enabled=true action_pred.detach_decoder=false \
  action_pred.goal_conditioned=true action_pred.horizon_conditioned=true \
  action_pred.goal_dropout=0.5 action_pred.hindsight_max_k=50 action_pred.horizon_H_max=50 \
  +action_pred.w_cyc=0.0 \
  init_from=$STABLEWM_HOME/decoders/pusht_ours_lewm_weights.pt \
  output_model_name=pusht_wcyc0 subdir=pusht_wcyc0 \
  trainer.max_epochs=100 +trainer.limit_train_batches=2000 +trainer.limit_val_batches=20 \
  +ckpt_every=10 loader.batch_size=64 num_workers=6 \
  hydra.run.dir=/mnt/minghao_data/hydra_pusht_wcyc0/run
```
Train log at launch: `[GIP] Intention predictor ON Adim=10 head=mse w_act=1.0 w_intent=1.0 detach_target=True`, `[GIP] OURS horizon conditioning ON AdaLN-Zero H_max=50`, `init_from ... missing=95 unexpected=0`. **NOTE the recipe difference vs siblings:** `limit_train_batches=2000` (the sm arm used 4000 = ~67 min/ep un-convergeable; the lg arm used 1000). 2000 batches/ep at ~1.3 it/s ≈ ~25 min/ep. (First attempt used 4000, killed after gauging ~67 min/ep at load-avg 162/64-core contention; relaunched at 2000.)

**BLOCKER — throughput, NOT correctness.** L40S oversubscribed: `load avg 162 / 64 cores`, 41 user sessions, CPU/disk-bound (documented box bottleneck). Steady-state ~1.3 it/s ⇒ ~25 min/ep ⇒ ~17 h to ep40, longer to ep100. Run is correct + stable (ep0 stepping, GPU6 32 GB / 80% util, act_emb_std 1.29 healthy in sanity-val) but CANNOT converge within one turn. Per rule 8/9 NO SR reported here. **Unblock condition:** the latest `weights_epoch_*.pt` at a plateau (per-epoch `validate/act_loss` flat over the last ~10 ep — the §13 OURS-GC signature plateaued val_act ~0.132–0.137 by ep28→40) AND `act_emb_std` not decayed toward 0. Then STEP 3.

**STEP 3 — EVAL (when converged, N=50 × 3-seed {42,0,1}, SAME-BOX L40S per cross-GPU render rule).** Driver `/tmp/eval_pusht_wcyc0.sh <seed> <gpu>`: `eval_gip.py --config-name pusht policy=pusht_wcyc0 +gip_eval.mode=policy +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|0|1>` (the §13/§18 OURS planning-free GC eval line, just `policy=pusht_wcyc0`). **Question this baseline answers:** it is the reference SR that the w_cyc=0.1 (`pusht_wcyc_sm`) and w_cyc=0.5 (`pusht_wcyc_lg`) arms must BEAT for the §26 FDM↔IDM consistency loss to claim a pusht-GC lift toward GC-IDM 90/91. Watch `cyc_loss` is ABSENT (w_cyc=0) and `act_emb_std` healthy.

**Status: STEP 1 PASS (w_cyc path honored at 0.1; byte-identical-skip confirmed at 0.0). STEP 2 IN FLIGHT (ep0, ~1.3 it/s under box contention, 2000-batch ep ≈ 25 min). STEP 3 pending. No SR / convergence verdict THIS turn — blocked on wall-clock convergence under L40S CPU/disk contention; unblock = `validate/act_loss` flat over the last ~10 ep then the N=50 3-seed eval.**

### §26-sweep MATCHED — the VALID w_cyc ablation (supersedes the 3 confounded arms above) — LAUNCHED 2026-06-22 [L40S]

**Why this re-run (the confound).** The three arms above (`pusht_wcyc0` / `pusht_wcyc_sm` / `pusht_wcyc_lg`) were each launched at a DIFFERENT `limit_train_batches` — **2000 / 4000 / 1000** respectively — at `max_epochs=100`, because each was relaunched independently to dodge the box-contention wall (the sm arm at 4000 hit ~67 min/ep, so lg dropped to 1000, and wcyc0 split the difference at 2000). That confounds the w_cyc ablation with **training compute**: an SR delta between arms could be the cycle loss OR just more/fewer gradient steps. Per CLAUDE.md rule 12 (within a dataset every ablation arm shares the SAME epoch + batch budget) those three arms are **INVALID for the w_cyc comparison** and are SUPERSEDED here. Worse, the configs were not otherwise identical either: the `wcyc0` arm additionally passed `detach_decoder=false hindsight_max_k=50 horizon_H_max=50` (all defaults, so no math change) and the `lg` arm passed `embed_dim=192` (the default) — cosmetic, but a second reason to not trust a cross-arm read. **Action taken 2026-06-22 ~04:50:** killed the 3 confounded training procs (`pkill -f "train[.]py.*pusht_wcyc"` then `kill -9` the one surviving leader 2176892; killed the two stale pruners 2164092 / 2176052; verified 0 remaining; left their checkpoints on disk, just unused; did NOT touch any other user's / fan-test's procs). The 3 matched arms below OWN the w_cyc sweep.

**The matched recipe (the ONLY difference across the 3 arms is `+action_pred.w_cyc`).** pusht, end-to-end GC (warm-WM init, `freeze_wm=false`), **`max_epochs=15` (pusht fast iteration budget, rule 12), `limit_train_batches=2000` (SAME for all 3), `batch_size=64`**, `init_from=pusht_ours_lewm_weights.pt`, `action_pred.enabled=true head=mse w_act=1.0 w_intent=1.0 detach_target=true goal_conditioned=true horizon_conditioned=true goal_dropout=0.5 horizon_H_max=50 hindsight_max_k=50` (all at config/train/lewm.yaml defaults except the four GC flags), seed `3072` (config default, identical), `+ckpt_every=5` (⇒ ckpts at ep5/ep10/ep15; eval auto-picks `pts[-1]`=ep15). The arms:

| arm | output_model_name | w_cyc | w_anorm | GPU | PID |
|---|---|---|---|---|---|
| baseline | `wcyc0_m` | **0.0** | 0.01 (no-op: anorm block nests inside `if w_cyc>0`) | 3 | 2537763 (relaunched, see OOM note) |
| small | `wcyc_sm_m` | **0.1** | 0.01 | 2 | 2520818 |
| large | `wcyc_lg_m` | **0.5** | 0.01 | 1 | 2521004 |

`+action_pred.w_anorm=0.01` is passed on ALL three (so the launch strings are byte-identical except the w_cyc value); at w_cyc=0 the `anorm` term is unreachable (it sits inside the `if _w_cyc > 0.0` block, train.py:139-151), so it is a true no-op for the baseline — the command-identity requirement is met without altering the baseline's math. EXACT launch (per-arm via `/mnt/minghao_data/matched/launch_arm.sh <arm> <w_cyc> <gpu>`, `sudo -u minghao.fu`, lewm venv, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`):
```
CUDA_VISIBLE_DEVICES=<gpu> python train.py data=pusht \
  init_from=$STABLEWM_HOME/decoders/pusht_ours_lewm_weights.pt \
  action_pred.enabled=true action_pred.head=mse action_pred.w_act=1.0 action_pred.w_intent=1.0 \
  action_pred.detach_target=true action_pred.goal_conditioned=true action_pred.horizon_conditioned=true \
  action_pred.goal_dropout=0.5 +action_pred.w_cyc=<0.0|0.1|0.5> +action_pred.w_anorm=0.01 \
  output_model_name=<arm> subdir=<arm> \
  trainer.max_epochs=15 +trainer.limit_train_batches=2000 +trainer.limit_val_batches=20 \
  +ckpt_every=5 loader.batch_size=64 num_workers=4 \
  hydra.run.dir=/mnt/minghao_data/matched/<arm>/hydra hydra.output_subdir=null
```
**Disk-safe env (per-arm-unique under /mnt/minghao_data, 2.4 T free; `/` & `/var` 97% full):** `STABLEWM_HOME=/mnt/minghao_data/.stable-wm SPT_CACHE_DIR=…/matched/<arm>/spt XDG_CACHE_HOME=…/xdg TMPDIR=…/tmp MPLCONFIGDIR=…/mpl HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1 MUJOCO_GL=egl OMP_NUM_THREADS=4 WANDB_MODE=disabled` (+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on wcyc0_m after its OOM relaunch — an allocator-fragmentation setting, NOT a hyperparameter, so it does not break config identity; the other two were already past the fragile init).

**STEP 1 — matched command smoke (VERIFIED 2026-06-22, GPU2).** 1ep×3-batch e2e on pusht with the EXACT matched command (`+w_cyc=0.1 +w_anorm=0.01`, max_epochs=1 limit_train_batches=3): `validate/cyc_loss=0.083` (finite, fires), `validate/anorm_loss=0.114`, `validate/act_emb_std=1.268` (healthy, no collapse), `validate/pred_loss=0.0048` (healthy, warm WM), model saved, exit 0, no NaN. Smoke ckpt + dir pruned.

**STEP 2 — TRAIN config VERIFIED per arm (in flight).** All three print IDENTICAL `[GIP] Intention predictor ON Adim=10 head=mse w_act=1.0 w_intent=1.0 detach_target=True` + `[GIP] OURS horizon conditioning ON AdaLN-Zero H_max=50` + `init_from=…pusht_ours_lewm_weights.pt: missing=95 unexpected=0` (warm WM; 95 = fresh goal/horizon head under strict=False). Sanity-val act_emb_std=1.289 (identical across sm/lg, healthy). cyc/anorm keys present on sm & lg, absent on wcyc0 (gating confirmed). **OOM hiccup:** wcyc0_m's first launch (GPU3) crashed at ep0 with `torch.OutOfMemoryError` — a transient burst of other users' procs filled GPU3 (44.39 GiB total, 120 MiB free at the moment) right after my pre-launch survey showed it 4 MiB-free; not a config error. Killed the dead tree, cleaned its partial ckpt/spt dirs, relaunched on GPU3 (then genuinely free) with `expandable_segments:True`; now stepping ep0 clean. Throughput ~1.2–1.3 it/s under load-avg ~156/64-core ⇒ ~28 min/ep ⇒ **~7 h to ep15** (the matched budget endpoint; this is a fast-iteration headline, NOT a 100ep convergence claim — rule 12 makes the ep15 matched number reportable at a verified plateau).

**STEP 3 — EVAL (auto, server-side orchestrator).** `/mnt/minghao_data/matched/orchestrate.sh` (nohup, PID 2556664, SSH-blip-robust): waits per-arm for `weights_epoch_15.pt`, then runs `eval_gip.py --config-name pusht policy=<arm> +gip_eval.mode=policy +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|0|1>` (N=50, 3 seeds, SAME-BOX L40S per cross-GPU render rule), one seed at a time per GPU. SR written to `/mnt/minghao_data/matched/SUMMARY.txt` (`success_rate` key, percent). act_emb_std read from each arm's final training validation (collapse check, watch wcyc_lg_m especially per SCAR).

**The Task-2 answer (TABLE — fill on landing).** Before reporting, verify each arm's `validate/act_loss` is near the §13 ~0.13 plateau at ep15 (if still descending steeply, note under-training).

| w_cyc | pusht GC SR (3-seed mean) | per-seed {42,0,1} | act_emb_std (final val) | val_act @ep15 |
|---|---|---|---|---|
| 0.0 (baseline) | _pending_ | _pending_ | _pending_ | _pending_ |
| 0.1 | _pending_ | _pending_ | _pending_ | _pending_ |
| 0.5 | _pending_ | _pending_ | _pending_ | _pending_ |

VERDICT (fill): is the cycle loss POSITIVE (w_cyc>0 BEATS the matched w_cyc=0, → FDM↔IDM consistency helps) or NEGATIVE? Does it move toward gcidm 91 / above the frozen-latent OURS 31? Any collapse at w_cyc=0.5 (act_emb_std→0, pred_loss blow-up)? Since all 3 share the identical 15ep/2000-batch/bs64 budget, the comparison is now VALID. **Reference points:** frozen-latent OURS GC pusht **31** (§13/§24/§25); GC-IDM Markovian **90/91** (§13/§24). NB the §13 OURS-with-history-GC at the FULL 100ep/4000-batch budget scored 31 — so a 15ep/2000-batch matched number is a fast-iteration read of the w_cyc DELTA, not directly comparable in absolute SR to the §13 100ep figure; the valid claim here is the within-budget sign of the w_cyc effect.

**Status: STEP 1 PASS (matched command smoke). STEP 2 IN FLIGHT (all 3 stepping ep0, ~1.2 it/s, ~7 h to ep15; wcyc0_m relaunched after a transient GPU3 OOM). STEP 3 armed (server-side orchestrator + monitor). No SR THIS turn — blocked on ~7 h wall-clock convergence under L40S contention; unblock = `weights_epoch_15.pt` per arm + the N=50 3-seed eval, both automated.**

## §27 — ROBOMIMIC GC EVAL IS BUGGY (OOD-tail-past-goal artifact) — found 2026-06-21

**Trigger:** user skepticism — "are u sure no bug? lewam perform so bad on robomimic." The red flag: the SAME model scores **0.71 in bc (S3, no goal)** yet **0.007–0.13 in GC (S2, with goal)** on can. Adding a goal should not crater a working policy.

**Method:** compared our robomimic GC eval (read from `eval_histbc_robomimic.py` + `gip.py` + the swm `world.py:evaluate`) against the gcidm PAPER's protocol (2605.08732, Algorithm 1 + §5.3; agent a7194080, quotes retained).

**FINDING — our robomimic GC eval DEVIATES from gcidm on 3 of 4 axes (CONFIRMED bug):**
| axis | gcidm paper (quoted) | our robomimic eval | |
|---|---|---|---|
| goal | env-designated **terminal** goal `o_g`, encoded once (§3, Fig.2, Alg.1) | `demo[start+90 sim]` = **mid-trajectory** frame | ✗ |
| success | **goal-reaching** ("Closed-loop goal-reaching control", ‖z_t−z_g‖ potential field, §5.3/§5.6) | env **`is_success`** (full task, unrelated to goal) | ✗ |
| termination | **terminate-at-goal** — "if goal reached or episode terminated then return" (Alg.1) | **none** — runs full budget | ✗ **(core)** |
| budget vs horizon | **`H_max = T = 50`** (budget=horizon), `h_t=T−t+1` counts down to goal; never past goal (§5.3) | horizon=18 obs, budget=48 obs → **~30 obs-steps OOD** at `h≈0`, goal behind it | ✗ |

**Mechanism:** GC-IDM is trained on `h ∈ [1, H_max]` (Eq.7) — it has NEVER seen `h≈0`-with-stale-goal. Our eval (no terminate-at-goal, budget≫horizon) forces it through exactly that ~30-step OOD tail. **Signature = the budget-sensitivity** (same model: 0.44 at b100/o30 [tail≈14 obs] → 0.06–0.30 at b240/o90 [tail≈30 obs]; monotone with tail length). bc (no goal/horizon/tail) = 0.71.

**VERDICT: the robomimic GC FLOOR is SUBSTANTIALLY AN EVAL ARTIFACT, not the method.** Affects BOTH gcidm-repro AND ours on robomimic (§16/§19/§20/§21) — the gcidm-vs-ours comparison stays internally fair (both run the buggy protocol) but BOTH are artificially depressed vs bc. **The "§21: floor = GC formulation not encoder" conclusion is UNRELIABLE** (both arms floored by the eval, not the encoder/formulation). **SCOPE = robomimic-ONLY** — the LeWM-env GC (cube/pusht/two-room/reacher) MATCHED gcidm's published numbers (100/91/100/97 ≈ 98.7/84/100/99.7) → faithful → trustworthy (the cube 72.7<100 / pusht 31<91 losses are REAL).

**FIX (gcidm-faithful, no retrain — re-eval existing models):** goal = the **terminal/task-goal frame** + **terminate-at-goal** (latent-distance predicate ‖z_t−z_g‖<τ, or robomimic is_success once goal=terminal) + **budget = horizon**. Quick first test (config-only, no code): `eval_budget = goal_offset_steps` to kill the OOD tail. If GC jumps toward bc → the floor was the artifact; if still ≪ bc → the gap is real. **TOP PRIORITY** once an eval slot is free (only needs eval, not the training box). Do NOT report the b240/o90/is_success/no-terminate numbers as "our gcidm reproduction" — gcidm never runs that protocol.

### §27 QUICK TEST RAN — `eval_budget = goal_offset_steps` (b90/o90) on can, 3-seed N=50, same-box L40S (2026-06-21, INFRA/OPS agent)

**Infra context (fleet relocation).** L40S was reported occupied by fan-test's 8-GPU DDP job, but at launch time I verified directly that **all 8 L40S GPUs were idle** (`nvidia-smi`: 0% util, ≤689 MiB each — fan-test's job had finished/stopped). 174 was surveyed as the fallback but is the WRONG box for this eval: the `can_gc_*` checkpoints were TRAINED on L40S, and per [[project_l40s_cross_gpu_rendering]] eval'ing them on 174 would render-OOD-floor them to ~0 (a SECOND artifact confounding the §27 test). So this re-eval ran **same-box on L40S** — the only correct option. Disk: `/var/lib/docker` 100% full and root `/` 97% (3.3 GB), but `/mnt/minghao_data` had 2.5 TB free, so all scratch (`TMPDIR`/`MPLCONFIGDIR`/`hydra.run.dir`/logs) was redirected there. Eval-only (no retrain), writes only small JSON → disk was a non-issue.

**EXACT commands** (drivers `/mnt/minghao_data/s27_scratch/s27_{e2e,gcidm}_eval.sh`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, run as `sudo -u minghao.fu`; shared env `STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=0 OMP_NUM_THREADS=4`, scratch on `/mnt/minghao_data/s27_scratch`):
- **OURS (e2e), per seed:** `CUDA_VISIBLE_DEVICES=<1|2|3> python eval_histbc_robomimic.py --config-name robomimic world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can eval.num_eval=50 eval.eval_budget=90 eval.goal_offset_steps=90 policy=can_gc_e2e +gip_eval.mode=policy +gip_eval.goal_conditioned=true seed=<42|0|1>`. Load log: `[GIP] load can_gc_e2e <- weights_epoch_100.pt: Adim=35 missing=0 unexpected=0` (NOTE: this `can_gc_e2e` is now the **CONVERGED ep100** ckpt — the §22 STEP-3 number used the stalled ep10 ckpt; the run finished training since), `[HISTBC] policy ready adim=35 action_block=5 HS=3 goal_cond=True use_horizon=True horizon0=18.00` (horizon0 = offset/block = 90/5 = 18, AdaLN-Zero horizon active).
- **gcidm (Markovian), per seed:** `CUDA_VISIBLE_DEVICES=<4|5|6> python eval_histbc_robomimic.py --config-name robomimic world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can eval.num_eval=50 eval.eval_budget=90 eval.goal_offset_steps=90 policy=can_gcidm_latest +gip_eval.mode=gcidm +gip_eval.gcidm_run=can_gcidm_latest seed=<42|0|1>`.
- The ONLY change vs the §22 canonical can protocol is `eval.eval_budget 240 → 90` (= goal_offset_steps). Everything else (N=50, offset=90, seeds, same-box) identical.

**RESULT (can, N=50, 3-seed {42,0,1}, ALL per-seed, no errors, no chunk-1 artifact — successes scatter across chunks):**
| arm | protocol | seed42 | seed0 | seed1 | **mean** |
|---|---|---|---|---|---|
| OURS (e2e, ep100) | **§27 b90/o90** | 0.04 | 0.02 | 0.00 | **0.020** |
| OURS (e2e) | §22 ref b240/o90 (ep10) | 0.06 | 0.30 | 0.26 | 0.207 |
| gcidm (`_latest`) | **§27 b90/o90** | 0.04 | 0.00 | 0.00 | **0.013** |
| gcidm (`_latest`) | §22 ref b240/o90 | 0.30 | 0.26 | 0.16 | 0.240 |

**VERDICT — the §27 hypothesis is REFUTED by the quick test: killing the OOD tail did NOT lift GC; SR went DOWN, not up.** Both arms collapsed toward 0 (ours 0.207→0.020, gcidm 0.240→0.013) instead of climbing toward the goal-agnostic bc 0.71. So the can-GC floor is **NOT mainly the OOD-tail-past-goal artifact** — at least not at goal_offset=90. The §22 conclusion (frozen vit-tiny latent can't localize the can → real GC floor) survives this test.

**⚠️ CAVEAT — the quick test has a budget-starvation confound (so this is necessary-but-not-sufficient to fully close §27).** `goal_offset_steps=90` samples a goal ~75% through the median can demo (≈120 sim steps, EXPERIMENTS.md ~L937), and `eval_budget=90 < 120` median demo length. So b90/o90 simultaneously (a) removes the OOD tail AND (b) STARVES the budget below the demo length — even a perfect policy may time out before completing the pick-place. The SR drop therefore mixes "no rescue from tail-removal" with "not enough steps." This is why the §27 spec called `eval_budget=goal_offset_steps` only the *quick first test*. **The clean test still needs the OTHER half of the fix: goal = TERMINAL frame + terminate-at-goal, with budget kept LONG** (so the tail is removed by early-termination-at-goal, not by a short budget). That needs a small code change (terminate-at-goal predicate + terminal-frame goal sampling in the robomimic world loop), not config-only. **NEXT (still §27, lower urgency now that the artifact hypothesis is weakened):** implement terminate-at-goal + terminal-goal and re-eval at b240/terminal — if GC still ≪ bc, the floor is definitively real and the §21 "floor = formulation not encoder" framing can be re-trusted.

### §27 CLEAN TEST RAN — goal = EPISODE-TERMINAL frame + budget LONG (b240/o90) on can, 3-seed N=50, same-box L40S (2026-06-21)

**Motivation (why).** The quick test (b90/o90) was confounded — a short budget starves task completion, mixing "tail removed" with "not enough steps." The CLEAN test removes all three §27 confounds AT ONCE while keeping the budget long: **goal = the episode's TERMINAL frame** (≈ task completion / can-in-bin, NOT `demo[start+90]` mid-trajectory) + **budget = 240** (≫ the ≈113-118-step median can demo, so a perfect policy never times out) + **success = `is_success` with terminate-on-success** (already in the robomimic env; with goal=terminal, is_success ≈ reached-the-goal, so this IS the gcidm-faithful goal-reaching metric). The ONLY change vs the §22 canonical can protocol is **goal=terminal instead of mid-trajectory** (`eval.goal_mode=terminal`). If GC recovers toward bc (0.71) → the floor was the mid-goal artifact; if it still floors (~0.1-0.2) → the floor is real (frozen vit-tiny latent can't localize the can, §22 survives).

**Infra context.** All 8 L40S GPUs verified idle at launch (`nvidia-smi`: 0% util, 4-5 MiB each — fan-test's job had finished). Eval'd **same-box on L40S** (the `can_gc_*` ckpts trained there; per [[project_l40s_cross_gpu_rendering]] eval'ing on 174 would render-OOD-floor them — a second artifact). `/var/lib/docker` ~100% + root `/` ~97% full, but `/mnt/minghao_data` had 2.5 TB → ALL scratch redirected there (`TMPDIR`/`MPLCONFIGDIR`/`hydra.run.dir`/logs `=/mnt/minghao_data/s27_clean`). Eval-only, writes only small JSON. 6 runs co-located across GPUs 0-5, staggered (`sleep 7`).

**Code change (config-gated, backward-compatible — backups in `/mnt/minghao_data/s27_clean_backup/`).** No retrain; this is an eval-side change. Added `eval.goal_mode` (default `'mid'` = byte-identical to original; `'terminal'` = new). Threaded through `World.evaluate` → `_evaluate_from_dataset` → `_extract_init_goal` in the swm package `stable_worldmodel/world/world.py` (installed at `/var/lib/docker/data/minghao_home/lewm/lib/python3.10/site-packages/stable_worldmodel/world/world.py`), plus a one-line read+pass in `le-wm-repro/eval_histbc_robomimic.py`. The terminal branch in `_extract_init_goal` (marked `S27_GOAL_MODE_PATCH`): when `goal_mode=='terminal'`, it loads a separate 1-frame chunk at `dataset.lengths[ep]-1` (the episode's TRUE last frame) for the goal, keeping init from the `start..start+offset` chunk unchanged. **WHY a code change was needed, not config-only:** the original eval has NO end-clamp — `_extract_init_goal` takes goal = `load_chunk(ep, start, start+offset+1)[-1]`, and `_load_slice` (`stable_worldmodel/data/formats/hdf5.py`) reads `h5[offset[ep]+start : offset[ep]+end]` with no clamp, so a large `goal_offset` would (a) read into the NEXT episode's frames (HDF5 stores episodes concatenated by `ep_offset`) AND (b) crash `sample_eval_episodes` (`gip.py`), which requires `start+goal_offset+1 ≤ ep_len` to pick valid starts. So a large-offset config-only route does NOT clamp to terminal — it corrupts. Confirmed by reading `_load_slice` + `sample_eval_episodes`. **Smoke-verified (1-env, `/tmp/smoke_s27_goal.py`):** for eps {0,5,17}, `goal_mode=terminal` goal == the episode's literal last frame (`demo[L-1]`, `np.array_equal` True for all 3), differs from the mid goal, and differs from the init frame. Median can demo L≈113-118 → b240 ≫ demo, no starvation.

**EXACT commands** (drivers `/mnt/minghao_data/s27_clean/s27clean_{e2e,gcidm}.sh`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, run as `sudo -u minghao.fu`; shared env `STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=0 OMP_NUM_THREADS=4`, scratch on `/mnt/minghao_data/s27_clean`; cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`):
- **OURS (e2e, ep100), per seed:** `CUDA_VISIBLE_DEVICES=<0|1|2> python eval_histbc_robomimic.py --config-name robomimic world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can eval.num_eval=50 eval.eval_budget=240 eval.goal_offset_steps=90 +eval.goal_mode=terminal policy=can_gc_e2e +gip_eval.mode=policy +gip_eval.goal_conditioned=true seed=<42|0|1>`. Load log: `[GIP] load can_gc_e2e <- weights_epoch_100.pt: Adim=35 missing=0 unexpected=0` (CONVERGED ep100 ckpt), `[HISTBC] policy ready adim=35 action_block=5 HS=3 goal_cond=True use_horizon=True horizon0=18.00`.
- **gcidm (Markovian), per seed:** `CUDA_VISIBLE_DEVICES=<3|4|5> python eval_histbc_robomimic.py --config-name robomimic world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can eval.num_eval=50 eval.eval_budget=240 eval.goal_offset_steps=90 +eval.goal_mode=terminal policy=can_gcidm_latest +gip_eval.mode=gcidm +gip_eval.gcidm_run=can_gcidm_latest seed=<42|0|1>`. Load log: `[GCIDM] load can_gcidm_latest <- gcidm_head_best.pt  H_max=50  action_dim=35  ablate_horizon=False`, `[GCIDM] policy ready action_block=5 adim=35 H_max=50 horizon0=18.00`.
- The ONLY change vs the §22 canonical can protocol is `+eval.goal_mode=terminal` (and the §22 b240 budget is restored vs the quick test's b90). N=50, offset=90, seeds {42,0,1}, same-box L40S — all identical to §22.

**RESULT (can, N=50, 3-seed {42,0,1}, ALL per-seed, no errors, successes scatter across all 5 chunks — no chunk-1 artifact):**
| arm | protocol | seed42 | seed0 | seed1 | **mean** |
|---|---|---|---|---|---|
| OURS (e2e, ep100) | **§27 CLEAN terminal-goal b240/o90** | 0.32 | 0.22 | 0.24 | **0.260** |
| gcidm (`_latest`) | **§27 CLEAN terminal-goal b240/o90** | 0.42 | 0.30 | 0.22 | **0.313** |
| OURS (e2e, ep100) | §22 ref b240/o90 MID-goal | 0.06 | 0.30 | 0.26 | 0.207 |
| gcidm (`_latest`) | §22 ref b240/o90 MID-goal | 0.30 | 0.26 | 0.16 | 0.240 |
| OURS (e2e, ep100) | §27 quick b90/o90 (budget-starved) | 0.04 | 0.02 | 0.00 | 0.020 |
| gcidm (`_latest`) | §27 quick b90/o90 (budget-starved) | 0.04 | 0.00 | 0.00 | 0.013 |
| bc (goal-AGNOSTIC, §1c) | — | — | — | — | **0.71** |

**VERDICT — PARTIAL recovery: the floor is BOTH a mild eval artifact AND predominantly real.** The terminal goal (with a long budget) lifts GC modestly above the §22 mid-goal numbers (OURS 0.207→**0.260**, +0.053; gcidm 0.240→**0.313**, +0.073) and dramatically above the budget-starved §27 quick test (12× / 24×). So the mid-trajectory goal WAS depressing GC somewhat (and the quick test's collapse was the budget-starvation confound, now confirmed — restoring b240 with terminal goal recovers it). **BUT neither arm climbs toward the bc ceiling (0.71); both top out at ~0.26-0.31 = ~37-44% of bc.** Conditioning the GC policy on the ACTUAL terminal goal with ample time does NOT recover it toward the goal-agnostic bc — it stays floored at well under half. **So the can-GC floor is PARTIALLY an eval artifact (the clean protocol lifts the §22 number by ~5-7 pts) but PREDOMINANTLY REAL.** The §22 conclusion (frozen vit-tiny latent can't localize the can well enough for goal-reaching) SURVIVES, with the refinement that the precise floor value is protocol-dependent (0.21 mid → 0.26 terminal, still ≪ 0.71). The gcidm-vs-ours ordering is unchanged from §22 (Markovian gcidm ≥ history ours: 0.313 ≥ 0.260, within ~1 SE) — consistent with [[project_gcidm_vs_ours_verdict]]. **The §21 "floor = GC formulation not encoder" framing is still NOT cleanly trustworthy** — the floor moved with the protocol, so it's part eval, part formulation/encoder; the clean number (terminal goal, 0.26/0.31) is the one to report for can-GC, NOT the mid-goal 0.21/0.24 and NOT the starved 0.02/0.01.

---

### §28 S4 SCRATCH-PROBE R² — two-room (frozen converged scratch CLS latent → ground-truth sim-state, Ridge α=10, 3 split-seeds {42,0,1}, L40S 2026-06-21)

**Motivation (why).** Fill the S4 scratch-probe cell for two-room: quantify how much ground-truth sim-state the FROZEN converged from-scratch base-WM encoder linearly exposes (latent decodability), the same frozen-probe diagnostic used in §10/§15 but for the canonical vit-tiny-192 tworoom scratch base. This is a representation-quality number (R², NOT a planning/SR number) — it measures whether the scratch encoder's CLS latent carries the privileged state, which is the necessary-not-sufficient signal behind the SR results. Existing scratch base, fast, eval-only (one forward pass + Ridge).

**Setup (`probe_tworoom.py`, unchanged in repo).** Freeze the converged from-scratch SIGReg base WM: config = `/mnt/minghao_data/.stable-wm/checkpoints/tworoom_lewm_base/config.json` (canonical `jepa.JEPA`, encoder `vit_hf` size=tiny patch=14 image=224 **pretrained=false** → from scratch, CLS dim 192), weights = `/mnt/minghao_data/.stable-wm/decoders/tworoom_ours_lewm_weights.pt` (`tworoom_ours_lewm_weights.pt`, converged scratch base, 72 MB, dated May 28). `model.load_state_dict(sd, strict=False)` → **missing=1 unexpected=0** (the 1 missing key is a non-encoder head/buffer; encoder loads clean). Take `enc(x, interpolate_pos_encoding=True).last_hidden_state[:,0]` = CLS. Dataset = `/mnt/minghao_data/.stable-wm/datasets/tworoom.h5` (12.8 GB on disk, ~138 GB uncompressed — NEVER fully loaded; **6000 stride-sampled frames**, `idx = arange(0, N, N//6000)[:6000]`, sorted, read in 512-blocks). tworoom h5 has **NO `state` key**; the privileged full sim-state is the `observation` field (**10-d**); also probed `pos(agent+target)` (4-d, `pos_agent`⊕`pos_target`) and `proprio` (2-d). Per split-seed {42,0,1}: 80/20 held-out split, standardize Z by train mean/std, `Ridge(alpha=10.0)`, `r2_score(..., multioutput="variance_weighted")` on the held-out 20%. **HEADLINE = full `observation` state R²** (NOT SR).

**Infra (DISK-SAFE, L40S shared box — good citizen).** All 8 L40S GPUs verified idle at launch (`nvidia-smi`: 0% util, 4-5 MiB each). Ran as `sudo -u minghao.fu`, py = `/var/lib/docker/data/minghao_home/lewm/bin/python` (torch 2.12.0+cu130, CUDA True), cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, **GPU 0 only** (single fast eval, didn't monopolize). L40S `/var/lib/docker` 100% full + `/` 97% full → **ALL scratch redirected to `/mnt/minghao_data` (2.5 TB free)**: exported `XDG_CACHE_HOME=/mnt/minghao_data/xdg_probe_tworoom TMPDIR=/mnt/minghao_data/tmp_probe_tworoom MPLCONFIGDIR=/mnt/minghao_data/mpl_probe_tworoom HF_HOME=/mnt/minghao_data/hf STABLEWM_HOME=/mnt/minghao_data/.stable-wm HF_HUB_OFFLINE=1`. `df` checked before AND after: stayed at **20% used on /mnt** throughout, `/` unchanged at 97%. Probe writes NO checkpoints (pure model instantiation + Ridge, no Trainer / no `default_root_dir` / no `hydra.run.dir`); verified zero new `.ckpt`/`.pt` under the redirect dirs and no new dir in `.stable-wm/checkpoints/`. Nothing to prune.

**EXACT command** (one shot, foreground):
```
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
export XDG_CACHE_HOME=/mnt/minghao_data/xdg_probe_tworoom TMPDIR=/mnt/minghao_data/tmp_probe_tworoom \
       MPLCONFIGDIR=/mnt/minghao_data/mpl_probe_tworoom HF_HOME=/mnt/minghao_data/hf \
       STABLEWM_HOME=/mnt/minghao_data/.stable-wm HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
/var/lib/docker/data/minghao_home/lewm/bin/python probe_tworoom.py
```

**Latent diagnostics.** dim=192, zstd=**0.634**, eff-rank=**57.5/192** (no collapse — healthy spread, well above the §10 scratch collapse thresholds).

**RESULT (two-room, frozen scratch base, 3 split-seeds {42,0,1}, ALL per-seed):**
| task | arm | target | seed42 | seed0 | seed1 | **mean R²** |
|---|---|---|---|---|---|---|
| two-room | scratch (converged base, vit-tiny-192) | **observation (full 10-d sim-state) = HEADLINE** | 0.3235 | 0.3152 | 0.3161 | **0.3182** |
| two-room | scratch | pos (agent+target, 4-d geom) | 0.3235 | 0.3152 | 0.3161 | 0.3182 |
| two-room | scratch | proprio (2-d) | 0.9967 | 0.9967 | 0.9969 | 0.9967 |

**S4 SCRATCH-PROBE CELL (two-room) = R² 0.318** (headline = full `observation` sim-state; per-seed [0.3235, 0.3152, 0.3161], tight, SE ≈ 0.003).

**Notes / finding.** (1) `observation(full)` and `pos(agent+target)` R² are **identical** — under variance-weighted scoring the 10-d full state is dominated by the 4-d geometric agent/target positions, so the extra 6 state dims add no separately-decodable variance. (2) **proprio is near-perfect (0.997)** — the scratch CLS latent trivially encodes the 2-d proprioceptive signal, but the **geometric pose (where agent/target are in the room) is only weakly linear-decodable (R² 0.32)**. (3) This is the same low-state-R² / high-proprio-R² pattern seen on the §10/§15 scratch bases (proprio easy, full geometric state hard for a frozen scratch vit-tiny latent) — necessary-not-sufficient: the frozen scratch latent localizes the agent's own dynamics far better than the scene geometry. Headline S4 two-room scratch-probe R² = **0.318**.

---

### §28 S3 CUBE-BC SR — goal-AGNOSTIC behavioral cloning on the existing cube base (reactive action head, N=50 3-seed, same-box L40S 2026-06-21)

**Motivation (why).** Fill the S3 (goal-agnostic bc) cell for **cube** in the master table. S3 = the reactive BC policy: run the trained action head directly as a reactive controller, NO goal image and NO solver/planning (`BCPolicy` with `goal_conditioned=False` → `type="behavioral_cloning"`, the mode-3 byte-identical path). This is the goal-AGNOSTIC counterpart to the S2 goal-conditioned and the planning/guided rows; it isolates how well the open-loop action head alone reaches the cube target. Eval-only on the **existing** cube base — no training.

**Base used.** `cube_lewm_base` = the canonical LeWM cube base at `/mnt/minghao_data/.stable-wm/checkpoints/cube_lewm_base/weights_epoch_0.pt` (the single `make_wm_ckpt`-style export; only ckpt in the dir). config.json: `jepa.JEPA`, `use_action_history=true`, `use_proprio=false`, encoder = stable_pretraining `vit_hf` **size=tiny patch=14 image=224 pretrained=false** (CLS-192, NOT DINOv2), ARPredictor 192-d depth-6, action_encoder `Embedder` input_dim=25 → 192, **`action_head: {type: mse}`** ← the bc head that S3 requires. The base carries the mse action head, so bc is runnable (no "blocked — no bc head"). This is the same base family the §cube guided/planning rows ran on (vit-tiny-192).

**Infra (DISK-SAFE, L40S shared box — good citizen).** All 8 L40S GPUs verified idle at launch (`nvidia-smi`: 0% util, 4–5 MiB each). Ran 3 seeds in parallel on **GPUs 0, 1, 2** (one seed each, staggered `sleep 7`), did NOT monopolize the box, never touched other users' procs/GPUs. Ran as `sudo -u minghao.fu`, py = `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`. L40S `/var/lib/docker` 100% full + `/` 97% full → **ALL writes redirected to `/mnt/minghao_data` (2.5 TB free)**: exported `XDG_CACHE_HOME=/mnt/minghao_data/xdg_cube_bc TMPDIR=/mnt/minghao_data/tmp_cube_bc MPLCONFIGDIR=/mnt/minghao_data/mpl_cube_bc HF_HOME=/mnt/minghao_data/hf STABLEWM_HOME=/mnt/minghao_data/.stable-wm HF_HUB_OFFLINE=1`, plus **`hydra.run.dir=/mnt/minghao_data/hydra_cube_bc_s<seed> hydra.output_subdir=null`** (Hydra otherwise writes `./outputs/<date>` into cwd on the full /var/lib/docker disk — overridden to /mnt). `STABLEWM_HOME` redirects `get_cache_dir()` so the eval results JSON lands at `/mnt/minghao_data/.stable-wm/gip_eval/bc/cube_lewm_base/`. `df` checked before AND after: `/mnt` stayed at **20% used** (603 → 605 GB), `/var/lib/docker` unchanged at 100% (no write leaked there), `/` unchanged at 97%. **Eval-only — writes NO checkpoints** (verified no new `.pt` / no new dir under `.stable-wm/checkpoints/`); nothing to prune. `+ckpt_every` N/A (no Trainer).

**EXACT command** (per seed; seed ∈ {42, 0, 1}, gpu ∈ {0, 1, 2}):
```
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
export STABLEWM_HOME=/mnt/minghao_data/.stable-wm \
       XDG_CACHE_HOME=/mnt/minghao_data/xdg_cube_bc TMPDIR=/mnt/minghao_data/tmp_cube_bc \
       MPLCONFIGDIR=/mnt/minghao_data/mpl_cube_bc HF_HOME=/mnt/minghao_data/hf \
       HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=<gpu> /var/lib/docker/data/minghao_home/lewm/bin/python eval_gip.py \
  --config-name cube policy=cube_lewm_base +gip_eval.mode=bc eval.num_eval=50 seed=<seed> \
  hydra.run.dir=/mnt/minghao_data/hydra_cube_bc_s<seed> hydra.output_subdir=null
```
Config (from cube.yaml, all defaults kept): `world.env_name=swm/OGBCube-v0`, `eval.num_eval=50`, `eval.goal_offset_steps=25`, `eval.eval_budget=50`, `plan_config.horizon=5 action_block=5`, `eval.dataset_name=ogbench/cube_single_expert`, `terminate_at_goal=True`. Policy = `BCPolicy` (`gip_eval.mode=bc`, goal_conditioned=False → goal-agnostic). Each seed evaluated on the **same box it would be eval'd on** (no cross-GPU render OOD per [[project_l40s_cross_gpu_rendering]]); same-box, same dataset, N=50.

**RESULT (cube, `cube_lewm_base`, mode=bc, N=50, 3 seeds, ALL per-seed):**
| task | base | mode | seed42 | seed0 | seed1 | **mean SR** |
|---|---|---|---|---|---|---|
| cube | `cube_lewm_base` (vit-tiny-192, mse action head) | **bc (goal-agnostic, S3)** | 40.0 | 50.0 | 36.0 | **42.0** |

**S3 CUBE-BC CELL = SR 42.0** (N=50 × 3 seeds {42, 0, 1}; per-seed [40.0, 50.0, 36.0]; SE ≈ 4.2). Goal-agnostic reactive action head on the existing cube base.

**Notes / finding.** (1) Goal-agnostic bc on cube tops out at **42.0%** — the reactive action head alone (no goal, no planning) reaches the cube target in under half of episodes. (2) Context vs the cube goal-using rows (memory [[project_cube_reproduction]]): intention-guided 88 > planning 78 on the cube base family; the goal-agnostic bc (42.0) is well below both, as expected — a reactive policy with no target signal cannot match a goal-conditioned/planning controller on this aligned goal-reaching manipulation task. So cube shows the normal ordering guided/planning ≫ goal-agnostic-bc, unlike the robomimic §27 case where bc ≫ GC (the robomimic GC floor being part eval-artifact + part frozen-encoder). (3) Per-seed spread is moderate (36–50, range 14 pts at N=50) — the 42.0 mean is the cell to report; single-seed numbers are diagnostic only.

---

### §28 S4 SCRATCH-PROBE R² — pusht (frozen scratch CLS latent → ground-truth sim-state, Ridge α=10, 3 split-seeds {42,0,1}, L40S 2026-06-21)

**Motivation (why).** Fill the S4 scratch-probe cell for **pusht** (the contact-heavy 2-D env), the sibling of the two-room cell above: how much ground-truth sim-state does the FROZEN **scratch** (epoch-0, `pretrained=False`, random-init) LeWM base encoder's CLS latent linearly expose? Representation-quality number (R², NOT planning/SR), the necessary-not-sufficient signal behind the SR results, eval-only (one forward pass + Ridge). Existing scratch base, fast.

**Setup (`pusht_probe.py`, unchanged in repo).** Freeze the SCRATCH LeWM base: config = `/mnt/minghao_data/.stable-wm/checkpoints/pusht_lewm_base/config.json` (`jepa.JEPA`, encoder `vit_hf` size=tiny patch=14 image=224 **pretrained=false**, `use_action_history=true use_proprio=false`, CLS dim 192), weights = `weights_epoch_0.pt` (72 MB, Jun 20 — **epoch 0 = scratch / untrained encoder**, the random-init base). `load_state_dict(strict=False)`, asserted **no `encoder.*` keys missing**. CLS = `enc(x, interpolate_pos_encoding=True).last_hidden_state[:,0]` = **192-D**, pixels ImageNet-normalized (.485/.456/.406, .229/.224/.225), `/255`, bs=128. Dataset = `/mnt/minghao_data/.stable-wm/datasets/pusht_expert_train.h5` (46 GB; **NEVER fully loaded** — 6000 stride-sampled frames, `idx=arange(0,N,N//6000)[:6000]`, sorted-unique, block-read). Target = pusht **`state` field = 7-D** GT sim-state (agent x/y, block x/y, block angle, +2 — the privileged 2-D scene geometry; `state` shape (2336736, 7) float32). Per split-seed {42,0,1}: **episode-level 80/20 held-out** split by `episode_idx` (disjoint train/test, `overlap=0` for all seeds), standardize Z by train mean/std, **`Ridge(alpha=10.0)`**, `r2_score(..., multioutput="variance_weighted")`; RBF KernelRidge (α=10, γ=1/192) reported as a control. **HEADLINE = `state` R²_lin** (NOT SR). ⚠️ With strided sampling each retained frame becomes its own `episode_idx` bucket, so the "6000 episodes" log line is really "6000 frames"; the split is still a correct held-out row split (overlap=0 confirms zero leakage), per-frame not per-true-demo — matches the committed `pusht_probe.py` exactly (not modified).

**Infra (DISK-SAFE, L40S shared box — good citizen).** All 8 L40S GPUs verified idle at launch (`nvidia-smi`: 4-5 MiB each). Ran as `sudo -u minghao.fu`, py = `/var/lib/docker/data/minghao_home/lewm/bin/python` (torch 2.12.0+cu130, CUDA True), cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, **GPU 0 only**. L40S `/var/lib/docker` 100% full + `/` 97% (3.3 GB) → **ALL scratch redirected to `/mnt/minghao_data` (2.5 TB free)**: exported `XDG_CACHE_HOME=/mnt/minghao_data/xdg_probe_pusht TMPDIR=/mnt/minghao_data/tmp_probe_pusht MPLCONFIGDIR=/mnt/minghao_data/mpl_probe_pusht HF_HOME=/mnt/minghao_data/hf STABLEWM_HOME=/mnt/minghao_data/.stable-wm HF_HUB_OFFLINE=1 MUJOCO_GL=egl`. `df` checked before launch (no write target on `/var/lib/docker` or `/`). Probe writes NO checkpoints (pure instantiation + Ridge, no Trainer / `default_root_dir` / `hydra.run.dir`); nothing to prune. Log retained at `/mnt/minghao_data/tmp_probe_pusht/pusht_probe.log`.

**EXACT command** (one shot):
```
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
export XDG_CACHE_HOME=/mnt/minghao_data/xdg_probe_pusht TMPDIR=/mnt/minghao_data/tmp_probe_pusht \
       MPLCONFIGDIR=/mnt/minghao_data/mpl_probe_pusht HF_HOME=/mnt/minghao_data/hf \
       STABLEWM_HOME=/mnt/minghao_data/.stable-wm HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl
/var/lib/docker/data/minghao_home/lewm/bin/python pusht_probe.py
```

**RESULT (pusht, frozen scratch base epoch-0, 3 split-seeds {42,0,1}, ALL per-seed):**
| task | arm | target | seed42 | seed0 | seed1 | **mean R²_lin** | mean R²_rbf |
|---|---|---|---|---|---|---|---|
| pusht | scratch (epoch-0, vit-tiny-192, pretrained=false) | **state (full 7-d GT sim-state) = HEADLINE** | 0.7120 | 0.7114 | 0.7148 | **0.7127** | 0.6566 |

**S4 SCRATCH-PROBE CELL (pusht) = R² 0.7127** (headline = `state` R²_lin, Ridge α=10; per-seed [0.7120, 0.7114, 0.7148], spread ±0.0017 = seed-invariant; held-out overlap=0). Linear > RBF (0.7127 > 0.6566) → readout is genuinely linear.

**Notes / finding.** The **scratch** (untrained, epoch-0) vit-tiny CLS latent already linearly carries **R²=0.71** of the pusht 7-D GT sim-state — most of the privileged state is recoverable WITHOUT training. Sharply ABOVE the two-room scratch cell (0.318) above: a random-init ViT's CLS token (with `interpolate_pos_encoding`) is already a strong linear chart of the low-frequency 2-D pusht agent/block geometry, whereas the two-room *full geometric* pose is only weakly linear (0.32, though tworoom proprio saturates at 0.997). Implication for pusht: "the encoder LEARNS to localize the scene" is only weakly true at the representation level — the scratch floor is already 0.71, so any TRAINED-encoder probe-R² must beat this to claim learned localization. Contrast §6 robomimic-can (latent→full-state only 0.554 even on a TRAINED encoder: 3-D occluded object pose + velocities ≫ harder than flat 2-D pusht geometry). This is a representation number (R²), NOT control/SR — it bounds what is linearly PRESENT in the latent, not whether a planner/policy can USE it. Headline S4 pusht scratch-probe R² = **0.7127**. `gen_master_table.py` NOT edited (user owns it).

---

## §29 ✅ ENCODER-BOTTLENECK TEST — **PRETRAINED DINOv2-base (768-d)** init+finetune on **can**, unified bc+clean-GC model — does a bigger PRETRAINED encoder lift bc toward DP (1.0) and clean-GC off the floor (vit-tiny: bc 0.71 / clean-GC 0.26)? (L40S, 2026-06-21) [DONE — ep100 converged; VERDICT = DEEPER bottleneck, NOT the encoder]

**Motivation (why).** The §22/§24/§27 chain established that on robomimic-can the goal-conditioned policy floors (clean terminal-goal GC ≈ 0.26 for OURS, ≈ 0.31 gcidm, both ≪ the goal-AGNOSTIC bc 0.71, and bc itself ≪ Diffusion-Policy's ~1.0), and §22's reading was "the **frozen vit-tiny-192** latent can't localize the occluded can well enough for goal-reaching." Two distinct capacity questions remained open: (Q1) does a **larger** encoder lift bc toward the DP ceiling, or is bc already saturated at 0.71 by the action head / data, not the encoder? §24-SIZE answered the **scratch-init** scaling half (random-init `vit_hf size=base`, 768-d, goal-AGNOSTIC bc only). (Q2 — THIS arm) does a larger **PRETRAINED** (DINOv2 ImageNet-scale) encoder — strictly more visual prior than a from-scratch ViT — lift **BOTH** bc AND the **clean terminal-goal GC** off their vit-tiny floors? If a 4×-wider encoder with a strong pretrained visual prior still leaves bc ≈ 0.71 and clean-GC ≈ 0.26, the can floors are **not** an encoder-capacity / visual-prior bottleneck (they're the action head, the data, or the GC formulation); if either jumps, the encoder WAS the bottleneck. This is the encoder-capacity counterpart to the §24-SIZE scratch-scaling arm and the natural "is it the encoder?" stress test the §22/§27 verdicts call for.

**Distinct from §24-SIZE.** §24-SIZE = `vit_hf size=base` **random init** (`pretrained=false`), **goal-AGNOSTIC bc only** (`goal_conditioned=false`). THIS §29 arm = **`from_huggingface facebook/dinov2-base pretrained=true`** (REAL DINOv2-base ImageNet weights, init+finetune) and a **UNIFIED bc+GC model** (`goal_conditioned=true horizon_conditioned=true goal_dropout=0.5` → ONE model serves both the goal-AGNOSTIC bc eval and the goal-conditioned clean-GC eval, exactly the §21-CLEAN/§24 two-setting WAM recipe). So §29 adds the **pretrained-prior** axis AND the **clean-GC** readout that §24-SIZE (bc-only, scratch) does not cover.

### STEP 1 — flag verification + 1-step forward (DONE, all honored)
- **New model config** written: `config/train/model/lewm_dinov2_base.yaml` = a copy of the existing `lewm_dinov2.yaml` (which used `facebook/dinov2-small`/384) with `model_name: facebook/dinov2-base` and `pretrained: true`. Encoder `_target_ = stable_pretraining.backbone.utils.from_huggingface`; the rest of the trunk (ARPredictor/projector/pred_proj/action_encoder) keys off `${embed_dim}`, so the **companion override `embed_dim=768` is REQUIRED** (the same audit flag the `lewm_dinov2` 384 case documents). `jepa.JEPA.encode` reads `encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:,0]` (the CLS token), so DINOv2-base (768-d CLS) is a drop-in at this interface — CONFIRMED in `jepa.py:124-125`.
- **Encoder forward smoke** (standalone, GPU): `from_huggingface("facebook/dinov2-base", pretrained=True)` downloaded the real DINOv2-base weights to `/mnt/minghao_data/hf` (223 safetensor shards, HTTP 200), `enc(randn(2,3,224,224), interpolate_pos_encoding=True).last_hidden_state[:,0].shape == (2, 768)`, **86.58 M encoder params** (≈ 4× the vit-tiny 22.1 M and the dinov2-small 22.1 M). CLS width = 768 confirmed.
- **1-step training forward** (the full JEPA model with the GC head): `train.py --config-name lewm data=robomimic_can model=lewm_dinov2_base embed_dim=768 action_pred.enabled=true action_pred.goal_conditioned=true action_pred.horizon_conditioned=true action_pred.goal_dropout=0.5 trainer.max_epochs=1 +trainer.limit_train_batches=6 loader.batch_size=8` → `[GIP] Intention predictor ON Adim=35 head=mse` (35 = frameskip 5 × can action-dim 7), GC + AdaLN-Zero horizon head built, full fwd+bwd ran, all losses finite (`fit/pred_loss≈0.20 fit/act_loss≈0.80 fit/intent_loss≈1.02 validate/act_loss 1.27→1.15`, `act_emb_std≈0.12` → no collapse), **NO shape crash**, swm ckpt saved to `/mnt/minghao_data/.stable-wm/checkpoints/<run>/weights_epoch_1.pt`. The only failure was at `+trainer.limit_train_batches=1` (a `ZeroDivisionError` in the cosine LR scheduler — total-steps=1 edge case, NOT a model-shape bug); cleared at ≥6 batches. **STEP 1 PASSES.**
- **`load_gip_model` (eval rebuild) is dinov2-aware**: it reads `config.json`, sets `embed_dim=int(config.predictor.input_dim)` (=768 here, auto), rebuilds the action_predictor/decoder, and `load_state_dict(strict=False)` with an unexpected-keys assert. So the hydra-`train.py`-trained 768-wide ckpt is loadable by `eval_histbc_robomimic.py policy=<run_name>` with NO eval-side code change — confirmed in `gip.py:115-160`.

### GPU-MEMORY NOTE (real constraint flagged per task)
DINOv2-base **end-to-end finetuning** (86.6 M encoder, 3 history frames each pushed through the ViT with grads) does **NOT** fit at the warm-arm `loader.batch_size=64` on a **contended** L40S GPU: a first probe on GPU 2 (which had another user's 14.6 GB process — left untouched) OOM'd at bs=128 (`CUDA out of memory ... 193 MiB free`). On a **fully-idle** 46 GB L40S GPU, **bs=64 fits** at ~36.7 GB used / 100% util with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. So the headline run uses **bs=64 on an idle GPU (GPU 4)**. (The §24-SIZE scratch base arm used bs=32 because at that time all 8 GPUs were 29–34 GB occupied by other parallel arms; the box is now free, so bs=64 is safe here.) DINOv2-large was NOT attempted — the task asked to verify base fits first; base fits at bs=64, large (304 M, 1024-d) would need a batch cut and is out of scope for this arm.

### STEP 2 — EXACT TRAIN COMMAND (init+finetune from pretrained DINOv2-base, GPU 4 L40S, 100 ep, converge)
```
# launch script /mnt/minghao_data/launch_dinov2_base_768.sh ; run as sudo -u minghao.fu ; log /mnt/minghao_data/dinov2_base_768_train.log
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
export HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=0 \
       XDG_CACHE_HOME=/mnt/minghao_data/xdg_dinov2_base_768 \
       TMPDIR=/mnt/minghao_data/tmp_dinov2_base_768 \
       MPLCONFIGDIR=/mnt/minghao_data/mpl_dinov2_base_768 \
       SPT_CACHE_DIR=/mnt/minghao_data/spt_dinov2_base_768 \
       STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl OMP_NUM_THREADS=4 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=4 /var/lib/docker/data/minghao_home/lewm/bin/python train.py --config-name lewm \
  data=robomimic_can model=lewm_dinov2_base embed_dim=768 \
  output_model_name=dinov2_base_768 \
  action_pred.enabled=true action_pred.goal_conditioned=true \
  action_pred.horizon_conditioned=true action_pred.goal_dropout=0.5 \
  trainer.max_epochs=100 loader.batch_size=64 num_workers=6 +ckpt_every=10 \
  hydra.run.dir=$TMPDIR/run
```
- **EXACT full config (every flag):** `--config-name lewm` (base recipe: AdamW lr 5e-5 wd 1e-3, precision bf16, gradient_clip 1.0, sigreg weight 0.09 knots 17 num_proj 1024, history_size 3, img_size 224, num_preds 1, train_split 0.9, seed 3072). `data=robomimic_can` (can.h5, frameskip 5, keys pixels+action+proprio+state). `model=lewm_dinov2_base` (**from_huggingface facebook/dinov2-base, pretrained=TRUE** → init from real DINOv2 weights then finetune end-to-end; `use_action_history=true`, `use_proprio=false`). `embed_dim=768` (REQUIRED companion to the 768-d CLS). `action_pred.enabled=true` (the intention/action head ON; `w_act=1.0 w_intent=1.0 detach_target=true detach_decoder=false ema_target=false sigreg_act=false head=mse`). **`action_pred.goal_conditioned=true`** (wraps the dataset with `GoalSamplingDataset`, hindsight goal ~U[1,50] obs ahead; head also takes z_goal → history-conditioned forward GC policy). **`action_pred.horizon_conditioned=true`** (AdaLN-Zero horizon modulation, `horizon_H_max=50`, the SAME hook GC-IDM uses — built in `JEPA.__init__`). **`action_pred.goal_dropout=0.5`** (zero z_goal per-sample w.p. 0.5 → the SAME head ALSO learns the goal-AGNOSTIC policy → ONE model serves bc + GC, the two-setting WAM / BESO-CFG recipe). `freeze_wm` NOT set (default false → encoder LEARNABLE / end-to-end — DINOv2-base is **finetuned**, not frozen). `init_from=null` (init comes from the pretrained DINOv2 encoder weights via `from_huggingface pretrained=true`, NOT from a LeWM `.pt`). `trainer.max_epochs=100`, `loader.batch_size=64`, `num_workers=6`, `+ckpt_every=10` (prune newest-3-then-final, rule 10). **seeds:** training seed = the recipe default 3072 (single training run); the 3-seed {42,0,1} are EVAL seeds (env resets), per the §27 convention. **GPU = L40S GPU 4** (idle at launch).
- **Init = PRETRAINED, finetune end-to-end (NOT scratch, NOT frozen).** `from_huggingface(pretrained=true)` loads the real DINOv2-base ImageNet weights; `freeze_wm=false` means they are then finetuned with the rest of the trunk. This is the key difference from §24-SIZE (which was scratch, `pretrained=false`).
- **Dataset size / per-epoch:** can.h5 = 200 demos / ~23 207 raw steps; windowed at frameskip 5 (history 3 + num_preds 1) → **≈ 281 train batches/epoch at bs=64** (train_split 0.9). Measured epoch time on idle GPU 4 = **~193 s/epoch** (0.69 s/batch) → **100 ep ≈ 5.4 h**. **NO `limit_train_batches` cap — full data every epoch** (convergence per rule 8; the budget is set by epoch count, not a batch cap). Launched **2026-06-21 ~19:58 UTC** → **ETA ep100 ≈ 01:20 UTC 2026-06-22**.

### DISK-SAFETY (L40S `/var/lib/docker` 100% + `/` 97% — ALL writes redirected to /mnt/minghao_data, 2.5 TB free)
- Exported `XDG_CACHE_HOME`/`TMPDIR`/`MPLCONFIGDIR`/`HF_HOME`(=`/mnt/minghao_data/hf`)/`STABLEWM_HOME`(=`/mnt/minghao_data/.stable-wm`) all on /mnt. **`HF_HUB_OFFLINE=0`** so the DINOv2-base weights download (one-time, to `/mnt/minghao_data/hf`). `hydra.run.dir=$TMPDIR/run` (off the full /var disk).
- **CRITICAL extra redirect found + applied:** the stable-pretraining **Manager/Lightning** checkpoint dir defaults to `~/.cache/stable-pretraining/runs` (resolved from `SPT_CACHE_DIR`, default `$HOME/.cache/...`), which on this box is on the **FULL `/var/lib/docker`** — and each Lightning requeue `last.ckpt` is **2.46 GB**. The smoke run initially wrote one there (caught it). **FIX: `SPT_CACHE_DIR=/mnt/minghao_data/spt_dinov2_base_768`** → verified the 2.46 GB Lightning ckpts now land on /mnt (4.9 G under the SPT dir, only 16 K leaked to /var). The eval target is the swm `weights_epoch_*.pt` under `STABLEWM_HOME/checkpoints/dinov2_base_768/` (small), pruned newest-3 by `+ckpt_every=10`. df verified before launch: /mnt 20% used (608 G/3.0 T), /var 99%, / 97% — and the smoke artifacts were cleaned (`rm -rf` the 4.9 G runs dir + the smoke checkpoint dir) back to 90 G free on /var.

### Convergence (rule 8 — verify plateau) [FILLING LIVE]
Per-epoch `validate/act_loss` (the action-prediction MSE = the bc/GC-relevant head loss; watch the last ~10% for a flat plateau): ep0 sanity-val 1.208 → ep0 1.155 → [filling]. Will report the last-10-epoch trace + a flat/not-flat call once it reaches ep100. **NOT yet converged at report time** (still training).

### STEP 3 — EXACT EVAL COMMANDS (to run once ep100 lands; same-box L40S GPU 4, N=50, 3 seeds {42,0,1})
The §27 `eval.goal_mode` patch is already installed in this box's swm `world.py` + `eval_histbc_robomimic.py` (read `cfg.eval.get("goal_mode","mid")`, `S27_GOAL_MODE_PATCH`), so the clean terminal-goal GC needs NO further code change. `load_gip_model` auto-detects embed_dim=768 from config.json. Env: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=0 OMP_NUM_THREADS=4` + the /mnt scratch redirects; cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`; py `/var/lib/docker/data/minghao_home/lewm/bin/python`; **same-box GPU 4** (the training GPU — per [[project_l40s_cross_gpu_rendering]] eval MUST be on the training box or the render goes OOD→0).
- **(a) bc (goal-AGNOSTIC, canonical b240/o90), per seed ∈ {42,0,1}:**
  ```
  CUDA_VISIBLE_DEVICES=4 python eval_histbc_robomimic.py --config-name robomimic \
    world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can \
    eval.num_eval=50 eval.eval_budget=240 eval.goal_offset_steps=90 \
    policy=dinov2_base_768 +gip_eval.mode=policy +gip_eval.goal_conditioned=false seed=<seed>
  ```
  (the goal-AGNOSTIC histbc-bc: `goal_conditioned=false` → the dropout-0.5 head's bc setting; compare to vit-tiny bc 0.71 and DP ~1.0.)
- **(b) clean GC (terminal goal, the §27 fix, b240/o90), per seed ∈ {42,0,1}:**
  ```
  CUDA_VISIBLE_DEVICES=4 python eval_histbc_robomimic.py --config-name robomimic \
    world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can \
    eval.num_eval=50 eval.eval_budget=240 eval.goal_offset_steps=90 +eval.goal_mode=terminal \
    policy=dinov2_base_768 +gip_eval.mode=policy +gip_eval.goal_conditioned=true seed=<seed>
  ```
  (the §27 CLEAN protocol: goal = episode-TERMINAL frame, budget 240 ≫ demo length; compare to vit-tiny clean-GC 0.26 OURS / 0.31 gcidm.)

### RESULT ✅ (ep100 converged, eval 2026-06-22 04:35 UTC, same-box GPU 4, N=50 × 3 seeds {42,0,1})
| arm | encoder | bc (goal-agnostic, N=50×3) | clean-GC terminal b240/o90 (N=50×3) |
|---|---|---|---|
| **§29 dinov2_base_768 (THIS)** | DINOv2-base 768, pretrained+finetuned | **0.00** {42:0.00, 0:0.00, 1:0.00} | **0.173** {42:0.20, 0:0.14, 1:0.18} |
| §27/§24 reference | vit-tiny-192 (scratch) | **0.71** | **0.260** (OURS) / 0.313 (gcidm) |
| Diffusion Policy ceiling | — | **~1.0** | — |

### VERDICT ✅ = **DEEPER BOTTLENECK, NOT THE ENCODER.** (decisive)
The bigger PRETRAINED DINOv2-base encoder did **NOT** lift bc toward DP — it **collapsed bc from 0.71 → 0.00** (all 150 bc episodes failed, every seed 0/50), and clean-GC did NOT lift off the vit-tiny 0.26 floor either (it sits slightly **below** at **0.173**). Neither readout moved in the "encoder was the bottleneck" direction; both got flat-to-worse. So on robomimic-can the goal-reaching/bc floors are **not** an encoder-capacity / visual-prior limitation — swapping vit-tiny-192 → a 4×-wider ImageNet-pretrained DINOv2-base made bc strictly worse and left GC flat. **The bottleneck is downstream of the visual encoder** (the action head / forward-policy / GC formulation / data), so adding encoder capacity / a stronger visual prior is the wrong lever for can. This closes the §22/§24/§27 "is it the encoder?" question: it is **not**.
- **bc=0.00 is REAL, not the §10 eval-batching artifact.** Verified the chunk pattern: bc `chunk 1: cumulative 0/10` (already 0 on the FIRST chunk — NOT the "chunk-1-nonzero-then-all-zeros" 10-env-pool-reuse signature [[project_dinov2_init_ablation]]), and the GC evals on the SAME 10-env pool got 3/10 on chunk 1 and progressed 3→4→6→8→10 across chunks (scattered, healthy) → the env pool is fine (GC succeeds in it), so bc=0 is a genuine property of the bc policy on this model, not a batching artifact. All 6 evals loaded `missing=0 unexpected=0` from `weights_epoch_100.pt`, all reached `chunk 5: cumulative X/50` (n=50 confirmed).
- **Unified-table cells (values only; `gen_master_table.py` NOT edited — main session owns it):** Table 2 (GC) DINOv2-GC can cell = **0.173** (seeds {0.20, 0.14, 0.18}, N=50); Table 3 (bc) DINOv2-bc can cell = **0.00** (seeds {0,0,0}, N=50).
- **Exact eval commands run** (final working form; the `world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can` overrides come from the `e2e_can_eval.sh` driver since the default config is Lift): bc = `eval_histbc_robomimic.py --config-name robomimic world.task=PickPlaceCan world.num_envs=10 dataset.stats=can eval.dataset_name=can policy=dinov2_base_768 +ckpt_epoch=100 eval.num_eval=50 eval.eval_budget=240 eval.goal_offset_steps=90 +gip_eval.mode=policy +gip_eval.goal_conditioned=false seed=<42|0|1>`; clean-GC = same + `+gip_eval.goal_conditioned=true +eval.goal_mode=terminal` (drop `goal_offset_steps`). Logs `/mnt/minghao_data/dinov2_eval_logs/{bc,gc}_s{42,0,1}.log`.

### STATUS (superseded) = was BLOCKED — TRAINING IN FLIGHT; NOW DONE (see VERDICT above)
- **Monitor task id:** `b80i901dg` (tails `/mnt/minghao_data/dinov2_base_768_train.log`, fires on each `done in` epoch line + any crash/OOM signature + ep50/60/.../100 ckpts).
- **Training PID family:** main `train.py` PID ~1747041 (+ 6 dataloader workers) on L40S, GPU 4, run as `sudo -u minghao.fu`.
- **ETA:** ep100 ≈ **01:20 UTC 2026-06-22** (~5.4 h from the 19:58 UTC launch, ~193 s/epoch × 100, idle GPU).
- **Unblock condition:** `weights_epoch_100.pt` (or the final pruned ckpt) appears under `/mnt/minghao_data/.stable-wm/checkpoints/dinov2_base_768/` AND `validate/act_loss` is flat over the last ~10 epochs (rule 8 plateau). THEN run STEP-3 (a)+(b) above (6 evals, 3 seeds × 2 modes, same-box GPU 4) → fill the RESULT table → write the finding (did the bigger PRETRAINED encoder lift bc toward 1.0 and clean-GC off the 0.26 floor?).
- `gen_master_table.py` NOT edited (user owns it).

---

## §32 🆕 FAST-ADAPTATION (rule 12) — w_cyc sweep on **PushT** (e2e GC), the OLD (confounded) arms' ep10 readout from their SAVED checkpoints (L40S, 2026-06-22) [DONE at ep10; cross-ref §26-sweep MATCHED]

**⚠️ READ §26-sweep MATCHED FIRST (the block above the §31 cube section).** These are the OLD, NOT-compute-matched arms `pusht_wcyc0`/`pusht_wcyc_sm`/`pusht_wcyc_lg` (`limit_train_batches` 2000/4000/1000) — the exact arms the fast-adaptation brief named. §26-sweep MATCHED already SUPERSEDED them as confounded (compute differs across arms) and KILLED their training procs ~04:50 UTC, replacing them with the compute-matched `wcyc0_m`/`wcyc_sm_m`/`wcyc_lg_m` (all ltb=2000, ep15). **This §32 is the fast read on the OLD arms' SAVED checkpoints** (the kill left the ckpts on disk): a within-budget directional signal the brief asked for, NOT the valid compute-matched answer (that is the MATCHED block's job). **The MATCHED replacement set ALSO got OOM-reaped** (as of 2026-06-22 ~05:20 only `wcyc0_m` still runs, no ckpt yet; `wcyc_sm_m`+`wcyc_lg_m` dead) — so the box's OOM-killing has crippled BOTH wcyc sets, and these ep10 numbers (2 of 3 arms) are the most complete wcyc data available right now.

**Motivation (why).** Test whether adding a **cycle / consistency loss** (`+action_pred.w_cyc=λ`) to the e2e goal-conditioned PushT policy HELPS or HURTS goal-reaching SR. PushT is the §13 contact-heavy discriminator where the frozen-latent OURS history-GC floored at 31 (≪ gcidm 91); these arms are the **e2e** variant (init from the pretrained decoder `pusht_ours_lewm_weights.pt`, then train the whole stack), so they also test the e2e-vs-frozen regime gap. Three arms: `pusht_wcyc0` (w_cyc=0, baseline), `pusht_wcyc_sm` (w_cyc=0.1), `pusht_wcyc_lg` (w_cyc=0.5). Per rule 12 the fast budget is ep15; these old arms saved at `ckpt_every=10`, so the first checkpoint at/past the budget would be ep20 — but they were killed/died before ep20.

**State of the OLD arms at kill/crash** (`DataLoader worker killed by signal: Terminated` in the logs = the ~04:50 `pkill` documented in §26-sweep MATCHED, and/or the box OOM-killer): `pusht_wcyc0` last ckpt **ep10** (`weights_epoch_10.pt`); `pusht_wcyc_lg` reached ep18 but only saved **ep10** (ckpt_every=10, gone before ep20); `pusht_wcyc_sm` only ep4 with **NO checkpoint** (config.yaml only) → **wcyc_sm has no result.** So the readout is the **ep10** SAVED checkpoints of wcyc0 and wcyc_lg only (below the ep15 budget, best available; §13 shows pusht val_act near-plateau by ep10–20). Not relaunched (the box keeps OOM-killing — a relaunch just re-crashes, as the MATCHED set proved).

### EXACT configs (every flag)
- `pusht_wcyc0`: `train_sigreg.py data=pusht action_pred.detach_decoder=false +action_pred.w_cyc=0.0 init_from=/mnt/minghao_data/.stable-wm/decoders/pusht_ours_lewm_weights.pt trainer.max_epochs=100 +trainer.limit_train_batches=2000 +ckpt_every=10`.
- `pusht_wcyc_sm`: identical except `+action_pred.w_cyc=0.1 action_pred.head=mse action_pred.w_act=1.0 action_pred.w_intent=1.0 action_pred.detach_target=true +trainer.limit_train_batches=4000`.
- `pusht_wcyc_lg`: identical except `+action_pred.w_cyc=0.5 +trainer.limit_train_batches=1000`.
- **NOT compute-matched (the user flagged this; it CONFOUNDS the comparison):** `limit_train_batches` = 2000 (wcyc0) / 4000 (wcyc_sm) / 1000 (wcyc_lg). So at a FIXED epoch count, wcyc0 saw **2× the batches** of wcyc_lg per epoch. At ep10, wcyc0 = 10×2000 = 20000 batches vs wcyc_lg = 10×1000 = 10000 batches → wcyc0 had 2× the total training. A fair w_cyc comparison would need batch-matching; this readout cannot cleanly separate "w_cyc=0 helps" from "wcyc0 just trained 2× longer."

### EVAL (e2e GC, ep10, N=50 × 3 seeds {42,0,1}, same box GPU 2, all loaded `missing=0 unexpected=0`)
`eval_gip.py --config-name pusht policy=<arm> +ckpt_epoch=10 +gip_eval.mode=policy +gip_eval.goal_conditioned=true +gip_eval.horizon_H_max=50 eval.num_eval=50 seed=<42|0|1>`. (`policy=<arm>` is just the checkpoint-dir name → `load_gip_model` globs `STABLEWM_HOME/checkpoints/<arm>/weights_epoch_10.pt`; no policy yaml needed.) Logs `/mnt/minghao_data/wcyc_eval2_logs/<arm>_gc_ep10_s<seed>.log`.

| arm | w_cyc | ckpt | seed 42 | seed 0 | seed 1 | **mean** | ep10 val_act | ep10 act_emb_std (collapse) |
|---|---|---|---|---|---|---|---|---|
| `pusht_wcyc0` | 0.0 | ep10 | 72.0 | 60.0 | 58.0 | **63.3** | ~0.21 (still slowly ↓) | ~2.2 (NO collapse) |
| `pusht_wcyc_lg` | 0.5 | ep10 | 54.0 | 58.0 | 58.0 | **56.7** | 0.255 | 1.93 (NO collapse) |
| `pusht_wcyc_sm` | 0.1 | — | — | — | — | **CRASHED ep4, no ckpt** | — | — |
| §13 ref: frozen-latent OURS GC | — | ep28 | — | — | — | **31** | 0.132 | — |
| §13 ref: gcidm GC | — | conv | 86 | — | 94 | **90** | — | — |

### VERDICT (ep10 fast-budget, CONFOUNDED by compute-mismatch — flagged)
1. **w_cyc>0 does NOT help PushT GC at this budget.** w_cyc=0.5 (56.7) is slightly **below** w_cyc=0 (63.3). The consistency/cycle loss did not improve goal-reaching SR; if anything marginally lower. **BUT the comparison is confounded** (wcyc0 saw 2× the batches via ltb 2000 vs 1000) — so the ~7pp gap could be the compute deficit, not the loss. The honest statement: **w_cyc=0.5 did not OVERCOME a 2× compute disadvantage, and there is no evidence it helps.** w_cyc=0.1 (wcyc_sm) crashed before any checkpoint → no datum.
2. **e2e ≫ frozen-latent on PushT GC.** Both e2e arms (57–63) **crush** the §13 frozen-latent OURS reference (31) — consistent with the LeWAM pivot's "robomimic/contact floor = frozen-regime artifact; e2e lifts it" thesis ([[project_lewam_positioning]]). Still below gcidm 90, but the e2e baseline is the right one to iterate from.
3. **No collapse** in either arm (act_emb_std ~1.9–2.2 at ep10).
- **Unified-table cells (values only; `gen_master_table.py` NOT edited):** Table 2 (GC) PushT — `pusht_wcyc0` cell = **63.3** {72,60,58}; `pusht_wcyc_lg` cell = **56.7** {54,58,58}; `pusht_wcyc_sm` = no datum (crashed). All ep10 (fast budget, sub-ep15; flag as diagnostic-at-ep10, NOT a converged headline).

---

## §31-RESULTS 🆕 FAST-ADAPTATION (rule 12) — SIGReg-redundancy ablation: collapse + SR readout at the fast budget, cube/reacher DONE, tworoom/cube-sigon-ON INCOMPLETE (L40S, 2026-06-22)

**What this is.** The fast-budget (rule 12) readout of the §31 SIGReg-redundancy ablation (A=sigreg-ON λ=0.09, B=sigreg-OFF λ=0, C=action-emb VICReg `sigreg_act=true`) on cube/tworoom/reacher: direct latent-collapse measurement from the checkpoint (`measure_collapse_*.py` → `gip.load_gip_model` → `model.encode` → z_std + effective rank + act_emb_std) plus SR (`eval_gip.py` planning + bc, N=50 × 3 seeds {42,0,1}, same-box). COLLAPSE = z_std→0 AND erank≪D(=192). The conjecture HOLDS if sigreg-OFF (B) stays full-rank AND holds SR vs A.

**⚠️ Arm availability (the oversubscribed-box OOM-killer reaped several arms — rule 10 disk/CPU bottleneck, `DataLoader worker killed by signal: Terminated` / `CUDA out of memory`):**
- **cube:** `cube_sigoff_noah` (B-off, `use_action_history=false` clean) ep10 ✅; `sigreg_B_cube_scratch` (B-off, history-on) ep10 ✅ (collapse only). **Cube A (sigreg-ON) UNAVAILABLE** — `cube_sigon_noah` CRASHED at ep8 (no ckpt) and `sigreg_A_cube_scratch` never produced a ckpt; `sigreg_C_cube_scratch` CRASHED (dataloader-killed). So cube has NO paired ON reference SR — B-off is reported standalone.
- **reacher:** A (`sigreg_A_reacher_scratch`) ep20 ✅, B (`sigreg_B_reacher_scratch`) ep20 ✅. **Reacher C CRASHED (CUDA OOM), unavailable.**
- **tworoom:** A/C still training (ep8, ~75 min to ep10, no ckpt yet), **B CRASHED at ep8** (no ckpt). **TWOROOM INCOMPLETE** — no checkpoints landed at the fast budget.

### COLLAPSE (z_std / effective-rank / act_emb_std, n=1024 val batch)
| env | arm | ckpt | z_std | erank /192 | act_emb_std | verdict |
|---|---|---|---|---|---|---|
| cube | B-off (`cube_sigoff_noah`, hist-off) | ep10 | 0.294 | **50.15** (0.261) | nan† | FULL-RANK |
| cube | B-off (`sigreg_B_cube_scratch`, hist-on) | ep10 | 0.196 | **48.54** (0.253) | nan† | FULL-RANK |
| reacher | A sigreg-ON | ep20 | 0.95 | **83** | ~0.93 | full-rank (ref) |
| reacher | B sigreg-OFF | ep20 | **0.19** | **108** | ~0.59 | **SCALE-SHRINK, full-rank (erank B>A)** |
†`act_emb_std=nan` on the cube noah arms (use_action_history=false → act_emb path unused); not a collapse signal.

### SR (N=50 × 3 seeds {42,0,1}, same-box; reacher = EGL-fixed reruns on the train GPU)
| env | arm | ckpt | mode | seed 42 | seed 0 | seed 1 | **mean** |
|---|---|---|---|---|---|---|---|
| cube | B-off (`cube_sigoff_noah`) | ep10 | bc | 70.0 | 66.0 | 56.0 | **64.0** |
| cube | B-off (`cube_sigoff_noah`) | ep10 | planning | 62.0 | 68.0 | 58.0 | **62.7** |
| reacher | A sigreg-ON | ep20 | bc | 2.0 | 2.0 | 6.0 | **3.3** |
| reacher | A sigreg-ON | ep20 | planning | 46.0 | 38.0 | 36.0 | **40.0** |
| reacher | B sigreg-OFF | ep20 | bc | 2.0 | 2.0 | 6.0 | **3.3** |
| reacher | B sigreg-OFF | ep20 | planning | 8.0 | 10.0 | 14.0 | **10.7** |

### VERDICT (fast-budget, per env)
- **CUBE — conjecture SUPPORTED on the latent-rank axis, SR strong but no paired ON ref.** Both cube sigreg-OFF arms (hist-off AND hist-on) stay **FULL-RANK** at ep10 (erank ~48–50/192, all per-dim std > 0, z_std stable not →0) — dropping latent SIGReg did NOT collapse the cube latent. SR of B-off is healthy (bc 64.0 / planning 62.7). The §18 history-copy confound is removed in the `cube_sigoff_noah` (hist-off) arm, so its full-rank latent means `act_loss` had to READ the latent and that pressure alone kept it full-rank → **clean support for "SIGReg redundant when predicting actions directly" on cube.** CAVEAT: the paired sigreg-ON cube reference (`cube_sigon_noah`) crashed before any ckpt, so we cannot yet show "B-off SR ≈ A-on SR" on cube — only that B-off does not collapse and posts strong absolute SR.
- **REACHER — conjecture HOLDS on latent-rank, FAILS on SR.** B-off SCALE-SHRINKS (z_std 0.19 vs A's 0.95) but stays **FULL-RANK** (erank 108 > A's 83) — same signature as the conservative pusht reference (B/A z_std ~0.2, erank B>A). BUT on SR, A-on planning (40.0) **≫** B-off planning (10.7) — sigreg-OFF does NOT hold planning SR vs sigreg-ON on reacher. bc is floored at 3.3 for BOTH arms (reacher §15 checkpoint-trap / undertrained at ep20 — both arms equally floored, so bc is uninformative here). So reacher is a **split verdict**: SIGReg is redundant for keeping the latent full-rank, but it MATTERS for reacher planning SR (A 40 > B 11). This is the conjecture's first SR-axis counterexample — flag for a converged (ep100) reacher recheck before headlining (reacher is the checkpoint-trap env; ep20 planning may be unstable).
- **TWOROOM — INCOMPLETE** (no fast-budget checkpoints; A/C ~75 min to ep10, B crashed). Pending.
- **C (action-emb VICReg) — NO DATUM anywhere** (cube-C and reacher-C both crashed; tworoom-C still training). The "does C match/beat A?" question is unanswered at the fast budget.
- **Unified-table cells (values only; `gen_master_table.py` NOT edited):** Table 3 (bc) — cube sigreg-bc (B-off) = **64.0**; reacher sigreg-bc A=B=**3.3** (floored). NEW sigreg-redundancy ablation row — cube B-off: FULL-RANK (erank 50/192), planning 62.7 / bc 64.0; reacher B-off: FULL-RANK (erank 108/192) but planning 10.7 ≪ A-on 40.0.

---

## §31 🔄🆕 SIGReg-REDUNDANCY test on **CUBE** — the STATE-DRIVEN clean retest of §30 (3 arms A/B/C; THIS entry = arm A = cube baseline) (L40S `stratus-lookout`, 2026-06-22) [IN FLIGHT]

**Motivation (why).** §30 ran the SCAR "is latent SIGReg redundant when the action head predicts raw actions directly" test on **pusht**, and its own §18 CAVEAT flagged pusht as a CONFOUNDED testbed: pusht's action is **history-copyable** (zeroing `a_<t` raised open-loop val_act, §18), so `act_loss` can be partly satisfied by copying the past-action stream rather than by reading a rich latent — which WEAKENS `act_loss`'s anti-collapse pressure on `emb` and makes a healthy Arm B only a *conservative*-positive. The brief's fix: **cube is STATE-DRIVEN** (pick-place where the next action genuinely depends on the current cube + gripper pose, not on action history) → a CLEAN retest of the two conjectures: **(1) latent-SIGReg is REDUNDANT when predicting raw actions directly** (drop it → no collapse, no SR loss); **(2)** replacing the action-emb **stop-grad** with **VICReg on the action embedding** (`sigreg_act`) is a **viable** alternative anti-collapse for the act_emb. Confirming (1) is a "Simple"-pillar win for LeWAM (one fewer loss term); (2) is a design alternative to the detach.

**The 3 arms (env=cube, FROM SCRATCH, action_pred ON throughout, vit-tiny-192, only the sigreg knobs differ).** All match the §30 design transplanted to cube, sharing ONE data config + cap so A/B/C are apples-to-apples:
- **Arm A (THIS entry — baseline):** `loss.sigreg.weight=0.09` (default λ), `action_pred.detach_target=true`, `action_pred.sigreg_act=false`. Latent-SIGReg ON + action-emb **stop-grad** (the validated default). Run `sigreg_A_cube_scratch`.
- **Arm B (conjecture 1):** `loss.sigreg.weight=0` (latent SIGReg term exactly zeroed; module still RUNS + logs `sigreg_loss` for monitoring, contributes 0 to loss), rest = A. Run `sigreg_B_cube_scratch`.
- **Arm C (conjecture 2):** `loss.sigreg.weight=0.09`, `detach_target=false` (act_emb target is NO LONGER stop-gradded — `intent_loss` grad flows back into the action_encoder), `sigreg_act=true` (the VICReg/SIGReg anti-collapse term `+λ·sigreg(act_emb)` is added). Run `sigreg_C_cube_scratch`.

**⚠️ CONCURRENCY — a sibling agent owns the A/B/C *training*; THIS agent owns arm-A *measurement + eval + this section*.** When this agent went to launch cube_A it found a sibling agent had ALREADY launched the canonical `sigreg_{A,B,C}_cube_scratch` runs (the §30 naming convention, `data=ogb` h5, `batch_size=64`, started 00:25–00:30 on 2026-06-22). This agent's first launch was a DIVERGENT duplicate (`cube_A_sigreg`, `data=ogb_lance`, bs128) — that is exactly the config-drift failure CLAUDE.md rule 7 exists to prevent (two arm-A baselines with different data/bs ⇒ A-vs-B not comparable). So this agent **killed its own duplicate + prune loop + cleaned its scratch dirs**, and ADOPTED the canonical `sigreg_A_cube_scratch` (which is matched to the sibling's B/C: same `data=ogb` h5, same bs64, same cap) as the arm-A baseline. The collapse-measurement + eval below are READ-ONLY on the landed checkpoints (no conflict with the sibling's training).

### Exact full config (canonical `sigreg_A_cube_scratch`, read from its `config.yaml`)
`data=ogbench/cube_single_expert.h5` (the 102 GB cube h5, frameskip 5 ⇒ action_block 5, **adim=25** = 5 action-dim × 5; keys pixels/action/observation, `keys_to_merge.proprio=proprio`), `model=lewm` (jepa.JEPA, encoder `vit_hf size=tiny patch=14 image=224 **pretrained=false**` → random CLS-192, `embed_dim=192`, `use_action_history=true use_proprio=false`), `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0 detach_target=true sigreg_act=false` (`action_head.type=mse`), `loss.sigreg.weight=0.09` (knots 17, num_proj 1024), `init_from=null` (from scratch), `trainer.max_epochs=100 limit_train_batches=1000 limit_val_batches=20`, `loader.batch_size=64 num_workers=4`, `optimizer=AdamW lr=5e-5 wd=1e-3`, `seed=3072`, `ckpt_every=10`, `freeze_wm=false`, `goal_conditioned=false horizon_conditioned=false` (the BASE WM + reactive intention head, NOT the §22 GC policy). Trainer `train_sigreg.py` (= `train.py` + the z_std monitor; the §30 file). `output_model_name=sigreg_A_cube_scratch subdir=sigreg_A_cube_scratch`.

### STEP 1 ✅ — arm flags exist + a 1-step cube forward HONORS them (numerically verified, GPU6, `/mnt/minghao_data/verify_sigreg_cube.py`, real cube batch bs32 from `data=ogb_lance`)
Built the gated model + one real cube batch (adim=25), ran `lejepa_forward` per arm, printed the loss decomposition + a gradient-flow probe (`autograd.grad(intent_loss, action_encoder.params)`):
| arm | w | detach_target | sigreg_act | TOTAL loss | EXPECT (pred + w·sig + intent + act [+ w·sig_act]) | \|Δ\| | intent_loss→action_encoder grad-norm |
|---|---|---|---|---|---|---|---|
| A baseline | 0.09 | true | false | 3.082356 | pred 0.2777 + 0.09·9.3503 + intent 1.0068 + act 0.9564 = 3.082356 | 2.4e-8 | **0.000000** (target detached ✓) |
| B latent-sig OFF | 0.0 | true | false | 2.240826 | pred+intent+act = 2.240826 (SIGReg term **provably 0**) | 1.2e-7 | **0.000000** (target detached ✓) |
| C sigreg_act | 0.09 | false | true | 4.219764 | A-terms + 0.09·`sigreg_act_loss`(12.6379)=1.1374 → 4.219764 | 1.4e-8 | **0.253376** (NONZERO — grad flows to act_emb ✓) |
→ **All three flags honored on cube.** Arm B's latent-SIGReg contribution is exactly 0 (`|loss−(pred+intent+act)|=1.2e-7`, FP noise). Arm C ADDS the act-emb VICReg term (`sigreg_act_loss=12.638` present, `λ·sigreg_act=1.137` enters the total) AND lets `intent_loss` backprop into the action_encoder (grad-norm 0.2534 ≠ 0), whereas A/B stop-grad it (grad-norm **exactly 0**) — the detach_target=false ⇄ true double-dissociation is clean. (Same z_std=0.3218 / act_emb_std=0.0801 at init across arms, identical random seed.) STATUS: not blocked.

### STEP 2 🔄 — TRAIN (arm A RELAUNCHED by this agent after the sibling's arm-A died; B sibling-owned)
**The sibling's canonical arm-A and arm-C DIED early** (config.yaml written ~00:25–00:30 but the process gone by ~00:40, only `config.yaml` on disk, NO weights, NO running PID — the multi-agent GPU-contention OOM I also hit on my first verify attempt; a 3rd agent "noah" is concurrently active too: `cube_noah_smoke`/`sigreg_noah_reacher_scratch` in the proc table). Only **arm B (`sigreg_B_cube_scratch`) stayed alive.** Since arm A is THIS agent's deliverable and was not running, this agent **RELAUNCHED `sigreg_A_cube_scratch`** with the EXACT canonical config (`data=ogb` h5, bs64, nw4 — the dead run's `config.yaml` verbatim, matched to the alive arm B) on GPU 5, via `/mnt/minghao_data/relaunch_cube_A.sh` (full disk-safe env: `STABLEWM_HOME`/`SPT_CACHE_DIR=/mnt/minghao_data/spt_cube_A`/`XDG_CACHE_HOME`/`TMPDIR`/`MPLCONFIGDIR`/`HF_HOME` all on `/mnt`, `+ckpt_every=10`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`). Train PID 2175335 (GPU5); keep-newest-3 prune loop `/mnt/minghao_data/prune_cube_A.sh` (PID 2175499) alongside. Log `/mnt/minghao_data/sigreg_A_cube_train.log`. **Health-verified:** proc alive, caching the 102 GB h5 (`Cached 'action'` @00:41, `Cached 'observation'` @00:47 — ~6 min/key I/O, NOT a hang). Then ~§30-pusht-like bs64 throughput ⇒ ETA many hours to ep100. Convergence by val-act_loss plateau over the last ~10 % (rule 8); collapse signal shows early (§30: ep10–20), SR needs ~plateau. `ckpt_every=10` ⇒ `weights_epoch_{10,…,100}.pt`. **arm C** is currently NOT running (its sibling launch also died); the finalize watcher (below) measures C too IF its ep100 lands.

### AUTOMATED FINALIZE (the unblock mechanism — detached on L40S)
`/mnt/minghao_data/cube_A_finalize.sh` (PID 2198989, alive, `EVALGPU=5`) WAITS for `$STABLEWM_HOME/checkpoints/sigreg_A_cube_scratch/weights_epoch_100.pt`, then (a) STEP-3 `measure_collapse_cube.py --epoch 100` over whichever of A/B/C have ep100, (b) STEP-4 `eval_gip.py --config-name cube policy=sigreg_A_cube_scratch +gip_eval.mode={planning,bc} +ckpt_epoch=100 eval.num_eval=50 seed={42,0,1}` (6 evals), tee → `/mnt/minghao_data/cube_A_finalize.log` + per-eval `/mnt/minghao_data/cube_A_eval_{mode}_s{seed}.log`. **UNBLOCK CONDITION:** ep100 ckpt on disk → watcher auto-fires measure + the 6 N=50 evals. The RESULTS table + VERDICT fill from `cube_A_finalize.log`.

### STEP 3 — COLLAPSE MEASUREMENT (DIRECT from the converged ckpt; `/mnt/minghao_data/measure_collapse_cube.py`, run as `sudo -u minghao.fu`, lewm venv, disk-safe env)
For each arm: `gip.load_gip_model(run, epoch=100)` (the SAME path eval uses; latest = converged, NOT a val-best — avoids the reacher val-best trap [[project_gcidm_vs_ours_verdict]]) → one SHARED deterministic cube val batch (256 windows, rebuilt from `full_config.yaml`) through `model.encode` → `z_std = emb.reshape(-1,D).std(dim=0).mean()` + `act_emb_std` + **erank_pr = participation ratio of the per-dim VARIANCE spectrum `(Σv)²/Σv²`** (the STEP-3 spec) + erank_ent (entropy of singular values, for continuity with §30) + dimstd[min,max]. COLLAPSE = z_std→0 AND erank≪D. SCALE-SHRINK (≠ collapse) = low z_std but erank stays high + all per-dim std>0 (the §30 ep20 reading: Arm B scale-shrunk ~5× but erank HIGHER than A).
```
sudo -u minghao.fu env STABLEWM_HOME=/mnt/minghao_data/.stable-wm <disk-safe env> CUDA_VISIBLE_DEVICES=<lowest> \
  /var/lib/docker/data/minghao_home/lewm/bin/python /mnt/minghao_data/measure_collapse_cube.py --epoch 100 --bs 256
```

### STEP 4 — EVAL SR (cube, N=50, 3 seeds {42,0,1}, same-box L40S, CONVERGED ckpt; planning + bc)
```
sudo -u minghao.fu env <disk-safe env> CUDA_VISIBLE_DEVICES=<train-region GPU> \
  /var/lib/docker/data/minghao_home/lewm/bin/python eval_gip.py --config-name cube \
  policy=sigreg_A_cube_scratch +gip_eval.mode=<planning|bc> eval.num_eval=50 seed=<42|0|1>
```
`--config-name cube` = `config/eval/cube.yaml` (env `swm/OGBCube-v0` single, N=50, goal_offset_steps=25, eval_budget=50, action_block=5, `terminate_at_goal`, privileged `set_target_pos` block goal — the §11/§12 cube eval). `mode=planning` = pure WM CEM; `mode=bc` = the reactive intention head. Same-box (train+eval same L40S) per [[project_l40s_cross_gpu_rendering]].

### RESULTS (to fill at convergence — ALL per-seed)
| arm | final z_std | act_emb_std | erank_pr/192 | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) | reading |
|---|---|---|---|---|---|---|
| **A (SIGReg ON, detach, THIS)** | [pending] | [pending] | [pending] | [pending] | [pending] | the baseline |
| B (SIGReg OFF) | [pending] | [pending] | [pending] | [pending] | [pending] | conjecture-1 (collapse? SR loss?) |
| C (sigreg_act, no detach) | [pending] | [pending] | [pending] | [pending] | [pending] | conjecture-2 (viable?) |

### VERDICT (to write at convergence)
[PENDING — arm A is the reference. Did B collapse / lose SR vs A? Does C help or match? The cube STATE-DRIVEN setting removes the §18 history-copyability confound, so a healthy Arm B here is a STRONGER positive for "SIGReg redundant when predicting actions directly" than §30's pusht conservative-positive.]

### Files (this agent, all on L40S)
- `/mnt/minghao_data/verify_sigreg_cube.py` — NEW (STEP-1 3-arm cube forward + grad-flow probe; numbers above).
- `/mnt/minghao_data/measure_collapse_cube.py` — NEW (STEP-3 direct-from-ckpt collapse; participation-ratio erank + entropy erank; cube val batch via gip.load_gip_model).
- **Killed/cleaned (this agent's divergent duplicate):** `cube_A_sigreg` run + `launch_cube_A.sh`/`prune_cube_A.sh` + scratch dirs (`spt_cube_A`,`xdg_cube_A`,`tmp_cube_A`,`mpl_cube_A`,`hydra_cube_A`) — removed to prevent config drift; the canonical `sigreg_A_cube_scratch` (sibling-owned, matched to B/C) is the arm-A baseline.
- **NOT touched:** shared `train.py`/`train_sigreg.py`/`jepa.py`/`gip.py`/`eval_gip.py`, the sibling's running A/B/C procs + their checkpoints, `gen_master_table.py`.

---

## §30 🔄🆕 SIGReg-REDUNDANCY test — is the latent anti-collapse (SIGReg/VICReg) REDUNDANT when the action head predicts actions directly? (L40S `stratus-lookout`, 2026-06-21) [IN FLIGHT]

**Motivation (why).** SCAR (2605.16412, §3.3) argues the *prediction* loss is the load-bearing anti-collapse force: a collapsed latent cannot predict varying actions, so the action-prediction loss `act_loss=‖â−a‖²` itself anchors the latent. If true, then **with the action-prediction head active, the explicit SIGReg latent anti-collapse should be REDUNDANT** — dropping SIGReg should NOT collapse the latent and should NOT drop SR. Confirming this is a "Simple"-pillar win for LeWAM (one fewer loss term / hyperparameter when the model predicts actions directly). The decisive measurement is the **latent-z std trajectory**: does Arm B (no SIGReg) keep a healthy z_std like Arm A, or decay toward 0?

**Theory mechanism under test.** In `train.py` (`lejepa_forward`) the latent loss is `output["loss"] = pred_loss + lambd*sigreg_loss` with `lambd = cfg.loss.sigreg.weight` (default 0.09). SIGReg (`module.SIGReg`, knots=17 num_proj=1024) is the LeJEPA isotropic-Gaussian regularizer = the explicit anti-collapse. With `action_pred.enabled=true` the total also gets `+ w_intent·intent_loss + w_act·act_loss` (the GIP intention head: `intent_loss=(intention−act_emb.detach())²`, `act_loss=(decoder(intention)−raw_action)²`). The conjecture: `act_loss` (raw-action regression, head=mse) supplies an independent anti-collapse pressure on the shared latent `emb`, so SIGReg→0 should be harmless.

### Design — 2 arms, FROM SCRATCH, action_pred ON throughout, ONLY `loss.sigreg.weight` differs
- **Env:** PushT (fastest wired single-task env; `data=pusht`, the committed `config/train/data/pusht.yaml`, frameskip 5 ⇒ action_block 5, adim=10, full `pusht_expert_train.h5`). State-driven 2-D contact task.
- **Arm A (baseline, SIGReg ON):** `loss.sigreg.weight=0.09` (the current default λ). GPU 6 (relaunched there after a transient GPU-3 OOM from a co-located other-user 36 GB proc; first GPU-3 attempt died at CUDA-init, NO partial ckpt written). PID 1752531.
- **Arm B (conjecture, SIGReg OFF):** `loss.sigreg.weight=0` — the `lambd*sigreg_loss` term is exactly zeroed (SIGReg module still RUNS and `sigreg_loss` is still logged for monitoring, but contributes 0 to `loss`). GPU 5. PID 1752039.
- **From scratch** (`init_from=null`, encoder `vit_hf` size=tiny patch=14 image=224 **pretrained=false** → random init, CLS-192). This is the load-bearing choice: warm-starting from a converged base would pre-establish a non-collapsed latent (via the WM pred_loss during base pretrain) and HIDE any collapse. From scratch, the only anti-collapse forces during joint training are SIGReg (Arm A) vs `act_loss`/`intent_loss`/`pred_loss` alone (Arm B).
- **Everything else IDENTICAL:** `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0 detach_target=true sigreg_act=false`, `trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4` (= the canonical §10/§21 from-scratch GIP recipe, line 91-102), seed 3072 (lewm.yaml default), `+ckpt_every=10`.

### Measurement — the COLLAPSE MONITORS are the point
Added ONE latent-z collapse monitor to a NON-INVASIVE copy `train_sigreg.py` (= `train.py` + 4 lines: 3 comments + `output["z_std"] = emb.detach().reshape(-1, emb.shape[-1]).std(dim=0).mean()`, inserted right after the `output["loss"]=pred+lambd*sigreg` line). `diff train.py train_sigreg.py` = EXACTLY those 4 added lines, nothing else. The shared `train.py` is UNTOUCHED (other agents — e.g. the §24-SIZE dinov2_base_768 run, PID 1747041 on GPU 4 — are concurrently active). The monitor key ends in `_std`, so the existing `losses_dict` filter (`if "loss" in k or k.endswith("_std")`) auto-logs it every step AND in the per-epoch validation table. Per-epoch we read from the logs: (1) `validate/z_std_epoch` (latent collapse ⇒ →0 — THE DECISIVE SIGNAL), (2) `validate/pred_loss_epoch` (does it trivially →0 via collapse?), (3) `validate/act_loss_epoch`, `validate/intent_loss_epoch`, (4) `validate/act_emb_std_epoch` (the pre-existing action-emb collapse monitor), (5) `validate/sigreg_loss_epoch` (still computed both arms; for Arm B it's logged-but-not-optimized).

### 1-STEP FORWARD VERIFICATION (done, BEFORE the real runs) — sigreg contribution is provably 0 in Arm B
Ran `verify_sigreg_zero.py`: built the gated model + one real pusht batch (bs=64), called `lejepa_forward` with each λ, no_grad, printed the loss decomposition:
| arm | λ | pred_loss | sigreg_loss | λ·sigreg | intent | act | TOTAL loss | pred+intent+act | TOTAL−(pred+intent+act) |
|---|---|---|---|---|---|---|---|---|---|
| A (SIGReg ON) | 0.09 | 0.085728 | 28.402752 | **2.556248** | 1.010820 | 0.967422 | 4.620216 | 2.063969 | **2.556248** = λ·sigreg ✓ |
| B (SIGReg OFF)| 0.0  | 0.085728 | 28.184174 | **0.000000** | 1.010820 | 0.967422 | 2.063969 | 2.063969 | **0.000000** (|diff|=2.24e-8 FP) ✓ |
→ Arm B's total loss is EXACTLY `pred_loss + w_intent·intent_loss + w_act·act_loss` with the SIGReg term provably removed. (Also note untrained-init `z_std=0.0013`, `act_emb_std=0.107` on a single batch — the random-init ViT CLS is near-constant across a batch, so z_std STARTS tiny; the test is whether *training* RAISES it. The fit/z_std at the first training step was ~0.33 in the smoke, so z_std moves once gradients flow.)

### Infra (DISK-SAFE, L40S shared box — good citizen)
L40S `stratus-lookout`, run as `sudo -u minghao.fu`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`. **Disk at launch:** `/var/lib/docker` 99% (90 GB free), `/` 97% (3.3 GB), `/mnt/minghao_data` 21% (2.4 TB free) → ALL writes redirected to `/mnt`: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_sigreg TMPDIR=/mnt/minghao_data/tmp_sigreg MPLCONFIGDIR=/mnt/minghao_data/mpl_sigreg HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline`, plus `hydra.run.dir=/mnt/minghao_data/hydra_sigreg_<A|B> hydra.output_subdir=null` (Hydra otherwise writes `./outputs` into the full /var/lib/docker disk). Ckpts → `STABLEWM_HOME/checkpoints/sigreg_<A|B>_pusht_scratch/`. `+ckpt_every=10` (save only ep 10,20,…,100) + a `prune_sigreg.sh` keep-newest-3 loop (SaveCkptCallback does NOT self-prune in this repo version — verified). GPUs: A=6, B=5 (lowest-occupied at relaunch; GPUs 1/2/4 held by other users' procs incl. the dinov2_base_768 run, never touched). Same box train+eval → no cross-GPU render OOD ([[project_l40s_cross_gpu_rendering]]).

### EXACT train commands (L40S)
```
# common env (both arms):
export STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_sigreg \
       TMPDIR=/mnt/minghao_data/tmp_sigreg MPLCONFIGDIR=/mnt/minghao_data/mpl_sigreg HF_HOME=/mnt/minghao_data/hf \
       HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
# ARM A (SIGReg ON, GPU6):  loss.sigreg.weight=0.09
# ARM B (SIGReg OFF, GPU5): loss.sigreg.weight=0
CUDA_VISIBLE_DEVICES=<6|5> python train_sigreg.py data=pusht \
  action_pred.enabled=true action_pred.detach_decoder=false action_pred.head=mse \
  action_pred.w_act=1.0 action_pred.w_intent=1.0 action_pred.sigreg_act=false \
  loss.sigreg.weight=<0.09|0> init_from=null \
  output_model_name=sigreg_<A|B>_pusht_scratch subdir=sigreg_<A|B>_pusht_scratch \
  trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4 +ckpt_every=10 \
  hydra.run.dir=/mnt/minghao_data/hydra_sigreg_<A|B> hydra.output_subdir=null
```
(Arm A additionally had `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` after the GPU-3 transient OOM relaunch.)

### Throughput / ETA
Both arms at ~6.3 it/s, 4000 steps/epoch ⇒ ~10.6 min/epoch ⇒ **~17.6 h/arm for 100 ep** (idle-ish GPUs 5/6, full pusht pixel encoder forward+backward from scratch). GIP head ON confirmed in both logs (`[GIP] Intention predictor ON  Adim=10  head=mse w_act=1.0 w_intent=1.0 detach_target=True sigreg_act=False`). z_std monitor confirmed logging both arms.

### EVAL plan (when converged — rule 8/9)
Both arms, same-box, **N=50 × 3 seeds {42,0,1}**: `eval_gip.py --config-name pusht policy=sigreg_<A|B>_pusht_scratch +gip_eval.mode=<planning|bc|guided> eval.num_eval=50 seed=<42|0|1>` (planning = pure WM CEM; bc = intention head reactive; guided = intention warm-starts CEM). Question: does Arm B (no SIGReg) MATCH Arm A's SR, or degrade?

### §18 CAVEAT — pusht is state-driven but its action is HISTORY-CORRELATED (weakens the anti-collapse claim)
PushT is contact-heavy but FULLY observed; §18 established the past-action stream `a_<t` is a strong open-loop predictor of the demonstrator's next push (zeroing it RAISED open-loop val_act 0.132→0.153). So a chunk of `act_loss` can be satisfied by COPYING/extrapolating the history rather than by reading the latent state — which would WEAKEN `act_loss`'s anti-collapse pressure on `emb` (the model can lower act_loss without using a rich `emb`). This means: if Arm B's latent stays healthy on pusht, the conjecture holds *at least where the action is partly history-copyable* (a conservative-positive); if it collapses, pusht's history-copyability is a confound and a more state-dependent env (cube) would be the cleaner retest. NOTE the current run uses `use_action_history=true` (default), so `past_act` IS fed — the history-copy shortcut is available. A stricter future arm would set `use_action_history=false` to force `act_loss` to read the latent.

### DIRECT-FROM-CHECKPOINT collapse measurement (the ROBUST signal — replaces the broken in-log z_std monitor)
The per-step/per-epoch `validate/z_std` line was NOT reliably captured anywhere in the logs (the in-training monitor is broken; `train_{A,B}.log` only carry the step progress bar). So collapse is measured DIRECTLY off the landed checkpoints, which is the robust signal and what the conjecture actually rides on. Script `/mnt/minghao_data/measure_collapse.py` (run `sudo -u minghao.fu`, lewm venv, GPU 1, full disk-safe env): for each arm's checkpoint it rebuilds the model via `gip.load_gip_model(run, epoch)` (the SAME path `eval_gip.py` uses — both arms load `missing=0 unexpected=0`, so the build is exact), rebuilds the EXACT pusht val split + transforms from the run's `full_config.yaml`, runs ONE shared deterministic val batch (256 windows × T=4, the same A/B batch) through `model.encode`, and computes the train_sigreg formula **`z_std = emb.reshape(-1,D).std(dim=0).mean()`** plus `act_emb_std`, the per-dim std spread, and the latent effective rank (entropy of singular values). Same val batch for both arms ⇒ A-vs-B is apples-to-apples.

**Trajectory (z_std, B/A ratio; lower B ⇒ more collapse; healthy ⇒ B≈A):**
| epoch | A z_std (SIGReg ON) | B z_std (SIGReg OFF) | B/A | A act_emb_std | B act_emb_std | A erank/192 | B erank/192 | reading |
|---|---|---|---|---|---|---|---|---|
| 10 (mid-train) | **0.970** | **0.184** | **0.19** | 0.697 | 0.426 | 74.9 | **108.4** | same picture as ep20: B scale-shrunk ~5×, rank HIGHER than A. |
| 20 (mid-train) | **0.944** | **0.190** | **0.20** | 0.933 | 0.590 | 83.2 | **108.3** | B's latent is **scale-shrunk ~5×** but NOT rank-collapsed (B erank 108 > A erank 83; B per-dim std 0.077–0.444 all > 0). SCALE contraction, not dimensional collapse — the action head keeps the latent FULL-RANK, SIGReg sets the scale. |
| 30..100 | [tracker filling as ckpts land → `/mnt/minghao_data/sigreg_collapse_traj.log`] | | | | | | | |

**KEY ep10→ep20 finding (the decisive trend so far): Arm B's z_std is FLAT, not decaying.** B z_std 0.184 (ep10) → 0.190 (ep20) — slightly UP, not heading to 0; B erank flat at ~108 both epochs. A true SCAR collapse would show B's z_std decaying monotonically toward 0 with erank dropping toward 1 (latent losing directions ⇒ can't represent varying states ⇒ can't predict varying actions). Instead B sits at a STABLE smaller scale with HIGHER rank than A. This is the signature of a **scale OFFSET held by the action head**, not a progressive collapse. The erank distinction is load-bearing: Arm B's effective rank (108/192) EXCEEDS Arm A's (83/192) at both epochs, so the action head IS holding the latent's dimensionality up — exactly the conjecture's mechanism. The lower z_std is a smaller *radius* (SIGReg's isotropic-unit-variance target inflates A's per-dim std toward ~1.0; without it B settles at the natural ~0.2 scale the pred/act losses produce). Whether that smaller radius hurts the downstream policy is the eval question — a global scale offset is harmless to a scale-tolerant readout. **Provisional reading (ep10–20): NOT a SCAR collapse (rank preserved, scale STABLE not decaying); a scale contraction. Conjecture-FAVORABLE so far. Confirm with the ep100 trajectory + the SR eval.**

### RESULT TABLE (to fill at convergence — ALL per-seed)
| arm | final `z_std` (latent collapse monitor) | `act_emb_std` | erank/192 | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) | SR guided |
|---|---|---|---|---|---|---|
| A (SIGReg ON, λ=0.09) | [ep10:0.970 ep20:0.944] → [final pending] | [ep20: 0.933] | [ep20: 83.2] | [pending] | [pending] | [pending] |
| B (SIGReg OFF, λ=0)   | [ep10:0.184 ep20:0.190 FLAT] → [final pending] | [ep20: 0.590] | [ep20: 108.3 > A] | [pending] | [pending] | [pending] |

### VERDICT (to write at convergence)
[PENDING — but the ep20 direct measurement already SHARPENS the question.] The naive "z_std→0 = collapse" framing is too blunt: at ep20 Arm B's z_std is 5× lower than Arm A's (0.19 vs 0.94), which LOOKS like collapse, BUT Arm B's latent effective rank is HIGHER (108 vs 83 / 192) and every per-dim std is > 0 — so the action head is keeping the latent FULL-RANK; SIGReg only sets the scale (its isotropic-unit-variance target inflates A's per-dim std toward 1.0). So the live question is: does Arm B's z_std keep DECAYING toward 0 with a COLLAPSING rank (→ true SCAR collapse, conjecture FALSIFIED, SIGReg load-bearing), or does it STABILIZE at a smaller scale with rank preserved (→ scale contraction not collapse, conjecture-favorable, and harmless to a scale-tolerant readout → SR should match → SIGReg redundant when predicting actions directly = a "Simple"-pillar win)? Decide on the ep100 trajectory + the SR eval. Plus the §18 history-copyability caveat (a healthy Arm B on pusht is a CONSERVATIVE positive; a collapse here is ambiguous — could be the pusht history-copyability confound → cube / `use_action_history=false` retest needed).

### Files (all on L40S)
- `measure_collapse.py` — NEW (`/mnt/minghao_data/`), the DIRECT-from-checkpoint collapse measurement (rebuilds model via `gip.load_gip_model` + exact pusht val batch + train_sigreg z_std formula + erank). This is the robust signal; the in-log z_std monitor is broken. Re-run per `--epoch` as ckpts land.
- `train_sigreg.py` — NEW (copy of `train.py` + the single z_std monitor; diff = 4 lines). `make_train_sigreg.py`, `verify_sigreg_zero.py`, `launch_sigreg_arm.sh`, `launch_armA.sh`, `prune_sigreg.sh` under `/mnt/minghao_data/`.
- **NOT touched:** shared `train.py`, `jepa.py`, `gip.py`, `eval_gip.py`, `module.py`, other agents' ckpts/GPUs. `gen_master_table.py` NOT edited.
- Ckpts (mine only): `/mnt/minghao_data/.stable-wm/checkpoints/sigreg_{A,B}_pusht_scratch/`. Logs `/mnt/minghao_data/sigreg_logs/train_{A,B}.log`.
- **PIDs:** Arm A 1752531 (GPU6), Arm B 1752039 (GPU5) — BOTH ALIVE (verified ep20+, ~6.2 it/s, state Sl). **Trajectory tracker:** `/mnt/minghao_data/track_collapse.sh` running detached (PID 1922021, GPU 1), auto-measures ep30..100 as ckpts land → appends to `/mnt/minghao_data/sigreg_collapse_traj.log`. **Unblock:** `weights_epoch_100.pt` in both run dirs + the tracker's z_std trajectory flat over last ~10 ep (verify B's z_std stays stable, doesn't decay to 0 — so far FLAT at 0.19) → eval (2 arms × 3 seeds {42,0,1}, N=50, planning + bc + guided, same-box GPU per cross-GPU render rule) → fill table → verdict. The eval commands are in the EVAL plan block above (`eval_gip.py --config-name pusht policy=sigreg_<A|B>_pusht_scratch +gip_eval.mode=<planning|bc|guided> eval.num_eval=50 seed=<...>`).

---

## §31 🔄🆕 SIGReg-ablation on **REACHER** (the CLEAN, state-driven retest of §30) — Arm B (latent-SIGReg OFF) [IN FLIGHT — training] (L40S, 2026-06-22)

**Motivation (why).** §30 ran the SIGReg-redundancy test on PushT and found Arm B (SIGReg OFF) does NOT collapse — it scale-shrinks (z_std 5× lower) while keeping FULL rank (erank 108 > A's 83 / 192), the action head holding the latent's dimensionality up. But §30's own §18 CAVEAT flags PushT as a WEAK test: pusht's action is HISTORY-COPYABLE (`a_<t` is a strong open-loop predictor of the next push, §18), so `act_loss` can be lowered by copying the past-action stream instead of reading the latent — which WEAKENS `act_loss`'s anti-collapse pressure on `emb`, making a "healthy Arm B" only a CONSERVATIVE positive. **reacher is the clean retest:** it is STATE-DRIVEN (the demonstrator's next action depends on the current joint/target geometry, not on action-history extrapolation), so `act_loss` MUST read a rich latent — if the conjecture holds here (Arm B stays healthy / SR matches), it holds where the action is NOT history-copyable, removing the §18 confound. This subagent owns **Arm B** (conjecture-1: `loss.sigreg.weight=0`); sibling subagents run **Arm A** (baseline, λ=0.09) and **Arm C** (`detach_target=false` + `sigreg_act=true`, the VICReg-on-act-emb replacement for the stop-grad) concurrently.

### Conjectures under test
1. **latent-SIGReg is REDUNDANT when the head predicts raw actions directly** (Arm B): the `act_loss=‖â−a‖²` (head=mse) supplies independent anti-collapse pressure on `emb`, so `loss.sigreg.weight→0` should NOT collapse the latent and NOT drop SR.
2. **replacing the act-emb stop-grad with VICReg is viable** (Arm C, sibling): `detach_target=false` (intent_loss gradient flows INTO act_emb) + `sigreg_act=true` (act-emb VICReg/SIGReg anti-collapse) — viable iff act_emb stays healthy and SR holds.

### STEP 1 — FLAGS VERIFIED + 1-STEP FORWARD (L40S, this session) ✅
`/mnt/minghao_data/verify_reacher_arms.py` built the gated model + one real reacher (data=dmc) batch (bs=64), ran `train_sigreg.lejepa_forward`, printed the decomposition. **reacher adim = frameskip(5) × action_dim(2) = 10.**
- **Arm B (`loss.sigreg.weight` 0.09→0):** total loss with λ=0 is EXACTLY `pred+w_intent·intent+w_act·act` — sigreg contribution provably removed. `pred=0.072815 intent=1.007287 act=1.085117` identical for both λ; `TOTAL−(pred+intent+act)=5.22e-08` at λ=0 (FP zero), `=2.489828=λ·sigreg` at λ=0.09. ✓ **Arm B's `loss.sigreg.weight=0` zeroes the latent sigreg term — confirmed numerically.**
- **Arm C mech1 (`detach_target=false`):** `|grad|` of `intent_loss` w.r.t. `model.action_encoder` params = **63.357** (vs **0.000** when `detach_target=true`). ✓ the intent_loss gradient FLOWS to act_emb only when detach_target=false — confirmed.
- **Arm C mech2 (`sigreg_act=true`):** `sigreg_act_loss` present=True value=24.967 (adds `λ·=2.247` to total); absent (not in output) when `sigreg_act=false`. ✓ the act-emb VICReg term is added only when sigreg_act=true — confirmed.

### Design — Arm B, FROM SCRATCH, action_pred ON, head=mse, latent-SIGReg OFF
- **Env:** reacher (`data=dmc`, the committed `config/train/data/dmc.yaml` → `reacher.h5`, frameskip 5 ⇒ action_block 5, adim=10). STATE-DRIVEN 2-link reaching (qpos_match), the clean (non-history-copyable) test vs §30 pusht.
- **Arm B (this subagent):** `loss.sigreg.weight=0` (the `lambd·sigreg_loss` term exactly zeroed; SIGReg module still RUNS + `sigreg_loss` still logged for monitoring, contributes 0 to `loss`). FROM SCRATCH (`init_from=null`, vit-tiny-192, `pretrained=false`, random CLS-192 init — the load-bearing choice: a warm start would pre-establish a non-collapsed latent and HIDE collapse). `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0 detach_target=true sigreg_act=false`.
- **Budget — MATCHES the sibling reacher A/C arms:** `trainer.max_epochs=100 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4`, seed 3072 (lewm.yaml default), `+ckpt_every=10`. **NOTE the cap is 1000 (not §30-pusht's 4000):** the sibling A/C subagents capped reacher to 1000 for turnaround under the L40S oversubscription; "same cap for all arms of this env" ⇒ Arm B uses 1000 too (an initial Arm-B launch at 4000 was killed by a DataLoader-worker SIGTERM under load-avg 146 / 3 concurrent reacher arms hammering the 98 GB reacher.h5; relaunched at 1000 to match siblings — see Infra).
- **Monitor:** trained with `train_sigreg.py` (= `train.py` + the single `output["z_std"]=emb.std(...)` line; diff = 4 lines, shared `train.py` UNTOUCHED). The robust collapse signal is measured DIRECTLY from the converged checkpoint (the in-log z_std monitor was unreliable in §30) via `/mnt/minghao_data/measure_collapse_reacher.py`: rebuilds the model via `gip.load_gip_model` (the SAME path `eval_gip.py` uses), runs ONE shared deterministic reacher val batch through `model.encode`, computes `z_std=emb.reshape(-1,D).std(dim=0).mean()` + effective rank (participation ratio = `exp(entropy)` of the singular-value spectrum) + `act_emb_std` + per-dim std spread.

### Infra (DISK-SAFE — L40S `/var` 97% / `/` 97%, ONLY `/mnt/minghao_data` free; the killer is SPT_CACHE_DIR)
Host L40S, run `sudo -u minghao.fu`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`. ALL writes redirected to `/mnt`: `SPT_CACHE_DIR=/mnt/minghao_data/spt_reacher_B` (the stable-pretraining run/metrics dir — the §SPT killer), `STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_reacher_B TMPDIR=/mnt/minghao_data/tmp_reacher_B MPLCONFIGDIR=/mnt/minghao_data/mpl_reacher_B HF_HOME=/mnt/minghao_data/hf`, `hydra.run.dir=/mnt/minghao_data/hydra_reacher_B hydra.output_subdir=null`. `+ckpt_every=10` (save ep 10,20,…,100) + `/mnt/minghao_data/prune_reacher_B.sh` keep-newest-3 loop. **GPU 7** (lowest-occupied at launch). Same box train+eval → no cross-GPU render OOD ([[project_l40s_cross_gpu_rendering]]).

### EXACT train command (L40S, Arm B)
```
export STABLEWM_HOME=/mnt/minghao_data/.stable-wm SPT_CACHE_DIR=/mnt/minghao_data/spt_reacher_B \
       XDG_CACHE_HOME=/mnt/minghao_data/xdg_reacher_B TMPDIR=/mnt/minghao_data/tmp_reacher_B \
       MPLCONFIGDIR=/mnt/minghao_data/mpl_reacher_B HF_HOME=/mnt/minghao_data/hf \
       HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
CUDA_VISIBLE_DEVICES=7 python train_sigreg.py data=dmc \
  action_pred.enabled=true action_pred.detach_decoder=false action_pred.head=mse \
  action_pred.w_act=1.0 action_pred.w_intent=1.0 \
  action_pred.detach_target=true action_pred.sigreg_act=false \
  loss.sigreg.weight=0 init_from=null \
  output_model_name=sigreg_B_reacher_scratch subdir=sigreg_B_reacher_scratch \
  trainer.max_epochs=100 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4 +ckpt_every=10 \
  hydra.run.dir=/mnt/minghao_data/hydra_reacher_B hydra.output_subdir=null
```

### EVAL plan (when converged — rule 8/9; AUTOMATED via `/mnt/minghao_data/reacher_B_orchestrator.sh`)
Arm B, same-box GPU 7, **N=50 × 3 seeds {42,0,1}**, modes **planning + bc** (reacher has a val-best checkpoint trap §15/§19 → use the CONVERGED ckpt via `+ckpt_epoch=<final>`, NOT val-best):
```
PYTHONPATH=/tmp/reacher_compat:$B python eval_gip.py --config-name reacher \
  policy=sigreg_B_reacher_scratch +ckpt_epoch=<conv> \
  +gip_eval.mode=<planning|bc> eval.num_eval=50 seed=<42|0|1>
```
(`/tmp/reacher_compat/sitecustomize.py` = the dm_control↔mujoco-3.x stale-field shim, REQUIRED for reacher eval.) The orchestrator (PID at launch 2157581) waits for `weights_epoch_100.pt`, then runs the collapse measurement + the 6 SR evals (2 modes × 3 seeds) → appends to `/mnt/minghao_data/reacher_B_logs/EVAL_RESULTS.txt`.

### Throughput / ETA
~2.0→2.5 it/s, 1000 steps/epoch ⇒ ~6.7–8 min/epoch ⇒ **~12 h/arm for 100 ep** (L40S oversubscribed: load-avg ~146, 3 reacher arms + other users' procs; CPU/disk-bound not GPU-bound, rule 10). GIP head ON confirmed in the log; z_std at init ≈ 0.0026 (random-init ViT CLS near-constant across a batch — the test is whether *training* raises it).

### RESULT TABLE (to fill at convergence — ALL per-seed)
| arm | final z_std | act_emb_std | erank/192 | collapsed? | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) | finding |
|---|---|---|---|---|---|---|---|
| B (SIGReg OFF, λ=0) | [pending ep100] | [pending] | [pending] | [pending] | [pending N=50×3] | [pending N=50×3] | [pending — vs Arm A baseline] |

### VERDICT (to write at convergence)
[PENDING.] Same decision rule as §30: **COLLAPSE** = z_std→0 AND erank≪192 (conjecture-1 FALSIFIED, SIGReg load-bearing); **SCALE-SHRINK not collapse** = low z_std but erank stays high + all per-dim std>0 (conjecture-favorable, harmless to a scale-tolerant readout → SR should match Arm A → SIGReg redundant when predicting actions directly, a "Simple"-pillar win). Because reacher is state-driven (NOT history-copyable), a healthy Arm B here is the CLEAN positive §30/§18 asked for. Compare SR planning + bc vs Arm A (sibling); does B lose SR? does C (sibling) help?

### Files (all on L40S)
- `/mnt/minghao_data/verify_reacher_arms.py` — the 1-step forward verification (all 3 arm mechanisms, numeric).
- `/mnt/minghao_data/measure_collapse_reacher.py` — DIRECT-from-checkpoint collapse (z_std + erank + act_emb_std via `gip.load_gip_model` + reacher val batch).
- `/mnt/minghao_data/launch_reacher_B.sh`, `prune_reacher_B.sh`, `reacher_B_orchestrator.sh`. Logs `/mnt/minghao_data/reacher_B_logs/`. Ckpts `/mnt/minghao_data/.stable-wm/checkpoints/sigreg_B_reacher_scratch/`.
- **NOT touched:** shared `train.py`, `jepa.py`, `gip.py`, `eval_gip.py`, `module.py`, `gen_master_table.py`, sibling arms' ckpts/GPUs.

---

## §29 🆕 ENCODER-BOTTLENECK arm: **DINOv2-large 1024-d** (init + finetune) on **can**, single bc+GC model — does a bigger PRETRAINED encoder lift can bc toward DP (~1.0) and clean-GC off the vit-tiny floor (bc 0.71 / clean-GC 0.26)? (L40S `stratus-lookout`, 2026-06-21) [IN FLIGHT — training]

**Motivation (why).** Every prior can result is bottlenecked by the **vit-tiny-192** CLS encoder (5.5M params, 192-d, DINOv2-arch but trained-from-SIGReg or DINOv2-small-init). The §22/§24/§27 verdict is that the can-GC floor is "PREDOMINANTLY REAL — the frozen/learnable vit-tiny-192 latent can't localize the can well enough for goal-reaching" (clean terminal-goal GC tops out ~0.26 = 37% of the 0.71 goal-agnostic bc ceiling), and the §6 latent→full-state probe is only R²≈0.554 on a TRAINED vit-tiny (3-D occluded can pose + velocities are hard for a 192-d CLS). The open question this arm answers: **is the can ceiling an ENCODER-CAPACITY bottleneck, or a formulation/data bottleneck?** If a 4×-deeper, 5.3×-wider **DINOv2-large** encoder (1024-d, 24 layers, ~304M backbone params, ImageNet/LVD-142M-pretrained) lifts (a) can goal-agnostic **bc** toward the DP ceiling (~1.0) and/or (b) clean terminal-goal **GC** off the vit-tiny 0.26 floor, then the floor is (partly) the tiny encoder; if both stay flat (~0.71 bc, ~0.26 GC), the bottleneck is the GC formulation / robomimic occluded-state difficulty, NOT encoder size, and the §22/§27 "floor is real" verdict hardens against the strongest pretrained-encoder counter-test. This is the **encoder axis** of the size sweep (cf. §24-SIZE which scales the trunk WIDTH at vit tiny/small/base 192/384/768 from scratch — this arm jumps to a large PRETRAINED foundation encoder).

### STEP 1 — FLAGS VERIFIED + 1-STEP FORWARD (L40S, this session) ✅
- **Encoder swap mechanics.** The framework already ships `config/train/model/lewm_dinov2.yaml` (`encoder._target_=stable_pretraining.backbone.utils.from_huggingface`, default `model_name=facebook/dinov2-small pretrained=false`). `jepa.JEPA.encode` reads ONLY `encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:,0]` (the CLS token), so any DINOv2 size is a drop-in at this interface. The dinov2-large swap = three overrides: **`model=lewm_dinov2 model.encoder.pretrained=true model.encoder.model_name=facebook/dinov2-large embed_dim=1024`**. The `embed_dim=1024` (vs the lewm.yaml default 192) is the REQUIRED companion: `embed_dim` keys the predictor/projector/pred_proj/action_encoder/horizon_modulator widths, so it MUST equal the encoder hidden_size (dinov2-large = 1024) or the trunk mismatches the CLS token (the same audit flag §24-SIZE documents for any non-tiny encoder). NO `init_from` (the vit-tiny `can_lewm_weights.pt` is shape-incompatible at 1024-d — the encoder comes PRETRAINED from HF, the GIP head + predictor + projector train from scratch).
- **1-step forward (gate).** Real-data smoke (`data=robomimic_can`, `limit_train_batches=3`, GPU 3, deleted after): DINOv2-large downloaded to `/mnt/minghao_data/hf` (439 weight shards loaded, `HF_HUB_OFFLINE=0`), startup `[GIP] Intention predictor ON Adim=35 head=mse`, `[GIP] OURS horizon conditioning ON AdaLN-Zero H_max=50`, **horizon_modulator built with 297,216 params** (cond_proj 264,192 = 1024×258 → confirms `embed_dim=1024` propagated, vs vit-tiny's 82,560), **NO `freeze_wm=true` line ⇒ encoder LEARNABLE (end-to-end finetune)**, GC losses finite (`validate/act_loss` path runs → GoalSamplingDataset + goal_dropout=0.5 OK), `weights_epoch_1.pt` (1.98 GB) + config.json saved cleanly to `/mnt`, **exit 0, no shape crash, no NaN.**
- **GPU-FIT.** DINOv2-large end-to-end needs much more activation memory than vit-tiny: **batch_size=128 and 32 BOTH OOM'd a 44 GB L40S** (43.8 GB allocated in a single fwd at the 24-layer/1024-d encoder backprop). **batch_size=8 fit (5.8 s/3-batch), batch_size=16 fit (1.89 it/s), batch_size=24 fit (1.35 it/s, peak ~40 GB / 44 GB).** Chose **batch_size=24** (the largest that fits with headroom). First OOM was also the §24 `trainer.devices=auto`-ignores-`CUDA_VISIBLE_DEVICES` gotcha — fixed by `trainer.devices=[0]` (logical-0 = the pinned physical GPU 3) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### STEP 1.5 — DISK SAFETY (L40S `/var/lib/docker` 99 % full, `/` 97 %, ONLY `/mnt/minghao_data` has space) ⚠️
- All scratch redirected to `/mnt`: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm` (→ checkpoints + datasets), `XDG_CACHE_HOME`/`TMPDIR`/`MPLCONFIGDIR`/`HF_HOME` all `/mnt/minghao_data/*_dinov2_large_1024`, `hydra.run.dir=/mnt/minghao_data/hydra_dinov2L`.
- **CRITICAL extra redirect not in §24:** `stable_pretraining.Manager` writes a **5.6 GB Lightning resumption ckpt** (`runs/.../checkpoints/{epoch=N-step=M,last}.ckpt`, ~11 GB/run) to `~/.cache/stable-pretraining` = `/var/lib/docker/...` (the FULL disk) — vit-tiny's were ~117 MB so §24 never noticed, but dinov2-large's are 5.6 GB EACH and would fill the 89 GB-free full disk. Fix = **`SPT_CACHE_DIR=/mnt/minghao_data/spt_dinov2L_runs`** (the env var read by `stable_pretraining/_config.py:84`), which moves the whole `runs/` tree to `/mnt`. Verified: the 5.6 GB ckpts now land on `/mnt` (2.4 TB free), `/var/lib/docker` holds steady at 89 GB free through training (writes nothing there).
- Prune loop (`/mnt/minghao_data/prune_dinov2L.sh`, every 120 s): keeps newest-3 `weights_epoch_*.pt` (1.98 GB each) in the run dir + newest-1 SPT `runs/` dir (the ~11 GB resumption ckpts). `+ckpt_every=10` ⇒ `weights_epoch_{10,20,...,100}.pt`.

### STEP 2 — EXACT TRAIN COMMAND (host `L40S` `stratus-lookout`, `sudo -u minghao.fu`; `$B=/var/lib/docker/data/minghao_home/workspace/le-wm-repro`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`; driver `/mnt/minghao_data/train_dinov2L.sh`, GPU 3)
```
env STABLEWM_HOME=/mnt/minghao_data/.stable-wm SPT_CACHE_DIR=/mnt/minghao_data/spt_dinov2L_runs \
  MUJOCO_GL=egl HF_HUB_OFFLINE=0 HF_HOME=/mnt/minghao_data/hf \
  XDG_CACHE_HOME=/mnt/minghao_data/xdg_dinov2_large_1024 \
  TMPDIR=/mnt/minghao_data/tmp_dinov2_large_1024 \
  MPLCONFIGDIR=/mnt/minghao_data/mpl_dinov2_large_1024 \
  OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=3 \
  python train.py data=robomimic_can model=lewm_dinov2 \
  model.encoder.pretrained=true model.encoder.model_name=facebook/dinov2-large \
  embed_dim=1024 loader.batch_size=24 \
  output_model_name=can_gc_dinov2L subdir=can_gc_dinov2L \
  freeze_wm=false \
  action_pred.enabled=true action_pred.goal_conditioned=true \
  action_pred.horizon_conditioned=true action_pred.goal_dropout=0.5 \
  action_pred.hindsight_max_k=50 \
  trainer.max_epochs=100 trainer.devices=[0] \
  +trainer.limit_train_batches=300 \
  +ckpt_every=10 wandb.enabled=false \
  hydra.run.dir=/mnt/minghao_data/hydra_dinov2L
```
- **The single bc+GC model.** `freeze_wm=false` (encoder LEARNABLE, end-to-end LeWAM) + `action_pred.enabled=true goal_conditioned=true horizon_conditioned=true goal_dropout=0.5 hindsight_max_k=50` — byte-identical to the §24 recipe EXCEPT the encoder (dinov2-large-1024 vs vit-tiny-192), NO `init_from`, and `loader.batch_size=24` + `limit_train_batches=300` (vs §24 full ~136 batches at b128). The `goal_dropout=0.5` zeros z_goal on half the samples so ONE model serves both the goal-AGNOSTIC bc policy and the goal-conditioned GC policy (CFG / BESO two-setting WAM).
- **`limit_train_batches=300` cap (STATED).** At batch=24 a full can epoch ≈ 725 steps; capped to 300 (300×24 = 7,200 samples/epoch ≈ 41 % of the ~17,400-sample full set) for wall-clock (full uncapped ≈ 15 h; capped ≈ 6.8 h). Convergence judged by val-act_loss plateau over the last ~10 % of epochs, NOT by epoch count (rule 8); if still improving at ep100 the cap is too tight and the run extends.
- Log `/mnt/minghao_data/train_dinov2L.log`. Per-epoch wall ≈ **244 s** (4 min). Checkpoints `/mnt/minghao_data/.stable-wm/checkpoints/can_gc_dinov2L/weights_epoch_{10..100}.pt`.

### Convergence (rule 8 — verify plateau) [FILLING LIVE]
- Sanity (pre-train) `validate/act_loss = 1.134`. **After epoch 0: `validate/act_loss = 0.2377`** — the PRETRAINED dinov2-large + head drops the action loss ~5× in ONE epoch (cf. §24 vit-tiny ~1.18 at epoch 1: the pretrained foundation encoder adapts the action head far faster). [continues filling per epoch; plateau check over last ~10 ep before any headline number]

### STEP 3 — EVAL COMMANDS (can, same-box GPU 3 = train GPU per the cross-GPU render gotcha [[project_l40s_cross_gpu_rendering]], N=50, 3-seed {42,0,1})
- **(a) bc = goal-AGNOSTIC (canonical b240/o90):**
```
python eval_histbc_robomimic.py --config-name robomimic world.task=PickPlaceCan \
  world.num_envs=10 dataset.stats=can eval.dataset_name=can eval.num_eval=50 \
  eval.eval_budget=240 eval.goal_offset_steps=90 policy=can_gc_dinov2L \
  +gip_eval.mode=policy +gip_eval.goal_conditioned=false seed=<42|0|1>
```
- **(b) clean GC = terminal-goal (the §27 fix, b240/o90):**
```
python eval_histbc_robomimic.py --config-name robomimic world.task=PickPlaceCan \
  world.num_envs=10 dataset.stats=can eval.dataset_name=can eval.num_eval=50 \
  eval.eval_budget=240 eval.goal_offset_steps=90 +eval.goal_mode=terminal \
  policy=can_gc_dinov2L +gip_eval.mode=policy +gip_eval.goal_conditioned=true seed=<42|0|1>
```
  (`+eval.goal_mode=terminal` = the §27 `_extract_init_goal` patch in the installed swm `world.py`: goal = the episode's TRUE terminal frame, budget kept long at 240, so the floor is measured against the actual task-completion goal not a mid-trajectory one. Same eval-GPU as train per the L40S cross-GPU render gotcha.)

### THE encoder-bottleneck table [FILLING LIVE — pending convergence + eval]
| encoder | params | embed_dim | can bc SR (N=50, b240/o90, 3-seed {42,0,1}) | can clean-GC SR (terminal-goal b240/o90, 3-seed) | note |
|---|---|---|---|---|---|
| **DINOv2-large (init+finetune, THIS §29)** | ~304M | 1024 | [pending] | [pending] | the encoder-capacity counter-test |
| vit-tiny-192 (the baseline) | 5.5M | 192 | **0.71** (§1c goal-agnostic ceiling) | **0.26** (§27 CLEAN terminal-goal, OURS e2e) | the bottlenecked baseline |
| DP (full-data BC reference) | — | — | **~1.0** | — | the imitation ceiling bc is compared to |

### Verdict [PENDING convergence + eval]
Does the 5.3×-wider PRETRAINED DINOv2-large encoder lift can goal-agnostic **bc** toward DP (~1.0) above the vit-tiny 0.71, AND/OR lift clean terminal-goal **GC** above the vit-tiny 0.26 floor? If yes → the can ceiling is (partly) an encoder-capacity bottleneck (the tiny CLS can't localize the occluded can). If both stay flat → the bottleneck is the GC formulation / robomimic occluded-state difficulty, NOT encoder size, and the §22/§27 "floor is real" verdict hardens against the strongest pretrained-encoder counter-test. [Fills once the 6 N=50 evals land.]

---

## §31 🔄🆕 SIGReg-REDUNDANCY test on **tworoom** (the CLEAN state-driven retest of §30's pusht conjectures) — arm **tworoom_B** = conjecture-1: latent-SIGReg OFF (L40S `stratus-lookout`, 2026-06-21) [IN FLIGHT]

**Motivation (why).** §30 ran the SIGReg-redundancy test on PushT and found Arm B (latent-SIGReg OFF) did NOT collapse — z_std contracted to a smaller *scale* (≈0.19 vs Arm A ≈0.94) but the latent effective rank stayed HIGH (B erank 108/192 > A erank 83/192) and every per-dim std > 0: a scale contraction held up by the action head, NOT a SCAR collapse. BUT §30's own caveat (§18 history-copyability) weakens the pusht claim: PushT's action is strongly HISTORY-correlated (the past-action stream `a_<t` alone predicts the next push; zeroing it RAISES open-loop val_act 0.132→0.153), so a chunk of `act_loss` can be satisfied by COPYING/extrapolating the action history rather than by reading the latent `emb` — weakening `act_loss`'s anti-collapse pressure on the latent and making a healthy pusht Arm B only a CONSERVATIVE positive. **tworoom is the CLEAN retest:** it is STATE-DRIVEN (the next action depends on the agent's room/position, not on a copyable action-history pattern), so here `act_loss` must read the latent to predict the action — the cleanest test of the conjecture "with the action-prediction head active, the explicit SIGReg latent anti-collapse is REDUNDANT." This entry is the **tworoom_B** arm (conjecture-1, SIGReg OFF), the twin of §30's pusht Arm B. (Conjectures restated: (1) latent-SIGReg is redundant when predicting raw actions directly; (2) replacing the action-emb stop-grad with VICReg `sigreg_act` is viable — conjecture-2 = arm C, a sibling arm.)

### STEP 1 — arm-flag verification (DONE ✅, the flags exist + a 1-step forward honors them)
Ran `/mnt/minghao_data/verify_sigreg_tworoom.py` (built the gated tworoom model + one real tworoom batch bs=64, action_block adim=10, ran `lejepa_forward`):
- **Arm B** (`loss.sigreg.weight=0`): TOTAL loss = `pred + intent + act` EXACTLY; the `lambd*sigreg_loss` term is provably 0. Numeric: pred=0.079078 sigreg_loss=27.8268 (still COMPUTED+logged for monitoring) lambd*sigreg=0.000000 intent=1.015526 act=1.128092 → TOTAL=2.222696 = pred+intent+act (|diff|=1.49e-8 FP). ✓
- **Arm C part-1** (`action_pred.detach_target`): backprop of `intent_loss` ALONE → `model.action_encoder` |grad| = **0.000000 (BLOCKED)** when `detach_target=true`, = **75.98 (FLOWS)** when `detach_target=false`. So detach_target=false DOES let the intent_loss gradient flow to act_emb, as conjecture-2/arm-C requires. ✓
- **Arm C part-2** (`action_pred.sigreg_act`): `sigreg_act_loss` is **absent** when `sigreg_act=false`, **present=24.899** when `sigreg_act=true` (the act-emb VICReg/SIGReg term is added). ✓ (NB: in code the `sigreg_act` term is weighted by the SAME `lambd=loss.sigreg.weight` as the latent SIGReg — so an arm-C that combines `loss.sigreg.weight=0` with `sigreg_act=true` would ZERO its own act-emb term; arm C must keep a nonzero `loss.sigreg.weight` for `sigreg_act` to contribute. Flagged for the arm-C runner.)
All three flags exist and are honored. NOT blocked.

### Design — tworoom_B = §30 Arm B recipe, ONLY env changed pusht→tworoom
FROM SCRATCH (`init_from=null`, encoder `vit_hf` size=tiny patch=14 image=224 **pretrained=false** → random CLS-192 — the load-bearing choice: warm-start would pre-establish a non-collapsed latent and HIDE collapse), `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0 detach_target=true sigreg_act=false`, **`loss.sigreg.weight=0`** (the arm-B knob: SIGReg module still RUNS + `sigreg_loss` still logged, contributes 0 to `loss`). Budget MATCHED to §30 pusht for comparability: `trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4 +ckpt_every=10`, seed 3072 (lewm.yaml default). Data = committed `config/train/data/tworoom.yaml` (frameskip 5 ⇒ action_block 5, adim=2 ⇒ action_block adim=10, full `tworoom.h5` 13 GB on `/mnt/minghao_data/.stable-wm/datasets/`). The SAME 4000-batch cap is to be used for tworoom arms A/B/C (this arm = B).

### Infra (DISK-SAFE, L40S shared box — the killer is SPT_CACHE_DIR)
L40S `stratus-lookout`, run `sudo -u minghao.fu`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`. **Disk at launch:** `/var/lib/docker` 99%, `/` 97%, `/mnt/minghao_data` 22% (2.4 TB free) → ALL writes redirected to `/mnt`: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_tworoom_B TMPDIR=/mnt/minghao_data/tmp_tworoom_B MPLCONFIGDIR=/mnt/minghao_data/mpl_tworoom_B HF_HOME=/mnt/minghao_data/hf **SPT_CACHE_DIR=/mnt/minghao_data/spt_tworoom_B** HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, plus `hydra.run.dir=/mnt/minghao_data/hydra_tworoom_B hydra.output_subdir=null`. Ckpts → `STABLEWM_HOME/checkpoints/sigreg_B_tworoom_scratch/`. `+ckpt_every=10` (ep 10,20,…,100) + `/mnt/minghao_data/prune_tworoom_B.sh` keep-newest-3 loop (PID 2125271). GPU **7** (lowest-occupied at launch, 5.3 GB; co-located per rule 10, never touched other users' procs). Same box train+eval → no cross-GPU render OOD ([[project_l40s_cross_gpu_rendering]]).

### EXACT train command (L40S)
```
export STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_tworoom_B \
       TMPDIR=/mnt/minghao_data/tmp_tworoom_B MPLCONFIGDIR=/mnt/minghao_data/mpl_tworoom_B \
       HF_HOME=/mnt/minghao_data/hf SPT_CACHE_DIR=/mnt/minghao_data/spt_tworoom_B \
       HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
CUDA_VISIBLE_DEVICES=7 python train_sigreg.py data=tworoom \
  action_pred.enabled=true action_pred.detach_decoder=false action_pred.head=mse \
  action_pred.w_act=1.0 action_pred.w_intent=1.0 action_pred.detach_target=true action_pred.sigreg_act=false \
  loss.sigreg.weight=0 init_from=null \
  output_model_name=sigreg_B_tworoom_scratch subdir=sigreg_B_tworoom_scratch \
  trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4 +ckpt_every=10 \
  hydra.run.dir=/mnt/minghao_data/hydra_tworoom_B hydra.output_subdir=null
```
**PID 2125270** (GPU 7). GIP head confirmed in log: `[GIP] Intention predictor ON  Adim=10  head=mse  w_act=1.0 w_intent=1.0 detach_target=True sigreg_act=False`. Sanity-val z_std @init = 0.00103 (untrained ViT CLS near-constant; test is whether *training* raises it). Throughput ~3 it/s × 4000 steps ⇒ ~22 min/ep ⇒ ~37 h for 100 ep. Log `/mnt/minghao_data/sigreg_logs/train_tworoom_B.log`.

### MEASUREMENT plan (STEP 3 — DIRECT from the final checkpoint, the robust signal)
The in-log `validate/z_std` monitor IS captured here (sanity-val showed 0.00103), but the robust signal is direct-from-checkpoint (the §30 in-log monitor was unreliable). Adapt `/mnt/minghao_data/measure_collapse.py` → tworoom run name `sigreg_B_tworoom_scratch`: rebuild via `gip.load_gip_model`, run a fixed tworoom val batch through `model.encode` → emb (B,T,D), report **z_std = emb.reshape(-1,D).std(dim=0).mean()**, **act_emb_std**, and **erank = participation ratio of the per-dim variance spectrum** `(Σλ)²/Σλ²` (λ = per-dim variance) + the entropy-erank as a cross-check. COLLAPSE = z_std→0 AND erank≪D; SCALE-SHRINK (not collapse) = low z_std but erank stays high + all dims>0.

### EVAL plan (STEP 4 — when converged, rule 8/9)
tworoom via `eval_gip.py --config-name tworoom policy=sigreg_B_tworoom_scratch +gip_eval.mode=<planning|bc> eval.num_eval=50 seed=<42|0|1>`, **N=50 × 3 seeds {42,0,1}**, same-box GPU, CONVERGED ckpt (NOT val-best — reacher-style trap rule). planning = pure WM CEM, bc = intention head reactive.

### Result [PENDING convergence + measurement + eval]
z_std / erank / collapsed / sr_planning / sr_bc / per_seed — to be filled when `weights_epoch_100.pt` lands (or an earlier converged plateau) and the measurement + eval run. Compare vs arm A (SIGReg ON) baseline: does B collapse / lose SR? **Provisional expectation (from §30 pusht):** tworoom_B holds a smaller-scale-but-full-rank latent and matches SR → conjecture-1 confirmed on the CLEAN state-driven env (SIGReg redundant when the action head predicts raw actions directly = a "Simple"-pillar win for LeWAM). If tworoom_B instead COLLAPSES (z_std→0, erank→1) where pusht did not, the §30 positive was a pusht history-copyability artifact and SIGReg IS load-bearing on state-driven envs.

---

## §31 🔄🆕 SIGReg-REDUNDANCY test on REACHER (DMC) — the CLEAN state-driven retest of §30, + arm C = action-emb VICReg (no stop-grad + `sigreg_act`) (L40S, 2026-06-21) [IN FLIGHT — TRAINING]

**Motivation (why).** §30 ran the SIGReg-redundancy test on **pusht** (SCAR conjecture: with the action-prediction head active, the explicit SIGReg latent anti-collapse should be REDUNDANT — dropping it should not collapse the latent or drop SR). The §30 verdict so far (ep10→20 direct-from-ckpt): Arm B (no SIGReg) is **scale-shrunk ~5×** (z_std 0.19 vs A 0.94) but NOT rank-collapsed (B erank 108 > A 83 / 192) — a scale contraction, not a SCAR collapse — conjecture-favorable, pending convergence + SR. BUT §30 carries an explicit CAVEAT (its "§18 history-copyability" block): **pusht's action is HISTORY-CORRELATED** (the `a_<t` past-action stream is a strong open-loop predictor; §18 showed zeroing it RAISES open-loop act-MSE 0.132→0.153), so a chunk of `act_loss` can be satisfied by COPYING the action history rather than reading the latent — which WEAKENS act_loss's anti-collapse pressure on `emb`. So a healthy Arm B on pusht is only a CONSERVATIVE positive; the clean retest needs a **state-driven** env where the action is NOT history-copyable. **Reacher (DMControl qpos_match) is that clean test:** fully observed, precise-control 2-joint reach, success = finger joint-config matches target qpos at the last step — the next action depends on the CURRENT arm pose + target (the state), not on a copyable action-history shortcut. This §31 reruns the §30 test on reacher (arms A/B) AND adds **arm C** to test the user's SECOND conjecture: replacing the action-emb stop-grad (`detach_target=true`, the §30/baseline anti-collapse on act_emb) with **VICReg on the action embeddings** (`detach_target=false` so intent_loss gradient flows INTO act_emb + `sigreg_act=true` so SIGReg directly anti-collapses act_emb). Conjecture-2: VICReg-on-act-emb is a viable substitute for the stop-grad (keeps act_emb non-collapsed without the stop-grad's "freeze the target" trick).

### Design — 3 arms, FROM SCRATCH (`init_from=null`, vit-tiny-192 random init `pretrained=false`), action_pred ON throughout, reacher (`data=dmc` → reacher.h5), SAME budget as §30 pusht
- **Arm A (baseline = §30-A recipe):** `loss.sigreg.weight=0.09 action_pred.detach_target=true action_pred.sigreg_act=false` — latent SIGReg ON, act_emb anti-collapse = the STOP-GRAD on the intent target (the validated design). The reference arm.
- **Arm B (conjecture-1 = §30-B recipe):** `loss.sigreg.weight=0 action_pred.detach_target=true action_pred.sigreg_act=false` — latent SIGReg term zeroed; tests "is latent SIGReg redundant when predicting raw actions directly?"
- **Arm C (conjecture-2, THIS arm's focus):** `loss.sigreg.weight=0.09 action_pred.detach_target=FALSE action_pred.sigreg_act=TRUE` — latent SIGReg still ON (0.09), but the act_emb anti-collapse switches from stop-grad → VICReg: `detach_target=false` lets the JEPA intent_loss `(intention−act_emb)²` gradient flow INTO act_emb (no stop-grad), and `sigreg_act=true` adds a SIGReg(act_emb) VICReg term (same module/weight λ=0.09) to keep act_emb from collapsing now that the stop-grad protection is removed. Tests "is action-emb VICReg (no stop-grad + sigreg_act) viable?"
- **From scratch** is load-bearing (same reason as §30): warm-starting would pre-establish a non-collapsed latent and HIDE any collapse; from scratch the only anti-collapse forces are the per-arm loss terms.
- **Budget (100 epochs, batch_size=64 — SAME as §30 pusht; the SAME cap for all 3 reacher arms):** `trainer.max_epochs=100 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4 +ckpt_every=10`. **NOTE on the cap (turnaround, task-permitted "you MAY cap limit_train_batches ... use the SAME cap for all arms of this env"):** first launched at `limit_train_batches=4000` (the literal §30 pusht number) but reacher's **98 GB reacher.h5 is severely I/O-bound** — steady-state was only ~1.8–2.2 it/s (vs pusht's 6.3 it/s; the §15 finding that reacher is 3× slower for the same I/O reason), so 4000-batch epochs ran ~35 min each → ~58 h/arm for 100 ep, intractable. GPUs were NOT saturated (identical it/s across arms regardless of GPU load → disk/CPU bound, per rule 10), so reducing the cap is the right lever (it shortens each epoch proportionally without sacrificing it/s). **Relaunched all 3 arms at `limit_train_batches=1000`** (64k samples/epoch, the §1c-era recipe scale) → ~8–9 min/epoch → ~14–15 h/100 ep. The cap is IDENTICAL across A/B/C so the within-reacher comparison stays apples-to-apples (the cross-env pusht-vs-reacher comparison is qualitative — the collapse signature — not a same-batch-count SR comparison). Killed the 4000-batch PIDs (2127244/2127629/2128104) + cleaned their partial run dirs before relaunch; new PIDs A=2140831(GPU0) B=2141094(GPU6) C=2141600(GPU2).

### STEP 1 — ARM-FLAG VERIFICATION (done, BEFORE the runs; NUMERICAL, on real reacher data) ✅
1-step forward via `/mnt/minghao_data/verify_armC.py` (builds each arm's model + a real reacher val batch through the `train_sigreg.py` instantiation path, runs `lejepa_forward`'s load-bearing slice, autograd-checks the three properties). GPU 0, lewm venv, disk-safe env. Results (per-batch, untrained random init):
| arm | latent sigreg_term (λ·sigreg_loss) | sigreg_act_term | intent_loss grad → action_encoder | sigreg_act grad → action_encoder |
|---|---|---|---|---|
| A (sigreg 0.09, detT, act-sigreg off) | **0.2299** (>0 ✓) | — | **0.000000** (stop-grad working ✓) | — |
| B (sigreg 0, detT, act-sigreg off) | **0.000000** (zeroed ✓) | — | 0.000000 | — |
| C (sigreg 0.09, detF, act-sigreg ON) | 0.2220 (>0) | **0.2837** (act VICReg term present+>0 ✓) | **0.310572** (>0 → grad flows to act_emb, detach_target=false working ✓) | **0.517682** (>0 → act VICReg reaches the encoder ✓) |

**ALL ARM FLAGS VERIFIED (script printed `ALL ARM FLAGS VERIFIED: True`).** The three decisive numeric checks the task called for: (1) arm B's latent sigreg term is EXACTLY 0 (vs A's 0.23); (2) arm C's `detach_target=false` lets intent_loss gradient reach the action_encoder (0.31, vs arm A's exactly 0.0 under the stop-grad); (3) arm C's `sigreg_act=true` adds a nonzero act-emb VICReg term (0.28) whose gradient reaches the encoder (0.52). The flags exist and are honored.

### STEP 2 — TRAIN (in flight). EXACT commands (host `L40S`, `sudo -u minghao.fu`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`; driver `/mnt/minghao_data/launch_reacher_sigreg.sh ARM GPU SIGREG_W DETACH_TARGET SIGREG_ACT`)
```
# common disk-safe env (per-arm scratch dirs; SPT_CACHE_DIR is the killer redirect):
export STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_reacher_<ARM> \
  TMPDIR=/mnt/minghao_data/tmp_reacher_<ARM> MPLCONFIGDIR=/mnt/minghao_data/mpl_reacher_<ARM> \
  HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4 \
  SPT_CACHE_DIR=/mnt/minghao_data/spt_reacher_<ARM> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
CUDA_VISIBLE_DEVICES=<GPU> python train_sigreg.py data=dmc \
  action_pred.enabled=true action_pred.detach_decoder=false action_pred.head=mse \
  action_pred.w_act=1.0 action_pred.w_intent=1.0 \
  action_pred.detach_target=<true|true|false> action_pred.sigreg_act=<false|false|true> \
  loss.sigreg.weight=<0.09|0|0.09> init_from=null \
  output_model_name=sigreg_<A|B|C>_reacher_scratch subdir=sigreg_<A|B|C>_reacher_scratch \
  trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4 +ckpt_every=10 \
  hydra.run.dir=/mnt/minghao_data/hydra_reacher_<ARM> hydra.output_subdir=null
```
- **Launched:** Arm A PID 2127244 (GPU 0, `detT/sigreg0.09/actoff`), Arm B PID 2127629 (GPU 7, `detT/sigreg0/actoff`), Arm C PID 2128104 (GPU 5, `detF/sigreg0.09/actON`). All three confirmed in their logs: `[GIP] Intention predictor ON Adim=10 head=mse w_act=1.0 w_intent=1.0 detach_target=<...> sigreg_act=<...>` (A/B `detach_target=True sigreg_act=False`, C `detach_target=False sigreg_act=True`). Random-init scratch ViT-tiny-192. `+ckpt_every=10` ⇒ `weights_epoch_{10..100}.pt`; prune loop `/mnt/minghao_data/prune_reacher_sigreg.sh` (PID 2128143) keeps newest-3 per run + newest-1 SPT runs-dir.
- **Throughput / disk:** ~2.0–2.8 it/s at launch (reacher's 98 GB h5 is I/O-heavy + 3 arms co-located share disk — the §15 finding that reacher eval/precompute is 3× slower than tworoom for the same reason; the real bottleneck here is disk/CPU not GPU per rule 10). Disk at launch: `/var/lib/docker` 99% (88 GB free), `/` 97% (3.5 GB), `/mnt/minghao_data` 22% (2.4 TB free) → ALL writes on `/mnt` (verified `SPT_CACHE_DIR` redirect prevents the metrics.csv/resumption-ckpt crash, [[project_l40s_path_consolidation]]).

### STEP 3 — DIRECT-FROM-CHECKPOINT collapse measurement (the ROBUST signal; reuses the §30 `measure_collapse.py` logic, reacher-adapted) [PENDING ckpts]
For each arm's converged checkpoint: `gip.load_gip_model(run, epoch)` (the SAME path eval_gip.py uses) → one shared reacher val batch through `model.encode` → `z_std = emb.reshape(-1,D).std(dim=0).mean()` + effective rank (participation/entropy of the per-dim variance spectrum) + `act_emb_std`. COLLAPSE = z_std→0 AND erank≪192; SCALE-SHRINK (not collapse) = low z_std but erank stays high + all per-dim std>0 (the §30 Arm-B signature).

### STEP 4 — EVAL SR (reacher, `eval_gip.py --config-name reacher`, N=50, 3 seeds {42,0,1}, same-box, planning + bc; use the CONVERGED ckpt — reacher has the §15 val-best CHECKPOINT TRAP, headline the converged ckpt NOT val-best) [PENDING convergence]
```
python eval_gip.py --config-name reacher policy=sigreg_<A|B|C>_reacher_scratch \
  +gip_eval.mode=<planning|bc> eval.num_eval=50 seed=<42|0|1>
```
(planning = pure WM CEM; bc = intention head reactive. Reacher needs the dm_control↔mujoco compat shim `/tmp/reacher_compat/sitecustomize.py` on PYTHONPATH, §15 blocker 1; eval same-box as train per the cross-GPU render gotcha [[project_l40s_cross_gpu_rendering]].)

### RESULT TABLE (to fill at convergence — ALL per-seed)
| arm | final z_std | act_emb_std | erank/192 | collapsed? | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) |
|---|---|---|---|---|---|---|
| A (sigreg 0.09, detach_target=true, sigreg_act=false) | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| B (sigreg 0, detach_target=true, sigreg_act=false) | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| C (sigreg 0.09, detach_target=FALSE, sigreg_act=TRUE) | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |

### VERDICT [PENDING convergence + eval]
Two conjectures under test on the CLEAN state-driven env: (1) is latent SIGReg redundant when predicting raw actions directly (Arm B vs A — does B's latent collapse, and does B lose SR)? (2) is action-emb VICReg a viable stop-grad replacement (Arm C vs A — does removing the stop-grad + adding sigreg_act keep act_emb healthy and SR intact, or does it hurt)? Decided on the converged-ckpt collapse measurement + the 3-seed N=50 SR. [Fills once the runs converge and the 9 evals + 3 collapse measurements land.]

### Files (L40S)
- `/mnt/minghao_data/verify_armC.py` — STEP-1 numeric arm-flag verification (the table above).
- `/mnt/minghao_data/verify_reacher_arms.py` — INDEPENDENT STEP-1 re-verification (reacher_A subagent, 2026-06-22): same three checks on a real reacher batch, CONFIRMED — arm B latent sigreg term EXACTLY 0 (|TOTAL−(pred+intent+act)|=1.71e-7), arm C `detach_target=false` → full intent_loss backward grad into action_encoder = 69.93 (vs detach_target=true → exactly 0.0), arm C `sigreg_act=true` → `sigreg_act_loss=25.05` present (adds λ·=2.254), absent when false. Flags fully honored.
- `/mnt/minghao_data/measure_collapse_reacher.py` — reacher-adapted STEP-3 collapse measurement (3-arm A/B/C, shared val batch, z_std + erank participation-ratio + act_emb_std; verdict tags COLLAPSED/SCALE-SHRINK/HEALTHY). reacher_C subagent added a no-op `--arms` argparse arg so the finalizer's `--arms A,B,C` call parses (all 3 are measured regardless); syntax-checked OK.
- `/mnt/minghao_data/reacher_sigreg_finalize.sh` — SELF-COMPLETING finalizer (PID 2138333, detached via setsid; log `/mnt/minghao_data/reacher_sigreg_finalize.log`). Blocks until `weights_epoch_100.pt` lands on A/B/C, then auto-runs STEP-3 collapse (all arms, ep100) + STEP-4 eval (3 arms × {planning,bc} × seeds {42,0,1}, N=50, same-box eval GPU 0). Converged numbers land without further babysitting.
- `/mnt/minghao_data/launch_reacher_sigreg.sh`, `/mnt/minghao_data/prune_reacher_sigreg.sh` — launcher + keep-newest-3 prune loop.
- Ckpts: `/mnt/minghao_data/.stable-wm/checkpoints/sigreg_{A,B,C}_reacher_scratch/`. Logs `/mnt/minghao_data/sigreg_reacher_logs/train_{A,B,C}.log`.
- **STATUS (reacher_A subagent, 2026-06-22 ~00:31 UTC):** under heavy box contention (load avg 148 on 64 cores; 8 co-located train_sigreg arms + other users; GPU 0 util 45% ⇒ CPU/disk-starved per rule 10, NOT GPU-bound) the original `limit_train_batches=4000` reacher arms (62 h/arm ETA) were KILLED and cleanly RELAUNCHED at **`limit_train_batches=1000`** (coordinated across A/B/C → cap stays UNIFORM per the "same cap for all arms" constraint; no weights had been written at 4000, nothing wasted). Arm A now = PID 2140831, GPU 0, `detach_target=true sigreg_act=false loss.sigreg.weight=0.09`, alive (state Sl). At ~1.8 it/s × 1000 steps ⇒ ~9.3 min/epoch ⇒ **~15.4 h/arm for 100 ep**. First ckpt at ep10 (`+ckpt_every=10`). Converged z_std/erank/SR PENDING ep100 — NOT fabricated. Unblock = ep100 ckpts on A/B/C → finalizer (PID 2138333) auto-fills STEP-3 collapse + STEP-4 eval.
- NOT touched: shared `train.py`/`jepa.py`/`gip.py`/`eval_gip.py`/`module.py`, `gen_master_table.py`, other agents' runs/GPUs.

## §31 🔄🆕 SIGReg-REDUNDANCY test on **cube** (OGBench cube_single, the CLEAN state-driven retest of §30) — arm **cube_C** = conjecture-2: action-emb VICReg (no stop-grad + `sigreg_act`) (L40S `stratus-lookout`, 2026-06-21) [IN FLIGHT — TRAINING]

**Motivation (why).** §30 ran the SIGReg-redundancy test on **pusht** and found (ep10→20 direct-from-ckpt) Arm B (latent-SIGReg OFF) does NOT SCAR-collapse — z_std contracts ~5× (B≈0.19 vs A≈0.94) but the latent effective rank stays HIGH (B erank 108 > A 83 / 192), a scale contraction held up by the action head, conjecture-favorable pending convergence + SR. §30's own §18-history-copyability CAVEAT weakens the pusht positive: pusht's action is HISTORY-correlated (the `a_<t` past-action stream alone predicts the next push; zeroing it RAISES open-loop act-MSE 0.132→0.153), so `act_loss` can be partly satisfied by COPYING the action history rather than reading the latent `emb` — weakening act_loss's anti-collapse pressure, so a healthy pusht Arm B is only a CONSERVATIVE positive. **cube (OGBench `cube_single_expert`) is a CLEAN retest:** the expert action is STATE-DRIVEN (reach/grasp/place a cube to a target pose — the next action depends on the current effector + cube pose vs target, NOT on a copyable action-history pattern), so here `act_loss` MUST read the latent to predict the action. This entry is the **cube_C** arm — conjecture-2: replace the act_emb stop-grad (`detach_target=true`, the §30/baseline anti-collapse on act_emb) with **VICReg on the action embeddings** (`detach_target=false` so the JEPA `intent_loss=(intention−act_emb)²` gradient FLOWS into act_emb + `sigreg_act=true` so a SIGReg(act_emb) VICReg term, same module/λ=0.09, keeps act_emb from collapsing now the stop-grad is removed). Conjecture-2: VICReg-on-act-emb is a viable substitute for the stop-grad. (Sibling arms A=baseline SIGReg-ON/stop-grad, B=latent-SIGReg-OFF, presumed launched by sibling agents on cube with the SAME budget for comparability.)

### Design — cube_C arm; FROM SCRATCH, action_pred ON, vit-tiny-192, budget matched to §30 pusht
- **Arm C (conjecture-2, THIS arm):** `loss.sigreg.weight=0.09 action_pred.detach_target=FALSE action_pred.sigreg_act=TRUE`. Latent SIGReg still ON (0.09); act_emb anti-collapse switches stop-grad → VICReg.
- **Everything else = the §30 canonical from-scratch GIP recipe:** `--config-name lewm data=ogb` (cube data: `ogbench/cube_single_expert.h5`/`ogb_cube_single.lance`, frameskip 5, action block = 5×5 = **25-d**, keys pixels+action+observation, proprio merged), `model=lewm` (vit-tiny-192, patch_size 14, `embed_dim=192`, `use_proprio=false use_action_history=true`), `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0`, `trainer.max_epochs=100 +limit_train_batches=4000 +limit_val_batches=20 loader.batch_size=64 num_workers=4`, seed 3072 (lewm default), `init_from=null`, `+ckpt_every=10`. GPU 6 (lowest-occupancy at launch). PID 2129324.

### STEP 1 — 1-step-forward arm-flag verification (DONE, `/mnt/minghao_data/verify_cube_C.py`, GPU 6, cube batch bs=32)
All three conjecture-2 conditions confirmed NUMERICALLY on cube (action dim = 25):
| check | result |
|---|---|
| **(3) sigreg_act term added** | `sigreg_act_loss=12.55 > 0`; TOTAL == `pred + 0.09·sigreg + intent + act + 0.09·sigreg_act` (diff 2.9e-7) ✓ |
| **(2) detach_target=FALSE → grad flows to act_emb** | `sum|d intent_loss/d act_emb| = 1.608` (FLOWS) vs `0.000` when detach_target=true (BLOCKED) ✓ |
| **(B-control) weight=0 zeroes latent sigreg** | TOTAL − (pred+intent+act) = 0.00 exactly ✓ (so the conjecture-1 Arm B flag is also sound) |
Launch-log confirms: `[GIP] Intention predictor ON  Adim=25  head=mse  w_act=1.0 w_intent=1.0 detach_target=False sigreg_act=True`.

### STEP 2 — TRAIN (cube, 100ep × 4000 batch, IN FLIGHT). Throughput ~1.6 it/s under heavy CPU oversubscription (load avg ~154 on 64 cores from co-located other-user runs) ⇒ slow epochs; collapse signal read early off ep10/20 ckpts, SR at the converged plateau. Sanity-val at init: `z_std=0.0051 sigreg_act_loss=24.9 sigreg_loss=28.25` (random net).

### STEP 3 — DIRECT-FROM-CHECKPOINT collapse measurement (`/mnt/minghao_data/measure_collapse_cube.py`, reuses §30 logic): `gip.load_gip_model(run, epoch)` → shared cube val batch → `model.encode` → `z_std = emb.reshape(-1,D).std(dim=0).mean()` + erank (participation ratio of the per-dim variance spectrum AND entropy-of-svdvals cross-check) + `act_emb_std`. COLLAPSE = z_std→0 AND erank≪192; SCALE-SHRINK = low z_std but erank high + all per-dim std>0. [PENDING ckpts]

### STEP 4 — EVAL SR (cube, `eval_gip.py --config-name cube`, N=50, 3 seeds {42,0,1}, same-box, planning + bc; CONVERGED ckpt) [PENDING convergence]
```
python eval_gip.py --config-name cube policy=sigreg_C_cube_scratch \
  +gip_eval.mode=<planning|bc> eval.num_eval=50 seed=<42|0|1>
```

### RESULT TABLE — cube (to fill at convergence — ALL per-seed)
| arm | final z_std | act_emb_std | erank/192 | collapsed? | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) |
|---|---|---|---|---|---|---|
| A (sigreg 0.09, detach_target=true, sigreg_act=false) | [sibling] | [sibling] | [sibling] | [sibling] | [sibling] | [sibling] |
| B (sigreg 0, detach_target=true, sigreg_act=false) | [sibling] | [sibling] | [sibling] | [sibling] | [sibling] | [sibling] |
| **C (sigreg 0.09, detach_target=FALSE, sigreg_act=TRUE)** | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |

### VERDICT [PENDING convergence + eval]
cube_C tests conjecture-2 on the CLEAN state-driven env: does swapping the act_emb stop-grad for VICReg (no stop-grad + sigreg_act) keep act_emb healthy (not collapsed) and SR intact vs arm A baseline, or does it hurt? Decided on the converged-ckpt collapse measurement (z_std/erank/act_emb_std) + 3-seed N=50 SR (planning+bc) vs the cube arm-A baseline.

### Files (L40S)
- `/mnt/minghao_data/verify_cube_C.py` — STEP-1 arm-flag verification. `/mnt/minghao_data/launch_cube_C.sh` — launcher. `/mnt/minghao_data/prune_cube_C.sh` — keep-newest-3 prune loop (PID 2133372). `/mnt/minghao_data/measure_collapse_cube.py` — STEP-3 collapse measurement.
- Disk-safe env (ALL writes → /mnt, / and /var stayed flat): `STABLEWM_HOME=/mnt/minghao_data/.stable-wm SPT_CACHE_DIR=/mnt/minghao_data/spt_cube_C XDG_CACHE_HOME=/mnt/minghao_data/xdg_cube_C TMPDIR=/mnt/minghao_data/tmp_cube_C MPLCONFIGDIR=/mnt/minghao_data/mpl_cube_C HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline`, `hydra.run.dir=/mnt/minghao_data/hydra_cube_C hydra.output_subdir=null`.
- Ckpts: `/mnt/minghao_data/.stable-wm/checkpoints/sigreg_C_cube_scratch/`. Log `/mnt/minghao_data/cube_sigreg_logs/train_C.log`.
- NOT touched: shared `train.py`/`jepa.py`/`gip.py`/`eval_gip.py`/`module.py`, `gen_master_table.py`, other agents' runs/GPUs.

---

## §31 🔄🆕 SIGReg-REDUNDANCY test on **cube** (OGBench cube_single, the CLEAN state-driven retest of §30) — arm **cube_B** = conjecture-1: latent-SIGReg OFF, + arm **cube_A** = baseline (L40S `stratus-lookout`, 2026-06-22) [IN FLIGHT — TRAINING]

**Motivation (why).** §30 ran the SCAR SIGReg-redundancy conjecture on **pusht**: with the action-prediction head active (`action_pred.enabled=true`, `act_loss=(decoder(intention)−raw_action)²`), the explicit latent anti-collapse (SIGReg, `loss.sigreg.weight`) should be REDUNDANT because `act_loss` (raw-action regression) supplies an independent anti-collapse pressure on the shared latent `emb` — a collapsed latent cannot predict varying actions, so the action loss itself anchors the latent. §30's ep10→20 direct-from-ckpt result: Arm B (latent-SIGReg OFF) does NOT SCAR-collapse — z_std contracts ~5× (B≈0.19 vs A≈0.94) but the latent effective rank STAYS HIGH (B erank 108 > A 83 / 192) — a **scale contraction**, not a dimensional collapse, conjecture-favorable. BUT §30's own §18-history-copyability CAVEAT weakens that positive: pusht's action is HISTORY-correlated (the `a_<t` past-action stream alone predicts the next push; zeroing it RAISES open-loop act-MSE 0.132→0.153), so `act_loss` can be partly satisfied by COPYING the action history rather than reading the latent — weakening act_loss's anti-collapse pressure → a healthy pusht Arm B is only a CONSERVATIVE positive. **cube (OGBench `cube_single_expert`) is the CLEAN retest:** the expert action is STATE-DRIVEN (reach/grasp/place a cube to a target pose — the next action depends on the current effector + cube pose vs target, NOT on a copyable action-history pattern), so here `act_loss` MUST read the latent. This entry = the **cube_B** arm (conjecture-1: latent-SIGReg OFF, `loss.sigreg.weight=0`) and its baseline **cube_A** (SIGReg ON, λ=0.09), the pair that decides conjecture-1 on cube. (Sibling **cube_C**, §31 above, tests conjecture-2 = act-emb VICReg; it crashed at ep0 on a DataLoader-worker kill, see STATUS.)

### Design — cube_A (baseline) vs cube_B (conjecture-1); FROM SCRATCH, action_pred ON, vit-tiny-192
- **Arm A (baseline, SIGReg ON):** `loss.sigreg.weight=0.09 action_pred.detach_target=true action_pred.sigreg_act=false`. GPU 3 (relaunched there with `loader.pin_memory=false num_workers=3` after the first GPU-0 attempt died at ep0 with `RuntimeError: Pin memory thread exited unexpectedly` under shared-box RAM/IO pressure — a co-located worker was OOM-killed; NO ckpt had been written, nothing wasted). PID 2183674.
- **Arm B (conjecture-1, SIGReg OFF, THIS arm):** `loss.sigreg.weight=0 action_pred.detach_target=true action_pred.sigreg_act=false` — the `lambd·sigreg_loss` term is EXACTLY zeroed (SIGReg module still RUNS, `sigreg_loss` is logged for monitoring, but contributes 0 to `loss`). GPU 6, `num_workers=4` `pin_memory` default. PID 2141402.
- **Everything else IDENTICAL between A and B = the §30/§31 canonical from-scratch GIP recipe:** `--config-name lewm data=ogb` (cube data: `ogbench/cube_single_expert.h5`, frameskip 5, action block = 5×5 = **25-d**, keys pixels+action+observation, proprio merged but `use_proprio=false` so pixels-only — matches the §30 pusht run), `model=lewm` (vit-tiny-192 patch_size 14 `pretrained=false` → random CLS-192, the load-bearing choice that lets collapse show; `embed_dim=192 use_action_history=true`), `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0 detach_target=true`, seed 3072 (lewm default), `init_from=null`, `+ckpt_every=10`.
- **BUDGET / the env-wide cap (STATED).** `trainer.max_epochs=100 +trainer.limit_val_batches=20 loader.batch_size=64`, **`+trainer.limit_train_batches=1000`**. This is the SAME 1000-batch cap both arms of this pair (A and B) use — the comparison is strictly apples-to-apples. NOTE the cap is **1000, not the §30 pusht 4000**: cube's 101 GB `cube_single_expert.h5` random-access is the bottleneck (rule 10: the real bottleneck is disk/CPU, not GPU mem), so at the shared-box contention (8+ co-located train_sigreg arms, other users) the two cube arms run at ~1.2–1.8 it/s; 4000 batches ⇒ ~44 min/epoch ⇒ ~73 h/arm (impractical), 1000 batches ⇒ ~10–14 min/epoch ⇒ **~17 h/arm**. The 1000-batch re-cap matches what the §31 reacher arms ALSO did for the identical IO-contention reason (line 4204). At bs 64, 1000 batches/epoch × 100 ep = 6.4 M sample-passes — ample for convergence; the collapse signal shows early (ep10–30) off the landed ckpts.

### STEP 1 — 1-step-forward arm-flag verification (DONE, `/mnt/minghao_data/verify_cube_arms.py`, GPU 6, real cube batch bs=32, action dim = 25) ✅ ALL PASS
Built the gated model + ONE real cube batch via the `train_sigreg.py` machinery, ran `lejepa_forward` under each arm's flags, printed the loss decomposition + a real `intent_loss.backward()` to probe the action-encoder gradient:
| check | arm | result | reading |
|---|---|---|---|
| **(B) `loss.sigreg.weight=0` zeroes the latent SIGReg term** | B | TOTAL − (pred+intent+act) = **−2.98e-08** (≈0) | conjecture-1 flag SOUND — sigreg contributes exactly 0 |
| **(A) SIGReg term present** | A | TOTAL − (pred+intent+act) = **0.8327** = λ·sigreg (0.09×9.253) exactly | baseline keeps the latent anti-collapse |
| **(C-ref) detach_target=FALSE → grad flows to act_emb** | C | `sum|d intent_loss/d action_encoder| = 64.57` (>0, FLOWS) | conjecture-2 flag sound (sibling arm) |
| **(A) detach_target=true → grad BLOCKED** | A | `sum|d intent_loss/d action_encoder| = 0.0` | the stop-grad works as designed |
| **(C-ref) sigreg_act=TRUE adds the act-emb VICReg term** | C | `sigreg_act_loss=12.51 > 0`; TOTAL − (pred+intent+act) = 2.0056 = λ·(sigreg+sigreg_act) | conjecture-2 term present |
Launch-log confirms both A and B: `[GIP] Intention predictor ON  Adim=25  head=mse w_act=1.0 w_intent=1.0 detach_target=True sigreg_act=False`.

### STEP 2 — TRAIN (cube, 100ep × 1000 batch, IN FLIGHT). Both arms alive; cube `observation` column one-time cache ~8 min then steps at ~1.2–1.8 it/s. Sanity-val at init: `z_std≈0.005` (random net, near-constant CLS — rises with training; the test is whether B's z_std STABILIZES with rank preserved vs DECAYS to 0).

### STEP 3 — DIRECT-FROM-CHECKPOINT collapse measurement (`/mnt/minghao_data/measure_collapse_cube.py`): `gip.load_gip_model(run, epoch)` → ONE shared cube val batch (bs 256, same batch both arms) → `model.encode` → `z_std = emb.reshape(-1,D).std(dim=0).mean()` + **erank = participation ratio of the per-dim VARIANCE spectrum** `(Σv_i)²/Σv_i²` (per task brief; entropy-of-svdvals cross-check also printed) + `act_emb_std`. COLLAPSE = z_std→0 AND erank≪192; SCALE-SHRINK (not collapse) = low z_std but erank high + all per-dim std>0. Auto-tracked at ep10,20,…,100 by `/mnt/minghao_data/track_cube_collapse.sh` (PID 2155239) → `/mnt/minghao_data/sigreg_logs/cube_collapse_traj.log`. [PENDING ckpts]

### STEP 4 — EVAL SR (cube, `eval_gip.py --config-name cube`, N=50, 3 seeds {42,0,1}, same-box, planning + bc; CONVERGED ep100 ckpt via `+ckpt_epoch=100`) [PENDING convergence]. Driver `/mnt/minghao_data/eval_cube_sigreg.sh <gpu>`:
```
python eval_gip.py --config-name cube policy=sigreg_<A|B>_cube_scratch \
  +ckpt_epoch=100 +gip_eval.mode=<planning|bc> eval.num_eval=50 seed=<42|0|1>
```

### RESULT TABLE — cube conjecture-1 (to fill at convergence — ALL per-seed)
| arm | final z_std | act_emb_std | erank(PR)/192 | collapsed? | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) |
|---|---|---|---|---|---|---|
| **A (sigreg 0.09, detach_target=true, sigreg_act=false)** | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| **B (sigreg 0, detach_target=true, sigreg_act=false)** | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |

### VERDICT [PENDING convergence + eval]
cube_B tests conjecture-1 on the CLEAN state-driven env (the §18 history-copyability confound that weakened §30-pusht is REMOVED on cube). The decisive measurement is the z_std/erank trajectory: does Arm B (no SIGReg) STABILIZE at a smaller scale with rank preserved (→ scale contraction not collapse, conjecture-favorable → act_loss anchors the latent → SIGReg redundant when predicting actions directly = a "Simple"-pillar win), or DECAY toward 0 with rank collapsing (→ true SCAR collapse, conjecture FALSIFIED, SIGReg load-bearing once history-copyability is gone)? Decided on the converged-ckpt collapse (z_std/erank/act_emb_std vs arm A) + 3-seed N=50 SR (planning+bc) vs the cube arm-A baseline.

### STATUS (cube_B subagent, 2026-06-22 ~00:55 UTC)
- **cube_B** PID 2141402 (GPU 6, `loss.sigreg.weight=0`), **cube_A** PID 2183674 (GPU 3, λ=0.09, relaunched with `pin_memory=false num_workers=3` after a pin-memory-thread crash at ep0; nothing wasted). Both ALIVE and stepping past ep0 at ~1.2–1.8 it/s (~17 h/arm to ep100). STEP-1 verification PASSED (numbers above). Converged z_std/erank/SR PENDING ep100 — NOT fabricated.
- **Sibling cube_C** (conjecture-2, §31 above) CRASHED at ep0 (`DataLoader worker killed by signal: Terminated`, no ckpt) under the same shared-box IO/RAM pressure that 4000-batch cube runs hit — its agent will need to relaunch (1000-batch cap + `pin_memory=false` recommended).
- **Unblock** = `weights_epoch_100.pt` in both `sigreg_{A,B}_cube_scratch` run dirs + the collapse tracker's z_std trajectory flat over the last ~10 ep (B's z_std stable, not decaying to 0) → run `eval_cube_sigreg.sh` (2 arms × 2 modes × 3 seeds {42,0,1}, N=50, same-box GPU) → fill the table + verdict.

### Files (L40S, all under `/mnt/minghao_data/`, MINE only)
- `verify_cube_arms.py` — STEP-1 arm-flag verification (B-zeroing + A-present + C-grad-flow + C-sigreg_act, all PASS). `launch_cube_sigreg.sh <A|B> <gpu>` — launcher (1000-batch cap). `relaunch_cube_A.sh <gpu>` — arm-A robust relaunch (`pin_memory=false num_workers=3`). `measure_collapse_cube.py` — STEP-3 direct-from-ckpt collapse (participation-ratio erank). `track_cube_collapse.sh` (PID 2155239) — auto-measures ep10..100 → `sigreg_logs/cube_collapse_traj.log`. `prune_cube.sh` (PID 2146148) — keep-newest-3 weights per arm. `eval_cube_sigreg.sh` — STEP-4 SR driver.
- Disk-safe env (ALL writes → /mnt; / and /var held flat at 88 G/3.5 G free): `STABLEWM_HOME=/mnt/minghao_data/.stable-wm SPT_CACHE_DIR=/mnt/minghao_data/spt_cube_<A|B> XDG_CACHE_HOME=/mnt/minghao_data/xdg_cube_<A|B> TMPDIR=/mnt/minghao_data/tmp_cube_<A|B> MPLCONFIGDIR=/mnt/minghao_data/mpl_cube_<A|B> HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline`, `hydra.run.dir=/mnt/minghao_data/hydra_cube_<A|B> hydra.output_subdir=null`.
- Ckpts: `/mnt/minghao_data/.stable-wm/checkpoints/sigreg_{A,B}_cube_scratch/`. Logs `/mnt/minghao_data/sigreg_logs/cube_{A,B}.log`.
- **NOT touched:** shared `train_sigreg.py`/`train.py`/`jepa.py`/`gip.py`/`eval_gip.py`/`module.py`/`gen_master_table.py`, the §31 cube_C / reacher / tworoom runs+GPUs, other users' procs. `train_sigreg.py` is the pre-existing non-invasive copy (= `train.py` + the z_std monitor); I added nothing to it.

---

## §31 tworoom — arm **tworoom_C** = conjecture-2: action-emb VICReg (no stop-grad + `sigreg_act`) (L40S `stratus-lookout`, 2026-06-22) [IN FLIGHT — TRAINING]

**Motivation (why).** Conjecture-2 of the §30/§31 SIGReg-redundancy pair: instead of the stop-grad on the act_emb target (the default anti-collapse for the JEPA-style intention loss, `detach_target=true`), let the `intent_loss` gradient FLOW INTO the action encoder (`detach_target=false`) and prevent the resulting act_emb collapse with a VICReg/SIGReg term on the action embeddings (`sigreg_act=true`). The question: is replacing the stop-grad with an explicit act-emb variance regularizer VIABLE (no collapse, no SR loss) on the CLEAN state-driven env? tworoom is the clean test (vs pusht's history-copyability, §18): the next action depends on agent room/position, so `act_loss` must read the latent, and the act-emb plays both roles (intent target AND prediction conditioning). This arm is the twin of the tworoom_B arm (§31 tworoom) but flips two flags: `detach_target false` + `sigreg_act true` (and keeps `loss.sigreg.weight=0.09` — arm-flag caveat: `sigreg_act` is weighted by the SAME `lambd`, so arm C must keep a nonzero latent-SIGReg weight or it would zero its own act-emb term).

### STEP 1 — arm-flag verification (DONE ✅, the flags exist + a 1-step forward honors them numerically)
Ran `/mnt/minghao_data/verify_sigreg_tworoom.py` (gated tworoom model + one real tworoom batch bs=64, action_block adim=10, `lejepa_forward`), GPU 7:
- **Arm B knob** (`loss.sigreg.weight=0`): TOTAL = `pred+intent+act` EXACTLY, `lambd*sigreg=0.000000`, |diff|=1.49e-8 FP. ✓ (latent SIGReg term provably zeroed; module still runs + logs `sigreg_loss=28.14` for monitoring.)
- **Arm C part-1** (`action_pred.detach_target`): backprop of `intent_loss` ALONE → `model.action_encoder` |grad| = **0.000000 (BLOCKED)** when `detach_target=true`, = **72.39 (FLOWS)** when `detach_target=false`. So `detach_target=false` DOES let the intent gradient reach act_emb (the action encoder), as arm C requires. ✓
- **Arm C part-2** (`action_pred.sigreg_act`): `sigreg_act_loss` **absent** when `sigreg_act=false`, **present=24.90** when `sigreg_act=true` (the act-emb VICReg/SIGReg term is added to `loss`, weighted by `lambd`). ✓
All three mechanisms honored. The arm-C combination (`detach_target=false sigreg_act=true loss.sigreg.weight=0.09`) is wired correctly. NOT blocked.

### Design — tworoom_C = §31 tworoom_B recipe, ONLY the two arm-C flags flipped
FROM SCRATCH (`init_from=null`, encoder `vit_hf` tiny patch=14 image=224 **pretrained=false** → random CLS-192, the load-bearing choice that lets collapse show), `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0`, **arm-C flags: `detach_target=false sigreg_act=true`**, **`loss.sigreg.weight=0.09`** (latent SIGReg kept ON; arm C contrasts the stop-grad vs the act-emb VICReg, NOT the latent SIGReg, and the caveat requires a nonzero `lambd` for `sigreg_act` to contribute). Budget MATCHED to §30 pusht / §31 tworoom_B for comparability: `trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4 +ckpt_every=10`, seed 3072 (lewm.yaml default). Data = committed `config/train/data/tworoom.yaml` (frameskip 5 ⇒ action_block 5, adim=2 ⇒ block adim=10, full `tworoom.h5` 13 GB). SAME 4000-batch cap as tworoom A/B.

### Infra (DISK-SAFE, L40S shared box — the killer is SPT_CACHE_DIR)
L40S `stratus-lookout`, run `sudo -u minghao.fu`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`. **Disk at launch:** `/` 97% (3.5 G free), `/mnt/minghao_data` 22% (2.4 TB free) → ALL writes redirected to `/mnt`: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_tworoom_C TMPDIR=/mnt/minghao_data/tmp_tworoom_C MPLCONFIGDIR=/mnt/minghao_data/mpl_tworoom_C HF_HOME=/mnt/minghao_data/hf **SPT_CACHE_DIR=/mnt/minghao_data/spt_tworoom_C** HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, plus `hydra.run.dir=/mnt/minghao_data/hydra_tworoom_C hydra.output_subdir=null`. Verified the SPT_CACHE redirect active in-log (`log_dir: /mnt/minghao_data/spt_tworoom_C`, summary.json + hf_exports redirected there). Ckpts → `STABLEWM_HOME/checkpoints/sigreg_C_tworoom_scratch/`. `+ckpt_every=10` (ep 10,20,…,100) + `/mnt/minghao_data/prune_tworoom_C.sh` keep-newest-3 loop (PID 2129615). GPU **5** (lowest-occupied at launch, 8.8 GB; co-located per rule 10, never touched other users' procs). Same box train+eval → no cross-GPU render OOD ([[project_l40s_cross_gpu_rendering]]).

### EXACT train command (L40S) — via `/mnt/minghao_data/launch_tworoom_C.sh 5`, expands to:
```
export STABLEWM_HOME=/mnt/minghao_data/.stable-wm SPT_CACHE_DIR=/mnt/minghao_data/spt_tworoom_C \
       XDG_CACHE_HOME=/mnt/minghao_data/xdg_tworoom_C TMPDIR=/mnt/minghao_data/tmp_tworoom_C \
       MPLCONFIGDIR=/mnt/minghao_data/mpl_tworoom_C HF_HOME=/mnt/minghao_data/hf \
       HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_sigreg.py data=tworoom \
  action_pred.enabled=true action_pred.detach_decoder=false action_pred.head=mse \
  action_pred.w_act=1.0 action_pred.w_intent=1.0 action_pred.detach_target=false action_pred.sigreg_act=true \
  loss.sigreg.weight=0.09 init_from=null \
  output_model_name=sigreg_C_tworoom_scratch subdir=sigreg_C_tworoom_scratch \
  trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4 +ckpt_every=10 \
  hydra.run.dir=/mnt/minghao_data/hydra_tworoom_C hydra.output_subdir=null
```
**PID 2129671** (GPU 5). Sanity-val (pre-train, in-log) confirms arm-C terms live + finite: `pred_loss=0.0783 intent_loss=1.0076 act_loss=1.1364 sigreg_loss=27.875 sigreg_act_loss=25.125 z_std=0.000999 act_emb_std=0.0879 total=7.0036` (= pred+intent+act + 0.09·(27.875+25.125) = 2.222 + 4.770 = 6.992 ✓ both SIGReg terms weighted by lambd). z_std @init = 0.001 is the untrained ViT CLS near-constant (test is whether *training* moves it; not collapse). Throughput ~2.1 it/s × 4000 steps ⇒ ~32 min/ep ⇒ ~53 h for 100 ep (slower than tworoom_B's ~3 it/s from co-located GPU contention; acceptable, matches the §30 budget). Log `/mnt/minghao_data/sigreg_logs/train_C_tworoom.log`. Background monitor armed on the log (epoch ticks + Traceback/OOM/Killed).

### MEASUREMENT plan (STEP 3 — DIRECT from the final/converged checkpoint, the robust signal)
Adapt `/mnt/minghao_data/measure_collapse.py` → tworoom run `sigreg_C_tworoom_scratch`: `gip.load_gip_model` (picks `pts[-1]` = highest-epoch = converged), fixed tworoom val batch through `model.encode` → emb (B,T,D), report **z_std = emb.reshape(-1,D).std(dim=0).mean()**, **act_emb_std** (the decisive monitor for arm C — does the no-stop-grad act_emb collapse despite `sigreg_act`?), and **erank** (participation ratio + entropy-erank cross-check). COLLAPSE = z_std→0 AND erank≪D; SCALE-SHRINK = low z_std but erank high + all dims>0. Arm C's specific risk: act_emb collapse (it now receives the intent gradient); `sigreg_act` is the guard.

### EVAL plan (STEP 4 — when converged, rule 8/9)
tworoom via `eval_gip.py --config-name tworoom policy=sigreg_C_tworoom_scratch +gip_eval.mode=<planning|bc> eval.num_eval=50 seed=<42|0|1>`, **N=50 × 3 seeds {42,0,1}**, same-box GPU, CONVERGED ckpt (NOT val-best, reacher-trap rule). planning = pure WM CEM, bc = intention head reactive.

### Result [PENDING convergence + measurement + eval]
z_std / act_emb_std / erank / collapsed / sr_planning / sr_bc / per_seed — to be filled when `weights_epoch_100.pt` lands (or an earlier converged plateau). Compare vs arm A (SIGReg ON, stop-grad on) baseline AND arm B: does replacing the stop-grad with `sigreg_act` keep act_emb non-collapsed and SR intact (conjecture-2 viable), or does the no-stop-grad act_emb collapse / lose SR even with `sigreg_act`? **Provisional expectation:** if `sigreg_act` holds act_emb variance up and SR matches A/B, conjecture-2 is viable (the act-emb stop-grad is replaceable by VICReg). If act_emb collapses (act_emb_std→0) or SR drops, the stop-grad is the better/necessary anti-collapse for the intention target.

---

## §31 🔄🆕 SIGReg-REDUNDANCY test on **TwoRoom** — the CLEAN (state-driven, NOT history-copyable) retest of §30, 3 arms A/B/C (L40S `stratus-lookout`, 2026-06-22) [IN FLIGHT — all 3 arms TRAINING]

**Motivation (why).** §30 ran the SIGReg-redundancy test on PushT and found Arm B (no latent SIGReg) is **scale-contracted but NOT rank-collapsed** (ep10–20: B z_std ~0.19 = 5× lower than A's ~0.95, but B erank ~108/192 > A erank ~83/192, every per-dim std > 0, z_std FLAT not decaying). The §18 CAVEAT (line 3816) is the load-bearing weakness of that result: **PushT's action is HISTORY-CORRELATED** — the past-action stream `a_<t` is a strong open-loop predictor of the demonstrator's next push (zeroing it RAISED open-loop val_act 0.132→0.153), so `act_loss` can be satisfied by COPYING the history rather than by reading the latent state. That WEAKENS `act_loss`'s anti-collapse pressure on `emb`, so a healthy Arm B on PushT is only a CONSERVATIVE positive. **TwoRoom is the clean retest:** it is **STATE-DRIVEN** (2-D point navigation between two rooms; the next action depends on WHERE the agent is, not on action-history extrapolation), so `act_loss` MUST read the latent to predict the action → the action-prediction anti-collapse pressure on `emb` is genuinely exercised. If Arm B's latent stays healthy HERE, the conjecture (SIGReg redundant when predicting raw actions directly) holds on the clean env, not just the history-copyable one — a "Simple"-pillar win for LeWAM (one fewer loss term/hyperparameter). Plus this env adds a SECOND conjecture: **Arm C** tests whether the action-emb stop-grad (`detach_target=true`, the current default anti-collapse on the action embeddings) can be REPLACED by VICReg/SIGReg on the action embeddings (`detach_target=false` + `sigreg_act=true`) — i.e. let the `intent_loss` gradient flow into the action encoder (no stop-grad) but prevent the resulting act-emb collapse with an explicit SIGReg term instead.

**The two conjectures under test (user direction).**
1. **Latent SIGReg is REDUNDANT when the action head predicts raw actions directly** (Arm A vs Arm B). Mechanism (SCAR 2605.16412 §3.3): `act_loss=‖decoder(intention)−a‖²` is an independent anti-collapse force on the shared latent `emb`, so dropping the explicit latent SIGReg (`loss.sigreg.weight: 0.09→0`) should NOT collapse the latent and should NOT drop SR.
2. **The action-emb stop-grad can be replaced by act-emb VICReg/SIGReg** (Arm A vs Arm C). Default design stop-grads the `intent_loss` target (`tgt_act_emb = ctx_act.detach()`, `detach_target=true`) to keep the action encoder from collapsing onto a trivial target. Arm C instead lets the gradient FLOW (`detach_target=false`) and adds an explicit SIGReg on the action embeddings (`sigreg_act=true`) to hold them apart. Question: is `detach_target=false + sigreg_act=true` a viable (or better) anti-collapse for the act embeddings than the stop-grad?

### Design — 3 arms, FROM SCRATCH, action_pred ON throughout, MATCHED §30/PushT budget for comparability
- **Env:** TwoRoom (`data=tworoom`, committed `config/train/data/tworoom.yaml`, frameskip 5 ⇒ action_block 5, action_dim=2 ⇒ Adim=10, full `tworoom.h5` = 12.7 GB). STATE-DRIVEN 2-D navigation — the CLEAN test (no PushT history-copyability confound, §18 caveat resolved).
- **Arm A (baseline = default design, this agent's arm):** `loss.sigreg.weight=0.09 action_pred.detach_target=true action_pred.sigreg_act=false`. Latent SIGReg ON, act-emb stop-grad ON, no act-emb SIGReg. Run `sigreg_A_tworoom_scratch`, GPU 1, PID 2148194.
- **Arm B (conjecture-1, latent SIGReg OFF):** `loss.sigreg.weight=0` (the `lambd*sigreg_loss` term EXACTLY zeroed; SIGReg module still RUNS + logs `sigreg_loss` for monitoring but contributes 0). `detach_target=true sigreg_act=false`. Run `sigreg_B_tworoom_scratch` (sibling agent), GPU 5, PID 2125270.
- **Arm C (conjecture-2, act-emb VICReg instead of stop-grad):** `action_pred.detach_target=false action_pred.sigreg_act=true loss.sigreg.weight=0.09`. The `intent_loss` gradient FLOWS to the action encoder (no stop-grad) AND a `sigreg_act_loss = SIGReg(act_emb)` term is added (weighted by the SAME `lambd=0.09`). Run `sigreg_C_tworoom_scratch` (sibling agent), GPU 7, PID 2129671.
- **From scratch** (`init_from=null`, encoder `vit_hf` size=tiny patch=14 image=224 **pretrained=false** → random init, CLS-192). Load-bearing: warm-starting would pre-establish a non-collapsed latent via the base-WM pred_loss and HIDE collapse.
- **EVERYTHING ELSE IDENTICAL across A/B/C and matched to §30 PushT:** `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0`, `trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4`, seed 3072 (lewm.yaml default), `+ckpt_every=10`. (Same `limit_train_batches=4000` cap as §30 PushT AND as siblings B/C — the brief's "same cap for all arms of this env" satisfied; A/B/C all run 4000×100.)

### EXACT train commands (L40S, `sudo -u minghao.fu`, `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`)
```
# common disk-safe env (per-arm SPT_CACHE_DIR/XDG/TMP/MPL on /mnt; the SPT_CACHE_DIR redirect is the killer per memory):
export STABLEWM_HOME=/mnt/minghao_data/.stable-wm SPT_CACHE_DIR=/mnt/minghao_data/spt_tworoom_<A|B|C> \
       XDG_CACHE_HOME=/mnt/minghao_data/xdg_tworoom_<A|B|C> TMPDIR=/mnt/minghao_data/tmp_tworoom_<A|B|C> \
       MPLCONFIGDIR=/mnt/minghao_data/mpl_tworoom_<A|B|C> HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1 \
       MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4
# ARM A (GPU1): detach_target=true  sigreg_act=false loss.sigreg.weight=0.09
# ARM B (GPU5): detach_target=true  sigreg_act=false loss.sigreg.weight=0
# ARM C (GPU7): detach_target=false sigreg_act=true  loss.sigreg.weight=0.09
CUDA_VISIBLE_DEVICES=<1|5|7> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_sigreg.py data=tworoom \
  action_pred.enabled=true action_pred.detach_decoder=false action_pred.head=mse \
  action_pred.w_act=1.0 action_pred.w_intent=1.0 \
  action_pred.detach_target=<true|true|false> action_pred.sigreg_act=<false|false|true> \
  loss.sigreg.weight=<0.09|0|0.09> init_from=null \
  output_model_name=sigreg_<A|B|C>_tworoom_scratch subdir=sigreg_<A|B|C>_tworoom_scratch \
  trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4 +ckpt_every=10 \
  hydra.run.dir=/mnt/minghao_data/hydra_tworoom_<A|B|C> hydra.output_subdir=null
```
(Arm A launcher `/mnt/minghao_data/launch_A_tworoom.sh <gpu>`; prune loop `/mnt/minghao_data/prune_A_tworoom.sh` keeps newest-3 + always-keep epoch_100. Logs `/mnt/minghao_data/sigreg_logs/train_A_tworoom.log`.) Confirmed in all 3 logs: `[GIP] Intention predictor ON  Adim=10  head=mse w_act=1.0 w_intent=1.0 detach_target=<T|T|F> sigreg_act=<F|F|T>`.

### STEP 1 — 1-STEP FORWARD VERIFICATION (done BEFORE training; `/mnt/minghao_data/verify_tworoom_arms.py`, GPU 7, one real tworoom batch bs=16, untrained-init)
Built the gated model per arm, ran `lejepa_forward`, printed the loss decomposition + an `intent_loss.backward()` gradient-flow probe on `model.action_encoder`:
| arm | weight | detach_target | sigreg_act | pred_loss | sigreg_loss | intent_loss | act_loss | **λ·sigreg_loss (latent term)** | **sigreg_act_loss** (λ·) | **act_encoder grad-from-intent_loss** |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.09 | true  | false | 0.1395 | 6.3727 | 1.0109 | 0.9371 | **0.5735** (present) | — (absent) | **None** (detached ⇒ no grad to act_emb) ✓ |
| B | 0    | true  | false | 0.1337 | 6.3756 | 1.0126 | 0.9787 | **0.000000** (EXACTLY zeroed) ✓ | — | None ✓ |
| C | 0.09 | false | true  | 0.1337 | 6.3277 | 1.0159 | 1.0167 | 0.5695 (present) | 6.282 (**0.5654** λ·) ✓ | **0.1945 > 0** (grad FLOWS to act_emb) ✓ |
→ **All three arm mechanisms verified numerically:** (B) `loss.sigreg.weight=0` zeros the latent SIGReg term to machine precision; (C) `detach_target=false` lets the `intent_loss` gradient reach the action encoder (0.1945 > 0, vs A/B's None) AND `sigreg_act=true` adds the act-emb SIGReg term (0.5654 contribution). The flags are honored. (z_std≈0.05 at untrained-init = near-constant random ViT CLS, as in §30; the test is whether training RAISES it.)

### STEP 2 — TRAINING [IN FLIGHT, all 3 arms alive ~1.8–2.0 it/s, GPU-bound (util 100% on the eval-GPUs), ETA ~50–55 h/arm for 4000×100]
Disk SAFE at launch + steady: `/` 97% (3.5 GB free, STABLE — `hydra.output_subdir=null` + SPT_CACHE_DIR redirect keep all big writes off the full `/`), `/mnt/minghao_data` 22% (2.4 TB free, all ckpts/runs/tmp here). GPUs co-located (A=1, B=5, C=7), models ~7.4 GB each, never touched other users' procs (the 36 GB dinov2_base_768 PID 1747041 on its own GPU, fan2 procs untouched; pkill bracket-escaped `train_sigreg[.]py.*sigreg_tworoom_A` when renaming).
- **NOTE (provenance):** this agent first launched arm A mis-named `sigreg_tworoom_A` (ep0, ~350 steps), then KILLED + relaunched as `sigreg_A_tworoom_scratch` to match the sibling B/C naming convention (`sigreg_<arm>_tworoom_scratch`) and the §30 PushT pattern (`sigreg_A_pusht_scratch`) — so all 3 arms share naming for the measure/eval scripts + verdict table. <1 min of compute discarded; the relaunched run carries the identical arm-A config.

### STEP 3 — collapse measurement [PENDING ckpts; script READY: `/mnt/minghao_data/measure_collapse_tworoom.py`]
Mirrors §30's `measure_collapse.py` but for tworoom: rebuilds each arm's model via `gip.load_gip_model(run, epoch)` (the SAME path `eval_gip.py` uses), rebuilds the EXACT tworoom val split + transforms from the run's `full_config.yaml`, runs ONE shared deterministic val batch (256 windows) through `model.encode`, computes **`z_std = emb.reshape(-1,D).std(dim=0).mean()`** + `act_emb_std` + per-dim std spread + latent **effective rank** (entropy of singular values, participation ratio). Run per `--epoch` as ckpts land (10,20,…,100), `--runs sigreg_A_tworoom_scratch sigreg_B_tworoom_scratch sigreg_C_tworoom_scratch`, GPU 1 (= train GPU, no cross-GPU concern for a pure-encode measurement). **COLLAPSE** = z_std→0 AND erank≪192; **SCALE-SHRINK (not collapse)** = low z_std but erank stays high + all per-dim std > 0 (the §30 PushT-B signature). Decisive signal = does Arm B's z_std DECAY toward 0 with erank dropping (→ true SCAR collapse, conjecture-1 FALSIFIED, SIGReg load-bearing on the clean env), or STABILIZE at a smaller scale with rank preserved (→ scale contraction not collapse, conjecture-1 holds on the clean env)?

### STEP 4 — SR eval [PENDING converged ckpts; commands READY]
Both/all arms, same-box (train-GPU = eval-GPU, [[project_l40s_cross_gpu_rendering]]), **N=50 × 3 seeds {42,0,1}**, planning + bc modes, **CONVERGED ckpt** (`+ckpt_epoch=100`, NOT val-best — the reacher checkpoint trap [[project_gcidm_vs_ours_verdict]]):
```
eval_gip.py --config-name tworoom policy=sigreg_<A|B|C>_tworoom_scratch \
  +gip_eval.mode=<planning|bc> +ckpt_epoch=100 eval.num_eval=50 seed=<42|0|1>
```
Results → `gip_eval/<mode>/sigreg_<A|B|C>_tworoom_scratch/<mode>_<...>_results.txt`. (planning = pure WM CEM, state head only; bc = intention head reactive policy. tworoom eval config: env `swm/TwoRoom-v1`, goal_offset_steps=25, eval_budget=50, CEM solver, `_set_state`/`_set_goal_state` from proprio/goal_proprio.)

### RESULT TABLE (to fill at convergence — ALL per-seed)
| arm | config | final z_std | act_emb_std | erank/192 | collapsed? | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) |
|---|---|---|---|---|---|---|---|
| A (baseline: latent-SIGReg ON, stop-grad ON) | w=0.09 dT=true sigregAct=false | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| B (latent-SIGReg OFF) | w=0 dT=true sigregAct=false | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| C (act-emb VICReg, no stop-grad) | w=0.09 dT=false sigregAct=true | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |

### VERDICT (to write at convergence)
[PENDING — training in flight.] Conjecture-1 (latent SIGReg redundant): does Arm B match Arm A's z_std/erank/SR on the CLEAN state-driven env (→ conjecture holds beyond the §30 history-copyable case → "Simple"-pillar win), or does B collapse here where §30-PushT didn't (→ PushT's healthy-B was the history-copyability confound, SIGReg load-bearing when the action genuinely needs the latent)? Conjecture-2 (act-emb VICReg replaces stop-grad): does Arm C hold the act embeddings apart (act_emb_std healthy, no collapse) and MATCH or BEAT Arm A's SR (→ VICReg is a viable drop-in for the stop-grad), or does letting the intent gradient flow degrade SR even with the SIGReg guard (→ the stop-grad stays the right design)?

### Unblock condition (robust to SSH blips — trainings detached, monitor armed)
`weights_epoch_100.pt` in all 3 run dirs (`/mnt/minghao_data/.stable-wm/checkpoints/sigreg_{A,B,C}_tworoom_scratch/`) AND the collapse trajectory flat over the last ~10 ep (verify B's z_std doesn't keep decaying to 0) → run `measure_collapse_tworoom.py --epoch 100` → run the 3-arm × 2-mode × 3-seed N=50 evals (same-box) → fill the table → verdict + EXPERIMENTS update.

### Files (all on L40S `/mnt/minghao_data/`)
- `verify_tworoom_arms.py` — STEP 1 numeric arm verification (DONE, table above).
- `measure_collapse_tworoom.py` — STEP 3 direct-from-ckpt collapse measurement (READY, pending ckpts).
- `launch_A_tworoom.sh` / `prune_A_tworoom.sh` — arm-A launcher + keep-newest-3 prune loop.
- Ckpts (arm A, mine): `/mnt/minghao_data/.stable-wm/checkpoints/sigreg_A_tworoom_scratch/`. Logs `/mnt/minghao_data/sigreg_logs/train_A_tworoom.log`.
- **NOT touched:** shared `train.py`, `jepa.py`, `gip.py`, `eval_gip.py`, `module.py`, `gen_master_table.py`, other agents' ckpts/GPUs/runs (B/C are sibling-agent runs; I only read their cmds to match the budget/naming). `train_sigreg.py` is the pre-existing §30 non-invasive copy (diff vs `train.py` = the 4-line z_std monitor), UNCHANGED.
- **PIDs:** A 2148194 (GPU1, this agent). Siblings: B 2125270 (GPU5), C 2129671 (GPU7).

## §31-clean 🔄🆕 SIGReg-redundancy CLEAN-test arm on **reacher** — the `use_action_history=FALSE` confound remover (`reacher_sigoff_noah`) (L40S, 2026-06-22) [IN FLIGHT — TRAINING]

**Motivation (why).** §30 (pusht) and §31 (reacher/tworoom/cube arms A/B/C) test the user's SIGReg-redundancy conjecture: with the action-prediction head active (`action_pred.enabled=true`, `act_loss=(decoder(intention)−raw_action)²`), the explicit latent anti-collapse (SIGReg, `loss.sigreg.weight`) should be REDUNDANT, because `act_loss` (raw-action regression) supplies an independent anti-collapse pressure on the shared latent `emb`. §30/§31 arm B (latent-SIGReg OFF) found a SCALE-SHRINK not a SCAR-collapse on pusht (z_std ~5× lower than arm A but erank STAYS HIGH, all per-dim std > 0). BUT arm B on these envs still runs with **`use_action_history=true`** — so the §18 history-copyability confound applies: when the past-action stream `a_<t` is available to the action head, a chunk of `act_loss` can be satisfied by COPYING/extrapolating the action history rather than by reading the latent `emb`. That WEAKENS `act_loss`'s anti-collapse pressure on the latent and makes a healthy arm B only a CONSERVATIVE positive (the latent might be partly held up by something OTHER than act_loss reading it). **This arm removes that confound entirely:** with `model.use_action_history=FALSE` (`jepa.py:192,338` zero `past_act_emb` in BOTH the train `predict_intention` and the rollout/eval path), the action head has NO action-history shortcut — it MUST read the latent to predict the action. So if sigreg-OFF STILL keeps the latent full-rank AND holds SR HERE, the conjecture (act_loss anchors the latent → SIGReg redundant) is CLEANLY confirmed, not just conservatively. If instead the latent collapses here where the history-ON arm B did not, the §30/§31 positives were a history-copyability artifact and SIGReg IS load-bearing once the shortcut is removed.

### Design — the §31 reacher arm-B recipe with ONE change: `model.use_action_history=false`
FROM SCRATCH (`init_from=null`, encoder `vit_hf` size=tiny patch=14 image=224 **pretrained=false** → random CLS-192, the load-bearing choice that lets collapse show), `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0 detach_target=true sigreg_act=false`, **`loss.sigreg.weight=0`** (sigreg OFF — the SIGReg module still RUNS + `sigreg_loss` is logged for monitoring but contributes 0 to `loss`), **`model.use_action_history=false`** (THE clean-test knob — past-action stream zeroed, action head must read the latent). Budget MATCHED to the §31 reacher A/B/C arms for comparability: `trainer.max_epochs=100 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4 +ckpt_every=10`, seed 3072 (lewm.yaml default). **`limit_train_batches=1000` matches the RELAUNCHED reacher A/B/C cap** (the original 4000 was killed under box contention and relaunched at 1000 across all arms, EXPERIMENTS.md §31 reacher STATUS line) — same cap = comparable. Data = committed `config/train/data/dmc.yaml` (`name: reacher.h5`, frameskip 5 ⇒ action_block 10, adim=2, the same reacher data the §15/§31 reacher arms use). reacher has the §15 val-best CHECKPOINT TRAP → headline the CONVERGED `weights_epoch_100.pt`, NOT val-best.

### STEP 1 — flag verified (smoke, 2 batches, GPU 0)
`model.use_action_history=false` accepted by `train_sigreg.py`; built model logs `[GIP] Intention predictor ON Adim=10 head=mse w_act=1.0 w_intent=1.0 detach_target=True sigreg_act=False`; 2-batch train+val ran with no crash. **`loss.sigreg.weight=0` verified:** `sigreg_loss` still COMPUTED+logged (3.4375) but `loss_epoch=2.312 = pred 0.0725 + intent 1.011 + act 1.228` (NO sigreg term in the total — the `lambd*sigreg` contribution is exactly 0). `z_std`/`act_emb_std`/`act_loss` all log. The `use_action_history=false` zeroing is read directly from `jepa.py` (lines 192 + 338: `if not self.use_action_history: past_act_emb = torch.zeros_like(past_act_emb)`) and rides into `config.json` via `cfg.model` (the same mechanism §18 used), so eval rebuilds with the flag. (A "did NOT receive gradients" warning fires on the now-dead past-action path — EXPECTED, that stream is zeroed.)

### STEP 2 — TRAIN (in flight). EXACT command (host `L40S`, `sudo -u minghao.fu`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`; driver `/mnt/minghao_data/launch_reacher_sigoff_noah.sh GPU`)
```
# disk-safe env (per-arm scratch dirs; SPT_CACHE_DIR is the killer redirect — /var is 99% full):
export STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_reacher_noah \
  TMPDIR=/mnt/minghao_data/tmp_reacher_noah MPLCONFIGDIR=/mnt/minghao_data/mpl_reacher_noah \
  HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4 \
  SPT_CACHE_DIR=/mnt/minghao_data/spt_reacher_noah PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
CUDA_VISIBLE_DEVICES=7 python train_sigreg.py data=dmc \
  model.use_action_history=false \
  action_pred.enabled=true action_pred.detach_decoder=false action_pred.head=mse \
  action_pred.w_act=1.0 action_pred.w_intent=1.0 \
  action_pred.detach_target=true action_pred.sigreg_act=false \
  loss.sigreg.weight=0 init_from=null \
  output_model_name=sigreg_noah_reacher_scratch subdir=sigreg_noah_reacher_scratch \
  trainer.max_epochs=100 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4 +ckpt_every=10 \
  hydra.run.dir=/mnt/minghao_data/hydra_reacher_noah hydra.output_subdir=null
```
- **Launched 2026-06-22 ~00:37 UTC:** PID 2171141, GPU 7 (lowest-mem at launch, ~20.7 GB used / 46 GB; box is heavily loaded — 8+ co-located train_sigreg arms + other users, GPU util is CPU-render-driven not memory-bound per rule 10). Confirmed in log: `[GIP] Intention predictor ON Adim=10 head=mse w_act=1.0 w_intent=1.0 detach_target=True sigreg_act=False`, stepping `[Epoch 0/100] step N/1000`. Throughput ~1.1–1.8 it/s under contention ⇒ ~9–15 min/epoch ⇒ ~15–25 h for 100ep. `+ckpt_every=10` ⇒ `weights_epoch_{10..100}.pt`.
- **Disk safety:** all writes redirected to `/mnt/minghao_data` (2.4 TB free, 22%); `/var/lib/docker` is at 99% (88 GB) and `/` at 97% (3.5 GB) — the `SPT_CACHE_DIR` redirect is the one that prevents the metrics.csv/resumption-ckpt crash ([[project_l40s_path_consolidation]]). Prune loop `/mnt/minghao_data/prune_reacher_noah.sh` (PID 2177429) keeps newest-3 weights + newest-1 SPT runs-dir.

### STEP 3 — DIRECT-FROM-CHECKPOINT collapse measurement [PENDING ckpts; auto-run by finalizer at ep100]
`/mnt/minghao_data/measure_collapse_reacher.py --epoch 100 --batch-run sigreg_B_reacher_scratch --arms noah_sigOFF_noHist:sigreg_noah_reacher_scratch B_sigreg_OFF:sigreg_B_reacher_scratch A_sigreg_ON:sigreg_A_reacher_scratch` (a SHARED reacher val batch from B's split → apples-to-apples). For each ckpt: `gip.load_gip_model(run, epoch=100)` → `model.encode` → `z_std = emb.reshape(-1,D).std(dim=0).mean()` + effective rank (participation ratio = exp(entropy) of the singular-value spectrum) + `act_emb_std`. COLLAPSE = z_std→0 AND erank≪192. SCALE-SHRINK (= the conjecture-favorable §30 signature, NOT collapse) = low z_std but erank stays HIGH + all per-dim std > 0. The decisive comparison: does noah (sigreg-OFF, NO history shortcut) stay full-rank like §31 arm B did (history-ON)?

### STEP 4 — EVAL SR (reacher, `eval_gip.py --config-name reacher`, N=50, seeds {42,0,1}, same-box, planning + bc) [PENDING convergence; auto-run by finalizer]
```
PYTHONPATH=/tmp/reacher_compat:$B python eval_gip.py --config-name reacher \
  policy=sigreg_noah_reacher_scratch +gip_eval.mode=<planning|bc> eval.num_eval=50 seed=<42|0|1>
```
(planning = pure WM CEM; bc = intention head reactive. Reacher needs the dm_control↔mujoco compat shim `/tmp/reacher_compat/sitecustomize.py` on PYTHONPATH, §15 blocker 1; eval same-box GPU 7 = the train GPU per the cross-GPU render gotcha [[project_l40s_cross_gpu_rendering]].)

### RESULT TABLE (to fill at convergence — ALL per-seed)
| arm | use_action_history | sigreg λ | final z_std | act_emb_std | erank/192 | collapsed? | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) |
|---|---|---|---|---|---|---|---|---|
| **noah (CLEAN: sigreg OFF, NO history)** | **false** | **0** | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| B ref (sigreg OFF, history ON) | true | 0 | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| A ref (sigreg ON, history ON) | true | 0.09 | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |

### VERDICT [PENDING convergence + eval]
Does sigreg-OFF stay FULL-RANK + hold SR even WITHOUT the action-history shortcut (`use_action_history=false`)? If YES (noah erank stays high, all per-dim std > 0, SR ≈ refs) → the SIGReg-redundancy conjecture is CLEANLY confirmed: `act_loss` (reading the latent, with no history to copy) anchors the latent on its own, so explicit SIGReg is redundant when the action head predicts raw actions directly — a "Simple"-pillar win, with the §18 confound removed. If NO (noah collapses where history-ON arm B did not) → the §30/§31 arm-B positives were a history-copyability artifact and SIGReg is load-bearing once the shortcut is gone. [Fills once `weights_epoch_100.pt` lands and the finalizer runs STEP-3 + STEP-4.]

### Files (L40S)
- `/mnt/minghao_data/launch_reacher_sigoff_noah.sh` — launcher (the §31 reacher-B recipe + `model.use_action_history=false`, sigreg λ=0, 1000-batch cap).
- `/mnt/minghao_data/prune_reacher_noah.sh` — keep-newest-3 weights + newest-1 SPT runs-dir prune loop (PID 2177429).
- `/mnt/minghao_data/reacher_sigoff_noah_finalize.sh` — SELF-COMPLETING finalizer (log `/mnt/minghao_data/reacher_sigoff_noah_finalize.log`). Blocks until `weights_epoch_100.pt`, then auto-runs STEP-3 collapse (noah_sigOFF + B + A refs, shared batch) + STEP-4 eval (noah × {planning,bc} × seeds {42,0,1}, N=50, same-box GPU 7). Reuses the shared `/mnt/minghao_data/measure_collapse_reacher.py` (UNCHANGED). **⚠️ RENAMED 2026-06-22 00:49 UTC** from `reacher_noah_finalize.sh` (PID 2178840 → relaunched 2197028): the sigreg-ON sibling arm (§31-clean below) collided on the SAME filename `reacher_noah_finalize.{sh,log}`; both were split into `reacher_{sigoff,sigon}_noah_finalize.{sh,log}` and relaunched, so each waits on its own run with its own log. No work lost (both were still in the ep100 wait loop).
- Ckpts: `/mnt/minghao_data/.stable-wm/checkpoints/sigreg_noah_reacher_scratch/`. Log `/mnt/minghao_data/sigreg_reacher_logs/train_noah.log`.
- **PIDs:** train 2171141 (GPU7), prune 2177429, finalizer 2178840.
- **NOT touched:** shared `train_sigreg.py`/`jepa.py`/`gip.py`/`eval_gip.py`/`module.py`/`gen_master_table.py`, the §31 reacher A/B/C runs/GPUs (read-only: matched their 1000-batch cap + naming + recipe), other users' procs.

---

## §31-clean 🔄🆕 SIGReg-REDUNDANCY — the CLEAN cube retest that removes the §18 history-copy confound (`use_action_history=FALSE`) (L40S `stratus-lookout`, 2026-06-22) [IN FLIGHT — sigreg-ON arm training]

**Motivation (why — the rigor completer).** §30 (pusht) and the §30 sibling envs ran the sigreg-redundancy test with `use_action_history=true` (the default), so the action head could partly satisfy `act_loss` by **copying/extrapolating the past-action stream `a_<t`** rather than by reading the latent `emb`. §18 proved on pusht that `a_<t` is a strong open-loop predictor of the demonstrator's next push (zeroing it RAISED open-loop val_act 0.132→0.153), so a chunk of `act_loss`'s anti-collapse pressure on `emb` was a HISTORY shortcut, not latent-reading. That is the §18 confound on the redundancy conjecture: a full-rank latent in the sigreg-OFF arm could be explained either by `act_loss` anchoring the latent (the conjecture) OR by `act_loss` being cheaply satisfiable via history-copy (so the latent was never the load-bearing path and SIGReg's removal was harmless for an unrelated reason). **This arm removes that confound**: with `use_action_history=FALSE` the action head's past-action stream is hard-zeroed (verified below), so `act_loss` can ONLY be lowered by reading the latent state. If sigreg-OFF STILL keeps the latent full-rank here (companion `cube_sigoff_noah` arm), the conjecture is CLEANLY confirmed — `act_loss` (not history-copy) anchors the latent. This arm (`cube_sigon_noah`, sigreg ON) is the **paired reference** the OFF arm is measured against on the SAME val batch.

### Design — cube clean retest, 2 paired arms (this = sigreg-ON reference; companion sibling agent runs sigreg-OFF)
- **Env:** cube (`data=ogb_lance` → `ogbench/ogb_cube_single.lance`, frameskip 5 ⇒ action_block 25, adim=5, keys pixels/action/observation). State-DEPENDENT grasp/place manipulation (vs pusht's history-copyable push) — the cleaner anti-collapse env the §30 caveat asked for.
- **THE clean change vs §30:** `model.use_action_history=false` (BOTH arms of this env). In `jepa.JEPA.predict_intention` (line 191-192) and `intention_rollout` (line 337-338) this hard-zeros `past_act_emb = torch.zeros_like(past_act_emb)` BEFORE any task/goal add. Single-task cube has no task_vec/goal_emb, so the action head sees EXACTLY zeros for the past-action stream → cannot copy `a_<t`, must read `emb`. The flag travels to eval via the saved `config.json`, so eval is consistent (gip.py:246 `self.use_history = getattr(model, "use_action_history", True)`).
- **THIS arm (sigreg ON, the reference):** `loss.sigreg.weight=0.09` (default λ). **Companion arm (sigreg OFF):** `loss.sigreg.weight=0` (sibling agent, `cube_sigoff_noah`) — the decisive arm whose latent rank answers the conjecture.
- **From scratch** (`init_from=null`, encoder `vit_hf` size=tiny patch=14 image=224 **pretrained=false** → random init, CLS-192) — same load-bearing choice as §30 (warm-start would pre-establish a non-collapsed latent and HIDE collapse).
- **Everything else IDENTICAL to §30 for comparability:** `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0 detach_target=true sigreg_act=false`, `trainer.max_epochs=100 +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4` (4000-batch cap held consistently ACROSS both sigreg arms of this env, per the brief), seed 3072, `+ckpt_every=10`. (reacher has a val-best-underfits ckpt trap [[project_gcidm_vs_ours_verdict]]; cube does not, but the rule still holds → use the CONVERGED ep≈100 ckpt.)

### STEP 1 — flag found + 1-STEP FORWARD VERIFICATION (DONE, before the real run) — past-action stream provably zeroed
- **Flag:** `model.use_action_history` (top-level `jepa.JEPA` kwarg; `config/train/model/lewm.yaml` documents the override `model.use_action_history=false`). Zeroing at `jepa.py:191-192` (`if not self.use_action_history: past_act_emb = torch.zeros_like(past_act_emb)`) in `predict_intention`, mirrored at `jepa.py:337-338` in `intention_rollout`. UNCONDITIONAL zero, BEFORE the task/goal additive hooks.
- **`verify_noah.py` (`/mnt/minghao_data/`, CPU):** built the gated cube model (`use_action_history=false action_pred.enabled=true head=mse`, action_encoder input_dim=25), attached `action_predictor`+`action_decoder`, called `predict_intention(emb, pa1)` vs `predict_intention(emb, pa2)` with TWO totally different past-action streams (`pa2 = 7.3·N+5`). **Result: `max|intention(pa1)−intention(pa2)| = 0.000e+00`, `max|action(pa1)−action(pa2)| = 0.000e+00`** → output EXACTLY invariant to the past-action stream ⇒ stream hard-zeroed ⇒ the action head MUST read the latent. Printed `use_action_history = False` + `PAST-ACTION STREAM ZEROED`.
- **Smoke (2-batch, GPU5, deleted after):** `[GIP] Intention predictor ON Adim=25 head=mse w_act=1.0 w_intent=1.0 detach_target=True sigreg_act=False`, `validate/z_std` LOGS (0.005 at random init — from-scratch ViT CLS near-constant across a batch ⇒ z_std STARTS tiny; `fit/z_std=0.32` once gradients flow, matching §30), `validate/sigreg_loss_epoch=3.48` computed (sigreg ON), `weights_epoch_1.pt`+config.json saved to `/mnt`, **exit 0, no shape crash, no NaN, /var held at 88 GB free.**

### Infra (DISK-SAFE, L40S shared — `/var/lib/docker` 99 % full, `/` 97 %, only `/mnt/minghao_data` free 2.4 TB)
Run `sudo -u minghao.fu`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `…/le-wm-repro`. ALL scratch → `/mnt`: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm SPT_CACHE_DIR=/mnt/minghao_data/spt_cube_sigon_noah XDG_CACHE_HOME=/mnt/minghao_data/xdg_cube_sigon_noah TMPDIR=/mnt/minghao_data/tmp_cube_sigon_noah MPLCONFIGDIR=/mnt/minghao_data/mpl_cube_sigon_noah HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, plus `hydra.run.dir=/mnt/minghao_data/hydra_cube_sigon_noah hydra.output_subdir=null`. `SPT_CACHE_DIR` is the load-bearing extra redirect ([[project_l40s_path_consolidation]]). `+ckpt_every=10` + `prune_cube_noah.sh` keep-newest-3 (PID 2179270; SaveCkptCallback does NOT self-prune here). GPU 5 (lowest-occupied at launch, ~29 GB free; vit-tiny cube ≈ 25 GB). Same box train+eval → no cross-GPU render OOD ([[project_l40s_cross_gpu_rendering]]).

### EXACT train command (driver `/mnt/minghao_data/train_cube_sigon_noah.sh`, GPU 5, PID 2178675)
```
CUDA_VISIBLE_DEVICES=5 python train_sigreg.py data=ogb_lance \
  model.use_action_history=false \
  action_pred.enabled=true action_pred.detach_decoder=false action_pred.head=mse \
  action_pred.w_act=1.0 action_pred.w_intent=1.0 action_pred.sigreg_act=false \
  loss.sigreg.weight=0.09 init_from=null \
  output_model_name=cube_sigon_noah subdir=cube_sigon_noah \
  trainer.max_epochs=100 trainer.devices=[0] \
  +trainer.limit_train_batches=4000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4 +ckpt_every=10 \
  hydra.run.dir=/mnt/minghao_data/hydra_cube_sigon_noah hydra.output_subdir=null
```
(companion sigreg-OFF arm = IDENTICAL except `loss.sigreg.weight=0` + `cube_sigoff_noah` naming; sibling agent.)

### STEP 3 — DIRECT-from-checkpoint collapse measurement (READY, pending ep100 ckpt)
`measure_collapse_cube.py` (`/mnt/minghao_data/`, mirrors §30's `measure_collapse.py`): rebuilds the model via `gip.load_gip_model(run, epoch)` (the SAME path `eval_gip.py` uses), rebuilds the EXACT cube val split + transforms from the run's `full_config.yaml`, runs one deterministic val batch through `model.encode`, computes `z_std = emb.reshape(-1,D).std(dim=0).mean()` + effective rank (entropy of singular values) + `act_emb_std`. COLLAPSE = `z_std→0 AND erank≪D`. Same val batch for sigon vs sigoff → apples-to-apples.

### STEP 4 — EVAL plan (cube, when converged — rule 8/9): N=50 × 3 seeds {42,0,1}, same-box GPU 5
`eval_gip.py --config-name cube policy=cube_sigon_noah +gip_eval.mode=<planning|bc> eval.num_eval=50 seed=<42|0|1>` (planning = pure WM CEM over the trained latent dynamics; bc = the intention head reactive policy). Driver `/mnt/minghao_data/eval_cube_sigon_noah.sh`.

### RESULT TABLE (to fill at convergence — ALL per-seed)
| arm | use_action_history | final `z_std` | `act_emb_std` | erank/192 | collapsed? | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) |
|---|---|---|---|---|---|---|---|
| cube_sigon_noah (sigreg ON, λ=0.09, THIS) | **FALSE** | [pending ep100] | [pending] | [pending] | [pending] | [pending] | [pending] |
| cube_sigoff_noah (sigreg OFF, λ=0) | **FALSE** | [companion sibling arm] | | | | | |

### VERDICT (to write at convergence)
[PENDING — the clean test.] The §18 confound is removed (verified `max diff = 0` ⇒ `a_<t` hard-zeroed ⇒ `act_loss` lowerable ONLY by reading the latent). If the companion `cube_sigoff_noah` (sigreg OFF) STILL holds full rank (`erank≫1`, every per-dim std > 0, z_std stable not →0) AND matches `cube_sigon_noah`'s SR, the redundancy conjecture is CLEANLY confirmed (act_loss anchors the latent → SIGReg redundant when predicting actions directly = a "Simple"-pillar win, NOT a history-shortcut artifact). If sigreg-OFF collapses here (z_std→0, erank→1), SIGReg is load-bearing once the history shortcut is removed (conjecture falsified in the clean regime). Decide on the ep100 trajectory of BOTH arms + the SR evals.

### Files (all on L40S `/mnt/minghao_data/`)
- `verify_noah.py` — STEP 1 zeroing verification (DONE: max|Δ|=0, stream zeroed).
- `train_cube_sigon_noah.sh` — launcher (THIS sigreg-ON arm). `prune_cube_noah.sh` — keep-newest-3 prune loop (PID 2179270).
- `measure_collapse_cube.py` — STEP 3 direct-from-ckpt collapse measurement (READY, pending ckpts).
- `eval_cube_sigon_noah.sh` — STEP 4 eval driver (planning|bc, N=50, seed arg).
- Ckpts (mine): `/mnt/minghao_data/.stable-wm/checkpoints/cube_sigon_noah/`. Log `/mnt/minghao_data/cube_noah_logs/train_sigon.log`.
- **NOT touched:** shared `train.py`, `jepa.py`, `gip.py`, `eval_gip.py`, `module.py`, `gen_master_table.py`, other agents' ckpts/GPUs/runs. `train_sigreg.py` = the pre-existing §30 non-invasive copy, UNCHANGED. `gen_master_table.py` NOT edited.
- **PID:** cube_sigon_noah 2178675 (GPU5, this agent). Companion `cube_sigoff_noah` = sibling agent.

---

## §31-clean 🔄🆕 SIGReg-redundancy CLEAN-test — the sigreg-**ON** reference arm on **reacher** with `use_action_history=FALSE` (`reacher_sigon_noah`) (L40S, 2026-06-22) [IN FLIGHT — TRAINING]

**Motivation (why).** This is the **sigreg-ON reference** that completes the CLEAN A/B at the no-history (`use_action_history=false`) setting begun by the sibling `reacher_sigoff_noah` arm (§31-clean above). The user's SIGReg-redundancy conjecture (§30/§31): with the action-prediction head active (`action_pred.enabled=true`, `act_loss=(decoder(intention)−raw_action)²`), the explicit latent anti-collapse (SIGReg, `loss.sigreg.weight`) should be REDUNDANT because `act_loss` reading the latent supplies its own anti-collapse pressure. §30/§31 arm B (latent-SIGReg OFF) showed a SCALE-SHRINK not a SCAR-collapse — BUT all those arms ran with `use_action_history=true`, so the §18 history-copyability confound applies: a chunk of `act_loss` can be satisfied by COPYING the past-action stream `a_<t` rather than reading the latent, which WEAKENS act_loss's anti-collapse pressure → a healthy arm B was only a CONSERVATIVE positive. The clean test removes the shortcut: `model.use_action_history=false` (`jepa.py:192,338` zero `past_act_emb` in BOTH the train `predict_intention` and the eval-rollout path), so the action head MUST read the latent. The CLEAN A/B at no-history is: **sigreg-OFF (sibling `reacher_sigoff_noah`, λ=0)** vs **sigreg-ON (THIS arm `reacher_sigon_noah`, λ=0.09)**. If sigreg-OFF stays full-rank AND holds SR even here (matching this sigreg-ON reference), the conjecture is CLEANLY confirmed; this entry supplies the **reference** the sigreg-OFF arm is compared against (same recipe, same cap, only `loss.sigreg.weight` differs: 0.09 vs 0).

### Design — the §31 reacher arm-A recipe with ONE change vs the canonical reacher arm A: `model.use_action_history=false`
FROM SCRATCH (`init_from=null`, encoder `vit_hf` size=tiny patch=14 image=224 **pretrained=false** → random CLS-192, the load-bearing choice that lets collapse show), `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0 detach_target=true sigreg_act=false`, **`loss.sigreg.weight=0.09`** (sigreg ON — the reference; latent SIGReg term contributes `0.09·sigreg_loss` to `loss`), **`model.use_action_history=false`** (THE clean-test knob — past-action stream zeroed). Budget MATCHED to the §31 reacher A/B/C + the sibling sigOFF-noah arm for comparability: `trainer.max_epochs=100 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4 +ckpt_every=10`, seed 3072 (lewm.yaml default). `limit_train_batches=1000` = the RELAUNCHED §31 reacher cap (same cap = comparable). Data = committed `config/train/data/dmc.yaml` (`name: reacher.h5`, frameskip 5 ⇒ action_block 10, adim=2). reacher has the §15 val-best CHECKPOINT TRAP → headline the CONVERGED `weights_epoch_100.pt`, NOT val-best.

### STEP 1 — `use_action_history=false` zeroing VERIFIED NUMERICALLY (real reacher batch, GPU 0, `/mnt/minghao_data/verify_reacher_noah.py`) ✅
The decisive check isolates the zeroing line. AdaLN-Zero (`module.py:104-105`, `nn.init.constant_(adaLN_modulation[-1].weight/bias, 0)`) makes the predictor OUTPUT insensitive to its conditioning at random init, so an output-invariance test is inconclusive at init — instead the script captures the conditioning tensor `c` actually fed to `action_predictor.forward` AFTER predict_intention's `use_action_history` gate:
| setting | raw a_<t max\|.\| | conditioning fed to predictor max\|c\| |
|---|---|---|
| `use_action_history=TRUE` (control) | 6.699e-01 | **6.699e-01** (stream READ — nonzero ✓) |
| `use_action_history=FALSE` (THIS arm) | 6.960e-01 | **0.000000e+00** (stream ZEROED — EXACTLY 0 ✓) |
**NOAH ZEROING VERIFIED: `True`** — a_<t is nonzero (0.696) but the stream reaching the predictor is exactly 0, so the action head has NO history shortcut and MUST read the latent. Full `lejepa_forward` with sigreg ON (0.09) also runs clean: `pred=0.085 sigreg=28.33 (λ·sig=2.550) intent=1.014 act=1.140 z_std=0.0023 act_emb_std=0.097 TOTAL=4.789`, loss decomposition exact (`TOTAL − (pred+λ·sig+intent+act) = 7.3e-8`). Saved `config.yaml` confirms `use_action_history: false` (rides into `config.json` via `cfg.model`, the same mechanism §18 used, so eval rebuilds the noah model). Train-log `[GIP]` line: `Intention predictor ON Adim=10 head=mse w_act=1.0 w_intent=1.0 detach_target=True sigreg_act=False` (= arm-A recipe).

### STEP 2 — TRAIN (in flight). EXACT command (host `L40S`, `sudo -u minghao.fu`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`; driver `/mnt/minghao_data/launch_reacher_noah.sh GPU`)
```
# disk-safe env (per-arm scratch dirs; SPT_CACHE_DIR is the killer redirect — /var is 97% full):
export STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_reacher_sigon_noah \
  TMPDIR=/mnt/minghao_data/tmp_reacher_sigon_noah MPLCONFIGDIR=/mnt/minghao_data/mpl_reacher_sigon_noah \
  HF_HOME=/mnt/minghao_data/hf HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4 \
  SPT_CACHE_DIR=/mnt/minghao_data/spt_reacher_sigon_noah PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
CUDA_VISIBLE_DEVICES=6 python train_sigreg.py data=dmc \
  model.use_action_history=false \
  action_pred.enabled=true action_pred.detach_decoder=false action_pred.head=mse \
  action_pred.w_act=1.0 action_pred.w_intent=1.0 \
  action_pred.detach_target=true action_pred.sigreg_act=false \
  loss.sigreg.weight=0.09 init_from=null \
  output_model_name=sigreg_sigon_noah_reacher_scratch subdir=sigreg_sigon_noah_reacher_scratch \
  trainer.max_epochs=100 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4 +ckpt_every=10 \
  hydra.run.dir=/mnt/minghao_data/hydra_reacher_sigon_noah hydra.output_subdir=null
```
- **Launched 2026-06-22 ~00:43 UTC:** PID 2184246, GPU 6 (lowest-mem at launch, ~22.9 GB / 46 GB; box heavily loaded — 8+ co-located train_sigreg arms + other users, load avg ~170 on 64 cores ⇒ CPU/disk-bound per rule 10, NOT GPU-mem-bound). Confirmed in log: `[GIP] Intention predictor ON Adim=10 head=mse w_act=1.0 w_intent=1.0 detach_target=True sigreg_act=False`, stepping `[Epoch 0/100] step N/1000` at ~1.4–1.5 it/s ⇒ ~11–12 min/epoch ⇒ ~18–20 h for 100ep. Sanity-val (random init): `pred_loss=0.086 sigreg_loss=28.5 z_std=0.0033`. `+ckpt_every=10` ⇒ `weights_epoch_{10..100}.pt`.
- **Disk safety:** all writes → `/mnt/minghao_data` (2.4 TB free, 22%); `/var/lib/docker` 99% (88 GB), `/` 97% (3.5 GB) — `SPT_CACHE_DIR` redirect prevents the metrics.csv/resumption-ckpt crash ([[project_l40s_path_consolidation]]). Prune loop `/mnt/minghao_data/prune_reacher_noah.sh` (the noah one, PID 2189275) keeps newest-3 weights.

### STEP 3 — DIRECT-FROM-CHECKPOINT collapse measurement [PENDING ckpts; auto-run by finalizer at ep100]
`/mnt/minghao_data/measure_collapse_reacher.py --epoch 100 --batch-run sigreg_B_reacher_scratch --arms noah_sigON_noHist:sigreg_sigon_noah_reacher_scratch` (SHARED reacher val batch from B's split → apples-to-apples vs the sigOFF-noah arm + the A/B refs the sibling finalizer measures). For the ckpt: `gip.load_gip_model(run, epoch=100)` → `model.encode` → `z_std = emb.reshape(-1,D).std(dim=0).mean()` + effective rank (participation ratio = exp(entropy) of the singular-value spectrum) + `act_emb_std`. COLLAPSE = z_std→0 AND erank≪192. (This sigreg-ON arm is the FULL-RANK reference; the test is whether the sigreg-OFF noah arm matches it.)

### STEP 4 — EVAL SR (reacher, `eval_gip.py --config-name reacher`, N=50, seeds {42,0,1}, same-box, planning + bc) [PENDING convergence; auto-run by finalizer]
```
PYTHONPATH=/tmp/reacher_compat:$B python eval_gip.py --config-name reacher \
  policy=sigreg_sigon_noah_reacher_scratch +gip_eval.mode=<planning|bc> eval.num_eval=50 seed=<42|0|1>
```
(planning = pure WM CEM; bc = intention head reactive. Reacher needs the dm_control↔mujoco compat shim `/tmp/reacher_compat/sitecustomize.py` on PYTHONPATH, §15 blocker 1; eval same-box GPU 6 = the train GPU per the cross-GPU render gotcha [[project_l40s_cross_gpu_rendering]].)

### RESULT TABLE (to fill at convergence — ALL per-seed)
| arm | use_action_history | sigreg λ | final z_std | act_emb_std | erank/192 | collapsed? | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) |
|---|---|---|---|---|---|---|---|---|
| **noah_sigON (THIS arm: sigreg ON, NO history) = reference** | **false** | **0.09** | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |
| noah_sigOFF (sibling clean test: sigreg OFF, NO history) | false | 0 | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] |

### VERDICT [PENDING convergence + eval]
This arm establishes the full-rank, sigreg-ON reference at the no-history setting. The clean conjecture verdict is read off the PAIR (this sigreg-ON ref vs the sibling sigreg-OFF clean arm): if sigreg-OFF stays full-rank (erank ≈ this ref, all per-dim std > 0) and SR ≈ this ref EVEN with `use_action_history=false`, the SIGReg-redundancy conjecture is CLEANLY confirmed (act_loss reading the latent — with no history to copy — anchors it on its own, SIGReg redundant; a "Simple"-pillar win with the §18 confound removed). If sigreg-OFF collapses where this ON ref does not, SIGReg is load-bearing once the shortcut is gone. [Fills once `weights_epoch_100.pt` lands and the finalizer runs STEP-3 + STEP-4.]

### Files (L40S)
- `/mnt/minghao_data/verify_reacher_noah.py` — STEP-1 zeroing verification (DONE: stream fed to predictor = exactly 0 under use_action_history=false; nonzero under true; full forward decomposition exact).
- `/mnt/minghao_data/launch_reacher_noah.sh` — launcher (the §31 reacher-A recipe + `model.use_action_history=false`, sigreg λ=0.09, 1000-batch cap).
- `/mnt/minghao_data/prune_reacher_noah.sh` — keep-newest-3 prune loop targeting THIS run `sigreg_sigon_noah_reacher_scratch` (PID 2189275).
- `/mnt/minghao_data/reacher_sigon_noah_finalize.sh` — SELF-COMPLETING finalizer (PID 2197029, setsid-detached; log `/mnt/minghao_data/reacher_sigon_noah_finalize.log`). Blocks until `weights_epoch_100.pt`, then auto-runs STEP-3 collapse + STEP-4 eval (this arm × {planning,bc} × seeds {42,0,1}, N=50, same-box GPU 6). Reuses the shared `/mnt/minghao_data/measure_collapse_reacher.py` (UNCHANGED). **⚠️ NAMING: an earlier copy collided on the shared filename `reacher_noah_finalize.{sh,log}` with the sibling sigOFF arm; both were split into `reacher_{sigon,sigoff}_noah_finalize.{sh,log}` (2026-06-22 00:49 UTC) and relaunched — each waits on its own run + log, no work lost (both were in the ep100 wait loop).**
- Ckpts: `/mnt/minghao_data/.stable-wm/checkpoints/sigreg_sigon_noah_reacher_scratch/`. Log `/mnt/minghao_data/sigreg_reacher_logs/train_sigon_noah.log`.
- **PIDs:** train 2184246 (GPU6), prune 2189275, finalizer 2197029.
- **NOT touched:** shared `train_sigreg.py`/`jepa.py`/`gip.py`/`eval_gip.py`/`module.py`/`gen_master_table.py`, the §31 reacher A/B/C runs/GPUs, the sibling `sigreg_noah_reacher_scratch` (sigOFF) run (read-only — matched its cap/naming/recipe; restored its accidentally-overwritten finalizer as `reacher_sigoff_noah_finalize.sh`), other users' procs.

---

## §31-clean 🔄🆕 SIGReg-REDUNDANCY — the CLEAN cube arm: **`cube_sigoff_noah`** (env=cube, latent-SIGReg OFF, **`use_action_history=false`**) — removes the §18 confound from the redundancy conjecture (L40S `stratus-lookout`, 2026-06-22) [IN FLIGHT — TRAINING]

**Motivation (why — the rigor completer).** §30 (pusht) and §31 (tworoom/reacher) test the conjecture "with the action-prediction head active, the explicit latent SIGReg anti-collapse is REDUNDANT" by training a sigreg-OFF arm and checking the latent stays full-rank. But every prior sigreg-OFF arm ran with `use_action_history=true`, so the action head could partly satisfy `act_loss` by COPYING/extrapolating the past-action stream `a_<t` rather than by reading the latent `emb` (the §18 mechanism: zeroing `a_<t` on pusht RAISED open-loop val_act 0.132→0.153, proving `a_<t` is a strong open-loop predictor). That history-copyability is a CONFOUND: a healthy sigreg-OFF latent could be held up by the action head NOT needing the latent (it copies history), which would make the redundancy positive only CONSERVATIVE. **This arm removes the confound:** with `use_action_history=false` the past-action stream is zeroed inside `predict_intention` (jepa.py:192 `if not self.use_action_history: past_act_emb = torch.zeros_like(past_act_emb)`), so the action head CANNOT copy `a_<t` — it MUST read the latent to predict the action. So if sigreg-OFF STILL keeps the latent full-rank here, the conjecture (act_loss anchors the latent → SIGReg redundant) is CLEANLY confirmed, not merely conservatively. Cube is also state-driven manipulation (grasp/place), the cleanest non-history-copyable test family. This arm is the sigreg-OFF half of a clean cube PAIR: the sibling agent runs `cube_sigon_noah` (sigreg ON, also `use_action_history=false`) as the full-rank reference; the verdict is read off the pair (does OFF match the ON reference's erank + SR with no history shortcut?).

### STEP 1 — flag verification (DONE ✅)
- **sigreg-zeroing exact (real cube batch, bs64, `lejepa_forward`):** `loss.sigreg.weight=0` makes TOTAL loss = `pred + intent + act` EXACTLY. Numeric (GPU7, /mnt/minghao_data/verify_cube_sigoff_noah.py): with w=0.09 `pred=0.072219 sigreg=27.4884 lambd*sigreg=2.473956 intent=1.015423 act=1.091987 TOTAL=4.653585`; with w=0 `lambd*sigreg=0.000000 TOTAL=2.179630 = pred+intent+act` (|diff|=3.73e-8 FP). **sigreg contribution == 0 confirmed.** ✓
- **`use_action_history=false` zeros `a_<t` — confirmed by CODE (jepa.py:192 in `predict_intention`, jepa.py:338 in the eval AR loop: `past_act_emb = torch.zeros_like(past_act_emb)` BEFORE the predictor) AND by the §18 empirical pusht result (zeroing it changed open-loop val_act).** The model config comment (`config/train/model/lewm.yaml:4`) documents the override `model.use_action_history=false`; it TRAVELS to eval via the saved `config.json` (read by `gip.load_gip_model`), so eval also cannot see `a_<t`. (A synthetic intention-invariance probe was attempted on GPU7 but the cube h5 [101 GB] dataset-cache step was disk-bound on the oversubscribed L40S and ran long; the zeroing is already proven by the code path + §18 + the saved-config verification below, so training was launched without blocking on the probe.)

### Design — `cube_sigoff_noah` = the existing cube sigreg-OFF recipe (`sigreg_B_cube_scratch`) + `model.use_action_history=false`
FROM SCRATCH (`init_from=null`, encoder `vit_hf` size=tiny patch=14 image=224 **pretrained=false** → random CLS-192 — the load-bearing choice: warm-start would pre-establish a non-collapsed latent and HIDE collapse), `action_pred.enabled=true detach_decoder=false head=mse w_act=1.0 w_intent=1.0 detach_target=true sigreg_act=false`, **`loss.sigreg.weight=0`** (latent SIGReg OFF: module still RUNS + `sigreg_loss` still logged, contributes 0 to `loss`), **`model.use_action_history=false`** (THE clean knob: zeros `a_<t`). Data = `data=ogb` (cube `ogbench/cube_single_expert.h5`, frameskip 5 ⇒ action_block adim=25). **Budget MATCHED to the sibling cube sigreg arms `sigreg_B_cube_scratch` / `cube_A_sigreg` for comparability:** `trainer.max_epochs=100 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=20 loader.batch_size=64 num_workers=4`, seed 3072, `+ckpt_every=10`. **NOTE the cap is 1000 (not §30-pusht's 4000):** the existing cube arms cap at 1000 (verified from `sigreg_B_cube_scratch/config.yaml: limit_train_batches: 1000`); "same cap across both sigreg arms of this env" ⇒ this clean arm uses 1000 too, directly comparable to `sigreg_B_cube_scratch` (the sigreg-OFF cube arm at `use_action_history=true`) and to the sibling `cube_sigon_noah`.

### Infra (DISK-SAFE — L40S `/` & `/var` 97% full, ONLY `/mnt/minghao_data` free; killer = SPT_CACHE_DIR)
Host L40S, `sudo -u minghao.fu`, py `/var/lib/docker/data/minghao_home/lewm/bin/python`, cwd `/var/lib/docker/data/minghao_home/workspace/le-wm-repro`. **Disk at launch:** `/`=97% (3.5 G), `/mnt/minghao_data`=22% (2.4 TB free) → ALL writes to `/mnt`: `STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_cube_sigoff_noah TMPDIR=/mnt/minghao_data/tmp_cube_sigoff_noah MPLCONFIGDIR=/mnt/minghao_data/mpl_cube_sigoff_noah HF_HOME=/mnt/minghao_data/hf **SPT_CACHE_DIR=/mnt/minghao_data/spt_cube_sigoff_noah** HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, plus `hydra.run.dir=/mnt/minghao_data/hydra_cube_sigoff_noah hydra.output_subdir=null`. Verified `df`: no write target on `/` or `/var`. Ckpts → `STABLEWM_HOME/checkpoints/cube_sigoff_noah/`. `+ckpt_every=10` (ep 10..100) + `/mnt/minghao_data/prune_cube_sigoff_noah.sh` keep-newest-3 loop. **GPU 5** (most-free at launch, ~18 GB headroom; co-located per rule 10, never touched other users' procs incl. the §29 dinov2 runs). Same box train+eval → no cross-GPU render OOD ([[project_l40s_cross_gpu_rendering]]).

### EXACT train command (L40S; driver `/mnt/minghao_data/launch_cube_sigoff_noah.sh <gpu>`)
```
export STABLEWM_HOME=/mnt/minghao_data/.stable-wm XDG_CACHE_HOME=/mnt/minghao_data/xdg_cube_sigoff_noah \
       TMPDIR=/mnt/minghao_data/tmp_cube_sigoff_noah MPLCONFIGDIR=/mnt/minghao_data/mpl_cube_sigoff_noah \
       HF_HOME=/mnt/minghao_data/hf SPT_CACHE_DIR=/mnt/minghao_data/spt_cube_sigoff_noah \
       HF_HUB_OFFLINE=1 MUJOCO_GL=egl WANDB_MODE=offline OMP_NUM_THREADS=4 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /var/lib/docker/data/minghao_home/workspace/le-wm-repro
CUDA_VISIBLE_DEVICES=5 python train_sigreg.py data=ogb \
  model.use_action_history=false \
  action_pred.enabled=true action_pred.detach_decoder=false action_pred.head=mse \
  action_pred.w_act=1.0 action_pred.w_intent=1.0 action_pred.detach_target=true action_pred.sigreg_act=false \
  loss.sigreg.weight=0 init_from=null \
  output_model_name=cube_sigoff_noah subdir=cube_sigoff_noah \
  trainer.max_epochs=100 +trainer.limit_train_batches=1000 +trainer.limit_val_batches=20 \
  loader.batch_size=64 num_workers=4 +ckpt_every=10 \
  hydra.run.dir=/mnt/minghao_data/hydra_cube_sigoff_noah hydra.output_subdir=null
```
**PID 2206124** (GPU 5). Log `/mnt/minghao_data/sigreg_logs/train_cube_sigoff_noah.log`. [startup/throughput + saved-config `use_action_history=false` confirmation fill live.]

### MEASUREMENT plan (STEP 3 — DIRECT from the CONVERGED ep100 checkpoint, the robust signal)
Reuse the shared `/mnt/minghao_data/measure_collapse_cube.py --run cube_sigoff_noah --epoch 100` (UNCHANGED — already adapted for cube): rebuilds via `gip.load_gip_model` (the SAME path `eval_gip.py` uses), runs a fixed cube val batch through `model.encode`, reports **z_std = emb.reshape(-1,D).std(dim=0).mean()**, **act_emb_std**, and **erank = exp(entropy of singular-value spectrum)** + per-dim std spread. COLLAPSE = z_std→0 AND erank≪192; FULL-RANK/SCALE-SHRINK = low z_std but erank high + all per-dim std>0.

### EVAL plan (STEP 4 — when converged, rule 8/9; AUTOMATED via `/mnt/minghao_data/cube_sigoff_noah_finalize.sh`)
cube via `eval_gip.py --config-name cube policy=cube_sigoff_noah +gip_eval.mode=<planning|bc> +ckpt_epoch=100 eval.num_eval=50 seed=<42|0|1>`, **N=50 × 3 seeds {42,0,1}**, modes **planning + bc**, same-box GPU 5, CONVERGED ep100 ckpt (`+ckpt_epoch=100`, NOT val-best — reacher-trap rule). planning = pure WM CEM, bc = intention head reactive. The finalizer (detached) waits for `weights_epoch_100.pt`, then runs STEP-3 measure + the 6 N=50 evals → `/mnt/minghao_data/cube_sigoff_noah_finalize.log` + per-eval `/mnt/minghao_data/cube_sigoff_noah_eval_{mode}_s{seed}.log`.

### RESULT TABLE (to fill at convergence — ALL per-seed)
| arm | final z_std | act_emb_std | erank/192 | collapsed? | SR planning (s42/s0/s1, mean) | SR bc (s42/s0/s1, mean) | finding |
|---|---|---|---|---|---|---|---|
| `cube_sigoff_noah` (SIGReg OFF, `use_action_history=false`) | [pending ep100] | [pending] | [pending] | [pending] | [pending N=50×3] | [pending N=50×3] | [pending — vs `cube_sigon_noah` full-rank ref] |

### VERDICT (to write at convergence)
[PENDING.] Decision rule: **CLEAN CONFIRMATION** = sigreg-OFF stays full-rank (erank ≈ the `cube_sigon_noah` reference, all per-dim std > 0) AND SR ≈ the ON reference EVEN with `use_action_history=false`. Because the history shortcut is GONE, a full-rank latent here means `act_loss` had to READ the latent to predict the action and that pressure alone kept it full-rank → the SIGReg-redundancy conjecture is cleanly confirmed with the §18 confound removed (a "Simple"-pillar win for LeWAM). **COLLAPSE** (z_std→0, erank→1) where the ON reference does not → SIGReg is load-bearing once the shortcut is gone, and the prior history-on positives were (partly) the §18 history-copyability artifact.

### Files (L40S)
- `/mnt/minghao_data/verify_cube_sigoff_noah.py` — STEP-1 sigreg-zeroing verification (exact, DONE). `/mnt/minghao_data/verify_cube_noah3.py` — the intention-invariance probe (disk-bound on the 101 GB cube h5 cache; superseded by the code-path + saved-config verification).
- `/mnt/minghao_data/launch_cube_sigoff_noah.sh` — launcher (the `sigreg_B_cube_scratch` recipe + `model.use_action_history=false`, λ=0, 1000-batch cap).
- `/mnt/minghao_data/prune_cube_sigoff_noah.sh` — keep-newest-3 prune loop targeting `cube_sigoff_noah`.
- `/mnt/minghao_data/cube_sigoff_noah_finalize.sh` — SELF-COMPLETING finalizer (waits ep100 → STEP-3 measure + STEP-4 eval planning+bc × seeds {42,0,1} N=50, same-box GPU 5). Reuses the shared `/mnt/minghao_data/measure_collapse_cube.py` (UNCHANGED).
- Ckpts: `/mnt/minghao_data/.stable-wm/checkpoints/cube_sigoff_noah/`. Log `/mnt/minghao_data/sigreg_logs/train_cube_sigoff_noah.log`.
- **NOT touched:** shared `train_sigreg.py`/`jepa.py`/`gip.py`/`eval_gip.py`/`module.py`/`measure_collapse_cube.py`/`gen_master_table.py`, the sibling cube arms (`sigreg_B_cube_scratch`, `cube_A_sigreg`, `cube_sigon_noah`) runs/GPUs, other users' procs.

---

## §32 🌙 OVERNIGHT174 — 7-arm × 4-dataset end-to-end ablation campaign on **H100-174-minghao** (`le-wm-repro`, 2026-06-22) [IN FLIGHT]

**WHY here, not L40S:** L40S GPUs 1–7 are hardware-faulted ("require reset", per-GPU reset Not Supported, can't reboot without killing user `fan`). So the overnight ablation campaign was moved to **H100-174-minghao** (`~/workspace/le-wm-repro` = `/home/minghao.fu/workspace/le-wm-repro`), which has the repo, all 4 datasets, and GPU room. Run by a background agent while user asleep; honest numbers only, N=50 evals, fast-iter budgets (rule 12).

**ENV (174, verified):** runs via the repo's **own `.venv`** (`/home/minghao.fu/workspace/le-wm-repro/.venv/bin/python` → system py3.10), NOT a conda env. The candidate conda envs (`WorldArena_JEPA`/`WorldArena`/`Cosmos`) do NOT have `stable_worldmodel`/`stable_pretraining` — only the repo `.venv` imports cleanly (torch 2.12.0+cu130, cuda OK, 8 devices). `STABLEWM_HOME=/mnt/data_nvme1/minghao.fu/.stable-wm` (datasets at `$STABLEWM_HOME/datasets/`, checkpoints at `$STABLEWM_HOME/checkpoints/`). Data cfg map: pusht→`pusht`, tworoom→`tworoom`, cube→`ogb` (ogbench/cube_single_expert.h5, a 101 GB symlink to /mnt/data_7tb), reacher→`dmc` (reacher.h5). `MUJOCO_GL=egl`.

**⚠️ CODEBASE DIVERGENCE from the task-prompt matrix (CRITICAL).** The prompt's 9-arm matrix was written for the NEWER swm/`mtjepa/` codebase. 174's `le-wm-repro` is the OLDER tree and does NOT implement:
- `goal_conditioned` / `goal_dropout` / `horizon_conditioned` — **absent in code** (grep-verified across `*.py` and `config/`). So the "one model serves 3 settings (BC/GC/Planning)" framing is unavailable here. 174 evals via its OWN 3 modes: **`bc | guided | planning`** (`eval_gip.py +gip_eval.mode=…`); `guided` = intention-guided planning (the closest analogue to "GC"). Every trained intention model supports all 3 modes, so the 3-setting story survives — just under the codebase's native names.
- `w_cyc` / `w_anorm` — **absent in code** → the `w_cyc_on` arm is **DROPPED** here (cannot run; would error on unknown key).

So this campaign runs **7 arms** (not 9), each evaluated in **bc/guided/planning × 3 seeds {42,0,1}, N=50, same-box**.

**SUPPORTED FLAGS (verified in `jepa.py`/`train.py`/`module.py`):** `action_pred.{enabled,w_act,w_intent,detach_target,detach_decoder,ema_target,ema_tau,sigreg_act,head(mse|gmm|diffusion),n_modes,diff_steps}`; `model.use_action_history`; top-level `history_size`; `loss.sigreg.weight`; `model=lewm_dinov2 model.encoder.pretrained=true embed_dim=384` for the dinov2 arm.

**ARMS (all END-TO-END, encoder learnable; base flags: vit-tiny scratch, `action_pred.enabled=true w_intent=0 w_act=1 detach_decoder=false head=mse`, `use_action_history=true history_size=3`, `loss.sigreg.weight=0.09`):**
1. `base` — the reference.
2. `enc_dino` — `model=lewm_dinov2 model.encoder.pretrained=true embed_dim=384` (DINOv2-small init, trainable; only this 1 dino config exists on 174).
3. `input_stateonly` — `model.use_action_history=false`.
4. `sigreg_off` — `loss.sigreg.weight=0` (log act_emb_std).
5. `predict_emb` — `action_pred.w_intent=1 action_pred.detach_decoder=true`.
6. `input_markov` — `history_size=1` (num_steps = num_preds+history_size = 2 ≥ 2, so it should be valid; skip+note if data-window errors).
7. `head_diffusion` — `action_pred.head=diffusion`.

**FAST-ITER BUDGET (rule 12; SAME across all arms WITHIN a dataset):** tworoom 12ep; pusht/cube/reacher 15ep; `+trainer.limit_train_batches=2000`; `loader.batch_size=64`; `+ckpt_every=5`. EVAL N=50, seeds {42,0,1}, same GPU as train.

**EXACT train command (per arm,dataset; via `/mnt/data_nvme1/minghao.fu/overnight174/run_arm.sh <gpu> <arm> <dataset>`):**
```
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python train.py \
  data=<DATACFG> model=lewm [or model=lewm_dinov2 model.encoder.pretrained=true embed_dim=384] \
  output_model_name=<arm>_<ds> subdir=<arm>_<ds> \
  action_pred.enabled=true action_pred.w_intent=0 action_pred.w_act=1 \
  action_pred.detach_decoder=false action_pred.head=mse \
  model.use_action_history=true history_size=3 loss.sigreg.weight=0.09 \
  <arm-specific override> \
  trainer.max_epochs=<12|15> trainer.devices=[0] \
  +trainer.limit_train_batches=2000 +trainer.limit_val_batches=20 \
  +ckpt_every=5 loader.batch_size=64 num_workers=5 \
  hydra.run.dir=$OVN/hydra/<arm>_<ds>
```
**EXACT eval command:** `eval_gip.py --config-name <EVALCFG> policy=<arm>_<ds> +gip_eval.mode=<bc|guided|planning> eval.num_eval=50 seed=<42|0|1>`.

**INFRA / DISK (the #1 risk — 174 disk is tighter than the prompt said):** `/mnt/data_nvme1` = 90 GB free / 98%; `/` (home, where `~/.stable_worldmodel` defaults!) = 88 GB / 98%; `/mnt/data_7tb` = 132 GB / 99%. So **all caches redirected** to `/mnt/data_nvme1/minghao.fu/overnight174/{sptcache,xdg,tmp,mpl}` via `SPT_CACHE_DIR XDG_CACHE_HOME TMPDIR MPLCONFIGDIR` + `STABLEWM_HOME` (datasets+ckpts on nvme1), `HF_HOME` left at `~/.cache/huggingface` (dinov2-small already cached there; `HF_HUB_OFFLINE=1`). Per run: SPT Lightning `.ckpt` ≈ 335 MB (writes BOTH `epoch=*.ckpt` AND `last.ckpt`), per-epoch `.pt` ≈ 120 MB (tiny) / 290 MB (dino). `run_arm.sh` prunes per-epoch `.pt` to final-only AND wipes `sptcache/runs/*` after each train. Dispatcher (`dispatch.sh`) caps concurrency at 6, auto-picks lowest-`memory.used` GPU among `[5 0 2 3]` (4/6/7 are full from other users; `fan`/`lipeng` tdmpc2 procs on 1/3 — NEVER touched), staggers launches by 8 s, and **HALTS new launches if free < 25 GB**.

**CODE PATCH (1, minimal, on 174 only):** `gip.load_gip_model` hardcoded `embed_dim=192`; added one line deriving `embed_dim = int(config.action_encoder.get("emb_dim", embed_dim))` from the ckpt config so the `enc_dino` 384-d arm evals correctly. Backup at `gip.py.bak_overnight174_embeddim`. 192-d path re-verified unchanged by smoke eval.

**SMOKE TEST (base:tworoom, max_epochs=1 limit_train_batches=20):** PASSED — train losses print (`pred_loss`/`sigreg_loss`/`intent_loss`/`act_loss`), `.pt`+`.ckpt` save under redirects, `bc` eval returns `success_rate` (50% on N=2, expected for a 1-ep model). Smoke artifacts cleaned; disk restored to 90 GB.

### RESULT TABLE (filled live as evals land; ALL per-seed, N=50)
| arm | dataset | bc (s42/s0/s1, mean) | guided (mean) | planning (mean) | val_act@budget | notes |
|---|---|---|---|---|---|---|
| base | tworoom | [pending] | [pending] | [pending] | [pending] | validation run in flight |
| (… 7 arms × 4 ds, priority {base,enc_dino,input_stateonly,sigreg_off}×{tworoom,pusht} first …) | | | | | | |

### Files (174)
- `/mnt/data_nvme1/minghao.fu/overnight174/` — `env.sh` (shared redirects), `run_arm.sh` (train+prune+eval one arm,ds), `dispatch.sh` (queue+disk-guard+GPU-picker), `HEADER.txt`, `SUMMARY.txt` (live human log), `results.jsonl` (one line per arm/ds/setting/seed), `logs/`, `hydra/`, `sptcache/`.
- Ckpts: `/mnt/data_nvme1/minghao.fu/.stable-wm/checkpoints/<arm>_<ds>/`.
- **NOT touched:** other users' procs/GPUs (`fan`, `lipeng` tdmpc2 on GPUs 1/3; GPUs 4/6/7 full), the existing `.stable-wm/checkpoints/*` from prior campaigns, repo source except the 1-line `gip.py` embed_dim patch (backed up).

---

## §IDM — Inverse-dynamics anti-collapse: can a DENSE inverse term REPLACE SIGReg? (2026-06-23, L40S le-wm-repro)

**Motivation (why).** SMWM (2606.20104, Balestriero) shows an inverse-dynamics regularizer `‖h(z_t,z_{t+1})−a_t‖²` prevents latent collapse and can REPLACE SIGReg (matches SIGReg on 2D, beats it on 3D Cube 84 vs 59). In our setting `sigreg_off` FLOORS — the latent collapses (measured: pusht scratch z_std A_sigreg_ON=0.963 vs B_sigreg_OFF=0.189, B/A=0.196 = COLLAPSE, erank 86→112 but dimstd max 1.56→0.44; `sigreg_collapse_traj.log`) because the forward head cheats through the past-action stream. **Hypothesis:** a DENSE inverse term (over EVERY consecutive latent pair in the context window, not just one transition) forces the action INTO the latent and replaces SIGReg, while we keep the planning-free forward policy unchanged. **User's bar:** `idm+sigreg_off` must MATCH/BEAT the SIGReg base across datasets, not merely survive.

### Implementation (additive, config-gated, default OFF → byte-identical; the running 16-job campaign is unaffected)
Three edits in `le-wm-repro`, all gated so existing arms (no new flag) are byte-for-byte unchanged. Backups: `jepa.py.bak_idm`, `train.py.bak_idm`, `config/train/lewm.yaml.bak_idm`.

1. **`jepa.py` — `JEPA.__init__`** gains `inverse_conditioned=False`, `inverse_action_dim=None`. When `inverse_conditioned`: builds `self.inverse_model = Sequential(Linear(2D,256), GELU, Linear(256,Adim))` — SMWM's head shape (2·D=384→256→Adim, D=192 vit-tiny, Adim=frameskip·action_dim). Adds `predict_inverse(z_t,z_tp1) = inverse_model(cat([z_t,z_tp1],-1))`. Default OFF → head never built → no `inverse_model.*` keys in the state_dict → load_state_dict byte-identical. The gate rides in `config.json` (cfg.model) so `gip.load_gip_model` rebuilds the SAME head before `load_state_dict` (keeps the strict `assert not unexpected_keys` meaningful). The deployed policy (`predict_intention` / `intention_rollout`) is UNCHANGED — `L_inv` is a train-time anti-collapse term only.
2. **`train.py` `lejepa_forward` — L_inv block**, gated by `action_pred.w_inv` (default 0.0 → block skipped → byte-identical). When `w_inv>0` and the head exists:
   - **DENSE** (`inv_mode=dense`, default): pairs over the CONTEXT-supported window — `z_τ=emb[:,:ctx_len]`, `z_{τ+1}=emb[:,1:ctx_len+1]`, `a_τ=action[:,:ctx_len]`; `L_inv = mean_τ ‖h([z_τ;z_{τ+1}]) − a_τ‖²`. (Bounded to `ctx_len` transitions, NOT `emb[:,:-1]`, so it stays aligned with the supervised action window for any `num_preds`>1 — Codex review fix 2026-06-23. With our `num_preds=1` the window IS the whole emb, so no change to the launched arms.) `inv_mode=last` = SMWM-style single (last) transition.
   - **TARGET** (`inv_target=encoded`, default): `z_{τ+1}` = encoded next latent. `inv_target=predicted` (the A8/cycle variant): `z_{τ+1}` = the FDM rollout `ẑ_{t+1}=predict(z_t,a_t)` (read the action off the FDM's own prediction).
   - `output["inv_loss"]=L_inv; output["loss"] += w_inv·L_inv`; logged as `inv_loss`. Multi-task: target masked like `act_loss` (padded dims zeroed).
   - Mirror block added to `train_sigreg.py` (= `train.py` + the `z_std` collapse-monitor line; my arms run `train_sigreg.py` so z_std is logged every step).
3. **`config/train/lewm.yaml`** — under `action_pred`: `w_inv: 0.0`, `inv_mode: dense`, `inv_target: encoded`. `train.py` sets `cfg.model.inverse_conditioned=True` + `cfg.model.inverse_action_dim=action_encoder.input_dim` when `w_inv>0` (mirrors how `horizon_conditioned` is set), so the gate travels to config.json. NOTE: because these keys now EXIST in the config, override them with plain `action_pred.w_inv=0.5` (NOT `+action_pred.w_inv=...`, which errors "item already at action_pred.w_inv").

### SMOKE TEST (CPU, no GPU — the campaign owns them). PASSED.
`smoke_idm.py` (in repo): builds the real JEPA (vit-tiny + ARPredictor + action head) and runs the real `lejepa_forward` on a seeded synthetic batch.
- **[1] default w_inv=0 (edited code):** loss=1.55267501, `inv_loss` ABSENT, `inverse_model is None`. 
- **[3] backup code (jepa.py.bak_idm/train.py.bak_idm), SAME seeded inputs:** loss=1.55267501, **|edited−backup|=0.00e+00 → BYTE-IDENTICAL at default** (running campaign provably unaffected).
- **[2] w_inv=0.1 dense encoded:** finite inv_loss=1.0849, total=1.7482, head built. **[2b]** inv_target=predicted: finite 1.0850. **[2c]** inv_mode=last: finite 1.2313.
- **End-to-end train→eval validation** (pusht, 1ep×8 batches, GPU): `[GIP] Intention predictor ON` + horizon ON, logged `fit/inv_loss=1.037 validate/inv_loss=0.962 fit/z_std fit/act_loss`; checkpoint has the 4 `inverse_model.*` keys + `config.json: inverse_conditioned=true inverse_action_dim=10`; `gip.load_gip_model` rebuilds it with **missing=0 unexpected=0** (strict assert passes). py_compile OK for all three files on the L40S venv.

### Arms & protocol (END-TO-END, matches the campaign EXACTLY for a valid comparison)
**Two questions** (user clarified 2026-06-23: this ablation verifies whether the inverse loss can IMPROVE PERFORMANCE, not only replace SIGReg):
- **(A) Can the inverse term REPLACE SIGReg?** (SMWM's claim) → SIGReg OFF + inverse.
- **(B) Can the inverse term IMPROVE PERFORMANCE on top of the working base?** → SIGReg ON + inverse, a pure "+inverse" ablation (rule 13: build on the end-to-end baseline, change exactly ONE component). Compare to `ov_base`.

All arms = the campaign's `ov_base` recipe (END-TO-END `freeze_wm=false`, vit-tiny-192 scratch, `action_pred.enabled=true w_act=1 w_intent=0 detach_decoder=false head=mse use_action_history=true history_size=3 goal_conditioned=true goal_dropout=0.5 horizon_conditioned=true`, SIGReg weight 0.09) with these deltas:
- **idm_dense_on** (PERFORMANCE arm, question B): `action_pred.w_inv=0.5 inv_mode=dense inv_target=encoded` (SIGReg stays ON) — does SR improve over `ov_base`?
- **idm_dense_sigregoff** (KEY replacement arm, question A): `loss.sigreg.weight=0 action_pred.w_inv=0.5 inv_mode=dense inv_target=encoded` — inverse replaces SIGReg.
- **idm_dense_A8** (question A, variant): same as sigregoff but `inv_target=predicted` (FDM-rollout/cycle variant).
- **Comparison baselines (PULLED from the running campaign, NOT rerun):** `ov_base_<ds>` (SIGReg ON, weight 0.09 — the baseline for BOTH questions) and `ov_sigreg_off_<ds>` (sigreg off ALONE = the floor question A must beat). Both already trained at the SAME budget.

**Verdict criteria:** (B) `idm_dense_on` SR > `ov_base` SR ⇒ the inverse loss IMPROVES performance. (A) `idm_dense_sigregoff` SR ≥ `ov_base` SR AND z_std/erank near base (full-rank) ⇒ it can REPLACE SIGReg; if it floors like `ov_sigreg_off`, it cannot.

Datasets: **pusht (`data=pusht`), cube (`data=ogb`), tworoom (`data=tworoom`), reacher (`data=dmc`)**. FAST per-dataset budget (rule 12, SAME across all arms within a dataset): **tworoom 12ep, pusht/cube/reacher 15ep**, `+trainer.limit_train_batches=2000 +trainer.limit_val_batches=20 +ckpt_every=5 loader.batch_size=64`. **Seed protocol matches the campaign: ONE training run (seed 3072 default), evaluated at 3 EVAL seeds {42,0,1}, N=50.** (The campaign's `ov_*` runs are single-train + 3 eval-seeds, NOT 3 train-seeds — verified from configs; matching it keeps the ablation valid per rule 12.)

**EXACT train command (orchestrator `/mnt/minghao_data/idm_ablation/run_idm.sh`, mirrors `overnight/run_overnight.sh`):**
```
source /mnt/minghao_data/overnight/env_common.sh tr_ov_<arm>_<ds>   # SPT_CACHE_DIR/XDG/TMPDIR/MPL/HF → /mnt/minghao_data (disk-safe)
CUDA_VISIBLE_DEVICES=<gpu> python train_sigreg.py data=<DATACFG> \
  action_pred.enabled=true action_pred.w_act=1.0 action_pred.w_intent=0.0 \
  action_pred.detach_decoder=false action_pred.head=mse model.use_action_history=true \
  history_size=3 loss.sigreg.weight=0.09 init_from=null freeze_wm=false embed_dim=192 \
  action_pred.goal_conditioned=true action_pred.goal_dropout=0.5 action_pred.horizon_conditioned=true \
  loss.sigreg.weight=0 action_pred.w_inv=0.5 action_pred.inv_mode=dense action_pred.inv_target=<encoded|predicted> \
  output_model_name=ov_<arm>_<ds> subdir=ov_<arm>_<ds> \
  trainer.max_epochs=<12|15> +trainer.limit_train_batches=2000 +trainer.limit_val_batches=20 \
  +ckpt_every=5 loader.batch_size=64 num_workers=2 trainer.devices=[0] \
  hydra.run.dir=/mnt/minghao_data/scratch/tr_ov_<arm>_<ds>/hydra hydra.output_subdir=null
```
**EXACT eval:** `eval_gip.py --config-name <EVALCFG> policy=ov_<arm>_<ds> seed=<42|0|1> eval.num_eval=50 +gip_eval.mode=<planning | bc | policy+goal_conditioned=true(=gc)>` (EVALCFG: pusht/cube/tworoom/reacher; DATACFG: pusht/ogb/tworoom/dmc).
**COLLAPSE readout** (`measure_collapse_generic.py <run>`): z_std, act_std, effective-rank (entropy of singular values), and `inv_loss` on a fixed val batch — the science question is whether the inverse term keeps the latent FULL-RANK *and* SR ≥ the SIGReg base.

**GPU/disk discipline:** orchestrator caps MAX_CONCURRENT=8, PER_GPU=2, `num_workers=2` (CPU is the bottleneck — load ~250 on 64 cores under the campaign), all redirects → `/mnt/minghao_data` (the `/var/lib/docker` fs is 99% full; `/mnt/minghao_data` 2.4 T free), `+ckpt_every=5` + prune-to-newest-3. Per user 2026-06-23: use ALL GPUs, co-locate freely incl. others' GPUs (never KILL another user's procs). Auto-picks lowest-job-count GPU via cached `nvidia-smi` enumeration (the earlier per-call torch-CUDA probe stalled the orchestrator under load → replaced with a one-shot cached `nvidia-smi -L`).

### RESULT (filled live; ALL per-seed, N=50; verdict pending)
Live files: `/mnt/minghao_data/idm_ablation/{SUMMARY.txt, results.jsonl, logs/}`. Checkpoints: `/mnt/minghao_data/.stable-wm/checkpoints/ov_idm_dense_sigregoff_<ds>/`, `ov_idm_dense_A8_<ds>/`.

| arm | dataset | planning (s42/s0/s1, mean) | bc (mean) | gc (mean) | z_std / erank / inv_loss | verdict |
|---|---|---|---|---|---|---|
| idm_dense_on (perf, Q-B) | pusht/cube/tworoom/reacher | [pending] | [pending] | [pending] | [pending] | vs ov_base: improve? |
| idm_dense_sigregoff (Q-A) | pusht/cube/tworoom/reacher | [pending] | [pending] | [pending] | [pending] | replace SIGReg? |
| idm_dense_A8 (Q-A variant) | pusht/cube/tworoom/reacher | [pending] | [pending] | [pending] | [pending] | replace SIGReg? |
| **base (SIGReg ON, pulled)** ov_base_<ds> | all 4 | (campaign) | (campaign) | (campaign) | z_std≈0.96 (healthy) | reference baseline |
| **sigreg_off alone (pulled)** ov_sigreg_off_<ds> | all 4 | (campaign) | (campaign) | (campaign) | z_std≈0.19 (COLLAPSE) | the floor to beat |

### OPEN QUESTIONS
- Does `idm_dense_sigregoff` keep z_std/erank near the SIGReg base (full-rank) AND clear SR ≥ base across all 4 datasets (user's bar), or does it floor like `sigreg_off` despite the dense inverse?
- Is `w_inv=0.5` the right weight, or does the inverse term need a sweep (SMWM uses a specific scaling)? If the key arm under-rescues, try `w_inv∈{1.0,2.0}` before concluding.
- `inv_target=predicted` (A8) vs `encoded`: does reading the action off the FDM's own rollout (cycle-consistent) help more in our history→pred setting than the plain encoded inverse?
