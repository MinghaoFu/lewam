# LeWAM handoff: migration manifest and inventory (2026-09-11)

For Minghao. The point of this document is the list of paths that must move with the project when the cluster or
this machine goes away: checkpoints, datasets, caches, code tarballs, entry scripts, job YAMLs, documents and
notes. Sections 1 and 2 are that list; section 3 is what is left to do; the appendices recap what each artifact
backs (results, data, code), for context.

Roots:
- `HROOT` = `/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam` (HDFS, the only mount a pod can read)
- `A2F` = `/mnt/hdfs/bi_algo_a2f/minghao.fu/lewam/data` (SSD mirror, devbox only; holds copies of a few datasets)
- `REPO` = `/home/tiger/lewam/.claude/worktrees/lewam-jointflow` = `git@github.com:MinghaoFu/lewam.git`, branch
  `lewam-jointflow` (HEAD `c461482` or later)
- `JOBS` = `/home/tiger/lewam_project/jobs` (Merlin YAMLs, the submit guard, copies of entry scripts)

---------------------------------------------------------------------------------------------------------------
## 1. The manifest: what to copy

`REPO/docs/handoff_manifest.txt` lists every path, one per line, tab-separated `tier  bytes  path  note`
(1,557 lines; `#` lines are comments; directories are copied whole). It was generated on 2026-09-11 by
`REPO/scripts/handoff_manifest.py`, which checks that every path exists and sums sizes; rerun it after any change
(it prints what is missing). The repo itself moves with git, not with the manifest.

| tier | what | size |
|---|---|---|
| 1-results | the 34 training run dirs behind every reported number (best, latest, full-state, snapshots, config, train log), all eval logs, the DP-T checkpoints, the published toolhang checkpoints and their eval logs, the LeWM release, the E3/E4 probe outputs | 26.9 GB |
| 1-code | the 11 code tarballs pods ran, the 692 entry scripts, helper scripts, DP task yamls, the diffusion_policy repo and env, the robosuite/dexmimicgen sim source, the LeWM code, the cube baselines code | 1.5 GB |
| 2-data | the 12 training h5 files in use, the drawer source file, the eval views, the multi-view link files, env metadata, the dataset README | 453.5 GB |
| 3-caches | the 17 preload caches the reported rows and the multi-view runs read (rebuildable from tier 2 with `scripts/make_preload_cache.py`: minutes for toolhang, hours for cube and pusht) | 730 GB |
| 4-devbox | `JOBS/`, the session memory directory, the coded job-name map | < 0.1 GB |

Copy (any host that mounts HDFS; keeps the absolute tree under DEST):
```
grep -v '^#' docs/handoff_manifest.txt | cut -f3 > /tmp/handoff_paths.txt
rsync -aR --files-from=/tmp/handoff_paths.txt / DEST
```
For HDFS-to-HDFS moves use `hdfs dfs -cp` on the same list. If space is short, the per-file minimum of tier 1 is
`jointflow_best.pt` + `jointflow_config.json` + `train.log` per run dir (about 2.1 GB for the 33 finished runs)
plus the eval logs (65 MB), the three DP-T checkpoints (1.2 GB), the published baselines (4.8 GB) and the LeWM
release (72 MB): about 9.5 GB. Tier 3 can be skipped entirely and rebuilt.

Not in the manifest on purpose: the caches no current row uses (about 2.0 TB under `preload_cache*/`), the DP-T
training replay buffers, older code tarballs (`HROOT/code/archive/`), the failure-video directories, and every
run trained before the AdaLN fix (`67e20e1`, 2026-09-06), whose numbers are invalid.

---------------------------------------------------------------------------------------------------------------
## 2. The paths, by category

Checkpoints (`HROOT/ckpts/`; each run dir holds `jointflow_best.pt` = the row, `jointflow_latest.pt`,
`jointflow_full.pt` = model + optimizer for resume, `snap_ep*.pt`, `jointflow_config.json`, `train.log`, `done`):
- TC arms: `jointflow_tc/tc_toolhang_fx_{mnm192,msig192,mfl192,mflsig192,mvsig192}_s42`,
  `jointflow_tc/tc_drawer_fx_{mnm192,msig192,mvsig192}_s42`, `jointflow_tc/tc_transport_fx_{mnm192,msig192,mvsig192}_s42`,
  `jointflow_tc/tc_twodoor2_fx_{flowvsig192,msevsig192}_s42` (E3a), and the running `jointflow_tc/tc_drawer_fx_mvsig192_3v_s42`.
- GR arms: `jointflow_gr_<cell>_<arm>/<arm>_s42` for pusht x {fx_nm192, fx_vsig192, fx_sig192, fx_fl192,
  fx_flsig192}, tworoom / pointmaze_large / cube / toolhang / drawer / transport x {fx_nm192, fx_vsig192},
  reacher_policy x {fx_nm192, fx_vsig192, fx_sig192}; their chained eval logs sit beside them (`*.log`).
- DP-T baselines: `wf8_uni/toolhang_dp_noprop/snap_ep120.ckpt`, `wf8_dp/drawer/latest.ckpt`,
  `wf8_dp/transport/epoch=0120-train_loss=0.0476.ckpt` (+ `wf8_dp/cube/latest.ckpt`, no row).
- DP-C trained on our caches (in progress): `dp_tc/dp_<cell>_dpc_s42/` (`dp_best.pt`, `dp_latest.pt`, `dp_full.pt`,
  `dp_config.json`, `train.log`).
- Published toolhang checkpoints and their eval logs: `official_baselines/tool_hang/` (bc_rnn, dp_cnn_train_0, eval).
- LeWM authors' release: `HROOT/code/lewm_main_eval/hf_release_native/{pusht,cube,reacher,tworooms}/`.
- Probe outputs: `probes_e3/`, `probes_e4/`.

Eval logs: `jf_grev/` (planner ladders, E7 grid, controls), `jointflow_tc/*.log` (`hb_`, `ev_`, `evtask_`,
`evshard_`, `evexp_`), `jointflow_gr_*/*.log`, `lewm_grid/`, `wf8_dp/{toolhang,drawer,transport}_eval/`,
`official_baselines/tool_hang/eval/` (with `base_frame/`).

Data (`HROOT/wf8/`): `train/{pusht,tworoom,pointmaze_large,cube,reacher_policy,toolhang,toolhang_eih,drawer,
drawer_3view,transport,transport_3view,twodoor2}.h5`, the external-link target of `drawer.h5` (listed in the
manifest), `train/_views/` (multi-view link files), `eval/` (eval views), `env/` (model xml h5 files, robomimic env
args), `README.md` (provenance and pod-side names). `A2F` holds copies of pusht, tworoom, cube and the old reacher.

Caches (`HROOT/preload_cache/<stem>/` and `HROOT/preload_cache_u8/<stem>/`, three files per tag: `frames.npy`,
`aux.npz`, `meta.json`): `pusht_expert_train_fs5_i224` (u8), `reacher_policy_fs5_i224` (u8), `tworoom_fs5_i224`,
`pointmaze_large_fs5_i224`, `cube_single_expert_fs5_i224`, `tool_hang_fs5_i224{_raw,,_raw.pixels_r0eih}`,
`drawer_cleanup_fixed_fs5_i224{_raw,,_raw.pixels_r0eih,_raw.pixels_r1eih}`,
`transport_fs5_i224{_raw,,_raw.pixels_r0eih,_raw.pixels_r1eih}`, `twodoor2_fs5_i224_raw`.

Code: the git repo (branch `lewam-jointflow`); `HROOT/code/lewam_jointflow_<sha>.tar.gz` for
`67e20e1 5cca99f b05a7f8 42560ab 2a5acc5 f4fb884 2b5805f ed314d6 75f0fa7 85b5df9 c461482`; `HROOT/code/*.sh`
(entry scripts) and `*.py`; `HROOT/code/dp_task_yamls/`, `lewam_scripts/`; external: `dp_repo.tar.gz`,
`dp_env.tar.gz`, `wf8_sim_src.tar.gz`, `sim_src.tgz`, `lewm_official_code.tgz`, `lewm_official.tgz`,
`lewm_main_eval/stable-worldmodel/`, `lewam_baselines_code.tgz`.

Devbox: `JOBS/` (YAMLs incl. `fix_wave/`, `fix_wave_ev/`, `e7-*`, `b1-*`, `dr-*`, `tr-*`, `dpc-*`; `submit_guard.sh`),
`/home/tiger/.claude/projects/-home-tiger-lewam/memory/` (rules and ops notes; `MEMORY.md` is the index),
`/home/tiger/.job_name_map.tsv` (coded job names to real names and ids). The session scratch
`/home/tiger/.claude/jobs/5f788414/tmp/` holds only working files; the board snapshot from it is now in the repo.

Documents (in the repo, so they move with git): `docs/HANDOFF.md` (this), `docs/handoff_manifest.txt`,
`docs/RECIPES.md` (the run log), `docs/PAPER_ROBUSTNESS_PLAN.md`, `CLAUDE.md`, `merlin/MERLIN.md`,
`docs/results/` (board, e3, e4, e7, wave1, older dashboards), `docs/EXPERIMENTS.md` (pre-August log),
`docs/WF8_CELLS.md`, `docs/SETTINGS.md`, `docs/MACHINE_TRANSFER.md`, `docs/MOT_SUMMARY.md`, `docs/proposal/`.
On HDFS: `HROOT/wf8/README.md`, `HROOT/code/REACHER_HANDOFF.md`, `HROOT/WHITELIST_TODO.md`.

---------------------------------------------------------------------------------------------------------------
## 3. What is left

TODO list of `REPO/docs/PAPER_ROBUSTNESS_PLAN.md` (lines 8-29, verbatim):

- [ ] **B1** Official DP-C toolhang checkpoint through our eval with the observation/controller
      adapter: 0/50 on all three seeds even with world-frame goals (2026-09-11); BC-RNN dropped
      (owner). Open: the cause (domain gap to robosuite 1.2 or a 1.5 controller detail), the
      audit table; DP-C retrained on our data replaces the published checkpoints (see the
      port `scripts/train_dp.py`). (owner + Minghao)
- [x] **B2** Dropped with BC-RNN (owner, 2026-09-11).
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
- [x] **E7** Done on PushT: the LeWAM grid, the LeWM row on the same episodes, and planning
      time per plan in both regimes (`docs/results/e7/timing_report.md`). Open: other cells,
      GC-IDM row, paired McNemar LeWM vs our arms.

Open items from the latest RECIPES records (2026-09-08 to 2026-09-11):
- B1 (RECIPES 2819-2826): both published toolhang policies score 0 in our stack. Unresolved: domain gap to the
  robosuite 1.2 / mujoco-py renders and physics they were trained on, or a remaining difference in robosuite
  1.5's absolute-pose handling (only the 1.4 orientation path was read). Owner's call: read the 1.5.1 controller
  code, take 0 as the B1 row, or retrain the baselines on our data. Transport official DP-C not started (the
  adapter can render shouldercamera0/1 + both wrists). The audit table is not written.
- Multi-view (RECIPES 2827-2858): code committed (`09137de`, `4aafe8c`, `85b5df9`). Toolhang 2-view training is
  STAGED, NOT SUBMITTED (`JOBS/th-mvsig192-2view.yaml`, entry `jf_tc_85b5df9.sh` with `--views
  pixels,pixels_r0eih`, ARMTAG `_fx_mvsig192_2v`; eval `jf_tc_ev_85b5df9_task_views.sh`, `JOBS/th-mvsig192-2view-ev.yaml`).
  The multi-view eval path was verified offline only; it has not run in a pod.
- Drawer 3-view training (RECIPES 2859-2878): job `94e7bd3536128c2a` (caption mf-8759ca3c, `JOBS/dr-mvsig192-3view.yaml`,
  algorithm aigcp H100, pod memory 200 GB) was `running` at 14:5x today. Run dir
  `HROOT/ckpts/jointflow_tc/tc_drawer_fx_mvsig192_3v_s42/` holds `snap_ep15.pt`, `snap_ep30.pt`, `jointflow_best.pt`
  (63,950,469, synced 14:43), `jointflow_full.pt`, `train.log`; no `done` marker yet. About 10 min per epoch, so it
  finishes around 05:45 on 2026-09-12; `jointflow_full.pt` + `--resume` continues it after a kill. The eval afterwards
  is `jf_tc_ev_85b5df9_task_views.sh drawer ... robot0_eye_in_hand,robot1_eye_in_hand` (never run).
- Transport TC task-only (RECIPES 2148-2160, 2178-2180, 2232-2233): sharding verified (10 envs = 51 min per shard);
  two shards scored 0/10. The full 2 arms x 3 seeds x 5 shards = 30 jobs run was held for the owner's scope call.
  DP transport task-only also needs the shard entry (not built for DP).
- Transport multi-view: the link file and the two wrist caches were built on 2026-09-11 (section 2). The 3-view
  training is the drawer YAML with the transport stem and a larger pod (the three caches are 190 GB in RAM).
- DP-C retrained on our data (owner 2026-09-11, replaces the published checkpoints of 1e): `scripts/train_dp.py`
  + `lewam/models/dp_policy.py` train diffusion_policy's published image model (imported unchanged) on the same
  caches, windows per epoch, validation rule and epoch count as the LeWAM arms; the design and the checks are in
  RECIPES (2026-09-11). Smoke-tested on the devbox. Next: the training entry and the three TC runs (toolhang one
  camera, drawer and transport three), then the task-only eval through `mode=dp_policy`, which already loads the
  new checkpoint format.
- E7 open (plan): other cells with their own ladders, GC-IDM as the goal-conditioned baseline, paired McNemar LeWM vs
  ours (the per-episode success flags are in the `[timing-json]` lines of the logs). The E2 script
  `scripts/sig_tests.py` does not exist yet; the paired tests used scratch scripts (`paired_ol25.py`, `paired_tags.py`
  in `/home/tiger/.claude/jobs/5f788414/tmp/`).
- Optional: second flow-head seed for the toy (E3a lean); E3b on toolhang or cube; pusht `fx_flsig192` SteerMPC and
  `fx_fl192` SteerMPC seeds 0/1 (about 3 h per checkpoint, RECIPES 210).
- Housekeeping: the `lance` 1.2.1 install on the devbox breaks `import lewam.models.motflow` unless `stable_worldmodel` is imported
  first (RECIPES 2414-2416).

---------------------------------------------------------------------------------------------------------------
Two facts to read first.
1. Everything on the board was retrained after the AdaLN fix (commit `67e20e1`, 2026-09-06). Only arms whose
   name starts with `fx_` count. Numbers from before that date are invalid (RECIPES.md line 8).
2. The board is not a file in the repo. It is regenerated from the heartbeat logs on HDFS by
   `REPO/scripts/collect_board.py` (reads `HROOT/ckpts/jointflow_gr_*/hb_*.log`, `HROOT/ckpts/jointflow_tc/hb_*.log`,
   `HROOT/ckpts/jf_grev/hb_*.log`; last line per seed wins; mean and sample std over eval seeds 42/0/1, 50 episodes
   each). The latest snapshot is `REPO/docs/results/board/board_2026-09-08_0551.md` and `.json` (2026-09-08 05:51,
   115 rows); older snapshots are in `REPO/docs/results/wave1/`. The DP rows, the GR-of-TC
   ladders finished after 05:51, the E7 grid and B1 are NOT in any board file; they live in RECIPES.md and the logs.

---------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------
## Appendix A. Results currently reported and what backs each number

Common facts for every LeWAM row.
- Model code: tarball `HROOT/code/lewam_jointflow_67e20e1.tar.gz` (71,438,797 bytes, git archive of `67e20e1`).
  Every checkpoint below was trained on it. Later tarballs (below) changed only eval scripts and entries.
- Trainer: `scripts/train_jointflow.py`. Width 192 throughout: `--z_dim 192 --proj_hidden 384 --d_model 192
  --depth 4 --n_heads 4 --model motflow --zstd_floor 0.005 --grad_probe_every 250`, encoder `resnet18dp`, seed 42,
  fp32. Verified from the dumped `jointflow_config.json` of `tc_toolhang_fx_mnm192_s42` and `fx_nm192_s42` (pusht).
- Arm names: `nm192` / `mnm192` = mse state head, no regularizer (`--w_reg 0`); `vsig192` / `mvsig192` = vanilla
  SIGReg (`--w_reg 0.04 --sigreg_proj_dim 0`); `sig192` / `msig192` = projected SIGReg with policy view (`--w_reg
  0.04 --sep_policy_state`, "pw_zp", dropped by the owner on 2026-09-07, history only); `fl192` / `mfl192` = flow
  state head (`--mot_state_head flow`); `flsig192` / `mflsig192` = flow + pw_zp. The action head is flow everywhere.
- Protocol: 50 episodes per eval seed, eval seeds 42, 0, 1. "TC task-only" = start at frame 0, run to the end, the
  env's own success predicate only (`+gip_eval.full_traj=true +gip_eval.task_only=true`, mode `jointflow_policy`,
  RECIPES tag JFTCTASK). "GR" = random start, goal 50 raw steps ahead (pusht/tworoom/reacher: 25, budget 50;
  pointmaze/transport/drawer: budget 100), goal-match or task predicate, 7 planner modes.
- Planner ladder (GR cells): reactive (goal-conditioned policy, tag JFGC) | best-of-K, K = 32 (JFROLL) | gradient,
  50 steps x 0.05 (JFGRAD) | gradient-TR, trust region 1e-2 (JFGRADTR) | SteerMPC, Adam 20 x 0.02, rho 0.3
  (JFSTEER) | CEM-policy (JFCEM) | random candidates (JFROLLRAND). Order in every table below is that one.
- Per-seed values: `REPO/docs/results/board/board_2026-09-08_0551.md` and the RECIPES lines cited per table. Seeds are listed in the
  order {42, 0, 1} in RECIPES; the board file sorts them 0, 1, 42.

### 1a. TC board (task completion, one arm per row; success %, mean +- std over 3 eval seeds)

| cell | arm | run dir under `HROOT/ckpts/jointflow_tc/` (jointflow_best.pt bytes) | task-only | old protocol (goal-match OR task) |
|---|---|---|---|---|
| toolhang | mse noreg | `tc_toolhang_fx_mnm192_s42` (63,924,480) | 84.0 +- 4.0 {84,80,88} | 84.7 +- 4.2 {86,80,88} |
| toolhang | mse pw_zp | `tc_toolhang_fx_msig192_s42` (64,221,263) | 80.7 +- 3.1 {84,80,78} | 82.7 +- 7.6 {86,88,74} |
| toolhang | flow noreg | `tc_toolhang_fx_mfl192_s42` (67,484,968) | 80.0 +- 4.0 {80,76,84} | 82.0 +- 0.0 {82,82,82} |
| toolhang | flow pw_zp | `tc_toolhang_fx_mflsig192_s42` (67,781,751) | 84.7 +- 9.5 {88,74,92} | 86.0 +- 8.0 {86,78,94} |
| toolhang | mse vanilla SIGReg | `tc_toolhang_fx_mvsig192_s42` (63,924,480) | 80.0 +- 3.5 {78,84,78} | not run |
| drawer | mse noreg | `tc_drawer_fx_mnm192_s42` (63,947,584) | 6.0 +- 3.5 {10,4,4} | (old +50-step rows invalid) |
| drawer | mse pw_zp | `tc_drawer_fx_msig192_s42` (64,244,367) | 5.3 +- 4.2 {10,2,4} | |
| drawer | mse vanilla SIGReg | `tc_drawer_fx_mvsig192_s42` (63,947,584) | 4.0 +- 2.0 {4,6,2} | |
| transport | mse noreg | `tc_transport_fx_mnm192_s42` (63,935,296) | 0/10 on shard 0 of 5, seeds 42 and 0 only; full 3-seed number NOT RUN | |
| transport | mse pw_zp | `tc_transport_fx_msig192_s42` (64,232,079) | NOT RUN (killed at the 3 h wall) | |
| transport | mse vanilla SIGReg | `tc_transport_fx_mvsig192_s42` (63,935,296) | NOT RUN | |

- Training flags (TC): `jf_tc_67e20e1.sh <cell> <h5> <cell> 42 120` = raw cache, `--epochs 120 --warmup_epochs
  10 --batch_size 64 --lr 1e-4 --num_actions_pred 10 --num_states_pred 1 --policy_history_len 2 --fp32`, plus the
  arm's EXTRA flags above; goal conditioning off (`H_max 50` in the dump is inert without goals).
  RECIPES.md lines 103-105, 191-193, 459-460.
- Entry scripts (all under `HROOT/code/`): training `jf_tc_67e20e1.sh` (5,634); task-only eval
  `jf_tc_ev_b05a7f8_task.sh` (5,573) and `jf_tc_ev_b05a7f8_task2.sh` (6,177; same, scratch on the pod overlay);
  transport shard eval `jf_tc_ev_b05a7f8_shard.sh` (6,782); failure videos `jf_tc_ev_b05a7f8_vid.sh`,
  `jf_tc_ev_b05a7f8_vidshard.sh`; drawer expert replay `jf_tc_ev_42560ab_expert.sh`. Eval tarball
  `lewam_jointflow_b05a7f8.tar.gz` (71,458,062) = 67e20e1 model code + the `task_only` eval flag.
- Eval logs (verified): `HROOT/ckpts/jointflow_tc/evtask_tc_<cell>_<arm>_s42_e<seed>.log` and
  `hb_evtask_tc_<cell>_<arm>_s42.log` for the 5 toolhang and 3 drawer arms (all three seeds present); transport:
  `evtask_tc_transport_fx_mnm192_s42_e{42,1}.log` (killed runs, no result) and
  `evshard_tc_transport_fx_mnm192_s42_e{42,0}_sh0.log` + `hb_evshard_tc_transport_fx_mnm192_s42.log` (the 0/10
  shards). Old-protocol toolhang logs: `ev_tc_toolhang_<arm>_s42_e<seed>.log`, `hb_ev_tc_toolhang_<arm>_s42.log`.
  Videos: `videos_tc_drawer_fx_mnm192_s42_e0/`, `videos_expert_tc_drawer_fx_mnm192_s42_e0/`,
  `videos_shard_tc_transport_fx_mnm192_s42_e0_sh0/`.
- Where written: RECIPES.md lines 249-251 (toolhang task rows), 268-269 (vanilla), 297-300 (drawer), 2204-2206
  (drawer vanilla), 2178-2180 and 2222-2233 (transport shards), 426-440 (drawer expert replay 72.0 = protocol
  ceiling). Timing per row: `REPO/docs/results/board/board_2026-09-08_0551.md`.
- Every TC run dir also holds `jointflow_config.json`, `train.log`, `done`, `jointflow_full.pt` (~191 MB, model +
  optimizer, resumable), `jointflow_latest.pt`, `snap_ep*.pt`, `grad_probe.jsonl`.

### 1b. GR board (goal reaching; 7-mode ladder; success %, mean +- std over 3 eval seeds)

Checkpoint = `HROOT/ckpts/jointflow_gr_<cell>_<arm>/<arm>_s42/jointflow_best.pt` (bytes in the table). Every one of
these dirs has `done`, `jointflow_config.json`, `train.log`, `jointflow_full.pt` (~191-203 MB), `snap_ep{15,25,40}.pt`.

| cell | arm | bytes | reactive | best-of-K | gradient | gradient-TR | SteerMPC | CEM | random |
|---|---|---|---|---|---|---|---|---|---|
| pusht | fx_nm192 | 63,916,800 | 72.7 +- 5.0 | 78.0 +- 2.0 | 83.3 +- 5.8 | 78.7 +- 2.3 | 80.0 +- 4.0 | 80.7 +- 6.4 | 37.3 +- 5.8 |
| pusht | fx_vsig192 | 63,916,800 | 80.0 +- 5.3 | 89.3 +- 4.2 | 94.0 +- 4.0 | 94.7 +- 4.2 | 92.7 +- 3.1 | 93.3 +- 4.2 | 50.7 +- 4.6 |
| pusht | fx_sig192 | 64,213,583 | 74.0 +- 9.2 | 78.0 +- 10.0 | 86.0 +- 9.2 | 80.7 +- 9.0 | 82.7 +- 8.1 | 83.3 +- 8.3 | 40.7 +- 2.3 |
| pusht | fx_fl192 | 67,477,288 | 66.0 +- 8.0 | 62.0 +- 6.9 | 61.3 +- 6.4 | 62.7 +- 7.6 | 48.0 (seed 42 only) | 59.3 +- 5.0 | 38.0 +- 6.0 |
| pusht | fx_flsig192 | 67,774,071 | 74.7 +- 4.2 | 76.7 +- 4.2 | 72.0 +- 8.7 | 76.0 +- 5.3 | NOT RUN | 76.7 +- 2.3 | 34.7 +- 2.3 |
| tworoom | fx_nm192 | 63,916,800 | 98.0 +- 2.0 | 100 +- 0.0 | 100 +- 0.0 | 100 +- 0.0 | 99.3 +- 1.2 | 100 +- 0.0 | 95.3 +- 3.1 |
| tworoom | fx_vsig192 | 63,916,800 | 98.7 +- 2.3 | 100 +- 0.0 | 100 +- 0.0 | 100 +- 0.0 | 100 +- 0.0 | 100 +- 0.0 | 84.0 +- 6.0 |
| pointmaze_large | fx_nm192 | 63,916,800 | 100 | 100 | 100 | 100 | 100 | 100 | 89.3 +- 2.3 |
| pointmaze_large | fx_vsig192 | 63,916,800 | 100 | 100 | 100 | 100 | 100 | 100 | 59.3 +- 6.1 |
| cube (with effector term 0.04) | fx_nm192 | 63,921,408 | 98.0 +- 2.0 | 100 +- 0.0 | 98.0 +- 0.0 | 100 +- 0.0 | 100 +- 0.0 | 98.0 +- 0.0 | 40.0 +- 6.9 |
| cube | fx_vsig192 | 63,921,408 | 94.0 +- 2.0 | 97.3 +- 3.1 | 100 +- 0.0 | 100 +- 0.0 | 98.7 +- 1.2 | 98.7 +- 1.2 | 42.7 +- 3.1 |
| reacher_policy (wrapped rule) | fx_nm192w | 63,916,800 (same ckpt as fx_nm192) | 100 | 100 | 100 | 100 | 100 | 99.3 +- 1.2 | 65.3 +- 6.1 |
| reacher_policy (wrapped rule) | fx_vsig192w | 63,916,800 | 100 | 100 | 100 | 100 | 100 | 100 | 52.7 +- 5.8 |
| reacher_policy (old rule, lower bound) | fx_nm192 | as above | 90.0 +- 5.3 | 93.3 +- 4.2 | 92.0 +- 5.3 | 92.0 +- 5.3 | 92.0 +- 7.2 | 92.7 +- 5.0 | 64.7 +- 6.4 |
| reacher_policy (old rule) | fx_sig192 | 64,213,583 | 91.3 +- 5.0 | 91.3 +- 5.8 | 90.7 +- 6.4 | 90.7 +- 6.4 | 90.7 +- 4.6 | 91.3 +- 5.8 | 66.7 +- 6.4 |
| toolhang as GR | fx_nm192 | 63,924,480 | 92.0 +- 2.0 | 92.0 +- 2.0 | 76.0 +- 4.0 | 90.7 +- 1.2 | 88.0 +- 0.0 | 88.0 +- 5.3 | 33.3 +- 2.3 |
| toolhang as GR | fx_vsig192 | 63,924,480 | 92.0 +- 0.0 | 94.7 +- 3.1 | 91.3 +- 1.2 | 93.3 +- 1.2 | 91.3 +- 4.2 | 93.3 +- 1.2 | 31.3 +- 5.8 |
| drawer as GR | fx_nm192 | 63,947,584 | 40.7 +- 11.0 | 46.7 +- 14.7 | 12.7 +- 8.1 | 38.0 +- 8.7 | 39.3 +- 3.1 | 14.0 +- 10.6 | 11.3 +- 4.6 |
| drawer as GR | fx_vsig192 | 63,947,584 | 52.0 +- 10.4 | 59.3 +- 1.2 | 19.3 +- 6.4 | 52.7 +- 1.2 | 47.3 +- 1.2 | 25.3 +- 6.1 | 10.7 +- 4.2 |
| transport as GR | fx_nm192 | 63,935,296 | 80.0 +- 6.0 | 76.0 +- 2.0 | 46.7 +- 3.1 | 79.3 +- 3.1 | 76.0 +- 4.0 | 68.7 +- 1.2 | 4.7 +- 1.2 |
| transport as GR | fx_vsig192 | 63,935,296 | 78.0 +- 7.2 | 77.3 +- 6.1 | 54.7 +- 11.0 | 80.7 +- 4.6 | 78.0 +- 3.5 | 69.3 +- 4.2 | 4.0 +- 0.0 |

- Training flags (GR): `jf_gr_67e20e1.sh` (pusht) / `jf_gr_67e20e1b.sh` (other cells) / `jf_gr_67e20e1d.sh` (the
  TC cells as GR, train-only, ladders by hand-off): strided cache, `--fs_strided --goal_conditioning
  --num_actions_pred 10 --num_states_pred 1 --policy_history_len 2 --p_drop_goal 0 --epochs 50 --warmup_epochs 5
  --batch_size 128 --lr 1.5e-4 --H_max 10` (10 anchors = 50 raw steps), plus the arm's EXTRA flags. Verified from
  the pusht `fx_nm192_s42/jointflow_config.json` (H_max 10, lr 0.00015, fp32 true, frames_cache preload_cache_u8).
  RECIPES.md lines 121-125, 194-195, 400-402, 461-467.
- Eval entries (HROOT/code/): pusht ladder `jf_grev_5cca99f.sh` (tarball `lewam_jointflow_5cca99f.tar.gz`,
  71,452,534); reacher_policy `jf_grev_5cca99fr.sh` (dm_control pin); cube `jf_grev_5cca99fc.sh` (effector term);
  reacher wrapped rule and the GR-of-TC ladders `jf_grev_42560ab.sh` (tarball `lewam_jointflow_42560ab.tar.gz`,
  71,470,292; contains Minghao's angle-wrap commit 4c33287). tworoom / pointmaze_large: reactive, best-of-K,
  gradient, random came from the chained eval inside the training job (67e20e1 tarball); gradient-TR, SteerMPC,
  CEM from `jf_grev_5cca99f.sh`.
- Eval logs (verified): pusht, tworoom, pointmaze_large, cube, reacher_policy ladders and the GR-of-TC ladders are in
  `HROOT/ckpts/jf_grev/`: `hb_<cell>_<arm>_s42.log` (result lines) and `ev_<cell>_<arm>_s42_<mode>_e<seed>.log`
  (timing json per run). The chained reactive/best-of-K/gradient/random rows of pusht, tworoom and pointmaze_large
  are in `HROOT/ckpts/jointflow_gr_<cell>_<arm>/hb_<arm>_s42.log` with `ev_<arm>_s42_{gc,plan,grad,rand}_e<seed>.log`
  beside them (verified for tworoom nm192, pointmaze_large vsig192, pusht vsig192).
- Where written: RECIPES.md lines 151-176 (wave-1 board), 202-221 and 281-285 (pusht ladders), 261-267 (pusht
  vanilla), 271-280 and 372-386 (reacher_policy), 504-535 (tworoom, pointmaze_large), 2163-2202 (GR-of-TC),
  `REPO/docs/results/board/board_2026-09-08_0551.md` (cube rows, all per-seed values, timing columns). Repo snapshots: `REPO/docs/results/wave1/
  board_wave1_2026-09-06.{md,json}`, `board_pusht_planners_2026-09-07.md`, `board_2026-09-07_0500.json`.
- The GR-of-TC rows use goal = 50 raw steps after a random start and success = goal-match OR task predicate
  (RECIPES 2207-2210). Drawer / transport task-only (start to finish) are the TC rows in 1a, not these.

### 1c. E7: success vs goal distance on PushT, with the LeWM row and planning time

Checkpoints: pusht `fx_nm192_s42` (noreg) and `fx_vsig192_s42` (SIGReg) from 1b. LeWM = the authors' released PushT
checkpoint `HROOT/code/lewm_main_eval/hf_release_native/pusht/weights.pt` (72,265,441) + `config.json` (1,313), run
through our port `REPO/scripts/eval.py` with its own CEM (300 x 30). NOTE: `HROOT/ckpts/hf_official/pusht_lewm_base`
is OUR repro, not the release; the first smoke job failed on it (RECIPES 2680-2685).

| H | noreg best-of-K | noreg gradient-TR | SIGReg best-of-K | SIGReg gradient-TR | LeWM CEM |
|---|---|---|---|---|---|
| 25 | 96.0 +- 3.5 {98,98,92} | 93.3 +- 4.6 {96,96,88} | 99.3 +- 1.2 {100,100,98} | 98.0 +- 2.0 {96,100,98} | 68.7 +- 3.1 {72,66,68} |
| 50 | 22.0 +- 9.2 {14,32,20} | 20.7 +- 11.0 {10,32,20} | 39.3 +- 9.2 {34,50,34} | 71.3 +- 3.1 {72,74,68} | 12.0 +- 4.0 {12,8,16} |
| 75 | 10.0 +- 0.0 {10,10,10} | 8.7 +- 1.2 {8,8,10} | 26.0 +- 9.2 {16,28,34} | 46.7 +- 6.4 {44,42,54} | 3.3 +- 2.3 {2,6,2} |
| 100 | 9.3 +- 4.2 {8,6,14} | 8.7 +- 2.3 {10,6,10} | 23.3 +- 3.1 {20,26,24} | 45.3 +- 6.4 {38,50,48} | 2.7 +- 1.2 {2,2,4} |

- Protocol: 50 episodes per seed from demos with at least 101 frames, the same 50 for every H, start frame 0, goal
  at frame H, budget 2H, imagine to the goal, cost at the goal, execute 25 actions, replan; K = 32; seeds 42/0/1.
  Flags `++gip_eval.exec_actions=25 +gip_eval.random_start=false +gip_eval.min_episode_len=101
  eval.goal_offset_steps=H eval.eval_budget=2H`. Episode draw `sorted(rng(seed).choice(n_eligible, 50, replace=False))`.
- Code: LeWAM grid `lewam_jointflow_f4fb884.tar.gz` (71,502,189; H 25/50/75 paired) and `_2a5acc5.tar.gz`
  (71,500,127; H 100); rerun with per-plan timing records `lewam_jointflow_ed314d6.tar.gz` (73,504,822). Entries:
  `jf_grev_f4fb884.sh`, `jf_grev_2a5acc5.sh`, `jf_e7grid_ed314d6.sh` -> `jf_grev_ed314d6.sh`; LeWM
  `lewm_grid_ed314d6_v2.sh` and the random-start control `lewm_ctrl_ed314d6.sh`. YAMLs `JOBS/e7-lewam-nm192.yaml`,
  `e7-lewam-vsig192.yaml`, `e7-lewm-grid.yaml`, `e7-lewm-ctrl.yaml`, `JOBS/fix_wave_ev/grev_pusht_lh*.yaml`, `lhp*.yaml`.
- Logs (verified): LeWAM `HROOT/ckpts/jf_grev/hb_pusht_<arm>lhp{25,50,75}_s42.log`, `hb_pusht_<arm>lh100_s42.log`
  (the table), `hb_pusht_<arm>tm{25,50,75,100}_s42.log` and `...tm{25,100}n1_s42.log` (timing rerun, batched and
  batch 1), `hb_e7grid_fx_{nm192,vsig192}.log`, plus the `ev_pusht_*` logs. LeWM `HROOT/ckpts/lewm_grid/
  ev_lewm_H{25,50,75,100}_s{42,0,1}.log`, `ev_lewm_n1_H*_s{42,0,1,2,3}.log`, control `ev_lewmrs_H25_s*.log`.
- Timing report: `REPO/docs/results/e7/timing_report.md` (9,516) and `.json` (36,719), built by
  `scripts/plan_timing_report.py` over those logs. Seconds per plan at batch 1, warm, by imagined blocks
  (5/10/15/20): best-of-K 0.19 / 0.37-0.39 / 0.55-0.57 / 0.93-0.95 (first call); gradient-TR 2.14 / 4.20-4.25 /
  6.32-6.33 / 8.70-8.82 (first call); LeWM CEM 0.40 / 0.79 / 1.16 / 1.58.
- Control rows (LeWAM under LeWM's open-loop scheme, random starts, H 25): noreg best-of-K 73.3 +- 5.0, noreg
  gradient-TR 74.7 +- 3.1, SIGReg best-of-K 83.3 +- 3.1, SIGReg gradient-TR 91.3 +- 2.3; LeWM 88.7. Logs
  `hb_pusht_<arm>ol25_s42.log`, `..ol25gt_s42.log`. LeWM random-start control 89.3 +- 5.0 {94,90,84}.
- Where written: RECIPES.md 2284-2306 (control), 2388-2401 (grid), 2637-2736 (LeWM row, timing, control);
  PAPER_ROBUSTNESS_PLAN.md section E7.

### 1d. DP-T baseline rows (official diffusion_policy code, DP-T, one 224 agentview image, no proprio, crop 202, 120 epochs)

| cell | checkpoint (verified, bytes) | task-only, start to finish | old protocol |
|---|---|---|---|
| toolhang | `HROOT/ckpts/wf8_uni/toolhang_dp_noprop/snap_ep120.ckpt` (412,783,191) | 68.0 +- 9.2 {58,70,76} | 71.0 |
| drawer | `HROOT/ckpts/wf8_dp/drawer/latest.ckpt` (412,907,095; = the epoch-120 artifact) | 2.0 +- 2.0 {4,0,2} | 52.7 |
| transport | `HROOT/ckpts/wf8_dp/transport/epoch=0120-train_loss=0.0476.ckpt` (412,840,535) | NOT RUN (both jobs killed at the 3 h wall; needs episode sharding) | 84.7 {84,86,84} |
| cube | `HROOT/ckpts/wf8_dp/cube/latest.ckpt` (412,766,743) | declined by the owner (DP is not goal-conditioned) | none |

- Training entry `HROOT/code/wf8_dp_cell_v5_0d2dcfc.sh` (4,376) with the v6b/v6c resume-chain fixes
  (`wf8_dp_cell_v6b_0d2dcfc.sh`, `v6c`); DP code `HROOT/code/dp_repo.tar.gz` (26,381,513) and env
  `HROOT/code/dp_env.tar.gz` (314,926,699); task yamls `HROOT/code/dp_task_yamls/`.
- Eval entry `dp_ev_cell_b05a7f8_task.sh` (5,518; tag DPTASK); old protocol `dp_ev_cell_b4fe8e7.sh`.
- Logs (verified): `HROOT/ckpts/wf8_dp/toolhang_eval/ev_dp_toolhang_task_{a_e42,a_e0,b_e1}.log` + `hb_dp_toolhang_task_{a,b}.log`;
  `wf8_dp/drawer_eval/ev_dp_drawer_task_{a_e42,b_e1,c_e0}.log`; old rows `wf8_dp/{drawer,transport}_eval/ev_ep120_e{42,0,1}.log`.
  collect_board.py does not scan these dirs; the numbers are only in RECIPES.md 231-253 and 270, and in the plan
  (B1 section). The DP rows' training-time replay buffers `replay.zarr.zip` (3.5 to 7.3 GB each) are not needed.

### 1e. B1: published toolhang checkpoints through our task-only protocol

| policy | checkpoint (verified, bytes) | result | logs |
|---|---|---|---|
| DP-C (diffusion_policy_cnn train_0, raw weights, no EMA), world-frame absolute goals on robosuite 1.5 | `HROOT/ckpts/official_baselines/tool_hang/dp_cnn_train_0/epoch=2150-test_mean_score=0.955.ckpt` (4,627,776,681) | 0/50, 0/50, 0/50 = 0.0 +- 0.0 (seeds 42/0/1), every episode at its 2x budget | `HROOT/ckpts/official_baselines/tool_hang/eval/ev_dpc_th_s{42,0,1}_e*.log`, `baseline_dpc_th_s*_dp_s*.json`, `hb_dpc_th_s*.log` |
| BC-RNN (robomimic model zoo v0.1) | `.../tool_hang/bc_rnn/tool_hang_ph_image_epoch_440_succ_74.pth` (140,402,076) | 0/50 on seed 42; dropped by the owner | `.../eval/ev_bcrnn_th_e42.log`, `baseline_bcrnn_th_bc_rnn_s42.json` |

- Code: `REPO/scripts/eval_official_baseline.py` (10,064) and `REPO/lewam/robomimic_checkpoint.py`; tarball
  `lewam_jointflow_75f0fa7.tar.gz` (73,529,606); entry `baseline_ev_75f0fa7.sh` (3,915); YAMLs
  `JOBS/b1-dpc-toolhang-s{42,0,1}e.yaml`. Jobs 543cd888a228de64 / dc6af8fab5157810 / 9ae0f96dabe226aa.
- The earlier base-frame DP-C runs (also 0/50, wrong by construction) are under
  `HROOT/ckpts/official_baselines/tool_hang/eval/base_frame/` (3 logs + 3 json, 35 KB).
- Where written: RECIPES.md 2737-2826; PAPER_ROBUSTNESS_PLAN.md section B1; memory `b1-official-baselines.md`.

### 1f. E3 and E4 (figures and jsons in the repo; raw outputs on HDFS)

- E3a two-door toy: checkpoints `HROOT/ckpts/jointflow_tc/tc_twodoor2_fx_flowvsig192_s42/jointflow_best.pt`
  (63,916,800) and `tc_twodoor2_fx_msevsig192_s42/jointflow_best.pt` (60,364,317); v1 pair `tc_twodoor_fx_*` (same
  sizes). Dataset `HROOT/wf8/train/twodoor2.h5` (2,049,355,186), cache `HROOT/preload_cache/twodoor2/` (2.04 GB).
  Entries `jf_toy2_f4fb884.sh`, `jf_toy2_eval_f4fb884.sh`. Outputs `HROOT/ckpts/probes_e3/v2_eval_s42_box1px/`;
  figures `REPO/docs/results/e3/v2/rollouts_box1px.png`, `first_chunks.png`, `demonstrations.png`,
  `summary_box1px.json`. Result: flow reached 82 (54 upper / 28 lower), crashed 5; mse reached 74 (0 / 74), crashed 26.
- E3b PushT chunk likelihood: entry `jf_e3b_f4fb884.sh`, outputs `HROOT/ckpts/probes_e3/e3b_pusht/`, repo
  `REPO/docs/results/e3/e3b_pusht.json`, `e3b_pusht_preference.png`, `e3b_pusht_ranks.png`.
- E4 latent probes: scripts and npz in `HROOT/ckpts/probes_e4/` (latent_geometry.py, latent_gaussianity.py,
  latent_pictures.py, e4_probes.sh, e4_tworoom.sh, latents_*.npz); jsons and pngs also in `REPO/docs/results/e4/`
  (geometry_{pusht,toolhang,tworoom}.json, dyn_/cost_/div_/discrim_*.json, latent_pictures_{pusht,toolhang,tworoom}.png).
  Checkpoints used: pusht fx_nm192 / fx_vsig192, toolhang tc_toolhang_fx_mnm192 / fx_mvsig192, tworoom fx_nm192 / fx_vsig192.
- Where written: RECIPES.md 2403-2470 (E4), 2472-2635 (E3); plan sections E3, E4.

---------------------------------------------------------------------------------------------------------------
## Appendix B. Data and caches in detail

Datasets, all under `HROOT/wf8/` (layout and history: `HROOT/wf8/README.md`, 12,567 bytes; pod-side names differ
from disk names, see its section 5). The a2f column says whether an identical copy exists under `A2F`.

| file | bytes | used by | a2f copy |
|---|---|---|---|
| `train/pusht.h5` | 46,300,921,856 | pusht GR arms, E3b, E4, E7 (pod name `pusht_expert_train.h5`) | `A2F/pusht_expert_train.h5` (same size) |
| `train/tworoom.h5` | 12,775,849,984 | tworoom GR, E4 | `A2F/tworoom.h5` |
| `train/pointmaze_large.h5` | 22,506,324,468 | pointmaze_large GR | none |
| `train/cube.h5` | 101,942,558,720 | cube GR (pod name `ogbench/cube_single_expert.h5`) | `A2F/ogbench/cube_single_expert.h5` |
| `train/reacher_policy.h5` | 75,648,536,114 | reacher_policy GR (linked as `reacher.h5` in the pod) | none |
| `train/toolhang.h5` | 5,030,456,346 | toolhang TC and as-GR, E4, B1 start states (pod name `tool_hang.h5`) | none |
| `train/toolhang_eih.h5` | 4,622,226,254 | 2-view toolhang cache (staged, not trained) | none |
| `train/drawer.h5` | 26,253,384 (external links) + `_source/drawer_raw.h5` 18,475,024,868 | drawer TC and as-GR (pod name `drawer_cleanup_fixed.h5`) | none |
| `train/drawer_3view.h5` | 42,575,765,750 | drawer 3-view training (running) | none |
| `train/transport.h5` | 32,412,252,851 | transport TC and as-GR | none |
| `train/transport_3view.h5` | 88,868,459,704 | not used yet (transport multi-view) | none |
| `train/twodoor2.h5` / `twodoor.h5` | 2,049,355,186 / 2,079,482,274 | E3a v2 / v1 | none |
| `train/reacher.h5` | 98,905,882,624 | old random-policy reacher; replaced, no current row | `A2F/reacher.h5` |
| `train/pointmaze.h5` | 24,820,266,155 | no current row | none |
| `train/drawer_flat.h5`, `train/reacher_mix.h5` | 18,472,642,772 / 316,184,802 | no current row | none |
| `train/_views/{tool_hang,drawer_cleanup_fixed,transport}.h5` | 8,289 / 8,289 / 8,327 | external-link joins for multi-view caches | none |
| `eval/{toolhang,transport,drawer}.h5` | 7,674 / 7,714 / 7,666 | eval views without `model_xml` (external links into train/) | none |
| `eval/{pusht,tworoom,reacher,cube,pointmaze,pointmaze_large}.h5` | symlinks to train/ | | |
| `eval_tc/pointmaze.h5` | 2,373,066,182 | old full-traj pointmaze; no current row | none |
| `env/transport_model_xml.h5` | 87,135,688 | transport env rebuild | none |
| `env/drawer_model_xml.h5` | 131,871,776 | drawer env rebuild | none |
| `env/dexmg_model_xml.h5` | 33,873,432 | kept, not referenced | none |
| `env/robomimic_raw/tool_hang/` | dir (holds a 6 KB env-args stub per RECIPES 2810) | toolhang env args | none |

`A2F/wf8/` is empty. No jointflow entry reads a2f (README section 7a).

Frame caches, `HROOT/preload_cache/<stem>/<stem>_fs5_i224[_raw][.<view>].{frames.npy,aux.npz,meta.json}`.
`_raw` = uint8, every frame (TC trainer); no suffix = fp16, one frame per 5 (GR trainer); `HROOT/preload_cache_u8/`
= uint8 strided copies. Sizes are the `frames.npy` bytes.

| cache | bytes | used by |
|---|---|---|
| `preload_cache_u8/pusht_expert_train/` | 71,425,235,072 | pusht GR arms (verified in the config dump) |
| `preload_cache_u8/reacher_policy/` | 76,262,301,824 | reacher_policy GR (RECIPES 194, 222) |
| `preload_cache/tworoom/tworoom_fs5_i224` | 57,369,833,600 | tworoom GR (RECIPES 464) |
| `preload_cache/pointmaze_large/pointmaze_large_fs5_i224` | 60,211,200,128 | pointmaze_large GR |
| `preload_cache/cube_single_expert/cube_single_expert_fs5_i224` | 123,432,960,128 | cube GR (RECIPES 467) |
| `preload_cache/tool_hang/tool_hang_fs5_i224_raw` | 14,444,968,064 | toolhang TC (verified in the config dump) |
| `preload_cache/tool_hang/tool_hang_fs5_i224` | 5,801,349,248 | toolhang as GR |
| `preload_cache/drawer_cleanup_fixed/..._raw` | 44,892,718,208 | drawer TC |
| `preload_cache/drawer_cleanup_fixed/drawer_cleanup_fixed_fs5_i224` | 18,120,560,768 | drawer as GR |
| `preload_cache/transport/transport_fs5_i224_raw` | 63,178,408,064 | transport TC |
| `preload_cache/transport/transport_fs5_i224` | 25,372,999,808 | transport as GR |
| `preload_cache/twodoor2/` raw | 2,042,815,616 | E3a v2 (`twodoor/` raw 2,063,287,424 = v1) |
| `preload_cache/tool_hang/..._raw.pixels_r0eih` | 14,444,968,064 | 2-view toolhang (staged) |
| `preload_cache/drawer_cleanup_fixed/..._raw.pixels_r0eih`, `..._raw.pixels_r1eih` | 44,892,718,208 each | drawer 3-view (running) |
| `preload_cache/transport/..._raw.pixels_r0eih`, `..._raw.pixels_r1eih` | 63,178,408,064 each (built 2026-09-11) | transport 3-view (staged) |
| not used by any current row: `pusht_expert_train` fp16 142.9 GB + raw 351.7 GB, `reacher` 123.4 + raw 302.6 GB, `preload_cache_u8/reacher` 100.9 GB, `cube_single_expert` raw 302.6 GB, `pointmaze` 60.2 + raw 150.5 GB, `pointmaze_large` raw 150.5 GB, `tworoom` raw 138.6 GB, `reacher_policy` fp16 152.5 GB, `dexmimicgen`, `drawer_cleanup`, `pointmaze_300`, `tool_hang_384ds` | | |

Rebuild: `REPO/scripts/make_preload_cache.py` (default = fp16 strided; `--anchor_rate raw` = uint8 every frame;
`--pixels_key <column>` = an extra camera; its `--out` is the stem directory), `REPO/scripts/convert_cache_u8.py`
(fp16 -> uint8), entry `HROOT/code/cache_entry.sh`. Devbox timings: toolhang raw view 99 s, drawer view ~7 min + 5
min copy. The cube raw cache took a dedicated job (`cube_cache_build2_600c696.sh`).

---------------------------------------------------------------------------------------------------------------
## Appendix C. Code in detail

- Repo: `REPO` (git worktree of `/home/tiger/lewam`), branch `lewam-jointflow`, HEAD `c461482`, pushed to
  `origin` = `git@github.com:MinghaoFu/lewam.git` (`origin/lewam-jointflow` contains HEAD). Second remote `gitlab`
  (code.byted.org) is not used for GitHub-facing work. The `live` mirror branch is stale (last snapshot 2026-07-21)
  and the watcher is not running.
- Committed with the DP-C port on 2026-09-11 (see `git log`): the B1, multi-view and kill-procedure records in
  `CLAUDE.md`, `docs/PAPER_ROBUSTNESS_PLAN.md`, `docs/RECIPES.md`, `merlin/MERLIN.md`; `lewam/models/dp_policy.py`,
  `scripts/train_dp.py` and the `gip.py` adapter change (DP-C trained on our caches, section 5); this document.
- Tarballs the reported numbers depend on (`HROOT/code/lewam_jointflow_<sha>.tar.gz`): `67e20e1` (71,438,797,
  training of every arm), `5cca99f` (71,452,534, pusht/reacher/cube ladders), `b05a7f8` (71,458,062, task-only TC
  evals, DP task evals), `42560ab` (71,470,292, wrapped reacher, GR-of-TC ladders, expert replay), `2a5acc5`
  (71,500,127) and `f4fb884` (71,502,189, E7 grid, E3), `2b5805f` (73,499,340, E7 smoke), `ed314d6` (73,504,822, E7
  timing rerun, LeWM grid), `75f0fa7` (73,529,606, B1), `85b5df9` (73,529,398, multi-view). Older tarballs stay for
  reading old (invalid) checkpoints.
- Scripts in `REPO/scripts/`: training `train_jointflow.py` (48,510); eval `eval_gip.py` (36,795; LeWAM harness),
  `eval.py` (10,799; LeWM port with the pairing guard), `eval_official_baseline.py`; caches `make_preload_cache.py`,
  `convert_cache_u8.py`; board `collect_board.py` (11,549); timing `plan_timing_stats.py`, `plan_timing_report.py`;
  converters `convert_swm_to_robomimic.py` (wf8 h5 -> robomimic hdf5 for DP), `build_pointmaze_dataset.py`;
  probes `probe_jf_dynamics.py`, `probe_jf_cost.py`, `probe_jf_diversity.py`, `probe_wm_discrim.py`. Model:
  `REPO/lewam/models/motflow.py` (num_views), `gip.py` (adapters, planners), `eval_timing.py` (PlanTimer),
  `robomimic_checkpoint.py`; envs `REPO/lewam/envs/`; eval configs `REPO/configs/eval/*.yaml` (pusht, tworoom,
  pointmaze_large, reacher_policy, cube, toolhang, transport, drawer, drawer_expert).
- Entry scripts: copies on HDFS `HROOT/code/*.sh` (the ones a pod ran; names above) and in `JOBS/` (YAMLs:
  `JOBS/fix_wave/*.yaml` = the 30 training jobs of waves 1-3 and GR-of-TC, `JOBS/fix_wave_ev/*.yaml` = the eval
  jobs, `JOBS/e7-*.yaml`, `JOBS/b1-*.yaml`, `JOBS/th-mvsig192-2view*.yaml`, `JOBS/dr-mvsig192-3view.yaml`).
  Submit guard `JOBS/submit_guard.sh`. How-to: `REPO/merlin/MERLIN.md`, `REPO/merlin/example_job.yaml`.
- External code and checkpoints: LeWM release `HROOT/code/lewm_main_eval/hf_release_native/{pusht,cube,reacher,
  tworooms}/` and `HROOT/code/lewm_main_eval/stable-worldmodel/`, `HROOT/code/lewm_official_code.tgz` (5,513,600),
  `lewm_official.tgz` (5,520,204); diffusion_policy `dp_repo.tar.gz`, `dp_env.tar.gz`; robomimic 0.5 source for
  the toolhang sims `wf8_sim_src.tar.gz` (31,505,487) / `sim_src.tgz` (31,525,618); official toolhang checkpoints
  (1e); Minghao's baselines code `lewam_baselines_code.tgz` (144,049,838; cube GR baselines).
- Simulator versions that matter: pods run robosuite 1.5.1 + mujoco 3; the devbox 1.4.1; reacher needs
  `dm_control==1.0.43 mujoco==3.10.0` (entries pin it); headless needs `MUJOCO_GL=egl`.

---------------------------------------------------------------------------------------------------------------
## Appendix D. Documents

Repo (`REPO/`):
- `docs/RECIPES.md` (the run log; about 2,900 lines)
- `docs/PAPER_ROBUSTNESS_PLAN.md` (experiments E1-E7 and B1)
- `CLAUDE.md` (project rules, GPU how-to, data paths) and `docs/CLAUDE.md` (10,122; historical, L40S era)
- `merlin/MERLIN.md` (Merlin job how-to incl. the kill procedure), `merlin/example_job.yaml`, `merlin/example_entry.sh`
- `docs/results/`: `board/` (the latest board snapshot, 2026-09-08 05:51), `e3/` (e3b_pusht.json, two pngs,
  `v2/` four files), `e4/` (23 json + 3 png), `e7/` (timing_report.md, timing_report.json), `wave1/` (4 board files), `dashboard/` (5 html, old campaign),
  `fdp/` (5 png, old), `master_results.html`, `lewam_equations.html`, `sr_comparison.png` (old)
- `docs/EXPERIMENTS.md` (658,316; the pre-August experiment log), `docs/WF8_CELLS.md`, `docs/SETTINGS.md`,
  `docs/MACHINE_TRANSFER.md`, `docs/MOT_SUMMARY.md`, `docs/pointmaze_sizes.md`, `docs/proposal/`
- `docs/HANDOFF.md` (this document)
- `scripts/collect_board.py` docstring (board semantics)

HDFS: `HROOT/wf8/README.md` (dataset provenance and the pod-side name table), `HROOT/code/REACHER_HANDOFF.md` (6,294),
`HROOT/WHITELIST_TODO.md`.

Devbox: memory directory `/home/tiger/.claude/projects/-home-tiger-lewam/memory/` (29 files, 360 KB), index
`MEMORY.md` (6,459). Most useful for a newcomer: `merlin-ops.md` (22,760; launch discipline, queues, kills),
`cell-protocol-map.md`, `horizon-units.md`, `lewm-eval-equivalence.md`, `b1-official-baselines.md`,
`launch-use-submitv2-pathB.md`, `unified-experiments.md` (176,620; the pre-jointflow architecture and results).
Scratch of this session: `/home/tiger/.claude/jobs/5f788414/tmp/` (board snapshots `board_0551.{md,json}`,
`board_1530.*`, `board_0945.*`, the E2 paired-test scripts, B1 diagnostics, commit messages). Project hub
`/home/tiger/lewam_project/` (`README.md`, `jobs/`, `sync_ctl.sh`).
