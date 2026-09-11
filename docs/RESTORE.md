# LeWAM restore guide for a new machine (written 2026-09-12, the day the ByteDance/Merlin server was cut off)

For an agent (or a person) who starts on a fresh filesystem and cluster with nothing but this repository. It
says where every artifact went, what the state of the research was when the server died, how to tell what is
finished from what is not, how to put the project back together, and which files carry paths and cluster
assumptions that must change. Read it before touching anything else. The companion documents:

- `docs/HANDOFF.md`: the migration manifest (every path with size and role) and the "what is left" list.
- `docs/RECIPES.md`: the chronological experiment log; the tail is the latest state. Recipes for every arm.
- `docs/PAPER_ROBUSTNESS_PLAN.md`: the experiment plan the results feed.
- `docs/results/board/`: the results board snapshots (latest file = latest numbers).
- `CLAUDE.md`, `merlin/MERLIN.md`: the old machine's instructions; cluster-specific, kept for reference.

## 1. Where everything is

Three copies exist. Check them in this order.

| what | where | layout |
|---|---|---|
| code | GitHub `git@github.com:MinghaoFu/lewam.git`, branch `lewam-jointflow` | the repo; every commit that ever ran on the cluster is on this branch (the pod tarballs `lewam_jointflow_<sha>.tar.gz` were `git archive` of these commits) |
| outside-the-repo working state | GitHub, branch `workspace-2026-09-12`, folder `workspace/` | `jobs/` (Merlin YAMLs + the submit guard), `hdfs_code/` (every pod entry script, ~700), `memory/` (the session memory: rules, ops, results notes), `global_CLAUDE.md`, `scratch/` (helper scripts and small logs, e.g. `make_view_links.py`, `hf_up.py`, monitors), `job_name_map.tsv` (coded job name -> real name -> job id), `merlin-docs/` |
| checkpoints | Hugging Face, private dataset repo `mh-hf/lewam-checkpoints-2026-09-12` | `ckpts/<same subpaths as HROOT/ckpts>` for the manifest's results tier plus `ckpts/dp_tc/` (DP-C ports) and the 3-view runs; `code/lewm_main_eval/hf_release_native` (the LeWM release checkpoint) |
| datasets | Hugging Face, private dataset repo `mh-hf/lewam-data-2026-09-12` | `wf8/train/*.h5`, `wf8/_source/drawer_raw.h5`, `wf8/env/`, `wf8/eval/`, `wf8/train/_views/`, `wf8/README.md`; files over 49 GB are split: `cat <name>.part-* > <name>`; `preload_cache/{drawer_cleanup_fixed,transport}/` holds the six 3-view strided caches |
| code + ops archive | Hugging Face, private dataset repo `mh-hf/lewam-code-ops-2026-09-12`, and the backup directory | `lewam_code_ops_2026-09-12.tar.gz` (~1.3 GB): a git bundle of every branch, the working tree at the cutoff, the workspace snapshot, and the manifest's code tier from HDFS (sim sources, LeWM eval stack, DP env); README inside |
| Minghao's backup | the tree `lewam_backup_2026-09-12/` he copied off the cluster | full original absolute paths underneath: `<backup>/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/{ckpts,code,wf8}` and `<backup>/home/tiger/{lewam_project/jobs,.claude/projects/-home-tiger-lewam/memory,.job_name_map.tsv}`; the same content as the manifest tiers 1-results, 1-code, 2-data, 4-devbox (caches were not staged) |

`HROOT` below means the old root `/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam`. Choose a new root and
recreate the subtree `wf8/`, `preload_cache/`, `preload_cache_u8/`, `ckpts/` under it.

Upload state at 15:30 PDT on the cutoff day: the checkpoints repo holds the whole results tier plus `ckpts/dp_tc`
and the 3-view runs; the data repo holds all 13 training files and `drawer_raw.h5` byte-exact against the manifest
(parts summed), `env/`, the three real `eval/` link files, `_views/`, the README, and the six 3-view strided caches
(`preload_cache/drawer_cleanup_fixed/*`, `preload_cache/transport/*`, scene + both wrists); the code archive is
1.7 GB. The running jobs' last epochs were re-uploaded by the 15:30 and 16:30 PDT sweeps.

Integrity checks: `docs/handoff_manifest.txt` lists every migrated path with its byte size (`tier bytes path
note`); compare sizes after download. Each checkpoint directory carries its own `train.log` and
`jointflow_config.json` (or `dp_config.json`); a finished run has a `done` file and a `TRAIN_OK` line in its
heartbeat log `hb_<arm>_s<seed>.log` next to it.

## 2. What was done (state at the cutoff)

The method is LeWAM: a joint latent world-action model (MoT transformer over a from-scratch encoder, flow-matching
action head, mse state head, optional SIGReg on the latents), evaluated in two protocols: task-conditioned (TC:
toolhang, transport, drawer, cube; reactive policy, success rate over 3 eval seeds x 50 episodes) and
goal-reaching (GR: pusht, tworoom, pointmaze_large, reacher_policy, plus the TC cells re-cast as GR; a 7-mode
planner ladder gc/plan/grad/gradtr/steer/cemp/rand, 3 seeds x 50). The two arms compared everywhere are mse+noreg
and mse+vanilla SIGReg at width 192 (`fx_nm192` / `fx_vsig192`; TC names `_fx_mnm192` / `_fx_mvsig192`).

Results as of 2026-09-12 (details, per-seed numbers and the exact commands: `docs/RECIPES.md`, the board files):
- All single-view TC and GR cells are trained and evaluated; the board is `docs/results/board/board_2026-09-08_0551.md`
  plus the later RECIPES entries (transport TC task-only, DP-C port).
- Baselines: DP-T (budget-matched diffusion policy on our data) on all TC cells; DP-C ported from the official
  diffusion_policy repo (`lewam/models/dp_policy.py`, `scripts/train_dp.py`): toolhang task-only 84.7 +- 3.1 vs LeWAM
  84.0 +- 4.0 (noreg) / 80.0 +- 3.5 (SIGReg); the published DP-C/BC-RNN checkpoints score 0/50 on our robosuite 1.5
  eval (domain gap; B1 closed, BC-RNN dropped by the owner).
- Multi-view (3 cameras: scene + both wrists) is implemented for TC and GR (`--views pixels,pixels_r0eih,pixels_r1eih`;
  GR goal views via `--goal_views`); the drawer 3-view TC SIGReg arm trained to epoch 120 (val action loss 0.194 vs
  0.211 single-view); its 3-seed eval was running at the cutoff (results, if they landed, are in
  `ckpts/jointflow_tc/hb_evtask_tc_drawer_fx_mvsig192_3v_s42.log`).
- Interrupted by the cutoff (their full training state was synced to HDFS after every epoch, so they resume):
  DP-C drawer (`ckpts/dp_tc/dp_drawer_dpc_s42`, ~epoch 112/120), DP-C transport (`ckpts/dp_tc/dp_transport_dpc_s42`,
  ~epoch 80/120), LeWAM transport 3-view TC (`ckpts/jointflow_tc/tc_transport_fx_mvsig192_3v_s42`, ~epoch 68/120).
  The exact last epoch is the last `PROGRESS`/`SNAP` line of each heartbeat log and the epoch stored in
  `dp_full.pt` / `jointflow_full.pt`.

## 3. What remains (the queue at the cutoff, in order)

1. Resume and finish the three interrupted trainings (section 5.6), then their task evals (3 seeds x 50).
2. Launch the four 3-view GR trainings (drawer, transport x noreg, SIGReg): the exact flags are in
   `workspace/jobs/{dr,tr}-gr-{nm,vsig}192-3view.yaml` (branch `workspace-2026-09-12`) and in RECIPES
   (2026-09-12 entry). They need the six 3-view strided caches (on HF under `preload_cache/`) and the entry
   `workspace/hdfs_code/jf_gr_cdc5b7f.sh` as the reference for the command line. Then the 7-mode ladders with
   `workspace/hdfs_code/jf_grev_cdc5b7f_views.sh <cell> <ckpt dir> gc,plan,grad,gradtr,steer,cemp,rand <tag>
   robot0_eye_in_hand,robot1_eye_in_hand`.
3. The owner's open decision: noreg twins of the 3-view TC runs (drawer, transport).
4. Update the board (`scripts/collect_board.py --ckpts <root>/ckpts`), RECIPES and the paper tables.

To recompute the queue yourself: run the board script (its INCOMPLETE-ladder report lists every GR cell x arm
missing planner modes or seeds), grep the heartbeat logs for `TRAIN_OK` / `ALL_DONE` / `success_rate`, and read
`docs/HANDOFF.md` section 3.

## 4. Environment

- Python 3.11, CUDA GPU (H100/A100 were used; V100 lacks the kernels our torch build needs). Install
  `requirements.txt` without the torch/nvidia pins, then `stable-worldmodel==0.1.1` and `torchvision<=0.24.1` (the
  pods did exactly `grep -vE '^torch==|^torchvision==|^nvidia-|^cuda-' requirements.txt | sed 's/==/<=/'`).
- The loader's home: set `STABLEWM_HOME` (any writable dir). Datasets are symlinked into
  `$STABLEWM_HOME/datasets/<pod name>` and checkpoints for eval into `$STABLEWM_HOME/checkpoints/<run>/`. Also set
  `SPT_CACHE_DIR=$STABLEWM_HOME/spt`, `XDG_CACHE_HOME=$STABLEWM_HOME/xdg`, `HF_HUB_OFFLINE=1`, `MPLCONFIGDIR`.
- Rendering: `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy`, `MUJOCO_EGL_DEVICE_ID=<gpu index>`.
- Sim stacks for the TC cells: the source tarball `wf8_sim_src.tar.gz` (code tier of the backup; contains robosuite
  and dexmimicgen sources) extracted and put first on `PYTHONPATH` (`<src>:<src>/dexmimicgen`), plus
  `pip install --target <dir> robosuite==1.5.1 mujoco==3.2.3 numpy==1.26.4 egl_probe matplotlib termcolor gymnasium`
  with that target dir on `PYTHONPATH`, and `pip uninstall robomimic`. Reacher needs `dm_control==1.0.43 mujoco==3.10.0`.
- Env metadata the sim envs read: `wf8/env/robomimic_raw` (`ROBOMIMIC_RAW`), `wf8/env/{drawer,transport}_model_xml.h5`
  (`DEXMG_XML_H5`), `assets/dexmg_{drawer,transport}_env_meta.json` in the repo (`DEXMG_ENV_META`), and the
  thresholds/flags the eval entry exports per cell (`workspace/hdfs_code/jf_grev_cdc5b7f_views.sh` lines 54-66:
  `DEXMG_DROP_DIMS=11,23` for drawer, `*_GOAL_THRESHOLD`, `*_EEF_THRESHOLD`, `ROBOMIMIC_VIEWS` / `DEXMG_VIEWS`).

## 5. Restore procedure

### 5.1 Code
`git clone git@github.com:MinghaoFu/lewam.git && git checkout lewam-jointflow`. For the outside-the-repo state:
`git checkout workspace-2026-09-12 -- workspace` (or clone the branch separately). Copy `workspace/memory/*` into
the new machine's Claude memory directory (`~/.claude/projects/<cwd with / replaced by ->/memory/`) so the standing
rules (naming, eval invariants, horizon units, confirm-before-launch) carry over.

### 5.2 Datasets
Download `mh-hf/lewam-data-2026-09-12` into `<root>/`; reassemble split files (`cat cube.h5.part-* > cube.h5`,
same for `transport_3view.h5` and `reacher_policy.h5`); delete the parts. Expected sizes: `docs/handoff_manifest.txt`
tier 2-data. Pod-side dataset names the configs expect (link `<root>/wf8/train/<file>` to
`$STABLEWM_HOME/datasets/<name>`): pusht.h5 -> `pusht_expert_train.h5`, tworoom.h5 -> `tworoom.h5`,
pointmaze_large.h5 -> `pointmaze_large.h5`, reacher_policy.h5 -> `reacher.h5`, toolhang.h5 -> `tool_hang.h5`,
transport.h5 -> `transport.h5`, drawer.h5 -> `drawer_cleanup_fixed.h5`, cube.h5 -> `ogbench/cube_single_expert.h5`;
eval variants from `wf8/eval/<cell>.h5` -> `<name>_ev.h5` where they exist.

`wf8/eval/` on the old root also held plain symlinks `<cell>.h5 -> ../train/<cell>.h5` for cube, pointmaze,
pointmaze_large, pusht, reacher and tworoom; they were not uploaded (they would have duplicated the training
files), recreate them with `ln -s`. Two kinds of h5 files are external-link files and store the absolute paths of
their targets, so they must be regenerated on the new root: `wf8/train/drawer.h5` (links into `wf8/_source/drawer_raw.h5`) and everything under
`wf8/train/_views/` and `wf8/eval/`. The builder is `workspace/scratch/make_view_links.py <out.h5> <primary.h5>
[<column>=<file.h5> ...]` (e.g. `pixels_r0eih=<root>/wf8/train/drawer_3view.h5`); `h5dump -H` or `h5py` shows the
old targets, replace the root. `wf8/README.md` documents the provenance and the column names.

### 5.3 Frame caches
Trainers read caches, never the h5 directly. Layout: `<root>/preload_cache/<stem>/<stem>_fs5_i224[<suffix>].{frames.npy,aux.npz,meta.json}`
where `<stem>` is the pod-side h5 name without `.h5`, suffix `` = strided fp16 (GR), `_raw` = every frame (TC),
`.pixels_r0eih` / `.pixels_r1eih` = a wrist camera (from a `_views` link h5), and `preload_cache_u8/` for the uint8
strided caches of pusht and reacher_policy. Build: `python3 scripts/make_preload_cache.py --h5 <h5> --out
<root>/preload_cache/<stem> [--anchor_rate raw] [--u8] [--pixels_key pixels_r0eih]` (needs the repo on PYTHONPATH).
Measured rate: ~20 MB/s of cache per process (drawer strided 18 GB in 16 min), so the full set (730 GB) is ~2 h of
wall time with 6 builders. The six 3-view strided caches are on HF; everything else is rebuilt. The trainers
locate the cache root from `--frames_cache <dir>` or the env `LEWAM_CACHE_DIR`.

### 5.4 Checkpoints
Download `mh-hf/lewam-checkpoints-2026-09-12` into `<root>/` (it recreates `ckpts/...`). A run directory holds
`jointflow_best.pt` (best val), `jointflow_latest.pt`, `jointflow_full.pt` (full training state: model, optimizer,
scheduler, epoch), `jointflow_config.json`, `train.log`, `grad_probe.jsonl`, `snap_ep<N>.pt`; DP-C runs hold
`dp_best.pt`, `dp_latest.pt`, `dp_full.pt`, `dp_config.json`. Eval stages `jointflow_best.pt` + `jointflow_config.json`
into `$STABLEWM_HOME/checkpoints/<run>/` and passes `policy=<run>`.

### 5.5 Files that carry the old machine's paths or cluster assumptions
| file | what to change |
|---|---|
| `scripts/train_jointflow.py` (2 places), `scripts/train_lewam_unified.py`, `scripts/train_lewam_gc.py`, `scripts/train_crossattn.py` | the default cache root; set `LEWAM_CACHE_DIR` or pass `--frames_cache` |
| `lewam/envs/robomimic_gc_env.py` | default `ROBOMIMIC_RAW`; set the env var to `<root>/wf8/env/robomimic_raw` |
| `scripts/collect_board.py` | `--ckpts` default = `<root>/ckpts` |
| `scripts/handoff_manifest.py` | the roots at the top (only if you regenerate the manifest) |
| `workspace/hdfs_code/*.sh` (pod entry scripts) | `HROOT=...` at the top, `/opt/tiger` scratch, `/tmp`; the logic (untar the code, pip, link datasets, run the trainer, heartbeat lines, snapshots, chained eval) is what to keep |
| `workspace/jobs/*.yaml`, `workspace/jobs/submit_guard.sh` | Merlin-specific (queue names, hdfs volumes, U13 env keys, coded job names); replace the launcher entirely, keep the `bash <entry> <args>` lines as the command record |
| `CLAUDE.md`, `merlin/MERLIN.md`, `merlin/example_*` | old-machine instructions; rewrite the working-space map for the new machine |

### 5.6 Resuming an interrupted training
Copy the run's `jointflow_full.pt` (or `dp_full.pt`) into a fresh run dir and start the trainer with the original
flags plus `--resume --run_dir <run dir> --ckpt_sync_dir <where to sync>`; it continues from the stored epoch with
the optimizer, scheduler and RNG state (`scripts/train_dp.py` lines 173-182; `scripts/train_jointflow.py` around
`args.resume`). The original flags of every run are the `python3 ... train_*.py` line in its `train.log` head and
the entry script named in `entry_<arm>_s<seed>.log`.

### 5.7 Evaluation commands
- GR ladder: `scripts/eval_gip.py --config-name <cell> policy=<run> +gip_eval.mode=jointflow_gc|jointflow_plan
  [+gip_eval.plan_mode=best_of_k|grad|cem|steer ...] eval.num_eval=50 seed=<42|0|1>`; the per-mode flag sets are
  lines 88-101 of `workspace/hdfs_code/jf_grev_cdc5b7f_views.sh`. `success_rate` is grepped from the log.
- TC task-only: `workspace/hdfs_code/jf_tc_ev_85b5df9_task_views.sh` (mode `jointflow_policy`, `task_only`, the
  camera list as the last arg); DP-C: `dp_tc_ev_c461482.sh`.
- Never add flags to `scripts/eval_gip.py` / `scripts/eval.py` (owner rule); use existing knobs and separate runs.

## 6. Pitfalls that cost time before (details in `workspace/memory/`)
- `--H_max` and `h_norm` count frame-skipped states (anchors), not raw steps: `--H_max 10` = 50 raw steps at fs 5.
- Every planning eval is the full 7-mode ladder, 3 seeds, mean +- std with timing; single-sample numbers are diagnostic only.
- Batch sizes: GR recipe 128 / lr 1.5e-4 / 50 epochs / warmup 5; TC recipe 64 / lr 1e-4 / 120 epochs / warmup 10 (fp32).
- Multi-camera TC at batch 64 fits on an 80 GB card with the raw caches in RAM (drawer 3 x 45 GB) or memory-mapped
  from local NVMe (transport 3 x 63 GB); `--cache_mmap` costs nothing measurable.
- Cluster copies over a fuse mount: plain `cp` streams at ~75 MB/s; `rsync` crawls at ~1 file/s. Hugging Face from
  the old devbox needed IPv4 forced (`workspace/scratch/hf_up.py`); check `curl -4` vs `curl -6` before blaming a block.

## 7. Final state (server cleared 15:57 PDT, 2026-09-11 Pacific = 06:57 CST 2026-09-12 on the box)

The HDFS tree `HROOT` and the repo checkout were deleted at 15:57 PDT, half an hour before the planned final sweep,
and the three running trainings were killed. The last captured state is therefore the 15:30 PDT sweep (run
15:31-15:36, verified byte-exact against the backup directory) and the HF re-uploads right after it:

- DP-C drawer `ckpts/dp_tc/dp_drawer_dpc_s42`: last synced epoch ~104 of 120 (`dp_full.pt` resumes it; val loss
  0.0230 at epoch 100, 0.0260 at 60, 0.0272 at 40).
- DP-C transport `ckpts/dp_tc/dp_transport_dpc_s42`: ~epoch 68 of 120 (val loss 0.0353 at epoch 60).
- LeWAM transport 3-view TC `ckpts/jointflow_tc/tc_transport_fx_mvsig192_3v_s42`: epoch 60 snapshot + the synced
  latest/full state a few epochs later (val action loss 0.449 at epoch 60 vs 0.464 at 40).
- Drawer 3-view TC eval (`tc_drawer_fx_mvsig192_3v_s42`, task protocol, both wrists rendered, 50 episodes per
  seed): seed 1 = 58.0, seed 0 = 52.0; seed 42 finished at 15:56 PDT but its log was deleted before it could be
  read, so the number is lost. Treat the cell as n = 2 x 50 (mean 55.0) until re-evaluated. The single-view drawer
  rows for comparison are in `docs/RECIPES.md` and the board.
- Everything else listed in section 1 was complete and verified before the deletion; the backup directory
  `lewam_backup_2026-09-12/` was still present at 15:57 PDT and is redundant with the HF repos.
