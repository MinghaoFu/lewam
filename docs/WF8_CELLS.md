# WF8: the four new cells (tool_hang · drawer_cleanup · transport · pointmaze)

How to train and evaluate the four new goal-reaching cells on this branch. Everything below has
been run end-to-end unless marked otherwise. The other four WF8 cells (tworoom, reacher, pusht,
cube) use the existing configs.

## Datasets and caches (already on the cluster)

Canonical names — **no suffix = official native-224**; variants carry a suffix:

| cell | train dataset (`$STABLEWM_HOME/datasets/`) | source on HDFS (byte_ad_audit …/worldforge10) | preload cache stem |
|---|---|---|---|
| pointmaze | `pointmaze.h5` (1000 eps × 1000 steps) · `pointmaze_300.h5` | `pm_render/` | `pointmaze` / `pointmaze_300` |
| tool_hang | `tool_hang.h5` (official re-render) · `tool_hang_384ds.h5` | `official_render/` | `tool_hang` / `tool_hang_384ds` |
| transport | `transport.h5` (official, 1029 eps) · `transport_84.h5` | `official_render/` | `transport` |
| drawer | `drawer_cleanup_fixed.h5` (22-dim view; keep the backing `drawer_cleanup.h5` beside it) | `datasets/` | `drawer_cleanup_fixed` |

Preload caches live at `…/lewam/preload_cache/<stem>/<stem>_fs5_i224.{frames.npy,aux.npz}` — the
default lookup of `--frames_cache auto`. Build new ones with `scripts/make_preload_cache.py`.

**Eval must use the `_ev` views** (`tool_hang_ev.h5`, `transport_ev.h5`,
`drawer_cleanup_fixed_ev.h5`, next to their sources): the flat files carry a per-episode
`model_xml` column and swm's loader row-indexes every column, so step-indexing a `(N_eps,)` column
IndexErrors. The views link every column except `model_xml`. pointmaze has no such column.

## Train (LeWAM-unified)

```bash
python3 scripts/train_lewam_unified.py \
  --dataset_name pointmaze.h5 --run_name uni_pointmaze \
  --frames_cache auto --cache_mmap --epochs 50 --H_max 50 \
  --ckpt_sync_dir <hdfs-dir>            # cache-backed: pointmaze = 16 min for 50 epochs
```

Same shape for the other cells (swap the dataset name). No env, no sim stack needed for training.
Baselines: `scripts/train.py model=lewm data={pointmaze,drawer_cleanup,robomimic_tool_hang,robomimic_transport} action_pred.enabled=false`
(online, no cache) and `scripts/train_gcidm.py --weights <lewm ckpt>` (frozen GC-IDM).

## Eval (policy / CEM planning / GC-IDM)

```bash
# reactive head            planning (CEM)                 GC-IDM
+gip_eval.mode=unified_policy | unified_cem              | gcidm
python3 scripts/eval_gip.py --config-name {pointmaze,toolhang,transport,drawer} \
  policy=<run_name> +gip_eval.mode=unified_policy \
  eval.dataset_name=<stem or _ev view> eval.num_eval=50 seed=42
# LeWM CEM baseline: scripts/eval.py --config-name <cell> policy=<run>/weights_epoch_<E>.pt solver=cem
```

Envs register automatically from the gym id in the config (`_register_env` in eval_gip.py /
eval.py). Do NOT override `world.num_envs`: the harness asserts `num_envs == eval.num_eval`.

Per-cell env vars (already encoded in the campaign wrappers under `~/wf8_render/campaign_*.sh`):

```bash
# tool_hang
ROBOMIMIC_RAW=<…>/env_inputs/robomimic_raw  ROBOMIMIC_GOAL_THRESHOLD=0.04  ROBOMIMIC_EEF_THRESHOLD=0.04
# transport   (model_xml MUST be the 1029-episode extraction, not the old 400-ep file)
DEXMG_ENV_META=assets/dexmg_transport_env_meta.json  DEXMG_XML_H5=<…>/official_render/transport_model_xml.h5
DEXMG_GOAL_THRESHOLD=0.04  DEXMG_EEF_THRESHOLD=0.10
# drawer
DEXMG_ENV_META=assets/dexmg_drawer_env_meta.json  DEXMG_XML_H5=<…>/env_inputs/drawer_model_xml.h5
DEXMG_DROP_DIMS=11,23  DEXMG_GOAL_THRESHOLD=0.04  DEXMG_EEF_THRESHOLD=0.04
# pointmaze: nothing (threshold defaults to OGBench's own goal_tol = 1.0)
```

Success = every task object within GOAL_THRESHOLD of the goal state AND end-effector(s) within
EEF_THRESHOLD, held for `GOAL_HOLD_K` consecutive steps (default 1 = reach-anytime; 0.04/0.04 are
calibrated values — at the 0.15 default most windows are already solved at t=0).

## Gotchas that cost real time (all hit at least once)

- `frameskip=5` and `num_steps=4` are hardcoded in the unified trainer; eval `action_block: 5`
  must stay in lockstep.
- The sim stack for manipulation evals: `pip install --target /tmp/u_sim robosuite==1.5.1
  mujoco==3.2.3 numpy==1.26.4 egl_probe matplotlib` + PYTHONPATH the robomimic-0.3.1/dexmimicgen
  source trees. Do NOT use PYTHONUSERBASE (it hides ~/.local where swm lives) and do NOT pip
  install robomimic (the wheel imports mujoco_py).
- h5py cannot write onto HDFS fuse — write locally, then `cp`.
- First measured numbers (n=25, seed 42, pointmaze): unified_policy 100.0, unified_cem 20.0.
