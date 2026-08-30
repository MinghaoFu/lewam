"""GIP step-1 success-rate eval.

Reuses eval.py's env/dataset machinery; the policy is chosen by gip_eval.mode:

  unified_policy        LeWAM-Unified reactive: the goal-conditioned action head run directly (the
                        representation channel; never touches the dynamics)
  unified_cem           LeWAM-Unified CEM: the dynamics head rolls candidate action blocks to the goal
  unified_grad          Gradient MPC: gc_head warm-start refined by Adam on the frozen dynamics
                        (terminal-latent cost); plan = best iterate by model cost
  unified_prompt_mpc    Prompt-MPC (ours): a per-episode prompt delta on the goal latent, optimized
                        through the frozen policy; pm_cost=anymin aligns the cost with reach-anytime
  split_policy          LeWAM-Split reactive action head
  gcidm                 frozen-LeWM + GCIDM head baseline
  bc (default)          a plain GIP action head run as a reactive policy

Example:
  python eval_gip.py --config-name pusht policy=<run_name> +gip_eval.mode=unified_policy
"""

import stable_worldmodel.data.formats.hdf5  # HDF5 self-registers on import
import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import gymnasium
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm

import lewam.models.gip as gip

# WF8 env modules register their gym ids on import (side effect), and nothing on this eval path
# imports them -- lewam/envs/__init__.py is deliberately empty. Resolve the module from the gym id
# so existing configs pay nothing and a new cell needs only its configs/eval yaml.
_ENV_MODULES = {
    "swm/RoboMimicGC-v0": "lewam.envs.robomimic_gc_env",
    "swm/OGBCube-v0": "lewam.envs.cube_env",
    "swm/DexMimicGen-v0": "lewam.envs.dexmimicgen_env",
    "swm/PointMaze-v0": "lewam.envs.pointmaze_env",
    "swm/RoboMimic-v0": "lewam.envs.robomimic_env",
}


def _register_env(env_name):
    mod = _ENV_MODULES.get(str(env_name))
    if mod:
        import importlib

        importlib.import_module(mod)


class NoEarlyStop(gymnasium.Wrapper):
    """Strip `terminated` so dataset eval ('wait' mode) never freezes an env at first success; every
    episode runs the full eval_budget. Latches the RAW env terminated per step so one run yields both
    metrics: `ever` = protocol-compatible ever-reached SR (identical to the frozen eval's SR — the
    latch fires at the same first touch), `last` = success judged from the FINAL state only."""

    def reset(self, **kwargs):
        self.ever = False
        self.last = False
        self.t = 0
        self.reach_at = None                   # first step the raw success predicate fired
        self.nan_at = None                     # first step the reported agent state went non-finite
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.t += 1
        raw = bool(terminated)
        self.ever = self.ever or raw
        self.last = raw
        if raw and self.reach_at is None:
            self.reach_at = self.t
        try:
            s = np.asarray(info["state"], dtype=float).reshape(-1)[-7:]
            if self.nan_at is None and not np.isfinite(s[:2]).all():
                self.nan_at = self.t
        except Exception:
            pass
        # ALSO strip the raw flag from info: the policy adapters read info['terminated'] to decide
        # which envs are active and hand NaN actions to "done" envs (gip.py action batch init) --
        # with continued stepping that nukes the agent body via a NaN PD target.
        if isinstance(info, dict) and "terminated" in info:
            info["terminated"] = False
        return obs, rew, False, truncated, info


class BudgetTruncate(gymnasium.Wrapper):
    """Truncate the env at its own step budget. `budget` is set per env after world construction."""

    budget = None

    def reset(self, **kwargs):
        self.t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.t += 1
        if self.budget is not None and self.t >= self.budget:
            truncated = True
        return obs, rew, terminated, truncated, info


def _subgoal_kwargs(cfg, dataset, episodes, starts):
    """gip_eval.subgoal_every > 0: per env, the demo's fs-strided frames from its start step
    (uint8 HWC), read straight from the eval h5 (the dataset column would load every pixel)."""
    ge = cfg.get("gip_eval", {})
    if int(ge.get("subgoal_every", 0)) <= 0:
        return None
    import h5py
    fs = int(cfg.plan_config.action_block)
    col = gip.episode_col(dataset)
    ep_arr = np.asarray(dataset.get_col_data(col)).reshape(-1)
    st_arr = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    h5_path = Path(swm.data.utils.get_cache_dir(), "datasets", str(cfg.eval.dataset_name) + ".h5")
    frames = []
    with h5py.File(h5_path, "r") as f:
        px = f["pixels"]
        for e, s0 in zip(episodes, starts):
            rows = np.nonzero(ep_arr == e)[0]
            rows = rows[np.argsort(st_arr[rows])][int(s0)::fs]
            order = np.argsort(rows)                  # h5py needs increasing fancy indices;
            frames.append(px[rows[order]][np.argsort(order)])   # then restore step order
    print(f"[subgoal] demo frame stacks for {len(frames)} envs, anchors "
          f"{min(len(a) for a in frames)}..{max(len(a) for a in frames)}", flush=True)
    return dict(subgoal_frames=frames)


def _policy_kwargs(cfg, dataset, episodes, starts):
    ok = _oracle_kwargs(cfg, dataset, episodes, starts)
    sk = _subgoal_kwargs(cfg, dataset, episodes, starts)
    if ok is None and sk is None:
        return None
    return {**(ok or {}), **(sk or {})}


def _oracle_kwargs(cfg, dataset, episodes, starts):
    """plan_mode=oracle_bok: the replayed demos' action sequences from each env's start step (raw
    env units), for the oracle candidate sets (jointflow _oracle_bok / OracleSolver)."""
    if str(cfg.get("gip_eval", {}).get("plan_mode", "")) != "oracle_bok":
        return None
    col = gip.episode_col(dataset)
    ep_arr = np.asarray(dataset.get_col_data(col)).reshape(-1)
    st_arr = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    act_arr = np.asarray(dataset.get_col_data("action"))
    expert = []
    for e, s0 in zip(episodes, starts):
        rows = np.nonzero(ep_arr == e)[0]
        rows = rows[np.argsort(st_arr[rows])]
        expert.append(act_arr[rows][int(s0):])
    print(f"[oracle] demo action sequences for {len(expert)} envs, lengths "
          f"{min(len(a) for a in expert)}..{max(len(a) for a in expert)}", flush=True)
    return dict(expert_actions=expert)


@hydra.main(version_base=None, config_path="../configs/eval", config_name="pusht")
def run(cfg: DictConfig):
    mode = cfg.get("gip_eval", {}).get("mode", "bc")
    assert cfg.policy != "random", "set policy=<gip_run_name>"

    # -- env + data context (shared helpers)
    _register_env(cfg.world.env_name)
    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    episodes, starts, goal_offsets = gip.sample_eval_episodes(cfg, dataset)
    if goal_offsets is not None:
        # full-traj protocol: budget = 2x the longest per-episode offset
        cfg.eval.eval_budget = 2 * int(max(goal_offsets))
        print(f"[full-traj] {len(goal_offsets)} episodes, offsets "
              f"{min(goal_offsets)}..{max(goal_offsets)}, eval_budget={cfg.eval.eval_budget}")

    # Slice into shards only after eval_budget is set above, so every shard shares the full-set budget.
    shard_count = int(cfg.get("gip_eval", {}).get("shard_count", 1))
    shard_idx = int(cfg.get("gip_eval", {}).get("shard_idx", 0))
    if shard_count > 1:
        sl = slice(shard_idx, None, shard_count)
        episodes, starts = episodes[sl], starts[sl]
        if goal_offsets is not None:
            goal_offsets = goal_offsets[sl]
        cfg.world.num_envs = len(episodes)
        print(f"[shard] {shard_idx}/{shard_count}: {len(episodes)} of the full pick-list", flush=True)

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "horizon*action_block must be <= eval_budget"
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    no_early_stop = bool(cfg.get("gip_eval", {}).get("no_early_stop", False))
    # 'per_episode' = 2x each episode's own offset; 'global' = 2x the max offset for every episode
    budget_mode = str(cfg.get("gip_eval", {}).get("budget_mode", "per_episode"))
    per_episode_budget = budget_mode == "per_episode" and goal_offsets is not None
    wrappers = ([BudgetTruncate] if per_episode_budget else []) + ([NoEarlyStop] if no_early_stop else [])
    world = swm.World(**cfg.world, image_shape=(224, 224),
                      extra_wrappers=wrappers or None)
    if no_early_stop:
        print("[GIP] no_early_stop ON: envs run the full eval_budget; reporting ever-reached AND final-state SR")
    if per_episode_budget:
        if world.num_envs == len(goal_offsets):
            for i, _e in enumerate(world.envs.envs):
                _w = _e
                while not isinstance(_w, BudgetTruncate):
                    _w = _w.env
                _w.budget = 2 * int(goal_offsets[i])
            print(f"[full-traj] budget_mode=per_episode: per-env budgets "
                  f"{2 * min(goal_offsets)}..{2 * max(goal_offsets)}", flush=True)
        else:
            per_episode_budget = False
            print(f"[full-traj] WARN budget_mode=per_episode needs num_envs==episodes "
                  f"({world.num_envs} vs {len(goal_offsets)}); falling back to the global budget", flush=True)

    # -- optional strict success criterion: judge the BLOCK pose only (pos dims [2:4] + angle dim 4),
    # agent excluded. Mirrors env.eval_state (incl. angle wrap); pusht state layout only.
    strict = str(cfg.get("gip_eval", {}).get("success", "default"))
    if strict == "strict":
        import types
        pos_thr = float(cfg.get("gip_eval", {}).get("strict_pos_thr", 20.0))
        ang_thr = float(cfg.get("gip_eval", {}).get("strict_ang_thr", np.pi / 9))

        def _strict_eval_state(self, goal_state, cur_state):
            g, c = np.asarray(goal_state), np.asarray(cur_state)
            pos_diff = np.linalg.norm(g[2:4] - c[2:4])
            angle_diff = np.abs(g[4] - c[4])
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            success = bool(pos_diff < pos_thr and angle_diff < ang_thr)
            return success, float(np.linalg.norm(g - c))

        n_patched = 0
        for _e in world.envs.envs:
            _u = _e.unwrapped
            if hasattr(_u, "eval_state"):
                _u.eval_state = types.MethodType(_strict_eval_state, _u)
                n_patched += 1
        assert n_patched == world.num_envs, f"strict criterion patched {n_patched}/{world.num_envs} envs"
        print(f"[GIP] STRICT success ON: block-only pos<{pos_thr:.0f}px angle<{ang_thr:.3f}rad ({n_patched} envs)")
    transform = {"pixels": gip.img_transform(cfg), "goal": gip.img_transform(cfg)}
    process = gip.build_process(cfg, dataset)

    # -- long-horizon eval: start = settle_t - H, goal = settle_t (the frame from which the BLOCK pose
    # stays within 10px/10deg of the demo's terminal pose -- ~= the drawn-T for success-terminated demos).
    # H = cfg.eval.goal_offset_steps. Truncated demos (len>=246: expert never matched the T) excluded.
    if bool(cfg.get("gip_eval", {}).get("longhorizon", False)):
        H = int(cfg.eval.goal_offset_steps)
        col = gip.episode_col(dataset)
        ep_arr = np.asarray(dataset.get_col_data(col)).reshape(-1)
        st_arr = np.asarray(dataset.get_col_data("state"))
        st_arr = st_arr.reshape(st_arr.shape[0], -1)
        eps_u, seg0 = np.unique(ep_arr, return_index=True)
        seg1 = np.append(seg0[1:], len(ep_arr))
        pos_tol, ang_tol = 10.0, np.deg2rad(10.0)
        cand_ep, cand_start = [], []
        for e, a, b in zip(eps_u, seg0, seg1):
            S = st_arr[a:b]
            T = S.shape[0]
            if T >= 246:
                continue
            blk, ang = S[:, 2:4], S[:, 4]
            ok = np.linalg.norm(blk - blk[-1], axis=1) < pos_tol
            d = np.abs(ang - ang[-1])
            ok &= np.minimum(d, 2 * np.pi - d) < ang_tol
            rev = np.flip(ok)
            run = int(np.argmin(rev)) if not rev.all() else T
            settle = T - run
            if H <= settle < T:
                cand_ep.append(int(e))
                cand_start.append(int(settle - H))
        g = np.random.default_rng(int(cfg.seed))
        pick = np.sort(g.choice(len(cand_ep), size=int(cfg.eval.num_eval), replace=False))
        episodes = [cand_ep[i] for i in pick]
        starts = [cand_start[i] for i in pick]
        print(f"[GIP] LONG-HORIZON eval ON: H={H} raw frames, start=settle-H, goal=settle "
              f"(pool {len(cand_ep)} qualifying episodes)", flush=True)

    # -- optional random goals (reachable by construction): from each real init state, roll the sim
    # forward goal_rollout_steps env-steps of random actions and take the resulting state as the goal.
    # NOTE: this is not recommended for pick-and-place style tasks where random actions only move the agent, not the object
    if bool(cfg.get("gip_eval", {}).get("random_goal", False)):
        import numpy as _np
        import stable_worldmodel.world.world as _wmod
        from stable_worldmodel.world.world import _apply_callables as _apply_cb
        _orig_eig = _wmod._extract_init_goal
        _rng = _np.random.default_rng(int(cfg.seed) + 20240729)
        _H = int(cfg.get("gip_eval", {}).get("goal_rollout_steps", cfg.eval.goal_offset_steps))
        _cbs = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True) or []
        # goal-setting callables consume goal_* columns; everything else sets the INIT (task-agnostic:
        # pusht/tworoom use _set_state, reacher/cube use set_state(qpos,qvel)).
        _GOAL_METHODS = {"_set_goal_state", "set_target_qpos", "set_target_pos"}
        _init_cbs = [c for c in _cbs if c.get("method") not in _GOAL_METHODS]
        _cols = [c for c in dataset.column_names if not str(c).startswith("goal")]
        # Generate random actions the same way the policy does -> un-z-score N(0,1) with the model's
        # action_mean/std. The raw dataset action column can live in a different (larger) space than the
        # env step's action-space box accepts (traced on tworoom: policy actions pass, raw dataset actions
        # don't); un-z-scored actions match the policy's valid output. Clamp for safety.
        _dbg = [True]

        # goal_policy: 'random' (default) or 'seek_block' -- aim actions at the block so the rollout
        # goal differs from the init in BLOCK pose, not just agent pose (pure random moves the block
        # only ~5px on pusht). seek_scale multiplies the expert action magnitude for the seek step.
        # seek_band="lo,hi" (px): snapshot each env's goal at the FIRST step its block displacement
        # enters [lo,hi] -- calibrated-difficulty OOD goals from one rollout (fallback: final state).
        _goal_policy = str(cfg.get("gip_eval", {}).get("goal_policy", "random"))
        _seek_scale = float(cfg.get("gip_eval", {}).get("seek_scale", 1.5))
        _band = cfg.get("gip_eval", {}).get("seek_band", None)
        if _band is not None:                                        # "lo:hi" (":" -- comma breaks hydra CLI)
            _blo, _bhi = [float(x) for x in str(_band).replace(",", ":").split(":")]

        def _rollout_goal(init_state, n):
            world.reset(seed=init_state.get("seed"))
            for i in range(n):                                        # set each env to its real init
                _apply_cb(world.envs.envs[i].unwrapped, _init_cbs, {k: v[i] for k, v in init_state.items()})
            _ucfg = getattr(model, "_unified_cfg", None) or {}
            _am = _np.asarray(_ucfg.get("action_mean", [0.0]), _np.float32).reshape(-1)
            _as = _np.asarray(_ucfg.get("action_std", [1.0]), _np.float32).reshape(-1)
            _aspc = world.envs.envs[0].unwrapped.action_space
            _lo, _hi = _np.asarray(_aspc.low, _np.float32), _np.asarray(_aspc.high, _np.float32)
            if _dbg[0]:
                print(f"[GIP] rollout act: mean={_am} std={_as} env_box=[{_lo},{_hi}] policy={_goal_policy}",
                      flush=True); _dbg[0] = False
            cur_state = (_np.asarray(init_state["state"]).reshape(n, -1)
                         if "state" in init_state else None)
            _i0s = cur_state.copy() if cur_state is not None else None
            _snaps = [None] * n

            def _snap_env(i, infos_):
                d = {}
                for col in _cols:
                    if col == "pixels":
                        f = infos_["pixels"]
                        d["goal"] = _np.asarray(f[i][-1] if _np.ndim(f[i]) > 3 else f[i]).copy()
                    elif col in infos_:
                        v = _np.asarray(infos_[col])
                        d["goal_" + col] = (v[i, -1] if v.ndim > 2 else v[i]).copy()
                return d

            infos = None
            for _ in range(_H):
                if _goal_policy == "seek_block" and cur_state is not None and cur_state.shape[1] >= 5:
                    # pusht state layout: agent xy [0:2], block xy [2:4]. Step toward the block + noise.
                    dvec = cur_state[:, 2:4] - cur_state[:, 0:2]
                    dn = _np.linalg.norm(dvec, axis=1, keepdims=True)
                    dn[dn < 1e-6] = 1.0
                    step_mag = float(_np.mean(_as)) * _seek_scale
                    acts = (dvec / dn * step_mag
                            + _rng.standard_normal((n, len(_as))) * _as * 0.5).astype(_np.float32)
                else:                                                # policy-space random actions
                    acts = (_rng.standard_normal((n, len(_as))).astype(_np.float32) * _as + _am)
                acts = _np.clip(acts, _lo, _hi)
                _, _, _, _, infos = world.envs.step(acts.reshape(*world.envs.action_space.shape))
                if "state" in infos:
                    _s = _np.asarray(infos["state"])
                    cur_state = _s[:, -1] if _s.ndim > 2 else _s
                if _band is not None and cur_state is not None and _i0s is not None:
                    _d = _np.linalg.norm(cur_state[:, 2:4] - _i0s[:, 2:4], axis=1)
                    for _ei in range(n):
                        if _snaps[_ei] is None and _blo <= _d[_ei] <= _bhi:
                            _snaps[_ei] = _snap_env(_ei, infos)
            if _goal_policy == "seek_block" and cur_state is not None and "state" in init_state:
                _i0 = _np.asarray(init_state["state"]).reshape(n, -1)
                _disp = _np.linalg.norm(cur_state[:, 2:4] - _i0[:, 2:4], axis=1)
                print(f"[GIP] seek_block goals: block displacement px p10/p50/p90 = "
                      f"{_np.percentile(_disp, 10):.1f}/{_np.percentile(_disp, 50):.1f}/"
                      f"{_np.percentile(_disp, 90):.1f}  (n={n})", flush=True)
            if _dbg[0]:
                print(f"[GIP] random-goal rollout infos keys: {sorted(infos.keys())}; "
                      f"dataset cols: {sorted(map(str,_cols))}", flush=True); _dbg[0] = False
            # emit all goal_* columns from the post-rollout env info, so each task's goal callable
            # (goal_state / goal_qpos / goal_privileged_* / goal_proprio) finds what it needs.
            # With seek_band: per-env snapshot from the first in-band step (fallback: final state).
            if _band is not None:
                _final = [_snap_env(i, infos) for i in range(n)]
                _per = [(_snaps[i] if _snaps[i] is not None else _final[i]) for i in range(n)]
                _nc = sum(s is not None for s in _snaps)
                if _i0s is not None and "goal_state" in _per[0]:
                    _gd = _np.linalg.norm(_np.stack([_per[i]["goal_state"] for i in range(n)])[:, 2:4]
                                          - _i0s[:, 2:4], axis=1)
                    print(f"[GIP] seek_band[{_blo:.0f},{_bhi:.0f}]px: {_nc}/{n} snapped; GOAL block "
                          f"displacement p10/p50/p90 = {_np.percentile(_gd,10):.1f}/"
                          f"{_np.percentile(_gd,50):.1f}/{_np.percentile(_gd,90):.1f}", flush=True)
                else:
                    print(f"[GIP] seek_band[{_blo:.0f},{_bhi:.0f}]px: {_nc}/{n} envs snapped in-band "
                          f"(rest = final rollout state)", flush=True)
                return {k: _np.stack([_per[i][k] for i in range(n)]) for k in _per[0]}
            goal_state = {}
            for col in _cols:
                if col == "pixels":
                    f = infos["pixels"]
                    goal_state["goal"] = _np.stack([_np.asarray(f[i][-1] if _np.ndim(f[i]) > 3 else f[i])
                                                    for i in range(n)])
                elif col in infos:
                    v = _np.asarray(infos[col])
                    # infos may carry a stack/history dim (e.g. pusht state is (n,1,7)); the dataset goal
                    # columns are (n, dim) -> take the last stacked frame so eval_state/callables match.
                    goal_state["goal_" + col] = v[:, -1] if v.ndim > 2 else v
            return goal_state

        def _random_goal_extract(ds, episodes_idx, start_steps, goal_offset):
            init_state, _drop, vids = _orig_eig(ds, episodes_idx, start_steps, 0)   # keep REAL init
            goal_state = _rollout_goal(init_state, len(episodes_idx))
            return init_state, goal_state, vids

        _wmod._extract_init_goal = _random_goal_extract
        print(f"[GIP] RANDOM-GOAL (rollout) eval ON: goal = init + {_H} random-action steps, "
              f"seed {int(cfg.seed)+20240729}", flush=True)

    # -- model + policy (mode is the single switch)
    # mode=gcidm loads its own frozen-LeWM + GCIDMHead, not the JEPA-checkpoint loader
    if mode == "gcidm":
        policy = gip.build_policy(cfg, None, None, process, transform, goal_offsets=goal_offsets)
    elif mode == "split_policy":
        # LeWAM-Split: its own loader + config; adim = the model's z-scored action block dim.
        model, split_cfg = gip.load_lewam_split_model(cfg.policy, which=cfg.get("seq_which", "best"))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        model._split_cfg = split_cfg
        adim = int(split_cfg["action_dim"])
        policy = gip.build_policy(cfg, model, adim, process, transform, goal_offsets=goal_offsets)
    elif mode == "crossattn_policy":
        model, ca_cfg = gip.load_crossattn_model(cfg.policy, which=cfg.get("seq_which", "best"))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        model._crossattn_cfg = ca_cfg
        adim = int(ca_cfg["action_dim"])
        policy = gip.build_policy(cfg, model, adim, process, transform, goal_offsets=goal_offsets)
    elif mode == "dp_policy":
        model, dp_cfg = gip.load_dp_model(cfg.policy)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        model._dp_cfg = dp_cfg
        adim = int(dp_cfg.task.shape_meta["action"]["shape"][0])
        policy = gip.build_policy(cfg, model, adim, process, transform, goal_offsets=goal_offsets)
    elif mode in ("jointflow_policy", "jointflow_plan", "jointflow_gc"):
        model, jf_cfg = gip.load_jointflow_model(cfg.policy, which=cfg.get("seq_which", "best"))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        model._jointflow_cfg = jf_cfg
        adim = int(jf_cfg["action_dim"])
        policy = gip.build_policy(cfg, model, adim, process, transform, goal_offsets=goal_offsets,
                                  policy_kwargs=_policy_kwargs(cfg, dataset, episodes, starts))
    elif mode in ("unified_policy", "unified_cem", "unified_grad", "unified_prompt_mpc"):
        # LeWAM-Unified: its own loader + config; adim = the model's z-scored action block dim.
        # unified_cem = CEM planner over the dynamics head (same loader, different policy in build_policy).
        model, uni_cfg = gip.load_lewam_unified_model(cfg.policy, which=cfg.get("seq_which", "best"))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        model._unified_cfg = uni_cfg
        adim = int(uni_cfg["action_dim"])
        policy = gip.build_policy(cfg, model, adim, process, transform, goal_offsets=goal_offsets)
    else:
        model, adim = gip.load_gip_model(cfg.policy, epoch=cfg.get("ckpt_epoch", None))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        # multi-task ckpt: select this task's conditioning vector + set the env<->trained action pad boundary
        if getattr(model, "task_proj", None) is not None:
            d_raw = int(dataset.get_dim("action"))
            f = int(cfg.plan_config.action_block)
            # registry name from the eval dataset name (e.g. "pusht" in "pusht_expert_train")
            dn = cfg.eval.dataset_name
            model.eval_task = next((t for t in model.mt_task_names if t in dn), dn)
            model.eval_action_pad = (d_raw, f)
            adim = f * d_raw
            print(f"[GIP] multi-task: eval_task={model.eval_task} of {model.mt_task_names}  "
                  f"action pad {d_raw}x{f} -> trained block")
        policy = gip.build_policy(cfg, model, adim, process, transform, goal_offsets=goal_offsets,
                                  policy_kwargs=_policy_kwargs(cfg, dataset, episodes, starts))
    print(f"[GIP] eval mode={mode} policy={type(policy).__name__}")

    # random-goal eval: a goal-conditioned policy fed an off-distribution goal can extrapolate to
    # out-of-box actions, which strict-checker envs (tworoom Box[-1,1]) reject -> crash (pusht clips, so
    # it was fine). Clamp executed actions to the env's action box so the eval runs; on-path actions are
    # already in-box, so this is a no-op there. Only when random_goal is on.
    if bool(cfg.get("gip_eval", {}).get("random_goal", False)):
        import numpy as _np2
        _albox = world.envs.envs[0].unwrapped.action_space
        _clo, _chi = _np2.asarray(_albox.low, _np2.float32), _np2.asarray(_albox.high, _np2.float32)
        _orig_ga = policy.get_action
        policy.get_action = lambda *a, **k: _np2.clip(
            _np2.asarray(_orig_ga(*a, **k), _np2.float32), _clo, _chi)

    results_path = Path(swm.data.utils.get_cache_dir(), "gip_eval", mode, cfg.policy)
    results_path.mkdir(parents=True, exist_ok=True)

    world.set_policy(policy)
    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=starts,
        goal_offset=(np.asarray(goal_offsets) if goal_offsets is not None
                     else cfg.eval.goal_offset_steps),
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=episodes,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    dt = time.time() - t0
    if no_early_stop:
        # library latch never fires (terminated stripped) -> read both SRs from the wrapper latches.
        evers, finals = [], []
        for _e in world.envs.envs:
            _w = _e
            while not isinstance(_w, NoEarlyStop):
                _w = _w.env
            evers.append(bool(_w.ever))
            finals.append(bool(_w.last))
        metrics["success_rate_ever"] = float(np.mean(evers) * 100.0)
        metrics["success_rate_final"] = float(np.mean(finals) * 100.0)
        metrics["success_rate"] = metrics["success_rate_final"]
        print(f"[GIP] SR ever-reached={metrics['success_rate_ever']:.1f}  "
              f"final-state={metrics['success_rate_final']:.1f}  (n={len(evers)})")
        # final-step distance decomposition. Final state from harness infos; the GOAL from the dataset
        # rows directly (episode, start+goal_offset -- same source _extract_init_goal reads), because
        # infos['goal_state'] carries NaN agent dims through the wrapper stack (debug print kept).
        try:
            s_raw = np.asarray(world.infos["state"], dtype=float)
            print(f"[GIP] debug infos state shape={s_raw.shape} row0={np.round(s_raw.reshape(s_raw.shape[0], -1)[0], 1)}")
            s_fin = s_raw.reshape(s_raw.shape[0], -1)[:, -7:]
            g_inf = np.asarray(world.infos["goal_state"], dtype=float)
            print(f"[GIP] debug infos goal_state row0 = {np.round(g_inf.reshape(g_inf.shape[0], -1)[0, -7:], 1)}")
            ep_col = np.asarray(dataset.get_col_data(gip.episode_col(dataset))).reshape(-1)
            st_col = np.asarray(dataset.get_col_data("state"))
            st_col = st_col.reshape(st_col.shape[0], -1)
            u_eps, u_seg = np.unique(ep_col, return_index=True)
            row0 = dict(zip(u_eps.tolist(), u_seg.tolist()))
            g_fin = np.stack([st_col[row0[int(e)] + int(t) + int(cfg.eval.goal_offset_steps)]
                              for e, t in zip(episodes, starts)])[:, -7:]
            for nm, sl in [("agent", slice(0, 2)), ("block", slice(2, 4)), ("pool4", slice(0, 4))]:
                v = np.linalg.norm(g_fin[:, sl] - s_fin[:, sl], axis=1)
                fin = np.isfinite(v)
                if fin.any():
                    print(f"[GIP] final-step {nm} dist px p10/p50/p90 = "
                          f"{np.percentile(v[fin],10):.1f}/{np.percentile(v[fin],50):.1f}/"
                          f"{np.percentile(v[fin],90):.1f}  (finite {int(fin.sum())}/{len(v)})")
            reach, nans, gaps = [], 0, []
            for _e in world.envs.envs:
                _w = _e
                while not isinstance(_w, NoEarlyStop):
                    _w = _w.env
                reach.append(_w.reach_at)
                if _w.nan_at is not None:
                    nans += 1
                    if _w.reach_at is not None:
                        gaps.append(_w.nan_at - _w.reach_at)
            print(f"[GIP] NaN-corruption: {nans}/{world.num_envs} envs; "
                  f"nan_at - reach_at steps = {sorted(gaps)[:12]}")
        except Exception as _ex:
            print(f"[GIP] final-step decomposition unavailable: {_ex}")
    print(f"==== GIP {mode} RESULTS ====")
    print(metrics)

    # Dump this shard's successes; a merge step pools them into the global SR.
    if shard_count > 1:
        import json as _json
        succ = np.asarray(metrics.get("episode_successes", [])).astype(bool).tolist()
        shard_file = results_path / f"{mode}_{cfg.policy}_shard{shard_idx}of{shard_count}.json"
        with shard_file.open("w") as f:
            _json.dump({"shard_idx": shard_idx, "shard_count": shard_count,
                        "episode_successes": succ, "n": len(succ),
                        "n_success": int(sum(succ))}, f)
        print(f"[shard] wrote {shard_file} ({int(sum(succ))}/{len(succ)})", flush=True)

    # trajectory-divergence probe: dump the executed per-obs-step latent trajectory if requested
    _dump = cfg.get("gip_eval", {}).get("dump_latents", "")
    if _dump and getattr(policy, "log_latents", False):
        policy.dump_latents(_dump)
        print(f"[divprobe] dumped executed latents -> {_dump}")
    _gd = cfg.get("gip_eval", {}).get("grad_diag", "")
    if _gd and hasattr(policy, "dump_diag"):
        policy.dump_diag(_gd)

    with (results_path / f"{mode}_{cfg.policy}_results.txt").open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"\n==== RESULTS ({mode}) ====\nmetrics: {metrics}\ntime: {dt:.1f}s\n")
    print("wrote", results_path)


if __name__ == "__main__":
    run()
