"""GIP step-1 success-rate eval.

Reuses eval.py's env/dataset machinery; the policy is chosen by gip_eval.mode:

  unified_policy        LeWAM-Unified reactive: the goal-conditioned action head run directly (the
                        representation channel; never touches the dynamics)
  unified_cem           LeWAM-Unified CEM: the dynamics head rolls candidate action blocks to the goal
  split_policy          LeWAM-Split reactive action head
  seq_policy / seq_cem  LeWAM-Seq reactive / CEM
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

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm

import lewam.models.gip as gip


@hydra.main(version_base=None, config_path="../configs/eval", config_name="pusht")
def run(cfg: DictConfig):
    mode = cfg.get("gip_eval", {}).get("mode", "bc")
    assert cfg.policy != "random", "set policy=<gip_run_name>"
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "horizon*action_block must be <= eval_budget"

    # -- env + data context (shared helpers)
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": gip.img_transform(cfg), "goal": gip.img_transform(cfg)}
    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    process = gip.build_process(cfg, dataset)
    episodes, starts = gip.sample_eval_episodes(cfg, dataset)

    # -- optional random goals (reachable by construction): from each real init state, roll the sim
    # forward goal_rollout_steps env-steps of random actions and take the resulting state as the goal.
    # This is off the expert's path (random != expert actions) so it isn't BC, yet it's reachable (a
    # random policy just reached it) so a competent planner can too -- fixing the "fully-random frame is
    # unreachable" confound. Precompute goals via a throwaway sim rollout (world.evaluate resets
    # afterward), then monkey-patch swm's module-global _extract_init_goal to serve them. Off by default.
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
                print(f"[GIP] rollout act: mean={_am} std={_as} env_box=[{_lo},{_hi}]", flush=True); _dbg[0] = False
            infos = None
            for _ in range(_H):                                      # policy-space random actions
                acts = (_rng.standard_normal((n, len(_as))).astype(_np.float32) * _as + _am)
                acts = _np.clip(acts, _lo, _hi)
                _, _, _, _, infos = world.envs.step(acts.reshape(*world.envs.action_space.shape))
            if _dbg[0]:
                print(f"[GIP] random-goal rollout infos keys: {sorted(infos.keys())}; "
                      f"dataset cols: {sorted(map(str,_cols))}", flush=True); _dbg[0] = False
            # emit all goal_* columns from the post-rollout env info, so each task's goal callable
            # (goal_state / goal_qpos / goal_privileged_* / goal_proprio) finds what it needs.
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
            # DIAG (trivial-goal check): how far did agent/block actually move over the random rollout?
            # pusht state = [agent_x, agent_y, block_x, block_y, block_angle]. A goal where the block
            # barely moved reduces to agent-navigation (easier) -> would skew SR up.
            _gs, _is = goal_state.get("goal_state"), init_state.get("state")
            if _gs is not None and _is is not None:
                _isb = _np.asarray(_is); _isb = _isb[:, -1] if _isb.ndim > 2 else _isb
                _gsb = _np.asarray(_gs)
                _blk = _np.linalg.norm(_gsb[:, 2:4] - _isb[:, 2:4], axis=-1)
                _agt = _np.linalg.norm(_gsb[:, 0:2] - _isb[:, 0:2], axis=-1)
                _ang = _np.abs(_gsb[:, 4] - _isb[:, 4]); _ang = _np.minimum(_ang, 2 * _np.pi - _ang)
                print(f"[DIAG] H={_H} n={n}  BLOCK-move mean={_blk.mean():.1f} med={_np.median(_blk):.1f} "
                      f"frac<5px={( _blk<5).mean():.2f} frac>20px={(_blk>20).mean():.2f}  "
                      f"ANGLE-move med={_np.median(_ang):.3f}rad(<pi/9={( _ang<_np.pi/9).mean():.2f})  "
                      f"AGENT-move med={_np.median(_agt):.1f}", flush=True)
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
        policy = gip.build_policy(cfg, None, None, process, transform)
    elif mode in ("seq_policy", "seq_cem"):
        # LeWAM-Seq: its own loader + config; adim = the model's z-scored action block dim.
        model, seq_cfg = gip.load_lewam_seq_model(cfg.policy, which=cfg.get("seq_which", "best"))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        model._seq_cfg = seq_cfg
        adim = int(seq_cfg["action_dim"])
        policy = gip.build_policy(cfg, model, adim, process, transform)
    elif mode == "split_policy":
        # LeWAM-Split: its own loader + config; adim = the model's z-scored action block dim.
        model, split_cfg = gip.load_lewam_split_model(cfg.policy, which=cfg.get("seq_which", "best"))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        model._split_cfg = split_cfg
        adim = int(split_cfg["action_dim"])
        policy = gip.build_policy(cfg, model, adim, process, transform)
    elif mode in ("unified_policy", "unified_cem"):
        # LeWAM-Unified: its own loader + config; adim = the model's z-scored action block dim.
        # unified_cem = CEM planner over the dynamics head (same loader, different policy in build_policy).
        model, uni_cfg = gip.load_lewam_unified_model(cfg.policy, which=cfg.get("seq_which", "best"))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        model._unified_cfg = uni_cfg
        adim = int(uni_cfg["action_dim"])
        policy = gip.build_policy(cfg, model, adim, process, transform)
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
        policy = gip.build_policy(cfg, model, adim, process, transform)
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
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=episodes,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    dt = time.time() - t0
    print(f"==== GIP {mode} RESULTS ====")
    print(metrics)

    # trajectory-divergence probe: dump the executed per-obs-step latent trajectory if requested
    _dump = cfg.get("gip_eval", {}).get("dump_latents", "")
    if _dump and getattr(policy, "log_latents", False):
        policy.dump_latents(_dump)
        print(f"[divprobe] dumped executed latents -> {_dump}")

    with (results_path / f"{mode}_{cfg.policy}_results.txt").open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write(f"\n==== RESULTS ({mode}) ====\nmetrics: {metrics}\ntime: {dt:.1f}s\n")
    print("wrote", results_path)


if __name__ == "__main__":
    run()
