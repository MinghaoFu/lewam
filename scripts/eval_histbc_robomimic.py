"""History-aware BC eval for RoboMimic (ADDITIVE -- touches no existing file).

The stock gip.BCPolicy runs the intention head on a SINGLE frame
(intention_rollout horizon=1, zero past-action conditioning), so it predicts
only the position-0 action with no memory. On Lift that makes the arm reach the
cube but never lift it (a reactive policy can't tell "I've grasped -> now lift").

This policy instead feeds the autoregressive head its proper training context:
the last `history_size` frame latents + the last `history_size-1` executed
(normalized) action blocks, shifted, and takes predict_intention's LAST
position. That matches how the head was trained (ctx_len = history_size), so the
grasp->lift transition is in-distribution.

Modes (cfg.gip_eval.mode, default "policy"):
  - policy   : OURS -- history-conditioned forward GC policy. With
               +gip_eval.goal_conditioned=true it encodes info_dict["goal"] ->
               z_goal and threads it (and the AdaLN-Zero horizon) into the head.
               goal_conditioned=false uses the goal-agnostic history-BC.
  - gcidm    : GC-IDM (Markovian) -- single-frame planning-free IDM. Each replan
               does ONE forward pass GCIDMHead(z_t, z_goal, h) -> action block.
               NO history, NO CEM. Faithful reproduction of `gcidm` (2605.08732),
               ported from gip.GCIDMPolicy to the robomimic histbc env loop.

Run (OURS goal-conditioned):
  python eval_histbc_robomimic.py --config-name robomimic policy=lift_gc_ours \
      world.task=Lift dataset.stats=lift eval.dataset_name=lift \
      eval.num_eval=50 eval.eval_budget=100 eval.goal_offset_steps=30 \
      +gip_eval.mode=policy +gip_eval.goal_conditioned=true

Run (GC-IDM Markovian):
  python eval_histbc_robomimic.py --config-name robomimic policy=lift_gcidm \
      world.task=Lift dataset.stats=lift eval.dataset_name=lift \
      eval.num_eval=50 eval.eval_budget=100 eval.goal_offset_steps=30 \
      +gip_eval.mode=gcidm
"""
import lewam.envs.robomimic_env as robomimic_env  # noqa: F401  -- registers swm/RoboMimic-v0 + robomimic ObsUtils
import stable_worldmodel.data.formats.hdf5  # noqa: F401  -- HDF5 self-registers
import os

os.environ["MUJOCO_GL"] = "egl"

import time
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.policy import BasePolicy

import lewam.models.gip as gip


class HistoryBCPolicy(BasePolicy):
    """Autoregressive BC with the real frame + executed-action history.

    goal_conditioned=False uses the goal-agnostic history-BC.
    goal_conditioned=True encodes info_dict["goal"] -> z_goal, threading it (and
    the AdaLN-Zero horizon, if the model carries a horizon_modulator) into the head
    via predict_intention(goal_emb=..., horizon=...). The history wedge is unchanged;
    the goal/horizon are optional additive hooks (a strict superset).
    """

    def __init__(self, model, action_block, action_dim, history_size=3,
                 process=None, transform=None, goal_conditioned=False,
                 H_max=50, horizon0=None, **kw):
        super().__init__(**kw)
        self.goal_conditioned = bool(goal_conditioned)
        self.type = "goal_cond_history_bc" if goal_conditioned else "history_bc"
        self.model = model.eval()
        self.action_block = int(action_block)
        self.action_dim = int(action_dim)
        self.HS = int(history_size)
        self.process = process or {}
        self.transform = transform or {}
        self.device = next(model.parameters()).device
        # AdaLN-Zero horizon countdown; active only if the model carries a horizon_modulator (else identity)
        self.use_horizon = getattr(self.model, "horizon_modulator", None) is not None
        self.H_max = int(H_max)
        self.horizon0 = float(horizon0) if horizon0 is not None else None
        self._steps_left = None

    def set_env(self, env):
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._abuf = [deque() for _ in range(n)]                 # raw actions to execute
        self._zhist = [deque(maxlen=self.HS) for _ in range(n)]  # frame latents (D,)
        self._ablk = [deque(maxlen=self.HS - 1) for _ in range(n)]  # executed blocks (Adim,)
        self._phist = [deque(maxlen=self.HS) for _ in range(n)]  # proprio latents (D,)
        if self.use_horizon and self.horizon0 is not None:
            self._steps_left = np.full(n, self.horizon0, dtype=np.float64)

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._abuf[i].clear()
                    self._zhist[i].clear()
                    self._ablk[i].clear()
                    self._phist[i].clear()
                    if self._steps_left is not None:
                        self._steps_left[i] = self.horizon0  # reset countdown for the new episode
        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        px = info_dict["pixels"]
        if not torch.is_tensor(px):
            px = torch.as_tensor(px)
        cur = px[:, -1] if px.ndim == 5 else px   # (n, C, H, W) -- most recent frame
        cur = cur.to(self.device).float()

        # goal frame (already transformed by _prepare_info; only used when goal_conditioned)
        goal_px = None
        if self.goal_conditioned and "goal" in info_dict:
            gp = info_dict["goal"]
            if not torch.is_tensor(gp):
                gp = torch.as_tensor(gp)
            goal_px = (gp[:, -1] if gp.ndim == 5 else gp).to(self.device).float()  # (n, C, H, W)

        for i in range(n):
            if dead[i] or len(self._abuf[i]) > 0:
                continue
            z = self.model.encode({"pixels": cur[i:i + 1].unsqueeze(1)})["emb"][0, 0]  # (D,)
            self._zhist[i].append(z)
            emb = torch.stack(list(self._zhist[i]))[None]    # (1, t, D)
            t, D = emb.size(1), emb.size(-1)
            past = torch.zeros(1, t, D, device=self.device)  # a_<p, leading zero = context
            for j, blk in enumerate(self._ablk[i]):          # block j executed at window frame j
                if j + 1 < t:
                    past[0, j + 1] = self.model.action_encoder(
                        blk.view(1, 1, -1).to(self.device))[0, 0]
            proprio_emb = None
            if getattr(self.model, "use_proprio", False) and "proprio" in info_dict:
                pr = info_dict["proprio"]
                if not torch.is_tensor(pr):
                    pr = torch.as_tensor(pr)
                pr_cur = (pr[:, -1] if pr.ndim == 3 else pr)[i:i + 1].to(self.device).float()
                # multi-task ckpt: pad proprio to p_max dims (post-normalize, as in training); single-task no-op
                ped = self.model.proprio_encoder.patch_embed.weight.shape[1]
                if pr_cur.shape[-1] < ped:
                    pr_cur = torch.nn.functional.pad(pr_cur, (0, ped - pr_cur.shape[-1]))
                pz = self.model.proprio_encoder(pr_cur.unsqueeze(1))[0, 0]  # (D,)
                self._phist[i].append(pz)
                proprio_emb = torch.stack(list(self._phist[i]))[None]  # (1, t, D)
            tv = self.model._eval_task_vec(1, self.device)  # (1, D) or None (multi-task only)
            # encode this env's goal frame -> z_goal
            goal_emb = None
            if goal_px is not None:
                goal_emb = self.model.encode({"pixels": goal_px[i:i + 1].unsqueeze(1)})["emb"][:, 0]  # (1, D)
            # remaining-horizon (normalized) for the AdaLN-Zero hook: per-env obs-step countdown,
            # init = goal_offset/frameskip, -1 per replan. Only when use_horizon, else None -> identity.
            horizon = None
            if self.use_horizon and self._steps_left is not None:
                steps = max(self._steps_left[i], 1.0)
                horizon = torch.tensor([min(steps, self.H_max) / self.H_max],
                                       device=self.device, dtype=torch.float32)  # (1,)
            _, acts = self.model.predict_intention(emb, past, proprio_emb=proprio_emb, task_vec=tv,
                                                   goal_emb=goal_emb, horizon=horizon)  # (1, t, Adim)
            blk_new = acts[0, -1].detach().cpu()               # (Adim,) -- last position
            self._ablk[i].append(blk_new)
            raw = blk_new.view(self.action_block, self.action_dim)  # (block, adim)
            self._abuf[i].extend(raw)
            if self._steps_left is not None:
                self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)  # one obs-step consumed

        action = torch.full((n, self.action_dim), float("nan"))
        for i in range(n):
            if not dead[i]:
                action[i] = self._abuf[i].popleft()
        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


# GC-IDM (Markovian, planning-free) for the robomimic histbc env loop.
# Faithful repro of `gcidm` (2605.08732), ported from gip.GCIDMPolicy; single frame, no history.
class GCIDMRobomimicPolicy(BasePolicy):
    """Planning-free goal-conditioned IDM in the robomimic histbc env loop.

    Each replan (per live env i):
      z_t   = frozen-LeWM encode(current pixels)['emb'][:,0]
      z_goal= frozen-LeWM encode(info_dict['goal'])['emb'][:,0]
      h     = remaining horizon in OBS-steps (per-env counter, init horizon0 =
              goal_offset/frameskip, decremented one obs-step per replan, clamped
              >=1, normalized min(h,H_max)/H_max).
      a     = GCIDMHead(z_t, z_goal, h_norm) -> ONE raw-action block.
    NO CEM, NO WM rollout, NO history. Action un-normalization mirrors
    HistoryBCPolicy (inverse_transform via the same StandardScaler).
    """

    def __init__(self, lewm, head, action_block, action_dim, H_max, horizon0,
                 process=None, transform=None, ablate_horizon=False, **kw):
        super().__init__(**kw)
        self.type = "gcidm_policy"
        self.lewm = lewm.eval()
        self.head = head.eval()
        self.process = process or {}
        self.transform = transform or {}
        self.action_block = int(action_block)
        self.action_dim = int(action_dim)
        self.H_max = int(H_max)
        self.horizon0 = float(horizon0)
        self.ablate_horizon = bool(ablate_horizon)
        self.device = next(head.parameters()).device

    def set_env(self, env):
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._abuf = [deque() for _ in range(n)]
        self._steps_left = np.full(n, self.horizon0, dtype=np.float64)

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._abuf[i].clear()
                    self._steps_left[i] = self.horizon0  # reset countdown for the new episode
        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        assert "goal" in info_dict, "GCIDM policy needs info_dict['goal'] (goal-reaching eval)"
        px = info_dict["pixels"]
        gp = info_dict["goal"]
        if not torch.is_tensor(px):
            px = torch.as_tensor(px)
        if not torch.is_tensor(gp):
            gp = torch.as_tensor(gp)
        cur = (px[:, -1] if px.ndim == 5 else px).to(self.device).float()   # (n, C, H, W)
        goal = (gp[:, -1] if gp.ndim == 5 else gp).to(self.device).float()  # (n, C, H, W)

        for i in range(n):
            if dead[i] or len(self._abuf[i]) > 0:
                continue
            z_t = self.lewm.encode({"pixels": cur[i:i + 1].unsqueeze(1)})["emb"][:, 0]   # (1, D)
            z_g = self.lewm.encode({"pixels": goal[i:i + 1].unsqueeze(1)})["emb"][:, 0]  # (1, D)
            steps = max(self._steps_left[i], 1.0)
            h_norm = torch.tensor([min(steps, self.H_max) / self.H_max],
                                  device=self.device, dtype=torch.float32)  # (1,)
            if self.ablate_horizon:
                h_norm = torch.zeros_like(h_norm)
            blk = self.head(z_t, z_g, h_norm)[0].detach().cpu()  # (Adim,)
            raw = blk.view(self.action_block, self.action_dim)   # (block, adim)
            self._abuf[i].extend(raw)
            self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)  # one obs-step consumed

        action = torch.full((n, self.action_dim), float("nan"))
        for i in range(n):
            if not dead[i]:
                action[i] = self._abuf[i].popleft()
        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


def _eval_loop(world, policy, dataset, episodes, starts, cfg, tag):
    """Shared chunked eval loop (one env per episode; same-box render)."""
    world.set_policy(policy)
    t0 = time.time()
    _ne = world.num_envs
    _cbl = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    _succ = []
    for _c in range(0, len(episodes), _ne):
        _ep = list(episodes[_c:_c + _ne]); _st = list(starts[_c:_c + _ne]); _k = len(_ep)
        while len(_ep) < _ne: _ep.append(_ep[0]); _st.append(_st[0])
        # CHUNK-RESET: eval runs mode='wait' (never sends _needs_flush), so a reused policy carries
        # chunk N-1's per-env deques/counters -> corrupted history -> 0 success after chunk 1. Re-init
        # per-env state each chunk via set_env(world.envs) (idempotent; must pass the vec env
        # world.set_policy uses so self.env matches the rollout).
        policy.set_env(world.envs)
        _gm = str(cfg.eval.get("goal_mode", "mid"))
        _m = world.evaluate(dataset=dataset, start_steps=_st, goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget, episodes_idx=_ep, callables=_cbl, video=None,
            goal_mode=_gm)
        _succ.extend(list(_m["episode_successes"])[:_k])
        print(f"[{tag}] chunk {_c // _ne + 1}: cumulative {int(sum(_succ))}/{len(_succ)}", flush=True)
    metrics = {"success_rate": float(np.mean(_succ)), "episode_successes": np.array(_succ),
               "n": len(_succ), "time": time.time() - t0}
    return metrics


@hydra.main(version_base=None, config_path="../configs/eval", config_name="robomimic")
def run(cfg: DictConfig):
    assert cfg.policy != "random", "set policy=<run_name>"
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": gip.img_transform(cfg), "goal": gip.img_transform(cfg)}
    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    process = gip.build_process(cfg, dataset)
    episodes, starts = gip.sample_eval_episodes(cfg, dataset)

    mode = str(cfg.get("gip_eval", {}).get("mode", "policy"))
    action_block = int(cfg.plan_config.action_block)

    # GC-IDM (Markovian)
    if mode == "gcidm":
        gc = cfg.gip_eval
        lewm, head, gcfg = gip.load_gcidm_model(gc.get("gcidm_run", cfg.policy))
        lewm = lewm.to("cuda").eval(); head = head.to("cuda").eval()
        lewm.requires_grad_(False); head.requires_grad_(False)
        lewm.interpolate_pos_encoding = True
        horizon0 = gc.get("gcidm_horizon", None)
        if horizon0 is None:
            horizon0 = float(cfg.eval.goal_offset_steps) / float(action_block)
        policy = GCIDMRobomimicPolicy(
            lewm=lewm, head=head, action_block=action_block,
            action_dim=int(gcfg["action_dim"]) // action_block,
            H_max=int(gcfg["H_max"]), horizon0=float(horizon0),
            process=process, transform=transform,
            ablate_horizon=bool(gc.get("ablate_horizon", gcfg.get("ablate_horizon", False))),
        )
        print(f"[GCIDM] policy ready action_block={action_block} adim={gcfg['action_dim']} "
              f"H_max={gcfg['H_max']} horizon0={horizon0:.2f} ablate_h={policy.ablate_horizon}")
        metrics = _eval_loop(world, policy, dataset, episodes, starts, cfg, "GCIDM")
        sub = "gcidm"
    # OURS (history GC, mode=policy)
    else:
        model, adim = gip.load_gip_model(cfg.policy, epoch=cfg.get("ckpt_epoch", None))
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        # multi-task ckpt: select this task's conditioning vector
        model.eval_task = cfg.eval.dataset_name if getattr(model, "task_proj", None) is not None else None
        if model.eval_task is not None:
            print(f"[HISTBC] multi-task conditioning: eval_task={model.eval_task} of {model.mt_task_names}")

        ab = int(cfg.plan_config.action_block)
        _gc = bool(cfg.get("gip_eval", {}).get("goal_conditioned", False))
        ge = cfg.get("gip_eval", {})
        horizon0 = ge.get("horizon0", None)
        if horizon0 is None:
            horizon0 = float(cfg.eval.goal_offset_steps) / float(ab)
        policy = HistoryBCPolicy(
            model=model, action_block=ab, action_dim=adim // ab,
            history_size=cfg.get("history_size", 3), process=process, transform=transform,
            goal_conditioned=_gc, H_max=int(ge.get("horizon_H_max", 50)), horizon0=float(horizon0),
        )
        print(f"[HISTBC] policy ready adim={adim} action_block={ab} HS={policy.HS} "
              f"goal_cond={_gc} use_horizon={policy.use_horizon} horizon0={horizon0:.2f}")
        metrics = _eval_loop(world, policy, dataset, episodes, starts, cfg, "HISTBC")
        sub = "histbc"

    results_path = Path(swm.data.utils.get_cache_dir(), "gip_eval", sub, cfg.policy)
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"==== {sub.upper()} RESULTS ====")
    print(metrics)
    with (results_path / f"{sub}_results.txt").open("a") as f:
        f.write(f"\nmode={mode} metrics: {metrics}\n")
    print("wrote", results_path)


if __name__ == "__main__":
    run()
