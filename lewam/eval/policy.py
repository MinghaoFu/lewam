"""
Policies wrap a trained model as a stable_worldmodel policy that encodes observations,
keeps per-env action buffers and goal-horizon countdowns, and emits the next action
"""

from collections import deque

import numpy as np
import torch
from stable_worldmodel.policy import BasePolicy

from lewam import views
from lewam.models.lewam import LeWAM


def history_steps(step: int, history_len: int, stride: int) -> list[int]:
    """Episode steps of the history frames at `step`, oldest first; negative steps precede the episode."""
    return [step - k * stride for k in range(history_len - 1, -1, -1)]


class LeWAMPolicy(BasePolicy):
    """
    LeWAM policy

    Each replan:
        - Encode the current frame into the rolling latent history
        - Predict action chunk from latent history, conditioned on the goal frame and the remaining goal
          horizon when the checkpoint's goal_type is not "none"
    """

    def __init__(
        self, model: LeWAM, cfg: dict, frameskip: int, action_dim: int, h0: float | np.ndarray | None = None,
        H_max: int = 50, transform: dict | None = None, num_exec_actions: int = 0, flow_seed: int = 0
    ):
        """
        Args:
            model (LeWAM): the trained model
            cfg (dict): the checkpoint config
            frameskip (int): env steps (one action each) between observations
            action_dim (int): raw action dim of one env step
            h0 (float | np.ndarray | None): goal horizon at episode start, in frameskip units: one number, or one per
                env; None = H_max
            H_max (int): goal-horizon cap
            transform (dict | None): per-key transforms applied to the info dict
            num_exec_actions (int): actions of each predicted chunk executed before the next replan;
                0 = frameskip
            flow_seed (int): seed of the generator every flow-matching draw uses
        """
        super().__init__()
        self.type = "lewam_policy"
        self.model = model.eval()
        self.frameskip = int(frameskip)
        self.action_dim = int(action_dim)
        self.H_max = int(cfg.get("H_max", H_max))
        self.h0 = h0 if h0 is not None else float(self.H_max)
        self.transform = transform or {}
        self.process = {}    # actions are un-z-scored with the checkpoint's own stats
        self.device = next(model.parameters()).device
        self._action_mean = torch.tensor(cfg["action_mean"], dtype=torch.float32, device=self.device)
        self._action_std = torch.tensor(cfg["action_std"], dtype=torch.float32, device=self.device).clamp_min(1e-6)
        self._action_buffer = None
        self._steps_left = None 
        self._flow_seed = int(flow_seed)
        self._flow_rng = None

        self.history_len = int(cfg["history_len"])
        self.history_stride = int(cfg["history_stride"])
        self.num_actions_pred = int(cfg["num_actions_pred"])
        self.num_exec_actions = int(num_exec_actions) or self.frameskip
        assert 0 < self.num_exec_actions <= self.num_actions_pred, \
            f"cannot execute {self.num_exec_actions} actions of a {self.num_actions_pred}-action chunk"
        
        assert "goal_type" in cfg, "checkpoint config has no goal_type: migrate it (none, terminal or sampled)"
        self.goal_type = cfg["goal_type"]

        self._frames = None        # per env: {episode step: (views, C, H, W) frame} over the history window
        self._latents = None       # per env: {episode step: (views, z) latent}, the window's encoded frames
        self._step = None          # per env: steps since the episode started
        self.view_keys = views.info_keys(views.columns(cfg.get("views")))
        for key in self.view_keys[1:]:
            if "pixels" in self.transform:
                self.transform[key] = self.transform["pixels"]
        
        self.goal_keys = views.goal_info_keys(cfg.get("goal_views") or views.columns(cfg.get("views"))[:1])
        self.goal_view_indices = [int(view) for view in cfg.get("goal_view_indices", [0])]
        for key in self.goal_keys[1:]:
            if "goal" in self.transform:
                self.transform[key] = self.transform["goal"]

    def _enc(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    def _goal_latents(self, info_dict: dict, replan: list[int]) -> torch.Tensor:
        """The goal latents of the replanning envs: (R, z) for one goal view, (R, goal views, z) for several."""
        latents = []
        for key in self.goal_keys:
            assert key in info_dict, f"goal-reaching eval needs info_dict[{key!r}]"
            goal = info_dict[key][replan]
            latents.append(self._enc((goal[:, -1] if goal.ndim == 5 else goal).to(self.device).float()))
        return latents[0] if len(latents) == 1 else torch.stack(latents, dim=1)

    def set_env(self, env) -> None:
        """Allocate per-env action buffers and latent histories and reset the goal-horizon countdown."""
        self.env = env
        num_envs = getattr(env, "num_envs", 1)
        self._action_buffer = [deque() for _ in range(num_envs)]
        self._env_h0 = np.broadcast_to(np.asarray(self.h0, dtype=np.float64), (num_envs,)).copy()
        self._steps_left = self._env_h0.copy()
        self._frames = [{} for _ in range(num_envs)]
        self._latents = [{} for _ in range(num_envs)]
        self._step = np.zeros(num_envs, dtype=np.int64)

    @torch.no_grad()
    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        info_dict = self._prepare_info(info_dict)
        num_envs = self.env.num_envs
        hist_len = self.history_len
        n_views = len(self.view_keys)
        if self._action_buffer is None:
            self.set_env(self.env)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(num_envs):
                if flush[i]:
                    self._flush_env(i)

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(num_envs, dtype=bool)

        alive = [i for i in range(num_envs) if not dead[i]]
        replan = [i for i in alive if len(self._action_buffer[i]) == 0]

        if alive:
            frames = []
            for key in self.view_keys:
                curr = info_dict[key][alive]
                frames.append(curr[:, -1] if curr.ndim == 5 else curr)
            c_obs = torch.stack(frames, dim=1)                                 # (R, views, C, H, W)
            window = (hist_len - 1) * self.history_stride
            for row, i in enumerate(alive):
                step = int(self._step[i])
                self._frames[i][step] = c_obs[row]
                self._frames[i] = {s: f for s, f in self._frames[i].items() if s >= step - window}
                self._latents[i] = {s: z for s, z in self._latents[i].items() if s >= step - window}

        if replan:
            history_rows = {i: [s for s in history_steps(int(self._step[i]), hist_len, self.history_stride) if s >= 0]
                            for i in replan}
            missing_latents = [(i, s) for i in replan for s in history_rows[i] if s not in self._latents[i]]
            if missing_latents:
                pixels = torch.stack([self._frames[i][s] for i, s in missing_latents])            # (M, views, C, H, W)
                latents = self._enc(pixels.flatten(0, 1).to(self.device).float()).view(len(missing_latents), n_views, -1)
                for (i, s), z in zip(missing_latents, latents):
                    self._latents[i][s] = z

            R, D = len(replan), self.model.z_dim
            history = torch.zeros(R, n_views, hist_len, D, device=self.device)
            history_pad = torch.ones(R, hist_len, dtype=torch.bool, device=self.device)
            for row, i in enumerate(replan):
                buf = [self._latents[i][s] for s in history_rows[i]]
                history[row, :, hist_len - len(buf):] = torch.stack(buf, dim=1)
                history_pad[row, hist_len - len(buf):] = False
            if n_views == 1:
                history = history[:, 0]

            actions = self._propose(info_dict, replan, history, history_pad)       # the actions to execute
            raw = (actions * self._action_std + self._action_mean).cpu()           # (R, executed actions, action_dim)
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(raw[row])
                self._steps_left[i] = max(self._steps_left[i] - raw.shape[1] / self.frameskip, 1.0)

        action = torch.full((num_envs, self.action_dim), float("nan"))
        for i in range(num_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
                self._step[i] += 1
        return action.reshape(*self.env.action_space.shape).float().numpy()

    def _flush_env(self, i: int) -> None:
        self._action_buffer[i].clear()
        self._frames[i].clear()
        self._latents[i].clear()
        self._step[i] = 0
        self._steps_left[i] = self._env_h0[i]

    def _flow_generator(self, device: torch.device | str) -> torch.Generator:
        dev = torch.device(device)
        if self._flow_rng is None or self._flow_rng.device != dev:
            self._flow_rng = torch.Generator(device=dev)
            self._flow_rng.manual_seed(self._flow_seed)
        return self._flow_rng

    def _h_norm(self, replan: list[int]) -> torch.Tensor | None:
        """Normalized goal horizon of each replanning env for sampled goals; terminal goals have none."""
        if self.goal_type != "sampled":
            return None
        steps = np.maximum(self._steps_left[replan], 1.0)
        return torch.tensor(np.minimum(steps, self.H_max) / self.H_max, device=self.device, dtype=torch.float32)

    def _propose(self, info_dict: dict, replan: list[int], history: torch.Tensor,
                 history_pad: torch.Tensor) -> torch.Tensor:
        """The actions to execute for each replanning env; called by get_action at every replan, and
        overridden by LeWAMPlanner."""
        goal = {}
        if self.goal_type != "none":
            goal = dict(z_goal=self._goal_latents(info_dict, replan), h_norm=self._h_norm(replan))
        actions = self.model.predict_actions(history, history_pad, generator=self._flow_generator(self.device), **goal)
        return actions[:, :self.num_exec_actions]
