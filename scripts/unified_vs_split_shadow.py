"""Shadow diagnostic: does the UNIFIED policy CHOOSE THE SAME ACTIONS as the working
split policy, on the split's own (goal-reaching) trajectories -- and does the agreement
HOLD near the goal (the precise endgame the split nails and the seq lost)?

Same idea as scripts/seq_vs_split_shadow.py, adapted to the unified model. We drive the env
with the WORKING split policy (LeWAMSplitPolicy) and, at every replan, ALSO query the unified
model's reactive action on the SAME state, WITHOUT letting the unified touch the env. The
unified needs a temporal WINDOW, so this shadow maintains its own per-env deque of the last
`window` frames along the split's trajectory (left-padded exactly like training/eval) and reads
the unified action off c_t = z_t + Aggr([z_{t-k..t}]).

Interpretation (near-vs-far contrast is the robust, calibration-free signal):
  * unified ~= split on the split's states, AND agreement holds/improves near-goal
    -> the unified preserves the split's endgame-precise action map. Green light for a full SR sweep.
  * agreement COLLAPSES near-goal (as the seq's did: r2 0.52 far -> 0.24 near)
    -> the temporal aggregator smoothed the endgame; rethink before spending a sweep.

Both models z-score actions with the SAME dataset normalizer, so raw action blocks are directly
comparable.

Usage (reacher):
  python scripts/unified_vs_split_shadow.py --config-name reacher \
      policy=reacher_lewam_gc +uni_run=reacher_lewam_unified eval.num_eval=20 seed=42
"""
import time
from collections import deque
from pathlib import Path

import stable_worldmodel.data.formats.hdf5  # HDF5 self-registers on import  # noqa: F401
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
import lewam.models.gip as gip


class ShadowUnifiedPolicy(gip.LeWAMSplitPolicy):
    """Drives the env with the split policy (parent logic) but also records, per replan, the
    unified model's action on the same (window, goal, horizon). No env effect."""

    def __init__(self, uni_model, uni_cfg, *a, **kw):
        super().__init__(*a, **kw)
        self.uni = uni_model.eval()
        self.uni_cfg = uni_cfg
        self.window = int(uni_cfg["window"])
        dev = next(uni_model.parameters()).device
        self._uni_amean = torch.tensor(uni_cfg["action_mean"], dtype=torch.float32, device=dev)
        self._uni_astd = torch.tensor(uni_cfg["action_std"], dtype=torch.float32, device=dev).clamp_min(1e-6)
        self._uni_fs = int(uni_cfg["frameskip"])
        self._uni_raw = int(uni_cfg["action_raw_dim"])
        self._frame_buf = None
        self.log_split, self.log_uni, self.log_h = [], [], []

    def set_env(self, env):
        super().set_env(env)
        n = getattr(env, "num_envs", 1)
        self._frame_buf = [deque(maxlen=self.window) for _ in range(n)]

    def _build_window(self, i):
        buf = list(self._frame_buf[i])
        pad = [buf[0]] * (self.window - len(buf))
        return torch.stack(pad + buf, dim=0)  # (W, C, H, W)

    @torch.no_grad()
    def _uni_block(self, replan, gpx, h_norm, dev):
        """Unified action (RAW) for the replanning envs, off the maintained frame windows."""
        windows = torch.stack([self._build_window(i) for i in replan], dim=0)  # (R,W,C,H,W)
        R, Wn = windows.shape[0], windows.shape[1]
        z_win = self.uni.encode(windows.reshape(R * Wn, *windows.shape[2:]).to(dev).float()).reshape(R, Wn, -1)
        z_g = self.uni.encode(gpx.to(dev).float())
        blk = self.uni.gc_action(z_win, z_g, h_norm.to(dev))      # (R, block_dim) z-scored
        raw = blk.reshape(blk.size(0), self._uni_fs, self._uni_raw) * self._uni_astd + self._uni_amean
        return raw.reshape(blk.size(0), -1)

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        dev = next(self.model.parameters()).device
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(n)]
            self._steps_left = np.full(n, self.horizon0, dtype=np.float64)
        if self._frame_buf is None:
            self._frame_buf = [deque(maxlen=self.window) for _ in range(n)]

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._action_buffer[i].clear()
                    self._frame_buf[i].clear()
                    self._steps_left[i] = self.horizon0

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        replan = [i for i in range(n) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            px = info_dict["pixels"][replan]
            assert "goal" in info_dict, "shadow eval needs info_dict['goal']"
            gpx = info_dict["goal"][replan]
            gpx = gpx[:, -1] if gpx.ndim == 5 else gpx      # (R,C,H,W)
            cpx = px[:, -1] if px.ndim == 5 else px         # (R,C,H,W)
            for row, i in enumerate(replan):
                self._frame_buf[i].append(cpx[row])
            z_t = self.model.encode(cpx.to(dev).float())
            z_g = self.model.encode(gpx.to(dev).float())
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=dev, dtype=torch.float32)
            z_blk = self.model.gc_head(z_t, z_g, h_norm)     # split z-scored block
            raw_split = (z_blk.reshape(len(replan), self.frameskip, self.raw_adim)
                         * self._astd + self._amean).reshape(len(replan), -1)
            # SHADOW: unified action on the SAME window/goal/horizon (no env effect)
            raw_uni = self._uni_block(replan, gpx, h_norm, dev)
            self.log_split.append(raw_split.cpu().numpy())
            self.log_uni.append(raw_uni.cpu().numpy())
            self.log_h.append(h_norm.cpu().numpy())
            raw_exec = raw_split.reshape(len(replan), self.action_block, self.action_dim).cpu()
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(raw_exec[row])
                self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)

        action = torch.full((n, self.action_dim), float("nan"))
        for i in range(n):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()


def _agree(split, uni, h):
    """Agreement stats between two (N, block_dim) raw-action arrays."""
    split = split.reshape(split.shape[0], -1); uni = uni.reshape(uni.shape[0], -1)
    diff = uni - split
    mse = float((diff ** 2).mean())
    sn = np.linalg.norm(split, axis=1); qn = np.linalg.norm(uni, axis=1)
    denom = np.clip(sn * qn, 1e-8, None)
    cos = float(((split * uni).sum(1) / denom).mean())
    rel = float(np.sqrt((diff ** 2).mean()) / (np.sqrt((split ** 2).mean()) + 1e-8))
    var = ((split - split.mean(0)) ** 2).mean()
    r2 = float(1.0 - (diff ** 2).mean() / (var + 1e-8))
    return dict(n=int(split.shape[0]), action_mse=mse, cosine=cos, rel_rmse=rel, r2=r2,
                split_rms=float(np.sqrt((split ** 2).mean())), uni_rms=float(np.sqrt((uni ** 2).mean())))


@hydra.main(version_base=None, config_path="../configs/eval", config_name="reacher")
def run(cfg: DictConfig):
    uni_run = cfg.get("uni_run")
    assert uni_run, "pass +uni_run=<unified_run_name>"
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": gip.img_transform(cfg), "goal": gip.img_transform(cfg)}
    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    process = gip.build_process(cfg, dataset)
    episodes, starts = gip.sample_eval_episodes(cfg, dataset)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    split_model, split_cfg = gip.load_lewam_split_model(cfg.policy)
    split_model = split_model.to(dev).eval(); split_model.requires_grad_(False)
    uni_model, uni_cfg = gip.load_lewam_unified_model(uni_run)
    uni_model = uni_model.to(dev).eval(); uni_model.requires_grad_(False)

    action_block = int(cfg.plan_config.action_block)
    horizon0 = float(cfg.eval.goal_offset_steps) / float(action_block)
    policy = ShadowUnifiedPolicy(
        uni_model, uni_cfg,
        model=split_model, cfg=split_cfg, action_block=action_block,
        action_dim=int(split_cfg["action_dim"]) // action_block, horizon0=horizon0,
        H_max=int(split_cfg.get("H_max", 50)), process=process, transform=transform,
    )
    print(f"[shadow] split={cfg.policy} uni={uni_run} | driving env with SPLIT, shadowing UNIFIED "
          f"(window={uni_cfg['window']})")

    world.set_policy(policy)
    t0 = time.time()
    metrics = world.evaluate(dataset=dataset, start_steps=starts,
                             goal_offset=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget,
                             episodes_idx=episodes,
                             callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
                             video=None)
    split = np.concatenate(policy.log_split, 0); uni = np.concatenate(policy.log_uni, 0)
    h = np.concatenate(policy.log_h, 0)
    print(f"==== SHADOW unified-vs-split (split SR this run: {metrics}) | {time.time()-t0:.0f}s ====")
    print("ALL      :", _agree(split, uni, h))
    # near-goal (small remaining horizon) vs far -- reacher success hinges on the precise endgame
    near = h <= np.quantile(h, 0.33); far = h >= np.quantile(h, 0.67)
    print("NEAR-goal:", _agree(split[near], uni[near], h[near]))
    print("FAR-goal :", _agree(split[far], uni[far], h[far]))


if __name__ == "__main__":
    run()
