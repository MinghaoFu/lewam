"""shadow_compare.py — generic head-to-head SHADOW diagnostic (any model vs any model).

Drive the env with a DRIVER model (its actions are executed, so the observed states are its
own goal-reaching trajectory) and, at every replan, query a SHADOW model on the SAME
(frames, goal, remaining-horizon) WITHOUT letting it touch the env. Log both raw action blocks
and report how well the shadow reproduces the driver's action MAP — overall and split by
remaining horizon (NEAR-goal vs FAR-goal, the precise endgame).

Why this is clean: the driver's own actions produced its frames, so the state distribution is
self-consistent AND competent. This isolates a model's per-step action map from its closed-loop
covariate shift (unlike teacher-forcing GT actions into a diverged rollout, which is ill-defined).
  * shadow ~= driver on the driver's states, agreement holds NEAR-goal
      -> the shadow preserves the driver's (endgame-precise) action map.
  * agreement COLLAPSES near-goal (e.g. the old seq: r2 0.52 far -> 0.24 near)
      -> the shadow smoothed the endgame; rethink before spending an SR sweep.

Both models z-score actions with their OWN dataset normalizer, so raw (un-z-scored) action blocks
are directly comparable. Replaces scripts/{seq,unified}_vs_split_shadow.py.

Extending to a new model: add a loader in lewam/models/gip.py, a <Name>Adapter below (implement
`act`; add per-env state via `reset`/`reset_env` if it needs temporal history), and a REGISTRY
entry. Driver and shadow are arbitrary — any registered kind against any other.

Usage (reacher; drive with split, shadow the unified):
  python scripts/shadow_compare.py --config-name reacher \
      +driver=split:reacher_lewam_gc +shadow=unified:reacher_lewam_unified \
      eval.num_eval=20 seed=42
  # spec = "<kind>:<run_name>"; kinds: split, seq, unified. Any kind may be driver or shadow.
"""
import time
from collections import deque

import stable_worldmodel.data.formats.hdf5  # HDF5 self-registers on import  # noqa: F401
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.policy import BasePolicy
import lewam.models.gip as gip


# --------------------------------------------------------------------------- #
# Model adapters: wrap a loaded (model, cfg) and expose a uniform `act(...)` that returns the
# model's RAW action block on the replanning envs. An adapter that needs temporal history keeps
# per-env state and resets it on episode boundaries. `act` also returns the z-scored block so the
# policy can feed the DRIVER's block to the SHADOW as its previous-action context.
# --------------------------------------------------------------------------- #
class ModelAdapter:
    kind = "base"

    def __init__(self, model, cfg, dev):
        self.model = model.eval()
        self.cfg = cfg
        self.dev = dev
        self.run = None
        self.fs = int(cfg["frameskip"])
        self.raw = int(cfg["action_raw_dim"])
        self.block_dim = int(cfg["action_dim"])          # frameskip * raw
        self.amean = torch.tensor(cfg["action_mean"], dtype=torch.float32, device=dev)
        self.astd = torch.tensor(cfg["action_std"], dtype=torch.float32, device=dev).clamp_min(1e-6)

    # per-env temporal state (no-op for reactive models)
    def reset(self, n):
        pass

    def reset_env(self, i):
        pass

    def desc(self):
        return f"{self.kind}:{self.run}"

    def _unnorm(self, blk):
        """z-scored block (R, block_dim) -> RAW action block (R, block_dim)."""
        return (blk.reshape(blk.size(0), self.fs, self.raw) * self.astd + self.amean).reshape(blk.size(0), -1)

    @torch.no_grad()
    def act(self, replan, cpx, gpx, h_norm, ext_zblk=None):
        """cpx, gpx: (R, C, H, W) transformed current/goal frames; h_norm: (R,); ext_zblk: (R, block)
        the driver's z-scored block for shadow prev-action context (None when this adapter is the
        driver). Returns (raw (R, block), zblk (R, block))."""
        raise NotImplementedError


class SplitAdapter(ModelAdapter):
    """LeWAM-Split: reactive one-step GC off raw z_t (head reads z_t, z_goal, h_norm)."""
    kind = "split"

    @torch.no_grad()
    def act(self, replan, cpx, gpx, h_norm, ext_zblk=None):
        z_t = self.model.encode(cpx.to(self.dev).float())
        z_g = self.model.encode(gpx.to(self.dev).float())
        zblk = self.model.gc_head(z_t, z_g, h_norm.to(self.dev))
        return self._unnorm(zblk), zblk


class SeqAdapter(ModelAdapter):
    """LeWAM-Seq at history-size 1 (pure reactive map): single-token predictor -> action head."""
    kind = "seq"

    @torch.no_grad()
    def act(self, replan, cpx, gpx, h_norm, ext_zblk=None):
        dev = self.dev
        z = self.model.encode_frames(cpx.to(dev).float().unsqueeze(1))       # (R,1,D)
        empty = z.new_zeros(z.size(0), 0, z.size(-1))
        pending = self.model._tokenize_pending(z, empty)                     # (R,1,D)
        h_act = self.model.predictor(pending)[:, 0::2][:, -1]                # (R,D)
        zg = self.model.state_encoder(gpx.to(dev).float())                   # (R,D)
        zblk = self.model.action_head(h_act, zg, h_norm.to(dev))            # (R, block) z-scored
        return self._unnorm(zblk), zblk


class UnifiedAdapter(ModelAdapter):
    """LeWAM-Unified: temporal-window head off c_t = z_t + Aggr([z_{t-k..t}]). Maintains a per-env
    latent buffer along the driver's trajectory (grows per episode, cleared on reset). If the arm is
    action-conditioned, each frame's previous-action is the DRIVER's executed block (fed via ext_zblk;
    or the model's own block when it is itself the driver)."""
    kind = "unified"

    def __init__(self, model, cfg, dev):
        super().__init__(model, cfg, dev)
        self.action_cond = bool(cfg.get("agg_action_cond", False))
        self._lat = self._pblk = self._prev = None

    def reset(self, n):
        self._lat = [[] for _ in range(n)]      # z_start..z_t per env
        self._pblk = [[] for _ in range(n)]     # prev-action block that led into each frame
        self._prev = [None] * n                 # last block source for the NEXT frame

    def reset_env(self, i):
        self._lat[i].clear(); self._pblk[i].clear(); self._prev[i] = None

    def desc(self):
        return f"{self.kind}:{self.run}(action_cond={self.action_cond})"

    @torch.no_grad()
    def act(self, replan, cpx, gpx, h_norm, ext_zblk=None):
        dev = self.dev
        if self._lat is None:                    # lazy init if set_env didn't run
            self.reset(max(replan) + 1 if replan else 1)
        z_new = self.model.encode(cpx.to(dev).float())
        z_g = self.model.encode(gpx.to(dev).float())
        for row, i in enumerate(replan):
            self._lat[i].append(z_new[row])
            self._pblk[i].append(self._prev[i])              # driver's PREVIOUS block (None at start)
        lens = [len(self._lat[i]) for i in replan]
        Lmax, R, Dd = max(lens), len(replan), z_new.shape[-1]
        seq = torch.zeros(R, Lmax, Dd, device=dev)
        for row, i in enumerate(replan):
            buf = torch.stack(self._lat[i], dim=0)
            seq[row, : buf.shape[0]] = buf
        a_prev = a_prev_mask = None
        if self.action_cond:
            a_prev = torch.zeros(R, Lmax, self.block_dim, device=dev)
            a_prev_mask = torch.zeros(R, Lmax, dtype=torch.bool, device=dev)
            for row, i in enumerate(replan):
                for k, blk in enumerate(self._pblk[i]):
                    if blk is not None:
                        a_prev[row, k] = blk.to(dev); a_prev_mask[row, k] = True
        c = self.model.aggregate(seq, a_prev, a_prev_mask)   # (R, Lmax, D)
        last_idx = torch.tensor([l - 1 for l in lens], device=dev)
        c_last = c[torch.arange(R, device=dev), last_idx]
        zblk = self.model.gc_head(c_last, z_g, h_norm.to(dev))
        for row, i in enumerate(replan):                     # prev source for next frame
            self._prev[i] = (ext_zblk[row].detach() if ext_zblk is not None else zblk[row].detach())
        return self._unnorm(zblk), zblk


REGISTRY = {
    "split":   (gip.load_lewam_split_model,   SplitAdapter),
    "seq":     (gip.load_lewam_seq_model,     SeqAdapter),
    "unified": (gip.load_lewam_unified_model, UnifiedAdapter),
}


def build_adapter(spec, dev):
    """spec = '<kind>:<run_name>' -> a loaded, eval-mode, grad-free ModelAdapter."""
    kind, _, run = str(spec).partition(":")
    assert run, f"spec must be '<kind>:<run_name>', got {spec!r}"
    assert kind in REGISTRY, f"unknown kind {kind!r}; known: {sorted(REGISTRY)}"
    loader, cls = REGISTRY[kind]
    model, cfg = loader(run)
    model = model.to(dev).eval(); model.requires_grad_(False)
    ad = cls(model, cfg, dev); ad.run = run
    return ad


# --------------------------------------------------------------------------- #
# Generic shadow policy: execute the DRIVER's action, log the SHADOW's action on the same state.
# --------------------------------------------------------------------------- #
class ShadowComparePolicy(BasePolicy):
    def __init__(self, driver, shadow, action_block, horizon0, H_max, process=None, transform=None):
        super().__init__(process=process or {}, transform=transform or {})
        self.type = "shadow_compare_policy"
        self.driver = driver
        self.shadow = shadow
        self.action_block = int(action_block)
        self.action_dim = int(driver.raw)              # per-step raw dim of the EXECUTED (driver) action
        self.H_max = int(H_max)
        self.horizon0 = float(horizon0)
        self._action_buffer = None
        self._steps_left = None
        self.log_driver, self.log_shadow, self.log_h = [], [], []

    def set_env(self, env):
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._action_buffer = [deque() for _ in range(n)]
        self._steps_left = np.full(n, self.horizon0, dtype=np.float64)
        self.driver.reset(n); self.shadow.reset(n)

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        if self._action_buffer is None:
            self.set_env(self.env)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._action_buffer[i].clear()
                    self._steps_left[i] = self.horizon0
                    self.driver.reset_env(i); self.shadow.reset_env(i)

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        replan = [i for i in range(n) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            assert "goal" in info_dict, "shadow eval needs info_dict['goal'] (goal-reaching)"
            px = info_dict["pixels"][replan]
            gpx = info_dict["goal"][replan]
            gpx = gpx[:, -1] if gpx.ndim == 5 else gpx      # (R,C,H,W)
            cpx = px[:, -1] if px.ndim == 5 else px         # (R,C,H,W)
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=self.driver.dev, dtype=torch.float32)
            raw_drv, zblk_drv = self.driver.act(replan, cpx, gpx, h_norm)          # executed
            raw_shd, _ = self.shadow.act(replan, cpx, gpx, h_norm, ext_zblk=zblk_drv)  # logged only
            self.log_driver.append(raw_drv.cpu().numpy())
            self.log_shadow.append(raw_shd.cpu().numpy())
            self.log_h.append(h_norm.cpu().numpy())
            exec_blocks = raw_drv.reshape(len(replan), self.action_block, self.action_dim).cpu()
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(exec_blocks[row])
                self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)

        action = torch.full((n, self.action_dim), float("nan"))
        for i in range(n):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()


def _agree(driver, shadow):
    """Agreement stats between two (N, block_dim) raw-action arrays (shadow reproducing driver)."""
    driver = driver.reshape(driver.shape[0], -1); shadow = shadow.reshape(shadow.shape[0], -1)
    diff = shadow - driver
    dn = np.linalg.norm(driver, axis=1); sn = np.linalg.norm(shadow, axis=1)
    cos = float(((driver * shadow).sum(1) / np.clip(dn * sn, 1e-8, None)).mean())
    rel = float(np.sqrt((diff ** 2).mean()) / (np.sqrt((driver ** 2).mean()) + 1e-8))
    var = ((driver - driver.mean(0)) ** 2).mean()
    r2 = float(1.0 - (diff ** 2).mean() / (var + 1e-8))     # 1=identical, 0=predicts mean, <0=worse
    return dict(n=int(driver.shape[0]), action_mse=float((diff ** 2).mean()), cosine=cos,
                rel_rmse=rel, r2=r2, driver_rms=float(np.sqrt((driver ** 2).mean())),
                shadow_rms=float(np.sqrt((shadow ** 2).mean())))


@hydra.main(version_base=None, config_path="../configs/eval", config_name="reacher")
def run(cfg: DictConfig):
    driver_spec = cfg.get("driver"); shadow_spec = cfg.get("shadow")
    assert driver_spec and shadow_spec, "pass +driver=<kind>:<run> +shadow=<kind>:<run>"
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": gip.img_transform(cfg), "goal": gip.img_transform(cfg)}
    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    process = gip.build_process(cfg, dataset)
    episodes, starts = gip.sample_eval_episodes(cfg, dataset)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    driver = build_adapter(driver_spec, dev)
    shadow = build_adapter(shadow_spec, dev)

    action_block = int(cfg.plan_config.action_block)
    horizon0 = float(cfg.eval.goal_offset_steps) / float(action_block)
    policy = ShadowComparePolicy(
        driver, shadow, action_block=action_block, horizon0=horizon0,
        H_max=int(driver.cfg.get("H_max", 50)), process=process, transform=transform,
    )
    print(f"[shadow] driver={driver.desc()} shadow={shadow.desc()} | "
          f"driving env with DRIVER, shadowing SHADOW")

    world.set_policy(policy)
    t0 = time.time()
    metrics = world.evaluate(dataset=dataset, start_steps=starts,
                             goal_offset=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget,
                             episodes_idx=episodes,
                             callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
                             video=None)
    drv = np.concatenate(policy.log_driver, 0); shd = np.concatenate(policy.log_shadow, 0)
    h = np.concatenate(policy.log_h, 0)
    print(f"==== SHADOW {shadow.kind}-vs-{driver.kind} (driver SR this run: {metrics}) "
          f"| {time.time()-t0:.0f}s ====")
    print("ALL      :", _agree(drv, shd))
    # near-goal (small remaining horizon) vs far -- success often hinges on the precise endgame
    near = h <= np.quantile(h, 0.33); far = h >= np.quantile(h, 0.67)
    print("NEAR-goal:", _agree(drv[near], shd[near]))
    print("FAR-goal :", _agree(drv[far], shd[far]))


if __name__ == "__main__":
    run()
