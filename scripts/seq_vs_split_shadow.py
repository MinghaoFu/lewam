"""Shadow diagnostic: does the seq policy CHOOSE THE SAME ACTIONS as the working split
policy, on the split's own (goal-reaching) trajectories?

We drive the env with the WORKING split policy (LeWAMSplitPolicy, reacher ~96 SR) and, at
every replan, ALSO query the seq model's reactive (history_size=1) action on the SAME current
frame + goal + remaining-horizon, WITHOUT letting the seq touch the env. We log both action
blocks and compare.

Why this is clean: the split's own actions produced its frames, so the state distribution is
self-consistent AND competent (unlike teacher-forcing GT actions into a diverged rollout, which
is ill-defined -- observations are determined by actions). This isolates the seq's per-step
action MAP from its closed-loop covariate shift:
  * seq ~= split on the split's states  -> the seq's action map is fine on good states; its
    37% is closed-loop drift (its own slightly-off actions compound off-distribution).
  * seq != split on the split's states  -> the seq's action pathway is genuinely worse even on
    good states -> the predictor/representation, not just drift.

Both models z-score actions with the SAME dataset normalizer, so raw (un-z-scored) action
blocks are directly comparable. HS=1 needs no past-action context, so the comparison is a pure
reactive-map comparison.

Usage (reacher):
  python scripts/seq_vs_split_shadow.py --config-name reacher \
      policy=reacher_lewam_gc +seq_run=reacher_seq eval.num_eval=20 seed=42
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


class ShadowSplitPolicy(gip.LeWAMSplitPolicy):
    """Drives the env with the split policy (parent get_action) but also records, per replan,
    the seq model's reactive HS=1 action on the same (frame, goal, horizon). No env effect."""

    def __init__(self, seq_model, seq_cfg, *a, **kw):
        super().__init__(*a, **kw)
        self.seq = seq_model.eval()
        self.seq_cfg = seq_cfg
        dev = next(seq_model.parameters()).device
        self._seq_amean = torch.tensor(seq_cfg["action_mean"], dtype=torch.float32, device=dev)
        self._seq_astd = torch.tensor(seq_cfg["action_std"], dtype=torch.float32, device=dev).clamp_min(1e-6)
        self._seq_fs = int(seq_cfg["frameskip"])
        self._seq_raw = int(seq_cfg["action_raw_dim"])
        self.log_split, self.log_seq, self.log_h = [], [], []

    @torch.no_grad()
    def _seq_hs1_block(self, cpx, gpx, h_norm):
        """cpx,gpx: (R,C,H,W) transformed frames; h_norm: (R,). -> (R, block_dim) RAW action."""
        dev = next(self.seq.parameters()).device
        z = self.seq.encode_frames(cpx.to(dev).float().unsqueeze(1))            # (R,1,D)
        empty = z.new_zeros(z.size(0), 0, z.size(-1))
        pending = self.seq._tokenize_pending(z, empty)                          # (R,1,D) single token
        h_act = self.seq.predictor(pending)[:, 0::2][:, -1]                     # (R,D)
        zg = self.seq.state_encoder(gpx.to(dev).float())                        # (R,D)
        blk = self.seq.action_head(h_act, zg, h_norm.to(dev))                   # (R, block_dim) z-scored
        raw = blk.reshape(blk.size(0), self._seq_fs, self._seq_raw) * self._seq_astd + self._seq_amean
        return raw.reshape(blk.size(0), -1)

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        dev = next(self.model.parameters()).device
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(n)]
            self._steps_left = np.full(n, self.horizon0, dtype=np.float64)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._action_buffer[i].clear()
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
            z_t = self.model.encode(cpx.to(dev).float())
            z_g = self.model.encode(gpx.to(dev).float())
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=dev, dtype=torch.float32)
            z_blk = self.model.gc_head(z_t, z_g, h_norm)     # split z-scored block
            raw_split = (z_blk.reshape(len(replan), self.frameskip, self.raw_adim)
                         * self._astd + self._amean).reshape(len(replan), -1)
            # SHADOW: seq HS=1 action on the SAME frame/goal/horizon (no env effect)
            raw_seq = self._seq_hs1_block(cpx, gpx, h_norm)
            self.log_split.append(raw_split.cpu().numpy())
            self.log_seq.append(raw_seq.cpu().numpy())
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


def _agree(split, seq, h):
    """Agreement stats between two (N, block_dim) raw-action arrays."""
    split = split.reshape(split.shape[0], -1); seq = seq.reshape(seq.shape[0], -1)
    diff = seq - split
    mse = float((diff ** 2).mean())
    # per-block cosine similarity of the action vectors
    sn = np.linalg.norm(split, axis=1); qn = np.linalg.norm(seq, axis=1)
    denom = np.clip(sn * qn, 1e-8, None)
    cos = float(((split * seq).sum(1) / denom).mean())
    # relative error vs the split action scale
    rel = float(np.sqrt((diff ** 2).mean()) / (np.sqrt((split ** 2).mean()) + 1e-8))
    # R^2 of seq predicting split (1 = identical, 0 = predicts the mean, <0 = worse than mean)
    var = ((split - split.mean(0)) ** 2).mean()
    r2 = float(1.0 - (diff ** 2).mean() / (var + 1e-8))
    return dict(n=int(split.shape[0]), action_mse=mse, cosine=cos, rel_rmse=rel, r2=r2,
                split_rms=float(np.sqrt((split ** 2).mean())), seq_rms=float(np.sqrt((seq ** 2).mean())))


@hydra.main(version_base=None, config_path="../configs/eval", config_name="reacher")
def run(cfg: DictConfig):
    seq_run = cfg.get("seq_run")
    assert seq_run, "pass +seq_run=<seq_run_name>"
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": gip.img_transform(cfg), "goal": gip.img_transform(cfg)}
    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    process = gip.build_process(cfg, dataset)
    episodes, starts = gip.sample_eval_episodes(cfg, dataset)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    split_model, split_cfg = gip.load_lewam_split_model(cfg.policy)
    split_model = split_model.to(dev).eval(); split_model.requires_grad_(False)
    seq_model, seq_cfg = gip.load_lewam_seq_model(seq_run)
    seq_model = seq_model.to(dev).eval(); seq_model.requires_grad_(False)

    action_block = int(cfg.plan_config.action_block)
    horizon0 = float(cfg.eval.goal_offset_steps) / float(action_block)
    policy = ShadowSplitPolicy(
        seq_model, seq_cfg,
        model=split_model, cfg=split_cfg, action_block=action_block,
        action_dim=int(split_cfg["action_dim"]) // action_block, horizon0=horizon0,
        H_max=int(split_cfg.get("H_max", 50)), process=process, transform=transform,
    )
    print(f"[shadow] split={cfg.policy} seq={seq_run} | driving env with SPLIT, shadowing SEQ (HS=1)")

    world.set_policy(policy)
    t0 = time.time()
    metrics = world.evaluate(dataset=dataset, start_steps=starts,
                             goal_offset=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget,
                             episodes_idx=episodes,
                             callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
                             video=None)
    split = np.concatenate(policy.log_split, 0); seq = np.concatenate(policy.log_seq, 0)
    h = np.concatenate(policy.log_h, 0)
    print(f"==== SHADOW seq-vs-split (split SR this run: {metrics}) | {time.time()-t0:.0f}s ====")
    print("ALL      :", _agree(split, seq, h))
    # near-goal (small remaining horizon) vs far -- reacher success hinges on the precise endgame
    near = h <= np.quantile(h, 0.33); far = h >= np.quantile(h, 0.67)
    print("NEAR-goal:", _agree(split[near], seq[near], h[near]))
    print("FAR-goal :", _agree(split[far], seq[far], h[far]))
    # reference: split action vs a shifted copy of itself would give r2~1; here r2 shows seq fidelity


if __name__ == "__main__":
    run()
