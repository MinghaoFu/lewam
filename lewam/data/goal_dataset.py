"""Goal-sampling dataset wrapper for the goal-conditioned policy.

Constructed only when `action_pred.goal_conditioned=true`. Wraps a swm EpisodeDataset; for each window
it adds a HINDSIGHT goal -- a frame at `t+h` (h ~ Uniform[1, hindsight_max_k]
observation-steps ahead of the window's last observed frame, clamped to the
episode end) -- plus the realized horizon `h`. This is what turns the demo set
into goal-reaching supervision: "to get from this window toward that future
state, take this action". Paired with `goal_dropout` in lejepa_forward (zero the
goal for a fraction of samples), one model learns BOTH the goal-conditioned and
the goal-agnostic policy (the two-setting WAM / BESO classifier-free-guidance).
"""
import numpy as np
import torch


class GoalSamplingDataset(torch.utils.data.Dataset):
    def __init__(self, base, hindsight_max_k: int = 50, goal_key: str = "pixels"):
        self.base = base
        self.frameskip = int(getattr(base, "frameskip", 1))
        self.span = int(getattr(base, "span", self.frameskip))
        self.hindsight_max_k = int(hindsight_max_k)
        self.goal_key = goal_key
        self.lengths = base.lengths
        self.clip_indices = base.clip_indices
        # NOTE: base._load_slice ALREADY applies dataset.transform, so the goal frame comes out
        # preprocessed like the window frames. Do NOT re-apply img-preproc here (double-normalizes -> [-11,9.9]).
        # passthrough attrs some trainers read off the dataset
        for a in ("get_dim", "get_col_data", "column_names", "offsets"):
            if hasattr(base, a) and not hasattr(self, a):
                setattr(self, a, getattr(base, a))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        ep, start = self.clip_indices[idx]
        L = int(self.lengths[ep])
        last = start + self.span - 1                       # window's last observed raw frame
        h = int(np.random.randint(1, self.hindsight_max_k + 1))   # obs-steps ahead
        gpos = min(last + h * self.frameskip, L - 1)        # clamp to episode end
        realized_h = max(1, (gpos - last) // self.frameskip)
        gslice = self.base._load_slice(ep, gpos, gpos + 1)  # one frame, preprocessed like the window
        gframe = gslice[self.goal_key]
        gframe = gframe[0] if getattr(gframe, "ndim", 0) >= 1 and len(gframe) >= 1 else gframe
        item["goal"] = torch.as_tensor(np.asarray(gframe))
        item["horizon"] = torch.tensor(float(realized_h))
        return item
