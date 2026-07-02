"""Multi-task dataset (enabled via data=multitask7).

Wraps N per-task swm datasets (each with its own column normalizers) behind
two-step task-balanced sampling (Newt / MT-JEPA convention):
  1. task ~ Uniform({0..n_tasks-1})
  2. sample ~ Uniform(task subset)

Per item it emits a collate-uniform dict:
  pixels      (T, C, H, W)   same img_size across tasks
  action      (T, f*d_max)   zero-padded per frame chunk; layout (f d), d fastest
  proprio     (T, p_max)     zero-padded; per-task source key (proprio | observation)
  task_id     ()             long, index into the task list / CLIP table
  action_mask (f*d_max,)     1 on valid dims: tile([1]*d_t + [0]*(d_max-d_t), f)

Padding happens AFTER the per-task normalizers (zeros are masked from the action
loss and zeroed again before the action encoder, so their value never matters).
"""

import numpy as np
import torch


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, subsets, task_names, action_dims, proprio_dims, proprio_keys,
                 frameskip, length=None, active_tasks=None, sampling_alpha=0.0):
        assert len(subsets) == len(task_names) == len(action_dims) == len(proprio_dims) == len(proprio_keys)
        self.subsets = subsets
        self.task_names = list(task_names)
        self.frameskip = int(frameskip)
        self.action_dims = [int(d) for d in action_dims]
        self.proprio_dims = [int(p) for p in proprio_dims]
        self.proprio_keys = list(proprio_keys)
        self.n_tasks = len(subsets)
        self.d_max = max(self.action_dims)
        self.p_max = max(self.proprio_dims)

        # per-task action masks in the (f d) flattened layout (d fastest within a frame)
        masks = []
        for d in self.action_dims:
            per_frame = torch.cat([torch.ones(d), torch.zeros(self.d_max - d)])
            masks.append(per_frame.repeat(self.frameskip))
        self.action_masks = torch.stack(masks)  # (n_tasks, f*d_max)

        self._len = int(length) if length else sum(len(s) for s in subsets)

        # Finetune (Newt-style): restrict SAMPLING to a task subset while the task
        # list, table order, and pad dims stay full-size (checkpoint-compatible).
        if active_tasks:
            self.active = [self.task_names.index(t) if isinstance(t, str) else int(t)
                           for t in active_tasks]
        else:
            self.active = list(range(self.n_tasks))

        # Task-sampling temperature: p_task ~ n_task^alpha.
        # alpha=0 -> uniform across TASKS (default); alpha=1 -> uniform across FRAMES
        # (big datasets dominate); 0.3-0.5 boosts large tasks without drowning small ones.
        sizes = np.array([len(self.subsets[i]) for i in self.active], dtype=np.float64)
        w = sizes ** float(sampling_alpha)
        self.task_p = w / w.sum()

    def __len__(self):
        return self._len

    def _pad_action(self, act, d):
        """(T, f*d) -> (T, f*d_max), zero-padding each frame chunk from d to d_max."""
        if d == self.d_max:
            return act
        T = act.shape[0]
        act = act.reshape(T, self.frameskip, d)
        pad = act.new_zeros(T, self.frameskip, self.d_max - d)
        return torch.cat([act, pad], dim=-1).reshape(T, self.frameskip * self.d_max)

    def __getitem__(self, idx):
        t = int(self.active[np.random.choice(len(self.active), p=self.task_p)])
        sub = self.subsets[t]
        item = sub[int(np.random.randint(len(sub)))]

        act = torch.as_tensor(item["action"]).float()
        pr = torch.as_tensor(item[self.proprio_keys[t]]).float()
        if pr.shape[-1] < self.p_max:
            pr = torch.nn.functional.pad(pr, (0, self.p_max - pr.shape[-1]))

        return {
            "pixels": item["pixels"],
            "action": self._pad_action(act, self.action_dims[t]),
            "proprio": pr,
            "task_id": torch.tensor(t, dtype=torch.long),
            "action_mask": self.action_masks[t],
        }
