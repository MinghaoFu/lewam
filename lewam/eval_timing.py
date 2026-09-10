"""Planning-time records for the evaluation harnesses.

One record per policy call: the whole call (observation encoding, planning, bookkeeping) and the
plan phase alone (from entering the planner to the finalized action chunk), the envs planned in
that call, each one's plan-cycle index within its episode, its time to goal, and the imagined
dynamics length. Every statistic of interest (time per cycle, the first plan only, amortized per
episode, per successful episode) is computed afterwards from the records.

Both harnesses batch the episodes: one call plans every env whose action buffer is empty, so a
record's `t_plan_s` is the batched cost of `len(envs_planned)` plans; `t_plan_s / len(envs_planned)`
is the amortized cost of one.
"""
import json
import time

import numpy as np
import torch


class _Timed:
    """A callable attribute replaced by a timing wrapper; other attribute access passes through to
    the original object. `mark` optionally wraps the call itself (used to flag the plan phase)."""

    def __init__(self, wrapped, on_done, cuda, mark=None):
        self._wrapped, self._on_done, self._cuda = wrapped, on_done, cuda
        self._call = wrapped if mark is None else mark(wrapped)

    def __call__(self, *args, **kwargs):
        if self._cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        out = self._call(*args, **kwargs)
        if self._cuda:
            torch.cuda.synchronize()
        self._on_done(time.perf_counter() - started)
        return out

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _resolve(root, dotted):
    owner = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


class PlanTimer:
    """Wraps `policy.get_action` and records one entry per call.

    plan_attr:       dotted attribute of the callable that runs the plan (our planners "_propose",
                     stable-worldmodel's planning policy "solver"); absent -> t_plan_s is None
    encode_attr:     dotted attribute of the encoder callable; its time inside the plan phase is
                     goal encoding, outside it observation encoding
    imagined_attr:   policy attribute holding the dynamics steps imagined by the last plan
    steps_left_attr: policy attribute holding each env's time to goal in blocks
    horizon_blocks:  the imagined length when the policy does not expose it (LeWM: its horizon)
    """

    def __init__(self, policy, num_envs, cuda, plan_attr=None, encode_attr=None, imagined_attr=None,
                 steps_left_attr=None, horizon_blocks=None):
        self.policy, self.num_envs, self.cuda = policy, int(num_envs), bool(cuda)
        self.imagined_attr, self.steps_left_attr, self.horizon_blocks = imagined_attr, steps_left_attr, horizon_blocks
        self.records = []
        self.cycle_count = np.zeros(self.num_envs, dtype=int)
        self.time_per_env = np.zeros(self.num_envs)
        self._in_plan = False
        self._t_plan = self._t_encode_in_plan = self._t_encode_outside = 0.0
        self._orig_get_action = policy.get_action
        policy.get_action = self.__call__
        if plan_attr and self._has(policy, plan_attr):
            owner, name = _resolve(policy, plan_attr)
            setattr(owner, name, _Timed(getattr(owner, name), self._plan_done, self.cuda, mark=self._mark_plan))
            self._plan_wrapper = getattr(owner, name)
        else:
            self._plan_wrapper = None
        if encode_attr and self._has(policy, encode_attr):
            owner, name = _resolve(policy, encode_attr)
            setattr(owner, name, _Timed(getattr(owner, name), self._encode_done, self.cuda))

    @staticmethod
    def _has(policy, dotted):
        try:
            owner, name = _resolve(policy, dotted)
            return hasattr(owner, name)
        except AttributeError:
            return False

    def _mark_plan(self, plan_fn):
        def marked(*args, **kwargs):
            self._in_plan = True
            try:
                return plan_fn(*args, **kwargs)
            finally:
                self._in_plan = False
        return marked

    def _plan_done(self, seconds):
        self._t_plan += seconds

    def _encode_done(self, seconds):
        if self._in_plan:
            self._t_encode_in_plan += seconds
        else:
            self._t_encode_outside += seconds

    def __call__(self, info_dict, **kwargs):
        terminated = info_dict.get("terminated")
        dead = (np.asarray(terminated, dtype=bool).reshape(-1) if terminated is not None
                else np.zeros(self.num_envs, dtype=bool))
        flush = info_dict.get("_needs_flush")                  # peek only; the policy pops it
        buffers = getattr(self.policy, "_action_buffer", None)  # None before the first call: everyone plans
        alive = [i for i in range(self.num_envs) if not dead[i]]
        planned = [i for i in alive if buffers is None or len(buffers[i]) == 0
                   or (flush is not None and bool(flush[i]))]
        if flush is not None:
            self.cycle_count[np.asarray(flush, dtype=bool).reshape(-1)] = 0
        steps_left = getattr(self.policy, self.steps_left_attr, None) if self.steps_left_attr else None
        if steps_left is not None:
            steps_left = np.array(steps_left, dtype=float).copy()   # the value at planning time; the policy decrements it in the call
        self._t_plan = self._t_encode_in_plan = self._t_encode_outside = 0.0
        if self.cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        out = self._orig_get_action(info_dict, **kwargs)
        if self.cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        imagined = getattr(self.policy, self.imagined_attr, None) if self.imagined_attr else None
        self.records.append(dict(
            call_idx=len(self.records), n_alive=len(alive), envs_planned=[int(i) for i in planned],
            cycle_idx=[int(self.cycle_count[i]) for i in planned],
            steps_left=([float(steps_left[i]) for i in planned] if steps_left is not None else None),
            imagined_blocks=(int(imagined) if imagined is not None else self.horizon_blocks) if planned else None,
            t_call_s=seconds, t_plan_s=(self._t_plan if self._plan_wrapper is not None and planned else None),
            t_encode_obs_s=self._t_encode_outside, t_encode_goal_s=self._t_encode_in_plan))
        if alive:
            self.time_per_env[alive] += seconds / len(alive)
        for i in planned:
            self.cycle_count[i] += 1
        return out

    def summary(self):
        """The per-run averages kept for continuity with earlier timing files."""
        calls = np.asarray([[r["t_call_s"], r["n_alive"], len(r["envs_planned"])] for r in self.records], dtype=float)
        calls = calls.reshape(-1, 3)
        plan_calls = calls[calls[:, 2] > 0]
        per_block = plan_calls[:, 0] / plan_calls[:, 2] if len(plan_calls) else np.zeros(0)
        std = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
        mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
        t_plan = [r["t_plan_s"] for r in self.records if r["t_plan_s"] is not None]
        return dict(n_calls=int(len(calls)), n_replan_calls=int(len(plan_calls)),
                    t_call_mean_s=mean(plan_calls[:, 0]), t_call_std_s=std(plan_calls[:, 0]),
                    t_plan_mean_s=mean(t_plan), t_plan_std_s=std(t_plan),
                    n_replanned_mean=mean(plan_calls[:, 2]),
                    t_block_amortized_mean_s=mean(per_block), t_block_amortized_std_s=std(per_block),
                    t_episode_mean_s=mean(self.time_per_env), t_episode_std_s=std(self.time_per_env),
                    replans_per_env_mean=mean(self.cycle_count))

    def per_episode(self):
        """Each episode's plans in order: index = plan number (0 = the first plan), one dict per plan
        with the batch's times, the batch size and the amortized share, so an episode's total planning
        time is a sum over its list and its number of replans is the list length minus one."""
        plans = {env: [] for env in range(self.num_envs)}
        for r in self.records:
            batch = len(r["envs_planned"])
            for row, env in enumerate(r["envs_planned"]):
                plans[env].append(dict(
                    plan_idx=r["cycle_idx"][row], call_idx=r["call_idx"], batch_size=batch,
                    steps_left=(r["steps_left"][row] if r["steps_left"] is not None else None),
                    imagined_blocks=r["imagined_blocks"],
                    t_call_s=r["t_call_s"], t_plan_s=r["t_plan_s"],
                    t_encode_obs_s=r["t_encode_obs_s"], t_encode_goal_s=r["t_encode_goal_s"],
                    t_call_amortized_s=r["t_call_s"] / batch,
                    t_plan_amortized_s=(r["t_plan_s"] / batch if r["t_plan_s"] is not None else None)))
        return plans

    def record_lines(self, prefix="[timing-record] "):
        return [prefix + json.dumps(r) for r in self.records if r["envs_planned"]]

    def write_per_episode(self, path, success=None):
        """The per-episode lists as json; `success` (per-env flags, same order) is stored alongside."""
        plans = self.per_episode()
        payload = {str(env): dict(success=(bool(success[env]) if success is not None else None), plans=plans[env])
                   for env in range(self.num_envs)}
        with open(path, "w") as f:
            json.dump(payload, f, indent=1)

    def write_records(self, path):
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")
