"""Oracle candidate-set "solver" for stable_worldmodel's WorldModelPolicy (owner sanity check
2026-08-29, LeWM as the reference verifier).

Instead of CEM, every replan scores a FIXED candidate set with the world model's own cost
(`model.get_cost` = LeWM's autoregressive rollout to the end of the plan horizon + final-latent
distance to the goal): candidate 0 is the replayed demo's expert action sequence at the env's
current raw step (horizon blocks), candidates 1..K-1 are wrong sequences -- "uniform" over the
dataset action box or "shuffle" = other demos' sequences. The cheapest candidate is returned as
the plan; the policy executes its first `receding_horizon` blocks. SR then measures whether the
world model as verifier recovers the demo's success when the expert IS in the set, and the
expert pick rate says how often it does pick it.

OracleWorldModelPolicy only adds an `env_idx` entry to the info dict so the solver can keep
per-env bookkeeping of how many raw steps have been executed (the demo's current step).
"""
import numpy as np
import torch
from gymnasium.spaces import Box

import stable_worldmodel as swm


class OracleSolver:
    def __init__(self, model, expert_actions, cands="uniform", k=32, seed=0, device="cuda"):
        self.model = model
        self.expert_actions = [np.asarray(a, np.float32) for a in expert_actions]   # per env (T, adim) raw
        assert cands in ("uniform", "shuffle"), cands
        self.cands, self.k = cands, int(k)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(int(seed) + 101)
        allv = np.concatenate(self.expert_actions, 0)
        self.act_lo, self.act_hi = allv.min(0), allv.max(0)
        self.raw_done = np.zeros(len(self.expert_actions), dtype=int)
        self.hits = self.n = 0
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    # ---------------------------------------------------------------- Solver protocol
    def configure(self, *, action_space, n_envs, config):
        assert isinstance(action_space, Box), type(action_space)
        self._raw_adim = int(np.prod(action_space.shape[1:]))
        self._n_envs = int(n_envs)
        self._config = config

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._raw_adim * int(self._config.action_block)

    @property
    def horizon(self):
        return int(self._config.horizon)

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    # ---------------------------------------------------------------- candidates
    def _sequence(self, src, t0, n_raw):
        """n_raw raw actions of `src` from t0, holding the last action past the demo's end."""
        seq = src[t0:t0 + n_raw]
        if len(seq) < n_raw:
            last = src[-1:] if len(src) else np.zeros((1, src.shape[-1]), np.float32)
            seq = np.concatenate([seq, np.repeat(last, n_raw - len(seq), 0)], 0)
        return seq

    def _candidates(self, env_ids):
        fs, H, K = int(self._config.action_block), self.horizon, self.k
        n_raw = H * fs
        adim = self._raw_adim
        out = np.zeros((len(env_ids), K, n_raw, adim), np.float32)
        for row, i in enumerate(env_ids):
            out[row, 0] = self._sequence(self.expert_actions[i], int(self.raw_done[i]), n_raw)
            if self.cands == "uniform":
                out[row, 1:] = self.rng.uniform(self.act_lo, self.act_hi, (K - 1, n_raw, adim))
            else:
                for c in range(1, K):
                    src = self.expert_actions[int(self.rng.integers(len(self.expert_actions)))]
                    s0 = int(self.rng.integers(0, max(1, len(src) - n_raw + 1)))
                    out[row, c] = self._sequence(src, s0, n_raw)
        # (n, K, H, fs*adim): the policy's block layout (row-major over (t, adim) inside a block)
        return out.reshape(len(env_ids), K, H, fs * adim)

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        env_ids = [int(v) for v in np.asarray(info_dict["env_idx"]).reshape(-1)]
        n, K = len(env_ids), self.k
        cands = torch.from_numpy(self._candidates(env_ids)).to(self.device, self._dtype)
        # one env per get_cost call (LeWM.criterion broadcasts the goal embedding against
        # (B, S, T, D) from the right, valid only for B == 1 -- the reason cem.yaml pins batch_size 1)
        costs = []
        for row in range(n):
            expanded = {}
            for key, v in info_dict.items():
                if key == "env_idx":
                    continue
                if torch.is_tensor(v):
                    target = self._dtype if v.is_floating_point() else None
                    expanded[key] = (v[row:row + 1].to(device=self.device, dtype=target)
                                     .unsqueeze(1).expand(1, K, *v.shape[1:]))
                elif isinstance(v, np.ndarray):
                    expanded[key] = np.repeat(v[row:row + 1, None, ...], K, axis=1)
                else:
                    expanded[key] = v
            c = self.model.get_cost(expanded, cands[row:row + 1])
            assert c.shape == (1, K), c.shape
            costs.append(c[0])
        costs = torch.stack(costs, 0)
        pick = costs.argmin(1)
        self.hits += int((pick == 0).sum()); self.n += n
        exec_raw = int(self._config.receding_horizon) * int(self._config.action_block)
        for i in env_ids:
            self.raw_done[i] += exec_raw
        if self.n % 50 < n:
            print(f"[oracle] expert picked {self.hits}/{self.n} = {100.0 * self.hits / max(1, self.n):.1f}% "
                  f"of replans (cands={self.cands}, K={K}, lewm rollout H={self.horizon})", flush=True)
        best = cands[torch.arange(n, device=cands.device), pick]
        return {"actions": best.detach().cpu(), "costs": costs.min(1).values.cpu().tolist()}


class OracleWorldModelPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy that tags the info dict with env indices (sliced per replan by the base
    class) so OracleSolver can index each env's demo."""

    def get_action(self, info_dict, **kwargs):
        info_dict = dict(info_dict)
        info_dict["env_idx"] = np.arange(getattr(self.env, "num_envs", 1), dtype=np.int64)
        return super().get_action(info_dict, **kwargs)
