"""Prompt-MPC (mode: unified_prompt_mpc; legacy alias unified_dgoal) -- ours.

A per-episode prompt vector delta is added to the goal latent fed to the frozen
goal-conditioned policy head, and optimized at test time against the frozen world model's
cost. Every action of every iterate is the frozen policy's own output, so the search can
never leave the policy's behavioral manifold -- the off-manifold exploitation channel that
breaks Gradient MPC does not exist as a coordinate. Iterate 0 (delta=0) is the pure policy
plan and stays in the candidate set: the floor is "do not optimize".

pm_cost="anymin" aligns the planning cost with the benchmark's reach-anytime success rule
(min over rolled steps instead of the terminal step); pm_seg>0 enables per-step prompts
(the capacity ablation); pm_random replaces the gradient with an equal-norm random
direction (the search-verification control).
"""

from collections import deque

import numpy as np
import torch

from lewam.models.gip import _h0_at
from lewam.models.grad_mpc import LeWAMUnifiedGradPolicy


class LeWAMUnifiedPromptMPCPolicy(LeWAMUnifiedGradPolicy):
    """Plan-through-the-policy (mode: unified_prompt_mpc). The optimization variable is a small
    offset delta on the goal embedding FED TO gc_head; every action of every iterate is the
    frozen policy's own output, so the off-manifold exploitation channel does not exist as a
    coordinate. Dynamics conditioning and the cost both use the TRUE goal; iterate 0
    (delta=0) is the pure policy rollout, so the floor is 'do not optimize'."""

    def __init__(self, model, cfg, *args, **kwargs):
        self.pm_steps = int(kwargs.pop("pm_steps", 20))
        self.pm_lr = float(kwargs.pop("pm_lr", 0.02))
        self.pm_clip = float(kwargs.pop("pm_clip", 5.0))
        self.pm_rho = float(kwargs.pop("pm_rho", 0.3))   # max ||delta|| relative to ||z_goal||
        self.pm_random = bool(kwargs.pop("pm_random", False))
        # per-step prompts (capacity ablation): per-timestep goal tilts with residual
        # parameterization delta_t = delta_base + cumsum(r)[seg(t)]. pm_seg=0 -> single prompt
        # (default), pm_seg=1 -> one residual per block. Residuals get a tighter budget pm_r_rho.
        self.pm_seg = int(kwargs.pop("pm_seg", 0))
        # 'terminal' = MSE at each env's data horizon (default); 'anymin' = min over the rolled
        # steps within each env's horizon -- aligns the planning cost with the reach-anytime
        # success rule (a plan that passes through the goal early scores what it deserves).
        self.pm_cost = str(kwargs.pop("pm_cost", "terminal"))
        self.pm_r_rho = float(kwargs.pop("pm_r_rho", 0.1))
        super().__init__(model, cfg, *args, **kwargs)
        self.type = "lewam_unified_prompt_mpc"

    def _policy_rollout(self, window0, win_len0, z_goal_tilt, z_goal_true, steps_left0,
                        H_env=None, H_max=None):
        """AR rollout with the graph OPEN through aggregate+gc_head+dynamics. Returns
        (U (R,H,B), terminal cost against the TRUE goal). With H_env, rolls the batch max and
        gathers each env's terminal at its own data-given H_i."""
        win, wl = window0.clone(), win_len0.clone()
        sl = steps_left0.copy()
        device = window0.device
        if H_max is None:
            H_max = self.grad_H
        z_steps = []
        blocks, z_term = [], None
        for _h in range(H_max):
            hn = torch.tensor(np.minimum(sl, self.H_max) / self.H_max,
                              device=device, dtype=torch.float32)
            if self.ablate_horizon:
                hn = torch.zeros_like(hn)
            cctx = self._context_at_head(win, wl)
            tilt = z_goal_tilt[:, _h] if z_goal_tilt.dim() == 3 else z_goal_tilt
            blk = self.model.gc_head.point(self.model.gc_head(cctx, tilt, hn))
            blocks.append(blk)
            z_term = self._dyn_step(cctx, blk, z_goal_true)
            z_steps.append(z_term)
            win, wl = self._append_latent(win, wl, z_term, int(self.ctx_cap))
            sl = np.maximum(sl - 1.0, 1.0)
        U = torch.stack(blocks, dim=1)
        Z = torch.stack(z_steps, dim=0)
        if H_env is not None:
            idx = torch.as_tensor(np.asarray(H_env) - 1, device=device, dtype=torch.long)
            z_term = Z[idx, torch.arange(Z.shape[1], device=device)]
        if getattr(self, "pm_cost", "terminal") == "anymin":
            d_all = ((Z - z_goal_true.unsqueeze(0)) ** 2).mean(-1)          # (H_max, R)
            if H_env is not None:
                steps = torch.arange(Z.shape[0], device=device).unsqueeze(1)
                lim = torch.as_tensor(np.asarray(H_env), device=device, dtype=torch.long).unsqueeze(0)
                d_all = d_all.masked_fill(steps >= lim, float("inf"))
            cost = d_all.min(dim=0).values
        else:
            cost = ((z_term - z_goal_true) ** 2).mean(-1)
        return U, cost

    def get_action(self, info_dict, **kwargs):
        info_dict = self._prepare_info(info_dict)
        n_envs = self.env.num_envs
        device = next(self.model.parameters()).device
        self._call += 1
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(n_envs)]
            self._steps_left = np.full(n_envs, self.horizon0, dtype=np.float64)
        if self._lat_buf is None:
            self._lat_buf = [[] for _ in range(n_envs)]
        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for env_i in range(n_envs):
                if flush[env_i]:
                    self._action_buffer[env_i].clear()
                    self._lat_buf[env_i].clear()
                    self._steps_left[env_i] = _h0_at(self.horizon0, env_i)
                    self._diag_prev.pop(env_i, None)
        term = info_dict.get("terminated")
        is_dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n_envs, dtype=bool)
        replan_envs = [i for i in range(n_envs) if len(self._action_buffer[i]) == 0 and not is_dead[i]]
        if replan_envs:
            with torch.no_grad():
                cur_px = info_dict["pixels"][replan_envs]
                goal_px = info_dict["goal"][replan_envs]
                goal_px = goal_px[:, -1] if goal_px.ndim == 5 else goal_px
                cur_px = cur_px[:, -1] if cur_px.ndim == 5 else cur_px
                z_cur = self.model.encode(cur_px.to(device).float())
                z_goal = self.model.encode(goal_px.to(device).float())
                for row, env_i in enumerate(replan_envs):
                    self._lat_buf[env_i].append(z_cur[row])
                    prev = self._diag_prev.pop(env_i, None)
                    if prev is not None:
                        realized = float(((z_cur[row] - z_goal[row]) ** 2).mean())
                        self._diag.append(("realized", prev[0], env_i, prev[1], realized))
                R, D = z_cur.shape
                ctx_cap = int(self.ctx_cap)
                window0 = torch.zeros(R, ctx_cap, D, device=device)
                win_len0 = torch.empty(R, dtype=torch.long, device=device)
                for row, env_i in enumerate(replan_envs):
                    real = torch.stack(self._lat_buf[env_i][-ctx_cap:], dim=0)
                    window0[row, : real.shape[0]] = real
                    win_len0[row] = real.shape[0]
                sl0 = np.maximum(self._steps_left[replan_envs], 1.0).astype(np.float64)
                gnorm = z_goal.norm(dim=-1, keepdim=True)
                H_env, H_max_b = self._env_H(replan_envs)

            n_seg = 0 if self.pm_seg <= 0 else max(int(np.ceil(H_max_b / self.pm_seg)), 1)
            with torch.enable_grad():
                delta = torch.zeros(R, D, device=device, requires_grad=True)
                resid = (torch.zeros(R, n_seg, D, device=device, requires_grad=True)
                         if n_seg > 0 else None)
                opt = torch.optim.Adam([delta] + ([resid] if resid is not None else []), lr=self.pm_lr)
                best_U = best_cost = best_delta = None
                best_iter = torch.zeros(R, dtype=torch.long)
                seg_idx = (torch.arange(H_max_b, device=device) // max(self.pm_seg, 1)).clamp(max=max(n_seg - 1, 0)) if n_seg > 0 else None
                def _tilts():
                    if resid is None:
                        return z_goal + delta
                    walk = torch.cumsum(resid, dim=1)[:, seg_idx]          # (R, H_max, D) residual walk
                    return z_goal.unsqueeze(1) + delta.unsqueeze(1) + walk
                for _k in range(self.pm_steps + 1):
                    U, cost = self._policy_rollout(window0, win_len0, _tilts(), z_goal, sl0,
                                                   H_env=H_env, H_max=H_max_b)
                    c = cost.detach()
                    if best_cost is None:
                        best_cost, best_U = c.clone(), U.detach().clone()
                        best_delta = delta.detach().clone()
                        c0_pm = c.clone()
                        U0_pm = U.detach().clone()
                    else:
                        m = c < best_cost
                        best_U[m] = U.detach()[m]
                        best_delta[m] = delta.detach()[m]
                        best_cost = torch.where(m, c, best_cost)
                        best_iter[m.cpu()] = _k
                    if _k == self.pm_steps:
                        break
                    opt.zero_grad(set_to_none=True)
                    params = [delta] + ([resid] if resid is not None else [])
                    grads = torch.autograd.grad(cost.mean(), params, allow_unused=True)
                    if self.pm_random:
                        g0 = grads[0]
                        r0 = torch.randn_like(g0)
                        grads = (r0 * (g0.norm(dim=-1, keepdim=True) / r0.norm(dim=-1, keepdim=True).clamp_min(1e-9)),) + tuple(grads[1:])
                    for p_, g_ in zip(params, grads):
                        p_.grad = g_ if g_ is not None else torch.zeros_like(p_)
                    torch.nn.utils.clip_grad_norm_(params, self.pm_clip)
                    opt.step()
                    with torch.no_grad():
                        scale = (self.pm_rho * gnorm / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
                        delta.mul_(scale)
                        if resid is not None:
                            rs = (self.pm_r_rho * gnorm.unsqueeze(1)
                                  / resid.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
                            resid.mul_(rs)
                plan = best_U

            with torch.no_grad():
                delta = best_delta                            # report/store the executed delta
                if not hasattr(self, "_plan_store"):
                    self._plan_store = {}
                if self._call not in self._plan_store:
                    self._plan_store[self._call] = ([int(e) for e in replan_envs],
                                                    U0_pm.cpu().numpy().copy(),
                                                    plan.detach().cpu().numpy().copy())
                if not hasattr(self, "_delta_store"):
                    self._delta_store = {}
                if self._call not in self._delta_store:
                    self._delta_store[self._call] = ([int(e) for e in replan_envs],
                                                     delta.detach().cpu().numpy().copy())
                dnorm = delta.detach().norm(dim=-1)
                for row, env_i in enumerate(replan_envs):
                    self._diag.append(("plan", self._call, env_i, float(self._steps_left[env_i]),
                                       float(c0_pm[row]), float(best_cost[row]),
                                       int(best_iter[row]), float(dnorm[row])))
                    self._diag_prev[env_i] = (self._call, float(best_cost[row]))
                Hp = plan.shape[1]
                plan_raw = (plan.reshape(R, Hp, self.frameskip, self.raw_adim)
                            * self._astd + self._amean)
                plan_raw = plan_raw.reshape(R, Hp * self.action_block, self.action_dim).cpu()
                for row, env_i in enumerate(replan_envs):
                    n_i = (min(self.grad_exec_k, int(H_env[row])) if self.grad_exec_k > 0
                           else (int(H_env[row]) if self.grad_exec_full else 1))
                    for t in range(n_i * self.action_block):
                        self._action_buffer[env_i].append(plan_raw[row, t])
                    self._steps_left[env_i] = max(self._steps_left[env_i] - float(n_i), 1.0)
        action = torch.full((n_envs, self.action_dim), float("nan"))
        for env_i in range(n_envs):
            if not is_dead[env_i]:
                action[env_i] = self._action_buffer[env_i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()
