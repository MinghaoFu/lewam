"""Gradient MPC (mode: unified_grad) over the frozen LeWAM-Unified model.

The action sequence is the parameter, warm-started from the policy (grad_steps=0 = execute
the warm plan as-is), refined by Adam against the terminal latent cost through the frozen
dynamics. Keeps the warm start as a floor and executes the best iterate by model cost.
Prompt-MPC (ours) lives in lewam/models/prompt_mpc.py and subclasses this planner's
execution/diagnostic machinery.
"""

from collections import deque

import numpy as np
import torch

from lewam.models.gip import LeWAMUnifiedCEMPolicy, _h0_at


class LeWAMUnifiedGradPolicy(LeWAMUnifiedCEMPolicy):
    """Receding-horizon gradient planner. Inherits the CEM policy's latent window cache,
    action buffers, horizon countdown, un-normalization, and dynamics helpers
    (_dyn_step/_context_at_head/_append_latent); overrides only planning itself."""

    def __init__(self, model, cfg, *args, **kwargs):
        self.grad_steps = int(kwargs.pop("grad_steps", 50))
        self.grad_lr = float(kwargs.pop("grad_lr", 0.05))
        self.grad_H = int(kwargs.pop("grad_H", 5))
        self.grad_clip = float(kwargs.pop("grad_clip", 10.0))
        _ac = kwargs.pop("grad_action_clip", None)
        self.grad_action_clip = None if _ac in (None, "", "none", "None") else float(_ac)
        _dm = str(kwargs.pop("grad_dyn_mode", "auto"))
        self.grad_dyn_mode = (("prefix" if getattr(model, "use_prefix", False) else "rollout")
                              if _dm == "auto" else _dm)
        self.grad_exec_full = bool(kwargs.pop("grad_exec_full", True))
        self.grad_warm = bool(kwargs.pop("grad_warm", True))
        # full-traj ragged horizons: per-env H from the steps_left countdown (which build_policy
        # seeds with the per-episode offsets); plans are padded to the batch max and each env's
        # terminal is gathered at its own H_i
        self.grad_H_auto = bool(kwargs.pop("grad_H_auto", False))
        # closed-loop full-traj (user-approved K=10, option A): execute K blocks per replan;
        # each replan plans to the trajectory END with H_t = h0 - executed (data + arithmetic,
        # tracked exactly by the steps_left countdown). 0 = disabled.
        self.grad_exec_k = int(kwargs.pop("grad_exec_k", 0))
        # action-space trust region: after each step, project U onto ||U - U0||_F <= grad_tr
        # (flattened plan norm; 0 = off). Separates "narrow search" from "on-manifold search".
        self.grad_tr = float(kwargs.pop("grad_tr", 0.0))
        super().__init__(model, cfg, *args, **kwargs)
        self.type = "lewam_unified_grad" + ("" if self.grad_warm else "_cold")
        # diagnostics: per-replan planning records + promise-vs-delivery across replans
        self._diag = []
        self._diag_prev = {}  # env -> (call, promised terminal cost of the executed plan)

    def dump_diag(self, path):
        import numpy as np
        rows = [r for r in self._diag if r[0] == "plan"]
        real = [r for r in self._diag if r[0] == "realized"]
        np.savez(path,
                 plan=np.array([r[1:] for r in rows], dtype=np.float64),      # call, env, h_left, c0, cb, best_iter, dU
                 realized=np.array([r[1:] for r in real], dtype=np.float64))  # call, env, promised, realized
        if getattr(self, "_delta_store", None):
            first = min(self._delta_store)
            envs_d, dl = self._delta_store[first]
            np.savez(path.replace(".npz", "_delta.npz"), envs=np.array(envs_d), delta=dl)
        if getattr(self, "_plan_store", None):
            first = min(self._plan_store)
            envs, u0, ub = self._plan_store[first]
            np.savez(path.replace(".npz", "_plans.npz"),
                     envs=np.array(envs), U0=u0, Ubest=ub,
                     astd=self._astd.cpu().numpy(), amean=self._amean.cpu().numpy())
        print(f"[graddiag] {len(rows)} plan rows, {len(real)} realized rows -> {path}")

    def _env_H(self, replan_envs):
        """Per-env plan lengths (blocks) and their batch max."""
        R = len(replan_envs)
        if self.grad_H_auto:
            if self.grad_exec_k > 0:
                # closed-loop: remaining horizon = h0 - executed, carried by steps_left (arithmetic)
                H_env = np.maximum(np.ceil(np.maximum(self._steps_left[replan_envs], 1.0)), 1).astype(int)
            else:
                # e2e: the episode's OWN horizon from the dataset, immutable -- never guessed
                H_env = np.array([max(int(np.ceil(_h0_at(self.horizon0, i))), 1) for i in replan_envs])
        else:
            H_env = np.full(R, self.grad_H, dtype=int)
        return H_env, int(H_env.max())

    def _terminal_cost_ragged(self, window0, win_len0, U, z_goal, H_env, H_max):
        """Rollout-mode terminal cost with per-env horizons: roll H_max blocks, gather each
        env's terminal latent at its own H_i."""
        device = window0.device
        window, win_len = window0.clone(), win_len0.clone()
        z_steps = []
        for h in range(H_max):
            context = self._context_at_head(window, win_len)
            z_next = self._dyn_step(context, U[:, h], z_goal)
            z_steps.append(z_next)
            window, win_len = self._append_latent(window, win_len, z_next, int(self.ctx_cap))
        Z = torch.stack(z_steps, dim=0)                                   # (H_max, R, D)
        idx = torch.as_tensor(H_env - 1, device=device, dtype=torch.long)
        z_term = Z[idx, torch.arange(Z.shape[1], device=device)]          # (R, D)
        cost = ((z_term - z_goal) ** 2).mean(-1)
        return cost

    def _terminal_cost(self, window0, win_len0, anchor, U, z_goal):
        """Differentiable terminal cost (R,). window0/win_len0/anchor/z_goal are constants
        (detached); the graph flows through U only."""
        if self.grad_dyn_mode == "prefix":
            goal_in = z_goal if self.model.dynamics.goal_cond else None
            pred = self.model.dynamics(anchor, U, goal_in)          # (R, H, D)
            z_term = pred[:, -1]
        else:
            window, win_len = window0.clone(), win_len0.clone()
            z_term = None
            for h in range(self.grad_H):
                context = self._context_at_head(window, win_len)
                z_term = self._dyn_step(context, U[:, h], z_goal)
                window, win_len = self._append_latent(window, win_len, z_term, int(self.ctx_cap))
        return ((z_term - z_goal) ** 2).mean(-1)

    def get_action(self, info_dict, **kwargs):
        # NOT @torch.no_grad: the refinement needs the action->dynamics->cost graph.
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
                assert "goal" in info_dict, "unified_grad needs info_dict['goal']"
                goal_px = info_dict["goal"][replan_envs]
                goal_px = goal_px[:, -1] if goal_px.ndim == 5 else goal_px
                cur_px = cur_px[:, -1] if cur_px.ndim == 5 else cur_px
                z_cur = self.model.encode(cur_px.to(device).float())
                z_goal = self.model.encode(goal_px.to(device).float())
                for row, env_i in enumerate(replan_envs):
                    self._lat_buf[env_i].append(z_cur[row])
                    prev = self._diag_prev.pop(env_i, None)
                    if prev is not None:  # promise vs delivery for the plan just executed
                        realized = float(((z_cur[row] - z_goal[row]) ** 2).mean())
                        self._diag.append(("realized", prev[0], env_i, prev[1], realized))
                if self.log_latents:
                    for row, env_i in enumerate(replan_envs):
                        self._lat_log.append((self._call, int(env_i),
                                              z_cur[row].detach().cpu(), z_goal[row].detach().cpu()))

                n_replan, latent_dim = z_cur.shape
                ctx_cap = int(self.ctx_cap)
                window0 = torch.zeros(n_replan, ctx_cap, latent_dim, device=device)
                win_len0 = torch.empty(n_replan, dtype=torch.long, device=device)
                for row, env_i in enumerate(replan_envs):
                    real = torch.stack(self._lat_buf[env_i][-ctx_cap:], dim=0)
                    window0[row, : real.shape[0]] = real
                    win_len0[row] = real.shape[0]
                anchor = self._context_at_head(window0, win_len0) if self.grad_dyn_mode == "prefix" else None

                H_env, H_max_b = self._env_H(replan_envs)
                if self.grad_warm:
                    # same AR warm start as unified_cem's cem_warm
                    steps_left = np.maximum(self._steps_left[replan_envs], 1.0).astype(np.float64)
                    window, win_len, warm_blocks = window0.clone(), win_len0.clone(), []
                    for _h in range(H_max_b):
                        horizon_norm = torch.tensor(np.minimum(steps_left, self.H_max) / self.H_max,
                                                    device=device, dtype=torch.float32)
                        if self.ablate_horizon:
                            horizon_norm = torch.zeros_like(horizon_norm)
                        context = self._context_at_head(window, win_len)
                        action_blk = self.model.gc_head.point(self.model.gc_head(context, z_goal, horizon_norm))
                        warm_blocks.append(action_blk)
                        window, win_len = self._append_latent(window, win_len,
                                                              self._dyn_step(context, action_blk, z_goal), ctx_cap)
                        steps_left = np.maximum(steps_left - 1.0, 1.0)
                    U0 = torch.stack(warm_blocks, dim=1)             # (R, H, block_dim)
                else:
                    U0 = torch.zeros(n_replan, H_max_b, self.block_dim, device=device)

            with torch.enable_grad():
                U = U0.detach().clone().requires_grad_(True)
                opt = torch.optim.Adam([U], lr=self.grad_lr)
                best_U, best_cost = U0.detach().clone(), None
                c0 = None
                best_iter = torch.zeros(n_replan, dtype=torch.long)

                def _track_best(cost, k):
                    nonlocal best_U, best_cost, c0
                    c = cost.detach()
                    if best_cost is None:
                        best_cost = c.clone()
                        best_U = U.detach().clone()
                        c0 = c.clone()
                    else:
                        m = c < best_cost
                        best_U[m] = U.detach()[m]
                        best_cost = torch.where(m, c, best_cost)
                        best_iter[m.cpu()] = k

                def _cost_of(U_):
                    if self.grad_H_auto:
                        return self._terminal_cost_ragged(window0, win_len0, U_, z_goal, H_env, H_max_b)
                    return self._terminal_cost(window0, win_len0, anchor, U_, z_goal)
                for _k in range(self.grad_steps):
                    cost = _cost_of(U)
                    _track_best(cost, _k)
                    opt.zero_grad(set_to_none=True)
                    U.grad = torch.autograd.grad(cost.mean(), U)[0]
                    torch.nn.utils.clip_grad_norm_([U], self.grad_clip)
                    opt.step()
                    if self.grad_tr > 0:
                        with torch.no_grad():
                            dUp = (U - U0).flatten(1)
                            n = dUp.norm(dim=1).clamp_min(1e-9)
                            sc = (self.grad_tr / n).clamp(max=1.0)
                            U.copy_(U0 + (dUp * sc.unsqueeze(1)).view_as(U))
                    if self.grad_action_clip is not None:
                        with torch.no_grad():
                            U.clamp_(-self.grad_action_clip, self.grad_action_clip)
                with torch.no_grad():
                    _track_best(_cost_of(U), self.grad_steps)
                plan = best_U
                with torch.no_grad():
                    if not hasattr(self, "_plan_store"):
                        self._plan_store = {}
                    if self._call not in self._plan_store:
                        self._plan_store[self._call] = (
                            [int(e) for e in replan_envs],
                            U0.detach().cpu().numpy().copy(),
                            plan.detach().cpu().numpy().copy())
                    dU = (plan - U0).flatten(1).norm(dim=1)
                    for row, env_i in enumerate(replan_envs):
                        self._diag.append(("plan", self._call, env_i, float(self._steps_left[env_i]),
                                           float(c0[row]), float(best_cost[row]),
                                           int(best_iter[row]), float(dU[row])))
                        self._diag_prev[env_i] = (self._call, float(best_cost[row]))

            with torch.no_grad():
                Hp = plan.shape[1]
                plan_raw = (plan.reshape(n_replan, Hp, self.frameskip, self.raw_adim)
                            * self._astd + self._amean)
                plan_raw = plan_raw.reshape(n_replan, Hp * self.action_block, self.action_dim).cpu()
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
