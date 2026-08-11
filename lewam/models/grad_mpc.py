"""Gradient MPC over the LeWAM-Unified dynamics (mode: unified_grad).

Warm-starts an H-block action plan from the reactive gc_head (the same AR rollout
unified_cem uses for cem_warm), then refines the plan with Adam against the frozen world
model: cost = ||z_terminal - z_goal||^2 through the dynamics. The model stays frozen
(requires_grad False); gradients flow only into the action sequence. The executed plan is
the best iterate by model cost, so it is never worse than the warm start under the model.
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
        super().__init__(model, cfg, *args, **kwargs)
        self.type = "lewam_unified_grad" + ("" if self.grad_warm else "_cold")
        # diagnostics: per-replan planning records + promise-vs-delivery across replans
        self._diag = []
        self._diag_prev = {}  # env -> (call, promised terminal cost of the executed plan)

    def dump_diag(self, path):
        import numpy as np
        rows = [r for r in self._diag if r[0] == "plan"]
        real = [r for r in self._diag if r[0] == "realized"]
        cand = [r for r in self._diag if r[0] == "cand"]
        np.savez(path,
                 plan=np.array([r[1:] for r in rows], dtype=np.float64),      # call, env, h_left, c0, cb, best_iter, dU
                 realized=np.array([r[1:] for r in real], dtype=np.float64),  # call, env, promised, realized
                 cand=np.array([r[1:] for r in cand], dtype=np.float64))      # call, env, winner, raw_winner, sm_U0, sm_win, raw_win
        print(f"[graddiag] {len(rows)} plan rows, {len(real)} realized rows -> {path}")

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

                if self.grad_warm:
                    # same AR warm start as unified_cem's cem_warm
                    steps_left = np.maximum(self._steps_left[replan_envs], 1.0).astype(np.float64)
                    window, win_len, warm_blocks = window0.clone(), win_len0.clone(), []
                    for _h in range(self.grad_H):
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
                    U0 = torch.zeros(n_replan, self.grad_H, self.block_dim, device=device)

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

                for _k in range(self.grad_steps):
                    cost = self._terminal_cost(window0, win_len0, anchor, U, z_goal)
                    _track_best(cost, _k)
                    opt.zero_grad(set_to_none=True)
                    U.grad = torch.autograd.grad(cost.mean(), U)[0]
                    torch.nn.utils.clip_grad_norm_([U], self.grad_clip)
                    opt.step()
                    if self.grad_action_clip is not None:
                        with torch.no_grad():
                            U.clamp_(-self.grad_action_clip, self.grad_action_clip)
                with torch.no_grad():
                    _track_best(self._terminal_cost(window0, win_len0, anchor, U, z_goal), self.grad_steps)
                plan = best_U                                        # (R, H, block_dim)
                with torch.no_grad():
                    dU = (plan - U0).flatten(1).norm(dim=1)
                    for row, env_i in enumerate(replan_envs):
                        self._diag.append(("plan", self._call, env_i, float(self._steps_left[env_i]),
                                           float(c0[row]), float(best_cost[row]),
                                           int(best_iter[row]), float(dU[row])))
                        self._diag_prev[env_i] = (self._call, float(best_cost[row]))

            with torch.no_grad():
                n_exec = self.grad_H if self.grad_exec_full else 1
                for h in range(n_exec):
                    block = plan[:, h]
                    raw = (block.reshape(n_replan, self.frameskip, self.raw_adim) * self._astd + self._amean)
                    raw = raw.reshape(n_replan, self.action_block, self.action_dim).cpu()
                    for row, env_i in enumerate(replan_envs):
                        self._action_buffer[env_i].extend(raw[row])
                for env_i in replan_envs:
                    self._steps_left[env_i] = max(self._steps_left[env_i] - float(n_exec), 1.0)

        action = torch.full((n_envs, self.action_dim), float("nan"))
        for env_i in range(n_envs):
            if not is_dead[env_i]:
                action[env_i] = self._action_buffer[env_i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()


class LeWAMUnifiedCandGradPolicy(LeWAMUnifiedGradPolicy):
    """Candidate planner (mode: unified_candgrad). Heterogeneous candidate pool per replan:
    the warm start U0 itself, early-stopped GD iterates, and noisy gc_head samples. All
    candidates are judged by a noise-SMOOTHED terminal cost (mean over m perturbed rollouts,
    common random numbers): thin fake minima that raw cost falls for blow up under the noise,
    wide honest basins survive. U0 sits in the pool, so the floor is 'do not optimize'."""

    def __init__(self, model, cfg, *args, **kwargs):
        snaps = str(kwargs.pop("cand_snapshots", "5,15,50"))
        self.cand_snapshots = sorted({int(x) for x in snaps.split(",") if x.strip()})
        self.cand_pol_K = int(kwargs.pop("cand_pol_K", 3))
        self.judge_noise = float(kwargs.pop("judge_noise", 0.3))
        self.judge_m = int(kwargs.pop("judge_m", 6))
        super().__init__(model, cfg, *args, **kwargs)
        self.type = "lewam_unified_candgrad"

    @staticmethod
    def _rep(t, k):
        return t.unsqueeze(1).expand(t.shape[0], k, *t.shape[1:]).reshape(t.shape[0] * k, *t.shape[1:])

    def get_action(self, info_dict, **kwargs):
        info_dict = self._prepare_info(info_dict)
        n_envs = self.env.num_envs
        device = next(self.model.parameters()).device
        self._call += 1
        if self._cem_gen is None:
            self._cem_gen = torch.Generator(device=device)
            self._cem_gen.manual_seed(self._cem_seed)
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
                anchor = self._context_at_head(window0, win_len0) if self.grad_dyn_mode == "prefix" else None

                def warm_rollout(k_samples, noise):
                    # AR gc_head rollout; k_samples noisy copies per env when noise, else 1 clean
                    b = R * k_samples
                    win = self._rep(window0, k_samples).clone()
                    wl = self._rep(win_len0, k_samples).clone()
                    zg = self._rep(z_goal, k_samples)
                    sl = np.repeat(np.maximum(self._steps_left[replan_envs], 1.0), k_samples)
                    blocks = []
                    for _h in range(self.grad_H):
                        hn = torch.tensor(np.minimum(sl, self.H_max) / self.H_max,
                                          device=device, dtype=torch.float32)
                        if self.ablate_horizon:
                            hn = torch.zeros_like(hn)
                        cctx = self._context_at_head(win, wl)
                        dist = self.model.gc_head(cctx, zg, hn)
                        if noise:
                            blk = self.model.gc_head.sample(dist, 1, noise=True,
                                                            generator=self._cem_gen).squeeze(1)
                        else:
                            blk = self.model.gc_head.point(dist)
                        blocks.append(blk)
                        win, wl = self._append_latent(win, wl, self._dyn_step(cctx, blk, zg), ctx_cap)
                        sl = np.maximum(sl - 1.0, 1.0)
                    return torch.stack(blocks, dim=1).reshape(R, k_samples, self.grad_H, self.block_dim)

                U0 = warm_rollout(1, noise=False)[:, 0]

            cands = [U0.detach().clone()]
            if self.grad_steps > 0:
                with torch.enable_grad():
                    U = U0.detach().clone().requires_grad_(True)
                    opt = torch.optim.Adam([U], lr=self.grad_lr)
                    for _k in range(self.grad_steps):
                        cost = self._terminal_cost(window0, win_len0, anchor, U, z_goal)
                        opt.zero_grad(set_to_none=True)
                        U.grad = torch.autograd.grad(cost.mean(), U)[0]
                        torch.nn.utils.clip_grad_norm_([U], self.grad_clip)
                        opt.step()
                        if (_k + 1) in self.cand_snapshots:
                            cands.append(U.detach().clone())
                if self.grad_steps not in self.cand_snapshots:
                    cands.append(U.detach().clone())
            with torch.no_grad():
                if self.cand_pol_K > 0:
                    pol = warm_rollout(self.cand_pol_K, noise=True)
                    for k in range(self.cand_pol_K):
                        cands.append(pol[:, k])
                C = len(cands)
                m = self.judge_m
                eps = torch.randn(R, m, self.grad_H, self.block_dim, device=device,
                                  generator=self._cem_gen) * self.judge_noise
                win_m = self._rep(window0, m)
                wl_m = self._rep(win_len0, m)
                zg_m = self._rep(z_goal, m)
                an_m = self._rep(anchor, m) if anchor is not None else None
                sm = torch.zeros(R, C, device=device)
                raw = torch.zeros(R, C, device=device)
                for c, Uc in enumerate(cands):
                    raw[:, c] = self._terminal_cost(window0, win_len0, anchor, Uc, z_goal)
                    Un = (self._rep(Uc, m) + eps.reshape(R * m, self.grad_H, self.block_dim))
                    sm[:, c] = self._terminal_cost(win_m, wl_m, an_m, Un, zg_m).reshape(R, m).mean(1)
                winner = sm.argmin(dim=1)
                raw_winner = raw.argmin(dim=1)
                rows = torch.arange(R, device=device)
                plan = torch.stack(cands, dim=1)[rows, winner]
                for row, env_i in enumerate(replan_envs):
                    w = int(winner[row])
                    self._diag.append(("cand", self._call, env_i, w, int(raw_winner[row]),
                                       float(sm[row, 0]), float(sm[row, w]), float(raw[row, w])))
                    self._diag_prev[env_i] = (self._call, float(raw[row, w]))
                n_exec = self.grad_H if self.grad_exec_full else 1
                for h in range(n_exec):
                    block = plan[:, h]
                    raw_a = (block.reshape(R, self.frameskip, self.raw_adim) * self._astd + self._amean)
                    raw_a = raw_a.reshape(R, self.action_block, self.action_dim).cpu()
                    for row, env_i in enumerate(replan_envs):
                        self._action_buffer[env_i].extend(raw_a[row])
                for env_i in replan_envs:
                    self._steps_left[env_i] = max(self._steps_left[env_i] - float(n_exec), 1.0)
        action = torch.full((n_envs, self.action_dim), float("nan"))
        for env_i in range(n_envs):
            if not is_dead[env_i]:
                action[env_i] = self._action_buffer[env_i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()
