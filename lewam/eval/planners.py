"""Planning with LeWAM"""

import numpy as np
import torch

from lewam.eval.policy import LeWAMPolicy
from lewam.models.lewam import LeWAM


class LeWAMPlanner(LeWAMPolicy):
    """
    Goal-reaching planner (mode=lewam_plan): candidate plans are imagined through LeWAM's dynamics
    and scored by the distance of the imagined latent at the goal time to the goal latent; the winner's
    first `exec_actions_per_plan` actions are executed.

    Planning modes:
    - `best_of_k`: num_proposals policy rollouts (or random proposals), cheapest wins
    - `grad`: Adam on a plan through the differentiable rollout, warm-started from the cheapest policy
        rollout, or from all of them jointly (grad_all_k)
    - `steer`: Adam on a bias added to the goal latent the policy is conditioned on; the cost uses the
        true goal
    - `cem`: cross-entropy method around the cheapest policy rollout (cem_init=policy) or zero
    """

    def __init__(
        self, model: LeWAM, cfg: dict, *args, plan_mode: str = "best_of_k", num_proposals: int = 32,
        rollout_steps: int = 1, exec_actions_per_plan: int = 0, plan_random_candidates: bool = False,
        plan_goal_time: bool = True, grad_steps: int = 50, grad_lr: float = 0.05, grad_clip: float = 10.0,
        grad_tr: float = 0.0, grad_action_clip: float = 3.0, grad_all_k: bool = False, steer_steps: int = 20,
        steer_lr: float = 0.02, steer_max_bias: float = 0.3, steer_k: int = 0, steer_env_batch: int = 8,
        cem_iters: int = 3, cem_elites: int = 6, cem_std: float = 0.5, cem_init: str = "policy", **kwargs
    ):
        """
        Args:
            model (LeWAM): the trained model (requires dynamics)
            cfg (dict): LeWAM checkpoint config
            plan_mode (str): "best_of_k", "grad", "steer" or "cem"
            num_proposals (int): candidate plans per env
            rollout_steps (int): dynamics steps imagined per plan. <= 1 = the goal
                horizon at episode start
            exec_actions_per_plan (int): number of actions from the best plan executed before the next replan;
                0 = frameskip
            plan_random_candidates (bool): best_of_k control: N(0, 1) action sequences instead of policy
                rollouts
            plan_goal_time (bool): score each plan at its env's goal time instead of after the last step to
                avoid overshooting the goal
            grad_steps (int): Adam steps on the plan
            grad_lr (float): Adam learning rate for `grad`
            grad_clip (float): gradient norm clip; 0 = off
            grad_tr (float): trust-region weight on ||plan - warm start||^2; 0 = off
            grad_action_clip (float): bound on the z-scored actions; 0 = off
            grad_all_k (bool): refine every policy rollout jointly and take the best instead of only the
                initial lowest-cost candidate
            steer_steps (int): Adam steps on the goal bias
            steer_lr (float): Adam learning rate for `steer`
            steer_max_bias (float): cap on ||bias|| as a fraction of ||z_goal||
            steer_k (int): policy samples per env; 0 = num_proposals
            steer_env_batch (int): envs optimized together, to bound memory
            cem_iters (int): CEM rounds
            cem_elites (int): candidates refitting the mean and std each round
            cem_std (float): initial std
            cem_init (str): "policy" starts at the lowest-cost initial rollout, "zero" at zero actions
            *args, **kwargs: LeWAMPolicy's arguments
        """
        super().__init__(model, cfg, *args, **kwargs)
        assert plan_mode in ("best_of_k", "grad", "steer", "cem"), plan_mode
        assert cem_init in ("policy", "zero"), cem_init
        assert self.model.num_states_pred > 0, "planning needs imagined state tokens"
        assert plan_mode != "steer" or model.goal_cond, "steer needs a goal-conditioned checkpoint"
        assert self.goal_type != "terminal", "terminal-goal checkpoints are goal-conditioned BC and are not planned"
        self.plan_mode = plan_mode
        self.plan_goal_time = plan_goal_time
        self.type = f"lewam_plan_{plan_mode}"

        self.num_proposals = num_proposals
        self.rollout_steps = rollout_steps if rollout_steps > 1 else \
            max(1, int(round(float(np.max(np.asarray(self.h0, dtype=float))))))
        plan_length = self.rollout_steps * self.frameskip if self.rollout_steps > 1 else self.num_actions_pred
        self.exec_actions_per_plan = exec_actions_per_plan or self.frameskip
        assert self.frameskip <= self.exec_actions_per_plan <= plan_length, \
            f"exec_actions_per_plan {self.exec_actions_per_plan} must be in {self.frameskip}..{plan_length}"

        self.grad_steps, self.grad_lr, self.grad_clip = grad_steps, grad_lr, grad_clip
        self.grad_tr, self.grad_action_clip, self.grad_all_k = grad_tr, grad_action_clip, grad_all_k

        self.steer_steps, self.steer_lr, self.steer_max_bias = steer_steps, steer_lr, steer_max_bias
        self.steer_k, self.steer_env_batch = steer_k, steer_env_batch

        self.cem_iters, self.cem_elites, self.cem_std, self.cem_init = cem_iters, cem_elites, cem_std, cem_init

        self.plan_random_candidates = plan_random_candidates

        print(f"[plan] {plan_mode}: {self.rollout_steps} dynamics steps imagined per plan, "
              f"{self.exec_actions_per_plan} actions executed per replan", flush=True)

    def _next_latent(self, z_imag: torch.Tensor) -> torch.Tensor:
        """Each camera's imagined latent one dynamics step ahead"""
        if self.model.num_views == 1:
            return z_imag[:, :1]   # (B, 1, z_dim)
        return z_imag.view(z_imag.shape[0], self.model.num_views, self.model.num_states_pred, -1)[:, :, :1]  # (B, views, 1, z_dim)

    @staticmethod
    def _slide_history(history: torch.Tensor, history_pad: torch.Tensor, next_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slide the history window forward one dynamics step, dropping the oldest latent and adding the new one"""
        time_axis = history.dim() - 2
        history = torch.cat([history.narrow(time_axis, 1, history.shape[time_axis] - 1), next_latent], dim=time_axis)
        return history, torch.cat([history_pad[:, 1:], torch.zeros_like(history_pad[:, :1])], dim=1)

    def _goal_cost(self, z_at_goal: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Mean squared distance of the imagined latent at the goal step to the goal latent, averaged over the goal views."""
        if z_at_goal.dim() == 2:
            return ((z_at_goal - z_goal) ** 2).mean(-1)
        z_goal = z_goal[:, None] if z_goal.dim() == 2 else z_goal
        return ((z_at_goal[:, self.goal_view_indices] - z_goal) ** 2).mean(-1).mean(-1)

    def _propose(self, info_dict: dict, replan: list[int], history: torch.Tensor, history_pad: torch.Tensor) -> torch.Tensor:
        """The actions to execute for each replanning env; called by LeWAMPolicy.get_action at every replan."""
        z_goal = self._goal_latents(info_dict, replan)
        h_norm = steps_to_goal = None
        if self.model.goal_cond:
            steps_left = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps_left, self.H_max) / self.H_max, device=self.device, dtype=torch.float32)
            steps_to_goal = torch.tensor(steps_left, device=self.device, dtype=torch.float32)
        if self.plan_mode == "best_of_k":
            plan = self._best_of_k(history, history_pad, z_goal, h_norm, steps_to_goal)
        elif self.plan_mode == "grad":
            plan = self._gradient_plan(history, history_pad, z_goal, steps_to_goal)
        elif self.plan_mode == "steer":
            plan = self._steer_rollout(history, history_pad, z_goal, steps_to_goal)
        else:
            plan = self._cem_plan(history, history_pad, z_goal, steps_to_goal)
        return plan[:, :self.exec_actions_per_plan]

    def _goal_step_idx(self, steps_to_goal: torch.Tensor | None) -> torch.Tensor | None:
        """Index of the imagined state at each row's goal step, or None when the cost is taken at the last
        imagined state (plan_goal_time off, or no horizon)."""
        if not self.plan_goal_time or steps_to_goal is None:
            return None
        return steps_to_goal.round().clamp(min=1.0, max=float(self.rollout_steps)).long() - 1

    def _dyn_steps_needed(self, steps_to_goal: torch.Tensor | None) -> int:
        """Dynamics steps to imagine: up to the largest goal step, and enough to cover the executed actions."""
        if not self.plan_goal_time or steps_to_goal is None:
            return self.rollout_steps
        steps_to_last_goal = int(self._goal_step_idx(steps_to_goal).max().item()) + 1
        steps_executed = (self.exec_actions_per_plan + self.frameskip - 1) // self.frameskip
        return max(1, min(self.rollout_steps, max(steps_to_last_goal, steps_executed)))

    def _pred_state_at(self, pred_states_per_step: torch.Tensor, step_idx: torch.Tensor | None) -> torch.Tensor:
        """Each row's imagined state at its step_idx (None = the last step)."""
        if step_idx is None:
            return pred_states_per_step[:, -1]
        return pred_states_per_step[torch.arange(pred_states_per_step.shape[0], device=self.device), step_idx]

    def _goal_inputs(self, z_goal: torch.Tensor, steps_to_goal: torch.Tensor | None, step: int, num_rows: int) -> dict:
        """The policy's goal inputs `step` dynamics steps into an imagined rollout: the goal and the horizon left."""
        if not self.model.goal_cond:
            return {}
        if steps_to_goal is None:
            h_norm = torch.zeros(num_rows, device=self.device)
        else:
            h_norm = (steps_to_goal - step).clamp(min=1.0).clamp(max=float(self.H_max)) / float(self.H_max)
        return dict(z_goal=z_goal, h_norm=h_norm)

    def _rollout_noise(self, n_envs: int, actions_given: bool) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        """
        Action and state flow noise for every step of a full plan, drawn up front so the generator advances the
        same however many steps are imagined. With given actions an env's proposals share one draw, so their costs
        differ only by their actions; the action noise is then unused but still drawn, keeping the sequence fixed.
        """
        generator = self._flow_generator(self.device)
        num_rows = n_envs * self.num_proposals
        noise = []
        for _ in range(self.rollout_steps):
            if actions_given:
                action_noise = self.model.sample_action_noise(n_envs, self.device, generator).repeat_interleave(self.num_proposals, 0)
                state_noise = self.model.sample_state_noise(n_envs, self.device, generator).repeat_interleave(self.num_proposals, 0)
            else:
                action_noise = self.model.sample_action_noise(num_rows, self.device, generator)
                state_noise = self.model.sample_state_noise(num_rows, self.device, generator) \
                    if self.model.state_head == "flow" else None
            noise.append((action_noise, state_noise))
        return noise

    def _imagine_rollout(
        self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor,
        steps_to_goal: torch.Tensor | None, candidate_actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Imagine each proposal autoregressively: every dynamics step predicts the next latent from the history and
        frameskip actions, then slides it into the history. Actions are `candidate_actions` when given (e.g.,
        CEM candidates or random proposals), otherwise a policy sample on the imagined history.

        Args:
            history (torch.Tensor): (R * num_proposals, ...) history latents of the R replanning envs, each env's
                rows repeated num_proposals times in a row (row = env * num_proposals + proposal)
            history_pad (torch.Tensor): (R * num_proposals, history_len) padding mask
            z_goal (torch.Tensor): (R * num_proposals, z) or (R * num_proposals, goal views, z) goal latents
            steps_to_goal (torch.Tensor | None): (R * num_proposals,) dynamics steps left to the goal
            candidate_actions (torch.Tensor | None): (R * num_proposals, rollout_steps, frameskip, action_dim)
                plans to imagine

        Returns:
            actions (torch.Tensor): (R * num_proposals, n_dyn_steps, frameskip, action_dim) the actions of each step
            state_at_goal (torch.Tensor): (R * num_proposals, z) or (..., views, z) the imagined state at the goal
        """
        num_rows = history.shape[0]
        actions_given = candidate_actions is not None
        assert not actions_given or candidate_actions.shape[1] == self.rollout_steps, \
            "candidate_actions must cover the whole plan"
        n_dyn_steps = self._dyn_steps_needed(steps_to_goal)
        noise = self._rollout_noise(num_rows // self.num_proposals, actions_given)

        actions_per_step, latents_per_step = [], []
        for step in range(n_dyn_steps):
            action_noise, state_noise = noise[step]
            if actions_given:
                actions = candidate_actions[:, step]
            else:
                actions = self.model.predict_actions(
                    history, history_pad, action_noise=action_noise,
                    **self._goal_inputs(z_goal, steps_to_goal, step, num_rows)
                )[:, :self.frameskip]

            pred_states = self.model.predict_states(history, history_pad, actions, state_noise=state_noise)
            actions_per_step.append(actions)
            next_latent = self._next_latent(pred_states)
            latents_per_step.append(next_latent.squeeze(-2))
            history, history_pad = self._slide_history(history, history_pad, next_latent)

        state_at_goal = self._pred_state_at(torch.stack(latents_per_step, 1), self._goal_step_idx(steps_to_goal))
        return torch.stack(actions_per_step, 1), state_at_goal

    def _steer_rollout(self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor, steps_to_goal: torch.Tensor | None) -> torch.Tensor:
        """SteerMPC over the replanning envs."""
        n_envs = history.shape[0]
        env_batch = max(1, self.steer_env_batch)
        if self.model.state_head == "flow":
            env_batch = max(1, env_batch // 4)    # a flow head multiplies the passes per step in the compute graph
        # one plan length for all envs, so every env batch returns plans of the same length
        n_dyn_steps = self._dyn_steps_needed(steps_to_goal)

        plans = []
        for start in range(0, n_envs, env_batch):
            envs = slice(start, start + env_batch)
            plans.append(
                self._steer_plan(history[envs], history_pad[envs], z_goal[envs],
                None if steps_to_goal is None else steps_to_goal[envs], n_dyn_steps)
            )
        return torch.cat(plans, 0)

    def _imagine_with_goal_bias(
        self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor, steps_to_goal: torch.Tensor,
        goal_bias: torch.Tensor, action_noise: list[torch.Tensor], state_noise: list[torch.Tensor], n_dyn_steps: int,
        goal_step_idx: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Policy rollouts conditioned on the goal plus each env's bias, scored against the true goal.

        Returns:
            plans (torch.Tensor): (n_envs, n_samples, n_dyn_steps, frameskip, action_dim)
            cost (torch.Tensor): (n_envs, n_samples)
        """
        n_envs, num_rows = goal_bias.shape[0], history.shape[0]
        bias = goal_bias.repeat_interleave(num_rows // n_envs, 0)
        biased_goal = z_goal + (bias if z_goal.dim() == 2 else bias[:, None])     # the same bias on every goal view
        actions_per_step, latents_per_step = [], []

        for step in range(n_dyn_steps):
            actions = self.model.predict_actions(history, history_pad, action_noise=action_noise[step],
                                                 **self._goal_inputs(biased_goal, steps_to_goal, step, num_rows))[:, :self.frameskip]
            pred_states = self.model.predict_states(history, history_pad, actions, state_noise=state_noise[step])
            actions_per_step.append(actions)
            next_latent = self._next_latent(pred_states)
            latents_per_step.append(next_latent.squeeze(-2))
            history, history_pad = self._slide_history(history, history_pad, next_latent)

        state_at_goal = self._pred_state_at(torch.stack(latents_per_step, 1), goal_step_idx)
        cost = self._goal_cost(state_at_goal, z_goal).view(n_envs, -1)
        plans = torch.stack(actions_per_step, 1).view(n_envs, cost.shape[1], n_dyn_steps, self.frameskip, -1)
        return plans, cost

    def _steer_plan(self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor,
                    steps_to_goal: torch.Tensor | None, n_dyn_steps: int) -> torch.Tensor:
        """
        SteerMPC: Adam on a bias added to the goal latent the policy is conditioned on, through the flow sampler
        and the imagined rollout, with the noise fixed per replan. The cost uses the true goal. The unbiased plans
        (step 0) stay candidates; the cheapest (sample, step) plan seen is returned.
        """
        assert self.model.goal_cond, "steer needs a goal-conditioned policy"
        n_envs = history.shape[0]
        n_samples = self.steer_k if self.steer_k > 0 else self.num_proposals
        generator = self._flow_generator(self.device)
        if steps_to_goal is None:
            steps_to_goal = torch.full((n_envs,), float(self.rollout_steps), device=self.device)
        history_rep = history.repeat_interleave(n_samples, 0)
        pad_rep = history_pad.repeat_interleave(n_samples, 0)
        goal_rep = z_goal.repeat_interleave(n_samples, 0)
        steps_rep = steps_to_goal.repeat_interleave(n_samples, 0)
        action_noise = [self.model.sample_action_noise(n_envs * n_samples, self.device, generator)
                        for _ in range(self.rollout_steps)]
        state_noise = [self.model.sample_state_noise(n_envs * n_samples, self.device, generator)
                       for _ in range(self.rollout_steps)]
        goal_norm = z_goal.reshape(n_envs, -1).norm(dim=-1, keepdim=True)
        env_rows = torch.arange(n_envs, device=self.device)
        goal_step_idx = self._goal_step_idx(steps_rep)

        with torch.enable_grad():
            goal_bias = torch.zeros(n_envs, self.model.z_dim, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([goal_bias], lr=self.steer_lr)
            best_plan = best_cost = unbiased_cost = None
            for iteration in range(self.steer_steps + 1):
                plans, cost = self._imagine_with_goal_bias(history_rep, pad_rep, goal_rep, steps_rep, goal_bias,
                                                           action_noise, state_noise, n_dyn_steps, goal_step_idx)
                sample_cost, cheapest = cost.detach().min(1)
                sample_plan = plans.detach()[env_rows, cheapest]
                if best_cost is None:
                    best_cost, best_plan, unbiased_cost = sample_cost.clone(), sample_plan.clone(), sample_cost.clone()
                else:
                    better = sample_cost < best_cost
                    best_plan[better] = sample_plan[better]
                    best_cost = torch.where(better, sample_cost, best_cost)
                if iteration == self.steer_steps:
                    break
                grad = torch.autograd.grad(cost.mean(), goal_bias)[0]
                optimizer.zero_grad(set_to_none=True)
                goal_bias.grad = grad
                optimizer.step()
                with torch.no_grad():
                    scale = (self.steer_max_bias * goal_norm / goal_bias.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
                    goal_bias.mul_(scale)

        stats = self.__dict__.setdefault("_steer_stats", dict(n=0, improved=0, unbiased_cost=0.0, best_cost=0.0, bias_ratio=0.0))
        stats["n"] += n_envs
        stats["improved"] += int((best_cost < unbiased_cost).sum())
        stats["unbiased_cost"] += float(unbiased_cost.sum())
        stats["best_cost"] += float(best_cost.sum())
        stats["bias_ratio"] += float((goal_bias.detach().norm(dim=-1) / goal_norm.squeeze(-1).clamp_min(1e-9)).sum())
        if stats["n"] % 50 < n_envs:
            print(f"[steer] {stats['improved']}/{stats['n']} replans improved on the unbiased plan (best of {n_samples}); "
                  f"mean cost {stats['unbiased_cost'] / stats['n']:.4f} -> {stats['best_cost'] / stats['n']:.4f}; "
                  f"mean ||bias||/||z_goal|| {stats['bias_ratio'] / stats['n']:.3f} "
                  f"(steps {self.steer_steps}, lr {self.steer_lr}, max_bias {self.steer_max_bias})", flush=True)
        return best_plan.reshape(n_envs, best_plan.shape[1] * self.frameskip, -1)

    def _rollout_cost_differentiable(
        self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor,
        steps_to_goal: torch.Tensor | None, plan: torch.Tensor, state_noise: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Planning cost of plans (rows, steps, frameskip, action_dim), differentiable in the plans. state_noise:
        the per-step state noise of a flow state head, fixed per replan."""
        latents_per_step = []
        for step in range(plan.shape[1]):
            pred_states = self.model.predict_states(
                history, history_pad, plan[:, step],state_noise=None if state_noise is None else state_noise[step]
            )
            next_latent = self._next_latent(pred_states)
            latents_per_step.append(next_latent.squeeze(-2))
            history, history_pad = self._slide_history(history, history_pad, next_latent)
        state_at_goal = self._pred_state_at(torch.stack(latents_per_step, 1), self._goal_step_idx(steps_to_goal))
        return self._goal_cost(state_at_goal, z_goal)

    def _gradient_plan(
        self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor, steps_to_goal: torch.Tensor | None
    ) -> torch.Tensor:
        """Adam on the plan through the differentiable rollout, from the cheapest policy rollout (or from all of
        them with grad_all_k); returns the cheapest plan seen."""
        n_envs, num_proposals, fs = history.shape[0], self.num_proposals, self.frameskip
        env_rows = torch.arange(n_envs, device=self.device)
        with torch.no_grad():
            history_rep = history.repeat_interleave(num_proposals, 0)
            pad_rep = history_pad.repeat_interleave(num_proposals, 0)
            goal_rep = z_goal.repeat_interleave(num_proposals, 0)
            steps_rep = steps_to_goal.repeat_interleave(num_proposals, 0) if steps_to_goal is not None else None
            rollout_actions, state_at_goal = self._imagine_rollout(history_rep, pad_rep, goal_rep, steps_rep)
            rollout_cost = self._goal_cost(state_at_goal, goal_rep).view(n_envs, num_proposals)
            cheapest = rollout_cost.argmin(1)
            rollout_plans = rollout_actions.view(n_envs, num_proposals, rollout_actions.shape[1], fs, -1)
            warm_plan = rollout_plans[env_rows, cheapest].detach()
            warm_cost = rollout_cost[env_rows, cheapest].detach()
        if self.grad_steps <= 0:
            return warm_plan.reshape(n_envs, warm_plan.shape[1] * fs, -1)

        generator = self._flow_generator(self.device)
        if self.grad_all_k:
            init_plan = rollout_plans.reshape(n_envs * num_proposals, rollout_actions.shape[1], fs, -1).detach()
            init_cost = rollout_cost.view(n_envs * num_proposals)
            opt_history, opt_pad, opt_goal, opt_steps = history_rep, pad_rep, goal_rep, steps_rep
        else:
            init_plan, init_cost = warm_plan, warm_cost
            opt_history, opt_pad, opt_goal, opt_steps = history, history_pad, z_goal, steps_to_goal
            
        state_noise = [self.model.sample_state_noise(init_plan.shape[0], self.device, generator)
                       for _ in range(self.rollout_steps)]      # fixed per replan
        best_plan, best_cost = init_plan.clone(), init_cost.clone()
        plan = init_plan.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([plan], lr=self.grad_lr)
        with torch.enable_grad():
            for iteration in range(self.grad_steps + 1):
                cost = self._rollout_cost_differentiable(opt_history, opt_pad, opt_goal, opt_steps, plan, state_noise)
                with torch.no_grad():
                    better = cost < best_cost
                    best_cost = torch.where(better, cost.detach(), best_cost)
                    best_plan[better] = plan.detach()[better]
                if iteration == self.grad_steps:
                    break
                loss = cost.sum()
                if self.grad_tr > 0:
                    loss = loss + self.grad_tr * ((plan - init_plan) ** 2).sum()
                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([plan], self.grad_clip)
                optimizer.step()
                if self.grad_action_clip > 0:
                    with torch.no_grad():
                        plan.clamp_(-self.grad_action_clip, self.grad_action_clip)

        if self.grad_all_k:
            cost_per_proposal = best_cost.view(n_envs, num_proposals)
            cheapest = cost_per_proposal.argmin(1)
            best_plan = best_plan.view(n_envs, num_proposals, best_plan.shape[1], fs, -1)[env_rows, cheapest]
            best_cost = cost_per_proposal[env_rows, cheapest]

        stats = self.__dict__.setdefault("_grad_stats", dict(n=0, improved=0, warm_cost=0.0, best_cost=0.0, plan_change=0.0))
        stats["n"] += n_envs
        stats["improved"] += int((best_cost < warm_cost).sum())
        stats["warm_cost"] += float(warm_cost.sum())
        stats["best_cost"] += float(best_cost.sum())
        stats["plan_change"] += float((best_plan.reshape(n_envs, -1) - warm_plan.reshape(n_envs, -1)).norm(dim=1).sum())
        if stats["n"] % 50 < n_envs:
            print(f"[grad] {stats['improved']}/{stats['n']} replans improved on the warm start; mean cost "
                  f"{stats['warm_cost'] / stats['n']:.4f} -> {stats['best_cost'] / stats['n']:.4f}; mean plan change "
                  f"{stats['plan_change'] / stats['n']:.3f} (steps {self.grad_steps}, lr {self.grad_lr}, "
                  f"tr {self.grad_tr}, all_k {self.grad_all_k})", flush=True)
        return best_plan.detach().reshape(n_envs, best_plan.shape[1] * fs, -1)

    def _cem_plan(self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor,steps_to_goal: torch.Tensor | None) -> torch.Tensor:
        """Cross-entropy method: cem_iters rounds of num_proposals Gaussian plans around a running mean (one of
        them the mean itself), refit on the cem_elites cheapest; returns the cheapest plan seen."""
        n_envs, num_proposals, fs, rollout_steps = history.shape[0], self.num_proposals, self.frameskip, self.rollout_steps
        action_dim = self.model.action_raw_dim
        history_rep = history.repeat_interleave(num_proposals, 0)
        pad_rep = history_pad.repeat_interleave(num_proposals, 0)
        goal_rep = z_goal.repeat_interleave(num_proposals, 0)
        steps_rep = steps_to_goal.repeat_interleave(num_proposals, 0) if steps_to_goal is not None else None
        env_rows = torch.arange(n_envs, device=self.device)
        
        with torch.no_grad():
            if self.cem_init == "policy":
                rollout_actions, state_at_goal = self._imagine_rollout(history_rep, pad_rep, goal_rep, steps_rep)
                rollout_cost = self._goal_cost(state_at_goal, goal_rep).view(n_envs, num_proposals)
                cheapest = rollout_cost.argmin(1)
                mean = rollout_actions.view(n_envs, num_proposals, rollout_actions.shape[1], fs, action_dim)[env_rows, cheapest].clone()
                best_plan, best_cost = mean.clone(), rollout_cost[env_rows, cheapest].clone()
                warm_cost = best_cost.clone()
            else:
                mean = torch.zeros(n_envs, rollout_steps, fs, action_dim, device=self.device)
                best_plan, best_cost = mean.clone(), torch.full((n_envs,), float("inf"), device=self.device)

            if mean.shape[1] < rollout_steps:
                # candidates always span the whole plan, so the random draws match a full rollout;
                # the part past the imagined steps is never scored or executed
                unimagined = mean.new_zeros(n_envs, rollout_steps - mean.shape[1], fs, action_dim)
                mean, best_plan = torch.cat([mean, unimagined], 1), torch.cat([best_plan, unimagined], 1)

            std = torch.full_like(mean, self.cem_std)
            for _ in range(self.cem_iters):
                candidates = mean[:, None] + std[:, None] * torch.randn(n_envs, num_proposals, rollout_steps, fs, action_dim,
                                                                        device=self.device, generator=self._flow_generator(self.device))
                candidates[:, 0] = mean
                _, state_at_goal = self._imagine_rollout(history_rep, pad_rep, goal_rep, steps_rep,
                                                         candidate_actions=candidates.reshape(n_envs * num_proposals, rollout_steps, fs, action_dim))
                cost = self._goal_cost(state_at_goal, goal_rep).view(n_envs, num_proposals)
                round_cost, round_cheapest = cost.min(1)
                better = round_cost < best_cost
                best_plan[better] = candidates[env_rows, round_cheapest][better]
                best_cost = torch.minimum(round_cost, best_cost)
                elite_indices = cost.topk(min(self.cem_elites, num_proposals), dim=1, largest=False).indices
                elites = candidates[env_rows[:, None], elite_indices]
                mean, std = elites.mean(1), elites.std(1, correction=0).clamp_min(0.02)

        stats = self.__dict__.setdefault("_cem_stats", dict(n=0, warm_cost=0.0, best_cost=0.0))
        stats["n"] += n_envs
        stats["best_cost"] += float(best_cost.sum())
        if self.cem_init == "policy":
            stats["warm_cost"] += float(warm_cost.sum())
        if stats["n"] % 50 < n_envs:
            warm = f" (warm start {stats['warm_cost'] / stats['n']:.4f})" if self.cem_init == "policy" else ""
            print(f"[cem] init={self.cem_init} iters={self.cem_iters} elites={self.cem_elites} std0={self.cem_std}: "
                  f"mean best cost {stats['best_cost'] / stats['n']:.4f}{warm}", flush=True)
        return best_plan.reshape(n_envs, best_plan.shape[1] * fs, action_dim)

    def _repeated_goal_inputs(self, z_goal: torch.Tensor, h_norm: torch.Tensor | None, repeat: int) -> dict:
        """The policy's goal inputs with each env's row repeated `repeat` times."""
        if not self.model.goal_cond:
            return {}
        return dict(z_goal=z_goal.repeat_interleave(repeat, 0),
                    h_norm=h_norm.repeat_interleave(repeat, 0) if h_norm is not None else None)

    def _best_of_k(
        self, history: torch.Tensor, history_pad: torch.Tensor, z_goal: torch.Tensor,
        h_norm: torch.Tensor | None = None, steps_to_goal: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The cheapest of num_proposals policy rollouts, or of random action sequences with plan_random_candidates."""
        n_envs, num_proposals = history.shape[0], self.num_proposals
        history_rep = history.repeat_interleave(num_proposals, 0)
        pad_rep = history_pad.repeat_interleave(num_proposals, 0)
        goal_rep = z_goal.repeat_interleave(num_proposals, 0)
        env_rows = torch.arange(n_envs, device=self.device)
        if self.rollout_steps <= 1:
            # one dynamics step - score each predicted chunk by its last predicted state
            assert self.model.num_views == 1, "one-step scoring imagines one camera"
            generator = self._flow_generator(self.device)
            actions = self.model.predict_actions(history_rep, pad_rep, generator=generator,
                                                 **self._repeated_goal_inputs(z_goal, h_norm, num_proposals))
            pred_states = self.model.predict_states(history_rep, pad_rep, actions, generator=generator)
            cost = self._goal_cost(pred_states[:, -1], goal_rep).view(n_envs, num_proposals)
            return actions.view(n_envs, num_proposals, *actions.shape[1:])[env_rows, cost.argmin(1)]

        steps_rep = steps_to_goal.repeat_interleave(num_proposals, 0) if steps_to_goal is not None else None
        random_proposals = None
        if self.plan_random_candidates:
            random_proposals = torch.randn(
                n_envs * num_proposals, self.rollout_steps, self.frameskip, self.model.action_raw_dim,
                device=self.device, generator=self._flow_generator(self.device)
            )
        rollout_actions, state_at_goal = self._imagine_rollout(history_rep, pad_rep, goal_rep, steps_rep, candidate_actions=random_proposals)
        cost = self._goal_cost(state_at_goal, goal_rep).view(n_envs, num_proposals)
        plans = rollout_actions.reshape(n_envs, num_proposals, -1, rollout_actions.shape[-1])

        return plans[env_rows, cost.argmin(1)]
