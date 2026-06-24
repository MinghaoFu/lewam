"""History-GUIDED eval for RoboMimic (task #43): history-aware intuition warm-starts CEM.

The missing cell. eval_histbc runs the history-aware intuition head OPEN-LOOP (no
planning); the stock guided runs CEM warm-started by a SINGLE-FRAME intention (no
action memory -> contact state lost). This policy combines them: at each replan it
builds the head's proper training context (frame-latent history + executed-action
history), predicts the next action block as the CEM warm-start, then lets CEM refine
it toward the goal over the world model. The model is left NON-Actionable, so the
solver only fills the remaining horizon with zeros (our block is the informative seed).

Run:
  python eval_histguided_robomimic.py --config-name robomimic policy=gip_robomimic_lift \
      world.task=Lift dataset.stats=lift eval.dataset_name=lift \
      eval.num_eval=15 eval.goal_offset_steps=30 eval.eval_budget=80
"""
import robomimic_env  # noqa: F401
import stable_worldmodel.data.formats.hdf5  # noqa: F401
import os

os.environ["MUJOCO_GL"] = "egl"

import time
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate as hydra_instantiate

import stable_worldmodel as swm

import gip
from eval_histbc_robomimic import HistoryBCPolicy


class HistoryGuidedPolicy(HistoryBCPolicy):
    """History-aware intuition block warm-starts CEM, which refines toward the goal."""

    def __init__(self, model, action_block, action_dim, solver, plan_config,
                 history_size=3, process=None, transform=None, **kw):
        super().__init__(model, action_block, action_dim, history_size, process, transform, **kw)
        self.type = "history_guided"
        self.solver = solver
        self.plan_config = plan_config
        self.receding = int(getattr(plan_config, "receding_horizon", getattr(plan_config, "action_block", 1)))

    def set_env(self, env):
        super().set_env(env)
        n = getattr(env, "num_envs", 1)
        self.solver.configure(action_space=env.action_space, n_envs=n, config=self.plan_config)

    def _warm_block(self, i, cur_i, info_dict):
        """History-aware next action block for env i (frame + executed-action history)."""
        z = self.model.encode({"pixels": cur_i.unsqueeze(1)})["emb"][0, 0]
        self._zhist[i].append(z)
        emb = torch.stack(list(self._zhist[i]))[None]
        t, D = emb.size(1), emb.size(-1)
        past = torch.zeros(1, t, D, device=self.device)
        for j, blk in enumerate(self._ablk[i]):
            if j + 1 < t:
                past[0, j + 1] = self.model.action_encoder(blk.view(1, 1, -1).to(self.device))[0, 0]
        proprio_emb = None
        if getattr(self.model, "use_proprio", False) and "proprio" in info_dict:
            pr = info_dict["proprio"]
            if not torch.is_tensor(pr):
                pr = torch.as_tensor(pr)
            pr_cur = (pr[:, -1] if pr.ndim == 3 else pr)[i:i + 1].to(self.device).float()
            ped = self.model.proprio_encoder.patch_embed.weight.shape[1]
            if pr_cur.shape[-1] < ped:
                pr_cur = torch.nn.functional.pad(pr_cur, (0, ped - pr_cur.shape[-1]))
            pz = self.model.proprio_encoder(pr_cur.unsqueeze(1))[0, 0]
            self._phist[i].append(pz)
            proprio_emb = torch.stack(list(self._phist[i]))[None]
        tv = self.model._eval_task_vec(1, self.device)
        _, acts = self.model.predict_intention(emb, past, proprio_emb=proprio_emb, task_vec=tv)
        return acts[0, -1].detach()  # (Adim,) raw frameskip-stacked block

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._abuf[i].clear(); self._zhist[i].clear()
                    self._ablk[i].clear(); self._phist[i].clear()
        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        px = info_dict["pixels"]
        if not torch.is_tensor(px):
            px = torch.as_tensor(px)
        cur = (px[:, -1] if px.ndim == 5 else px).to(self.device).float()

        replan = [i for i in range(n) if not dead[i] and len(self._abuf[i]) == 0]
        if replan:
            Adim = self.action_block * self.action_dim
            warm = torch.zeros(len(replan), Adim, device=self.device)
            for row, i in enumerate(replan):
                warm[row] = self._warm_block(i, cur[i:i + 1], info_dict)
            # slice info to replan envs for the solver
            sliced = {}
            for k, v in info_dict.items():
                if torch.is_tensor(v):
                    sliced[k] = v[torch.as_tensor(replan)]
                elif isinstance(v, np.ndarray):
                    sliced[k] = v[replan]
                elif isinstance(v, list):
                    sliced[k] = [v[i] for i in replan]
                else:
                    sliced[k] = v
            # CEM refine: our history-aware block is the warm-start (1 block); solver zero-pads rest
            out = self.solver.solve(sliced, init_action=warm.unsqueeze(1))
            plan = out["actions"]  # (R, horizon, Adim)
            for row, i in enumerate(replan):
                blk = plan[row, 0].detach().cpu()
                self._ablk[i].append(blk)
                raw = blk.view(self.action_block, self.action_dim)
                self._abuf[i].extend(raw)

        action = torch.full((n, self.action_dim), float("nan"))
        for i in range(n):
            if not dead[i]:
                action[i] = self._abuf[i].popleft()
        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


@hydra.main(version_base=None, config_path="./config/eval", config_name="robomimic")
def run(cfg: DictConfig):
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": gip.img_transform(cfg), "goal": gip.img_transform(cfg)}
    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    process = gip.build_process(cfg, dataset)
    episodes, starts = gip.sample_eval_episodes(cfg, dataset)

    model, adim = gip.load_gip_model(cfg.policy, epoch=cfg.get("ckpt_epoch", None))
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if getattr(model, "task_proj", None) is not None:
        model.eval_task = cfg.eval.dataset_name

    # solver over the world model (model NON-Actionable -> our block seeds the warm-start)
    solver = hydra_instantiate(cfg.solver, model=model)
    pc = swm.PlanConfig(**cfg.plan_config)
    ab = int(cfg.plan_config.action_block)
    policy = HistoryGuidedPolicy(
        model=model, action_block=ab, action_dim=adim // ab, solver=solver,
        plan_config=pc, history_size=cfg.get("history_size", 3),
        process=process, transform=transform,
    )
    print(f"[HISTGUIDED] ready adim={adim} action_block={ab} HS={policy.HS}")

    results_path = Path(swm.data.utils.get_cache_dir(), "gip_eval", "histguided", cfg.policy)
    results_path.mkdir(parents=True, exist_ok=True)
    world.set_policy(policy)
    t0 = time.time()
    import numpy as _np
    _ne = world.num_envs
    _cbl = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    _succ = []
    for _c in range(0, len(episodes), _ne):
        _ep=list(episodes[_c:_c+_ne]); _st=list(starts[_c:_c+_ne]); _k=len(_ep)
        while len(_ep)<_ne: _ep.append(_ep[0]); _st.append(_st[0])
        _m = world.evaluate(dataset=dataset, start_steps=_st, goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget, episodes_idx=_ep, callables=_cbl, video=None)
        _succ.extend(list(_m["episode_successes"])[:_k])
        print(f"[GUIDED] chunk {_c//_ne+1}: cumulative {int(sum(_succ))}/{len(_succ)}", flush=True)
    metrics = {"success_rate": float(_np.mean(_succ)), "episode_successes": _np.array(_succ), "n": len(_succ)}
    print("==== HISTGUIDED RESULTS ====")
    print(metrics)
    with (results_path / "histguided_results.txt").open("a") as f:
        f.write(f"\nmetrics: {metrics}\ntime: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    run()
