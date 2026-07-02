"""Honest single-env eval (ADDITIVE) for the history-aware BC policy.

Re-runs the SAME policy on the SAME eval start states as eval_histbc_robomimic.py,
but checks env.is_success() at EVERY step, so it can report both:
  latched   = is_success true at ANY step  (the harness convention -> should match the 66/92/64 numbers)
  final     = is_success true at the LAST step          (honest: did it END solved)
  held10    = is_success true for the last 10 steps      (stricter: solved and stable)

Single env, manual history-aware BC rollout (mirrors HistoryBCPolicy exactly).
Run:  python honest_eval_robomimic.py --config-name robomimic policy=gip_robomimic_can \
        world.task=PickPlaceCan dataset.stats=can eval.dataset_name=can eval.num_eval=20 eval.eval_budget=150
"""
import lewam.envs.robomimic_env as robomimic_env  # noqa
import stable_worldmodel.data.formats.hdf5  # noqa
import os
os.environ["MUJOCO_GL"] = "egl"
from collections import deque

import hydra
import numpy as np
import torch
from torchvision import tv_tensors

import lewam.models.gip as gip
from lewam.envs.robomimic_env import RoboMimicEnv


@hydra.main(config_path="./config/eval", config_name="robomimic", version_base=None)
def run(cfg):
    dev = "cuda"
    model, adim = gip.load_gip_model(cfg.policy)
    model = model.to(dev).eval(); model.requires_grad_(False); model.interpolate_pos_encoding = True
    HS = int(cfg.get("history_size", 3))
    ab = int(cfg.plan_config.action_block); adn = adim // ab

    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    process = gip.build_process(cfg, dataset)
    tfm = gip.img_transform(cfg)
    episodes, starts = gip.sample_eval_episodes(cfg, dataset)
    col = gip.episode_col(dataset)
    ep_idx = dataset.get_col_data(col); step_idx = dataset.get_col_data("step_idx")
    states = dataset.get_col_data("state")
    budget = int(cfg.eval.eval_budget)

    env = RoboMimicEnv(task=cfg.world.task); env.reset()
    ascaler = process["action"]

    def encode(frame):  # frame HWC uint8 -> (D,)
        x = tfm(tv_tensors.Image(np.transpose(frame, (2, 0, 1))))
        return model.encode({"pixels": x[None, None].to(dev).float()})["emb"][0, 0]

    rows = []
    for ep, st in zip(episodes, starts):
        m = (ep_idx == ep) & (step_idx == st)
        env.set_state(states[m][0])
        zhist = deque(maxlen=HS); ablk = deque(maxlen=HS - 1)
        succ = []  # is_success per env step
        steps = 0
        while steps < budget:
            z = encode(env.render()); zhist.append(z)
            emb = torch.stack(list(zhist))[None]; t = emb.size(1); D = emb.size(-1)
            past = torch.zeros(1, t, D, device=dev)
            for j, blk in enumerate(ablk):
                if j + 1 < t:
                    past[0, j + 1] = model.action_encoder(blk.view(1, 1, -1).to(dev))[0, 0]
            _, acts = model.predict_intention(emb, past)
            blk_new = acts[0, -1].detach().cpu(); ablk.append(blk_new)
            raw = ascaler.inverse_transform(blk_new.view(ab, adn).numpy())
            for a in raw:
                _, _, term, _, _ = env.step(a); succ.append(bool(term)); steps += 1
                if steps >= budget:
                    break
        succ = np.array(succ, dtype=bool)
        rows.append(dict(
            latched=succ.any(),
            final=bool(succ[-1]) if len(succ) else False,
            held10=bool(succ[-10:].all()) if len(succ) >= 10 else False,
            first=int(np.argmax(succ)) if succ.any() else -1,
            n=len(succ),
        ))

    def pct(k):
        return 100.0 * sum(r[k] for r in rows) / len(rows)
    nfirst = [r["first"] for r in rows if r["first"] >= 0]
    print(f"HONEST {cfg.world.task} (n={len(rows)}, budget={budget}): "
          f"latched={pct('latched'):.0f}%  final-step={pct('final'):.0f}%  held-last10={pct('held10'):.0f}%  "
          f"| of successes, mean first-success-step={np.mean(nfirst):.0f}" if nfirst else
          f"HONEST {cfg.world.task}: latched={pct('latched'):.0f}% final={pct('final'):.0f}% held10={pct('held10'):.0f}%")
    # explicit per-episode for auditing
    print("PEREP", [(r["first"], int(r["final"]), r["n"]) for r in rows])


if __name__ == "__main__":
    run()
