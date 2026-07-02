"""Diagnostic (additive): for each eval start state, check env.is_success() at
step 0 (right after set_state, before any action). Quantifies how many 'successes'
are pre-solved start states (the latched-OR success convention then counts them)."""
import lewam.envs.robomimic_env as robomimic_env  # noqa
import stable_worldmodel.data.formats.hdf5  # noqa
import os
os.environ["MUJOCO_GL"] = "egl"
import hydra, numpy as np
import lewam.models.gip as gip
from lewam.envs.robomimic_env import RoboMimicEnv

@hydra.main(config_path="./config/eval", config_name="robomimic", version_base=None)
def run(cfg):
    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    episodes, starts = gip.sample_eval_episodes(cfg, dataset)
    col = gip.episode_col(dataset)
    ep_idx = dataset.get_col_data(col); step_idx = dataset.get_col_data("step_idx")
    states = dataset.get_col_data("state")
    env = RoboMimicEnv(task=cfg.world.task); env.reset()
    n0 = 0
    for ep, st in zip(episodes, starts):
        m = (ep_idx == ep) & (step_idx == st)
        env.set_state(states[m][0])
        s0 = bool(env.env.is_success()["task"]); n0 += s0
    print(f"RESULT {cfg.world.task}: {n0}/{len(episodes)} start states ALREADY is_success at step0 "
          f"| start_step range [{int(min(starts))},{int(max(starts))}]")

if __name__ == "__main__":
    run()
