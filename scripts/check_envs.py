"""Build every eval environment, reset it and take random steps: a check that the install runs all tasks.

Uses each task's eval config (configs/eval/<task>.yaml), plus swm's TwoRoom, which has no config here. Drawer and
transport read each episode's model xml from their dataset in $STABLEWM_HOME/datasets (see docs/DATA.md).
Usage: MUJOCO_GL=egl python scripts/check_envs.py [task ...]
"""

import sys
import time
import traceback
from pathlib import Path

import stable_worldmodel as swm
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_lewam import register_env  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "eval"
TASKS = sorted(p.stem for p in CONFIGS.glob("*.yaml") if p.stem != "base") + ([] if (CONFIGS / "tworoom.yaml").exists() else ["tworoom"])


def world_kwargs(task: str) -> dict:
    if task == "tworoom":
        return {"env_name": "swm/TwoRoom-v1"}
    cfg = OmegaConf.merge(OmegaConf.load(CONFIGS / "base.yaml"), OmegaConf.load(CONFIGS / f"{task}.yaml"))
    return OmegaConf.to_container(cfg.world, resolve=True)


def check(task: str, steps: int = 20) -> bool:
    start = time.time()
    world = None
    try:
        kwargs = world_kwargs(task)
        register_env(kwargs["env_name"])
        kwargs.update(num_envs=1, max_episode_steps=steps + 10)
        world = swm.World(**kwargs, image_shape=(224, 224))
        world.envs.reset(seed=0)
        for _ in range(steps):
            world.envs.step(world.envs.action_space.sample())
        print(f"OK    {task:10s} {kwargs['env_name']:40s} action {world.envs.action_space.shape}  {time.time() - start:.1f}s",
              flush=True)
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"FAIL  {task:10s} {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
        return False
    finally:
        if world is not None:
            world.envs.close()


if __name__ == "__main__":
    tasks = sys.argv[1:] or TASKS
    failed = [task for task in tasks if not check(task)]
    print(f"{len(tasks) - len(failed)}/{len(tasks)} environments OK" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
