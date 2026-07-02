"""Thin entry: register the RoboMimic env, then run eval.py unchanged.
robomimic_env import has the side effect of gym.register('swm/RoboMimic-v0').
runpy executes eval.py as __main__ in THIS process (registration persists) so
Hydra resolves its config_path relative to eval.py correctly.
"""
import lewam.envs.robomimic_env as robomimic_env  # noqa: F401  -- registers swm/RoboMimic-v0 + robomimic ObsUtils
import runpy
runpy.run_path("eval.py", run_name="__main__")
