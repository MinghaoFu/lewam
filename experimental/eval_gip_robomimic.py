"""Thin entry: register the RoboMimic env, then run eval_gip.py unchanged.
Same pattern as eval_robomimic.py but delegates to eval_gip.py so the GIP
bc/guided/planning modes run on robomimic. Override config + task on the CLI:
  python eval_gip_robomimic.py --config-name robomimic policy=gip_robomimic_lift \
      +gip_eval.mode=bc world.task=Lift dataset.stats=lift eval.dataset_name=lift
"""
import robomimic_env  # noqa: F401  -- registers swm/RoboMimic-v0 + robomimic ObsUtils
import runpy
runpy.run_path("eval_gip.py", run_name="__main__")
