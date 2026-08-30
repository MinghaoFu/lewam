"""Add an end-effector term to OGBench cube's success predicate (lewam side).

OGBench cube success (stable_worldmodel/envs/ogbench/cube_env.py:_compute_successes) is
object-position-only: the cube COM within 0.04 m of its target, gripper-agnostic. Under the
goal-reaching protocol the goal frame is only goal_offset_steps ahead, so on many episodes the
cube has not been displaced yet and a do-nothing / random policy already sits at the goal (the
measured chance floor is ~40 on cube GR). Adding an end-effector term removes those trivial
successes:

    success = object_success AND ||effector - goal_effector|| < eef_threshold

This mirrors how dexmimicgen_env / robomimic_gc_env / pointmaze_env fold the goal-reaching
predicate into the env itself. Cube's env lives in stable_worldmodel (vendored OGBench), not in
this repo, so the term is installed by monkeypatching CubeEnv rather than editing a class here --
no new gym id, no "GC" variant: swm/OGBCube-v0 is still the env, now with the term. The eval
applies goal callables to the UNWRAPPED env (world.py:_evaluate_from_dataset), so the hook must
live on CubeEnv itself (a gym.Wrapper method is not reachable).

Goal effector is the goal-frame proprio_effector_pos, supplied per episode by the set_goal_effector
callable (configs/eval/cube.yaml). eef_threshold is read from CUBE_EEF_THRESHOLD and defaults to
+inf, so cube is byte-for-byte the object-only OGBench predicate unless that variable is set --
existing cube runs are unaffected. Importing this module installs the patch; eval_gip's
_register_env imports it for env_name swm/OGBCube-v0.
"""

import os

import numpy as np

from stable_worldmodel.envs.ogbench.cube_env import CubeEnv


def _eef_threshold():
    return float(os.environ.get("CUBE_EEF_THRESHOLD", "inf"))


def set_goal_effector(self, goal_effector=None):
    """Goal callable: store the goal-frame end-effector position. None (arg absent because the
    goal column was not loaded) leaves the predicate object-only."""
    self._goal_eef = (None if goal_effector is None
                      else np.asarray(goal_effector, dtype=np.float64).reshape(-1)[:3])


_ORIG_POST_STEP = CubeEnv.post_step
_ORIG_RESET = CubeEnv.reset


def _post_step(self):
    _ORIG_POST_STEP(self)                       # sets self._success (object-only, all cubes)
    thr = _eef_threshold()
    goal_eef = getattr(self, "_goal_eef", None)
    if self._success and goal_eef is not None and np.isfinite(thr):
        cur = np.asarray(self._data.site_xpos[self._pinch_site_id][:3], dtype=np.float64)
        self._eef_dist = float(np.linalg.norm(cur - goal_eef))
        if self._eef_dist >= thr:
            self._success = False


def _reset(self, *args, **kwargs):
    self._goal_eef = None
    return _ORIG_RESET(self, *args, **kwargs)


CubeEnv.set_goal_effector = set_goal_effector
CubeEnv.post_step = _post_step
CubeEnv.reset = _reset
