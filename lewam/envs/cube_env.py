"""Add an end-effector term to OGBench cube's success predicate (lewam side).

OGBench cube success (stable_worldmodel/envs/ogbench/cube_env.py:_compute_successes) is
object-position-only: the cube COM within 0.04 m of its target, gripper-agnostic. Under the
goal-reaching protocol the goal frame is only goal_offset_steps ahead, so on many episodes the
cube has not been displaced yet and a do-nothing / random policy already sits at the goal (the
measured chance floor is ~40 on cube GR). Adding an end-effector term removes those trivial
successes:
    success = object_success AND ||effector - goal_effector|| < eef_threshold
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from stable_worldmodel.envs.ogbench.cube_env import CubeEnv


def set_goal_effector(self, goal_effector=None):
    """Goal callable: store the goal-frame end-effector position. None (arg absent because the
    goal column was not loaded) leaves the predicate object-only."""
    self._goal_eef = (None if goal_effector is None
                      else np.asarray(goal_effector, dtype=np.float64).reshape(-1)[:3])


_ORIG_INIT = CubeEnv.__init__
_ORIG_POST_STEP = CubeEnv.post_step
_ORIG_RESET = CubeEnv.reset


def _init(self, *args, eef_threshold: float = float("inf"), **kwargs):
    self._eef_threshold = float(eef_threshold)
    _ORIG_INIT(self, *args, **kwargs)


def _post_step(self):
    _ORIG_POST_STEP(self)                       # sets self._success (object-only, all cubes)
    goal_eef = getattr(self, "_goal_eef", None)
    if self._success and goal_eef is not None and np.isfinite(self._eef_threshold):
        cur = np.asarray(self._data.site_xpos[self._pinch_site_id][:3], dtype=np.float64)
        self._eef_dist = float(np.linalg.norm(cur - goal_eef))
        if self._eef_dist >= self._eef_threshold:
            self._success = False


def _reset(self, *args, **kwargs):
    self._goal_eef = None
    return _ORIG_RESET(self, *args, **kwargs)


CubeEnv.__init__ = _init
CubeEnv.set_goal_effector = set_goal_effector
CubeEnv.post_step = _post_step
CubeEnv.reset = _reset
