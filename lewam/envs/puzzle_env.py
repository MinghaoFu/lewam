"""OGBench puzzle under a swm-style id, `swm/OGBPuzzle-v0`.

stable_worldmodel vendors cube, scene and maze but not puzzle, so this points the id at OGBench's
own PuzzleEnv (its dynamics and rendering unchanged, as the data was collected with) and adds two
things: a goal callable that sets the full button configuration, and the end-effector success term
implemented locally.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np

import ogbench  # noqa: F401  registers puzzle-<size>-v0
from ogbench.manipspace.envs.puzzle_env import PuzzleEnv


ENV_ID = "swm/OGBPuzzle-v0"


def make_puzzle_env(env_type: str = "3x3", **kwargs) -> PuzzleEnv:
    """Build OGBench's puzzle-<env_type>-v0. swm's World passes `multiview`, which only its vendored
    envs accept, so it is dropped here."""
    kwargs.pop("multiview", None)
    return gym.make(f"puzzle-{env_type}-v0", env_type=env_type, **kwargs).unwrapped


def set_target_button_states(self, target_button_states=None):
    """Goal callable: make the goal frame's full button configuration the target. OGBench only
    exposes set_new_target (flips one random button, for data collection), while outside
    data-collection mode success needs every button to match the target."""
    if target_button_states is None:
        return
    self._target_button_states = np.rint(
        np.asarray(target_button_states, dtype=np.float64).reshape(-1)).astype(int)
    mujoco.mj_kinematics(self._model, self._data)


PuzzleEnv.set_target_button_states = set_target_button_states
def set_goal_effector(self, goal_effector=None):
    """Goal callable: store the goal-frame end-effector position. None (arg absent because the
    goal column was not loaded) leaves the predicate object-only."""
    self._goal_eef = (None if goal_effector is None
                      else np.asarray(goal_effector, dtype=np.float64).reshape(-1)[:3])


_ORIG_INIT = PuzzleEnv.__init__
_ORIG_POST_STEP = PuzzleEnv.post_step
_ORIG_RESET = PuzzleEnv.reset


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


PuzzleEnv.__init__ = _init
PuzzleEnv.set_goal_effector = set_goal_effector
PuzzleEnv.post_step = _post_step
PuzzleEnv.reset = _reset

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=make_puzzle_env, max_episode_steps=500)
