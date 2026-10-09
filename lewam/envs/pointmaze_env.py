"""Goal-conditioned OGBench PointMaze, `swm/PointMaze-v0`.

The maze is picked by `task` ('medium' / 'large' / 'giant' / 'teleport'). The state is the full
simulator state [qpos, qvel] (2 + 2), and replaying it is exact to float32 precision, so the
dataset is rendered through this same class. Success is the agent within `goal_threshold`
(OGBench's own tolerance by default) of the goal frame's position.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ogbench.locomaze.maze import make_maze_env

ENV_ID = "swm/PointMaze-v0"


class PointMazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task: str = "medium", resolution: int = 224, goal_threshold: float | None = None,
                 **kwargs):
        """
        Args:
            task (str): OGBench maze type
            resolution (int): square render size, in pixels
            goal_threshold (float | None): success distance to the goal; None = OGBench's
        """
        super().__init__()
        self.task = str(task)
        self.res = int(resolution)
        self.env = make_maze_env(ob_type="states", loco_env_type="point", maze_env_type="maze",
                                 maze_type=self.task)
        self.u = self.env.unwrapped
        self.nq, self.nv = int(self.u.model.nq), int(self.u.model.nv)
        self._configure_renderer()
        self.goal_threshold = float(getattr(self.u, "_goal_tol", 0.5) if goal_threshold is None
                                    else goal_threshold)
        self._goal_state = None
        self._goal_xy = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(int(self.u.model.nu),), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(self.nq + self.nv,), dtype=np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(self.nq + self.nv,), dtype=np.float32),
        })
        self.render_mode = "rgb_array"

    def _configure_renderer(self):
        """Render natively at self.res: OGBench's renderer size and the model's offscreen framebuffer
        are both pinned at 200 and mujoco.Renderer refuses to exceed the latter, so both are raised;
        dropping custom_renderer forces a rebuild at the new size."""
        vis = self.u.model.vis.global_
        vis.offwidth, vis.offheight = max(self.res, int(vis.offwidth)), max(self.res, int(vis.offheight))
        self.u.width = self.u.height = self.res
        self.u.custom_renderer = None

    def _state(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.u.data.qpos, np.float32), np.asarray(self.u.data.qvel, np.float32)])

    def _obs(self) -> dict:
        state = self._state()
        return {"state": state, "proprio": state}

    def _get_goal_state(self):
        return self._goal_state

    def _set_goal_state(self, goal_state):
        """Goal callable: the agent position of the goal frame's state."""
        self._goal_state = np.asarray(goal_state, np.float64).ravel()
        self._goal_xy = self._goal_state[:2]

    def _set_goal_proprio(self, goal_proprio):
        """No-op (proprio == state here); kept so the shared goal callables apply unchanged."""

    def _goal_distance(self) -> float:
        if self._goal_xy is None:
            return float("inf")
        return float(np.linalg.norm(np.asarray(self.u.get_xy(), np.float64) - self._goal_xy))

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        # keep OGBench's goal marker at the model-default cell, where every training frame shows it
        self.u.set_goal(goal_xy=np.zeros(2))
        self._goal_state = None
        self._goal_xy = None
        if options and options.get("state") is not None:
            self.set_state(options["state"])
        return self._obs(), {}

    def set_state(self, state):
        state = np.asarray(state, np.float64).ravel()
        self.u.set_state(state[:self.nq].copy(), state[self.nq:self.nq + self.nv].copy())

    def step(self, action):
        self.u.step(np.asarray(action, np.float64))
        dist = self._goal_distance()
        terminated = self._goal_xy is not None and dist < self.goal_threshold
        reward = 0.0 if self._goal_xy is None else -dist
        return self._obs(), float(reward), bool(terminated), False, {"goal_distance": dist}

    def close(self):
        self.env.close()

    def render(self):
        frame = np.asarray(self.u.render(), np.uint8)
        assert frame.shape[:2] == (self.res, self.res), f"rendered {frame.shape[:2]}, expected {self.res}"
        return frame


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=PointMazeEnv)
