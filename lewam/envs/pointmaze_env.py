"""swm env-layer plugin: goal-conditioned OGBench PointMaze for the WF8 protocol.

Registers `swm/PointMaze-v0`; maze size via `world.task` ('medium'/'large'/'giant'/'teleport').
State replay is exact to float32 precision (3.4e-07), so the dataset for this cell is rendered
through this same class. OGBench's released observation column is a 2-D xy feature, not simulator
state -- datasets must be pulled with `add_info=True` for real qpos/qvel. (DINO-WM's point maze is
a D4RL maze2d rewrite, not OGBench; same physics, different data.)
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ogbench.locomaze.maze import make_maze_env

_DIST_LOG = os.environ.get("POINTMAZE_DIST_LOG", "")

# Suite-wide render size; cells whose data is not 224 override `resolution` in their eval config.
DEFAULT_RESOLUTION = int(os.environ.get("WF8_RESOLUTION", 224))


class PointMazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task="medium", resolution=None, goal_threshold=None, **kwargs):
        super().__init__()
        self.task = str(task)
        self.res = int(DEFAULT_RESOLUTION if resolution is None else resolution)
        self.env = make_maze_env(ob_type="states", loco_env_type="point",
                                 maze_env_type="maze", maze_type=self.task)
        u = self.env.unwrapped
        self.u = u
        self.nq, self.nv = int(u.model.nq), int(u.model.nv)
        self._configure_renderer()

        # Inherit OGBench's own tolerance; env var overrides for calibration sweeps.
        official = float(getattr(u, "_goal_tol", 0.5))
        self.goal_threshold = float(
            goal_threshold if goal_threshold is not None
            else os.environ.get("POINTMAZE_GOAL_THRESHOLD", official))
        # 1 = reach-anytime (official harness); >1 requires the predicate to hold K steps.
        self.hold_k = int(os.environ.get("GOAL_HOLD_K", 1))
        self._hold = 0
        self._goal_state = None
        self._goal_xy = None
        self._nstep = 0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(int(u.model.nu),), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(self.nq + self.nv,), dtype=np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(self.nq + self.nv,), dtype=np.float32),
        })
        self.render_mode = "rgb_array"
        print(f"[pointmaze-gc] maze={self.task} nq={self.nq} nv={self.nv} nu={u.model.nu} "
              f"frame_skip={getattr(u, 'frame_skip', '?')} goal_tol={self.goal_threshold} "
              f"(official {official})", flush=True)

    def _configure_renderer(self):
        """Render natively at self.res: both the OGBench renderer size and the model's offscreen
        framebuffer are pinned at 200, and mujoco.Renderer refuses to exceed the latter, so both
        must be raised; dropping custom_renderer forces a rebuild at the new size."""
        u = self.u
        u.model.vis.global_.offwidth = max(int(self.res), int(u.model.vis.global_.offwidth))
        u.model.vis.global_.offheight = max(int(self.res), int(u.model.vis.global_.offheight))
        u.width = u.height = int(self.res)
        u.custom_renderer = None

    # -- state ------------------------------------------------------------------
    def _state(self):
        """[qpos, qvel] -- the complete simulator state (2+2, no free joints/actuators)."""
        return np.concatenate([np.asarray(self.u.data.qpos, np.float32),
                               np.asarray(self.u.data.qvel, np.float32)])

    def _proprio(self):
        return self._state()

    def _xy(self, state_vec=None):
        if state_vec is None:
            return np.asarray(self.u.get_xy(), np.float64)
        return np.asarray(state_vec, np.float64).ravel()[:2]

    def _obs(self):
        s = self._state()
        return {"state": s, "proprio": s}

    # -- goal -------------------------------------------------------------------
    def _get_goal_state(self):
        return self._goal_state

    def _set_goal_state(self, goal_state):
        """Goal = agent position from the dataset state at start+goal_offset."""
        g = np.asarray(goal_state, np.float64).ravel()
        self._goal_state = g
        self._goal_xy = g[:2]

    def _set_goal_proprio(self, goal_proprio):
        """No-op: proprio == state here; kept so shared configs apply unchanged."""
        return

    def _goal_distance(self):
        if self._goal_xy is None:
            return float("inf")
        return float(np.linalg.norm(self._xy() - self._goal_xy))

    # -- gym contract -----------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        # Pin the OGBench marker to the model-default cell: the dataset renderer never resets, so
        # every training frame shows it there; letting reset move it would be a train/eval shift.
        try:
            self.u.set_goal(goal_xy=np.zeros(2))
        except Exception:
            pass
        self._goal_state = None
        self._goal_xy = None
        self._nstep = 0
        self._hold = 0
        if options and options.get("state") is not None:
            self.set_state(options["state"])
        return self._obs(), {}

    def set_state(self, state):
        s = np.asarray(state, np.float64).ravel()
        self.u.set_state(s[:self.nq].copy(), s[self.nq:self.nq + self.nv].copy())
        self._nstep = 0
        self._hold = 0

    def step(self, action):
        self.u.step(np.asarray(action, np.float64))
        dist = self._goal_distance()
        if self._goal_xy is None:
            terminated, reward = False, 0.0
        else:
            reached = dist < self.goal_threshold
            self._hold = self._hold + 1 if reached else 0
            terminated = self._hold >= self.hold_k
            reward = -dist
        info = {"goal_distance": dist, "goal_hold": self._hold}
        self._nstep += 1
        if _DIST_LOG:
            try:
                with open(_DIST_LOG, "a") as fh:
                    fh.write(f"{os.getpid()}_{id(self)}\t{self._nstep}\t{dist:.6f}\n")
            except Exception:
                pass
        return self._obs(), float(reward), bool(terminated), False, info

    _warned_resize = False

    def render(self):
        """Resize when the native render size differs from self.res. Interpolation is picked by
        direction (INTER_AREA degenerates when upsampling); a missing cv2 is a hard error."""
        f = np.asarray(self.u.render(), np.uint8)
        if f.shape[0] == self.res and f.shape[1] == self.res:
            return f
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                f"pointmaze renders {f.shape[1]}x{f.shape[0]} but resolution={self.res} was asked "
                f"for, and cv2 is not installed to resize it. Install opencv-python, or set "
                f"resolution={f.shape[0]}.") from exc
        interp = cv2.INTER_LINEAR if self.res > f.shape[0] else cv2.INTER_AREA
        if not PointMazeEnv._warned_resize:
            PointMazeEnv._warned_resize = True
            print(f"[pointmaze-gc] rendering {f.shape[1]}x{f.shape[0]} and resampling to "
                  f"{self.res}x{self.res} ({'up' if self.res > f.shape[0] else 'down'})", flush=True)
        return cv2.resize(f, (self.res, self.res), interpolation=interp)


if "swm/PointMaze-v0" not in gym.registry:
    gym.register(id="swm/PointMaze-v0", entry_point=PointMazeEnv)
