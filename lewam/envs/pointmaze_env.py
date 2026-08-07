"""swm env-layer plugin: goal-conditioned OGBench PointMaze for the WF8 protocol.

This is the navigation cell that reproduces its own demonstrations. Restore a recorded state, execute
the recorded actions, and the trajectory comes back at 3.4e-07 on step one and 4.4e-07 after fifty --
below one float32 ULP at the magnitude of these states, so identical to the precision the dataset was
stored at. Restore and self-replay are both exactly 0.000e+00, and the error does not grow.

Nothing else in the suite gets close. The manipulation cells reach 0.7-5 % of a natural step even with
robosuite and MuJoCo pinned to what their data records, antmaze sits at 2.7 %, and humanoidmaze at
311 % is not even deterministic against itself. The reason is structural and worth stating, because
it predicts which future cells will behave: a manipulation `env.step` runs 25 MuJoCo substeps through
an OSC controller that inverts a mass matrix every substep, which is the most version-sensitive thing
in the loop. PointMaze has `nu=2` and no controller at all -- the action is a force on a point mass,
and its only contacts are with walls.

Registers `swm/PointMaze-v0`; maze size via `world.task` ('medium' / 'large' / 'giant' / 'teleport').

Built on OGBench's maze, matching stable_worldmodel's own `scripts/data/collect_pointmaze.py`, which
collects through `swm/OGBMaze-v0` -- the same env behind a wrapper. Not to be confused with
`swm/SimplePointMaze-v0`, a pure-numpy 2-D toy with matplotlib rendering and `state + speed * action`
dynamics: exact because there is no physics in it, so nothing a world model would be tested on.

CORRECTION (2026-08-06): an earlier version of this docstring said DINO-WM's point maze traces to
OGBench as well. It does not. DINO-WM builds on **D4RL maze2d** -- `env/pointmaze/maze_model.py`
does `from d4rl import offline_env`, declares D4RL's `U_MAZE`/`MEDIUM_MAZE`/`LARGE_MAZE` layouts, and
generates its own point-mass XML. The two are independent rewrites of the same idea, and they agree
on the physics that matters here: `nq = nv = 2`, `frame_skip = 5`, a point mass whose only contacts
are walls. So the choice between them is about the DATA, not the dynamics -- and DINO-WM ships
rendered 224 pixels where OGBench ships none. See the README before building a pointmaze dataset.

The predicate is agent position alone: there is no object to place and no end-effector to pose, so
unlike the manipulation cells there is no second term and no free-start subtlety introduced by
objects that do not move. The default tolerance is read off OGBench's own `_goal_tol` rather than
invented, and can be overridden.

Data note: OGBench's released observation column is a 2-D xy *feature*, not simulator state. Building
a dataset from it and replaying against it would measure nothing. Pull with `add_info=True`, which
adds the real `qpos` and `qvel`.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ogbench.locomaze.maze import make_maze_env

_DIST_LOG = os.environ.get("POINTMAZE_DIST_LOG", "")


# One knob for the whole suite. Every WF8 env carries this same default and honours the same
# variable, so the render size is a property of the suite rather than of whichever env class you
# happened to construct. Cells whose data is not 224 override `resolution` in their eval config;
# see the resolution section of the README before changing it.
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

        # OGBench ships its own tolerance for this maze; inherit it rather than invent one, and let
        # an env var override for calibration sweeps -- the same pattern the manipulation cells use.
        official = float(getattr(u, "_goal_tol", 0.5))
        self.goal_threshold = float(
            goal_threshold if goal_threshold is not None
            else os.environ.get("POINTMAZE_GOAL_THRESHOLD", official))
        # Consecutive steps the predicate must hold before the episode counts. 1 is reach-anytime,
        # which is what the official harness does; above 1 rejects a trajectory that merely passes
        # through the goal region. See LIMITATIONS.md.
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
        """Render at `self.res` natively instead of rendering 200 and resampling.

        OGBench sizes its renderer from `width`/`height` (both 200) and mujoco.Renderer refuses to
        exceed the model's offscreen framebuffer, which the maze XML also pins at 200 --
        `ValueError: Image width 224 > framebuffer width 200`. Unlike robosuite, mujoco does not
        grow it on demand, so both have to be raised. Dropping `custom_renderer` forces OGBench to
        rebuild it at the new size on the next render.

        This matters beyond tidiness: the dataset for this cell is rendered through THIS class, so
        training pixels and evaluation frames come from one code path. Leaving the resize in place
        would have trained on 224 upsampled from 200 and evaluated on the same -- consistent, but
        needlessly soft, and one refactor away from becoming a domain gap.
        """
        u = self.u
        u.model.vis.global_.offwidth = max(int(self.res), int(u.model.vis.global_.offwidth))
        u.model.vis.global_.offheight = max(int(self.res), int(u.model.vis.global_.offheight))
        u.width = u.height = int(self.res)
        u.custom_renderer = None

    # -- state ------------------------------------------------------------------
    def _state(self):
        """[qpos, qvel] -- the whole simulator state. PointMaze has no free joints and no actuator
        activation, so these 2+2 numbers are the entire thing."""
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
        """The World hands over the dataset `state` column at start+goal_offset. For this cell the
        goal is the agent's position in it -- there is no object to place, so unlike the manipulation
        envs there is no second term, and no need to move the simulator to read anything."""
        g = np.asarray(goal_state, np.float64).ravel()
        self._goal_state = g
        self._goal_xy = g[:2]
        # Visualization-only: sync OGBench target marker (the pink sphere) to the conditioned
        # goal. GATED OFF by default on purpose -- training frames carry the collector-time marker
        # (uncorrelated with hindsight goals), so a marker sitting ON the goal at eval would be an
        # observation feature the model never trained with (shortcut + distribution shift). Scored
        # evals keep the env untouched; video capture exports POINTMAZE_SYNC_GOAL_MARKER=1.
        if os.environ.get("POINTMAZE_SYNC_GOAL_MARKER") == "1":
            try:
                self.u.set_goal(goal_xy=np.asarray(self._goal_xy, np.float64))
            except Exception:
                pass

    def _set_goal_proprio(self, goal_proprio):
        """No-op: proprio and state are the same 4 numbers here, and the goal is already set from the
        state. Kept so a shared config's callable list applies unchanged."""
        return

    def _goal_distance(self):
        if self._goal_xy is None:
            return float("inf")
        return float(np.linalg.norm(self._xy() - self._goal_xy))

    # -- gym contract -----------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        # Clear the previous episode's goal too, so a reused env cannot terminate against it.
        self._goal_state = None
        self._goal_xy = None
        self._nstep = 0
        self._hold = 0
        if options and options.get("state") is not None:
            self.set_state(options["state"])
        return self._obs(), {}

    def set_state(self, state):
        """Per-episode entry point. The maze geometry is fixed for a given size, so the flattened
        [qpos, qvel] restores the episode completely -- there is no per-demo model to replay."""
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
        """OGBench renders this maze at 200x200, so the suite's 224 default means a real resize.

        Two things that were wrong here and are worth keeping written down. INTER_AREA is a
        DOWNsampling filter -- correct going 224 -> 112, wrong going 200 -> 224, where it degenerates
        toward nearest-neighbour; pick by direction instead. And a missing cv2 used to be swallowed,
        returning a 200x200 frame that then tripped swm's EnsureImageShape with an error naming the
        wrong cause. Fail here, where the cause is visible.

        Note for building the dataset: because the native size is 200, `resolution: 224` upsamples.
        A pointmaze cell that is genuinely 224 needs OGBench's camera configured, not this resize.
        """
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
