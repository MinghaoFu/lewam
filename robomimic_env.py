"""swm env-layer plugin: RoboMimic (robosuite) closed-loop env for CEM planning.
Adapts robomimic's EnvRobosuite to the gymnasium contract the World/MegaWrapper
expect (reset/step/render + a `set_state` callable). One generic env; the task
(Lift / PickPlaceCan / NutAssemblySquare) is passed via `world.task`, exactly
like ReacherDMControl passes `world.task: qpos_match`. Additive only — does not
touch any existing file.
"""
import os, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import h5py
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.env_utils import create_env_from_metadata

_DEMO_ROOT = os.environ.get("ROBOMIMIC_RAW", "/mnt/data_nvme1/minghao.fu/robomimic")
_TASK_TO_DIR = {"Lift": "lift", "PickPlaceCan": "can", "NutAssemblySquare": "square", "ToolHang": "tool_hang", "TwoArmTransport": "transport"}

# robomimic requires the obs-modality mapping set once before any env produces obs
ObsUtils.initialize_obs_modality_mapping_from_dict({
    "rgb": ["agentview_image"],
    "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel"],
})

# proprio-as-token: 11-D vector matching scripts/convert_robomimic_h5.py PROPRIO_KEYS
_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel"]


class RoboMimicEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, task="Lift", resolution=224, **kwargs):
        super().__init__()
        self.task = task
        self.res = int(resolution)
        demo_dir = _TASK_TO_DIR.get(task, str(task).lower())
        raw = os.path.join(_DEMO_ROOT, demo_dir, "ph", "image_384_v15.hdf5")
        with h5py.File(raw, "r") as f:
            env_meta = json.loads(f["data"].attrs["env_args"])
        self.env = create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True)
        adim = int(self.env.action_dimension)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(adim,), dtype=np.float32)
        sdim = int(np.asarray(self.env.get_state()["states"]).shape[0])
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(sdim,), dtype=np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32),
        })
        self.render_mode = "rgb_array"

    def _state(self):
        return np.asarray(self.env.get_state()["states"], np.float32)

    def _proprio(self):
        o = self.env.get_observation()
        return np.concatenate([np.asarray(o[k], np.float32).ravel() for k in _PROPRIO_KEYS]).astype(np.float32)

    # per-stage instrumentation: when $GRASP_LOG is set, append one "task,grasped,success"
    # CSV line per finished episode.
    def _grasped(self):
        try:
            rs = self.env.env  # raw robosuite env under robomimic's EnvRobosuite
            g = rs.robots[0].gripper
            if isinstance(g, dict):  # robosuite 1.5 multi-arm mapping
                g = list(g.values())[0]
            if hasattr(rs, "cube"):                       # Lift
                return bool(rs._check_grasp(gripper=g, object_geoms=rs.cube))
            if getattr(rs, "objects", None):              # PickPlace (can)
                i = int(getattr(rs, "object_id", 0))
                return bool(rs._check_grasp(gripper=g, object_geoms=rs.objects[i]))
            if getattr(rs, "nuts", None):                 # NutAssembly (square)
                i = int(getattr(rs, "nut_id", 0))
                return bool(rs._check_grasp(gripper=g, object_geoms=rs.nuts[i]))
        except Exception:
            return False
        return False

    def _flush_ep(self):
        if (not getattr(self, "_ep_started", False) or getattr(self, "_ep_flushed", True)
                or getattr(self, "_ep_steps", 0) == 0):
            return
        self._ep_flushed = True
        with open(os.environ["GRASP_LOG"], "a") as f:
            f.write(f"{self.task},{int(self._ep_grasped)},{int(self._ep_success)}\n")

    def _dbg(self, tag):
        if os.environ.get("GRASP_DEBUG"):
            with open(os.environ["GRASP_DEBUG"], "a") as f:
                f.write(f"{tag} steps={getattr(self, '_ep_steps', -1)}\n")

    def _begin_ep(self):
        """Flush any pending episode and start fresh tracking. NOTE (verified by
        entry-point trace): the eval World builds ONE env instance PER EPISODE in
        parallel -- reset+set_state fire once at startup and never again, so the
        only flush points are terminated (success) and TEARDOWN (close/atexit).
        Failures are recovered exclusively by the teardown flush."""
        self._flush_ep()
        self._ep_started, self._ep_flushed = True, False
        self._ep_grasped, self._ep_success = False, False
        self._ep_steps = 0
        if not getattr(self, "_atexit_armed", False):
            import atexit
            atexit.register(self._flush_ep)
            self._atexit_armed = True

    def close(self):
        if os.environ.get("GRASP_LOG"):
            self._flush_ep()
        try:
            super().close()
        except Exception:
            pass

    def reset(self, seed=None, options=None):
        self._dbg("RESET")
        if os.environ.get("GRASP_LOG"):
            self._begin_ep()
        self.env.reset()
        if options and options.get("state") is not None:
            self.set_state(options["state"])
        return {"state": self._state(), "proprio": self._proprio()}, {}

    def set_state(self, state):
        # new-episode entry point; guard avoids double-begin when called from reset() (0 steps taken)
        self._dbg("SET_STATE")
        if os.environ.get("GRASP_LOG") and getattr(self, "_ep_steps", 0) > 0:
            self._begin_ep()
        self.env.reset_to({"states": np.asarray(state, np.float64)})

    def step(self, action):
        _, reward, _, info = self.env.step(np.asarray(action, np.float64))
        terminated = bool(self.env.is_success()["task"])
        if os.environ.get("GRASP_LOG") and getattr(self, "_ep_started", False):
            self._ep_steps = getattr(self, "_ep_steps", 0) + 1
            if not self._ep_grasped and self._grasped():
                self._ep_grasped = True
            if terminated:
                self._ep_success = True
                self._flush_ep()  # success ends the episode; flush now
        return {"state": self._state(), "proprio": self._proprio()}, float(reward), terminated, False, info

    def render(self):
        return np.asarray(self.env.render(
            mode="rgb_array", height=self.res, width=self.res, camera_name="agentview"), np.uint8)


gym.register(id="swm/RoboMimic-v0", entry_point=RoboMimicEnv)
