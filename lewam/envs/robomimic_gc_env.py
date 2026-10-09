"""swm env-layer plugin: goal-conditioned RoboMimic (robosuite) env for the WF8 protocol.

Scores goal-reaching against a dataset state at start+goal_offset, mirroring DexMimicGenEnv's
predicate (object positions AND end-effector within threshold) so the manipulation
families stay comparable; with no goal set, success is the task predicate.

Registers `swm/RoboMimicGC-v0`; task (ToolHang / PickPlaceCan / ...) via `world.task`.
"""
import json
import os
from collections.abc import Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
import mujoco
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
from gymnasium import spaces
from robomimic.utils.env_utils import create_env_from_metadata

# 9-D to match the converted dataset's proprio column; must equal the training width.
_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
_EEF = slice(0, 3)

ObsUtils.initialize_obs_modality_mapping_from_dict({
    "rgb": ["agentview_image"],
    "low_dim": _PROPRIO_KEYS,
})


class RoboMimicGCEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    _warned_no_ctrl = False

    def __init__(
        self, env_meta: str, task: str = "ToolHang", resolution: int = 224, goal_threshold: float = 0.15,
        eef_threshold: float = float("inf"), extra_views: Sequence[str] = (), **kwargs
    ):
        """
        Args:
            env_meta (str): robomimic env-metadata json of the task
            task (str): robomimic task
            resolution (int): rendered image size
            goal_threshold (float): max object distance to the goal state for goal success (m); 0.15 = robocasa's
            eef_threshold (float): max end-effector distance to the goal state's for goal success (m)
            extra_views (Sequence[str]): cameras rendered into the observation as pixels.<camera>
        """
        super().__init__()
        self.task = task
        self.res = int(resolution)
        self.goal_threshold = float(goal_threshold)
        self.eef_threshold = float(eef_threshold)
        self._goal_state = None
        self._goal_obj = None
        self._goal_eef = None
        self._start_state = None
        self._proprio_cache = None      # must exist before observation_space calls _proprio()

        with open(env_meta) as f:
            env_meta = json.load(f)
        import robosuite as _rs

        cc = env_meta.get("env_kwargs", {}).get("controller_configs")
        if isinstance(cc, dict) and "body_parts" not in cc and not _rs.__version__.startswith("1.4"):
            from robosuite.controllers import load_composite_controller_config
            rb = env_meta["env_kwargs"].get("robots") or ["Panda"]
            rb = rb[0] if isinstance(rb, (list, tuple)) else rb
            env_meta["env_kwargs"]["controller_configs"] = load_composite_controller_config(
                controller=None, robot=rb)
        self.env = create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True)

        adim = int(self.env.action_dimension)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(adim,), dtype=np.float32)
        sdim = int(np.asarray(self.env.get_state()["states"]).shape[0])
        self.extra_views = list(extra_views)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(sdim,), dtype=np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(len(self._proprio()),), dtype=np.float32),
            **{f"pixels.{c}": spaces.Box(0, 255, shape=(self.res, self.res, 3), dtype=np.uint8)
               for c in self.extra_views},
        })
        self.render_mode = "rgb_array"
        self._build_obj_slices()
        self._build_fk()
        self._model_xml = self.env.env.sim.model.get_xml()

    def _build_fk(self):
        """Private model+data pair for forward kinematics, so goal poses can be evaluated
        without moving the episode's simulator."""
        sim = self.env.env.sim
        self._fk_model = getattr(sim.model, "_model", sim.model)
        self._fk_data = mujoco.MjData(self._fk_model)
        rb = self.env.env.robots[0]
        sid = getattr(rb, "eef_site_id", None)
        if isinstance(sid, dict):           # robosuite 1.5 keys these per arm
            sid = sid.get("right", next(iter(sid.values())))
        if sid is None:
            sid = mujoco.mj_name2id(self._fk_model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        self._eef_site = int(sid)
        print(f"[robomimic-gc] fk eef site id={self._eef_site} "
              f"name={mujoco.mj_id2name(self._fk_model, mujoco.mjtObj.mjOBJ_SITE, self._eef_site)}",
              flush=True)

    # -- goal-state geometry --
    def _build_obj_slices(self):
        """Map task objects into the flattened state ([time]+qpos+qvel): free joints (7 qpos)
        are exactly the manipulable objects; arm joints are hinges (1 qpos)."""
        model = self.env.env.sim.model
        chosen = []
        for nm in list(getattr(model, "joint_names", [])):
            try:
                adr = model.get_joint_qpos_addr(nm)
            except Exception:
                continue
            if isinstance(adr, (tuple, list)) and (adr[1] - adr[0]) == 7:
                chosen.append((nm, int(adr[0])))
        self._obj_names = [nm for nm, _ in chosen]
        self._obj_pos_slices = [slice(1 + a, 1 + a + 3) for _, a in chosen]
        print(f"[robomimic-gc] task={self.task} goal objects={self._obj_names} "
              f"slices={[(s.start, s.stop) for s in self._obj_pos_slices]}", flush=True)

    def _obj_pos(self, state_vec):
        v = np.asarray(state_vec, np.float64).ravel()
        return np.stack([v[s] for s in self._obj_pos_slices])

    def _eef_at(self, state_vec):
        """End-effector position for a state via FK on a scratch MjData."""
        v = np.asarray(state_vec, np.float64).ravel()
        m, d = self._fk_model, self._fk_data
        d.qpos[:] = v[1:1 + m.nq]
        d.qvel[:] = v[1 + m.nq:1 + m.nq + m.nv]
        mujoco.mj_forward(m, d)
        return np.asarray(d.site_xpos[self._eef_site], np.float64).ravel().copy()

    def _get_goal_state(self):
        return self._goal_state

    def _set_goal_state(self, goal_state):
        g = np.asarray(goal_state, np.float64).ravel()
        self._goal_state = g
        self._goal_obj = self._obj_pos(g)
        self._goal_eef = self._eef_at(g)

    def _set_goal_proprio(self, goal_proprio):
        """No-op: the goal eef comes from FK (the dataset proprio column is degenerate)."""
        return

    def _state(self):
        return np.asarray(self.env.get_state()["states"], np.float32)

    def _proprio(self):
        """Cached per step: get_observation() re-renders every camera, and this is read
        more than once per step on an unmoved sim."""
        if self._proprio_cache is None:
            o = self.env.get_observation()
            self._proprio_cache = np.concatenate(
                [np.asarray(o[k], np.float32).ravel() for k in _PROPRIO_KEYS]).astype(np.float32)
        return self._proprio_cache

    def _eef_pos(self):
        return self._proprio().astype(np.float64)[_EEF]

    def _obs(self):
        obs = {"state": self._state(), "proprio": self._proprio()}
        for camera in self.extra_views:
            obs[f"pixels.{camera}"] = np.asarray(self.env.render(
                mode="rgb_array", height=self.res, width=self.res, camera_name=camera), np.uint8)
        return obs

    def _goal_distance(self):
        """MAX over task objects, plus the eef term when goal proprio was supplied."""
        if self._goal_obj is None:
            return float("inf"), float("inf")
        d = float(np.linalg.norm(self._obj_pos(self._state()) - self._goal_obj, axis=-1).max())
        e = float("inf")
        if self._goal_eef is not None:
            e = float(np.linalg.norm(self._eef_pos() - self._goal_eef))
        return d, e

    # -- gym --
    def reset(self, seed=None, options=None):
        self.env.reset()
        self._goal_state = self._goal_obj = self._goal_eef = None
        self._proprio_cache = None
        if options and options.get("state") is not None:
            self.set_state(options["state"])
        return self._obs(), {}

    def set_state(self, state):
        self._start_state = np.asarray(state, np.float64)
        # rebuilding from the captured xml buys determinism at the cost of an inexact get_xml round trip
        self.env.reset_to({"model": self._model_xml, "states": self._start_state})
        self._sync_controller()
        self._proprio_cache = None

    def _sync_controller(self):
        """reset_to restores qpos/qvel only; the OSC nullspace reference (initial_joint) keeps
        the pre-restore pose."""
        n = 0
        for rb in getattr(getattr(self.env, "env", None), "robots", []) or []:
            cands = []
            c = getattr(rb, "controller", None)
            if c is not None:
                cands.append(c)
            cc = getattr(rb, "composite_controller", None)
            if cc is not None:
                cands.append(cc)
                cands.extend((getattr(cc, "part_controllers", {}) or {}).values())
            for c in cands:
                if hasattr(c, "update_initial_joints"):
                    try:
                        c.update_initial_joints(rb._joint_positions)
                        n += 1
                    except Exception:
                        pass
        if n == 0 and not RoboMimicGCEnv._warned_no_ctrl:
            RoboMimicGCEnv._warned_no_ctrl = True
            print("[robomimic-gc] WARNING: no controller exposed update_initial_joints; "
                  "the nullspace-reference fix is NOT active", flush=True)

    def step(self, action):
        self._proprio_cache = None          # the sim is about to move
        _, _, _, info = self.env.step(np.asarray(action, np.float64))
        task_ok = bool(self.env.is_success()["task"])
        dist, eef = self._goal_distance()
        if self._goal_obj is None:
            terminated, reward = task_ok, 0.0
        else:
            terminated = (dist < self.goal_threshold and eef < self.eef_threshold) or task_ok
            reward = -dist
        info = dict(info or {})
        info["goal_distance"] = dist
        info["goal_eef_distance"] = eef
        info["task_success"] = task_ok
        return self._obs(), float(reward), terminated, False, info

    def render(self):
        return np.asarray(self.env.render(
            mode="rgb_array", height=self.res, width=self.res, camera_name="agentview"), np.uint8)


if "swm/RoboMimicGC-v0" not in gym.registry:
    gym.register(id="swm/RoboMimicGC-v0", entry_point=RoboMimicGCEnv)
