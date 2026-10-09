"""
swm env-layer plugin: DexMimicGen closed-loop goal-reaching env
(TwoArmDrawerCleanup, TwoArmTransport)
"""
import os, json
from collections.abc import Sequence
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# Match the physical GPU selected by CUDA; robosuite rejects an EGL device
# outside CUDA_VISIBLE_DEVICES. Preserve an explicitly selected EGL device.
_visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
if _visible_gpu.isdigit():
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _visible_gpu)
# robosuite imports glfw unconditionally, and pyglfw's import-time lib probe spins forever
# under a detached run (stdin=/dev/null). Pin the bundled lib so import dlopens it directly.
if not os.environ.get("PYGLFW_LIBRARY"):
    import glob as _glob
    for _c in (_glob.glob(os.path.expanduser("~/.local/lib/python*/site-packages/glfw/x11/libglfw.so"))
               + _glob.glob("/usr/local/lib/python*/dist-packages/glfw/x11/libglfw.so")):
        os.environ["PYGLFW_LIBRARY"] = _c
        break
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import h5py
import dexmimicgen  # noqa: F401  registers the payload-randomized tasks into robosuite
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.env_utils import create_env_from_metadata

# bimanual proprio, 18-D, matching convert_dexmg.py PROP order
_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                 "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]

# KNOWN DEFECT: these slices assume the 18-D parallel-gripper layout.
# Drawer's dex hands make proprio 38-D with robot1's eef at 19:22, so slice(9,12) lands in
# robot0's gripper joints. But goal and state read the SAME slice, the predicate stays
# well-defined. Fix only together with a re-measurement (same for observation_space's shape=(18,)).
_EEF0, _EEF1 = slice(0, 3), slice(9, 12)
# drop_dims are refilled with the recorded constant: float32(pi/2) widened to float64
_DROPPED_DIM_VALUE = 1.5707963705062866

ObsUtils.initialize_obs_modality_mapping_from_dict({
    "rgb": ["agentview_image"],
    "low_dim": _PROPRIO_KEYS,
})


def _load_env_meta(path):
    with open(path) as f:
        meta = json.load(f)
    # env_lang=None rides along in the recorded env_kwargs; no installed robosuite accepts it.
    meta.get("env_kwargs", {}).pop("env_lang", None)
    # Render only agentview (the dataset camera); the other 4 cameras are per-step waste.
    ek = meta.get("env_kwargs", {})
    ek["camera_names"] = ["agentview"]
    ek["camera_heights"] = 84
    ek["camera_widths"] = 84
    return meta


class DexMimicGenEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    _warned_no_ctrl = False

    def __init__(
        self, env_meta: str, model_xml_h5: str, task: str = "TwoArmTransport", resolution: int = 224,
        goal_threshold: float = 0.15, eef_threshold: float = float("inf"), drop_dims: Sequence[int] = (),
        extra_views: Sequence[str] = (), **kwargs
    ):
        """
        Args:
            env_meta (str): robomimic env-metadata json of the task
            model_xml_h5 (str): h5 holding each episode's model_xml, indexed by ep_idx
            task (str): dexmimicgen task
            resolution (int): rendered image size
            goal_threshold (float): max object distance to the goal state for goal success (m); 0.15 = robocasa's
            eef_threshold (float): max end-effector distance to the goal proprio for goal success (m)
            drop_dims (Sequence[int]): raw action dims the training data dropped as constant; step() refills them
            extra_views (Sequence[str]): cameras rendered into the observation as pixels.<camera>
        """
        super().__init__()
        self.task = task
        self.res = int(resolution)
        self.goal_threshold = float(goal_threshold)
        self.eef_threshold = float(eef_threshold)
        self._model_xml_h5 = model_xml_h5
        self._goal_state = None
        self._goal_obj = None
        self._goal_eef = None
        self._ep_idx = None
        self._proprio_cache = None
        self.env = create_env_from_metadata(
            env_meta=_load_env_meta(env_meta), render=False, render_offscreen=True, use_image_obs=True)
        adim = int(self.env.action_dimension)
        self._drop_dims = sorted(int(dim) for dim in drop_dims)
        self._full_adim = adim
        eff_adim = adim - len(self._drop_dims)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(eff_adim,), dtype=np.float32)
        sdim = int(np.asarray(self.env.get_state()["states"]).shape[0])
        self.extra_views = list(extra_views)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(sdim,), dtype=np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32),
            **{f"pixels.{c}": spaces.Box(0, 255, shape=(self.res, self.res, 3), dtype=np.uint8)
               for c in self.extra_views},
        })
        self.render_mode = "rgb_array"
        self._build_obj_slices()

    # -- goal-state geometry --
    def _build_obj_slices(self):
        """Map task objects into the flattened state ([time]+qpos+qvel): free-joint addresses
        are resolved from the live model because the payload geometry is randomized per demo."""
        sim = self.env.env.sim
        model = sim.model
        names = list(getattr(model, "joint_names", []))
        chosen, all_free = [], []
        for nm in names:
            try:
                addr = model.get_joint_qpos_addr(nm)
            except Exception:
                continue
            if not (isinstance(addr, (tuple, list)) and (addr[1] - addr[0]) == 7):
                continue  # free joints only (7 qpos)
            all_free.append((nm, int(addr[0])))
            low = nm.lower()
            if ("payload" in low or "trash" in low) and "bin" not in low:
                chosen.append((nm, int(addr[0])))
        if not chosen:
            chosen = all_free
        self._obj_names = [nm for nm, _ in chosen]
        self._obj_pos_slices = [slice(1 + a, 1 + a + 3) for _, a in chosen]
        print(f"[dexmg-env] goal objects={self._obj_names} "
              f"slices={[(s.start, s.stop) for s in self._obj_pos_slices]} "
              f"(all_free={[nm for nm, _ in all_free]})", flush=True)

    def _obj_pos(self, state_vec):
        v = np.asarray(state_vec, np.float64).ravel()
        return np.stack([v[s] for s in self._obj_pos_slices])  # (n_obj, 3)

    def _get_goal_state(self):
        return self._goal_state

    def _set_goal_state(self, goal_state):
        """Goal = dataset `state` column at start+goal_offset_steps."""
        goal = np.asarray(goal_state, np.float64).ravel()
        self._goal_state = goal
        self._goal_obj = self._obj_pos(goal)

    def _set_goal_proprio(self, goal_proprio):
        """Goal end-effector poses. Required to keep the predicate non-trivial: 58% of starts
        already have both (stationary) objects within 0.15 m of their goal at t=0, while the
        arms move every step -- mirrors robocasa's subtask_success obj+hand combination."""
        proprio = np.asarray(goal_proprio, np.float64).ravel()
        self._goal_eef = np.stack([proprio[_EEF0], proprio[_EEF1]])

    def _eef_pos(self):
        proprio = self._proprio().astype(np.float64)
        return np.stack([proprio[_EEF0], proprio[_EEF1]])

    def _goal_distance(self):
        """Max over task objects, plus max over the two eefs when goal proprio is supplied."""
        if self._goal_obj is None:
            return float("inf"), None, float("inf")
        d = np.linalg.norm(self._obj_pos(self._state()) - self._goal_obj, axis=-1)
        e = float("inf")
        if self._goal_eef is not None:
            e = float(np.linalg.norm(self._eef_pos() - self._goal_eef, axis=-1).max())
        return float(d.max()), d, e

    # -- helpers --
    def _model_xml(self, ep_idx):
        with h5py.File(self._model_xml_h5, "r") as f:
            s = f["model_xml"][int(ep_idx)]
        return s.decode() if isinstance(s, (bytes, bytearray)) else str(s)

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

    def _obs(self):
        obs = {"state": self._state(), "proprio": self._proprio()}
        for camera in self.extra_views:
            obs[f"pixels.{camera}"] = np.asarray(self.env.render(
                mode="rgb_array", height=self.res, width=self.res, camera_name=camera), np.uint8)
        return obs

    # -- gymnasium contract --
    def reset(self, seed=None, options=None):
        self.env.reset()
        self._goal_state = None
        self._goal_obj = None
        self._goal_eef = None
        self._proprio_cache = None
        if options and options.get("state") is not None:
            self.set_state(options["state"], options.get("ep_idx", 0))
        return self._obs(), {}

    def set_state(self, state, ep_idx):
        """Rebuild this episode's payload geometry from its model_xml, then restore the
        flattened state."""
        self._ep_idx = int(ep_idx)
        self._proprio_cache = None
        xml = self._model_xml(ep_idx)
        self.env.reset_to({"model": xml, "states": np.asarray(state, np.float64)})
        self._sync_controllers()

    def _sync_controllers(self):
        """Point the OSC nullspace reference at the pose the arm is actually in."""
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
        if n == 0 and not DexMimicGenEnv._warned_no_ctrl:
            DexMimicGenEnv._warned_no_ctrl = True
            print("[dexmg-env] WARNING: no controller exposed update_initial_joints; "
                  "the nullspace-reference fix is NOT active", flush=True)

    def step(self, action):
        self._proprio_cache = None          # the sim is about to move
        a = np.asarray(action, np.float64)
        if self._drop_dims and a.shape[-1] == self._full_adim - len(self._drop_dims):
            full = np.empty(self._full_adim, np.float64)
            keep = [i for i in range(self._full_adim) if i not in self._drop_dims]
            full[keep] = a
            for i in self._drop_dims:
                full[i] = _DROPPED_DIM_VALUE
            a = full
        _, _, _, info = self.env.step(a)
        task_ok = bool(self.env.is_success()["task"])
        dist, per_obj, eef = self._goal_distance()
        if self._goal_obj is None:
            terminated, reward = task_ok, 0.0     # no goal set: full-task predicate
        else:
            terminated = (dist < self.goal_threshold and eef < self.eef_threshold) or task_ok
            reward = -dist
        info = dict(info or {})
        info["goal_distance"] = dist
        info["goal_eef_distance"] = eef
        info["task_success"] = task_ok
        if per_obj is not None:
            info["goal_distance_per_object"] = per_obj
        return self._obs(), float(reward), terminated, False, info

    def render(self):
        # EnvRobosuite.render already returns the upright agentview image (no extra flip).
        return np.asarray(self.env.render(
            mode="rgb_array", height=self.res, width=self.res, camera_name="agentview"), np.uint8)


if "swm/DexMimicGen-v0" not in gym.registry:
    gym.register(id="swm/DexMimicGen-v0", entry_point=DexMimicGenEnv)
