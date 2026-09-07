"""swm env-layer plugin: DexMimicGen closed-loop goal-reaching env (TwoArmDrawerCleanup,
TwoArmTransport) for the reactive head and CEM planning.

Additive; mirrors the contract in lewam/envs/robomimic_env.py that swm's World/EnvPool
drive: gymnasium reset/step/render + a per-episode `set_state`, with terminated ==
"episode success". Two dexmimicgen-specific facts drive the design: payload geometry is
randomized per demo (each episode rebuilds from its own model_xml via reset_to), and
model_xml is a per-episode column swm's row-indexing loader cannot deliver, so the env
reads it from its own h5 keyed by ep_idx.
"""
import os, json, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
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

_XML_H5 = os.environ.get("DEXMG_XML_H5", os.path.expanduser("~/dexmg_model_xml.h5"))
_ENV_META = os.environ.get("DEXMG_ENV_META", os.path.expanduser("~/dexmg_env_meta.json"))

# bimanual proprio, 18-D, matching convert_dexmg.py PROP order
_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                 "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]

_DIST_LOG = os.environ.get("DEXMG_DIST_LOG", "")
# KNOWN DEFECT, kept deliberately: these slices assume the 18-D parallel-gripper layout.
# Drawer's dex hands make proprio 38-D with robot1's eef at 19:22, so slice(9,12) lands in
# robot0's gripper joints -- but goal and state read the SAME slice, the predicate stays
# well-defined, and every drawer number on record was measured with it. Fix only together
# with a re-measurement (same for observation_space's shape=(18,)).
_EEF0, _EEF1 = slice(0, 3), slice(9, 12)
# positive control: execute the demo's recorded actions through the full eval path
_EXPERT_OVERRIDE = os.environ.get("DEXMG_EXPERT_OVERRIDE") == "1"
_EXPERT_CACHE = None
_ACT_LOG = os.environ.get("DEXMG_ACT_LOG", "")

ObsUtils.initialize_obs_modality_mapping_from_dict({
    "rgb": ["agentview_image"],
    "low_dim": _PROPRIO_KEYS,
})

# Suite-wide render size; cells whose data is not 224 override `resolution` in their eval config.
DEFAULT_RESOLUTION = int(os.environ.get("WF8_RESOLUTION", 224))


def _load_env_meta():
    with open(_ENV_META) as f:
        meta = json.load(f)
    # env_lang=None rides along in the recorded env_kwargs; no installed robosuite accepts it.
    meta.get("env_kwargs", {}).pop("env_lang", None)
    # Render only agentview (the dataset camera); the other 4 cameras are per-step waste.
    ek = meta.get("env_kwargs", {})
    _rc = os.environ.get("DEXMG_RENDER_CAM", "agentview")
    ek["camera_names"] = ["agentview"] if _rc == "agentview" else ["agentview", _rc]
    # Sizes the env's own obs images, which nothing here reads -- render() passes height/width
    # explicitly and robosuite grows the offscreen buffer on demand. Small keeps steps cheap.
    ek["camera_heights"] = 84
    ek["camera_widths"] = 84
    return meta


class DexMimicGenEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    _warned_no_ctrl = False

    def __init__(self, task="TwoArmTransport", resolution=None, goal_threshold=None, **kwargs):
        super().__init__()
        self.task = task
        self.res = int(DEFAULT_RESOLUTION if resolution is None else resolution)
        # 0.15 m == robocasa's place_threshold (subtask_success)
        self.goal_threshold = float(
            goal_threshold if goal_threshold is not None
            else os.environ.get("DEXMG_GOAL_THRESHOLD", 0.15))
        self.eef_threshold = float(os.environ.get("DEXMG_EEF_THRESHOLD", "inf"))
        # 1 = reach-anytime (matches earlier numbers); >1 requires the predicate to hold K steps.
        self.hold_k = int(os.environ.get("GOAL_HOLD_K", 1))
        self._hold = 0
        self._goal_state = None
        self._goal_obj = None
        self._goal_eef = None
        self._ep_idx = None
        self._start_step = None
        self._nstep = 0
        self._proprio_cache = None
        env_meta = _load_env_meta()
        self.env = create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True)
        adim = int(self.env.action_dimension)
        # DEXMG_DROP_DIMS: raw-action indices the TRAINING data dropped (constant dims, e.g.
        # drawer's fixed dex-hand joints 11/23). The policy emits (adim - k) dims and step()
        # re-inserts the constants. Conditional on the incoming width, so 24-dim checkpoints
        # pass straight through; with the normalizer std guard new runs need none of this.
        _dd = os.environ.get("DEXMG_DROP_DIMS", "").strip()
        self._drop_dims = sorted(int(x) for x in _dd.split(",") if x != "") if _dd else []
        # The recorded constant is float32(pi/2) widened to float64 (1.5707963705062866),
        # not pi/2 itself; default to what the data actually says.
        _REC_PI_2 = 1.5707963705062866
        self._fill_val = float(os.environ.get("DEXMG_FILL_VAL", str(_REC_PI_2)))
        self._full_adim = adim
        eff_adim = adim - len(self._drop_dims)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(eff_adim,), dtype=np.float32)
        sdim = int(np.asarray(self.env.get_state()["states"]).shape[0])
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(sdim,), dtype=np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32),
        })
        self.render_mode = "rgb_array"
        self._build_obj_slices()

    # -- goal-state geometry ---------------------------------------------------
    def _build_obj_slices(self):
        """Map task objects into the flattened state ([time]+qpos+qvel): free-joint addresses
        are resolved from the live model because the payload geometry is randomized per demo.
        Task objects = payload + trash, the two bodies the env's own success predicate tests."""
        sim = self.env.env.sim
        model = sim.model
        names = list(getattr(model, "joint_names", []))
        chosen, all_free = [], []
        for nm in names:
            try:
                adr = model.get_joint_qpos_addr(nm)
            except Exception:
                continue
            if not (isinstance(adr, (tuple, list)) and (adr[1] - adr[0]) == 7):
                continue  # free joints only (7 qpos)
            all_free.append((nm, int(adr[0])))
            low = nm.lower()
            if ("payload" in low or "trash" in low) and "bin" not in low:
                chosen.append((nm, int(adr[0])))
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
        g = np.asarray(goal_state, np.float64).ravel()
        self._goal_state = g
        self._goal_obj = self._obj_pos(g)

    def _set_goal_proprio(self, goal_proprio):
        """Goal end-effector poses. Required to keep the predicate non-trivial: 58% of starts
        already have both (stationary) objects within 0.15 m of their goal at t=0, while the
        arms move every step -- mirrors robocasa's subtask_success obj+hand combination."""
        p = np.asarray(goal_proprio, np.float64).ravel()
        self._goal_eef = np.stack([p[_EEF0], p[_EEF1]])

    def _eef_pos(self):
        p = self._proprio().astype(np.float64)
        return np.stack([p[_EEF0], p[_EEF1]])

    def _goal_distance(self):
        """MAX over task objects, plus MAX over the two eefs when goal proprio was supplied
        -- the distance analogue of the env's own `payload AND trash` conjunction."""
        if self._goal_obj is None:
            return float("inf"), None, float("inf")
        d = np.linalg.norm(self._obj_pos(self._state()) - self._goal_obj, axis=-1)
        e = float("inf")
        if self._goal_eef is not None:
            e = float(np.linalg.norm(self._eef_pos() - self._goal_eef, axis=-1).max())
        return float(d.max()), d, e

    # -- helpers ---------------------------------------------------------------
    def _model_xml(self, ep_idx):
        with h5py.File(_XML_H5, "r") as f:
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
        return {"state": self._state(), "proprio": self._proprio()}

    # -- gymnasium contract ----------------------------------------------------
    def reset(self, seed=None, options=None):
        self.env.reset()
        self._goal_state = None
        self._goal_obj = None
        self._goal_eef = None
        self._nstep = 0
        self._hold = 0
        self._proprio_cache = None
        if options and options.get("state") is not None:
            self.set_state(options["state"], options.get("ep_idx", 0))
        return self._obs(), {}

    def _set_start_step(self, step_idx):
        """Dataset row this episode starts at, for the expert-action positive control."""
        self._start_step = int(step_idx)

    def _expert_action(self):
        if self._ep_idx is None or self._start_step is None:
            return None
        try:
            global _EXPERT_CACHE
            if _EXPERT_CACHE is None:
                with h5py.File(os.environ.get("DEXMG_DS", ""), "r") as f:
                    # the pod copies of the dexmg files carry ep_len, not ep_offset: derive the offsets
                    ep_off = (f["ep_offset"][:] if "ep_offset" in f
                              else np.concatenate([[0], np.cumsum(f["ep_len"][:-1])]))
                    _EXPERT_CACHE = (f["action"][:], ep_off)
            acts, ep_off = _EXPERT_CACHE
            row = int(ep_off[self._ep_idx]) + self._start_step + self._nstep
            if 0 <= row < len(acts):
                return np.asarray(acts[row], np.float64)
        except Exception:
            return None
        return None

    def set_state(self, state, ep_idx):
        """Rebuild this episode's payload geometry from its model_xml, then restore the
        flattened state -- robomimic's canonical replay, one reset_to call."""
        self._ep_idx = int(ep_idx)
        self._hold = 0
        self._proprio_cache = None
        xml = self._model_xml(ep_idx)
        self.env.reset_to({"model": xml, "states": np.asarray(state, np.float64)})
        self._sync_controllers()

    def _sync_controllers(self):
        """Point the OSC nullspace reference at the pose the arm is actually in: the xml rebuild
        initialises controllers at the rebuilt pose, then set_state moves the arms. Measured
        step-1 replay error without this: 3.4 (drawer) / 3.0 (transport) on EVERY episode; with
        it 4.4e-02 / 2.4e-01. Walks both robosuite layouts (1.4 controller, 1.5 composite)."""
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
        if _EXPERT_OVERRIDE:
            # positive control: the demo's recorded action replaces the policy's BEFORE the drop-dims
            # expansion, so a recorded action stored without the constant dims is expanded like any other
            ea = self._expert_action()
            if ea is not None:
                a = ea
        if self._drop_dims and a.shape[-1] == self._full_adim - len(self._drop_dims):
            full = np.empty(self._full_adim, np.float64)
            keep = [i for i in range(self._full_adim) if i not in self._drop_dims]
            full[keep] = a
            for i in self._drop_dims:
                full[i] = self._fill_val
            a = full
        if _ACT_LOG:
            try:
                with open(_ACT_LOG, "a") as fh:
                    fh.write("\t".join(f"{x:.6f}" for x in np.asarray(a).ravel()) + "\n")
            except Exception:
                pass
        _, _, _, info = self.env.step(a)
        task_ok = bool(self.env.is_success()["task"])
        dist, per_obj, eef = self._goal_distance()
        if self._goal_obj is None:
            terminated, reward = task_ok, 0.0     # no goal set: full-task predicate
        else:
            reached = dist < self.goal_threshold and eef < self.eef_threshold
            self._hold = self._hold + 1 if reached else 0
            # task_ok is latched, so it is not hold-gated; only goal-reaching can be accidental.
            terminated = self._hold >= self.hold_k or task_ok
            reward = -dist
        info = dict(info or {})
        info["goal_distance"] = dist
        info["goal_eef_distance"] = eef
        info["goal_hold"] = self._hold
        info["task_success"] = task_ok
        if per_obj is not None:
            info["goal_distance_per_object"] = per_obj
        self._nstep += 1
        if _DIST_LOG:
            try:
                with open(_DIST_LOG, "a") as fh:
                    # key on pid AND instance: EnvPool may co-locate several envs per process
                    fh.write(f"{os.getpid()}_{id(self)}\t{self._nstep}\t{dist:.6f}\t{int(task_ok)}"
                             f"\t{eef:.6f}\n")
            except Exception:
                pass
        return self._obs(), float(reward), terminated, False, info

    def render(self):
        # EnvRobosuite.render already returns the upright agentview image (no extra flip).
        cam = os.environ.get("DEXMG_RENDER_CAM", "agentview")
        return np.asarray(self.env.render(
            mode="rgb_array", height=self.res, width=self.res, camera_name=cam), np.uint8)


if "swm/DexMimicGen-v0" not in gym.registry:
    gym.register(id="swm/DexMimicGen-v0", entry_point=DexMimicGenEnv)
