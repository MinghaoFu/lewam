"""swm env-layer plugin: DexMimicGen closed-loop goal-reaching env (TwoArmDrawerCleanup,
TwoArmTransport) for the reactive head and CEM planning.

Additive; mirrors the PROVEN contract in lewam/envs/robomimic_env.py that swm's
World/EnvPool actually drive: gymnasium reset/step/render + a `set_state` callable
applied once per episode on the unwrapped env, with terminated = is_success()["task"]
(swm scores an episode a success iff the env ever returns terminated=True).

Two dexmimicgen-specific facts drive the design (both verified from the data):
  1. The payload object geometry is RANDOMIZED per demo (all 400 model_xml differ in
     the payload <geom> SIZES), so a fixed env + set_state is physically wrong. Each
     episode is rebuilt from its own model_xml via robomimic reset_to({"model": xml}),
     which runs edit_model_xml to rewrite the absolute /home/yuqix asset paths onto the
     local robosuite install before reset_from_xml_string.
  2. model_xml is stored per-episode (400,) in the eval h5. swm's flat HDF5 loader
     row-indexes EVERY column by [ep_offset+start : ...], so it cannot deliver a
     per-episode column through load_chunk. Therefore the env reads model_xml itself
     from the h5, keyed by the per-row `ep_idx` column which the callable DOES deliver.
"""
import os, json, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
# robosuite's robot_env.py imports glfw UNCONDITIONALLY, so glfw must stay importable (removing it
# breaks `import robosuite` with ModuleNotFoundError). But pyglfw resolves its native lib at IMPORT
# time by spawning a python subprocess per candidate path and reading the filename from STDIN --
# under a detached run (stdin=/dev/null) every probe hits EOF and respawns, an infinite silent CPU
# spin. MUJOCO_GL=egl cannot prevent that (it picks the rendering backend, not pyglfw's import-time
# search). Pin PYGLFW_LIBRARY to the bundled lib so `import glfw` dlopens it directly, no probe.
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
import dexmimicgen  # noqa: F401  registers dexmimicgen's payload-randomized TwoArmTransport + its success check into robosuite
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.env_utils import create_env_from_metadata

# Per-episode model_xml lives in its OWN file, not the eval dataset: swm's flat HDF5 loader
# row-indexes every column by [ep_offset+start : ...], and a per-episode (400,) column blows up with
# "IndexError: Fancy indexing out of range for (0-399)" against 162701 rows. So model_xml is split
# out and the env reads it here, keyed by the per-row ep_idx the callable delivers.
_XML_H5 = os.environ.get("DEXMG_XML_H5", os.path.expanduser("~/dexmg_model_xml.h5"))
_ENV_META = os.environ.get("DEXMG_ENV_META", os.path.expanduser("~/dexmg_env_meta.json"))

# bimanual proprio, 18-D, exactly matching scripts convert_dexmg.py PROP order
_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                 "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]

# optional: append (pid, step, goal_distance, task_success, per-object distances) per step so
# the mean final goal-distance can be reported alongside the success rate.
_DIST_LOG = os.environ.get("DEXMG_DIST_LOG", "")
# proprio layout: [r0_eef_pos 0:3, r0_eef_quat 3:7, r0_grip 7:9,
#                 r1_eef_pos 9:12, r1_eef_quat 12:16, r1_grip 16:18]
#
# KNOWN DEFECT, kept deliberately. That layout is the 18-D two-arm PARALLEL-gripper case it was
# written for. TwoArmDrawerCleanup uses dex hands, whose gripper_qpos is 12-D, so its proprio is
# 38-D and robot1's eef actually sits at 19:22 -- slice(9,12) lands inside robot0's gripper joints.
# The goal and the current state are read through the SAME slice, so the predicate stays
# well-defined and every drawer number on record was measured with it; it simply is not "the
# second arm's end-effector". Widening it changes the predicate and invalidates those numbers, so
# fix it only together with a re-measurement. Same for observation_space's declared shape=(18,).
_EEF0, _EEF1 = slice(0, 3), slice(9, 12)
# positive control: execute the demo's recorded actions through the full eval path
_EXPERT_OVERRIDE = os.environ.get("DEXMG_EXPERT_OVERRIDE") == "1"
_EXPERT_CACHE = None
# log every action the env actually receives, to compare the planner-emitted distribution
# against the dataset's raw actions (a missing inverse z-score would show up as std~1).
_ACT_LOG = os.environ.get("DEXMG_ACT_LOG", "")

ObsUtils.initialize_obs_modality_mapping_from_dict({
    "rgb": ["agentview_image"],
    "low_dim": _PROPRIO_KEYS,
})


# One knob for the whole suite. Every WF8 env carries this same default and honours the same
# variable, so the render size is a property of the suite rather than of whichever env class you
# happened to construct. Cells whose data is not 224 override `resolution` in their eval config;
# see the resolution section of the README before changing it.
DEFAULT_RESOLUTION = int(os.environ.get("WF8_RESOLUTION", 224))


def _load_env_meta():
    with open(_ENV_META) as f:
        meta = json.load(f)
    # `env_lang` rides along in the dataset's env_kwargs with the value None -- a task annotation the
    # recording build accepted and no robosuite available here declares, and none of them take
    # **kwargs, so forwarding it raises
    # `TypeError: ManipulationEnv.__init__() got an unexpected keyword argument 'env_lang'` and the
    # env cannot be constructed at all. dexmimicgen's source never references the name, so dropping
    # it changes nothing physical.
    meta.get("env_kwargs", {}).pop("env_lang", None)
    # render only agentview: dataset pixels come from the agentview camera; dropping the
    # other 4 cameras avoids 5x offscreen render cost across the 50 parallel envs.
    ek = meta.get("env_kwargs", {})
    _rc = os.environ.get("DEXMG_RENDER_CAM", "agentview")
    ek["camera_names"] = ["agentview"] if _rc == "agentview" else ["agentview", _rc]
    # Deliberately NOT tied to `resolution`. These size the env's own observation images, which
    # this class never reads -- render() passes height/width explicitly, so the frame the policy
    # sees is `resolution` regardless. Raising them to 224 would just make every step render a
    # larger image nothing consumes. Rendering 224 out of an 84 buffer is safe: robosuite grows the
    # offscreen buffer on demand (utils/binding_utils.py, render() -> update_offscreen_size).
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
        # 0.15 m == robocasa's place_threshold in swm/envs/robocasa.py (subtask_success)
        self.goal_threshold = float(
            goal_threshold if goal_threshold is not None
            else os.environ.get("DEXMG_GOAL_THRESHOLD", 0.15))
        # eef tolerance; inf disables the eef term (objects-only predicate)
        self.eef_threshold = float(os.environ.get("DEXMG_EEF_THRESHOLD", "inf"))
        # Consecutive steps the goal predicate must hold before the episode counts. At 1 this is
        # the reach-anytime rule and reproduces the earlier numbers exactly. Above 1 it rejects a
        # trajectory that merely passes through the goal region on its way somewhere else: staying
        # is evidence the policy recognised the goal, a single frame inside it is not.
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
        # DEXMG_DROP_DIMS: comma-separated raw-action indices that the TRAINING data dropped
        # (constant dims, e.g. drawer's fixed dex-hand joints 11 and 23). The policy then emits
        # (adim - k) dims and we re-insert the constants in step() before robosuite sees it.
        #
        # This exists only because the trainer divides by a per-dim action std with no guard
        # (`train_lewam_gc.py`: `a = (a - act_mean_t) / act_std_t`), so a dim whose std is exactly 0
        # produces 0/0 and an entire run comes back NaN with a zero exit code. The workaround was to
        # drop those dims from the dataset rather than to fix the division. `patches/` carries the
        # one-line clamp that removes the need: with it a constant dim normalises to 0, the head
        # learns to emit 0 there, and the inverse transform restores the constant -- so the action
        # width stays equal to the env's and none of this re-insertion is needed.
        #
        # Kept because the adapter is conditional on the incoming width: a 24-dim policy passes
        # straight through. Existing 22-dim checkpoints keep working; new ones need neither.
        _dd = os.environ.get("DEXMG_DROP_DIMS", "").strip()
        self._drop_dims = sorted(int(x) for x in _dd.split(",") if x != "") if _dd else []
        # The recorded constant is float32(pi/2) widened to float64, not pi/2 itself: measured over
        # all 298,235 drawer steps, dims 11 and 23 hold 1.5707963705062866 and nothing else, while
        # pi/2 is 1.5707963267948966 -- a 4.4e-08 difference. Default to what the data actually says.
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
        """Map the task objects' positions into the flattened mujoco state vector.

        robosuite's get_state()['states'] is [time] + qpos + qvel, so a free joint whose
        qpos address is `a` has its 3-D position at state[1+a : 1+a+3]. We resolve those
        addresses from the live model instead of hardcoding indices, because the payload
        geometry (and therefore the model) is randomized per demo.
        Task objects = payload + trash: TwoArmTransport's own success predicate is
        `payload_in_target_bin AND trash_in_trash_bin`, so goal-reaching is scored on the
        same two bodies -- just as a distance to the goal state rather than a bin test.
        """
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
                continue  # free joint only (7 qpos: 3 pos + 4 quat)
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
        """swm per-episode goal entry point (mirrors pusht/_set_goal_state). The World
        hands us the dataset `state` column at start+goal_offset_steps, i.e. the same
        115-D vector space as the start state."""
        g = np.asarray(goal_state, np.float64).ravel()
        self._goal_state = g
        self._goal_obj = self._obj_pos(g)

    def _set_goal_proprio(self, goal_proprio):
        """Goal end-effector poses (World supplies the dataset `proprio` at start+goal_offset).

        Needed because the two task objects are STATIONARY over a short goal offset in most
        windows -- measured on this dataset, 58% of starts already have both objects within
        0.15 m of their goal at t=0, which is why an objects-only predicate scored random at
        70%. The arms, by contrast, move on essentially every step (median |d_eef| = 0.18 m
        over 25 steps), so requiring the eefs to reach their goal pose as well makes "reached
        the goal state" non-trivial. This mirrors robocasa's subtask_success, which combines
        obj_goal_dist with a hand distance rather than using object distance alone.
        """
        p = np.asarray(goal_proprio, np.float64).ravel()
        self._goal_eef = np.stack([p[_EEF0], p[_EEF1]])

    def _eef_pos(self):
        p = self._proprio().astype(np.float64)
        return np.stack([p[_EEF0], p[_EEF1]])

    def _goal_distance(self):
        """Distance to the goal state: MAX over the two task objects, and (when goal proprio
        was supplied) MAX over the two end-effectors. Aggregating with MAX is the distance
        analogue of the env's own `payload AND trash` conjunction."""
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
        """Cached per step. `get_observation()` re-renders every camera (robomimic calls
        `_get_observations(force_update=True)`, and use_image_obs=True makes that a real render),
        and this was called twice per step -- once through `_obs()`, once through `_eef_pos()` --
        so each step paid for three renders where one is needed. The sim does not move between
        those calls, so the cached value is identical, not an approximation."""
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
        # `reset` cleared the goal but not the dwell counter, so a reused env could carry a
        # partially satisfied hold into the next episode.
        self._hold = 0
        self._proprio_cache = None
        if options and options.get("state") is not None:
            self.set_state(options["state"], options.get("ep_idx", 0))
        return self._obs(), {}

    def _set_start_step(self, step_idx):
        """Record which dataset row this episode starts at, so the expert-action positive
        control can look up the demo's recorded actions."""
        self._start_step = int(step_idx)

    def _expert_action(self):
        if self._ep_idx is None or self._start_step is None:
            return None
        try:
            global _EXPERT_CACHE
            if _EXPERT_CACHE is None:
                with h5py.File(os.environ.get("DEXMG_DS", ""), "r") as f:
                    _EXPERT_CACHE = (f["action"][:], f["ep_offset"][:])
            acts, ep_off = _EXPERT_CACHE
            row = int(ep_off[self._ep_idx]) + self._start_step + self._nstep
            if 0 <= row < len(acts):
                return np.asarray(acts[row], np.float64)
        except Exception:
            return None
        return None

    def set_state(self, state, ep_idx):
        """swm per-episode entry point. Rebuild this episode's payload geometry from its
        model_xml (asset paths rewritten by edit_model_xml), then restore the flattened
        mujoco state -- robomimic's canonical replay: reset -> reset_from_xml_string ->
        set_state_from_flattened, all in one reset_to call. ep_idx names the episode."""
        self._ep_idx = int(ep_idx)
        self._hold = 0
        self._proprio_cache = None
        xml = self._model_xml(ep_idx)
        self.env.reset_to({"model": xml, "states": np.asarray(state, np.float64)})
        self._sync_controllers()

    def _sync_controllers(self):
        """Point every controller's nullspace reference at the pose the arm is actually in.

        `reset_from_xml_string` rebuilds the whole env and initialises the controllers at the
        rebuilt pose; `set_state_from_flattened` then moves the arms somewhere else. The reference
        is left describing a configuration that no longer exists, and the first action gets a large
        spurious nullspace pull.

        Measured by replaying recorded actions against their own recorded states: without this, the
        step-1 error is 3.4 on drawer and 3.0 on transport, on EVERY episode -- these cells rebuild
        the model each time, so unlike tool_hang the problem is not confined to the first one. With
        it, 4.4e-02 and 2.4e-01 respectively.

        robosuite 1.5 nests the part controllers under `robot.composite_controller`, 1.4 exposes a
        single `robot.controller`; walk both.
        """
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
        # Silence here is expensive: this sync is worth a step-1 error of 3.4 -> 4.4e-02 on drawer,
        # so an API change that turns it into a no-op must not pass unnoticed.
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
                full[i] = self._fill_val
            a = full
        if _EXPERT_OVERRIDE:
            # positive control: ignore the policy's action and execute the demo's RECORDED
            # action for this (episode, step) through the exact same World/EnvPool/wrapper
            # path. If this does not reproduce the demo, the action path is broken.
            ea = self._expert_action()
            if ea is not None:
                a = ea
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
            # no goal set -> fall back to the full-task predicate (old protocol)
            terminated, reward = task_ok, 0.0
        else:
            # goal-reaching protocol, mirroring pusht's eval_state(): terminate on a threshold
            # distance to the goal, reward = -distance. When goal proprio was supplied the eef
            # must reach its goal too (see _set_goal_proprio for why objects alone are trivial).
            reached = dist < self.goal_threshold and eef < self.eef_threshold
            self._hold = self._hold + 1 if reached else 0
            # task_ok is latched -- the drawer IS cleaned up -- so it is not gated. Only
            # goal-reaching is, since that is the one a passing trajectory can satisfy by accident.
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
                    # key on pid AND instance: EnvPool may co-locate several envs in one process
                    fh.write(f"{os.getpid()}_{id(self)}\t{self._nstep}\t{dist:.6f}\t{int(task_ok)}"
                             f"\t{eef:.6f}\n")
            except Exception:
                pass
        return self._obs(), float(reward), terminated, False, info

    def render(self):
        # robomimic EnvRobosuite.render already returns the upright agentview image,
        # matching the dataset's stored obs["agentview_image"] (no extra flip).
        cam = os.environ.get("DEXMG_RENDER_CAM", "agentview")
        return np.asarray(self.env.render(
            mode="rgb_array", height=self.res, width=self.res, camera_name=cam), np.uint8)


if "swm/DexMimicGen-v0" not in gym.registry:
    gym.register(id="swm/DexMimicGen-v0", entry_point=DexMimicGenEnv)
