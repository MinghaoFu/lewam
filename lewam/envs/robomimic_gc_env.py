"""swm env-layer plugin: goal-conditioned RoboMimic (robosuite) env for the WF8 protocol.

Additive: does NOT touch lewam/envs/robomimic_env.py. That env terminates on the
robomimic task predicate (`is_success()["task"]`), which is the full-task protocol.
WF8 scores goal-reaching against a dataset state at start+goal_offset, so this env
mirrors DexMimicGenEnv's predicate exactly -- object positions AND end-effector within
threshold of the goal state -- keeping the manipulation families comparable.

Registers `swm/RoboMimicGC-v0`; task (ToolHang / PickPlaceCan / ...) via `world.task`.
"""
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import gymnasium as gym
import h5py
import mujoco
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
from gymnasium import spaces
from robomimic.utils.env_utils import create_env_from_metadata

_DEMO_ROOT = os.environ.get(
    "ROBOMIMIC_RAW", "/mnt/hdfs/bi_algo_a2f/minghao.fu/lewam/data/worldforge10/robomimic_raw")
_TASK_TO_DIR = {"Lift": "lift", "PickPlaceCan": "can", "NutAssemblySquare": "square",
                "ToolHang": "tool_hang", "TwoArmTransport": "transport"}
# 9-D to match the converted dataset's proprio column (eef pos + quat + gripper qpos);
# the policy consumes proprio as a token, so this must equal the training width.
_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
_EEF = slice(0, 3)
_DIST_LOG = os.environ.get("ROBOMIMIC_DIST_LOG", "")

ObsUtils.initialize_obs_modality_mapping_from_dict({
    "rgb": ["agentview_image"],
    "low_dim": _PROPRIO_KEYS,
})


# One knob for the whole suite. Every WF8 env carries this same default and honours the same
# variable, so the render size is a property of the suite rather than of whichever env class you
# happened to construct. Cells whose data is not 224 override `resolution` in their eval config;
# see the resolution section of the README before changing it.
DEFAULT_RESOLUTION = int(os.environ.get("WF8_RESOLUTION", 224))


class RoboMimicGCEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    _warned_no_ctrl = False

    def __init__(self, task="ToolHang", resolution=None, goal_threshold=None, **kwargs):
        super().__init__()
        self.task = task
        self.res = int(DEFAULT_RESOLUTION if resolution is None else resolution)
        # 0.15 m: same place_threshold DexMimicGenEnv and robocasa's subtask_success use
        self.goal_threshold = float(
            goal_threshold if goal_threshold is not None
            else os.environ.get("ROBOMIMIC_GOAL_THRESHOLD", 0.15))
        self.eef_threshold = float(os.environ.get("ROBOMIMIC_EEF_THRESHOLD", "inf"))
        # Consecutive steps the goal predicate must hold before the episode counts. At 1 this is
        # the reach-anytime rule and reproduces the earlier numbers exactly. Above 1 it rejects a
        # trajectory that merely passes through the goal region on its way somewhere else: staying
        # is evidence the policy recognised the goal, a single frame inside it is not.
        self.hold_k = int(os.environ.get("GOAL_HOLD_K", 1))
        self._hold = 0
        self._goal_state = None
        self._goal_obj = None
        self._goal_eef = None
        self._start_state = None
        self._nstep = 0
        # Must exist before the observation_space line below, which calls _proprio().
        self._proprio_cache = None

        raw = os.path.join(_DEMO_ROOT, _TASK_TO_DIR.get(task, str(task).lower()),
                           "ph", "image_384_v15.hdf5")
        with h5py.File(raw, "r") as f:
            env_meta = json.loads(f["data"].attrs["env_args"])
        # v141 env_args carry a flat OSC_POSE block; robosuite 1.5 wants the composite form.
        # The 1.5 BASIC default for Panda is numerically identical (output +-0.05/+-0.5,
        # kp 150, damping_ratio 1, uncoupled, delta input), so this is a re-nesting only.
        # Only re-nest when the installed robosuite actually needs it. Measured on tool_hang, whose
        # demos carry env_version 1.4.1: running robosuite 1.4.1 and handing it the dataset's own
        # flat block reproduces the demo ~4x more accurately than running 1.5.1 on the re-nested
        # form (per-step state error 4.7e-03 against 1.8e-02). The re-nesting is a compatibility
        # shim, not a no-op, so it should fire only when the versions differ.
        import robosuite as _rs

        cc = env_meta.get("env_kwargs", {}).get("controller_configs")
        if isinstance(cc, dict) and "body_parts" not in cc and not _rs.__version__.startswith("1.4"):
            from robosuite.controllers import load_composite_controller_config
            rb = env_meta["env_kwargs"].get("robots") or ["Panda"]
            rb = rb[0] if isinstance(rb, (list, tuple)) else rb
            env_meta["env_kwargs"]["controller_configs"] = load_composite_controller_config(
                controller=None, robot=rb)
        # NOTE: env_kwargs["camera_heights"] is deliberately left at the dataset's 84 and NOT
        # raised to self.res. It sizes the env's own observation images, which this class never
        # reads -- render() passes height/width explicitly. The dataset sets use_camera_obs=False,
        # but create_env_from_metadata(use_image_obs=True) turns it back on, so every step renders
        # an obs image nothing consumes; keeping it at 84 keeps that waste small. Rendering the
        # 224 frame from an 84 buffer is safe -- robosuite grows the offscreen buffer on demand
        # (utils/binding_utils.py, render() -> update_offscreen_size), once, on the first call.
        self.env = create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True)

        adim = int(self.env.action_dimension)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(adim,), dtype=np.float32)
        sdim = int(np.asarray(self.env.get_state()["states"]).shape[0])
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(sdim,), dtype=np.float32),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(len(self._proprio()),), dtype=np.float32),
        })
        self.render_mode = "rgb_array"
        self._build_obj_slices()
        self._build_fk()
        # Captured once, and passed to every reset_to. ToolHang's scene is fixed across demos -- all
        # 200 stored model_xml hash identically -- so this is the same model every time and carries
        # no per-episode information. It is here for determinism, not geometry: `reset_to` with only
        # `states` writes qpos and qvel and leaves everything else as the previous episode left it,
        # and two rollouts from the same restored state then diverge in contact-heavy windows.
        # Measured: replaying an episode's actions against a goal produced by that same rollout ends
        # 0.13 and 0.23 away on two windows in twenty. drawer never shows this because its
        # per-episode geometry forces it through the full `reset_from_xml_string` path every time.
        self._model_xml = self.env.env.sim.model.get_xml()

    def _build_fk(self):
        """A private model+data pair for forward kinematics, so goal poses can be evaluated without
        moving the episode's simulator. Also resolves the end-effector site once."""
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

    # -- goal-state geometry ---------------------------------------------------
    def _build_obj_slices(self):
        """Map task-object positions into the flattened mujoco state vector.

        get_state()['states'] is [time] + qpos + qvel, so a free joint at qpos address
        `a` has its position at state[1+a : 1+a+3]. Only task objects are free joints
        (7 qpos); the arm's joints are hinges (1 qpos), so taking every free joint
        selects exactly the manipulable objects -- for ToolHang the frame and the tool,
        which are the two bodies its own success predicate tests.
        """
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
        """End-effector position for a state, computed WITHOUT touching the live simulator.

        The goal end-effector pose has to come from the sim, because tool_hang's converted proprio
        column is constant to within 1e-5 and would hand every episode the same goal. The obvious
        way to get it -- restore the goal, read the pose, restore the start -- moves the simulator
        twice, and those two restores leave it in a subtly different condition than a plain
        set_state: replaying an episode's own actions against a goal generated by that very rollout
        then scored 90% instead of the 100% it is by construction, with two windows ending 0.13 and
        0.23 away from a goal they had just produced. Syncing the controller afterwards recovered
        part of it and not all.

        Forward kinematics on a scratch MjData answers the same question and disturbs nothing. The
        state layout is [time] + qpos + qvel, as robosuite's get_state() writes it."""
        v = np.asarray(state_vec, np.float64).ravel()
        m, d = self._fk_model, self._fk_data
        d.qpos[:] = v[1:1 + m.nq]
        d.qvel[:] = v[1 + m.nq:1 + m.nq + m.nv]
        mujoco.mj_forward(m, d)
        return np.asarray(d.site_xpos[self._eef_site], np.float64).ravel().copy()

    def _get_goal_state(self):
        return self._goal_state

    def _set_goal_state(self, goal_state):
        """Cache the goal, and read the goal end-effector pose FROM THE SIM rather than from
        the dataset's proprio column: tool_hang's converted proprio is a constant column
        (std ~0 on every dimension), so a dataset-derived goal eef would be the same fixed
        pose for every episode. Restoring the goal state, reading robot0_eef_pos, then
        restoring the start is exact and works for any robomimic task."""
        g = np.asarray(goal_state, np.float64).ravel()
        self._goal_state = g
        self._goal_obj = self._obj_pos(g)
        self._goal_eef = self._eef_at(g)

    def _set_goal_proprio(self, goal_proprio):
        """No-op: kept so the shared config's callable list applies unchanged. The goal eef
        comes from the sim (see _set_goal_state) because the dataset column is degenerate."""
        return

    def _state(self):
        return np.asarray(self.env.get_state()["states"], np.float32)

    def _proprio(self):
        """Cached per step. `get_observation()` re-renders every camera (robomimic calls
        `_get_observations(force_update=True)`, and use_image_obs=True makes that a real render),
        and this used to be called twice per step -- once through `_obs()`, once through
        `_eef_pos()` -- so each step paid for three renders where one is needed. The sim does not
        move between those calls, so the cached value is identical, not an approximation."""
        if self._proprio_cache is None:
            o = self.env.get_observation()
            self._proprio_cache = np.concatenate(
                [np.asarray(o[k], np.float32).ravel() for k in _PROPRIO_KEYS]).astype(np.float32)
        return self._proprio_cache

    def _eef_pos(self):
        return self._proprio().astype(np.float64)[_EEF]

    def _obs(self):
        return {"state": self._state(), "proprio": self._proprio()}

    def _goal_distance(self):
        """MAX over task objects, plus the eef term when goal proprio was supplied."""
        if self._goal_obj is None:
            return float("inf"), float("inf")
        d = float(np.linalg.norm(self._obj_pos(self._state()) - self._goal_obj, axis=-1).max())
        e = float("inf")
        if self._goal_eef is not None:
            e = float(np.linalg.norm(self._eef_pos() - self._goal_eef))
        return d, e

    # -- gym contract ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        self.env.reset()
        # Clear the previous episode's goal and dwell counter. In a scoring run every env runs
        # exactly one episode so this never bit, but a reused env would otherwise terminate
        # against a goal belonging to the episode before it.
        self._goal_state = self._goal_obj = self._goal_eef = None
        self._proprio_cache = None
        self._nstep = 0
        self._hold = 0
        if options and options.get("state") is not None:
            self.set_state(options["state"])
        return self._obs(), {}

    def set_state(self, state):
        """Per-episode entry point. ToolHang's assets are fixed across demos (only their
        placement varies, and that lives in qpos), so the flattened state alone restores
        the episode -- no per-demo model_xml replay is needed here."""
        self._start_state = np.asarray(state, np.float64)
        # The rebuild buys determinism and costs fidelity: it doubles the disagreement with the
        # recorded states, because get_xml() -> reset_from_xml_string() does not round-trip the
        # compiled model exactly. Under a stack that matches the recording it should be unnecessary,
        # so it is a switch rather than a fact. TOOLHANG_MODEL_REBUILD=0 turns it off.
        if os.environ.get("TOOLHANG_MODEL_REBUILD", "1") != "0":
            self.env.reset_to({"model": self._model_xml, "states": self._start_state})
        else:
            self.env.reset_to({"states": self._start_state})
        self._sync_controller()
        self._proprio_cache = None
        self._nstep = 0
        self._hold = 0

    def _sync_controller(self):
        """`reset_to` restores qpos and qvel and nothing else. An OSC controller also carries
        `initial_joint`, the reference its nullspace term pulls toward, and that is left holding
        whatever configuration the arm was in before -- the env's construction pose on the first
        episode, the previous episode's final pose after that.

        Measured on tool_hang: without this the first episode after construction replays its own
        demo with a step-1 state error of 3.19 while later episodes sit at 8e-03. With it, the first
        episode drops to 1.5e-02 and the steady state improves to 5e-03. It matters more than it
        looks, because a scoring run sets `num_envs == num_eval` and every env runs exactly one
        episode -- so in a real eval, every episode is a first episode.

        Note a full `env.reset()` is the wrong fix and measurably worse (step-1 error 2.0): it
        re-randomises object placement, initialises the controller against that new pose, and then
        `reset_to` puts the objects back, leaving the reference matched to a scene that no longer
        exists.

        Walk both layouts: robosuite 1.4 exposes a single `robot.controller`, 1.5 nests the part
        controllers under `robot.composite_controller`. Looking only at `robot.controller` means
        that under 1.5 -- which this file explicitly supports, see the version-conditional
        re-nesting above -- the sync silently finds nothing and the fidelity fix quietly stops
        applying. Kept identical to DexMimicGenEnv._sync_controllers on purpose.
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
        # Silence here is expensive: this sync is worth a step-1 error of 3.19 -> 1.5e-02, so an
        # API change that makes it a no-op must not pass unnoticed.
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
            reached = dist < self.goal_threshold and eef < self.eef_threshold
            self._hold = self._hold + 1 if reached else 0
            # task_ok is latched -- the tool IS hung -- so it is not gated. Only goal-reaching is,
            # since that is the one a passing trajectory can satisfy by accident.
            terminated = self._hold >= self.hold_k or task_ok
            reward = -dist
        info = dict(info or {})
        info["goal_distance"] = dist
        info["goal_eef_distance"] = eef
        info["goal_hold"] = self._hold
        info["task_success"] = task_ok
        self._nstep += 1
        if _DIST_LOG:
            try:
                with open(_DIST_LOG, "a") as fh:
                    fh.write(f"{os.getpid()}_{id(self)}\t{self._nstep}\t{dist:.6f}"
                             f"\t{int(task_ok)}\t{eef:.6f}\n")
            except Exception:
                pass
        return self._obs(), float(reward), terminated, False, info

    def render(self):
        return np.asarray(self.env.render(
            mode="rgb_array", height=self.res, width=self.res, camera_name="agentview"), np.uint8)


if "swm/RoboMimicGC-v0" not in gym.registry:
    gym.register(id="swm/RoboMimicGC-v0", entry_point=RoboMimicGCEnv)
