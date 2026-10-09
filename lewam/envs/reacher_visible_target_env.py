"""Dataset-replay Reacher with the policy-collection target kept visible.

The released LeWM Reacher evaluation uses ``task="qpos_match"``.  In
``stable-worldmodel==0.1.1`` that task has two independent notions of target:

* ``target_qpos`` is the future arm configuration used only for termination;
* the MuJoCo ``target`` geom is randomized on reset and hidden by default.

Our policy-collected Reacher data contains a visible, fixed target geom in
every frame.  During dataset replay we therefore have to restore that geom from
the episode's ``target_pos`` column.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np

from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper


ENV_ID = "swm/ReacherVisibleTargetDMControl-v0"
_TARGET_RGBA = np.array([0.6, 0.3, 0.3, 1.0], dtype=np.float64)


class ReacherVisibleTargetDMControlWrapper(ReacherDMControlWrapper):
    """Reacher wrapper whose task target can be restored from an HDF5 row."""

    def _set_target_visible(self, visible: bool) -> None:
        rgba = _TARGET_RGBA.copy()
        if not visible:
            rgba[3] = 0.0
        self.env.physics.named.model.mat_rgba["target"] = rgba

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        self._set_target_visible(False)

        if options is not None and options.get("task_target_pos") is not None:
            self.set_task_target_pos(options["task_target_pos"])
            obs = self._obs_to_array(
                self.env.task.get_observation(self.env.physics)
            )
            info = self.info
        return obs, info

    def set_task_target_pos(self, target_pos, goal_target_pos=None):
        """Restore the fixed, visible task target used during collection.

        ``goal_target_pos`` is optional and is used only as a consistency
        check.  For a valid Reacher episode the target geom is static, so its
        initial and goal-frame values must agree.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(-1)
        if target_pos.shape != (2,) or not np.isfinite(target_pos).all():
            raise ValueError(
                "target_pos must be a finite vector with shape (2,), got "
                f"{target_pos!r}"
            )

        if goal_target_pos is not None:
            goal_target_pos = np.asarray(
                goal_target_pos, dtype=np.float64
            ).reshape(-1)
            if goal_target_pos.shape != (2,) or not np.allclose(
                target_pos, goal_target_pos, rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    "Reacher task target changed inside one episode: "
                    f"initial={target_pos}, goal={goal_target_pos}"
                )

        physics = self.env.physics
        physics.named.model.geom_pos["target", ["x", "y"]] = target_pos
        self._set_target_visible(True)
        physics.forward()
        self._install_wrapped_qpos_match()

    def _install_wrapped_qpos_match(self):
        """Angle-correct the qpos_match success predicate."""
        task = self.env.task
        if getattr(task, "_wrapped_qpos_patched", False):
            return
        if not hasattr(task, "qpos_threshold"):
            return  # not the qpos_match task ('easy'/'hard'): nothing to fix
        threshold = task.qpos_threshold

        def get_termination(physics, _task=task, _thr=threshold):
            if _task.target_qpos is None:
                return None
            limited = np.asarray(physics.model.jnt_limited, dtype=bool)
            raw = np.abs(np.asarray(physics.data.qpos) - _task.target_qpos)
            wrapped = np.abs(np.remainder(raw + np.pi, 2.0 * np.pi) - np.pi)
            diff = np.where(limited, raw, wrapped)
            if np.all(diff < _thr):
                return 0.0
            return None

        task.get_termination = get_termination
        task._wrapped_qpos_patched = True


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=ReacherVisibleTargetDMControlWrapper)
