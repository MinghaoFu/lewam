"""Real-robot rollout eval (`backend: lerobot`), robot-agnostic.

The embodiment is pure config: `robot:` is a lerobot `RobotConfig` in data form (the
draccus `type` field picks the robot class, camera entries pick camera classes the same
way; see configs/embodiment/). Observations and actions are mapped through the same
lerobot feature helpers the record pipeline uses, so batch fields carry the exact
dataset-column names training saw (`observation.state`, `observation.images.<cam>`,
`action`) on any embodiment.

Offline-first: main.py attaches the returned EvalResult afterwards, and the only wandb this
module touches is the active run's id, parked in the rollout dir so `eval.resume` can
reattach (None when wandb is off). Success is human-tagged from the terminal after each
episode.

Inference runs in-process by default; `server:` points it at scripts/serve.py instead, and
only the driver differs (see chunking.py).
"""

import contextlib
import importlib
import json
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from ...utils.session import (
    CYAN,
    YELLOW,
    announce,
    flush_pending_stdin,
    install_graceful_sigint_handler,
    prompt,
    quiet_console_logging,
)
from .base import EvalResult
from .chunking import RemoteDriver, make_driver

SLOW_TICK_WARN_FRACTION = 0.30


def _decode_robot_config(raw):
    """dict -> lerobot RobotConfig.

    A robot class registers with draccus at import time and lerobot keeps those imports
    lazy, so the module named after `type` must be imported first -- sweeping the robots
    package for types whose module is named differently (so101_follower in so_follower).
    """
    import draccus
    from lerobot.robots.config import RobotConfig

    from ...utils.cameras import build_cameras
    from ...utils.ports import resolve_arm_port

    def registered(kind):
        try:
            RobotConfig.get_choice_class(kind)
            return True
        except Exception:  # noqa: BLE001 - draccus raises plain KeyError subclasses
            return False

    raw = dict(raw)
    cameras = build_cameras(raw.pop("cameras", None))
    kind = raw["type"]
    raw["port"] = resolve_arm_port(kind, raw.get("port"))
    if not registered(kind):
        try:
            importlib.import_module(f"lerobot.robots.{kind}")
        except ModuleNotFoundError:
            pass
    if not registered(kind):
        import pkgutil

        import lerobot.robots as robots_pkg
        for mod in pkgutil.iter_modules(robots_pkg.__path__):
            if not mod.ispkg:
                continue
            try:
                importlib.import_module(f"lerobot.robots.{mod.name}")
            except Exception:  # noqa: BLE001 - optional hardware deps may be absent
                continue
            if registered(kind):
                break
    cfg = draccus.decode(RobotConfig, raw)
    cfg.cameras = cameras
    return cfg


SESSION_LOG = "eval_session.json"


def read_session_log(record_root):
    """What an earlier session of this rollout dir recorded: its scored `episodes`, and the
    `wandb_run_id` a resume reattaches to so both halves score one run.

    Rewritten after every kept episode, so a session killed mid-episode still resumes with
    everything scored before it. A malformed log stops the resume rather than silently
    restarting the count and reporting a success rate over the wrong denominator.
    """
    if record_root is None:
        return {}
    path = Path(record_root) / SESSION_LOG
    if not path.exists():
        return {}
    log = json.loads(path.read_text())
    if not isinstance(log, dict) or not isinstance(log.get("episodes", []), list):
        raise ValueError(f"{path} is not a rollout session log")
    return log


def _save_session_log(record_root, rows, wandb_run_id):
    if record_root is None:
        return
    Path(record_root, SESSION_LOG).write_text(
        json.dumps({"wandb_run_id": wandb_run_id, "episodes": rows}, indent=2)
    )


def _concat_episode_videos(dataset_root, camera):
    """One real-time mp4 of every recorded episode of `camera`, concatenated in order.

    The dataset writer already encoded the episodes as h264 chunk files under
    videos/<camera>/; those share one encoder configuration, so ffmpeg's concat demuxer
    joins them by stream copy -- no re-encode, a few seconds for any length. A single
    chunk file is returned as-is. Returns the mp4 path, or None (no files, or ffmpeg
    missing for a multi-file join -- the dataset still holds every episode either way).
    """
    import subprocess
    import tempfile

    cam_dir = Path(dataset_root) / "videos" / camera
    files = sorted(cam_dir.rglob("*.mp4"))
    files = [f for f in files if "all_episodes" not in f.name]
    if not files:
        return None
    out = Path(dataset_root) / f"{camera.rsplit('.', 1)[-1]}_all_episodes.mp4"
    if len(files) == 1:
        return files[0]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for f in files:
            listing.write(f"file '{f.resolve()}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", listing.name, "-c", "copy", str(out)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as err:
        print(f"could not concatenate {camera} episode videos ({err}); "
              f"the per-episode files remain under {cam_dir}")
        return files[0]
    finally:
        Path(listing.name).unlink(missing_ok=True)
    return out


class LeRobotEval:
    def __init__(self, cfg):
        self.cfg = cfg
        self.checkpoint_step = None
        self.sampling = {}

    def _make_driver(self, model, obs_features):
        """Local inference, or the remote server named by `server:` (see chunking.py)."""
        if not self.cfg.get("server"):
            driver = make_driver(model, self.cfg, obs_features)
            self.sampling = driver.sampling
            return driver
        run = self.cfg.get("run")
        driver = RemoteDriver(self.cfg, obs_features, run)
        self.checkpoint_step = driver.checkpoint_step
        self.sampling = driver.sampling
        announce(
            f"Remote inference: {self.cfg.server} serving {run}"
            + (f" @ step {driver.checkpoint_step}" if driver.checkpoint_step else ""),
            CYAN,
        )
        announce(
            f"  execute_len={driver.execute_len} "
            + " ".join(f"{k}={v}" for k, v in driver.sampling.items()),
            CYAN,
        )
        return driver

    def _make_robot(self):
        from lerobot.robots.utils import make_robot_from_config

        raw = OmegaConf.to_container(self.cfg.robot, resolve=True)
        return make_robot_from_config(_decode_robot_config(raw))

    def run(self, model):
        robot = self._make_robot()
        # a camera slower than the control loop can't feed every step; caught here
        # rather than as duplicated frames mid-episode
        cameras = getattr(getattr(robot, "config", None), "cameras", None) or {}
        fps = float(self.cfg.get("fps", 30))
        too_slow = {n: c.fps for n, c in cameras.items() if c.fps is not None and c.fps < fps}
        if too_slow:
            raise ValueError(
                f"cameras {too_slow} run below the control fps={fps}. "
                "Raise their fps in the embodiment config, or lower fps."
            )
        if cameras:
            from ...utils.stderr_filter import (CORRUPT_JPEG, LIBAV_TAGGED, SVT_INFO,
                                                suppress_stderr_lines)
            suppress_stderr_lines([CORRUPT_JPEG, LIBAV_TAGGED, SVT_INFO])

        # imported here so this module stays importable without lerobot installed
        from ...utils.stream import (
            CameraPanel,
            LeRobotObservationPanel,
            TimeSeriesPanel,
            start_stream,
        )

        robot.connect()
        panels = []
        live_cameras = getattr(robot, "cameras", None) or {}
        if live_cameras and self.cfg.get("stream_cameras", True):
            panels.append(
                CameraPanel(
                    live_cameras,
                    height=int(self.cfg.get("stream_height", 360)),
                    fps=int(self.cfg.get("stream_fps", 15)),
                    quality=int(self.cfg.get("stream_quality", 95)),
                )
            )
        window = int(self.cfg.get("stream_metrics_window", 300))
        observations = None
        if self.cfg.get("stream_metrics", True):
            observations = LeRobotObservationPanel(window=window)
            panels.append(observations)
        latency = None
        if self.cfg.get("stream_latency", True):
            latency = TimeSeriesPanel(name="latency", window=window)
            panels.append(latency)
        stream = start_stream(panels, port=int(self.cfg.get("stream_port", 8090)))
        if stream is not None:
            announce(f"Watch the rollout: {stream.url}", CYAN)
        try:
            return self._rollout(model, robot, observations, latency)
        finally:
            # before disconnect(): read_latest() raises once the cameras are gone
            if stream is not None:
                stream.close()
            try:
                robot.disconnect()
            except Exception as err:  # noqa: BLE001 - never mask what ended the rollout
                announce(f"robot did not disconnect cleanly: {err}", YELLOW)

    def _pause_listener(self, listener):
        """TerminalKeyListener holds stdin in no-echo cbreak mode; pause it around
        prompts so typed characters show up."""
        from lerobot.utils.keyboard_input import TerminalKeyListener

        if isinstance(listener, TerminalKeyListener):
            listener.stop()
            return lambda: listener.start()
        return lambda: None

    def _ask(self, listener, robot, text, drop_torque=False):
        """Prompt with the listener paused and stale keys flushed. `drop_torque` is for
        open-ended prompts on an idle arm (an enabled-but-uncommanded motor faults on
        its own comm-timeout); it must stay False while a homing ramp is commanding."""
        resume = self._pause_listener(listener)
        # is_connected, here and again after the prompt: a robot that dropped out is
        # handled by the rollout loop, and must not turn a prompt into a traceback.
        can_torque = (
            drop_torque
            and robot.is_connected
            and hasattr(robot, "disable_torque")
            and hasattr(robot, "enable_torque")
        )
        if can_torque:
            robot.disable_torque()
        flush_pending_stdin()
        answer = prompt(text).strip().lower()
        if can_torque and robot.is_connected:
            robot.enable_torque()
        resume()
        return answer

    def _recover_robot(self, robot, err, play_sounds):
        """Bring the robot back after it dropped out mid-episode, returning whether it
        came back.

        `eval.robot_recovery_attempts` bounds the tries; 0 ends the session on the first
        drop. A robot that can rebuild its motor link alone -- b601's follower process,
        whose cameras never went down with it -- does that, since a full reconnect costs
        seconds of camera startup it does not need.
        """
        attempts = int(self.cfg.get("robot_recovery_attempts", 3) or 0)
        delay = float(self.cfg.get("robot_recovery_delay_s", 2.0))
        reason = robot.follower_error() if hasattr(robot, "follower_error") else None
        announce(f"Robot dropped out mid-episode: {reason or err}", YELLOW,
                 speak="Robot fault", play_sounds=play_sounds)
        for attempt in range(1, attempts + 1):
            time.sleep(delay)
            announce(f"  reconnecting ({attempt}/{attempts})...", YELLOW)
            try:
                if hasattr(robot, "restart_follower_process"):
                    robot.restart_follower_process()
                else:
                    with contextlib.suppress(Exception):
                        robot.disconnect()
                    robot.connect()
            except Exception as e:  # noqa: BLE001 - whatever it is, it gets another try
                err = e
                continue
            if robot.is_connected:
                announce("  robot is back; redoing the episode.", CYAN,
                         speak="Robot back", play_sounds=play_sounds)
                return True
        announce(f"  robot did not come back ({err}); stopping here. Everything scored so "
                 "far is saved, and eval.resume picks the session up.", YELLOW,
                 speak="Robot lost", play_sounds=play_sounds)
        return False

    @staticmethod
    def _hold_action(frame, action_names, obs_features):
        """The raw action that holds this arm where it stands.

        These followers command absolute joint targets, so holding means re-commanding the
        pose already measured, matched by feature name. A dimension the observation does
        not report -- a velocity action on a mobile base, say -- is relative and holds at
        zero instead.
        """
        names = list(obs_features["observation.state"]["names"])
        state = np.asarray(frame["observation.state"], dtype=np.float32)
        at = {name: i for i, name in enumerate(names)}
        return np.array([state[at[n]] if n in at else 0.0 for n in action_names],
                        dtype=np.float32)

    def _rollout(self, model, robot, observations=None, latency=None):
        from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
        from lerobot.utils.keyboard_input import init_keyboard_listener
        from lerobot.utils.robot_utils import precise_sleep

        cfg = self.cfg
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        action_names = hw_to_dataset_features(robot.action_features, "action")["action"]["names"]
        driver = self._make_driver(model, obs_features)

        fps = float(cfg.get("fps", 30))
        episodes = int(cfg.get("episodes", 10))
        max_steps = int(cfg.get("max_episode_steps", 600))
        play_sounds = not cfg.get("quiet", False)
        can_home = bool(cfg.get("home_between_episodes", True)) and hasattr(robot, "go_home")
        profile = open(cfg.profile_log, "w") if cfg.get("profile_log") else None

        if observations is not None:
            observations.track_features(obs_features, action_names)
        metric_stride = max(1, round(fps / float(cfg.get("stream_metrics_fps", 10))))

        # rollouts are recorded as a real LeRobotDataset; the writer's per-episode
        # h264 becomes the wandb video and nothing buffers in RAM
        recorder, record_root = None, None
        record_dataset = bool(cfg.get("record_dataset", True))
        resume_from = cfg.get("resume")
        if resume_from and not record_dataset:
            raise ValueError(
                "eval.resume needs eval.record_dataset=true: an interrupted session is "
                "resumed from the rollout dataset it left on disk."
            )
        if record_dataset:
            import os

            from lerobot.configs.video import RGBEncoderConfig
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            # errors-only for the SVT codecs, whose per-encode config banner goes out over
            # their own fprintf and so escapes quiet_console_logging
            os.environ.setdefault("SVT_LOG", "1")
            try:
                # save_episode's parquet write runs a datasets.Map with a progress bar
                import datasets

                datasets.disable_progress_bars()
            except Exception:  # noqa: BLE001 - cosmetic only
                pass
            if resume_from:
                record_root = Path(resume_from)
                if not (record_root / "meta").is_dir():
                    raise ValueError(
                        f"eval.resume={record_root} holds no rollout dataset to resume."
                    )
            else:
                record_root = Path(cfg.get("record_root") or
                                   Path("outputs/rollouts") / time.strftime("%Y%m%d_%H%M%S"))
            vcodec = cfg.get("record_vcodec")
            writer_threads = cfg.get("record_image_writer_threads")
            if writer_threads is None:
                writer_threads = len(getattr(getattr(robot, "config", None), "cameras", None) or {})
            writer_kwargs = dict(
                image_writer_threads=int(writer_threads),
                image_writer_processes=int(cfg.get("record_image_writer_processes") or 0),
                encoder_threads=cfg.get("record_encoder_threads"),
                streaming_encoding=bool(cfg.get("record_streaming_encoding", True)),
                rgb_encoder=RGBEncoderConfig(vcodec=vcodec) if vcodec else None,
                image_suffix=f".{cfg.get('record_image_format', 'jpg')}",
                jpeg_quality=int(cfg.get("record_jpeg_quality", 90)),
            )
            if resume_from:
                recorder = LeRobotDataset.resume(
                    "local/eval_rollouts", root=record_root, **writer_kwargs
                )
            else:
                features = {**obs_features,
                            **hw_to_dataset_features(robot.action_features, "action")}
                recorder = LeRobotDataset.create(
                    "local/eval_rollouts", fps=int(round(fps)), features=features,
                    root=record_root, **writer_kwargs,
                )
            announce(f"Recording rollouts to {record_root}", CYAN)
        record_task = str(cfg.get("record_task") or "eval rollout")

        quiet_console_logging()
        listener, events = init_keyboard_listener()
        install_graceful_sigint_handler(events)

        def home():
            from ...scripts.b601.common import start_homing
            return start_homing(robot, play_sounds)

        from ...utils.wandb import active_run_id

        run_id = active_run_id()
        rows = read_session_log(record_root).get("episodes", []) if resume_from else []
        _save_session_log(record_root, rows, run_id)
        videos = {}
        if rows:
            done = sum(r["success"] for r in rows)
            announce(f"Resuming at episode {len(rows) + 1}/{episodes} "
                     f"({done}/{len(rows)} successful so far)", CYAN)
        if model is not None:
            model.eval()
        try:
            if can_home:
                # the first episode starts from the zero pose too, not wherever the arm powered up
                home()()
            ep, first = len(rows), True
            while ep < episodes and not events["stop_recording"]:
                self._ask(listener, robot,
                          f"[episode {ep + 1}/{episodes}] reset the scene, then press ENTER ",
                          drop_torque=True)
                announce(
                    f"Episode {ep + 1}/{episodes}", CYAN,
                    speak=f"Episode {ep + 1}", play_sounds=play_sounds, bell=True,
                )
                if first:
                    announce("  [n / right] end episode   [r / left] discard & redo"
                             "   [q / esc] stop", CYAN)
                    first = False
                driver.reset(lambda frame: self._hold_action(frame, action_names, obs_features))
                events["exit_early"] = False
                steps, slow, t_start = 0, 0, time.perf_counter()
                fault = None
                for steps in range(1, max_steps + 1):
                    if events["exit_early"] or events["rerecord_episode"] or events["stop_recording"]:
                        break
                    t0 = time.perf_counter()
                    try:
                        frame = build_dataset_frame(obs_features, robot.get_observation(), "observation")
                        t_obs = time.perf_counter()
                        action = driver.step(frame)
                        t_pred = time.perf_counter()
                        robot.send_action(
                            {name: float(a) for name, a in zip(action_names, action, strict=True)}
                        )
                    except Exception as err:  # noqa: BLE001 - only a dropped robot is ours
                        # Anything raised while the robot is still up is a real failure
                        # (a driver/server error, a bad action); what this catches is the
                        # hardware going away mid-episode, handled below.
                        if robot.is_connected:
                            raise
                        fault = err
                        break
                    if recorder is not None:
                        recorder.add_frame({**frame,
                                            "action": np.asarray(action, dtype=np.float32),
                                            "task": record_task})
                    if observations is not None and steps % metric_stride == 0:
                        observations.push_step(frame, action)
                    if latency is not None:
                        spans = driver.take_latency()
                        if spans:
                            latency.push({f"replan.ms/{k}": v * 1e3 for k, v in spans.items()})
                    elapsed = time.perf_counter() - t0
                    if profile:
                        profile.write(
                            f"step={steps} obs={t_obs - t0:.4f} "
                            f"predict={t_pred - t_obs:.4f} tick={elapsed:.4f}\n"
                        )
                    slow += elapsed > 1 / fps
                    precise_sleep(max(0.0, 1 / fps - elapsed))
                events["exit_early"] = False
                if fault is not None:
                    # The arm stopped being commanded part-way through, so the episode is
                    # not a rollout of anything; drop it and redo it on a robot that is
                    # back, or end the session cleanly on one that is not.
                    if recorder is not None and recorder.has_pending_frames():
                        recorder.clear_episode_buffer()
                    if self._recover_robot(robot, fault, play_sounds):
                        # The stopped arm fell where it stood; walk it back to zero before
                        # the operator resets the scene around it.
                        if can_home:
                            try:
                                home()()
                            except Exception as err:  # noqa: BLE001 - as in the step loop
                                if robot.is_connected:
                                    raise
                                announce(f"homing after the fault failed: {err}", YELLOW)
                        continue
                    events["stop_recording"] = True
                    break
                if steps and slow / steps > SLOW_TICK_WARN_FRACTION:
                    announce(
                        f"Control rate: {slow}/{steps} ticks ran slower than fps={fps:g}; "
                        f"the policy saw a slower world than training did.", YELLOW,
                    )

                finish_homing = home() if can_home and not events["stop_recording"] else None
                discard = events["rerecord_episode"]
                events["rerecord_episode"] = False
                if not discard:
                    tag = self._ask(listener, robot,
                                    "success? [y/N] (r = discard & redo, q = tag failure and stop): ")
                    discard = tag == "r"
                if discard:
                    announce("Discarding episode.", YELLOW, speak="Discarding",
                             play_sounds=play_sounds)
                    if recorder is not None and recorder.has_pending_frames():
                        recorder.clear_episode_buffer()
                else:
                    rows.append({
                        "episode": ep,
                        "success": tag == "y",
                        "steps": steps,
                        "seconds": round(time.perf_counter() - t_start, 1),
                    })
                    if recorder is not None and recorder.has_pending_frames():
                        # encoding runs now, while the operator resets the scene
                        recorder.save_episode()
                    _save_session_log(record_root, rows, run_id)
                    if tag == "q":
                        events["stop_recording"] = True
                    ep += 1
                if finish_homing is not None:
                    try:
                        finish_homing()
                    except Exception as err:  # noqa: BLE001 - as in the step loop above
                        if robot.is_connected:
                            raise
                        if not self._recover_robot(robot, err, play_sounds):
                            events["stop_recording"] = True
        finally:
            driver.close()
            if recorder is not None:
                try:
                    if recorder.has_pending_frames():
                        recorder.clear_episode_buffer()
                    recorder.finalize()
                except Exception as err:  # noqa: BLE001 - never lose the eval over the recording
                    announce(f"rollout dataset finalize failed: {err}", YELLOW)
            if listener is not None:
                listener.stop()
            if profile is not None:
                profile.close()

        if recorder is not None and rows:
            announce(f"Rollout dataset: {record_root} ({len(rows)} episodes)", CYAN)
            for col in sorted(driver.visual):
                out = _concat_episode_videos(record_root, col)
                if out is not None:
                    videos[col.rsplit(".", 1)[-1]] = str(out)

        succ = [r["success"] for r in rows]
        metrics = {
            "success_rate": 100.0 * sum(succ) / max(1, len(succ)),
            "episodes": len(rows),
            "mean_episode_steps": float(np.mean([r["steps"] for r in rows])) if rows else 0.0,
        }
        return EvalResult(metrics=metrics, videos=videos, episodes=rows)
