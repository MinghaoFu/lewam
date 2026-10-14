#!/usr/bin/env python
"""Record a LeRobot dataset by teleoperating the reBot B601-DM follower with the
reBot Arm 102 leader. Invoked by record.sh, which resolves the arm ports and resets the
follower's control mode first.

A new task instruction can be chosen after every episode. Cameras come from the config's
`cameras` dict (see lewam_real_robot.utils.cameras.build_cameras), and the camera feeds
plus joint charts are served as an MJPEG page to any browser on the network. Between
episodes the follower ramps to its zero pose and waits for the leader to match.
`dry_run=true` teleoperates without writing a dataset.

Controls (arrow keys, or n/r/q over SSH):
  Right / n  -> end episode (or end the scene reset)
  Left  / r  -> discard and re-record episode
  Esc   / q  -> stop recording entirely
"""

import hashlib
import itertools
import json
import logging
import math
import re
import signal
import time
import warnings
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
import datasets
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.rebot_b601_follower import RebotB601Follower, RebotB601FollowerRobotConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.keyboard_input import TerminalKeyListener, init_keyboard_listener
from lerobot.utils.utils import log_say
from lewam_real_robot.scripts.b601.common import (
    CYAN,
    GREEN,
    MAGENTA,
    YELLOW,
    announce,
    confirm,
    confirm_phrase,
    flush_pending_stdin,
    install_graceful_sigint_handler,
    prompt,
    quiet_console_logging,
    start_homing,
)
from lewam_real_robot.utils.cameras import build_cameras
from lewam_real_robot.utils.stream import CameraPanel, LeRobotObservationPanel, start_stream
from lewam_real_robot.utils.stderr_filter import CORRUPT_JPEG, LIBAV_TAGGED, suppress_stderr_lines
from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class RecordConfig:
    """Schema + defaults for record.py, registered with Hydra as `record_schema`.

    Per-dataset config files under configs/ compose on top of this (`defaults:
    - record_schema`) and only need to set what differs from these defaults --
    typically repo_id, task, and cameras, so recording settings stay consistent
    across sessions of the same dataset. repo_id and the port fields have no
    default: repo_id must come from the dataset config, the ports are filled
    in by record.sh at invocation time.
    """

    repo_id: str = MISSING  # <hf_username>/<dataset_name>
    root: str | None = None
    task: str | None = None  # omit to be prompted before episode 1
    repeat_task: bool = False
    # how many episodes the DATASET ends up with, not how many to record this session
    num_episodes: int = 50
    # None runs unbounded, ending on a keypress; a float caps the phase in seconds
    episode_time_s: float | None = None
    reset_time_s: float | None = None
    fps: int = 30
    control_fps: int | None = 60  # must be a whole multiple of fps

    # `{camera_key: {type: <lerobot backend>, ...}}`; each key names an observation
    # stream, and the remaining fields are that backend's own (utils/cameras.py)
    cameras: dict[str, Any] = field(default_factory=dict)

    resume: bool = True
    private: bool = False
    # encode as frames arrive rather than at episode end: save_episode() becomes
    # near-instant, at the cost of CPU during the episode
    streaming_encoding: bool = True
    encoder_threads: int | None = None

    # lerobot's own default (libsvtav1/AV1) is far more CPU-expensive on the Pi
    vcodec: str = "h264"
    image_writer_threads: int | None = None
    image_writer_processes: int | None = None
    image_format: str = "jpg"  # png | jpg
    jpeg_quality: int = 90  # 0-100

    # failing to start the stream never stops a recording
    stream_cameras: bool = True
    stream_port: int = 8090
    stream_fps: int = 10
    stream_height: int = 240  # per camera, before the panels are stacked
    stream_quality: int = 85  # 0-100

    stream_metrics: bool = True
    stream_metrics_fps: int = 10
    stream_metrics_window: int = 150  # rows, so 15 s at 10 Hz

    # teleoperate under this exact config writing nothing; the dataset half is ignored
    dry_run: bool = False

    quiet: bool = False
    profile_log: str | None = None

    smoothing_time_constant: float | None = 0.02  # seconds
    no_smoothing: bool = False

    home_between_episodes: bool = True
    home_tolerance_deg: float = 10.0
    # below ~1.5s the ease's peak speed is felt as a jolt
    home_duration_s: float = 1.25
    # per-joint override, for a joint that would swing into frame on its way to zero
    home_joint_durations_s: dict[str, float] = field(
        default_factory=lambda: {"elbow_flex": 2.0}
    )

    # supplied by record.sh, which resolves these first
    follower_port: str = MISSING
    leader_port: str = MISSING
    follower_id: str = "b601_follower"
    leader_id: str = "b601_leader"
    control_mode: str = "mit"  # mit | pos_vel


ConfigStore.instance().store(name="record_schema", node=RecordConfig)


# report the control rate only when this share of an episode's ticks ran slower
# than fps: an isolated slow tick is absorbed by the fixed-rate dispatch
SLOW_TICK_WARN_FRACTION = 0.30


class SlowTickFilter(logging.Filter):
    """Swallows record_loop's per-tick "running slower" warnings and counts them,
    so main() can report one summary line per episode instead of a wall of
    identical warnings.

    Counted two ways, because they mean different things: record_loop warns
    whenever a tick misses the *control* rate (60 Hz here), which happens
    constantly and harmlessly, while a tick slower than the rate frames are
    actually recorded at is what threatens the dataset."""

    RATE_RE = re.compile(r"Record loop is running slower \(([\d.]+) Hz\)")

    def __init__(self, fps: int):
        super().__init__()
        self.fps = fps
        self.below_fps = 0
        self.below_control = 0

    def filter(self, record: logging.LogRecord) -> bool:
        match = self.RATE_RE.search(record.getMessage())
        if match is None:
            return True
        self.below_control += 1
        if float(match.group(1)) < self.fps:
            self.below_fps += 1
        return False

    def take_counts(self) -> tuple[int, int]:
        counts = (self.below_fps, self.below_control)
        self.below_fps = self.below_control = 0
        return counts


def report_control_rate(
    label: str, below_fps: int, below_control: int, ticks: int, fps: int, control_fps: int
) -> None:
    """Always reported, so a rate that is quietly drifting is visible before it
    becomes a problem; highlighted only once enough ticks missed the recording
    rate to actually threaten the data (see SLOW_TICK_WARN_FRACTION)."""
    if ticks <= 0:
        announce(f"{label}: no ticks to report.")
        return
    share = below_fps / ticks
    warn = share >= SLOW_TICK_WARN_FRACTION
    announce(
        f"{label}: {below_fps}/{ticks} ticks ({share:.1%}) below the {fps} Hz record rate, "
        f"{below_control} below the {control_fps} Hz control rate."
        + (" Frames may have been dropped." if warn else ""),
        YELLOW if warn else "",
    )


def teleoperate(cfg, robot, teleop, processors, observations, play_sounds, slow_tick_filter):
    """`dry_run=true`: drive the arms under this config and record nothing.

    record_loop with no dataset is already exactly teleoperation -- it is how the
    between-episode scene reset runs -- so the rehearsal goes through the same code
    path a real episode does, minus the writing.
    """
    teleop_action_processor, robot_action_processor, robot_observation_processor = processors
    listener = None
    stream = None
    try:
        stream, listener, events = connect_and_watch(cfg, robot, teleop, observations)

        # the first episode starts from the zero pose too, not wherever the arm powered up
        if cfg.home_between_episodes and not home_and_wait(
            cfg, robot, teleop, events, play_sounds
        ):
            events["stop_recording"] = True

        announce("Dry run: teleoperating, nothing is being recorded.", CYAN,
                 speak="Dry run", play_sounds=play_sounds, bell=True)
        announce("  [q / esc] stop   [n / right] re-home and carry on", CYAN)

        control_fps = cfg.control_fps or cfg.fps
        while not events["stop_recording"]:
            slow_tick_filter.take_counts()
            started = time.perf_counter()
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=math.inf,
                display_data=False,
                control_fps=cfg.control_fps,
            )
            below_fps, below_control = slow_tick_filter.take_counts()
            report_control_rate(
                "Control rate", below_fps, below_control,
                round((time.perf_counter() - started) * control_fps), cfg.fps, control_fps,
            )
            # anything but a stop means re-home and carry on, so the arms can be re-zeroed
            # without restarting
            if events["stop_recording"]:
                break
            events["exit_early"] = False
            if cfg.home_between_episodes and not home_and_wait(
                cfg, robot, teleop, events, play_sounds
            ):
                break
    finally:
        announce("Stopping.", YELLOW)
        log_say("Stop teleoperating", play_sounds, blocking=True)
        stop_arms(robot, teleop, stream, listener)
        print("\nDry run finished. Nothing was written.")


def connect_and_watch(cfg: RecordConfig, robot, teleop, observations):
    """Bring up the arms, the page watching them, and the keyboard, in that order --
    the stream peeks at cameras that connect() is what opens.

    Returns (stream, listener, events). Anything raising part-way leaves the caller
    with nothing to tear down: the stream and listener come last and are returned
    together or not at all.
    """
    robot.connect()
    teleop.connect()
    stream = start_watch_stream(cfg, robot, observations)
    listener, events = init_keyboard_listener()
    install_graceful_sigint_handler(events)
    return stream, listener, events


def stop_arms(robot, teleop, stream, listener):
    """Undo connect_and_watch, in the order the hardware requires.

    The stream peeks at the cameras, so it goes down before disconnect() closes them
    -- read_latest() raises once they are gone. disconnect() drops torque and clears
    any latched motor fault, and an impatient repeat Ctrl-C interrupting it mid-call
    is how that fault gets left stuck, so SIGINT is ignored until it returns.
    """
    if stream is not None:
        stream.close()
    previous_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
    if listener is not None:
        listener.stop()


def home_and_wait(cfg: RecordConfig, robot, teleop, events, play_sounds) -> bool:
    """Ramp the follower to its zero pose and wait for the leader to be brought there
    too. False if the session was stopped while waiting.

    Without it the next send_action snaps the follower across the workspace to meet
    wherever the leader happens to be resting.

    The ramp runs against the operator's own reset rather than before it: it is the
    follower process that drives it, and waiting on the leader only reads the leader,
    so the two cost whichever is slower instead of both. Joined before returning, so
    the next episode can never start mid-ramp.
    """
    finish_homing = start_homing(robot, play_sounds)
    homed = wait_for_leader_home(teleop, events, cfg.home_tolerance_deg, play_sounds)
    finish_homing()
    if not homed:
        return False
    events["exit_early"] = False
    return True


def start_watch_stream(cfg: RecordConfig, robot, observations):
    """The camera feed and the state/action charts on one page, or None if neither is
    wanted. Must be called after robot.connect(), which is what opens the cameras."""
    panels = []
    if cfg.stream_cameras and robot.cameras:
        panels.append(
            CameraPanel(
                robot.cameras,
                height=cfg.stream_height,
                fps=cfg.stream_fps,
                quality=cfg.stream_quality,
            )
        )
    if observations is not None:
        panels.append(observations)
    if not panels:
        return None
    stream = start_stream(panels, port=cfg.stream_port)
    if stream is not None:
        announce(f"Watch the rig: {stream.url}", CYAN)
    return stream


def build_arms(cfg: RecordConfig, cameras: dict):
    """The follower and leader a session drives, configured from `cfg`.

    Shared with teleop.py so a dry run puts the arms in exactly the state recording
    will: control mode, smoothing and gains all decide what a given action produces,
    so a rehearsal under different ones proves nothing about the session that follows.
    """
    smoothing_kwargs = {}
    if cfg.no_smoothing:
        smoothing_kwargs["enable_trajectory_smoothing"] = False
    if cfg.smoothing_time_constant is not None:
        smoothing_kwargs["smoothing_time_constant_s"] = cfg.smoothing_time_constant
    robot = RebotB601Follower(
        RebotB601FollowerRobotConfig(
            port=cfg.follower_port,
            id=cfg.follower_id,
            can_adapter="damiao",
            control_mode=cfg.control_mode,
            cameras=cameras,
            home_duration_s=cfg.home_duration_s,
            home_joint_durations_s=cfg.home_joint_durations_s,
            # the follower runs in a separate OS process and cannot share this process's
            # --profile-log FileHandler, so it gets the path and attaches its own
            profile_log_path=cfg.profile_log,
            **smoothing_kwargs,
        )
    )
    teleop = RebotArm102Leader(
        RebotArm102LeaderTeleopConfig(port=cfg.leader_port, id=cfg.leader_id)
    )
    return robot, teleop


def observation_panel(cfg: RecordConfig, processors: tuple):
    """A panel charting the follower's state against the leader's commanded action,
    plus the processors rewired to feed it.

    record_loop only ever calls its processors, so tapping them needs no hook in it
    and no change to lerobot. Returns (panel, processors) unchanged when the config
    has the charts turned off.
    """
    teleop_action_processor, robot_action_processor, robot_observation_processor = processors
    if not cfg.stream_metrics:
        return None, processors

    panel = LeRobotObservationPanel(window=cfg.stream_metrics_window)
    stride = max(1, round((cfg.control_fps or cfg.fps) / max(1, cfg.stream_metrics_fps)))
    latest_obs: dict = {}
    ticks = itertools.count()

    def see_observation(obs):
        latest_obs.clear()
        latest_obs.update(obs)

    def see_action(action):
        # the observation is processed first each tick, so it is the state this action
        # was taken from rather than the one after it
        if next(ticks) % stride == 0:
            panel.push_raw(latest_obs, action)

    return panel, (
        tap(teleop_action_processor, see_action),
        robot_action_processor,
        tap(robot_observation_processor, see_observation),
    )


def tap(processor, sink):
    """Wrap a lerobot processor so something else sees what flows through it.

    record_loop only ever calls these, so a tap needs no hook in it and no change to
    lerobot: the value is handed to `sink` and then returned untouched. A failing sink
    would take the recording with it, so it must not be able to raise.
    """
    def tapped(*args, **kwargs):
        value = processor(*args, **kwargs)
        sink(value)
        return value

    return tapped


def jsonable(value: Any) -> Any:
    """Best-effort conversion of a config value to something json can write.

    Configs hold dataclasses, Paths and tuples; anything else unexpected is
    stringified rather than raising, since failing to record the settings must
    never be what ends a recording session.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def recording_settings(robot: RebotB601Follower, teleop: RebotArm102Leader) -> dict:
    """The arm settings this session records under, for meta/info.json.

    Gains, control mode and gravity compensation decide what state a given
    action actually produces, so episodes recorded under different settings are
    not interchangeable however alike their frames look. Recording it makes that
    checkable after the fact.
    """
    return {
        "robot": {"type": robot.name, "id": robot.id, **jsonable(robot.config)},
        "teleop": {"type": teleop.name, "id": teleop.id, **jsonable(teleop.config)},
    }


# settings that change what a recorded action MEANS, so a resumed dataset
# differing on one of these is mixing regimes
CONTROL_KEYS = (
    "control_mode",
    "gripper_control_mode",
    "mit_kp",
    "mit_kd",
    "gripper_mit_kp",
    "gripper_mit_kd",
    "pos_vel_velocity",
    "gripper_torque_ratio",
    "gravity_compensation",
    "gravity_gain",
    "gravity_scale",
    "gravity_payload_kg",
    "gravity_payload_com",
    "gravity_max_torque",
    "enable_trajectory_smoothing",
    "smoothing_time_constant_s",
    "max_relative_target",
    "joint_limits",
    "send_rate_hz",
)


def control_differences(stored: dict, current: dict) -> list[str]:
    """Which control-relevant settings a resumed dataset disagrees with."""
    was = (stored or {}).get("robot", {})
    now = (current or {}).get("robot", {})
    return [key for key in CONTROL_KEYS if key in was and was[key] != now.get(key)]


def format_duration(seconds: float) -> str:
    """Seconds as hh:mm:ss. Hours are not wrapped or capped, so a dataset that
    passes 99 hours reads 123:01:05 rather than starting over."""
    total = int(seconds)
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def recorded_duration(dataset: LeRobotDataset) -> str:
    """How much footage the whole dataset holds, hh:mm:ss.

    Frames over fps, so this is the duration of what was recorded rather than
    how long the session took -- scene resets and task prompts are not in it.
    """
    return format_duration(dataset.num_frames / dataset.fps)


def pending_frames(dataset: LeRobotDataset) -> int:
    """How many frames the current episode has buffered but not yet saved. Used
    to size the control-rate report; any failure to read it just means no
    report, so it is never worth raising over."""
    try:
        return int(dataset.writer.episode_buffer["size"])
    except (AttributeError, KeyError, TypeError):
        return 0


def wait_for_leader_home(teleop: RebotArm102Leader, events: dict, tolerance_deg: float, play_sounds: bool) -> bool:
    """Block until every leader joint sits within `tolerance_deg` of the zero
    pose the follower was just homed to. Returns False if the session was
    stopped while waiting.

    Without this the next episode's first send_action snaps the follower from
    home to wherever the leader happens to be resting, which is violent enough
    to move the scene, and is recorded as an opening transition no policy
    trained on the data could reproduce."""
    announce(
        f"Move the leader arm back to its home (zero) pose, within {tolerance_deg:.0f}deg on every joint.",
        MAGENTA,
        speak="Return the leader to home",
        play_sounds=play_sounds,
    )
    status_width = 0
    while not events["stop_recording"]:
        offsets = {
            key.removesuffix(".pos"): abs(value)
            for key, value in teleop.get_action().items()
            if key.endswith(".pos")
        }
        if not offsets:
            return True
        worst_joint, worst = max(offsets.items(), key=lambda item: item[1])
        if worst <= tolerance_deg:
            print("\r" + " " * status_width + "\r", end="")
            announce("Leader is home.", GREEN)
            return True
        off_count = sum(1 for value in offsets.values() if value > tolerance_deg)
        status = f"  {off_count} joint(s) out: worst {worst_joint} at {worst:.1f}deg"
        status_width = max(status_width, len(status))
        print(f"\r{status.ljust(status_width)}", end="", flush=True)
        time.sleep(0.1)
    print()
    return False


# named from the local side (HUB_BEHIND = the Hub trails us). Only the first three
# can be pushed without risking data that exists nowhere else.
HUB_NO_REPO = "no_repo"
HUB_IN_SYNC = "in_sync"
HUB_BEHIND = "behind"
HUB_AHEAD = "ahead"
HUB_DIVERGED = "diverged"
HUB_UNKNOWN = "unknown"
SAFE_TO_PUSH = (HUB_NO_REPO, HUB_IN_SYNC, HUB_BEHIND)

# where an episode's bytes were filed, not what they are: re-chunking rewrites
# them without changing a recorded frame, so they stay out of the fingerprints
LAYOUT_COLUMN_SUFFIXES = ("chunk_index", "file_index")


class HubUnreachable(Exception):
    """The Hub copy may exist but could not be read, so its state is unknown."""


@dataclass(frozen=True)
class RemoteEpisodes:
    """What the Hub holds for a repo, read at a single commit."""

    revision: str  # full commit sha the fingerprints were read at
    fingerprints: list[str]


@dataclass(frozen=True)
class HubComparison:
    """Where the local recording folder stands against its Hub copy at upload time."""

    status: str
    local_episodes: int
    remote_episodes: int | None = None
    revision: str | None = None  # Hub commit the comparison was made against
    first_divergent_episode: int | None = None  # episode_index the two stop agreeing at
    error: str | None = None

    @property
    def safe_to_push(self) -> bool:
        return self.status in SAFE_TO_PUSH


def episode_metadata_files(root: Path) -> list[Path]:
    """The per-episode metadata parquet files of a lerobot dataset rooted at `root`."""
    return sorted((root / "meta" / "episodes").rglob("*.parquet"))


def episode_fingerprints(paths: list[Path]) -> list[str]:
    """One content hash per episode, ordered by episode_index.

    Hashes lerobot's own per-episode metadata row: length, tasks, frame range, video
    timestamps, and the per-feature statistics (min/max/mean/std/quantiles) of every
    recorded signal, cameras included. Those statistics are computed from the frames
    themselves, so two recordings hash alike only if they hold the same data -- which is
    what makes this an identity check rather than an educated guess. Both sides of a
    comparison are read through this same parquet path, so the values being hashed are
    the same doubles bit for bit.
    """
    rows: list[dict] = []
    for path in paths:
        table = pq.read_table(path)
        content = [name for name in table.schema.names if not name.endswith(LAYOUT_COLUMN_SUFFIXES)]
        rows.extend(table.select(content).to_pylist())
    rows.sort(key=lambda row: row["episode_index"])
    return [hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() for row in rows]


def fetch_remote_episodes(repo_id: str) -> RemoteEpisodes | None:
    """Per-episode fingerprints of `repo_id` on the Hub, or ``None`` if no such dataset repo
    exists or one exists but was never pushed to. Raises HubUnreachable when the repo may
    hold data that could not be read, so a network failure is never read as an empty Hub.

    Everything is read at one commit, resolved up front: the Hub repo is a git repo that
    can be pushed to while this runs, and a comparison spread across two commits could
    describe a state that never existed. Only the small meta/episodes parquet files are
    downloaded, into the Hub's own cache, so this never touches the local recording folder
    and is safe to call for a repo we are writing to.
    """
    api = HfApi()
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
        files = api.list_repo_files(repo_id, repo_type="dataset", revision=info.sha)
    except RepositoryNotFoundError:
        return None
    except Exception as e:
        raise HubUnreachable(str(e)) from e
    if info.sha is None:
        raise HubUnreachable("the Hub reported no commit for this repo")
    revision = info.sha

    episode_files = sorted(
        f for f in files if f.startswith("meta/episodes/") and f.endswith(".parquet")
    )
    if not episode_files:
        # an empty repo was created but never pushed to, and is safe to push to; metadata
        # with no findable episodes is a layout this script does not know
        if not any(f.startswith("meta/") for f in files):
            return None
        raise HubUnreachable(f"no meta/episodes/*.parquet at commit {revision[:7]}")

    try:
        downloaded = [
            Path(
                hf_hub_download(
                    repo_id=repo_id, filename=f, repo_type="dataset", revision=revision
                )
            )
            for f in episode_files
        ]
        return RemoteEpisodes(revision, episode_fingerprints(downloaded))
    except Exception as e:
        raise HubUnreachable(str(e)) from e


def compare_with_hub(dataset: LeRobotDataset, repo_id: str) -> HubComparison:
    """Classify the local recording folder against its Hub copy, episode by episode.

    push_to_hub uploads the whole local folder through upload_folder, which overwrites
    files sharing a path and never deletes remote-only ones. Appending is therefore only
    safe when the Hub holds exactly a prefix of what we hold: otherwise a push either
    replaces remote episodes with unrelated local ones, or strands remote data files under
    metadata that no longer references them.

    "Exactly a prefix" is settled by comparing per-episode content fingerprints (see
    episode_fingerprints), so a second machine recording into the same repo_id is caught
    however closely its episodes happen to line up with ours in count and length.

    Never raises: this runs while the session is already shutting down, and a comparison
    that cannot be made is reported as HUB_UNKNOWN -- which gates the push behind the typed
    confirmation -- rather than replacing whatever ended the session.
    """
    try:
        local = episode_fingerprints(episode_metadata_files(Path(dataset.root)))
    except Exception as e:
        return HubComparison(
            HUB_UNKNOWN, int(dataset.meta.total_episodes), error=f"local metadata unreadable: {e}"
        )
    try:
        remote = fetch_remote_episodes(repo_id)
    except Exception as e:
        return HubComparison(HUB_UNKNOWN, len(local), error=str(e))
    if remote is None:
        return HubComparison(HUB_NO_REPO, len(local))

    shared = min(len(local), len(remote.fingerprints))
    first_divergent = next(
        (i for i in range(shared) if local[i] != remote.fingerprints[i]), None
    )
    if first_divergent is not None:
        status = HUB_DIVERGED
    elif len(remote.fingerprints) > len(local):
        status = HUB_AHEAD
    elif len(remote.fingerprints) == len(local):
        status = HUB_IN_SYNC
    else:
        status = HUB_BEHIND
    return HubComparison(
        status,
        len(local),
        remote_episodes=len(remote.fingerprints),
        revision=remote.revision,
        first_divergent_episode=first_divergent,
    )


def describe_hub_comparison(c: HubComparison, repo_id: str, recorded_episodes: int) -> list[str]:
    """The three counts that matter -- what is local, what is on the Hub, what this session
    added -- followed by what uploading would actually do. Kept separate from the prompt so
    the wording can be exercised without a recording session."""
    on_hub = {HUB_NO_REPO: "none yet", HUB_UNKNOWN: "unknown"}.get(c.status, str(c.remote_episodes))
    lines = [
        f"Episodes -- local: {c.local_episodes} | on the Hub: {on_hub} | "
        f"recorded this session: {recorded_episodes}"
    ]
    if c.status == HUB_NO_REPO:
        lines.append(f"Uploading creates '{repo_id}' with all {c.local_episodes} local episode(s).")
        return lines
    if c.status == HUB_UNKNOWN:
        lines.append(f"Could not compare '{repo_id}' with its Hub copy: {c.error}")
        lines.append("Uploading would push the whole local dataset blind, over whatever is there.")
        return lines
    if c.remote_episodes is None:
        return lines

    at = f"at commit {c.revision[:7]}" if c.revision else "on the Hub"
    if c.status == HUB_IN_SYNC:
        lines.append(
            f"The Hub {at} holds these same {c.remote_episodes} episode(s), frame statistics and "
            "all. Uploading changes nothing unless a previous push was interrupted partway."
        )
    elif c.status == HUB_BEHIND:
        lines.append(
            f"The Hub {at} holds exactly the first {c.remote_episodes} of these episodes. "
            f"Uploading pushes the whole local dataset, so the Hub gains "
            f"{c.local_episodes - c.remote_episodes} episode(s), for {c.local_episodes} total."
        )
    elif c.status == HUB_AHEAD:
        lines.append(
            f"WARNING: the Hub {at} has {c.remote_episodes - c.local_episodes} episode(s) this "
            f"local copy does not. Uploading would strand their files on the Hub under "
            f"{c.local_episodes}-episode metadata, corrupting the dataset."
        )
    elif c.status == HUB_DIVERGED and c.first_divergent_episode == 0:
        lines.append(
            f"WARNING: the Hub {at} holds a different recording -- the two disagree from their "
            "very first episode. Uploading would overwrite the Hub's."
        )
    elif c.status == HUB_DIVERGED:
        lines.append(
            f"WARNING: this copy and the Hub {at} share their first {c.first_divergent_episode} "
            f"episode(s) but hold different recordings from episode_index "
            f"{c.first_divergent_episode} on. Uploading would overwrite the Hub's from there."
        )
    return lines


def archive_dataset_root(dataset_root: Path) -> Path:
    """Move a recording folder aside rather than deleting it, returning where it went.
    Episodes that were never uploaded exist nowhere else, so "start fresh" has to stay
    recoverable; deleting the archive once the data is known to be junk is a manual step."""
    archived = dataset_root.with_name(
        f"{dataset_root.name}.superseded-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    dataset_root.rename(archived)
    print(f"Moved the existing dataset aside to {archived}")
    return archived


def pick_task(task_history: list[str], current_task: str | None) -> str:
    while True:
        print("\n" + "=" * 50)
        print("TASK SELECTION")
        if current_task:
            print(f"  Current: {current_task}")
        if task_history:
            print("  Previous tasks:")
            for i, t in enumerate(task_history):
                print(f"    [{i + 1}] {t}")
        print("  [Enter] Keep current task" if current_task else "  Type a new task description")
        if task_history:
            print("  [number] Select a previous task")
        print("  [text] Type a new task")
        print("=" * 50)

        choice = prompt("Task> ").strip()

        if not choice:
            if current_task:
                return current_task
            print("No task set. Please type a task description.")
            continue

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(task_history):
                return task_history[idx]
            print(f"Invalid number. Choose 1-{len(task_history)}.")
            continue

        if len(choice) < 5:
            print("Task too short, please enter a real description.")
            continue

        return choice


@hydra.main(version_base=None, config_path="configs", config_name="record")
def main(cfg: RecordConfig) -> None:
    suppress_stderr_lines([CORRUPT_JPEG, LIBAV_TAGGED])
    # kills the per-episode "Map: ..." bar without touching the Hub's upload bars.
    # Not enough alone: resuming reaches the Hub client, which re-enables them, so
    # record.sh also exports HF_DATASETS_DISABLE_PROGRESS_BARS=1, which outranks it.
    warnings.filterwarnings("ignore", message="Cannot enable progress bars")
    datasets.disable_progress_bars()
    if cfg.image_format not in ("png", "jpg"):
        raise ValueError(f"image_format must be 'png' or 'jpg', got {cfg.image_format!r}.")
    if cfg.control_mode not in ("mit", "pos_vel"):
        raise ValueError(f"control_mode must be 'mit' or 'pos_vel', got {cfg.control_mode!r}.")

    play_sounds = not cfg.quiet
    slow_tick_filter = SlowTickFilter(cfg.fps)
    quiet_console_logging(slow_tick_filter)
    # record_loop() ends on a keypress unless a cap was configured; `while timestamp
    # < inf` never exits on its own
    episode_time_s = cfg.episode_time_s if cfg.episode_time_s is not None else math.inf
    reset_time_s = cfg.reset_time_s if cfg.reset_time_s is not None else math.inf

    if cfg.profile_log:
        # attached to the root logger so the "running slower" warnings land in the same
        # file as the per-step breakdown, and the robot's own per-substep debug logs too
        root_logger = logging.getLogger()
        handler = logging.FileHandler(cfg.profile_log)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        root_logger.addHandler(handler)
        # importing lerobot triggers an implicit logging.basicConfig(), adding a NOTSET
        # StreamHandler that would leak the DEBUG levels set below to the console.
        # Cap any such handler at WARNING so the extra detail only goes to the file.
        for h in root_logger.handlers:
            if h is not handler and isinstance(h, logging.StreamHandler):
                h.setLevel(logging.WARNING)
        for logger_name in (
            "lerobot.record_loop_profile",
            "lerobot.robots.rebot_b601_follower.rebot_b601_follower",
            "lerobot.teleoperators.rebot_102_leader.rebot_102_leader",
        ):
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
        print(f"Profiling record_loop() to {cfg.profile_log}")

    cameras = build_cameras(cfg.cameras)
    print("Cameras:")
    for name, cam in cameras.items():
        print(f"  {name}: {cam}")
    if not cameras:
        print("  none configured -- this dataset will hold robot state only")
    # caught here rather than as dropped frames mid-episode: a camera slower than
    # the dataset's rate cannot feed every recorded step
    too_slow = {name: cam.fps for name, cam in cameras.items() if cam.fps is not None and cam.fps < cfg.fps}
    if too_slow:
        raise ValueError(
            f"cameras {too_slow} run below the dataset's fps={cfg.fps}. "
            "Raise their fps in the `cameras` config, or lower fps."
        )
    robot, teleop = build_arms(cfg, cameras)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    if cfg.dry_run:
        # before anything touches the dataset root: a rehearsal must not create, resume
        # or archive a folder
        observations, processors = observation_panel(
            cfg,
            (teleop_action_processor, robot_action_processor, robot_observation_processor),
        )
        teleoperate(cfg, robot, teleop, processors, observations, play_sounds, slow_tick_filter)
        return

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    # tapped after the dataset features are derived from the originals, since those
    # helpers inspect the pipeline rather than call it
    observations, (
        teleop_action_processor,
        robot_action_processor,
        robot_observation_processor,
    ) = observation_panel(
        cfg,
        (teleop_action_processor, robot_action_processor, robot_observation_processor),
    )

    # these knobs only matter with --image-format=png: more workers drains the write
    # queue faster but starves the CAN read, fewer backs the queue up into save_episode()
    image_writer_processes = cfg.image_writer_processes if cfg.image_writer_processes is not None else 0
    image_writer_threads = (
        cfg.image_writer_threads if cfg.image_writer_threads is not None else len(cameras)
    )
    rgb_encoder = RGBEncoderConfig(vcodec=cfg.vcodec) if cfg.vcodec else None
    image_suffix = f".{cfg.image_format}"
    dataset_root = Path(cfg.root) if cfg.root else HF_LEROBOT_HOME / cfg.repo_id
    create_kwargs = dict(
        root=cfg.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        encoder_threads=cfg.encoder_threads,
        streaming_encoding=cfg.streaming_encoding,
        rgb_encoder=rgb_encoder,
        image_suffix=image_suffix,
        jpeg_quality=cfg.jpeg_quality,
    )

    def resume_dataset() -> LeRobotDataset:
        return LeRobotDataset.resume(
            cfg.repo_id,
            root=str(dataset_root),
            encoder_threads=cfg.encoder_threads,
            streaming_encoding=cfg.streaming_encoding,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            rgb_encoder=rgb_encoder,
            image_suffix=image_suffix,
            jpeg_quality=cfg.jpeg_quality,
        )

    if dataset_root.exists():
        # LeRobotDataset.create() requires a brand-new root and would raise FileExistsError
        # here. An incomplete folder fails metadata loading the same way a corrupt one does:
        # LeRobotDatasetMetadata falls back to a Hub lookup, which 404s if never pushed.
        try:
            existing_episodes = LeRobotDatasetMetadata(cfg.repo_id, root=dataset_root).total_episodes
        except Exception as e:
            print(f"\nFound an existing folder at {dataset_root}, but couldn't read its metadata: {e}")
            print(
                "That folder is either an incomplete leftover or a real dataset whose metadata "
                "is temporarily unreadable, and the two look identical from here."
            )
            if not confirm("Move it aside and start a fresh recording?"):
                print("Leaving the existing folder in place. Exiting.")
                raise SystemExit(1)
            archive_dataset_root(dataset_root)
            dataset = LeRobotDataset.create(cfg.repo_id, cfg.fps, **create_kwargs)
        else:
            if cfg.resume:
                dataset = resume_dataset()
            else:
                print(
                    f"\nDataset '{cfg.repo_id}' already has {existing_episodes} episode(s) at "
                    f"{dataset_root}."
                )
                if confirm("Resume and append to it?"):
                    dataset = resume_dataset()
                else:
                    # declining resume starts a new dataset at the same root, so the old one has to
                    # move first -- archived, not deleted, since it may never have been pushed
                    archive_dataset_root(dataset_root)
                    dataset = LeRobotDataset.create(cfg.repo_id, cfg.fps, **create_kwargs)
    else:
        dataset = LeRobotDataset.create(cfg.repo_id, cfg.fps, **create_kwargs)

    listener = None
    camera_stream = None
    # declared out here so the finally block can report them however the session
    # ended, including before the recording loop is ever reached
    session_rate = {"below_fps": 0, "below_control": 0, "ticks": 0}
    recorded_episodes = 0
    try:
        camera_stream, listener, events = connect_and_watch(cfg, robot, teleop, observations)

        settings = recording_settings(robot, teleop)
        differences = control_differences(dataset.meta.robot_config, settings)
        if differences:
            announce(
                f"This dataset's episodes were recorded with different {', '.join(differences)}. "
                "Appending mixes control regimes: the same action produces a different state "
                "under each, which no policy trained on the mix can tell apart.",
                YELLOW,
            )
            if not confirm("Append anyway?"):
                raise SystemExit("Not appending. Start a separate dataset, or restore the settings.")
        elif dataset.meta.robot_config is None:
            dataset.meta.robot_config = settings

        task_history: list[str] = []
        current_task = cfg.task
        if current_task:
            task_history.append(current_task)

        if dataset.num_episodes >= cfg.num_episodes:
            announce(
                f"'{cfg.repo_id}' already has {dataset.num_episodes} of {cfg.num_episodes} episodes, "
                "so there is nothing to record. Raise num_episodes to collect more.",
                YELLOW,
            )
            events["stop_recording"] = True

        # the first episode starts from the zero pose too, not wherever the arm powered up
        if (
            cfg.home_between_episodes
            and not events["stop_recording"]
            and not home_and_wait(cfg, robot, teleop, events, play_sounds)
        ):
            events["stop_recording"] = True

        with VideoEncodingManager(dataset):
            # episode numbering follows the dataset, not the session, so a resumed dataset
            # picks up where it left off
            while dataset.num_episodes < cfg.num_episodes and not events["stop_recording"]:
                if not (cfg.repeat_task and current_task):
                    # TerminalKeyListener (the headless fallback) puts stdin into no-echo cbreak
                    # mode and never releases it, so input() here would be unechoed. Only that
                    # listener supports being restarted like this; a pynput.Listener cannot be.
                    is_terminal_listener = isinstance(listener, TerminalKeyListener)
                    if is_terminal_listener:
                        listener.stop()
                    # toss keys typed during teleop/reset so they cannot appear in the task text
                    flush_pending_stdin()
                    # an enabled-but-uncommanded motor faults on its own comm-timeout, and
                    # pick_task()'s input() can block indefinitely, so drop torque for the prompt
                    robot.disable_torque()
                    current_task = pick_task(task_history, current_task)
                    robot.enable_torque()
                    if is_terminal_listener:
                        listener.start()
                    if current_task not in task_history:
                        task_history.append(current_task)

                announce(
                    f"Episode {dataset.num_episodes + 1}/{cfg.num_episodes}: {current_task}",
                    CYAN,
                    speak=f"Recording episode {dataset.num_episodes + 1}",
                    play_sounds=play_sounds,
                    bell=True,
                )
                if recorded_episodes == 0:
                    announce("  [n / right] finish   [r / left] re-record   [q / esc] stop", CYAN)
                slow_tick_filter.take_counts()
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=episode_time_s,
                    single_task=current_task,
                    display_data=False,
                    control_fps=cfg.control_fps,
                )
                below_fps, below_control = slow_tick_filter.take_counts()
                ticks = pending_frames(dataset) * max(1, (cfg.control_fps or cfg.fps) // cfg.fps)
                report_control_rate(
                    "Control rate", below_fps, below_control, ticks, cfg.fps, cfg.control_fps or cfg.fps
                )
                session_rate["below_fps"] += below_fps
                session_rate["below_control"] += below_control
                session_rate["ticks"] += ticks

                if not events["stop_recording"] and (
                    dataset.num_episodes + 1 < cfg.num_episodes or events["rerecord_episode"]
                ):
                    announce(
                        "Reset the scene, then press [n / right] when it is ready.",
                        YELLOW,
                        speak="Reset the scene",
                        play_sounds=play_sounds,
                    )
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=reset_time_s,
                        single_task=current_task,
                        display_data=False,
                        control_fps=cfg.control_fps,
                    )
                    # the reset loop swallows whichever key ended it, but esc/q also leaves
                    # exit_early set; clear it so it cannot end the next episode on its first tick
                    events["exit_early"] = False

                discard = events["rerecord_episode"]
                if discard:
                    announce("Discarding episode.", YELLOW, speak="Re-recording episode", play_sounds=play_sounds)
                    events["rerecord_episode"] = False
                    dataset.clear_episode_buffer()

                # started before the encode and joined after it, so the ramp and the encoding
                # share the same dead time; the ramp runs in the follower process, so the encode
                # cannot stall it however long it holds this process's GIL
                finish_homing = None
                if cfg.home_between_episodes and not events["stop_recording"]:
                    finish_homing = start_homing(robot, play_sounds)

                if not discard:
                    dataset.save_episode()
                    recorded_episodes += 1
                    announce(
                        f"Saved episode {dataset.num_episodes}/{cfg.num_episodes} "
                        f"(task: \"{current_task}\") -- {recorded_duration(dataset)} recorded",
                        GREEN,
                    )

                if finish_homing is not None:
                    finish_homing()

                if cfg.home_between_episodes and not events["stop_recording"]:
                    if not wait_for_leader_home(teleop, events, cfg.home_tolerance_deg, play_sounds):
                        break
                    events["exit_early"] = False
    finally:
        announce("Stopping.", YELLOW)
        log_say("Stop recording", play_sounds, blocking=True)
        # a failure here must not skip the disconnect below: that is what drops torque
        # and clears any latched motor fault. It does rule out uploading: finalize() is
        # what closes the parquet writers, so nothing on disk is trustworthy until it returns.
        finalize_error = None
        try:
            dataset.finalize()
        except Exception as e:
            finalize_error = e
            logger.exception("dataset.finalize() failed")
            announce(f"Failed to finalize the dataset: {e}", YELLOW)
        stop_arms(robot, teleop, camera_stream, listener)

        print()
        report_control_rate(
            "Session control rate",
            session_rate["below_fps"],
            session_rate["below_control"],
            session_rate["ticks"],
            cfg.fps,
            cfg.control_fps or cfg.fps,
        )

        if finalize_error is not None:
            print(
                f"\nSkipping upload: the local dataset at {dataset_root} did not finalize "
                "cleanly, so what is on disk may be incomplete. Check it before pushing."
            )
        elif dataset.num_episodes > 0:
            comparison = compare_with_hub(dataset, cfg.repo_id)
            print()
            for line in describe_hub_comparison(comparison, cfg.repo_id, recorded_episodes):
                print(line)
            # a safe push only ever adds to the Hub, so it takes a plain yes; the rest can
            # destroy episodes that exist only there
            if comparison.safe_to_push:
                approved = confirm("Upload to the Hub now?")
            else:
                approved = confirm_phrase(
                    "Uploading anyway can destroy data that exists only on the Hub.", "overwrite"
                )
            if approved:
                dataset.push_to_hub(private=cfg.private)
            elif comparison.safe_to_push:
                print("Skipping upload.")
            else:
                print(
                    f"Skipping upload. The local dataset is untouched at {dataset_root} -- "
                    "reconcile it with the Hub before pushing."
                )
        else:
            print("No episodes saved, skipping upload.")

    print(f"\nDone. Recorded {recorded_episodes} episodes across {len(task_history)} tasks.")
    print("Tasks recorded:")
    for t in task_history:
        print(f"  - {t}")


if __name__ == "__main__":
    main()
