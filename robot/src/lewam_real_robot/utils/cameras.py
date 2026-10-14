"""Camera device detection and config building shared by hardware scripts
(src/lewam_real_robot/scripts/b601/record.py, src/lewam_real_robot/debug/hardware/*.py). Resolves
which /dev/video* nodes belong to which physical camera and turns a plain
`{name: {type: ..., ...}}` config into lerobot `CameraConfig` objects; does not
open or connect anything.
"""

import contextlib
import dataclasses
import importlib
import re
import subprocess
import sys
from pathlib import Path

# an omitted serial_number / index_or_path means the same thing
AUTO = "auto"

# lerobot does not import these from lerobot.cameras (each pulls in its own
# backend dependency), so a backend is imported only once a config asks for it
_BACKEND_MODULES = {
    "opencv": "lerobot.cameras.opencv",
    "zed": "lerobot.cameras.zed",
    "intelrealsense": "lerobot.cameras.realsense",
    "reachy2_camera": "lerobot.cameras.reachy2_camera",
    "zmq": "lerobot.cameras.zmq",
}


def _is_capture_node(dev):
    """Whether this /dev/video* node's Device Caps (not the driver-wide
    Capabilities of the whole physical device) include Video Capture -- most
    cameras also expose a second, metadata-only node alongside the real one."""
    out = subprocess.run(["v4l2-ctl", "-d", dev, "--all"], capture_output=True, text=True).stdout
    m = re.search(r"Device Caps\s*:.*\n((?:\t\t.*\n?)*)", out)
    return bool(m) and "Video Capture" in m.group(1)


def _v4l2_device_groups():
    """Groups /dev/video* indices by physical device, per `v4l2-ctl --list-devices`
    (which prints one heading per physical device followed by its node paths).
    Returns a list of (heading, {indices}) pairs."""
    out = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True).stdout
    groups = []
    for line in out.splitlines():
        if line and not line[0].isspace():
            groups.append((line.strip(), set()))
        elif line.strip().startswith("/dev/video") and groups:
            groups[-1][1].add(int(re.search(r"\d+", line.strip()).group()))
    return groups


def find_zed_cameras():
    """Every connected ZED, via the ZED SDK's own enumeration. Returns an empty
    list when no ZED is plugged in or the zed backend isn't installed, so a
    setup with no ZED at all still works."""
    try:
        from lerobot.cameras.zed import ZedCamera
    except ImportError:
        return []
    return ZedCamera.find_cameras()


def find_webcam_path(exclude_indices, exclude_paths=()):
    """Lowest-numbered USB video-capture node that isn't one of the ZED's own
    nodes (`exclude_indices`) or already claimed by another camera in the same
    config (`exclude_paths`). Restricted to USB devices (v4l2-ctl --list-devices
    headings containing "usb"), not just anything that reports a Video Capture
    capability: on the Pi, the onboard ISP/video-decode blocks (`pispbe`,
    `rpivid`) expose their own capture-capable nodes too, and would otherwise be
    picked as "the webcam".

    Uses v4l2-ctl rather than probing with cv2 or the ZED SDK, so it never has
    to open (and risk disturbing) a node it's about to skip -- notably the
    ZED's, which corrupts frames if opened as plain UVC."""
    if subprocess.run(["which", "v4l2-ctl"], capture_output=True).returncode != 0:
        sys.exit("v4l2-ctl not found (apt install v4l-utils) -- needed to auto-detect the webcam device")

    # a claimed device may be a path or a bare index; both must keep the search
    # off the same node
    exclude_indices = set(exclude_indices)
    claimed = set()
    for device in exclude_paths:
        if isinstance(device, int):
            exclude_indices.add(device)
        else:
            claimed.add(str(device))

    usb_indices = {idx for heading, indices in _v4l2_device_groups() if "usb" in heading.lower() for idx in indices}
    candidates = sorted(Path("/dev").glob("video*"), key=lambda p: int(re.search(r"\d+", p.name).group()))
    for path in candidates:
        idx = int(re.search(r"\d+", path.name).group())
        if str(path) in claimed:
            continue
        if idx in usb_indices and idx not in exclude_indices and _is_capture_node(str(path)):
            return str(path)
    return None


def zed_node_indices():
    """Every /dev/video* index belonging to a connected ZED's physical device.

    Excludes the ZED's whole device group, not just the node the SDK reports
    +/- 1: its capture and metadata nodes aren't always at consecutive indices
    (other USB video devices enumerated in between can separate them)."""
    zed_indices = {int(info["index"]) for info in find_zed_cameras()}
    exclude = set(zed_indices)
    for _heading, indices in _v4l2_device_groups():
        if indices & zed_indices:
            exclude |= indices
    return exclude


def resolve_zed_serial(zed_serial=None):
    """The serial of the ZED to open, auto-detecting the first one if not given.

    Working around a real crash: ZedCamera's own auto-detect (serial_number=None ->
    dev_id=-1) scans video nodes itself, and if another camera already holds one of
    them (a ZED enumerates as a UVC device too, and can land at a low index like
    /dev/video0) that scan hits a busy device and the native library aborts the
    whole process (SIGABRT, not a catchable exception) instead of failing cleanly.
    Looking the serial up through the SDK's find_cameras() first -- which never
    touches cv2 -- skips that scan. Callers that open several cameras should also
    connect the ZED first, as extra insurance against the same race (build_cameras()
    orders them that way)."""
    if zed_serial is not None:
        return zed_serial
    zed_cameras = find_zed_cameras()
    if not zed_cameras:
        sys.exit("No ZED camera detected (ZedCamera.find_cameras() found none).")
    return zed_cameras[0]["id"]


def resolve_webcam_path(opencv_index_or_path=None, exclude_paths=()):
    """The device path/index of a plain UVC webcam, auto-detecting one that is
    neither a ZED node nor already claimed (`exclude_paths`) if not given."""
    if opencv_index_or_path is None:
        path = find_webcam_path(zed_node_indices(), exclude_paths)
        if path is None:
            sys.exit("Could not auto-detect a non-ZED webcam device; pass an explicit index/path.")
        return path
    try:
        return int(opencv_index_or_path)
    except (TypeError, ValueError):
        return opencv_index_or_path


def _camera_config_class(type_name):
    """The registered CameraConfig subclass for a `type:` value, importing the
    backend that registers it first."""
    from lerobot.cameras import CameraConfig

    module = _BACKEND_MODULES.get(type_name)
    if module is not None:
        try:
            importlib.import_module(module)
        except ImportError as e:
            raise ValueError(
                f"camera type '{type_name}' needs '{module}', which failed to import: {e}"
            ) from e
    else:
        # a type this map doesn't know: import every backend that will import, in case
        # a newer lerobot registers one under a name predating this map
        for candidate in _BACKEND_MODULES.values():
            with contextlib.suppress(ImportError):
                importlib.import_module(candidate)

    try:
        return CameraConfig.get_choice_class(type_name)
    except Exception as e:
        known = sorted(CameraConfig.get_known_choices())
        raise ValueError(f"unknown camera type '{type_name}'. registered types: {known}") from e


def build_cameras(cameras_cfg):
    """`{name: {type: ..., <backend fields>}}` -> `{name: CameraConfig}`, ready to
    hand to a lerobot `Robot(cameras=...)`. The yaml equivalent of lerobot's own
    `--robot.cameras='{front: {type: opencv, index_or_path: 0}}'`.

    `type` picks the lerobot camera backend by its registered name (`zed`,
    `opencv`, `intelrealsense`, `zmq`, ...); every other key is passed straight
    to that backend's config dataclass, so anything lerobot supports is
    configurable without touching this function. The dict key becomes the
    dataset's camera/observation key, so what a camera *is* (`wrist_cam`,
    `front`, ...) is named by the config rather than by this code. An empty
    config is not an error -- it records robot state with no image streams, and
    an entry whose value is null is dropped (see the loop below).

    Beyond plain lerobot, two device fields are auto-detected when omitted or
    set to "auto": `serial_number` on a zed (all auto zed entries then share the
    one detected camera, which is what lets left/right eyes share a single
    capture) and `index_or_path` on an opencv camera (the lowest-numbered USB
    capture node that is not a ZED's and not already claimed by another entry,
    so several auto webcams resolve to different devices).

    Any zed entries are returned first regardless of their order in the config,
    so Robot.connect() opens them before any UVC camera -- see
    resolve_zed_serial() for why that ordering matters.
    """
    from omegaconf import OmegaConf

    if OmegaConf.is_config(cameras_cfg):
        cameras_cfg = OmegaConf.to_container(cameras_cfg, resolve=True)
    if not cameras_cfg:
        return {}

    zed_serial = None
    claimed_paths = []
    cameras = {}
    for name, spec in cameras_cfg.items():
        # a null entry drops an inherited camera: yaml has no equivalent of the CLI's
        # `~cameras.<name>`, so a shared arrangement can only be overridden, not trimmed
        if spec is None:
            continue
        spec = dict(spec)
        type_name = spec.pop("type", None)
        if type_name is None:
            raise ValueError(f"camera '{name}' has no `type` (e.g. zed, opencv, intelrealsense)")
        config_cls = _camera_config_class(type_name)

        if type_name == "zed":
            if spec.get("serial_number", AUTO) in (None, AUTO):
                zed_serial = resolve_zed_serial(zed_serial)
                spec["serial_number"] = zed_serial
        elif type_name == "opencv":
            given = spec.get("index_or_path", AUTO)
            path = resolve_webcam_path(
                None if given in (None, AUTO) else given, exclude_paths=claimed_paths
            )
            spec["index_or_path"] = path
            claimed_paths.append(path)

        try:
            cameras[name] = config_cls(**spec)
        except TypeError as e:
            fields = sorted(f.name for f in dataclasses.fields(config_cls))
            raise ValueError(
                f"camera '{name}' (type={type_name}): {e}. valid fields: {fields}"
            ) from e

    return dict(
        sorted(cameras.items(), key=lambda item: item[1].type != "zed")
    )
