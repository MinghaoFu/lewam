"""USB serial port detection for the arms -- the /dev counterpart of
utils/cameras.py, turning `port: auto` in an embodiment config into a device path.

`/dev/ttyACM*` numbering is kernel enumeration order, so a hardcoded path silently
points at the wrong device once another CDC-ACM peripheral is plugged in or the arm
is re-plugged. The USB vendor/product pair of the arm's serial bridge identifies it
regardless of enumeration, and needs no interactive unplug step to disambiguate the
way lerobot-find-port does.
"""

from pathlib import Path

# also what an omitted port means; matches utils/cameras.py's convention
AUTO = "auto"

_TTY_CLASS = Path("/sys/class/tty")

# robot type -> USB (idVendor, idProduct) of its serial bridge; the ids are the
# bridge chip's, so every B601-DM presents the same pair. A leader's CH340 is
# deliberately absent: it would just as happily match an unrelated adapter.
ARM_USB_IDS = {
    "rebot_b601_follower": ("2e88", "4603"),
    "bi_rebot_b601_follower": ("2e88", "4603"),
}


def _usb_ids(tty):
    """(idVendor, idProduct) of the USB device behind a tty, or None for a tty
    that isn't USB at all. The ids live on the USB device node, several levels
    above the interface the tty itself links to, so walk up until they appear."""
    node = tty / "device"
    if not node.exists():
        return None
    node = node.resolve()
    while node != node.parent and not (node / "idVendor").exists():
        node = node.parent
    if not (node / "idVendor").exists():
        return None
    return (
        (node / "idVendor").read_text().strip(),
        (node / "idProduct").read_text().strip(),
    )


def find_arm_ports(robot_type):
    """Every tty whose USB bridge matches `robot_type`, as /dev paths."""
    ids = ARM_USB_IDS.get(robot_type)
    if ids is None:
        raise ValueError(
            f"no USB id known for robot type '{robot_type}', so its port cannot be "
            f"auto-detected; set an explicit `port:`. Known: {sorted(ARM_USB_IDS)}"
        )
    return sorted(
        f"/dev/{tty.name}" for tty in _TTY_CLASS.glob("tty*") if _usb_ids(tty) == ids
    )


def resolve_arm_port(robot_type, port=None):
    """The arm's device path, auto-detecting when `port` is None or "auto".

    An explicit port is returned untouched, including for robot types this module
    knows nothing about -- only asking for auto-detection requires a known type.
    """
    if port not in (None, AUTO):
        return port
    matches = find_arm_ports(robot_type)
    if not matches:
        raise ValueError(
            f"no {robot_type} found on any USB serial port (looking for USB id "
            f"{':'.join(ARM_USB_IDS[robot_type])}). Is the arm plugged in and powered on?"
        )
    if len(matches) > 1:
        raise ValueError(
            f"several {robot_type} bridges are connected ({', '.join(matches)}); "
            "name the one to use with an explicit `port:`."
        )
    return matches[0]
