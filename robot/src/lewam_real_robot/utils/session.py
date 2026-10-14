"""Operator-session scaffolding shared by on-robot entry points (b601 record/teleop
scripts, the lerobot eval backend): how milestones are printed and spoken, how prompts
survive a raw-mode keyboard listener and camera-thread stderr spam, and how Ctrl-C is
turned into a graceful stop instead of a KeyboardInterrupt mid-CAN-write. Embodiment-
specific pieces (the b601 homing ramp) stay in scripts/b601/common.py.
"""

from __future__ import annotations

import logging
import signal
import sys

try:
    import termios
except ImportError:
    termios = None

from lerobot.utils.utils import log_say

RESET_COLOR = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"


def announce(
    text: str,
    color: str = "",
    speak: str | None = None,
    play_sounds: bool = False,
    bell: bool = False,
) -> None:
    """Print one milestone line (episode boundaries, prompts, warnings) in
    `color`, so it stands out from lerobot's own output. Colors are skipped when
    stdout isn't a terminal, e.g. when the session is piped to a file.

    `speak` additionally says the line aloud, in place of lerobot's log_say(),
    which announces at INFO and so never reaches the console. `bell`
    rings the terminal, to catch attention from across the room when the
    terminal isn't in focus; unlike `speak` it ignores `quiet`, which is about
    the spoken announcements."""
    if bell and sys.stdout.isatty():
        sys.stdout.write("\a")
        sys.stdout.flush()
    print(f"{color}{text}{RESET_COLOR}" if color and sys.stdout.isatty() else text)
    if speak is not None:
        try:
            log_say(speak, play_sounds)
        except OSError:
            pass


def quiet_console_logging(extra_filter: logging.Filter | None = None) -> None:
    """Leave the console showing warnings and above, so the milestone lines
    announce() prints stay readable. Hydra's own file handler is untouched and
    still records everything, including anything `extra_filter` drops.

    Importing lerobot triggers an implicit logging.basicConfig() (a bare
    logging.debug() call in lerobot.utils.import_utils), so there is a console
    handler to find here whether or not Hydra installed one."""
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(logging.WARNING)
            if extra_filter is not None:
                handler.addFilter(extra_filter)


def flush_pending_stdin() -> None:
    """Discard any keystrokes typed but not yet read (e.g. episode controls hit
    during the control loop) so they don't leak as text into the next
    prompt."""
    if termios is None or not sys.stdin.isatty():
        return
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


def prompt(text: str) -> str:
    """Like input(text), but input()'s tty readline path races with the camera
    thread's libjpeg stderr writes and leaks the occasional "Corrupt JPEG data"
    warning past the fd-2 filter. EOF returns "" rather than raising EOFError.
    """
    sys.stdout.write(text)
    sys.stdout.flush()
    return sys.stdin.readline().rstrip("\n")


def confirm(question: str) -> bool:
    """[y/N] prompt. Anything but an explicit yes -- including a bare Enter or EOF -- is no,
    so every caller's default has to be the recoverable branch."""
    flush_pending_stdin()
    return prompt(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def confirm_phrase(question: str, phrase: str) -> bool:
    """Confirmation for an action that can destroy data held only on the Hub. Requires the
    word typed out, so a reflexive "y" declines it -- these cases are rare enough that
    making them cost a moment's thought is the point."""
    flush_pending_stdin()
    typed = prompt(
        f"{question}\nType '{phrase}' to proceed, anything else to skip: "
    ).strip()
    return typed == phrase


def install_graceful_sigint_handler(events: dict) -> None:
    """Route the first Ctrl-C to the same graceful stop the 'esc' key already
    triggers, instead of letting KeyboardInterrupt unwind mid-CAN-write inside
    robot.send_action() and leave the follower comm-faulted. Loops that check
    events["exit_early"] once per iteration stop within one tick.

    Restores normal SIGINT behavior after the first press, so a second
    Ctrl-C force-exits."""

    def handle_sigint(signum, frame):
        print("\nStopping (Ctrl-C again to force an immediate exit)...")
        events["exit_early"] = True
        events["stop_recording"] = True
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, handle_sigint)
