"""Drops known-noisy lines from file descriptor 2.

Both sources this filters are C libraries writing with fprintf(stderr) from
their own threads, not through sys.stderr, so neither logging config nor
contextlib.redirect_stderr can silence them. This splices fd 2 through a
filtering pipe and forwards every other line through unchanged.
"""

import os
import re
import threading

# the C270's MJPG stream triggers one of these per decoded frame; they decode fine
CORRUPT_JPEG = rb"^Corrupt JPEG data: \d+ extraneous bytes before marker 0x[0-9a-f]+$"

# every libav message carries a "[component @ 0xADDRESS]" tag, failures included.
# A failed encode still raises out of encode_video_frames() rather than only printing.
LIBAV_TAGGED = rb"^\[[\w,.\-/ ]+ @ 0x[0-9a-f]+\]"

# SVT-AV1 prints its whole config banner with its own fprintf, underneath libav's log
# callback; SVT_LOG=1 (set where the rollout recorder is created) silences it at the
# source, and this catches builds that ignore the env var. [warn]/[error] pass through.
SVT_INFO = rb"^Svt\[info\]"


def suppress_stderr_lines(patterns) -> None:
    """Call once, as early as possible (before any camera connects or any video
    is encoded). `patterns` are bytes regexes matched against each line."""
    matchers = [re.compile(p) for p in patterns]

    def dropped(line: bytes) -> bool:
        return any(m.match(line) for m in matchers)

    real_stderr_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def _pump() -> None:
        with os.fdopen(read_fd, "rb", buffering=0) as pipe_in, os.fdopen(real_stderr_fd, "wb", buffering=0) as real_err:
            buf = b""
            while True:
                chunk = pipe_in.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not dropped(line):
                        real_err.write(line + b"\n")
                # progress bars redraw with a bare carriage return and never emit a newline, so
                # without this they sit in the buffer until the stream closes
                if b"\r" in buf:
                    head, buf = buf.rsplit(b"\r", 1)
                    real_err.write(head + b"\r")
            if buf and not dropped(buf):
                real_err.write(buf)

    threading.Thread(target=_pump, daemon=True, name="stderr-line-filter").start()
