"""A stream panel showing already-open lerobot cameras as one merged MJPEG feed.

Peeks at cameras someone else owns and is already reading, rather than opening its
own: frames are taken with `read_latest()`, which copies a reference under the
camera's frame lock and leaves `new_frame_event` alone. `async_read()` would clear
that event and so compete for frames with the real consumer; `read_latest` cannot.
That makes this safe to point at a robot's cameras mid-recording.

One background thread polls every camera in turn, merges the frames side by side and
JPEG-encodes them, exactly as the standalone preview scripts in
src/lewam_real_robot/debug/hardware/ do -- see stream_lerobot_cameras.py's module docstring for
why the polling is sequential in one thread rather than one per camera. Encoding runs
on another core rather than stalling the owner: OpenCV releases the GIL for the resize
and the encode.
"""

import logging
import threading
import time

import cv2

from lerobot.cameras.configs import ColorMode

from .server import StreamPanel

logger = logging.getLogger(__name__)


def merge(frames, height):
    """Scale every frame to a common height and lay them out left to right."""
    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        scale = height / h
        resized.append(cv2.resize(frame, (max(1, int(w * scale)), height)))
    return cv2.hconcat(resized)


def _to_bgr(frame, camera):
    """cv2 encodes BGR, while lerobot cameras hand back whichever layout their
    config asked for -- and the ZED backend has no `color_mode` at all because it
    always returns RGB, to match what the dataset expects."""
    if getattr(camera, "color_mode", ColorMode.RGB) == ColorMode.BGR:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


class CameraPanel(StreamPanel):
    """Keeps the latest merged+encoded JPEG of `cameras` ready to serve, and serves it
    as an endless multipart stream."""

    kind = "video"

    def __init__(self, cameras: dict, name: str = "cameras", height: int = 240,
                 fps: int = 10, quality: int = 60):
        super().__init__(name)
        self.cameras = dict(cameras)
        self.height = height
        self.period = 1 / fps
        self.quality = quality
        self.frames: dict = {}
        self.jpg: bytes | None = None
        self.lock = threading.Lock()
        self.running = True
        self._warned_empty = False
        self.thread = threading.Thread(target=self._loop, daemon=True, name="camera-stream")
        self.thread.start()

    def _loop(self):
        started = time.perf_counter()
        while self.running:
            t0 = time.perf_counter()
            for name, cam in self.cameras.items():
                try:
                    self.frames[name] = _to_bgr(cam.read_latest(), cam)
                except Exception:
                    # a stale, unstarted or disconnected camera keeps its last frame; a preview is
                    # never worth raising over
                    continue
            # held back until every camera has produced a frame, so the layout does not
            # shift as cameras warm up at different rates
            if len(self.frames) == len(self.cameras) and self.frames:
                merged = merge(list(self.frames.values()), self.height)
                ok, jpg = cv2.imencode(".jpg", merged, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ok:
                    with self.lock:
                        self.jpg = jpg.tobytes()
            elif not self._warned_empty and time.perf_counter() - started > 5:
                missing = sorted(set(self.cameras) - set(self.frames))
                logger.warning(f"Camera stream has no frames yet from: {', '.join(missing)}")
                self._warned_empty = True
            time.sleep(max(0.0, self.period - (time.perf_counter() - t0)))

    def latest_jpg(self) -> bytes | None:
        with self.lock:
            return self.jpg

    def serve(self, handler):
        handler.send_response(200)
        handler.send_header("Age", "0")
        handler.send_header("Cache-Control", "no-cache, private")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        handler.end_headers()
        while self.running:
            jpg = self.latest_jpg()
            if jpg is not None:
                handler.wfile.write(b"--FRAME\r\n")
                handler.send_header("Content-Type", "image/jpeg")
                handler.send_header("Content-Length", str(len(jpg)))
                handler.end_headers()
                handler.wfile.write(jpg)
                handler.wfile.write(b"\r\n")
            time.sleep(self.period)

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
