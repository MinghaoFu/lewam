"""A small HTTP server that puts whatever a session wants to watch on one page:
    http://<pi-ip>:8090/

The server knows nothing about cameras, metrics or robots. It holds a list of
StreamPanels, serves each at /panel/<name>, and publishes a manifest at /panels.json
saying what each one is. The page builds itself from that manifest, picking a renderer
by the panel's `kind` (see page.py).

So adding something new to watch is a StreamPanel subclass; this module and every
existing panel are untouched, and a session composes the panels it wants at the call
site rather than the server growing a keyword argument per thing it might show.

Whether that subclass also needs javascript depends on how it draws. Anything
chart-shaped -- a metric trace, a planner's cost curve -- subclasses ChartPanel and is
defined entirely in python, because the page's `chart` renderer draws whatever spec it
is sent. A genuinely new way of drawing (an image, a video) still needs a renderer for
its `kind` in page.py, which is the one file a new panel kind can require.

A panel writes its own response, so pulled-once payloads (json, a jpeg) and endless
ones (an MJPEG multipart stream) are the same kind of object here.
"""

import json
import logging
import mimetypes
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from io import BufferedIOBase
from typing import Protocol

from .page import PAGE

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8090


class PanelWriter(Protocol):
    """The part of the request handler a panel is allowed to use, so what a panel may
    call is a stated contract rather than whatever BaseHTTPRequestHandler happens to
    expose. Signatures mirror that class exactly, since it is what gets passed."""

    wfile: BufferedIOBase

    def send_bytes(self, body: bytes, content_type: str) -> None:
        """Send one complete response. The usual case."""

    def send_response(self, code: int) -> None: ...
    def send_header(self, keyword: str, value: str) -> None: ...
    def end_headers(self) -> None: ...


class StreamPanel:
    """One thing the page shows.

    `kind` picks the renderer on the page; `name` is unique within a server and is
    both the route segment and the heading.
    """

    kind = "panel"

    def __init__(self, name: str):
        self.name = name

    @property
    def src(self) -> str:
        return f"/panel/{self.name}"

    def spec(self) -> dict | None:
        """Static description of this panel for the manifest, if its renderer needs
        one up front. Panels whose shape only emerges from data return None and put it
        in the payload instead."""
        return None

    def serve(self, handler: PanelWriter) -> None:
        """Write this panel's HTTP response, on the server thread. Use
        `handler.send_bytes(body, content_type)` for anything that fits in one
        response; write to `handler.wfile` directly to stream indefinitely."""
        raise NotImplementedError

    def close(self) -> None:
        """Release whatever the panel started. Called once, on shutdown."""


def local_ip() -> str:
    """The address of the interface that reaches the network, so the printed URL
    is one another machine can actually open."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        s.close()


def _load_external() -> dict[str, tuple[bytes, str]]:
    """Read external/ once at import, so serving never touches the filesystem and the
    files travel with the wheel."""
    assets = {}
    for path in files(__package__).joinpath("external").iterdir():
        if path.name.endswith((".js", ".css")):
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            assets[path.name] = (path.read_bytes(), content_type)
    return assets


EXTERNAL = _load_external()


def _make_handler(panels: list[StreamPanel]):
    """Bound to one set of panels via a closure rather than a module global, so the
    handler can't outlive them or be pointed at the wrong server."""
    by_name = {panel.name: panel for panel in panels}
    manifest = json.dumps(
        {
            "panels": [
                {"name": p.name, "kind": p.kind, "src": p.src, "spec": p.spec()}
                for p in panels
            ]
        }
    ).encode()

    class StreamHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self.send_bytes(PAGE, "text/html; charset=utf-8")
            elif path == "/panels.json":
                self.send_bytes(manifest, "application/json")
            elif path.startswith("/external/"):
                self._send_external(path[len("/external/"):])
            elif path.startswith("/panel/"):
                panel = by_name.get(path[len("/panel/"):])
                if panel is None:
                    self.send_error(404)
                    return
                try:
                    panel.serve(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception:
                    logger.exception(f"stream panel {panel.name!r} failed")
            else:
                self.send_error(404)

        def _send_external(self, filename: str):
            """Third-party assets the page loads (see external/README.md). Vendored
            rather than fetched from a CDN, since the rig usually has no route to the
            internet. The name is matched against a fixed set, so nothing outside
            external/ is reachable however the path is spelled."""
            asset = EXTERNAL.get(filename)
            if asset is None:
                self.send_error(404)
                return
            body, content_type = asset
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def send_bytes(self, body: bytes, content_type: str):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 - matches the base signature
            pass

    return StreamHandler


class StreamServer:
    """The HTTP server and the panels it serves, closed together."""

    def __init__(self, panels: list[StreamPanel], server: ThreadingHTTPServer, url: str):
        self.panels = panels
        self.server = server
        self.url = url
        self._thread = threading.Thread(target=server.serve_forever, daemon=True, name="stream-http")
        self._thread.start()

    def close(self):
        # stops serving before the panels, so no client thread is mid-write when a
        # panel's source goes away; camera panels must be down before the robot
        # disconnects, since read_latest() raises once the cameras are closed
        self.server.shutdown()
        self.server.server_close()
        for panel in self.panels:
            panel.close()
        self._thread.join(timeout=2)


def start_stream(panels, port: int = DEFAULT_PORT) -> StreamServer | None:
    """Serve `panels` as one page, or return None if that isn't possible -- nothing to
    show, or a port already in use. `panels` may contain None, so a caller can build it
    with plain conditionals.

    A preview that won't start is never a reason to bring down whatever is doing the
    actual work, so this reports the problem and lets the caller carry on without one.
    """
    panels = [panel for panel in panels if panel is not None]
    if not panels:
        return None
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(panels))
    except OSError as e:
        for panel in panels:
            panel.close()
        logger.warning(f"Stream disabled: could not listen on port {port} ({e}).")
        return None
    return StreamServer(panels, server, f"http://{local_ip()}:{port}/")
