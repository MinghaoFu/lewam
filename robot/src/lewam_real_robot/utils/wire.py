"""Length-prefixed pickle framing for the rollout link between a robot box and a remote
inference server (scripts/serve.py <-> the `lerobot` eval backend's RemoteDriver).

One blocking socket, strictly request/response: a 4-byte big-endian length followed by
that many pickle bytes. Payloads are plain dicts of numpy arrays and JPEG blobs, which
pickle carries without either side agreeing on a schema first -- the point of this layer
is that adding a field to a request needs no protocol change.

Pickle executes arbitrary code on load, so only ever point a client or a server at a host
you control on a trusted network.
"""

import pickle
import socket
import struct

# Refuse absurd frame headers rather than preallocating on a desynced or hostile stream.
MAX_MESSAGE_BYTES = 256 << 20
_CHUNK = 4 << 20


def connect(address, timeout=None):
    """`host:port` -> a connected TCP socket with Nagle off. A stalled control loop is
    waiting on this round trip, so small requests must never sit in a coalescing buffer.
    """
    host, _, port = address.rpartition(":")
    sock = socket.create_connection((host or "localhost", int(port)), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def listen(host, port):
    """A bound, listening server socket that survives an immediate restart."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    return sock


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_msg(sock):
    length = struct.unpack(">I", _recvall(sock, 4))[0]
    if length > MAX_MESSAGE_BYTES:
        raise ConnectionError(f"framed message of {length} bytes exceeds the cap")
    return pickle.loads(_recvall(sock, length))


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), _CHUNK))
        if not chunk:
            raise ConnectionError("connection closed by peer")
        buf.extend(chunk)
    return bytes(buf)


def encode_jpeg(rgb, quality=95):
    """RGB uint8 frame -> JPEG bytes. Camera frames dominate a request, and raw ones do
    not fit a 30 Hz replan over a LAN.
    """
    import cv2

    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise ValueError(f"failed to JPEG-encode a frame of shape {rgb.shape}")
    return buf.tobytes()


def decode_jpeg(data):
    import cv2
    import numpy as np

    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode a JPEG frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
