"""Live view of a running session, served as one page on the rig's network.

    from lewam_real_robot.utils.stream import CameraPanel, LeRobotObservationPanel, start_stream

    stream = start_stream([
        CameraPanel(robot.cameras, height=360, fps=15, quality=95),
        observations := LeRobotObservationPanel(window=300),
    ], port=8090)
    ...
    observations.track_features(obs_features, action_names)
    observations.push_step(frame, action)
    ...
    stream.close()   # before whatever the panels read from goes away

`start_stream` never raises and returns None when it can't serve (nothing to show, or
the port is taken), so a preview is never what brings down a recording or a rollout.

Panels are independent. To add one that plots numbers, subclass ChartPanel and return
`line_chart(...)` specs from `charts()` -- no javascript is involved. To add one that
draws some other way, subclass StreamPanel (see server.py) and give its `kind` a
renderer in page.py.
"""

from .camera import CameraPanel, merge
from .chart import PALETTE, ChartPanel, line_chart
from .metrics import LeRobotObservationPanel, TimeSeriesPanel, group_key
from .server import DEFAULT_PORT, PanelWriter, StreamPanel, StreamServer, local_ip, start_stream

__all__ = [
    "DEFAULT_PORT",
    "PALETTE",
    "CameraPanel",
    "ChartPanel",
    "LeRobotObservationPanel",
    "PanelWriter",
    "StreamPanel",
    "StreamServer",
    "TimeSeriesPanel",
    "group_key",
    "line_chart",
    "local_ip",
    "merge",
    "start_stream",
]
