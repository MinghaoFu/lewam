"""Chart panels fed by whatever loop is being watched.

Camera panels poll something they don't own; these are the opposite -- the loop owns
the numbers and pushes them. Pushes are cheap enough to sit inside a 30 Hz control
loop: they append to a deque and never serialize. The charts are built on the HTTP
thread with the lock already released, so a slow viewer can never stall the loop it is
watching.

TimeSeriesPanel is the generic half: push named scalars, get one chart per name prefix.
LeRobotObservationPanel is the b601/lerobot use of it, and holds everything that knows
what a motor feature is called.
"""

import math
import threading
import time
from collections import deque

from .chart import ChartPanel, line_chart

DEFAULT_WINDOW = 300


def group_key(prefix: str, name: str) -> str:
    """`action` + `joint1.pos` -> `action.pos/joint1`: which chart a value belongs on,
    then which line within that chart.

    lerobot names every motor feature `<motor>.<field>`, so the field picks the chart
    and the motor picks the line. That split is what makes the charts automatic:
    positions, torques and velocities differ by orders of magnitude and can never
    share a y axis, while the joints within one field always can. A name with no
    field is its own single-line chart.
    """
    motor, _, field = name.rpartition(".")
    return f"{prefix}.{field}/{motor}" if motor else f"{prefix}/{name}"


class TimeSeriesPanel(ChartPanel):
    """The last `window` pushed rows, as one chart per `<chart>/<line>` name prefix.

    Knows nothing about robots: anything that can name a number can push to it.
    """

    def __init__(self, name: str = "metrics", window: int = DEFAULT_WINDOW):
        super().__init__(name)
        self.rows: deque = deque(maxlen=max(1, int(window)))
        self.lock = threading.Lock()

    def push(self, values: dict, t: float | None = None) -> None:
        """Append one row, stamped with a monotonic clock.

        Anything that isn't a finite number is dropped rather than raised over: a
        metric that goes NaN must not be what ends a rollout, and json cannot
        represent it anyway (json.dumps would emit a bare NaN, which no browser's
        JSON.parse accepts).
        """
        row = {}
        for key, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                row[str(key)] = number
        with self.lock:
            self.rows.append((time.perf_counter() if t is None else float(t), row))

    def snapshot(self) -> dict:
        """Rows transposed into one array per series, on a shared time axis starting
        at zero. A series that started late, or that skipped a row, gets a null
        there, which uPlot draws as a gap rather than joining across it."""
        with self.lock:
            rows = list(self.rows)
        if not rows:
            return {"t": [], "series": {}}
        t0 = rows[0][0]
        keys = sorted({key for _, row in rows for key in row})
        return {
            "t": [round(t - t0, 3) for t, _ in rows],
            "series": {key: [row.get(key) for _, row in rows] for key in keys},
        }

    def precision(self, chart: str) -> int:
        """Decimals on this chart's y axis. Overridden where different charts hold
        quantities of very different magnitude."""
        return 2

    def charts(self) -> list[dict]:
        snap = self.snapshot()
        # (label, series key) per chart, so a pushed name carrying no "/" still finds
        # its own data
        groups: dict[str, list[tuple[str, str]]] = {}
        for key in snap["series"]:
            chart, sep, line = key.partition("/")
            groups.setdefault(chart, []).append((line if sep else chart, key))

        charts = []
        for chart, lines in sorted(groups.items()):
            lines.sort(key=lambda pair: _natural(pair[0]))
            charts.append(
                line_chart(
                    chart,
                    [label for label, _ in lines],
                    [snap["t"]] + [snap["series"][key] for _, key in lines],
                    precision=self.precision(chart),
                )
            )
        return charts


class LeRobotObservationPanel(TimeSeriesPanel):
    """A rollout's executed action against the state it was sent from.

    Which charts appear is derived from the robot's own feature names, so this follows
    an embodiment wherever its features change and there is no series list to maintain:
    a b601 yields action.pos, state.pos, state.torq and state.vel, one line per joint.
    """

    def __init__(self, name: str = "observation", window: int = DEFAULT_WINDOW):
        super().__init__(name, window)
        self.state_keys: list[str] = []
        self.action_keys: list[str] = []

    def track_features(self, obs_features: dict, action_names: list[str]) -> None:
        """Name the charts from lerobot's dataset features. Called once, before the
        episode loop; until it is, push_step has nothing to file values under."""
        state_names = obs_features.get("observation.state", {}).get("names", [])
        self.state_keys = [group_key("state", name) for name in state_names]
        self.action_keys = [group_key("action", name) for name in action_names]

    def push_step(self, frame: dict, action, t: float | None = None) -> None:
        """One control step of a rollout: the dataset frame it was decided from, and
        the action vector that was sent. Needs track_features to name the columns."""
        if not self.state_keys and not self.action_keys:
            return
        state = frame.get("observation.state", [])
        self.push(
            dict(zip(self.state_keys, state, strict=True))
            | dict(zip(self.action_keys, action, strict=True)),
            t,
        )

    def push_raw(self, observation: dict, action: dict, t: float | None = None) -> None:
        """One step of teleoperation, from the robot's and teleoperator's own dicts
        (`{"joint1.pos": 1.0, ...}`) rather than the dataset frames a rollout builds.

        These are already keyed by feature name, so nothing has to be tracked first.
        Camera entries come through as image arrays and are dropped on the way in,
        since push only keeps what converts to a finite number.
        """
        self.push(
            {group_key("state", key): value for key, value in observation.items()}
            | {group_key("action", key): value for key, value in action.items()},
            t,
        )

    def precision(self, chart: str) -> int:
        # torques and velocities sit near zero while positions run to the tens, so one
        # fixed number of decimals would crowd one or flatten the other
        return 2 if chart.endswith((".torq", ".vel")) else 1


def _natural(label: str):
    """joint2 before joint10, so a legend reads in joint order rather than ASCII."""
    digits = "".join(c for c in label if c.isdigit())
    return (label.rstrip("0123456789"), int(digits) if digits else 0)
