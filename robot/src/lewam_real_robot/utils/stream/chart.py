"""Panels that draw themselves as line charts, defined entirely in Python.

A ChartPanel serves a list of charts, each carrying its own uPlot spec (which series
exist, their labels and colours, the axes) next to the data for them. The page has one
generic renderer for the whole `chart` kind: it hands the spec straight to uPlot and
calls setData, rebuilding a chart only when its series change.

That is what keeps new plots out of the page's javascript. A subclass decides what to
plot by returning specs and data from `charts()` -- a metric trace, a planner's cost
curve, a reward history -- and the page renders it without knowing what any of it is.
The seam is only crossed again for a genuinely new way of *drawing*, not for a new
thing to draw.

The spec is json, so it holds no functions: uPlot options that take callbacks
(value formatters, custom strokes) cannot be expressed here. `precision` covers the
common case by naming how many decimals an axis shows, which the renderer turns back
into a formatter on the other side.
"""

import json

from .server import StreamPanel

# reused by every ChartPanel, so two charts on one page never disagree about
# what colour a first series is
PALETTE = [
    "#4ea1ff", "#ff6b6b", "#4ecb71", "#ffd166", "#c792ea",
    "#00c2c7", "#ff9f43", "#a0aec0", "#f78fb3", "#7bed9f",
]


def line_chart(name: str, labels: list[str], columns: list[list], precision: int = 2) -> dict:
    """One chart's spec and data, in the shape the page's chart renderer expects.

    `columns` is uPlot's own layout: the x values first, then one array per label.
    Nulls inside a column are drawn as gaps rather than joined across.
    """
    return {
        "name": name,
        "spec": {
            "title": name,
            "precision": precision,
            "series": [{}] + [
                {"label": label, "stroke": PALETTE[i % len(PALETTE)], "width": 1.4}
                for i, label in enumerate(labels)
            ],
        },
        "data": columns,
    }


class ChartPanel(StreamPanel):
    """Serves `charts()` as json. Subclasses only decide what to plot."""

    kind = "chart"

    def charts(self) -> list[dict]:
        """The charts to draw right now, each from `line_chart`."""
        raise NotImplementedError

    def json(self) -> bytes:
        return json.dumps({"charts": self.charts()}).encode()

    def serve(self, handler):
        handler.send_bytes(self.json(), "application/json")
