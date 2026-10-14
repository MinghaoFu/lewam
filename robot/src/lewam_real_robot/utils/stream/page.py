"""The one page served at `/`, which builds itself from the server's panel manifest.

Kept as a string rather than a file so the server has no asset to locate at runtime
beyond external/.

The page fetches /panels.json, then for each panel picks a renderer out of RENDERERS by
the panel's `kind` and hands it an element plus that panel's `src`. A renderer owns its
own polling and drawing from there, so panel kinds don't interact.

RENDERERS is the only place a panel kind is known to javascript, and it is deliberately
small: the `chart` renderer draws whatever uPlot spec a ChartPanel sends, so every
chart-shaped panel -- a metric trace, a planner's cost curve -- is defined entirely in
python and needs no entry here. A new entry is only warranted by a genuinely new way of
drawing, not by a new thing to draw.

Drawing happens on the viewer's machine, not the rig: the robot only serializes the
numbers.
"""

PAGE = b"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rig</title>
<link rel="stylesheet" href="/external/uplot.css">
<script src="/external/uplot.js"></script>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; padding: 12px; background: #14161a; color: #e6e8eb;
    font: 13px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .panel { margin-bottom: 14px; }
  .panel img { display: block; width: 100%; border-radius: 6px; background: #000; }
  .note { padding: 20px; text-align: center; color: #8b929c; }
  /* two to a row, so a tall chart of one field sits beside the next rather than
     pushing it off the screen. One column once that would make them unreadable. */
  .charts { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
  @media (max-width: 700px) { .charts { grid-template-columns: 1fr; } }
  .chart { background: #1b1e24; border-radius: 6px; padding: 6px 8px; }
  .uplot, .u-wrap { width: 100% !important; }
  .u-title { font-size: 13px; font-weight: 600; letter-spacing: .02em; }
  .u-legend { font-size: 11px; }
  .u-legend .u-marker { width: 8px; height: 8px; border-radius: 2px; }
  .u-axis, .u-legend th, .u-legend td { color: #8b929c; }
</style>
</head>
<body>
<div id="root"></div>
<script>
const POLL_MS = 500;
// uPlot needs pixels, not css. The width follows the grid cell; only the height is
// ours to pick, and a tall narrow chart resolves a joint's motion far better than a
// wide flat one, which is what these are read for.
const CHART_HEIGHT = 300;

function el(tag, cls) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  return node;
}

// uPlot options that take a function can't survive json, so the python side names the
// one thing it needs (how many decimals an axis shows) and it becomes a formatter here.
function toOptions(spec, width) {
  const fmt = v => v == null ? "--" : v.toFixed(spec.precision ?? 2);
  return {
    title: spec.title,
    width: width,
    height: CHART_HEIGHT,
    cursor: {y: false},
    legend: {live: true},
    scales: {x: {time: false}},
    series: spec.series.map((s, i) =>
      i === 0 ? {value: (u, v) => v == null ? "--" : v.toFixed(1) + "s"}
              : Object.assign({}, s, {value: (u, v) => fmt(v)})),
    axes: [
      {stroke: "#8b929c", grid: {stroke: "#2b3038"}, ticks: {stroke: "#2b3038"}},
      {stroke: "#8b929c", grid: {stroke: "#2b3038"}, ticks: {stroke: "#2b3038"},
       values: (u, splits) => splits.map(fmt)},
    ],
  };
}

const RENDERERS = {
  video(root, panel) {
    const img = el("img");
    img.alt = panel.name;
    img.onerror = () => { img.style.display = "none"; };
    img.src = panel.src;
    root.appendChild(img);
  },

  // Draws any ChartPanel: the spec arrives with the data, so what gets plotted is
  // decided in python and nothing here needs to know what it is looking at.
  chart(root, panel) {
    const note = el("div", "note");
    note.textContent = "waiting for data...";
    root.appendChild(note);
    const grid = el("div", "charts");   // the boxes live here, not on the panel itself
    root.appendChild(grid);
    const charts = new Map();   // chart name -> {plot, box, signature}

    // One box per chart name, created once and registered before anything that could
    // throw. A half-built chart must not leave an orphan behind for the next poll to
    // create another of, which is how the page grows a blank stripe per tick forever.
    function boxFor(name) {
      let entry = charts.get(name);
      if (entry) return entry;
      const box = el("div", "chart");
      box.dataset.name = name;          // before the sort below, which reads it
      grid.appendChild(box);
      // charts appear as their data does, so re-sort into a stable order each time
      const boxes = [...grid.querySelectorAll(".chart")];
      boxes.sort((a, b) => (a.dataset.name || "").localeCompare(b.dataset.name || ""));
      for (const child of boxes) grid.appendChild(child);
      entry = {box, signature: null, plot: null};
      charts.set(name, entry);
      return entry;
    }

    function ensure(spec) {
      // a chart is rebuilt only when its series change; otherwise setData is enough
      const entry = boxFor(spec.title);
      const signature = spec.series.map(s => s.label || "").join("\\u0000");
      if (entry.signature === signature) return entry;
      if (entry.plot) entry.plot.destroy();
      entry.box.replaceChildren();
      entry.plot = new uPlot(toOptions(spec, entry.box.clientWidth - 16), [], entry.box);
      entry.signature = signature;
      return entry;
    }

    async function tick() {
      let payload;
      try {
        const res = await fetch(panel.src, {cache: "no-store"});
        if (!res.ok) return;
        payload = await res.json();
      } catch (e) { return; }           // a dropped poll is not worth reporting
      note.style.display = payload.charts.length ? "none" : "";
      for (const chart of payload.charts) {
        // per chart, so one that cannot be drawn costs only itself -- the whole
        // point of the panel is the charts you can still see
        try {
          ensure(chart.spec).plot.setData(chart.data);
        } catch (e) {
          console.error("chart " + chart.name + ":", e);
        }
      }
    }

    tick();
    setInterval(tick, POLL_MS);
    // uPlot needs an explicit size, so it has to be told when the column changes
    window.addEventListener("resize", () => {
      for (const {plot, box} of charts.values()) plot.setSize({width: box.clientWidth - 16, height: CHART_HEIGHT});
    });
  },
};

async function main() {
  const root = document.getElementById("root");
  let panels;
  try {
    panels = (await (await fetch("/panels.json", {cache: "no-store"})).json()).panels;
  } catch (e) {
    root.innerHTML = '<div class="note">could not reach the stream server</div>';
    return;
  }
  for (const panel of panels) {
    const section = el("section", "panel");
    root.appendChild(section);
    const render = RENDERERS[panel.kind];
    if (render) {
      render(section, panel);
    } else {
      const note = el("div", "note");
      note.textContent = 'no renderer for panel kind "' + panel.kind + '"';
      section.appendChild(note);
    }
  }
}

main();
</script>
</body>
</html>
"""
