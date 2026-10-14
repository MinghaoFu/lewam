# External assets

Served by the stream server at `/external/<file>` and loaded by the page. Vendored
rather than fetched from a CDN because the rig is usually on a network with no
route to the internet, and a preview that only works when unpkg is reachable is
worse than no preview.

| file | version | source | license |
| --- | --- | --- | --- |
| `uplot.js` | 1.6.32 | `https://unpkg.com/uplot@1.6.32/dist/uPlot.iife.min.js` | MIT (`uplot.LICENSE`) |
| `uplot.css` | 1.6.32 | `https://unpkg.com/uplot@1.6.32/dist/uPlot.min.css` | MIT (`uplot.LICENSE`) |

The IIFE build, so the page needs no module loader: it defines a global `uPlot`.

To update, refetch both files at the new version and change the version here.
Nothing generates these, and nothing but the stream page reads them.
