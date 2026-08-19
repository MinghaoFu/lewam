# Choosing a maze size for the pointmaze cell

OGBench's `locomaze` ships five layouts for each of three agents (point / ant / humanoid), plus
`visual-` pixel variants and antsoccer. This records what they actually are, measured rather than
estimated, and why the suite adds **large** rather than giant.

## Layouts

| layout | grid | shape | free cells | mean shortest path | official task paths |
|---|---|---|---|---|---|
| arena | 8×8 | square | 36 | — | open, no interior walls |
| **medium** (the original cell) | 8×8 | square | 26 | 5.1 | 6–10 |
| **large** (added) | 9×12 | rect | 46 | 7.9 (1.56×) | 12–19 |
| giant | 12×16 | rect | 86 | 13.3 (2.61×) | 17–30 |
| teleport | 9×12 | rect | 45 | — | contains portals |

Difficulty tracks **path length, not area**. By free-cell count large looks 1.8× medium, but by
mean BFS shortest path it is 1.56×. Giant is the real jump — its longest path is 31 cells against
medium's 11.

## Episode structure

Medium and large are both 1000 episodes × 1001 steps. **Giant is 500 × 2001** — the horizon
doubles, so evaluation wall-clock doubles with it. Large therefore changes exactly one variable
(path length) while giant changes two.

## Render framing

OGBench's camera is top-down with `distance = 5*(width-2)`. Against mujoco's 45° vertical FOV a
rectangular maze still nearly fills a square frame. Measured at 224×224:

| layout | vertical content | horizontal | px per maze cell |
|---|---|---|---|
| medium | 100% | 100% | 28 |
| large | **91%** | 100% | 18.7 |
| giant | 86% | 100% | 14 |

So large needs **no cropping, stretching or aspect change** — it drops into the suite's 224×224
pipeline unchanged, losing only a 10-pixel band top and bottom. Giant's target marker is down to a
few pixels, which would confound any result there with a visibility change; if giant is ever used
it should be rendered at 448 to hold px/cell constant.

## Why large

Large raises path length while leaving horizon, render size, agent visibility and the data pipeline
untouched. Giant moves three things at once (path length, horizon, target visibility), so a drop in
planning performance there could not be attributed.

## Building the dataset

`scripts/build_pointmaze_dataset.py --size large` renders the released trajectories through
`lewam/envs/pointmaze_env.py` (which already accepts `task='large'`) into the same swm HDF5 schema
as the other cells — verified key-for-key and dtype-for-dtype against the medium file.

State replay is exact: OGBench's released `observations` column is a 2-D xy feature, **not**
simulator state, so qpos/qvel from the npz are what get set.

Rendering needs a GPU worker. On a CPU-only devbox the software EGL path gives 7.6 fps and
parallelism makes it *worse* (4 processes → 4 fps aggregate, the rasteriser is already
multi-threaded), so 1M frames takes 36.5 h there against ~1 h on one A100.
