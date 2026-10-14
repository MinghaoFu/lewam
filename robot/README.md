# LeWorldActionModel (LeWAM)

## Tasks

| task | dataset | episodes | instruction |
| --- | --- | --- | --- |
| Pick and Place T | [`ehalicki/b601_pusht_pick_and_place`](https://huggingface.co/datasets/ehalicki/b601_pusht_pick_and_place) | 200 | Pick up the T block and place it onto the outline. |
| Push-T | [`ehalicki/b601_pusht`](https://huggingface.co/datasets/ehalicki/b601_pusht) | 100 | Push the T block onto the outline. |
| Tape Stack | [`ehalicki/b601_tape_stack`](https://huggingface.co/datasets/ehalicki/b601_tape_stack) | 100 | Stack the green roll of tape on the white roll of tape. |

Each episode records a ZED-Mini scene view, a wrist camera, 7-D joint-target actions and a
21-D follower state at 30 Hz.

## Results

Successes out of 30 trials per task (Table 2 of the paper).

| method | PnP-T | Push-T | Tape Stack | average | GFLOPs / chunk |
| --- | --- | --- | --- | --- | --- |
| Diffusion Policy | 19/30 | 16/30 | 0/30 | 38.9% | 270.0 |
| π0.5 | 14/30 | 3/30 | 9/30 | 28.9% | 3603.6 |
| LeWAM (frozen V-JEPA 2.1 encoder) | 15/30 | 12/30 | 5/30 | 35.6% | 270.2 |
| LeWAM | 16/30 | 8/30 | 6/30 | 33.3% | 270.2 |

## Installation

Uses [`uv`](https://docs.astral.sh/uv/).

This folder is its own Python project (`lewam-real-robot`, imported as `lewam_real_robot`),
separate from the simulation code at the repository root.

```bash
cd robot
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"      # or a lighter extra, e.g. ".[b601]"
```

Extras: `b601` (the real arm), `baselines` (Diffusion Policy, π0.5), `serve` (inference
server, no lerobot or hardware deps), `vjepa`, `wandb`, `all`.

lerobot is installed from a fork ([ErykHalicki/lerobot](https://github.com/ErykHalicki/lerobot)),
pinned in `pyproject.toml`, which carries the B601 arm and ZED camera backends; lerobot
from PyPI will not drive the robot. `external/vjepa2` holds the few V-JEPA 2.1 model
files the encoder needs, appended to `sys.path` at import.

## Training

The paper's arms, shown for PnP-T (swap `_pnpt` for `_pusht`). Tape Stack has a
recording config but no training config here.

```bash
python main.py run=selfwam_mot_mse_b601_pnpt           # LeWAM
python main.py run=selfwam_mot_mse_frozen_b601_pnpt    # LeWAM, frozen V-JEPA 2.1 encoder
python main.py run=diffusion_b601_pnpt                 # Diffusion Policy
python main.py run=pi05_frozen_b601_pnpt               # π0.5, action expert only
```

Every runnable arm is one file in [`configs/run/`](configs/README.md). Any config value can
be overridden on the CLI, e.g. `experiment.max_steps=30000 wandb.mode=disabled`.

## Data collection

Demonstrations are recorded by joint-space teleoperation from a reBot Arm 102 leader to the
B601-DM follower, with a ZED-Mini scene camera and a wrist-mounted Logitech C270. Printable
parts (the T block and the wrist camera mount) are in [`external/hardware/`](external/hardware).

```bash
./src/lewam_real_robot/scripts/b601/teleop.sh                              # teleoperate only
./src/lewam_real_robot/scripts/b601/record.sh --config-name=b601_pnpt      # record a dataset
```

One recording config per task lives in `src/lewam_real_robot/scripts/b601/configs/`. Each
script takes `--help`.

## Running on the robot

The robot host sends its observation window to a GPU box, receives a 36-action chunk, and
executes the first 18 at 30 Hz before asking again.

```bash
# GPU box
./src/lewam_real_robot/scripts/serve.sh --port 8080

# robot host
python main.py run=b601_irl load=<wandb_run_id> eval.server=<gpu-host>:8080
```

## Layout

```
configs/             Hydra groups: run, algorithm, dataset, eval, experiment, embodiment
src/lewam_real_robot/
  algorithms/        encoders, MoT predictor, losses, lerobot baseline wrappers
  datasets/          LeRobot dataset loading, normalization, augmentation
  experiments/       training, held-out validation and on-robot eval
  scripts/           teleop, recording, inference server
  utils/
external/
  vjepa2/            V-JEPA 2.1 model files used by the encoder
  hardware/          STL files for the T block and wrist camera mount
```
