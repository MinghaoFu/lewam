# Data

Every task is one h5 file. Training reads it from `dataset_path`; eval reads
`$STABLEWM_HOME/datasets/<eval.dataset_name>.h5`, where the task configs use the file names below. To keep a file
under another name or path, point eval at it with `eval.dataset_name=<path relative to $STABLEWM_HOME/datasets,
without .h5>`.

## Download

| task | Hugging Face dataset | file |
|---|---|---|
| Push-T | [LeWAM/lewam-pusht](https://huggingface.co/datasets/LeWAM/lewam-pusht) | `pusht.h5.zst` |
| OGBench Cube | [LeWAM/lewam-cube](https://huggingface.co/datasets/LeWAM/lewam-cube) | `cube.h5.zst` |
| OGBench Scene | [LeWAM/lewam-scene](https://huggingface.co/datasets/LeWAM/lewam-scene) | `scene_expert.h5.zst` |
| OGBench Puzzle 3x3 | [LeWAM/lewam-puzzle](https://huggingface.co/datasets/LeWAM/lewam-puzzle) | `puzzle_expert.h5.zst` |
| PointMaze Large | [LeWAM/lewam-pointmaze-large](https://huggingface.co/datasets/LeWAM/lewam-pointmaze-large) | `pointmaze_large.h5` |
| TwoRoom | [LeWAM/lewam-tworoom](https://huggingface.co/datasets/LeWAM/lewam-tworoom) | `tworoom.h5` |
| DMControl Reacher | [LeWAM/lewam-reacher](https://huggingface.co/datasets/LeWAM/lewam-reacher) | `reacher_policy.h5` |
| DexMimicGen Drawer Cleanup | [LeWAM/lewam-drawer](https://huggingface.co/datasets/LeWAM/lewam-drawer) | `drawer.h5.zst` |
| DexMimicGen Transport | [LeWAM/lewam-transport](https://huggingface.co/datasets/LeWAM/lewam-transport) | `transport.h5` |
| RoboMimic Tool Hang | [LeWAM/lewam-toolhang](https://huggingface.co/datasets/LeWAM/lewam-toolhang) | `toolhang.h5` |

Download each file into `$STABLEWM_HOME/datasets`, and decompress the `.zst` files there:

```bash
hf download LeWAM/lewam-pusht pusht.h5.zst --repo-type dataset --local-dir $STABLEWM_HOME/datasets
zstd -d --long=27 --rm $STABLEWM_HOME/datasets/pusht.h5.zst
```

Drawer, transport and tool hang also come with their wrist cameras, in the
[appendix collection](https://huggingface.co/collections/LeWAM/appendix): download the whole repository, keep
its files together, and train on its `*_multiview.h5`, which links the cameras into one file.

| task | Hugging Face dataset | cameras (`data.views`) |
|---|---|---|
| DexMimicGen Drawer Cleanup | [LeWAM/lewam-drawer-3view](https://huggingface.co/datasets/LeWAM/lewam-drawer-3view) | `pixels`, `pixels_r0eih`, `pixels_r1eih` |
| DexMimicGen Transport | [LeWAM/lewam-transport-3view](https://huggingface.co/datasets/LeWAM/lewam-transport-3view) | `pixels`, `pixels_r0eih`, `pixels_r1eih` |
| RoboMimic Tool Hang | [LeWAM/lewam-toolhang-2view](https://huggingface.co/datasets/LeWAM/lewam-toolhang-2view) | `pixels`, `pixels_r0eih` |

```bash
hf download LeWAM/lewam-drawer-3view --repo-type dataset --local-dir $STABLEWM_HOME/datasets/drawer_3view
python scripts/train_lewam.py --config-name tc dataset_path=$STABLEWM_HOME/datasets/drawer_3view/drawer_multiview.h5 \
    run_name=tc_drawer "data.views=[pixels,pixels_r0eih,pixels_r1eih]"
```

## Uncompressed frames

The h5 files store frames compressed (with HDF5 filter plugins, read through `hdf5plugin`). By default training
reads an uncompressed copy of the frames instead (`data.source=decomp`), written once per dataset:

```bash
python scripts/decompress_h5.py --dataset_path $STABLEWM_HOME/datasets/pusht.h5 \
    --decomp_dir $STABLEWM_HOME/decomp
```

This writes `<decomp_dir>/<stem>/<column>.npy` for each column in `--views` (default `pixels`). The script checks
free disk space before writing, and a column only gets its final name once complete. Rebuild the copy whenever
the h5 changes: the trainer only checks that the shapes match.

| `data.*` | disk | memory | per frame read |
|---|---|---|---|
| `source=decomp load_in_ram=true` (default) | h5 + copy | the whole copy | nothing |
| `source=decomp load_in_ram=false` | h5 + copy | the parts read so far, while memory allows | nothing |
| `source=h5 load_in_ram=false` | h5 | small | one decompression |

All three give identical samples. Without the memory for the whole copy, memory-map it (`load_in_ram=false`); with
little disk, or data on a network drive, read the h5 directly and raise `data.num_workers`.

Each camera's copy takes N x 3 x 224 x 224 bytes for N frames:

| dataset | h5 | copy |
|---|---|---|
| toolhang | 5.0 GB | 14.4 GB |
| drawer | 18.5 GB | 44.9 GB |
| transport | 32.4 GB | 63.2 GB |
| tworoom | 12.8 GB | 139 GB |
| cube | 102 GB | 303 GB |
| scene, puzzle (each) | | 302 GB |
| pusht | 46 GB | 352 GB |

## Using your own data

An h5 needs these columns:

| column | shape | |
|---|---|---|
| `pixels` (extra cameras, e.g. `pixels_r0eih`) | (N, H, W, 3) or (N, 3, H, W) uint8 | cameras chosen with `data.views` |
| `action` | (N, action_dim) | |
| `ep_offset`, `ep_len` | per episode | episode boundaries |
| `success` | (N,) bool | only for `data.terminal_state=success` |

H and W must equal `data.img_size`; frames are not resized. Actions are z-scored per dimension over the dataset,
and the statistics are saved with the checkpoint for eval.

## What one sample contains

Every frame at least `data.frameskip` (fs) frames before its episode's terminal frame (the last frame, or the first
`success` frame) starts one sample. Starts are split `data.train_split` / rest into train and validation, and each
epoch visits every training start once.

| part | frames |
|---|---|
| history (`model.history_len`) | start - (history_len - 1) * `model.history_stride`, ..., start |
| action targets | the next `model.num_actions_pred` actions |
| state targets | start + fs, start + 2fs, ... (`model.num_states_pred`) |
| goal (`model.goal_type=sampled`) | start + h * fs, h ~ U[1, `model.H_max`], clamped to the terminal frame |
| goal (`model.goal_type=terminal`) | the terminal frame |

History frames before the episode's start are replaced by its oldest frame inside the episode and masked out of
attention; targets past the
terminal frame are masked out of the loss. `goal_type=none` has no goal.
