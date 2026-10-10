<h1 align="center">LeWAM</h1>
<h3 align="center">End-to-end World and Action Modeling with JEPAs</h3>

<p align="center">
  <a href="https://minghaofu.com/">Minghao Fu*</a>,
  <a href="https://tavis-siebert.github.io/">Tavis Siebert*</a>,
  <a href="https://eryk.ca/">Eryk Halicki</a> and
  <a href="https://randallbalestriero.github.io/">Randall Balestriero</a>
</p>

<p align="center">
  <a href="https://le-wam.github.io/">Website</a> &middot;
  <a href="https://minghaofu.github.io/files/lewam.pdf">Paper</a> &middot;
  <a href="https://huggingface.co/collections/LeWAM/lewam-simulation-checkpoint">Checkpoints</a> &middot;
  <a href="https://huggingface.co/collections/LeWAM/lewam-simulation-data">Datasets</a>
</p>

<p align="center">
  <img src="assets/lewam_teaser.gif" alt="LeWAM animated teaser" width="100%">
</p>

## Using the code

**Installation:**
```bash
bash install.sh lewam cu128
conda activate lewam
export STABLEWM_HOME=/path/to/storage
export MUJOCO_GL=egl
```

Change `cu128` to match your CUDA setup. MuJoCo rendering requires `libEGL`.

## Data

Datasets use HDF5 format. Datasets are available in the [Hugging Face collection](https://huggingface.co/collections/LeWAM/lewam-simulation-data). Decompress downloaded files with:

```bash
zstd -d --long=27 ${dataset_name}.h5.zst
```

Store datasets under `$STABLEWM_HOME/datasets/`. See [docs/DATA.md](docs/DATA.md) for the dataset format and loading options.

Prepare each dataset once before training:
```bash
python scripts/decompress_h5.py \
    --dataset_path "$STABLEWM_HOME/datasets/pusht.h5" \
    --decomp_dir "$STABLEWM_HOME/decomp"
```

## Training

Training configs live under `configs/train/`. Use `gr` for goal-conditioned visual planning and `tc` for task completion (behavior cloning).

To train a model for goal-conditioned visual planning:
```bash
python scripts/train_lewam.py --config-name gr \
    dataset_path="$STABLEWM_HOME/datasets/pusht.h5" run_name=gr_pusht
```

To train a model for task completion:
```bash
python scripts/train_lewam.py --config-name tc \
    dataset_path="$STABLEWM_HOME/datasets/toolhang.h5" run_name=tc_toolhang
```

Override options with `key=value`, such as `seed=0` or `train.w_reg=0`. Checkpoints are saved to `$STABLEWM_HOME/checkpoints/<run_name>/`.

## Evaluation

Evaluation configs live under `configs/eval/`. Set `policy` to the training run name.

**Visual Planning:**

```bash
python scripts/eval_lewam.py --config-name pusht \
    policy=gr_pusht eval.mode=lewam_plan \
    eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01
```

**Behavior Cloning:**

```bash
python scripts/eval_lewam.py --config-name toolhang \
    policy=tc_toolhang eval.mode=lewam_policy \
    eval.full_traj=true eval.task_only=true
```

Replace `pusht` or `toolhang` with the corresponding task config. Results are saved under `$STABLEWM_HOME/eval/`.
See [docs/REPRODUCE.md](docs/REPRODUCE.md) for our checkpoints and the paper's evaluation settings.

## Main results

Goal-reaching visual planning: mean success rate across three seeds.

| Task | SR (%) | Checkpoint | Dataset |
|---|---:|---|---|
| Two-Room | 100.0 | [Checkpoint](https://huggingface.co/LeWAM/lewam-tworoom) | [Data](https://huggingface.co/datasets/LeWAM/lewam-tworoom) |
| Reacher | 100.0 | [Checkpoint](https://huggingface.co/LeWAM/lewam-reacher) | [Data](https://huggingface.co/datasets/LeWAM/lewam-reacher) |
| Push-T | 98.7 | [Checkpoint](https://huggingface.co/LeWAM/lewam-pusht) | [Data](https://huggingface.co/datasets/LeWAM/lewam-pusht) |
| Cube | 100.0 | [Checkpoint](https://huggingface.co/LeWAM/lewam-cube) | [Data](https://huggingface.co/datasets/LeWAM/lewam-cube) |
| PointMaze | 100.0 | [Checkpoint](https://huggingface.co/LeWAM/lewam-pointmaze-large) | [Data](https://huggingface.co/datasets/LeWAM/lewam-pointmaze-large) |
| Scene | 97.3 | [Checkpoint](https://huggingface.co/LeWAM/lewam-scene) | [Data](https://huggingface.co/datasets/LeWAM/lewam-scene) |
| Puzzle | 80.7 | [Checkpoint](https://huggingface.co/LeWAM/lewam-puzzle) | [Data](https://huggingface.co/datasets/LeWAM/lewam-puzzle) |
| ToolHang | 94.7 | [Checkpoint](https://huggingface.co/LeWAM/lewam-toolhang) | [Data](https://huggingface.co/datasets/LeWAM/lewam-toolhang) |
| Transport | 83.3 | [Checkpoint](https://huggingface.co/LeWAM/lewam-transport) | [Data](https://huggingface.co/datasets/LeWAM/lewam-transport) |
| DrawerCleanup | 80.7 | [Checkpoint](https://huggingface.co/LeWAM/lewam-drawer) | [Data](https://huggingface.co/datasets/LeWAM/lewam-drawer) |

Reacher uses the original LeWM dataset. See [docs/REPRODUCE.md](docs/REPRODUCE.md) for download and evaluation commands.

## Acknowledgments

We thank [LeWorldModel](https://github.com/lucas-maes/le-wm) and [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for their open-source contributions.
