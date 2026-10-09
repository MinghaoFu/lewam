# LeWAM
### End-to-end World and Action Modeling with JEPAs

[Minghao Fu*](https://minghaofu.com/), [Tavis Siebert*](https://tavis-siebert.github.io/), [Eryk Halicki](https://eryk.ca/) and [Randall Balestriero](https://randallbalestriero.github.io/)

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

## Acknowledgments

We thank [LeWorldModel](https://github.com/lucas-maes/le-wm) and [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for their open-source contributions.
