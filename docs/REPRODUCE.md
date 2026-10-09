# Reproducing the results

Evaluation needs a checkpoint under `$STABLEWM_HOME/checkpoints/<run_name>/`: download one of ours, or train
your own. Datasets are set up as in [DATA.md](DATA.md).

## Checkpoints

| task | Hugging Face model |
|---|---|
| Push-T | [LeWAM/lewam-pusht](https://huggingface.co/LeWAM/lewam-pusht) |
| OGBench Cube | [LeWAM/lewam-cube](https://huggingface.co/LeWAM/lewam-cube) |
| OGBench Scene | [LeWAM/lewam-scene](https://huggingface.co/LeWAM/lewam-scene) |
| OGBench Puzzle 3x3 | [LeWAM/lewam-puzzle](https://huggingface.co/LeWAM/lewam-puzzle) |
| PointMaze Large | [LeWAM/lewam-pointmaze-large](https://huggingface.co/LeWAM/lewam-pointmaze-large) |
| TwoRoom | [LeWAM/lewam-tworoom](https://huggingface.co/LeWAM/lewam-tworoom) |
| DMControl Reacher | [LeWAM/lewam-reacher](https://huggingface.co/LeWAM/lewam-reacher) |
| DexMimicGen Drawer Cleanup | [LeWAM/lewam-drawer](https://huggingface.co/LeWAM/lewam-drawer) |
| DexMimicGen Transport | [LeWAM/lewam-transport](https://huggingface.co/LeWAM/lewam-transport) |
| RoboMimic Tool Hang | [LeWAM/lewam-toolhang](https://huggingface.co/LeWAM/lewam-toolhang) |

All are goal-reaching (`gr`) checkpoints. Each holds `lewam_best.pt` and `lewam_config.json`:
<!-- TODO: task-completion (tc) checkpoints for drawer, transport and tool hang -->

```bash
hf download LeWAM/lewam-pusht --local-dir $STABLEWM_HOME/checkpoints/lewam-pusht
```

## Training

There are two training protocols, one config each in `configs/train/`:

- **Goal reaching** (`gr`): the policy is conditioned on a goal frame sampled up to `model.H_max` chunks ahead.
- **Task completion** (`tc`): classic behavior cloning, without a goal.

```bash
python scripts/train_lewam.py --config-name gr dataset_path=$STABLEWM_HOME/datasets/pusht.h5 run_name=gr_pusht
python scripts/train_lewam.py --config-name tc dataset_path=$STABLEWM_HOME/datasets/toolhang.h5 run_name=tc_toolhang
```

NOTE: the paper's task-completion runs on *Tool Hang* use only its scene camera (`pixels`, the default `data.views`),
although the dataset also has a wrist camera ([DATA.md](DATA.md#download)). *Drawer* and *Transport* use all three
cameras of their multi-camera datasets:

```bash
python scripts/train_lewam.py --config-name tc run_name=tc_drawer \
    dataset_path=$STABLEWM_HOME/datasets/drawer_3view/drawer_multiview.h5 "data.views=[pixels,pixels_r0eih,pixels_r1eih]"
python scripts/train_lewam.py --config-name tc run_name=tc_transport \
    dataset_path=$STABLEWM_HOME/datasets/transport_3view/transport_multiview.h5 "data.views=[pixels,pixels_r0eih,pixels_r1eih]"
```

Both protocols read the uncompressed frames by default (see [DATA.md](DATA.md#uncompressed-frames)). Every option is
declared in `configs/train/base.yaml` with the paper's value. Override any as `key=value`, e.g. `seed=0` or
`train.w_reg=0` (no SIGReg). The run is written to `$STABLEWM_HOME/checkpoints/<run_name>`.

## Evaluation

One script evaluates every task. `--config-name` picks the task's config (`configs/eval/<task>.yaml`) and `policy`
the run under `$STABLEWM_HOME/checkpoints`:

```bash
python scripts/eval_lewam.py --config-name <task> policy=<run_name> [key=value ...]
```

`<task>` is one of `pusht`, `cube`, `scene`, `puzzle`, `pointmaze`, `reacher`, `tworoom`, `drawer`,
`transport`, `toolhang`. Every option is declared in `configs/eval/base.yaml`.
<!-- TODO: when we add more configs, be sure to update this -->

Results and videos are written to `$STABLEWM_HOME/eval/<mode>/<run_name>/`. The paper reports 50 episodes for
each of the seeds 42, 0 and 1 (`seed=...`).
<!-- TODO: we might report more in the arxiv version or after rebuttal, so this might change-->

**Policy and planners:**

| | overrides |
|---|---|
| reactive policy | `eval.mode=lewam_policy` |
| best-of-K | `eval.mode=lewam_plan eval.plan_mode=best_of_k` |
| gradient planner | `eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01` |

Planners score 32 proposals (`eval.num_proposals`) over a horizon of the goal offset divided by the frameskip,
and execute one chunk (frameskip actions, `eval.exec_actions`) before replanning.

**Protocols:**

- **Goal reaching** (the task configs' defaults, with a `gr` checkpoint): start at a sampled step of a dataset
  episode and reach the frame `eval.goal_offset_steps` later within `eval.eval_budget` actions. Episodes come
  from the training data.
- **Task completion** (with a `tc` checkpoint): `eval.full_traj=true eval.task_only=true`. Each episode runs from
  its first frame, and success is the task's own success check. Used on drawer, transport and tool hang.
- **Long horizon**: `eval.random_start=false eval.min_episode_len=101 eval.exec_actions=25
  eval.goal_offset_steps=<H> eval.eval_budget=<2H>`, with a planner.

| tasks | goal offset | budget |
|---|---:|---:|
| Push-T, Cube, Scene, Puzzle, Reacher | 25 | 50 |
| Drawer, Transport, Tool Hang, PointMaze | 50 | 100 |
| TwoRoom | 100 | 150 |

Push-T, Cube, Reacher and TwoRoom follow [LeWM](https://arxiv.org/abs/2603.19312) (Appendix F.1).

For example, the gradient planner with our Push-T checkpoint:

```bash
python scripts/eval_lewam.py --config-name pusht policy=lewam-pusht seed=42 \
    eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01
```

## Results of the released checkpoints

Goal-reaching success (%), mean ± std over seeds 42, 0 and 1:
<!-- TODO: PointMaze, TwoRoom, Reacher, Transport, Tool Hang -->

| task | reactive | best-of-32 | gradient planner |
|---|---|---|---|
| Push-T | 85.3 ± 5.0 | 95.3 ± 2.3 | 98.7 ± 1.2 |
| OGBench Cube | 99.3 ± 1.2 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| OGBench Scene | 90.7 ± 5.8 | 96.0 ± 2.0 | 97.3 ± 1.2 |
| OGBench Puzzle 3x3 | 76.7 ± 8.1 | 76.7 ± 10.1 | 78.7 ± 9.5 |
| DexMimicGen Drawer Cleanup | 64.0 ± 3.5 | 64.0 ± 8.7 | 60.0 ± 7.2 |
