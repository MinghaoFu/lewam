# configs

Hydra config groups. Everything runnable is one file in `configs/run/`, which selects the
dataset, algorithm and eval together — so one name is a whole run.

```bash
source .venv/bin/activate
python main.py run=selfwam_mot_mse_b601_pnpt
```

`base.yaml` is Hydra's default config and the entry point, so `run=` is the whole command
line. It owns the run-level keys — `debug`, `wandb`, `resume`/`load`, `posthoc_prefix` —
and any of them can be overridden on the CLI or by a run config.

New variants are new yaml files, not code branches.

## The groups

| group | holds |
| --- | --- |
| `run/` | one file per runnable arm; the only thing `run=` accepts |
| `algorithm/` | the model: its modality spec, encoders and losses |
| `dataset/` | which repo, which episodes, which columns |
| `eval/` | how a trained model is scored |
| `experiment/` | training-loop knobs (batch size, steps, checkpoint cadence) |
| `embodiment/` | the physical rig an on-robot eval drives |

`algorithm/vision/` is a group nested inside `algorithm/`: it holds the camera head,
chosen with `- override vision: <option>` at the END of an algorithm's defaults list.
