# training

`TrainingMixin`: supplies the `training` task, a thin Accelerate loop (DDP + mixed
precision) with wandb metric logging and checkpointing. Mix into any experiment that
trains. The wandb run itself is opened and closed by `main.py` around all tasks.

`max_steps`, the LR schedule and the save/eval cadences all count optimizer steps, not
micro-batches. Knobs are in `configs/experiment/`.
