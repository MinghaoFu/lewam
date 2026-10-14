# utils

Cross-cutting helpers called as hooks by the training loop: wandb logging, checkpoint I/O, viz, sync. `wandb.py` is the only wandb-aware module.

`normalize.py` and `augment.py` are source wrappers rather than loop hooks: `build_dataset` stacks them around a backend, and both operate only on the batch dict, so neither knows which backend produced it.

`cameras.py` is unrelated to the training loop: ZED/webcam device-detection shared by the hardware scripts under `src/lewam_real_robot/scripts/`.
