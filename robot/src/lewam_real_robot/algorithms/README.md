# algorithms

The method layer: models, loss/step, and inference API. May import external packages
(torch, encoders, backend libs) but never from `datasets/`, `experiments/`, or `eval/`.

In this layer, the `PredictiveModel`
base is the only code that reads `from`/`encoder`; the `GenericViTTrunk` it feeds sees
only geometry keys and tensors.

There is one predictor (`name: generic_vit_predictor`). Each predicted stream declares
`type: flow | direct`, and `_forward_train` derives its seeding and target from that:
`flow` seeds noise and targets `x1 - x0`; `direct` reads its `input` stream's tokens and
targets the next latent. Adding a paradigm is a config change, not a subclass.

`name: lerobot_policy` (`lerobot_policy.py`) wraps lerobot's Diffusion Policy and π0.5
behind the same training and inference API, for the baselines.
