# eval

`EvalMixin` supplies the `validation` task; `build_eval` is the lazy per-backend factory.
Each runner consumes the algorithm's inference API and returns a wandb-free `EvalResult`
(metrics, videos, episodes). Same `backend=` as the dataset, so training and eval share
one world.

`offline.py` is the exception: `backend: offline` has no world, scoring `model.loss` over
the episodes `dataset.validation_split` held out, so `eval/loss` compares directly against
`train/loss`. It takes the dataset config through `${dataset}` and pairs with any backend
that can hold episodes out.

`chunking.py`'s drivers buffer what a rollout conditions on. Observation columns come off
the rig; past actions cannot, since no frame carries the action that produced it, so the
driver replays what it commanded itself.
