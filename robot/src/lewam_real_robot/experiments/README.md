# experiments

The runners. `base.py` is `BaseExperiment` (builds algo/dataset, dispatches string `tasks`
via `exec_task`); `__init__.py` is the `exp_registry` + `build_experiment`. Concrete
experiments (e.g. `generic.py`) compose task mixins: `training/` supplies `TrainingMixin`,
`eval/` supplies `EvalMixin`. An experiment only mixes the tasks it needs.
