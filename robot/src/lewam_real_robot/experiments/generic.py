from .base import BaseExperiment
from .eval.base import EvalMixin
from .training.base import TrainingMixin


class GenericExperiment(TrainingMixin, EvalMixin, BaseExperiment):
    """Default train-then-evaluate experiment for the shared `base` schedule: everything
    task-specific (dataset, algorithm, eval world) comes from the other config groups,
    so one class serves every config that needs no bespoke behavior.

    That includes the real-robot runs. An embodiment with no sim env validates offline --
    held-out loss on the episodes `dataset.validation_split` kept back -- and its task
    success is measured post-hoc on the robot machine, which is an eval config rather than
    an experiment class (`eval=b601_zed_and_wrist`).
    """
