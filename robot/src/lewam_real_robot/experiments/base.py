from abc import ABC
from pathlib import Path

from ..algorithms import build_algorithm
from ..datasets import build_dataset
from ..utils.spec import spec_fields


class BaseExperiment(ABC):
    """Orchestration scaffolding shared by every experiment. Holds the config + output
    paths, builds the algorithm/dataset through the lazy factories, and dispatches string
    `tasks` to methods via exec_task.

    Task methods (training, validation, ...) are supplied by mixins, so an experiment only
    implements the tasks it actually needs: a train+eval run mixes both, an eval-only
    robotics run mixes just the eval task.
    """

    def __init__(self, root_cfg, output_dir, ckpt_path=None):
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.output_dir = Path(output_dir)
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.algo = None
        self.dataset = None
        self._trained = False

    def _build_algo(self):
        if self.algo is None:
            self.algo = build_algorithm(self.root_cfg.algorithm)
        return self.algo

    def _build_dataset(self):
        if self.dataset is None:
            # the training task builds its own augmented source; this one only has to
            # answer what modalities exist
            self.dataset = build_dataset(self.root_cfg.dataset, augment=False)
        return self.dataset

    def _check_modalities(self):
        def fields(group):
            # list form: entries are the batch fields themselves; dict form: each entry
            # reads the field(s) named by `from` (default: the entry name)
            if hasattr(group, "items"):
                return {f for name, spec in group.items() for f in spec_fields(name, spec)}
            return set(group)

        algo = self.root_cfg.algorithm
        required = fields(algo.conditioning) | fields(algo.predict)
        missing = required - set(self._build_dataset().provided_modalities)
        if missing:
            raise ValueError(
                f"dataset (backend={self.root_cfg.dataset.backend}) lacks modalities "
                f"required by algorithm: {sorted(missing)}"
            )

    def exec_task(self, task):
        """Run one stage named by string. Each task is a method the experiment's mixins
        provide; a declared task with no method is a clean error, so composing a subset
        of tasks is safe.
        """
        if hasattr(self, task) and callable(getattr(self, task)):
            print(f"Executing task: {task} out of {list(self.cfg.get('tasks', [task]))}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"task '{task}' not defined for {self.__class__.__name__} or is not callable."
            )
