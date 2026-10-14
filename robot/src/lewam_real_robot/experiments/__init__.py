from .base import BaseExperiment
from .generic import GenericExperiment

exp_registry = dict(
    base=GenericExperiment,
)


def build_experiment(cfg, output_dir=None, ckpt_path=None):
    """Look up the experiment class by the selected experiment yaml (cfg.experiment._name,
    stamped in main.py from Hydra's group choices) and instantiate it.
    """
    name = cfg.experiment.get("_name", None)
    if name not in exp_registry:
        raise ValueError(
            f"experiment '{name}' not found in registry {list(exp_registry.keys())}. "
            "Register it in experiments/__init__.py under the same name as its yaml file."
        )
    return exp_registry[name](cfg, output_dir, ckpt_path)
