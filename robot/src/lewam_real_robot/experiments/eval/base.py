from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """The eval contract. Backends return this plain object and never touch wandb, so
    control loops never block on network I/O. utils/wandb.py is what wraps it for logging.
    """

    metrics: dict
    videos: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)
    episodes: list = field(default_factory=list)


class EvalMixin:
    """Supplies the `validation` task: run this backend's eval environment on the model and
    log the EvalResult. Mix into any experiment that should be evaluated, including
    eval-only ones (no training task). If the model was not trained in this run, the
    requested checkpoint is loaded first.
    """

    def validation(self):
        from . import build_eval
        from ...utils.wandb import log_eval

        eval_cfg = self.root_cfg.eval
        model, step = None, None
        if eval_cfg.get("server"):
            # remote inference: the server holds the model, so this box builds nothing and
            # owes the server only the name of the run to serve
            self._name_remote_run(eval_cfg)
        else:
            from accelerate import Accelerator

            from ...utils.checkpoint import load_checkpoint

            model = self._build_algo()
            if not self._trained and self.ckpt_path is not None:
                step = load_checkpoint(model, None, self.ckpt_path)
                # eval-only runs build and load on CPU, unlike a model the training loop
                # prepared, so move it or the whole eval runs on CPU
                model = Accelerator().prepare(model)

        backend = build_eval(eval_cfg)
        result = backend.run(model)
        # under remote inference the served checkpoint's step is known only once the
        # server answers the handshake
        step = step if step is not None else getattr(backend, "checkpoint_step", None)
        log_eval(result, step=step)
        # main.py reads these back for the post-hoc summary write-back (load= + eval-only)
        self.last_eval_result, self.last_eval_step = result, step

    def _name_remote_run(self, eval_cfg):
        """Fill in `eval.run` from `load=` when it wasn't set explicitly. A bare run id is
        qualified with this config's wandb entity/project, the same path main.py resolves
        checkpoints through.
        """
        from omegaconf import open_dict

        run = eval_cfg.get("run") or self.root_cfg.get("load")
        if run and "/" not in str(run):
            wandb_cfg = self.root_cfg.wandb
            run = f"{wandb_cfg.entity}/{wandb_cfg.project}/{run}"
        with open_dict(eval_cfg):
            eval_cfg.run = run
