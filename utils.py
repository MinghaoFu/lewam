from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained.

    Also persists the FULL run config (incl. ``action_pred`` / training flags) once per run, into
    ``checkpoints/<run_name>/full_config.yaml``. ``save_pretrained``'s ``config.json`` only stores
    ``cfg.model`` — it does NOT carry ``action_pred`` (``detach_decoder``, ``w_act``, ``head`` …),
    which is how a whole batch silently ran ``detach_decoder=false`` undetected (EXPERIMENTS.md §0).
    Now every run's exact settings are recoverable from its own folder.
    """

    def __init__(self, run_name, cfg, epoch_interval: int = 1, full_cfg=None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.full_cfg = full_cfg
        self._dumped_full = False

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _dump_full_config(self):
        """Write the complete cfg (action_pred included) to the run folder, once."""
        if self._dumped_full or self.full_cfg is None:
            return
        try:
            import stable_worldmodel as swm
            from omegaconf import OmegaConf
            run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), self.run_name)
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / 'full_config.yaml', 'w') as f:
                OmegaConf.save(self.full_cfg, f)
            self._dumped_full = True
        except Exception as e:  # never let config-dump kill a training run
            print(f'[SaveCkpt] full_config dump failed: {e!r}')

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        self._dump_full_config()
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )

class EMAActionEncoderCallback(Callback):
    """Opt-in EMA (momentum) update for ``model.action_encoder_ema`` (BYOL / I-JEPA).

    Only attached when ``action_pred.ema_target=true``. The EMA encoder is the
    stop-grad TARGET for the intention/act_emb prediction loss; the online
    ``action_encoder`` (which also feeds the conditioning ``past_act``) is the
    student. We update the teacher AFTER the optimizer step, mirroring
    ``stable_pretraining.callbacks.TeacherStudentCallback``: we key off
    ``trainer.global_step`` so the update fires once per OPTIMIZER step (correct
    under gradient accumulation / multi-optimizer frequencies), not once per
    micro-batch. Update rule (both params and buffers):
        p_ema = tau * p_ema + (1 - tau) * p_online
    Done under ``no_grad``; the teacher carries ``requires_grad_(False)``.
    """

    def __init__(self, tau: float = 0.99):
        super().__init__()
        self.tau = float(tau)
        self._last_global_step = -1

    @torch.no_grad()
    def _update(self, online, ema):
        tau = self.tau
        for p_ema, p_online in zip(ema.parameters(), online.parameters()):
            p_ema.mul_(tau).add_(p_online.detach(), alpha=1.0 - tau)
        # copy buffers verbatim (e.g. norm running stats); EMA on buffers is
        # unnecessary for the action encoder but we keep teacher == student here.
        for b_ema, b_online in zip(ema.buffers(), online.buffers()):
            b_ema.copy_(b_online)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        model = getattr(pl_module, "model", None)
        ema = getattr(model, "action_encoder_ema", None) if model is not None else None
        if ema is None:
            return  # ema_target off -> no teacher attached -> no-op
        # only update on iterations where the optimizer actually stepped
        step = trainer.global_step
        if step == self._last_global_step:
            return
        self._update(model.action_encoder, ema)
        self._last_global_step = step
