from omegaconf import OmegaConf
import torch

_OPTIMIZERS = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


def build_optimizer(params, cfg):
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    name = kwargs.pop("name").lower()
    if name not in _OPTIMIZERS:
        raise ValueError(f"unknown optimizer '{name}'. available: {list(_OPTIMIZERS)}")
    return _OPTIMIZERS[name](params, **kwargs)


def build_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0 — the Diffusion Policy schedule."""
    import math

    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


class EMA:
    """Exponential moving average of model weights; eval/checkpoint use the EMA copy
    (Diffusion Policy deploys the EMA model, not the raw one).
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)
