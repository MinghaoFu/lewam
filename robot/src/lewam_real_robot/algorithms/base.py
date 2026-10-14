import torch.nn as nn


class BaseAlgorithm(nn.Module):
    """The method. Backend-free: must NOT import from datasets/, experiments/, or eval/.

    It may import any external package it needs (torch, encoders, model libraries).
    Two methods form the whole contract the rest of the repo relies on:

      loss(batch)  -> dict containing a "loss" scalar for the training loop
      predict(obs) -> outputs, the framework-neutral inference API used by eval
    """

    # the key a prediction dict uses and the batch field its normalization stats are
    # keyed by. None means this algorithm predicts no actions (a world model), which
    # is what makes eval choose between executing a chunk and planning one.
    action_stream = None
    action_field = None

    def loss(self, batch):
        raise NotImplementedError

    def optim_params(self):
        """What the training loop hands the optimizer. Every parameter as one group by
        default; override to give some of them their own settings (a wrapped policy that
        trains its vision backbone at a different learning rate, say).
        """
        return self.parameters()

    def set_gradient_checkpointing(self, enabled):
        """Recompute blocks in the backward pass instead of storing their activations:
        lower peak memory for ~20-30% more compute. Called by the training loop from
        `experiment.gradient_checkpointing`; reaches every submodule that exposes the flag
        (trunk and encoders), and is a no-op for algorithms whose modules don't.
        """
        for module in self.modules():
            if module is not self and hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = bool(enabled)

    def predict(self, obs):
        raise NotImplementedError

    def summary(self):
        """Static stats logged to wandb.config at startup. Defaults to param counts;
        override (and call super().summary()) to add per-algorithm stats like encoder
        size, block count, or latent dims.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"params/total": total, "params/trainable": trainable}
