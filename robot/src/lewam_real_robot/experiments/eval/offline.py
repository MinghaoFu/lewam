"""Held-out loss on the episodes training never saw."""

from .base import EvalResult


class OfflineEval:
    """Scores a model on held-out episodes with the same `model.loss` the training loop
    optimizes, so `eval/loss` and `train/loss` are directly comparable and the gap between
    them reads as overfitting. No environment and no robot: this measures whether the
    policy generalizes to unseen data, while the rollout backends measure whether it does
    the task. A run that is fine here and fails on the arm is an execution problem, not a
    data one -- which is the whole reason to have both.

    The dataset config arrives as `cfg.dataset` (the eval yaml interpolates `${dataset}`),
    built with `split="val"` and augmentation off, so it holds exactly the episodes
    `dataset.validation_split` kept out of training.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = int(cfg.get("batch_size", 32))
        self.num_workers = int(cfg.get("num_workers", 0))
        self.max_batches = cfg.get("max_batches")
        # loss terms draw noise per item, so an unseeded score wanders between
        # checkpoints on the draw alone
        self.seed = int(cfg.get("seed", 0))
        self._dataset = None

    def _build(self, model):
        if self._dataset is None:
            from ...datasets import build_dataset

            self._dataset = build_dataset(
                self.cfg.dataset,
                # the stats training used, never stats recomputed from the held-out episodes:
                # those are a different distribution and make the two losses incomparable
                norm_stats_override=getattr(model, "norm_stats", None),
                augment=False,
                split="val",
            )
        return self._dataset

    def run(self, model):
        import torch
        from torch.utils.data import DataLoader

        dataset = self._build(model)
        if not len(dataset):
            raise ValueError(
                f"held-out split of '{self.cfg.dataset.get('repo_id')}' produced no "
                f"samples; check dataset.validation_split and the episode windows"
            )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            # samples are laid out episode by episode, so an unshuffled `max_batches` would
            # score the first few held-out episodes rather than the split
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
        )
        device = next(model.parameters()).device
        totals, seen, batches = {}, 0, 0
        was_training = model.training
        # fork rather than reseed: the training loop's own sampling must not inherit
        # the eval seed when this runs mid-training
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            model.eval()
            try:
                with torch.no_grad():
                    for batch in loader:
                        if self.max_batches and batches >= int(self.max_batches):
                            break
                        batch = {
                            k: v.to(device) if torch.is_tensor(v) else v
                            for k, v in batch.items()
                        }
                        out = model.loss(batch)
                        n = next(len(v) for v in batch.values() if torch.is_tensor(v))
                        for key, value in out.items():
                            if key == "loss" or key.startswith(("loss/", "var/")):
                                totals[key] = totals.get(key, 0.0) + value.item() * n
                        seen += n
                        batches += 1
            finally:
                model.train(was_training)

        return EvalResult(
            metrics={**{k: v / seen for k, v in totals.items()}, "holdout_samples": seen}
        )
