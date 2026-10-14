import copy
import math
import signal
import time
from contextlib import nullcontext

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader

from ...datasets import build_dataset
from ...utils.checkpoint import load_checkpoint, run_ckpt_dir, save_checkpoint
from ...utils.optim import EMA, build_lr_scheduler, build_optimizer
from ...utils.wandb import log_eval, log_metrics, log_model_summary, read_summary, update_summary

# accelerate's own mixed_precision string, translated to the autocast dtype it implies
# (`"no"` -> None, full precision). Used wherever a training-owned autocast region
# (e.g. the encoding cache build) needs to match the run's configured precision rather
# than a hardcoded dtype.
_AMP_DTYPE = {"no": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def _improved(value, best, mode):
    """Whether `value` beats `best` under `best_mode` (configs/base.yaml)."""
    return value > best if mode == "max" else value < best


class TrainingMixin:
    """Supplies the `training` task: a thin Accelerate loop (DDP + mixed precision) with
    wandb logging and checkpointing. Mix into an experiment that should train. Reads the
    algorithm/dataset/paths from the BaseExperiment it is composed with. The wandb run
    itself is opened/closed by main.py around all tasks, not here.
    """

    def training(self):
        exp = self.cfg
        acc = Accelerator(mixed_precision=exp.mixed_precision)
        self._build_algo()

        self.algo.set_gradient_checkpointing(exp.get("gradient_checkpointing", False))

        opt = build_optimizer(self.algo.optim_params(), exp.optimizer)
        start_step = load_checkpoint(self.algo, opt, self.ckpt_path) if self.ckpt_path else 0

        # a model already carrying norm_stats was filled from a checkpoint: reuse them, or
        # the run drifts from what those weights were trained on
        recompute_stats = exp.get("recompute_norm_stats", False)
        reuse_stats = None if recompute_stats else getattr(self.algo, "norm_stats", None)
        self._build_training_dataset(acc, reuse_stats)

        if acc.is_main_process:
            log_model_summary(self.algo)

        model, opt = acc.prepare(self.algo, opt)
        batch_size = exp.batch_size
        effective_batch = exp.get("effective_batch", None)
        if str(batch_size) == "auto":
            batch_size = self._calibrate_batch_size(acc, model, opt, exp)
            if effective_batch:
                per_rank = max(1, math.ceil(int(effective_batch) / acc.num_processes))
                batch_size = self._fit_micro_batch(batch_size, per_rank)
        accum_steps = self._accumulation_steps(acc, batch_size, effective_batch)

        num_workers = exp.get("num_workers", 0)
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=exp.get("pin_memory", False),
            persistent_workers=num_workers > 0,
        )
        loader = acc.prepare(loader)
        total_steps = exp.max_steps
        sched = build_lr_scheduler(opt, exp.get("lr_warmup_steps", 0), total_steps)
        # on resume the scheduler is fresh while the optimizer state is not: fast-forward
        # it, or the LR re-enters warmup at the resume point
        for _ in range(start_step):
            sched.step()
        ema_decay = exp.get("ema_decay", None)
        ema = EMA(acc.unwrap_model(model), decay=ema_decay) if ema_decay else None

        save_every = exp.get("save_every", 0)
        eval_every = exp.get("eval_every", 0)
        save_intermittent = exp.get("save_intermittent", False)
        ckpt_dir = run_ckpt_dir(fallback=self.output_dir)

        best_key = self.root_cfg.get("best_metric", None)
        best_key = best_key.removeprefix("eval/") if best_key else None
        best_mode = self.root_cfg.get("best_mode", "min")
        if best_key and best_mode not in ("min", "max"):
            raise ValueError(f"best_mode must be 'min' or 'max', got '{best_mode}'")
        # a resume must not call its first eval a new best just because the loop's state
        # is gone; the reattached wandb run's summary carries the score across
        best_value = read_summary("best/value") if best_key == read_summary("best/metric") else None

        def checkpoint(current_step, best=False):
            algo = acc.unwrap_model(model)
            # `copy_to` writes the EMA weights into the live tensors the optimizer still
            # holds, so without this stash every save rewinds training to the lagged average
            backup = None
            if ema is not None:
                backup = copy.deepcopy(algo.state_dict())
                ema.copy_to(algo)
            algo.norm_stats = getattr(self.algo, "norm_stats", None)
            algo.norm_method = getattr(self.algo, "norm_method", None)
            try:
                save_checkpoint(algo, opt, ckpt_dir, current_step,
                                save_intermittent=save_intermittent, best=best,
                                config=self.root_cfg)
            finally:
                if backup is not None:
                    algo.load_state_dict(backup)

        if acc.is_main_process:
            def _on_signal(signum, frame):
                acc.print(f"received signal {signum}: saving checkpoint before exit")
                checkpoint(step)
                raise SystemExit(128 + signum)
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, _on_signal)

        step = start_step
        loss_ema = None
        ema_beta = exp.get("loss_ema_beta", 0.98)
        done = step >= total_steps
        # `step` counts optimizer steps, not micro-batches: max_steps and the schedulers
        # are in units of weight updates
        micro = 0
        window = {}
        data_time = compute_time = aug_time = 0.0
        t_last = time.perf_counter()
        while not done:
            for batch in loader:
                t_batch = time.perf_counter()
                data_time += t_batch - t_last
                if self._batch_augmenter is not None:
                    batch = self._batch_augmenter.apply_batch(batch)
                    # async on CUDA, so this is launch time; the kernels' cost lands in
                    # `compute` at the next sync
                    t_aug = time.perf_counter()
                    aug_time += t_aug - t_batch
                    t_batch = t_aug
                micro += 1
                last_micro = micro % accum_steps == 0
                sync = nullcontext() if last_micro else acc.no_sync(model)
                with sync:
                    # `model.loss` calls `self.predictor(...)` etc rather than
                    # `self(...)`/`self.forward(...)`, so accelerate's forward-patching
                    # mixed precision (installed by `acc.prepare`) never sees this call --
                    # it only wraps the exact `.forward` of the object passed to
                    # `prepare()`, not a method that dispatches to nested submodules.
                    # Applying autocast explicitly here is accelerate's documented
                    # workaround for a custom loop that doesn't invoke `model(...)`
                    # directly (see `Accelerator.autocast`); a `nullcontext` when
                    # `mixed_precision: no`, so this is a no-op for existing fp32 runs.
                    with acc.autocast():
                        out = model.loss(batch)
                    acc.backward(out["loss"] / accum_steps)
                for k, v in out.items():
                    if k == "loss" or k.startswith(("loss/", "var/")):
                        window[k] = window.get(k, 0.0) + v.item() / accum_steps
                if last_micro:
                    if exp.get("grad_clip"):
                        acc.clip_grad_norm_(model.parameters(), float(exp.grad_clip))
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    if ema is not None:
                        ema.update(acc.unwrap_model(model))
                t_last = time.perf_counter()
                compute_time += t_last - t_batch
                if not last_micro:
                    continue
                loss_val = window["loss"]
                loss_ema = loss_val if loss_ema is None else ema_beta * loss_ema + (1 - ema_beta) * loss_val
                if step % exp.log_every == 0:
                    acc.print(f"step {step}  loss {loss_val:.4f}  ema {loss_ema:.4f}"
                              f"  data {data_time:.3f}s  compute {compute_time:.3f}s"
                              + (f"  aug {aug_time:.3f}s" if self._batch_augmenter is not None else ""))
                    if acc.is_main_process:
                        stream_metrics = {
                            f"train/{k}": v for k, v in window.items()
                            if k.startswith(("loss/", "var/"))
                        }
                        log_metrics(
                            {"train/loss": loss_val, "train/loss_ema": loss_ema,
                             "train/lr": sched.get_last_lr()[0],
                             "train/data_time": data_time, "train/compute_time": compute_time,
                             "train/aug_time": aug_time, **stream_metrics},
                            step=step,
                        )
                window = {}
                data_time = compute_time = aug_time = 0.0
                step += 1
                if acc.is_main_process and save_every and step % save_every == 0:
                    checkpoint(step)
                if acc.is_main_process and eval_every and step % eval_every == 0:
                    acc.print(f"step {step}  eval starting")
                    # an eval that dies must not take the rest of the run with it, nor the
                    # arms after it under a multirun
                    try:
                        result = self._eval_during_training(acc.unwrap_model(model), ema, step)
                    except Exception:
                        import traceback
                        acc.print(f"step {step}  EVAL FAILED -- training continues:")
                        acc.print(traceback.format_exc())
                        log_metrics({"eval/failed": 1.0}, step=step)
                        result = None
                    if result is None:
                        pass
                    elif best_key and best_key not in result.metrics:
                        acc.print(f"best_metric '{best_key}' is not reported by this eval "
                                  f"(it has {', '.join(sorted(result.metrics))}); "
                                  f"no best checkpoint will be saved")
                        best_key = None
                    value = (result.metrics.get(best_key)
                             if result is not None and best_key else None)
                    if value is not None and (best_value is None
                                              or _improved(value, best_value, best_mode)):
                        best_value = value
                        acc.print(f"step {step}  new best {best_key} {value:.5f}")
                        checkpoint(step, best=True)
                        log_metrics({f"best/{best_key}": value, "best/step": step}, step=step)
                        update_summary({"best/metric": best_key, "best/value": value,
                                        "best/step": step})
                if step >= total_steps:
                    done = True
                    break
                t_last = time.perf_counter()

        if acc.is_main_process:
            self.algo = acc.unwrap_model(model)
            if ema is not None:
                ema.copy_to(self.algo)
            self._trained = True
            save_checkpoint(self.algo, opt, ckpt_dir, step,
                            save_intermittent=save_intermittent, config=self.root_cfg,
                            final=True)

    def cache(self):
        """Build the encoding cache and stop, without training anything.

            python main.py run=<any arm> +name=build_cache experiment.tasks=[cache]

        A cache is identified by its contents, so any arm whose frozen-encoder output is
        identical resolves to the same directory: any one of a sweep builds it for the rest.
        """
        acc = Accelerator(mixed_precision=self.cfg.mixed_precision)
        self._build_algo()
        if not self.root_cfg.get("cache_encodings"):
            raise ValueError(
                "the `cache` task builds the encoding cache, but `cache_encodings` is not "
                "set on this config, so there is nothing to build."
            )
        self._build_training_dataset(acc, getattr(self.algo, "norm_stats", None))

    def _build_training_dataset(self, acc, reuse_stats):
        """The training dataset, plus the encoding cache when one is configured."""
        cache_cfg = self.root_cfg.get("cache_encodings")
        cache_streams = {}
        if cache_cfg:
            cache_streams = getattr(self.algo, "cacheable_streams", dict)()
            if not cache_streams:
                raise ValueError(
                    "cache_encodings is set but the algorithm has no cacheable stream "
                    "(an encoder chain rooted in a frozen encoder over one visual field)"
                )
        cache_fields = {i["field"] for i in cache_streams.values()}
        pixel_needed = getattr(self.algo, "raw_pixel_fields", set)() if cache_cfg else set()
        # where `dataset.augment` runs: `workers` wraps the source (each DataLoader worker
        # augments its items on the CPU), `device` applies the same Augmenter per sample
        # to every micro-batch after it lands on the accelerator (see the train loop)
        augment_on = self.cfg.get("augment_on")
        if augment_on not in ("device", "workers"):
            raise ValueError(
                f"experiment.augment_on must be 'device' or 'workers', got {augment_on!r}"
            )
        dataset_kwargs = {
            "augment": augment_on == "workers",
            "subsample": self.root_cfg.get("subsample_streams"),
            "cache_fields": sorted(cache_fields) or None,
            "cache_draws": int(cache_cfg.get("augmentation_draws", 1)) if cache_cfg else 1,
            "keep_pixels": sorted(cache_fields & pixel_needed) or None,
        }
        # rank 0 builds first so the download and norm-stats pass happen once; the other
        # ranks block in the broadcast below, which doubles as the barrier
        if acc.is_main_process:
            self.dataset = build_dataset(self.root_cfg.dataset,
                                         norm_stats_override=reuse_stats, split="train",
                                         **dataset_kwargs)
        if acc.num_processes > 1:
            payload = [getattr(self.dataset, "stats", None)] if acc.is_main_process else [None]
            broadcast_object_list(payload, from_process=0)
            if not acc.is_main_process:
                self.dataset = build_dataset(self.root_cfg.dataset,
                                             norm_stats_override=payload[0], split="train",
                                             **dataset_kwargs)
        self._check_modalities()
        if hasattr(self.dataset, "stats"):
            self.algo.norm_stats = self.dataset.stats
        if hasattr(self.dataset, "method"):
            self.algo.norm_method = self.dataset.method

        self._batch_augmenter = None
        aug_cfg = self.root_cfg.dataset.get("augment")
        if augment_on == "device" and aug_cfg and aug_cfg.get("streams"):
            from ...datasets import live_fields
            from ...utils.augment import Augmenter, live_augment_streams

            source = self.dataset
            while hasattr(source, "dataset"):
                source = source.dataset
            streams = live_augment_streams(aug_cfg["streams"],
                                           live_fields(source) - cache_fields)
            if streams:
                seed = aug_cfg.get("seed")
                # ranks must not replay one another's draws
                seed = None if seed is None else int(seed) + acc.process_index
                self._batch_augmenter = Augmenter(streams, seed=seed)
                acc.print(f"augmentation on {acc.device.type} per micro-batch: "
                          f"{ {p: sorted(o) for p, o in streams.items()} }")

        if cache_cfg:
            # must run before acc.prepare / batch calibration, so `batch_size: auto` probes
            # the cached-mode step and lands on the batch the run will train with
            self._setup_encoding_cache(acc, cache_cfg, cache_streams)

    def _setup_encoding_cache(self, acc, cache_cfg, streams_info):
        """Load or precompute the frozen-encoder cache and attach it to the algorithm.

        Multi-rank builds split the episodes: each rank writes a shard directory, rank 0
        concatenates them in rank order (shards are contiguous slices of the key-sorted
        arrays), and every rank then loads the whole assembled cache.
        """
        import tempfile
        from pathlib import Path

        from ...utils.enc_cache import (EncodingCache, assemble_shards,
                                        build_encoding_cache, cache_fingerprint)

        source = self.dataset
        while hasattr(source, "dataset"):
            source = source.dataset

        draws = int(cache_cfg.get("augmentation_draws", 1))
        store_on = cache_cfg.get("store_on", "auto")
        fingerprint = cache_fingerprint(
            streams_info,
            self.root_cfg.algorithm.get("encoders"),
            self.root_cfg.dataset.get("augment"),
            source._anchor_stride,
            draws,
            {
                "repo_id": self.root_cfg.dataset.get("repo_id"),
                "revision": self.root_cfg.dataset.get("revision"),
                "episodes": source._episodes,
                "image_size": self.root_cfg.dataset.get("image_size"),
            },
        )
        base = cache_cfg.get("disk_path") or Path(tempfile.gettempdir()) / "lewam-enc-cache"
        directory = EncodingCache.cache_dir(base, fingerprint)

        if (directory / "manifest.json").exists():
            acc.print(f"encoding cache: loading {directory} (store_on={store_on})")
            cache = EncodingCache.load(directory, fingerprint, store_on)
        else:
            acc.print(f"encoding cache: building {directory} "
                      f"(K={draws}, {len(streams_info)} streams, "
                      f"{acc.num_processes} rank(s))")
            world = acc.num_processes
            build_dir = directory if world == 1 else directory / f"shard{acc.process_index}"
            cache = build_encoding_cache(
                source, self.algo, streams_info,
                self.root_cfg.dataset.get("augment"),
                draws=draws, store_on=store_on, directory=build_dir,
                fingerprint=fingerprint, device=acc.device,
                encode_batch=cache_cfg.get("encode_batch", "auto"),
                amp_dtype=_AMP_DTYPE[acc.mixed_precision],
                decode_workers=int(cache_cfg.get("decode_workers", 8)),
                rank=acc.process_index, world=world, log=acc.print,
            )
            cache.save(build_dir)
            if world > 1:
                acc.wait_for_everyone()
                if acc.is_main_process:
                    assemble_shards(directory, world)
                acc.wait_for_everyone()
                cache = EncodingCache.load(directory, fingerprint, store_on)
        self.algo.attach_encoding_cache(cache)

    @staticmethod
    def _fit_micro_batch(safe_batch, per_rank_target):
        """Feed the per-rank target in the fewest accumulation windows the safe batch
        allows, then shrink the micro-batch so the windows land on the target instead
        of overshooting it: safe 120, target 256 -> 3 windows of 86 (= 258), not
        3 windows of the full 120 (= 360)."""
        if safe_batch >= per_rank_target:
            return per_rank_target
        windows = math.ceil(per_rank_target / safe_batch)
        return math.ceil(per_rank_target / windows)

    @staticmethod
    def _accumulation_steps(acc, batch_size, effective_batch):
        """`experiment.effective_batch: N`: how many micro-batches to accumulate before
        stepping the optimizer, so one update sees >= N samples regardless of what fits in
        VRAM (`batch_size: auto`) or how many ranks are running. Unset means one update per
        micro-batch.
        """
        per_micro = batch_size * acc.num_processes
        if not effective_batch:
            return 1
        steps = max(1, math.ceil(int(effective_batch) / per_micro))
        acc.print(
            f"effective batch {per_micro * steps} = {batch_size} x {steps} accum"
            f" x {acc.num_processes} rank(s), target {effective_batch}"
        )
        if steps == 1 and per_micro > int(effective_batch):
            acc.print(
                f"warning: batch_size {batch_size} already exceeds effective_batch "
                f"{effective_batch}; no accumulation applied"
            )
        return steps

    def _calibrate_batch_size(self, acc, model, opt, exp):
        """`experiment.batch_size: auto`: probe the largest per-device batch size by
        running full train steps on one repeated dataset sample, binary-searching against
        `vram_target_fraction` (default 0.85) of device memory. Model and optimizer state
        are snapshotted to CPU and restored afterwards, so the probe steps never leak
        into training; with DDP every rank probes and the minimum wins.
        """
        import torch
        from torch.utils.data import default_collate

        from ...utils.auto_batch import find_max_batch_size

        def to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.detach().to("cpu", copy=True)
            if isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_cpu(v) for v in obj]
            return obj

        unwrapped = acc.unwrap_model(model)
        model_state = to_cpu(unwrapped.state_dict())
        opt_state = to_cpu(opt.state_dict())
        sample = self.dataset[0]

        def step_fn(batch_size):
            batch = default_collate([sample] * batch_size)
            batch = {
                k: v.to(acc.device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            if self._batch_augmenter is not None:
                batch = self._batch_augmenter.apply_batch(batch)
            # must match the real step's precision (see the training loop's own
            # `acc.autocast()`), or this probes fp32's larger footprint and picks a
            # batch smaller than what mixed precision can actually fit
            with acc.autocast():
                out = model.loss(batch)
            acc.backward(out["loss"])
            opt.step()
            opt.zero_grad()

        acc.print("calibrating batch size...")
        effective = exp.get("effective_batch", None)
        batch_size = find_max_batch_size(
            step_fn,
            target_fraction=exp.get("vram_target_fraction", 0.85),
            device=acc.device,
            log=acc.print,
            target_cap=math.ceil(int(effective) / acc.num_processes) if effective else None,
            fit_threshold=exp.get("vram_fit_threshold", 0.15),
            min_fit_points=exp.get("vram_min_fit_points", 4),
        )
        if acc.num_processes > 1:
            gathered = acc.gather(torch.tensor([batch_size], device=acc.device))
            batch_size = int(gathered.min().item())

        unwrapped.load_state_dict(model_state)
        opt.load_state_dict(opt_state)
        opt.zero_grad()
        batch_size = min(batch_size, len(self.dataset))
        acc.print(f"auto batch size: {batch_size}")
        return batch_size

    def _eval_during_training(self, model, ema, step):
        """Run the configured eval on the current weights mid-training and log to wandb, and
        return the EvalResult so the caller can score it for best-checkpoint selection. EMA
        weights are what get evaluated (same as validation/checkpoint); training weights
        are stashed and restored so the run continues unperturbed.
        """
        from ..eval import build_eval

        model.norm_stats = getattr(self.algo, "norm_stats", None)
        model.norm_method = getattr(self.algo, "norm_method", None)
        backup = None
        if ema is not None:
            backup = copy.deepcopy(model.state_dict())
            ema.copy_to(model)
        was_training = model.training
        try:
            result = build_eval(self.root_cfg.eval).run(model)
            log_eval(result, step=step)
            return result
        finally:
            if backup is not None:
                model.load_state_dict(backup)
            model.train(was_training)
