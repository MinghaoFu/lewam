import os
from pathlib import Path


def _has_wandb():
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def wandb_enabled():
    """The single source of truth for whether wandb is live. Set WANDB_MODE=disabled to
    force stdout/json fallback (used by the smoke test and no-network runs).
    """
    return os.environ.get("WANDB_MODE", "") != "disabled" and _has_wandb()


def active_run_id():
    """The id of the run this process opened, or None when wandb is off. Parked in a
    rollout dir, it is what lets a resumed eval reattach instead of opening a second run.
    """
    if not wandb_enabled():
        return None
    import wandb
    return wandb.run.id if wandb.run is not None else None


def init_wandb(cfg, eval_source=None, source_name=None, resume_id=None):
    """Open the run bracketing all tasks. With `eval_source` (a training run id), this is
    a post-hoc eval: the new run is tagged job_type="eval" and grouped under the source
    run, so results attach to the exact model that produced them. `resume_id` reattaches
    to an eval run an earlier, interrupted session opened.
    """
    wandb_cfg = cfg.get("wandb", {})
    mode = wandb_cfg.get("mode", None)
    if mode:
        os.environ["WANDB_MODE"] = mode
    if not wandb_enabled():
        return
    import wandb
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(cfg, resolve=True)
    if eval_source:
        wandb.init(
            id=resume_id,
            resume="must" if resume_id else None,
            name=None if resume_id
            else (cfg.get("name", None) or f"eval-{source_name or eval_source}"),
            entity=wandb_cfg.get("entity", None),
            project=wandb_cfg.get("project", None) or "lewam-real-robot",
            group=eval_source,
            job_type="eval",
            config={**config, "source_run_id": eval_source},
        )
        return
    resume_id = cfg.get("resume", None)
    wandb.init(
        id=resume_id,
        resume="must" if resume_id else None,
        name=None if resume_id else cfg.get("name", None),
        entity=wandb_cfg.get("entity", None),
        project=wandb_cfg.get("project", None) or "lewam-real-robot",
        config=config,
    )


def log_model_summary(model):
    """Push the algorithm's static stats (model.summary()) into wandb.config."""
    summary = model.summary()
    if not wandb_enabled():
        print(summary)
        return
    import wandb
    wandb.config.update(summary, allow_val_change=True)


def log_metrics(metrics, step=None):
    if not wandb_enabled():
        print(metrics)
        return
    import wandb
    wandb.log(metrics, step=step)


def log_eval(result, step=None):
    if not wandb_enabled():
        print(result.metrics)
        return
    import wandb
    wandb.log({f"eval/{k}": v for k, v in result.metrics.items()}, step=step)
    for name, video in result.videos.items():
        # a str/Path is an encoded file (the real-robot eval's concatenated real-time
        # mp4); an array is sim-eval frames, kept on the old gif path
        if isinstance(video, (str, Path)):
            wandb.log({f"eval/video/{name}": wandb.Video(str(video), format="mp4")}, step=step)
        else:
            wandb.log({f"eval/video/{name}": wandb.Video(video, fps=10, format="gif")}, step=step)
    for name, image in result.images.items():
        image, caption = image if isinstance(image, tuple) else (image, None)
        wandb.log({f"eval/image/{name}": wandb.Image(image, caption=caption)}, step=step)
    if result.episodes:
        import pandas as pd
        wandb.log({"eval/episodes": wandb.Table(dataframe=pd.DataFrame(result.episodes))}, step=step)


def update_summary(values):
    """Write into the *active* run's summary. State parked here survives a `resume=`
    (which reattaches to the same run), unlike anything held in the training loop.
    """
    if not wandb_enabled():
        return
    import wandb
    if wandb.run is not None:
        wandb.run.summary.update(values)


def read_summary(key, default=None):
    """Read back what update_summary wrote, on this run or the one being resumed."""
    if not wandb_enabled():
        return default
    import wandb
    if wandb.run is None:
        return default
    return wandb.run.summary.get(key, default)


def update_run_summary(run_path, metrics):
    """Write metrics into a (possibly finished) run's summary via the public API — how a
    post-hoc eval lands numbers on the training run id after the fact.
    """
    if not wandb_enabled():
        print({run_path: metrics})
        return
    import wandb
    wandb.Api().run(run_path).summary.update(metrics)


def finish_wandb(exit_code=None):
    """Close the active run. `exit_code` marks it failed on wandb; passing it matters
    under multiruns, where a run left open swallows the next arm's wandb.init."""
    if not wandb_enabled():
        return
    import wandb
    wandb.finish(exit_code=exit_code)
