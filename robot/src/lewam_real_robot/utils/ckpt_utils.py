from pathlib import Path


def is_run_id(run_id):
    """True if a string looks like a wandb run id (8 alphanumeric chars)."""
    return len(run_id) == 8 and run_id.isalnum()


def split_alias(spec):
    """Split a trailing artifact alias off a checkpoint reference:
    `abc12345:best` -> ('abc12345', 'best'). Anything without one -- a bare run id, an
    entity/project/id path, a local .pt path -- comes back unchanged with alias None.
    """
    spec = str(spec)
    head, sep, alias = spec.rpartition(":")
    return (head, alias) if sep and head and alias else (spec, None)


def _version_to_int(artifact):
    """Convert an artifact version of the form vX to the int X."""
    return int(artifact.version[1:])


def _reorder_like(stored, template, path=""):
    """`stored`'s VALUES in `template`'s KEY ORDER, recursively.

    wandb hands configs back with alphabetically sorted keys, but the model derives its
    token layout and rope band layout from dict declaration order (GenericViTTrunk:
    "reordering `conditioning:` moves every token position") -- a checkpoint still
    load_state_dicts strictly into the scrambled build, because every weight is
    name-keyed, and the model silently computes a systematically wrong function. This
    restored the exact offline gap (3.5 -> 6.7 cm FK on u6vblv66) and was what every
    robot serve of a LeWAM arm ran. Values always come from `stored` (the config as
    trained is the source of truth); only the ORDER comes from the template. Stored-only
    keys (removed from the repo since training) keep their stored position at the end,
    with a warning, because their intended position is unrecoverable.
    """
    if not isinstance(stored, dict) or not isinstance(template, dict):
        return stored
    out = {}
    for key in template:
        if key in stored:
            out[key] = _reorder_like(stored[key], template[key], f"{path}{key}.")
    extras = [k for k in stored if k not in template]
    for key in extras:
        print(f"fetch_run_config: '{path}{key}' is not in the local arm config; its "
              f"declaration position cannot be restored (appending last)")
        out[key] = stored[key]
    return out


def config_from_checkpoint(ckpt_path):
    """The run config a checkpoint embedded at save time (`config_yaml`, an ORDERED yaml
    string), as OmegaConf -- or None for checkpoints from before configs were embedded.

    This is the preferred config source for a rebuild: it never round-trips through
    wandb's alphabetized storage, so declaration order (which the model's token and rope
    layouts follow) survives exactly as trained.
    """
    import torch
    from omegaconf import OmegaConf

    yaml_str = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("config_yaml")
    return OmegaConf.create(yaml_str) if yaml_str else None


def fetch_run_config(run_path):
    """The full hydra config a wandb run stored at init, as OmegaConf. Post-hoc eval
    restores the algorithm section from here, so the model rebuild cannot drift from
    what actually trained even if the local yamls changed since.

    Prefer `config_from_checkpoint` when a checkpoint file is at hand: wandb SORTS keys
    alphabetically in storage, and key order is semantic here (token and rope layouts
    follow declaration order). For checkpoints without an embedded config, the stored
    `algorithm._name` / `dataset._name` recover which local config files define the
    canonical order, and `_reorder_like` restores it while keeping every stored value.
    """
    from pathlib import Path

    import wandb
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    stored = dict(wandb.Api().run(run_path).config)
    overrides = []
    for group in ("algorithm", "dataset", "eval", "experiment"):
        name = (stored.get(group) or {}).get("_name")
        if not name:
            continue
        overrides.append(f"{group}={name}")
    if overrides:
        from hydra.core.global_hydra import GlobalHydra

        # inside a running hydra app (main.py) the global instance already exists and
        # re-initializing raises; its search path is this same configs dir, so compose
        # against it directly
        try:
            if GlobalHydra.instance().is_initialized():
                template = OmegaConf.to_container(
                    compose(config_name="base", overrides=overrides), resolve=False
                )
            else:
                cdir = Path(__file__).resolve().parents[3] / "configs"
                with initialize_config_dir(config_dir=str(cdir), version_base=None):
                    template = OmegaConf.to_container(
                        compose(config_name="base", overrides=overrides), resolve=False
                    )
        except Exception as err:  # noqa: BLE001 - hydra raises many flavors
            raise ValueError(
                f"{run_path} trained under config(s) that no longer exist locally "
                f"({' '.join(overrides)}): {err}"
            ) from err
        stored = _reorder_like(stored, template)
    else:
        print(
            "fetch_run_config: stored config carries no _name entries, so its "
            "alphabetized key order cannot be restored -- a model whose layout depends "
            "on declaration order (GenericViTPredictor) will rebuild WRONG. Prefer a "
            "local rebuild from configs/run/ for such checkpoints."
        )
    return OmegaConf.create(stored)


def _cached_pt(root):
    """The .pt already sitting in a run's download dir, or None."""
    return next(root.glob("*.pt"), None) if root.is_dir() else None


def download_latest_checkpoint(run_path, download_dir, alias=None):
    """The local .pt path of a wandb run's 'model' artifact, downloading only when
    needed.

    With `alias` (currently only "best") the tagged version is fetched instead of the
    newest one, which is how a run's pre-overfit weights are reached after training has
    moved past them.

    Each run's download lives in its own subdir with a `.artifact_digest` sidecar
    written after a completed download. Resolving which artifact an alias points at is
    a small metadata call; when the resolved digest matches the sidecar, the cached .pt
    is returned without touching the network again -- so a moved alias (`best`/`latest`
    advancing) still re-downloads, and an unchanged one costs nothing. When wandb is
    unreachable entirely, a cached copy (if present) is served with a warning, which
    keeps a robot serving through a network blip. Only the run's own subdir is ever
    cleared (a stale or partial download); use lewam_real_robot.scripts.clean_checkpoints to
    reclaim the space of runs no longer needed.
    """
    import shutil

    import wandb

    root = Path(download_dir) / run_path
    sidecar = root / ".artifact_digest"
    cached = _cached_pt(root)

    try:
        if alias:
            entity, project, run_id = run_path.split("/")[-3:]
            name = f"{entity}/{project}/model-{run_id}:{alias}"
            try:
                wanted = wandb.Api().artifact(name, type="model")
            except Exception as err:  # noqa: BLE001 - wandb raises many flavors
                raise ValueError(
                    f"run {run_path} has no '{alias}' checkpoint ({err}). Runs trained "
                    f"before best-checkpointing, or with best_metric=null, only have the "
                    f"newest one: drop the ':{alias}' suffix."
                ) from err
        else:
            run = wandb.Api().run(run_path)
            wanted = None
            for artifact in run.logged_artifacts():
                if artifact.type != "model" or artifact.state != "COMMITTED":
                    continue
                if wanted is None or _version_to_int(artifact) > _version_to_int(wanted):
                    wanted = artifact
            if wanted is None:
                raise ValueError(f"no model checkpoint artifact found for run {run_path}")
    except Exception as err:  # noqa: BLE001 - network failures arrive in many shapes,
        # including as the ValueErrors above wrapping a connection error; with a
        # completed (sidecar'd) download on disk, keep working on it rather than die on
        # the wandb round trip. In this workflow aliases are never deleted, so "cannot
        # resolve the artifact" with a cache present is overwhelmingly a network problem.
        if cached is not None and sidecar.is_file():
            print(f"could not resolve the wandb artifact ({err}); "
                  f"using the cached checkpoint {cached}")
            return cached
        raise

    if cached is not None and sidecar.is_file() and sidecar.read_text() == wanted.digest:
        print(f"checkpoint cache hit: {cached} ({wanted.name})")
        return cached

    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    wanted.download(root=str(root))
    path = next(root.glob("*.pt"))
    sidecar.write_text(wanted.digest)
    return path


def source_run_name(run_path, checkpoint_path=None):
    """The training run's own name (`selfwam_mot_mse_b601_pnpt`), so a post-hoc eval can be named
    after the model it scores instead of an opaque run id. The checkpoint's baked config is
    tried first (no network); wandb's display name is the fallback, None if neither answers.
    """
    if checkpoint_path:
        try:
            stored = config_from_checkpoint(checkpoint_path)
            name = stored.get("name") if stored else None
            if name:
                return str(name)
        except Exception:  # noqa: BLE001 - an unresolvable or absent name is not fatal
            pass
    try:
        import wandb
        return wandb.Api().run(run_path).name or None
    except Exception:  # noqa: BLE001 - offline workers have no public api
        return None
