"""Checkpoint loaders"""

import json
from pathlib import Path

import torch
from stable_worldmodel.data.utils import get_cache_dir

from lewam.models.lewam import LeWAM, build_model


def get_checkpoint_path(run_dir: Path, prefix: str, which: str) -> Path:
    """Find `<prefix>_<which>.pt` in `run_dir`, falling back to `<prefix>_latest.pt`."""
    path = run_dir / f"{prefix}_{which}.pt"
    if not path.exists():
        path = run_dir / f"{prefix}_latest.pt"
    assert path.exists(), f"no {prefix}_*.pt checkpoint in {run_dir}"
    return path


def load_lewam(run_name: str, which: str = "best") -> tuple[LeWAM, dict]:
    """Load a LeWAM checkpoint and its config from $STABLEWM_HOME/checkpoints/<run_name>."""
    run_dir = Path(get_cache_dir(sub_folder="checkpoints")) / run_name
    cfg = json.loads((run_dir / "lewam_config.json").read_text())
    # the trainer stores frameskip and the raw action dim, not the flattened chunk length
    cfg.setdefault("action_dim", int(cfg["fs"]) * int(cfg["action_raw_dim"]))
    model = build_model(cfg)
    model.load_state_dict(torch.load(get_checkpoint_path(run_dir, "lewam", which), map_location="cpu"), strict=True)
    return model, cfg
