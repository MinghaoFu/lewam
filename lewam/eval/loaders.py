"""Checkpoint loaders"""
import json
import re
import torch
import torch.nn as nn

from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf
from stable_worldmodel.data.utils import get_cache_dir

from lewam.models.lewam import LeWAM, build_model

# transformers 4 -> 5 compatibility
VIT_RENAMES = [
    (r"\.attention\.attention\.query\.", ".attention.q_proj."),
    (r"\.attention\.attention\.key\.", ".attention.k_proj."),
    (r"\.attention\.attention\.value\.", ".attention.v_proj."),
    (r"\.attention\.output\.dense\.", ".attention.o_proj."),
    (r"\.intermediate\.dense\.", ".mlp.fc1."),
    (r"\.output\.dense\.", ".mlp.fc2."),
    (r"^encoder\.encoder\.layer\.", "encoder.layers."),
]


class LeWMEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        cls = self.model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        return self.model.projector(cls)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)


def rename_vit_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    renamed = {}
    for key, value in state_dict.items():
        for old, new in VIT_RENAMES:
            key = re.sub(old, new, key)
        renamed[key] = value
    return renamed


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


def load_lewm(ckpt_dir: str, encoder_only: bool = False) -> tuple[nn.Module, dict]:
    """Load LeWM checkpoint from a folder with `config.json` and a `.pt`.
    See: https://huggingface.co/collections/quentinll/lewm"""
    ckpt_path = Path(ckpt_dir)
    weights = list(ckpt_path.glob("*.pt"))
    assert len(weights) == 1, f"expected one .pt in {ckpt_path}, found {len(weights)}"

    cfg = json.loads((ckpt_path / "config.json").read_text())
    model = instantiate(OmegaConf.create(cfg))
    ckpt_state_dict = torch.load(weights[0], map_location="cpu")
    if ckpt_state_dict.keys() != model.state_dict().keys():
        ckpt_state_dict = rename_vit_keys(ckpt_state_dict)
    model.load_state_dict(ckpt_state_dict)

    if encoder_only:
        model = LeWMEncoder(model)

    return model, cfg
