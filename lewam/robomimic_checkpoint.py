"""Load a published robomimic checkpoint (the 2021 model zoo, robomimic v0.1) with the installed robomimic.

Two things changed since those checkpoints were written. The config: `observation.encoder` was one flat
block (visual_core, visual_core_kwargs, visual_feature_dimension, use_spatial_softmax,
spatial_softmax_kwargs, obs_randomizer_class/kwargs) and is now one section per modality with
core_class / core_kwargs / obs_randomizer_class / obs_randomizer_kwargs; keys added later (algo.transformer,
...) are read unconditionally. The weights: the image encoder's modules were named vis_core and pool_net,
now backbone and pool. `load_rollout_policy` translates both and returns robomimic's rollout policy.
"""
import json

import robomimic
import robomimic.utils.file_utils as file_utils
from robomimic.config import config_factory

RENAMED_MODULES = ((".vis_core.", ".backbone."), (".pool_net.", ".pool."))


def installed_wraps_state_dict():
    """robomimic 0.3.1 serializes a model as {nets, optimizers, lr_schedulers} and reads all three back;
    earlier versions read the bare state dict."""
    version = tuple(int(part) for part in robomimic.__version__.split(".")[:3])
    return version >= (0, 3, 1)


def merge_into(config, saved):
    """Set every leaf of `saved` into `config`, recursing through nested sections."""
    for key, value in saved.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merge_into(config[key], value)
        else:
            config[key] = value


def translate_v01_encoder(encoder):
    """The v0.1 flat encoder block -> the per-modality layout (rgb only; low_dim has no encoder)."""
    if "visual_core" not in encoder:
        return encoder
    return {"rgb": dict(
        core_class="VisualCore",
        core_kwargs=dict(feature_dimension=encoder["visual_feature_dimension"], flatten=True,
                         backbone_class=encoder["visual_core"], backbone_kwargs=encoder["visual_core_kwargs"],
                         pool_class="SpatialSoftmax" if encoder["use_spatial_softmax"] else None,
                         pool_kwargs=encoder["spatial_softmax_kwargs"]),
        obs_randomizer_class=encoder["obs_randomizer_class"],
        obs_randomizer_kwargs=encoder["obs_randomizer_kwargs"])}


def translate_v01_config(saved):
    """The v0.1 observation section in place: the modality `image` became `rgb`, the encoder block
    became per-modality sections."""
    observation = saved["observation"]
    for group in observation.get("modalities", {}).values():
        if isinstance(group, dict) and "image" in group:
            group["rgb"] = group.pop("image")
    observation["encoder"] = translate_v01_encoder(observation["encoder"])
    return saved


def load_rollout_policy(ckpt_path, device):
    """robomimic's RolloutPolicy and the checkpoint dict, for a checkpoint of any robomimic version."""
    ckpt_dict = file_utils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    saved = translate_v01_config(json.loads(ckpt_dict["config"]))
    config = config_factory(saved["algo_name"])
    with config.unlocked():
        merge_into(config, saved)
    ckpt_dict["config"] = config.dump()
    model = ckpt_dict["model"]
    state = model["nets"] if "nets" in model else model
    weights = {}
    for key, value in state.items():
        for old, new in RENAMED_MODULES:
            key = key.replace(old, new)
        weights[key] = value
    if installed_wraps_state_dict():
        ckpt_dict["model"] = dict(nets=weights, optimizers=model.get("optimizers", {}),
                                  lr_schedulers=model.get("lr_schedulers", {}))
    else:
        ckpt_dict["model"] = weights
    return file_utils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=False)
