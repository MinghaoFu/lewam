"""Diffusion Policy (DP-C) as a LeWAM baseline: the model, its normalizers and its checkpoint files, shared
by the trainer (scripts/train_dp.py) and the eval adapter (DPTPolicy in gip.py) so both build the same policy.

The network (one ResNet18 with GroupNorm and spatial softmax per camera, a conditional 1-D UNet over the
action chunk), the DDPM noise schedule, the loss, the crop augmentation and the normalizer classes are
diffusion_policy's own (Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", MIT
license), imported from the diffusion_policy package unchanged, with the settings of its published image
experiments. Ours are the camera list (the h5 image columns a LeWAM model trains on, named as in
lewam.views), the action normalizer rule, and the checkpoint layout: dp_config.json, dp_best.pt (lowest
validation loss), dp_latest.pt (the last epoch's weights), dp_full.pt (full training state for resume).
"""
import json
import types
from pathlib import Path

import numpy as np
import torch

from lewam import views

PUBLISHED_IMAGE_SETTINGS = dict(
    horizon=16, n_action_steps=8, n_obs_steps=2, num_inference_steps=100, obs_as_global_cond=True,
    diffusion_step_embed_dim=128, down_dims=(512, 1024, 2048), kernel_size=5, n_groups=8,
    cond_predict_scale=True, obs_encoder_group_norm=True, eval_fixed_crop=True)
NOISE_SCHEDULE = dict(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                      beta_schedule="squaredcos_cap_v2", variance_type="fixed_small", clip_sample=True,
                      prediction_type="epsilon")
CROP_RATIO = 0.9          # 76 of 84 and 216 of 240 in the published runs; 202 of 224 here


def import_diffusion_policy():
    """Import the policy class; robomimic 0.3 moved CropRandomizer out of base_nets, where the class looks."""
    import robomimic.models.base_nets as base_nets
    if not hasattr(base_nets, "CropRandomizer"):
        import robomimic.models.obs_core as obs_core
        base_nets.CropRandomizer = obs_core.CropRandomizer
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
    return DiffusionUnetHybridImagePolicy


def camera_keys(columns):
    """diffusion_policy observation key per h5 image column: agentview_image, robot0_eye_in_hand_image, ..."""
    return [f"{views.camera_of(column)}_image" for column in columns]


def shape_meta(columns, image_size, action_dim):
    obs = {key: dict(shape=[3, image_size, image_size], type="rgb") for key in camera_keys(columns)}
    return dict(obs=obs, action=dict(shape=[action_dim]))


def make_config(columns, image_size, action_dim, crop=None, **overrides):
    """The checkpoint config: cameras, sizes and every policy setting, published values unless overridden."""
    config = dict(PUBLISHED_IMAGE_SETTINGS)
    config.update(overrides)
    config.update(views=list(columns), image_size=int(image_size), action_dim=int(action_dim),
                  crop=int(crop if crop is not None else round(CROP_RATIO * image_size)),
                  action_normalizer="identity")
    return config


def build_policy(config):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    policy_class = import_diffusion_policy()
    settings = {key: config[key] for key in PUBLISHED_IMAGE_SETTINGS}
    settings["down_dims"] = tuple(settings["down_dims"])
    return policy_class(shape_meta=shape_meta(config["views"], config["image_size"], config["action_dim"]),
                        noise_scheduler=DDPMScheduler(**NOISE_SCHEDULE),
                        crop_shape=(config["crop"], config["crop"]), **settings)


def fit_normalizer(policy, actions, config):
    """diffusion_policy's normalizers from the training data. Images: [0, 1] to [-1, 1]. Actions: the
    identity when the recorded actions lie within [-1, 1], as diffusion_policy assumes of robomimic's delta
    actions (its DDPM clips samples to [-1, 1]); otherwise its min-max range normalizer, which it uses for
    absolute actions. The rule used is recorded in the config."""
    from diffusion_policy.common.normalize_util import (array_to_stats, get_identity_normalizer_from_stat,
                                                        get_image_range_normalizer,
                                                        get_range_normalizer_from_stat)
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    actions = np.asarray(actions, np.float32)
    stat = array_to_stats(actions)
    # the cache stores z-scored half-precision actions, so a recorded 1.0 comes back as 1.0 plus rounding
    within_unit = float(np.abs(actions).max()) <= 1.0 + 1e-2
    normalizer = LinearNormalizer()
    normalizer["action"] = (get_identity_normalizer_from_stat(stat) if within_unit
                            else get_range_normalizer_from_stat(stat))
    for key in camera_keys(config["views"]):
        normalizer[key] = get_image_range_normalizer()
    policy.set_normalizer(normalizer)
    config["action_normalizer"] = "identity" if within_unit else "range"
    return config["action_normalizer"]


def save_config(run_dir, config):
    Path(run_dir, "dp_config.json").write_text(json.dumps(config, indent=1))


def load_trained(run_dir, which="best"):
    """The trained policy (weights and normalizer) and its config from a run directory."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "dp_config.json").read_text())
    policy = build_policy(config)
    state = torch.load(run_dir / f"dp_{which}.pt", map_location="cpu")
    policy.load_state_dict(state["model"] if "model" in state else state)
    return policy.eval(), config


def adapter_config(config):
    """What the eval adapter reads: the observation keys with their info-dict source (`pixels` for the first
    camera, `pixels.<camera>` for the others), the observation history and executed-chunk lengths."""
    keys = camera_keys(config["views"])
    return types.SimpleNamespace(
        task=types.SimpleNamespace(shape_meta=shape_meta(config["views"], config["image_size"], config["action_dim"])),
        n_obs_steps=int(config["n_obs_steps"]), n_action_steps=int(config["n_action_steps"]),
        camera_info_keys=dict(zip(keys, views.info_keys(config["views"]))))
