"""lerobot's own policies (ACT, diffusion, ...) as LeWAM algorithms.

Normalization is this repo's, not the policy's: a lerobot policy carries no Normalize
layers, since they live in the processor pipelines `make_*_pre_post_processors` builds,
which this wrapper does not use. The dataset normalizes these arms and the rollout driver
unnormalizes from the same stats, so turning it off in the data layer leaves nothing
normalizing at all -- which diffusion cannot survive, its sampler clipping every denoising
step to [-1, 1] while the training loss goes on looking reasonable.
"""

from pathlib import Path
import torch
from omegaconf import OmegaConf

from ..datasets.lerobot import resolve_slice
from ..utils.spec import action_entry, parse_index, raw_index, spec_fields, split_index
from .base import BaseAlgorithm

# lerobot's VLA policies (smolvla, pi05) consume PRE-TOKENIZED language: their models
# read `observation.language.tokens`/`.attention_mask`, produced by the processor
# pipelines this wrapper bypasses. The wrapper tokenizes `batch["task"]` itself,
# replicating each policy's own processor step.
LANGUAGE_POLICIES = ("smolvla", "pi05")


def pi05_prompts(tasks, state, max_state_dim):
    """openpi's PaliGemma prompt: the task text plus the state, zero-padded to
    `max_state_dim` and discretized into 256 bins. Mirrors
    Pi05PrepareStateTokenizerProcessorStep (lerobot processor_pi05.py), which runs on
    the NORMALIZED state -- this repo's normalization stands in for lerobot's, so the
    state is clamped to the [-1, 1] range the bin edges assume."""
    import numpy as np

    state = torch.nn.functional.pad(state.float(), (0, max_state_dim - state.shape[-1]))
    bins = np.digitize(state.clamp(-1, 1).cpu().numpy(),
                       bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    prompts = []
    for task, row in zip(tasks, bins):
        cleaned = str(task).strip().replace("_", " ").replace("\n", " ")
        prompts.append(f"Task: {cleaned}, State: {' '.join(map(str, row))};\nAction: ")
    return prompts


class LeRobotPolicy(BaseAlgorithm):
    """Adapts a lerobot `PreTrainedPolicy` to the `loss()` / `predict()` contract.

    Two translations. The batch: this repo serves every field as a window `(B, T, ...)`
    while a lerobot policy expects a time axis only on the keys it asked for a window of,
    so windows of length one are squeezed. And the attributes the rollout driver reads off
    a model (chunk length, conditioning offsets, which stream is the action), derived from
    the same spec every other algorithm here declares.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conditioning = cfg.conditioning
        self.predict_spec = cfg.predict

        action = action_entry(cfg.predict)
        if action is None:
            raise ValueError(
                "a lerobot policy predicts actions, but no entry of `predict:` carries "
                "them. Mark one with `role: action`."
            )
        self.action_stream, self.action_field = action
        action_spec = cfg.predict[self.action_stream]
        self.chunk_len = len(parse_index(action_spec["index"]))
        self.action_dim = int(action_spec["dim"])
        self.execute_len = int(cfg.get("execute_len", self.chunk_len))
        if not 1 <= self.execute_len <= self.chunk_len:
            raise ValueError(
                f"execute_len={self.execute_len} is outside the {self.chunk_len}-step chunk"
            )

        offsets = {}
        for name, spec in cfg.conditioning.items():
            relative, has_last = split_index(raw_index(spec))
            if has_last:
                raise ValueError(
                    f"conditioning '{name}' asks for a goal ('last') frame, which a "
                    f"lerobot policy has no input for"
                )
            for field in spec_fields(name, spec):
                offsets.setdefault(field, set()).update(relative)
        self.field_offsets = {f: sorted(steps) for f, steps in offsets.items()}
        self.obs_len = max(
            (1 - min(steps) for steps in self.field_offsets.values()), default=1
        )
        self._last_fields = set()

        self.norm_stats = None
        self.norm_method = None

        self.policy = self._build_policy(cfg)
        self._check_features()
        self._language = self._build_language()

    def _build_policy(self, cfg):
        from lerobot.configs.types import FeatureType
        from lerobot.datasets import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_policy_config
        from lerobot.utils.feature_utils import dataset_to_policy_features

        source = cfg.dataset
        # a stored config bakes a machine-local `root`; chasing it on another machine
        # makes lerobot mkdir the missing tree and die. Fall back to root=None
        # (HF_LEROBOT_HOME / hub) -- only metadata is read here, to size input layers
        root = source.get("root")
        if root is not None and not Path(root).exists():
            print(f"dataset.root '{root}' not present here; reading '{source.repo_id}' "
                  f"metadata from the hub cache instead")
            root = None
        meta = LeRobotDatasetMetadata(
            source.repo_id, root=root, revision=source.get("revision")
        )
        overrides = OmegaConf.to_container(cfg.policy, resolve=True)
        policy_type = overrides.pop("type")
        policy_cfg = make_policy_config(policy_type, **overrides)

        # the spec must decide what the policy reads: left to the dataset's column list,
        # lerobot takes every non-action feature
        features = dataset_to_policy_features(meta.features)
        wanted = set(self.field_offsets)
        unknown = wanted - set(features)
        if unknown:
            raise ValueError(
                f"`conditioning:` reads {sorted(unknown)}, which '{source.repo_id}' does "
                f"not record. available: {sorted(features)}"
            )
        policy_cfg.input_features = {
            key: feature for key, feature in features.items() if key in wanted
        }
        from dataclasses import replace

        size = source.get("image_size")
        if size:
            # the dataset wrapper resizes every visual stream to one square size, but the
            # metadata still records native shapes, which lerobot's Diffusion refuses
            height, width = (int(v) for v in size)
            policy_cfg.input_features = {
                key: (replace(feature, shape=(feature.shape[0], height, width))
                      if feature.type is FeatureType.VISUAL else feature)
                for key, feature in policy_cfg.input_features.items()
            }
        # same story for `slices:`: the wrapper hands over the kept dimensions while the
        # metadata reports the whole column, so the policy is built for the wrong width
        for key, patterns in dict(source.get("slices") or {}).items():
            feature = policy_cfg.input_features.get(key)
            if feature is None or feature.type is FeatureType.VISUAL:
                continue
            kept = len(resolve_slice(patterns, meta.features[key]["names"], key))
            policy_cfg.input_features[key] = replace(
                feature, shape=(*feature.shape[:-1], kept)
            )
        policy_cfg.output_features = {
            key: feature
            for key, feature in features.items()
            if feature.type is FeatureType.ACTION
        }
        return make_policy(policy_cfg, ds_meta=meta)

    def _build_language(self):
        """The tokenizer a VLA policy's processor pipeline would have applied, or None.

        Built eagerly so a missing tokenizer (no HF cache on an offline worker) fails at
        construction rather than at the first training step.
        """
        cfg = self.policy.config
        ptype = getattr(cfg, "type", None)
        if ptype == "smolvla":
            from transformers import AutoProcessor

            tok = AutoProcessor.from_pretrained(cfg.vlm_model_name).tokenizer
            tok.padding_side = "right"
            return {"tokenizer": tok, "padding": cfg.pad_language_to,
                    "max_length": int(cfg.tokenizer_max_length), "state_in_prompt": False}
        if ptype == "pi05":
            from transformers import AutoTokenizer

            # the tokenizer pi05's processor pins, independent of any policy weights
            tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
            tok.padding_side = "right"
            return {"tokenizer": tok, "padding": "max_length",
                    "max_length": int(cfg.tokenizer_max_length), "state_in_prompt": True}
        assert ptype not in LANGUAGE_POLICIES
        return None

    def _tokenize_language(self, out, batch):
        from lerobot.utils.constants import (OBS_LANGUAGE_ATTENTION_MASK,
                                             OBS_LANGUAGE_TOKENS, OBS_STATE)

        tasks = batch.get("task")
        if tasks is None:
            raise ValueError(
                f"policy '{self.policy.config.type}' is language-conditioned, but the "
                f"batch carries no 'task' strings. The lerobot dataset wrapper emits "
                f"them; a rollout driver has to supply the instruction itself."
            )
        if isinstance(tasks, str):
            tasks = [tasks]
        lang = self._language
        if lang["state_in_prompt"]:
            state = out[OBS_STATE]
            state = state[:, -1] if state.ndim > 2 else state
            prompts = pi05_prompts(tasks, state, self.policy.config.max_state_dim)
        else:
            # smolvla's pipeline guarantees a trailing newline before tokenizing
            prompts = [t if str(t).endswith("\n") else str(t) + "\n" for t in tasks]
        enc = lang["tokenizer"](prompts, padding=lang["padding"], truncation=True,
                                max_length=lang["max_length"], return_tensors="pt")
        device = next(t.device for t in out.values() if torch.is_tensor(t))
        out[OBS_LANGUAGE_TOKENS] = enc["input_ids"].to(device)
        # bool, not the tokenizer's long: smolvla's eager attention torch.where()s on it
        out[OBS_LANGUAGE_ATTENTION_MASK] = enc["attention_mask"].to(device, torch.bool)

    def _check_features(self):
        """Fail at build time when the spec and the policy disagree about the inputs.

        Uncaught, the same mismatch surfaces as a shape error several frames deep inside
        lerobot at the first training step.
        """
        provided = set(self.field_offsets)
        required = set(getattr(self.policy.config, "input_features", {}) or {})
        missing = required - provided
        if missing:
            raise ValueError(
                f"policy '{self.policy.config.type}' reads {sorted(missing)}, which no "
                f"`conditioning:` entry provides. Add them, or drop them from the "
                f"policy's input_features."
            )
        extra = provided - required
        if extra:
            raise ValueError(
                f"`conditioning:` provides {sorted(extra)}, which policy "
                f"'{self.policy.config.type}' does not read. A window this policy ignores "
                f"is decode cost for nothing -- remove it from the spec."
            )
        n_obs = getattr(self.policy.config, "n_obs_steps", None)
        if n_obs is not None:
            wrong = {
                field: len(steps)
                for field, steps in self.field_offsets.items()
                if len(steps) != int(n_obs)
            }
            if wrong:
                raise ValueError(
                    f"policy '{self.policy.config.type}' takes n_obs_steps={n_obs}, but "
                    f"these fields are windowed differently: {wrong}. The spec decides "
                    f"what the dataset loads, so it is the one that has to agree."
                )

    def _to_policy_batch(self, batch, with_action=True):
        """This repo's windowed batch -> the batch a lerobot policy expects.

        Images arrive as `(B, T, C, H, W)`, uint8 from the dataset default and from the
        rollout driver alike; lerobot policies want float in [0, 1].

        Whether a length-1 window keeps its time axis is the policy's own declaration:
        `observation_delta_indices` is None for one taking a bare observation (ACT), a list
        of offsets for one that always windows (Diffusion asserts the axis at one step).
        """
        windowed = getattr(self.policy.config, "observation_delta_indices", None) is not None
        out = {}
        for field in self.field_offsets:
            tensor = batch[field]
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            elif not tensor.dtype.is_floating_point:
                tensor = tensor.float()
            if tensor.shape[1] == 1 and not windowed:
                tensor = tensor.squeeze(1)
            out[field] = tensor
        if with_action and self.action_field in batch:
            actions = batch[self.action_field].float()
            out[self.action_field] = actions
            # ACT's VAE encoder reads this unconditionally and KeyErrors without it.
            # All-false is correct: `drop_boundary` means no chunk is ever padded.
            pad_key = f"{self.action_field}_is_pad"
            out[pad_key] = batch.get(
                pad_key,
                torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device),
            )
        if "task" in batch:
            out["task"] = batch["task"]
        if self._language is not None:
            self._tokenize_language(out, batch)
        return out

    def optim_params(self):
        """The policy's own parameter groups (ACT gives its vision backbone a separate
        learning rate)."""
        return self.policy.get_optim_params()

    def loss(self, batch):
        loss, output = self.policy.forward(self._to_policy_batch(batch))
        metrics = {"loss": loss}
        for key, value in (output or {}).items():
            if torch.is_tensor(value) and value.ndim == 0:
                metrics[f"loss/{key}"] = value
        return metrics

    def predict(self, obs):
        chunk = self.policy.predict_action_chunk(self._to_policy_batch(obs, with_action=False))
        return {self.action_stream: chunk[:, : self.chunk_len]}

    def reset(self):
        self.policy.reset()

    def summary(self):
        return {
            **super().summary(),
            "policy/type": self.policy.config.type,
            "policy/chunk_len": self.chunk_len,
            "obs_len": self.obs_len,
        }
