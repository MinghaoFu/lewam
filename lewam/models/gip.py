"""
GIP (Generalized Intention Prediction) evaluation infrastructure.

Handles everything the policy needs at eval time:
  - loading a checkpoint
  - the shared env/dataset/process/episode-sampling helpers,
  - the two policies (reactive BC, and the Actionable adapter for guided
    planning)
  - a single `build_policy` factory dispatched by `gip_eval.mode`.

The success-rate mode is the one config knob, mirroring how `action_pred.enabled`
gates training: gip_eval.mode = bc | guided | planning
  - bc:         run the intention head directly as the policy (action head only)
  - guided:     intention-guided planning -- the action head warm-starts CEM,
                the world-model state head rolls candidates to the goal
  - planning:   plain world-model CEM planning (state head only; LeWM baseline)
"""

import json
import types
from collections import deque
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from hydra.utils import instantiate as hydra_instantiate
from stable_worldmodel.policy import BasePolicy
from stable_worldmodel.data.utils import get_cache_dir

from lewam import views
from lewam.models.module import MLP

#####################################
#  Env / dataset / process helpers  #
#####################################
def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def episode_col(dataset):
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    _nm = str(dataset_name)[:-6] if str(dataset_name).endswith(".lance") else str(dataset_name)
    _cands = [dataset_path / (_nm + ".lance"), dataset_path / "datasets" / (_nm + ".lance")]
    _lance = next((c for c in _cands if c.exists()), None)
    if _lance is not None:
        _kl = ["pixels", "action", "observation", "step_idx", "episode_idx", "ep_idx"]
        _ds = swm.data.LanceDataset(path=str(_lance), keys_to_cache=cfg.dataset.keys_to_cache)
        _kl = [k for k in _kl if k in getattr(_ds, "column_names", [])]
        return swm.data.LanceDataset(path=str(_lance), keys_to_load=_kl, keys_to_cache=cfg.dataset.keys_to_cache)
    return swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=cfg.dataset.keys_to_cache, cache_dir=dataset_path
    )


def build_process(cfg, dataset):
    """Per-column StandardScaler fitted on dataset stats (z-score normalizers)."""
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler.fit(col_data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def sample_eval_episodes(cfg, dataset):
    """Pick cfg.eval.num_eval random (episode, start_step) pairs for evaluation.

    Each pair defines a start frame; the goal frame is goal_offset_steps ahead of it,
    except when cfg.gip_eval.to_end is set, in which case the goal is each episode's
    last frame and only one start per episode qualifies. cfg.gip_eval.random_start=false
    starts each picked episode at frame 0 (the draw is then over episodes, not frames).
    cfg.gip_eval.min_episode_len keeps only episodes of at least that many frames, so one
    seed picks the same episodes for every goal offset. Episodes too short to reach the
    goal are excluded. Sampling is deterministic given cfg.seed.

    With cfg.gip_eval.full_traj, each pick starts at the episode's FIRST frame and the goal
    is its LAST frame, so the goal offset varies per episode and is returned as a third list
    (None in the default mode).

    Returns:
        (episode_ids, start_step, goal_offsets): lists of size cfg.eval.num_eval
    """
    col = episode_col(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)

    lengths = []
    step_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    ep_idx = np.asarray(dataset.get_col_data(col)).reshape(-1)
    for ep_id in ep_indices:
        lengths.append(np.max(step_idx[ep_idx == ep_id]) + 1)
    lengths = np.array(lengths)

    full_traj = bool(cfg.get("gip_eval", {}).get("full_traj", False))
    if full_traj:
        valid = np.nonzero(step_idx == 0)[0]
        g = np.random.default_rng(cfg.seed)
        picks = np.sort(valid[g.choice(len(valid), size=cfg.eval.num_eval, replace=False)])
        episodes = dataset.get_row_data(picks)[col]
        len_by_ep = {e: int(lengths[i]) for i, e in enumerate(ep_indices)}
        offsets = [len_by_ep[e] - 1 for e in episodes]
        return episodes.tolist(), [0] * len(episodes), offsets

    max_start = lengths - cfg.eval.goal_offset_steps - 1
    max_start_by_ep = {e: max_start[i] for i, e in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_by_ep[e] for e in ep_idx])
    gip_eval = cfg.get("gip_eval", {})
    # episodes eligible for the draw: long enough for the goal offset and at least min_episode_len
    # frames; the same length bar for every goal offset makes one seed pick the same episodes for all
    frames_per_row = lengths[np.searchsorted(ep_indices, ep_idx)]
    eligible_row = (max_start_per_row >= 0) & (frames_per_row >= int(gip_eval.get("min_episode_len", 0)))
    g = np.random.default_rng(cfg.seed)
    if not bool(gip_eval.get("random_start", True)):
        first_frames = np.nonzero((step_idx == 0) & eligible_row)[0]
        picks = np.sort(first_frames[g.choice(len(first_frames), size=cfg.eval.num_eval, replace=False)])
        return dataset.get_row_data(picks)[col].tolist(), [0] * len(picks), None
    if bool(gip_eval.get("to_end", False)):
        valid = np.nonzero((step_idx == max_start_per_row) & eligible_row)[0]
    else:
        valid = np.nonzero((step_idx <= max_start_per_row) & eligible_row)[0]
    picks = np.sort(valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
    episodes = dataset.get_row_data(picks)[col]
    starts = dataset.get_row_data(picks)["step_idx"]
    if len(episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")
    return episodes.tolist(), starts.tolist(), None

###################
#  Model Loading  #
###################

#NOTE: deprecated: GIP is no longer a model
def load_gip_model(run_name, embed_dim=None, epoch=None):
    """Rebuild a GIP model (vanilla JEPA + runtime action_predictor/decoder) and
    load trained weights. load_pretrained() cannot: its config.json is cfg.model
    (no GIP modules) and it rejects a folder with multiple weights_epoch_*.pt.
    We pick the latest (or requested) epoch and read config.json directly.
    Returns (model, adim) with adim = frameskip * action_dim.
    """
    cache = get_cache_dir(sub_folder="checkpoints")
    run_dir = Path(cache, run_name)
    pts = sorted(run_dir.glob("weights_epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not pts:
        pts = sorted(run_dir.glob("*.pt"))
    assert pts, f"no .pt checkpoint in {run_dir}"
    ckpt = (run_dir / f"weights_epoch_{epoch}.pt") if epoch is not None else pts[-1]
    assert ckpt.exists(), f"missing {ckpt}"

    from omegaconf import OmegaConf

    config = OmegaConf.create(json.loads((run_dir / "config.json").read_text()))
    model = hydra_instantiate(config)
    if embed_dim is None:
        embed_dim = int(config.predictor.input_dim)
    adim = int(config.action_encoder.input_dim)
    model.action_predictor = hydra_instantiate(config.predictor)
    hspec = config.get("action_head", None)
    htype = hspec.get("type", "mse") if hspec is not None else "mse"
    if htype == "gmm":
        from lewam.models.module import GMMHead
        model.action_decoder = GMMHead(embed_dim, 2048, adim, n_modes=int(hspec.get("n_modes", 5)))
    elif htype == "diffusion":
        from lewam.models.module import DiffusionHead
        model.action_decoder = DiffusionHead(embed_dim, 2048, adim, n_steps=int(hspec.get("n_steps", 50)))
    else:
        model.action_decoder = MLP(embed_dim, 2048, adim)
    sd = torch.load(ckpt, map_location="cpu")
    # drop alignment_projection: train-time-only (TC-WM InfoNCE), no inference role; keeps the strict key audit meaningful.
    sd = {k: v for k, v in sd.items() if not k.startswith("alignment_projection")}
    if "proprio_encoder.patch_embed.weight" in sd:
        from lewam.models.module import Embedder
        pdim = sd["proprio_encoder.patch_embed.weight"].shape[1]
        model.proprio_encoder = Embedder(input_dim=pdim, emb_dim=embed_dim)
        model.proprio_pred_proj = MLP(embed_dim, 2048, embed_dim)
        model.use_proprio = True
    # multi-task: rebuild the frozen CLIP task table + projection from checkpoint shapes.
    if "task_table" in sd:
        n_t, td = sd["task_table"].shape
        model.task_table = torch.nn.Parameter(torch.zeros(n_t, td), requires_grad=False)
        model.task_proj = torch.nn.Linear(td, embed_dim)
    res = model.load_state_dict(sd, strict=False)
    print(
        f"[GIP] load {run_name} <- {ckpt.name}: Adim={adim} "
        f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}"
    )
    assert not res.unexpected_keys, "unexpected keys -> model build mismatch"
    return model, adim


def load_lewam_split_model(run_name, which="best"):
    """Load a trained LeWAM-Split model (`lewam.models.lewam_split.LeWAMSplit`) + config 
    (saved in `scripts/train_lewam_gc.py`)

    Args:
        run_name: Load from checkpoints/<run_name>/lewam_gc_config.json 
            (arch dims + action z-score stats: action_mean/std, frameskip, action_raw_dim)
        which: select checkpoints/<run_name>/lewam_gc_{which}.pt (best or latest). Defaults to best

    Returns:
        (model, cfg). This is the direct reactive-GC eval path for the split model
    """
    from lewam.models.lewam_split import LeWAMSplit

    cache = Path(get_cache_dir(sub_folder="checkpoints"))
    run_dir = cache / run_name
    cfg = json.loads((run_dir / "lewam_gc_config.json").read_text())
    ckpt = run_dir / ("lewam_gc_latest.pt" if which == "latest" else "lewam_gc_best.pt")
    if not ckpt.exists():
        ckpt = run_dir / "lewam_gc_latest.pt"
    assert ckpt.exists(), f"no lewam_gc_*.pt checkpoint in {run_dir}"
    sd = torch.load(ckpt, map_location="cpu")
    # infer the projector width from the checkpoint so older ckpts load
    proj_w = sd.get("encoder.projector.net.0.weight")
    proj_hidden = int(proj_w.shape[0]) if proj_w is not None else None
    model = LeWAMSplit(
        embed_dim=int(cfg["z_dim"]), action_dim=int(cfg["action_dim"]),
        hidden_dim=int(cfg["hidden_dim"]), img_size=224,
        dropout=float(cfg.get("dropout", 0.1)), proj_hidden=proj_hidden,
    )
    res = model.load_state_dict(sd, strict=True)
    print(f"[SPLIT] load {run_name} <- {ckpt.name}: action_block={cfg['action_dim']} "
          f"(raw {cfg['action_raw_dim']}x{cfg['frameskip']}) H_max={cfg['H_max']} "
          f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    return model, cfg


def load_lewam_unified_model(run_name, which="best"):
    """Load a trained LeWAM-Unified model (`lewam.models.lewam_unified.LeWAMUnified`) + 
    its config (saved by `scripts/train_lewam_unified.py`).
    This is the direct reactive-GC eval path for the unified model;
    like the split loader it does NOT go through the frozen-LeWM/JEPA rebuild.

    Args:
        run_name: Load from checkpoints/<run_name>/lewam_gc_config.json 
            (arch dims + action z-score stats: action_mean/std, frameskip, action_raw_dim)
        which: select checkpoints/<run_name>/lewam_gc_{which}.pt (best or latest). Defaults to best

    Returns:
        (model, cfg). This is the direct reactive-GC eval path for the split model
    """
    from lewam.models.lewam_unified import LeWAMUnified

    cache = Path(get_cache_dir(sub_folder="checkpoints"))
    run_dir = cache / run_name
    cfg = json.loads((run_dir / "lewam_unified_config.json").read_text())
    ckpt = run_dir / ("lewam_unified_latest.pt" if which == "latest" else "lewam_unified_best.pt")
    if not ckpt.exists():
        ckpt = run_dir / "lewam_unified_latest.pt"
    assert ckpt.exists(), f"no lewam_unified_*.pt checkpoint in {run_dir}"
    sd = torch.load(ckpt, map_location="cpu")
    proj_w = sd.get("encoder.projector.net.0.weight")
    proj_hidden = int(proj_w.shape[0]) if proj_w is not None else None
    model = LeWAMUnified(
        encoder_size=str(cfg.get("encoder_size", "tiny")),
        encoder_backbone=str(cfg.get("encoder_backbone", "scratch")),
        encoder_ckpt="defer",
        embed_dim=int(cfg["z_dim"]), action_dim=int(cfg["action_dim"]),
        hidden_dim=int(cfg["hidden_dim"]), img_size=224,
        dropout=float(cfg.get("dropout", 0.1)), proj_hidden=proj_hidden,
        agg_depth=int(cfg["agg_depth"]), agg_heads=int(cfg.get("agg_heads", 4)),
        agg_dim_head=(int(cfg["agg_dim_head"]) or None) if cfg.get("agg_dim_head") else None,
        agg_mlp_dim=(int(cfg["agg_mlp_dim"]) or None) if cfg.get("agg_mlp_dim") else None,
        agg_residual=bool(cfg.get("agg_residual", False)),
        agg_gate=bool(cfg.get("agg_gate", False)),
        agg_action_cond=bool(cfg.get("agg_action_cond", False)),
        dyn_goal_cond=bool(cfg.get("dyn_goal_cond", True)),
        head_type=str(cfg.get("head_type", "mse")), 
        n_mix=int(cfg.get("n_mix", 5)),
        flow_H=int(cfg.get("flow_H", 1)), 
        num_chunks=int(cfg.get("num_chunks", 1)),
        drop_goal=bool(cfg.get("drop_goal", False)), 
        crop_size=int(cfg.get("crop_size", 0) or 0),
        dyn_action_embed_dim=int(cfg.get("dyn_action_embed_dim", 0) or 0),
        use_idm=bool(float(cfg.get("w_idm", 0) or 0) > 0),
        use_prefix=bool(cfg.get("use_prefix", False)),
        prefix_H=int(cfg.get("prefix_H", 5)),
        prefix_depth=int(cfg.get("prefix_depth", 2)),
        prefix_heads=int(cfg.get("prefix_heads", 4)),
        # head must be rebuilt with flags it was trained under (load is strict)
        latent_h=str(cfg.get("latent_h", "") or ""),
        h_codes=int(cfg.get("h_codes", 16)),
        h_code_dim=int(cfg.get("h_code_dim", 64)),
    )
    res = model.load_state_dict(sd, strict=True)
    print(f"[UNIFIED] load {run_name} <- {ckpt.name}: action_block={cfg['action_dim']} "
          f"(raw {cfg['action_raw_dim']}x{cfg['frameskip']}) H_max={cfg['H_max']} "
          f"agg_depth={cfg['agg_depth']} action_cond={cfg.get('agg_action_cond', False)} "
          f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    return model, cfg


# GC-IDM: planning-free goal-conditioned IDM on frozen LeWM latents (arXiv 2605.08732). mode=gcidm. NO CEM / NO WM.
def load_crossattn_model(run_name, which="best"):
    """Load a trained cross-attention LeWAM (scripts/train_crossattn.py) + its config, which carries
    the action z-score stats / frameskip / block dim used to un-normalize predicted actions."""
    from lewam.models.lewam_crossattn import build_model

    cache = Path(get_cache_dir(sub_folder="checkpoints"))
    run_dir = cache / run_name
    cfg = json.loads((run_dir / "crossattn_config.json").read_text())
    ckpt = run_dir / ("crossattn_latest.pt" if which == "latest" else "crossattn_best.pt")
    if not ckpt.exists():
        ckpt = run_dir / "crossattn_latest.pt"
    assert ckpt.exists(), f"no crossattn_*.pt checkpoint in {run_dir}"
    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model, cfg


def load_dp_model(run_name):
    """Load a diffusion_policy checkpoint (dill payload: hydra cfg + EMA weights) from
    $STABLEWM_HOME/checkpoints/<run_name>/dp.ckpt. Needs the diffusion_policy repo on
    PYTHONPATH."""
    import dill
    import hydra as _hydra
    import robomimic.models.base_nets as _rmbn
    if not hasattr(_rmbn, "CropRandomizer"):   # robomimic 0.3 moved it out of base_nets
        import robomimic.models.obs_core as _rmoc
        _rmbn.CropRandomizer = _rmoc.CropRandomizer
    ckpt = Path(get_cache_dir(sub_folder="checkpoints")) / run_name / "dp.ckpt"
    assert ckpt.exists(), f"no dp.ckpt in {ckpt.parent}"
    payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = _hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["ema_model"])
    return policy.eval(), cfg


def load_jointflow_model(run_name, which="best"):
    """Load a trained jointflow (scripts/train_jointflow.py) + its config. `action_dim` (the
    frameskip block dim the eval harness sizes adim with) is derived here; the trainer config
    stores only fs and action_raw_dim."""
    from lewam.models.jointflow import build_model as build_jointflow

    cache = Path(get_cache_dir(sub_folder="checkpoints"))
    run_dir = cache / run_name
    cfg = json.loads((run_dir / "jointflow_config.json").read_text())
    cfg.setdefault("action_dim", int(cfg["fs"]) * int(cfg["action_raw_dim"]))
    ckpt = run_dir / ("jointflow_latest.pt" if which == "latest" else "jointflow_best.pt")
    if not ckpt.exists():
        ckpt = run_dir / "jointflow_latest.pt"
    assert ckpt.exists(), f"no jointflow_*.pt checkpoint in {run_dir}"
    kind = cfg.get("model", "jointflow")
    if kind == "twinflow":
        from lewam.models.twinflow import build_model as build_twinflow
        model = build_twinflow(cfg)
    elif kind == "motflow":
        from lewam.models.motflow import build_model as build_motflow
        model = build_motflow(cfg)
    else:
        model = build_jointflow(cfg)
    sd = torch.load(ckpt, map_location="cpu")
    res = model.load_state_dict(sd, strict=False)
    # state_mu/state_sd (the --state_target_norm running stats) were added on 2026-08-27; older
    # checkpoints lack them and keep the identity stats they were built with. Nothing else may be missing.
    assert set(res.missing_keys) <= {"state_mu", "state_sd"} and not res.unexpected_keys, \
        f"checkpoint/model mismatch: missing={res.missing_keys} unexpected={res.unexpected_keys}"
    return model, cfg


def load_gcidm_model(run_name):
    """Load a trained GC-IDM: the FROZEN LeWM encoder + the GCIDMHead.

    The trainer (train_gcidm.py) writes checkpoints/<run_name>/gcidm_config.json
    (head dims, frozen-LeWM weights path, action z-score stats) and
    gcidm_head_best.pt. We rebuild the frozen LeWM exactly like train_gcidm
    (so z = encode({pixels})['emb'][:,0] matches) and the GCIDMHead,
    then load the head weights. Returns (lewm, head, gcfg)."""
    import lewam.models.gcidm as _gcidm
    from lewam.models.jepa import build_frozen_lewm

    cache = Path(get_cache_dir(sub_folder="checkpoints"))
    run_dir = cache / run_name
    gcfg = json.loads((run_dir / "gcidm_config.json").read_text())
    head_pt = run_dir / "gcidm_head_best.pt"
    if not head_pt.exists():
        head_pt = run_dir / "gcidm_head_latest.pt"
    assert head_pt.exists(), f"no gcidm head weights in {run_dir}"

    lewm = build_frozen_lewm(
        gcfg["weights"], embed_dim=int(gcfg["emb_dim"]), history_size=3,
        img_size=224, action_block_dim=int(gcfg["action_dim"]),
    )

    if gcfg.get("from_scratch") or gcfg.get("weights") == "self":
        full_pt = run_dir / "gcidm_full_model_best.pt"
        if not full_pt.exists():
            full_pt = run_dir / "gcidm_full_model_latest.pt"
        assert full_pt.exists(), f"no gcidm_full_model_*.pt in {run_dir}"
        full = torch.load(full_pt, map_location="cpu", weights_only=False)
        enc_sd = {k[len("encoder."):]: v for k, v in full.items() if k.startswith("encoder.")}
        r = lewm.load_state_dict(enc_sd, strict=False)
        assert not r.unexpected_keys, f"unexpected encoder keys: {r.unexpected_keys[:5]}"
        print(f"[GCIDM] from_scratch encoder <- {full_pt.name}: "
              f"loaded={len(enc_sd)} missing={len(r.missing_keys)}")
    head = _gcidm.GCIDMHead(
        emb_dim=int(gcfg["emb_dim"]), action_dim=int(gcfg["action_dim"]),
        hidden_dim=int(gcfg["hidden_dim"]), n_freqs=int(gcfg["n_freqs"]),
        dropout=float(gcfg["dropout"]),
    )
    sd = torch.load(head_pt, map_location="cpu")
    res = head.load_state_dict(sd, strict=True)
    print(f"[GCIDM] load {run_name} <- {head_pt.name}  H_max={gcfg['H_max']}  "
          f"action_dim={gcfg['action_dim']}  ablate_horizon={gcfg.get('ablate_horizon', False)}")
    return lewm, head, gcfg


#NOTE: deprecated: GIP is no longer a model
def attach_intention_actor(model, history_size=3, goal_conditioned=False):
    """Make a GIP model satisfy the Actionable protocol so CEM warm-starts from
    the intention. Binds get_action(info, horizon, prefix_actions) onto the
    INSTANCE only -> vanilla planning models stay zero-initialized.
    goal_conditioned=True -> the warm-start prior is GOAL-AWARE (rolls toward z_goal);
    this is mode 4 (policy-as-prior) with a goal-conditioned prior. Falls back to the
    goal-agnostic prior (existing behaviour) if no 'goal' is in the planning info."""

    def get_action(self, info_dict, horizon, prefix_actions=None):
        goal_emb = None
        if goal_conditioned and isinstance(info_dict, dict) and "goal" in info_dict:
            g = info_dict["goal"]
            g = g[:, -1:] if g.ndim == 5 else g.unsqueeze(1)  # -> (B,1,C,H,W) single goal frame
            g = g.to(next(self.parameters()).device).float()
            goal_emb = self.encode({"pixels": g})["emb"][:, 0]
        # horizon_modulator: feed AdaLN-Zero the remaining-horizon (h_norm) at rollout start; None -> identity.
        horizon_norm = getattr(self, "_guided_horizon_norm", None) if (
            getattr(self, "horizon_modulator", None) is not None) else None
        return self.intention_rollout(
            info_dict, horizon, prefix_actions=prefix_actions,
            history_size=history_size, goal_emb=goal_emb, horizon_norm=horizon_norm,
        )

    model.get_action = types.MethodType(get_action, model)
    return model


####################
#  Policy Classes  #
####################

def _as_h0(h0):
    """Normalize horizon0: None stays None, scalars become float, per-env sequences arrays."""
    if h0 is None:
        return None
    a = np.asarray(h0, dtype=np.float64)
    return a if a.ndim else float(a)


def _h0_at(h0, i):
    return float(h0[i]) if isinstance(h0, np.ndarray) else float(h0)


# Policy factory
def build_policy(cfg, model, adim, process, transform, goal_offsets=None, policy_kwargs=None):
    """Configure policy from cfg. goal_offsets: per-env raw-frame goal offsets (full-traj
    eval); when given, horizon0 becomes a per-env array offsets/action_block."""
    mode = cfg.get("gip_eval", {}).get("mode", "bc")
    goal_conditioned = bool(cfg.get("gip_eval", {}).get("goal_conditioned", False))
    action_block = int(cfg.plan_config.action_block)
    h0_full = (np.asarray(goal_offsets, np.float64) / float(action_block)
               if goal_offsets is not None else None)

    # mode=split_policy: LeWAM-Split adapter. `model` is a loaded LeWAMSplit with its config
    # attached as model._split_cfg (done in eval_gip.py)
    if mode == "split_policy":
        ge = cfg.get("gip_eval", {})
        split_cfg = getattr(model, "_split_cfg")
        horizon0 = ge.get("horizon0", None)
        if horizon0 is None:
            horizon0 = h0_full if h0_full is not None else \
                float(cfg.eval.goal_offset_steps) / float(action_block)
        return LeWAMSplitPolicy(
            model=model, cfg=split_cfg, action_block=action_block,
            action_dim=adim // action_block, horizon0=_as_h0(horizon0),
            H_max=int(ge.get("horizon_H_max", split_cfg.get("H_max", 50))),
            process=process, transform=transform,
            ablate_horizon=bool(ge.get("ablate_horizon", split_cfg.get("ablate_horizon", False))),
        )

    if mode == "crossattn_policy":
        ge = cfg.get("gip_eval", {})
        ca_cfg = getattr(model, "_crossattn_cfg")
        horizon0 = ge.get("horizon0", None)
        if horizon0 is None:
            horizon0 = h0_full if h0_full is not None else \
                float(cfg.eval.goal_offset_steps) / float(action_block)
        if ge.get("flow_steps"):
            model.policy_head.n_flow_steps = int(ge["flow_steps"])
        return CrossAttnPolicy(
            model=model, cfg=ca_cfg, action_block=action_block,
            action_dim=adim // action_block, horizon0=_as_h0(horizon0),
            H_max=int(ge.get("horizon_H_max", ca_cfg.get("H_max", 50))),
            process=process, transform=transform,
            ablate_horizon=bool(ge.get("ablate_horizon", ca_cfg.get("ablate_horizon", False))),
            exec_actions=int(ge.get("exec_actions", 0)))

    if mode == "dp_policy":
        dp_cfg = getattr(model, "_dp_cfg")
        return DPTPolicy(model=model, cfg=dp_cfg, action_dim=adim, process={}, transform={})

    if mode in ("jointflow_policy", "jointflow_plan", "jointflow_gc"):
        ge = cfg.get("gip_eval", {})
        jf_cfg = getattr(model, "_jointflow_cfg")
        horizon0 = ge.get("horizon0", None)
        if horizon0 is None:
            horizon0 = h0_full if h0_full is not None else \
                float(cfg.eval.goal_offset_steps) / float(action_block)
        if ge.get("flow_steps"):
            model.n_flow_steps = int(ge["flow_steps"])
        shared = dict(
            model=model, cfg=jf_cfg, action_block=action_block,
            action_dim=adim // action_block, horizon0=_as_h0(horizon0),
            H_max=int(ge.get("horizon_H_max", 50)),
            process=process, transform=transform,
            pixel_scale=str(ge.get("pixel_scale", "norm")),
            exec_actions=int(ge.get("exec_actions", 0)),
            flow_seed=int(cfg.seed))
        if mode == "jointflow_policy":
            return JointFlowPolicy(**shared)
        if mode == "jointflow_gc":
            return JointFlowGCPolicy(shuffle_goal=bool(ge.get("shuffle_goal", False)),
                                     dyn_log=bool(ge.get("dyn_log", False)),
                                     dyn_random_p=float(ge.get("dyn_random_p", 0.0)), **shared)
        return JointFlowPlanPolicy(
            plan_mode=str(ge.get("plan_mode", "best_of_k")),
            plan_score=str(ge.get("plan_score", "joint")),
            plan_k=int(ge.get("plan_k", 32)),
            plan_rollout=int(ge.get("plan_rollout", 1)),
            plan_random_candidates=bool(ge.get("plan_random_candidates", False)),
            cem_iters=int(ge.get("cem_iters", 3)),
            cem_elites=int(ge.get("cem_elites", 6)),
            cem_std=float(ge.get("cem_std", 0.5)),
            cem_init=str(ge.get("cem_init", "policy")),
            ext_wm_path=str(ge.get("ext_wm_path", "")),
            ext_wm_action_block_dim=int(ge.get("ext_wm_action_block_dim", 0)),
            ext_wm_embed_dim=int(ge.get("ext_wm_embed_dim", 192)),
            ext_wm_hist=int(ge.get("ext_wm_hist", 3)),
            pm_steps=int(ge.get("pm_steps", 20)),
            pm_lr=float(ge.get("pm_lr", 0.02)),
            pm_rho=float(ge.get("pm_rho", 0.3)),
            pm_random=bool(ge.get("pm_random", False)),
            oracle_cands=str(ge.get("oracle_cands", "uniform")),
            subgoal_every=int(ge.get("subgoal_every", 0)),
            grad_steps=int(ge.get("grad_steps", 50)),
            grad_lr=float(ge.get("grad_lr", 0.05)),
            grad_clip=float(ge.get("grad_clip", 10.0)),
            grad_tr=float(ge.get("grad_tr", 0.0)),
            grad_action_clip=float(ge.get("grad_action_clip", 3.0)),
            pm_K=int(ge.get("pm_K", 0)),
            pm_chunk=int(ge.get("pm_chunk", 8)),
            plan_goal_time=bool(ge.get("plan_goal_time", True)),
            **(policy_kwargs or {}), **shared)

    # mode=unified_policy: LeWAM-Unified adapter. `model` is a loaded LeWAMUnified with its config
    # attached as model._unified_cfg (done in eval_gip.py).
    if mode in ("unified_policy", "unified_cem", "unified_grad", "unified_prompt_mpc"):
        ge = cfg.get("gip_eval", {})
        uni_cfg = getattr(model, "_unified_cfg")
        horizon0 = ge.get("horizon0", None)
        if horizon0 is None:
            horizon0 = h0_full if h0_full is not None else \
                float(cfg.eval.goal_offset_steps) / float(action_block)
        common = dict(model=model, cfg=uni_cfg, action_block=action_block,
                      action_dim=adim // action_block, horizon0=_as_h0(horizon0),
                      H_max=int(ge.get("horizon_H_max", uni_cfg.get("H_max", 50))),
                      process=process, transform=transform,
                      ablate_horizon=bool(ge.get("ablate_horizon", uni_cfg.get("ablate_horizon", False))),
                      exec_blocks=int(ge.get("exec_blocks", 1)),
                      exec_actions=int(ge.get("exec_actions", 0)))
        # flow head eval-time ODE steps
        if ge.get("flow_steps") and hasattr(model.gc_head, "n_steps"):
            model.gc_head.n_steps = int(ge["flow_steps"])
        if mode == "unified_cem":
            return LeWAMUnifiedCEMPolicy(
                cem_K=int(ge.get("cem_K", 256)), cem_M=int(ge.get("cem_M", 32)),
                cem_iter=int(ge.get("cem_iter", 4)), cem_H=int(ge.get("cem_H", 5)),
                cem_std=float(ge.get("cem_std", 1.0)),
                cem_warm=bool(ge.get("cem_warm", False)),
                cem_state=str(ge.get("cem_state", "c")),
                cem_exec_full=bool(ge.get("cem_exec_full", False)),
                cem_propose=str(ge.get("cem_propose", "cem")),
                cem_cost=str(ge.get("cem_cost", "final")),
                cem_dyn_mode=str(ge.get("cem_dyn_mode", "auto")),
                cem_seed=int(cfg.seed),
                log_latents=bool(ge.get("dump_latents", "")),
                ctx_cap=int(ge.get("ctx_cap", uni_cfg.get("context_len", 5))), **common)
        # ---- Prompt-MPC (OURS; paper name). Per-episode prompt delta on the goal latent
        # through the frozen policy. pm_seg>0 = per-step prompts; pm_cost=anymin = the
        # reach-anytime-aligned cost (task-completion default).
        if mode == "unified_prompt_mpc":
            from lewam.models.prompt_mpc import LeWAMUnifiedPromptMPCPolicy
            return LeWAMUnifiedPromptMPCPolicy(
                pm_steps=int(ge.get("pm_steps", 20)), pm_lr=float(ge.get("pm_lr", 0.02)),
                pm_clip=float(ge.get("pm_clip", 5.0)), pm_rho=float(ge.get("pm_rho", 0.3)),
                pm_cost=str(ge.get("pm_cost", "terminal")),
                pm_random=bool(ge.get("pm_random", False)),
                pm_seg=int(ge.get("pm_seg", 0)),
                pm_r_rho=float(ge.get("pm_r_rho", 0.1)),
                pm_H=int(ge.get("pm_H", 5)),
                pm_H_auto=bool(ge.get("pm_H_auto", False)),
                pm_exec_k=int(ge.get("pm_exec_k", 0)),
                pm_exec_full=bool(ge.get("pm_exec_full", True)),
                cem_seed=int(cfg.seed),
                log_latents=bool(ge.get("dump_latents", "")),
                ctx_cap=int(ge.get("ctx_cap", uni_cfg.get("context_len", 5))), **common)
        # ---- Gradient MPC (Jyothir et al., arXiv:2312.17227 lineage): the action sequence
        # is the parameter, warm-started from the policy.
        if mode == "unified_grad":
            from lewam.models.grad_mpc import LeWAMUnifiedGradPolicy
            return LeWAMUnifiedGradPolicy(
                grad_steps=int(ge.get("grad_steps", 50)), grad_lr=float(ge.get("grad_lr", 0.05)),
                grad_H=int(ge.get("grad_H", 5)), grad_clip=float(ge.get("grad_clip", 10.0)),
                grad_action_clip=ge.get("grad_action_clip", None),
                grad_dyn_mode=str(ge.get("grad_dyn_mode", "auto")),
                grad_exec_full=bool(ge.get("grad_exec_full", True)),
                grad_warm=bool(ge.get("grad_warm", True)),
                grad_H_auto=bool(ge.get("grad_H_auto", False)),
                grad_exec_k=int(ge.get("grad_exec_k", 0)),
                grad_tr=float(ge.get("grad_tr", 0.0)),
                cem_seed=int(cfg.seed),
                log_latents=bool(ge.get("dump_latents", "")),
                ctx_cap=int(ge.get("ctx_cap", uni_cfg.get("context_len", 5))), **common)
        # ctx_cap defaults to the TRAINED context window (context_len in the ckpt config)
        return LeWAMUnifiedPolicy(
            ctx_cap=int(ge.get("ctx_cap", uni_cfg.get("context_len", 0))),
            log_latents=bool(ge.get("dump_latents", "")), **common)

    # mode=gcidm: the `model` arg is unused; GCIDM loads its OWN frozen-LeWM + head via load_gcidm_model.
    if mode == "gcidm":
        gc = cfg.gip_eval
        lewm, head, gcfg = load_gcidm_model(gc.get("gcidm_run", cfg.policy))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lewm = lewm.to(device); head = head.to(device)
        # horizon0 = goal_offset(raw frames)/frameskip(=action_block) = obs-steps to the goal frame. Override via gip_eval.gcidm_horizon.
        horizon0 = gc.get("gcidm_horizon", None)
        if horizon0 is None:
            horizon0 = h0_full if h0_full is not None else \
                float(cfg.eval.goal_offset_steps) / float(action_block)
        return GCIDMPolicy(
            lewm=lewm, head=head, action_block=action_block,
            action_dim=int(gcfg["action_dim"]) // action_block,
            H_max=int(gcfg["H_max"]), horizon0=_as_h0(horizon0),
            process=process, transform=transform,
            ablate_horizon=bool(gc.get("ablate_horizon", gcfg.get("ablate_horizon", False))),
        )

    # mode=bc and mode=policy are the SAME direct forward policy; `policy` exposes the goal_conditioned
    # switch (bc = policy with goal_conditioned=false). One forward pass, NO solver.
    if mode in ("bc", "policy"):
        ge = cfg.get("gip_eval", {})
        # horizon0 = goal_offset/frameskip (obs-steps to goal); override via gip_eval.horizon0.
        # H_max MUST match training H_max (action_pred.horizon_H_max, default 50).
        horizon0 = ge.get("horizon0", None)
        if horizon0 is None:
            horizon0 = h0_full if h0_full is not None else \
                float(cfg.eval.goal_offset_steps) / float(action_block)
        return BCPolicy(
            model=model,
            action_block=action_block,
            action_dim=adim // action_block,
            process=process,
            transform=transform,
            goal_conditioned=(mode == "policy" and goal_conditioned),
            H_max=int(ge.get("horizon_H_max", 50)),
            horizon0=_as_h0(horizon0),
            history_size=int(cfg.get("history_size", 3)),
        )

    # guided + planning both use the WM CEM planner; `guided` makes the model Actionable so CEM
    # warm-starts from the intention proposal (goal-aware when goal_conditioned).
    if mode == "guided":
        attach_intention_actor(model, history_size=cfg.get("history_size", 3),
                               goal_conditioned=goal_conditioned)
        # if the model has a horizon_modulator, feed the warm-start the initial horizon
        # (goal_offset/frameskip, normalized by H_max).
        if getattr(model, "horizon_modulator", None) is not None:
            ge = cfg.get("gip_eval", {})
            H_max = float(ge.get("horizon_H_max", 50))
            h0 = ge.get("horizon0", None)
            if h0 is None:
                h0 = float(cfg.eval.goal_offset_steps) / float(action_block)
            model._guided_horizon_norm = float(min(h0, H_max) / H_max)
    elif mode != "planning":
        raise ValueError(f"unknown gip_eval.mode={mode!r})")

    config = swm.PlanConfig(**cfg.plan_config)
    ge = cfg.get("gip_eval", {})
    if mode == "planning" and str(ge.get("plan_mode", "")) == "oracle_bok":
        # owner sanity check: the world model scores a fixed set (expert + K-1 wrong sequences)
        # with its own rollout cost instead of running CEM (lewam/models/oracle_solver.py)
        from lewam.models.oracle_solver import OracleSolver, OracleWorldModelPolicy
        pk = policy_kwargs or {}
        assert "expert_actions" in pk, "oracle_bok needs the demo actions (eval_gip passes them)"
        solver = OracleSolver(model, pk["expert_actions"], cands=str(ge.get("oracle_cands", "uniform")),
                              k=int(ge.get("plan_k", 32)), seed=int(cfg.seed),
                              device="cuda" if torch.cuda.is_available() else "cpu")
        return OracleWorldModelPolicy(solver=solver, config=config, process=process, transform=transform)
    solver = hydra_instantiate(cfg.solver, model=model)
    return swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

# reactive BC policy (mode = bc)
class BCPolicy(BasePolicy):
    """Behavioral-cloning policy: run the intention head directly. Each replan
    encodes the current frame (position 0, as trained), proposes one
    frameskip-stacked block via the action head, unstacks it into `action_block`
    raw actions, and executes them before re-observing. Buffer/inverse_transform
    handling mirrors WorldModelPolicy."""

    def __init__(self, model, action_block, action_dim, process=None, transform=None,
                 goal_conditioned=False, H_max=50, horizon0=None, history_size=3, **kwargs):
        super().__init__(**kwargs)
        # goal_conditioned: False -> goal-agnostic bc; True -> encode goal -> z_goal, one forward pass, NO solver.
        self.goal_conditioned = bool(goal_conditioned)
        self.type = "goal_cond_policy" if goal_conditioned else "behavioral_cloning"
        self.model = model.eval()
        self.process = process or {}
        self.transform = transform or {}
        self.action_block = int(action_block)
        self.action_dim = int(action_dim)
        self._action_buffer = None
        # horizon countdown (obs-steps): init horizon0=goal_offset/frameskip, -1 per replan, clamp >=1,
        # norm min(h,H_max)/H_max, fed to intention_rollout. Active only if horizon_modulator present.
        self.use_horizon = getattr(self.model, "horizon_modulator", None) is not None
        self.H_max = int(H_max)
        self.horizon0 = _as_h0(horizon0)
        self._steps_left = None  # per-env remaining obs-steps
        # buffer last HS observed frames + last HS-1 emitted action blocks per env so the eval
        # context (z_{t-HS+1..t} + a_{<t}) matches training. Active only for the history-conditioned GC head.
        self.history_size = int(history_size)
        self.use_history = bool(getattr(self.model, "use_action_history", True))
        self._frame_buf = None      # per-env deque of the last HS preprocessed frames
        self._past_act_buf = None   # per-env deque of the last HS-1 emitted raw action blocks

    def set_env(self, env):
        self.env = env
        num_envs = getattr(env, "num_envs", 1)
        self._action_buffer = [deque() for _ in range(num_envs)]
        self._frame_buf = [deque(maxlen=self.history_size) for _ in range(num_envs)]
        self._past_act_buf = [deque(maxlen=max(self.history_size - 1, 0)) for _ in range(num_envs)]
        if self.use_horizon and self.horizon0 is not None:
            self._steps_left = np.full(num_envs, self.horizon0, dtype=np.float64)

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        info_dict = self._prepare_info(info_dict)
        num_envs = self.env.num_envs
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(num_envs)]
        if self.use_horizon and self.horizon0 is not None and self._steps_left is None:
            self._steps_left = np.full(num_envs, self.horizon0, dtype=np.float64)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(num_envs):
                if flush[i]:
                    self._action_buffer[i].clear()
                    if self._steps_left is not None:
                        self._steps_left[i] = _h0_at(self.horizon0, i)  # reset countdown for the new episode
                    if self._frame_buf is not None:
                        self._frame_buf[i].clear(); self._past_act_buf[i].clear()  # reset history

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(num_envs, dtype=bool)

        # push the current obs-frame into each live env's HS buffer every replan (one replan == one obs-step).
        cur_px = info_dict["pixels"]  # (n, 1, C, H, W) preprocessed
        if self.use_horizon and self._frame_buf is not None:
            for i in range(num_envs):
                if not dead[i] and len(self._action_buffer[i]) == 0:
                    fr = cur_px[i]
                    fr = fr[-1] if (hasattr(fr, "ndim") and fr.ndim == 4) else fr  # (C,H,W)
                    self._frame_buf[i].append(fr)

        replan = [i for i in range(num_envs) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            device = next(self.model.parameters()).device
            # goal-conditioned: encode the (already-transformed) goal frame -> z_goal.
            goal_emb = None
            if self.goal_conditioned and "goal" in info_dict:
                gpx = info_dict["goal"][replan]
                gpx = gpx[:, -1:] if gpx.ndim == 5 else gpx.unsqueeze(1)  # -> (R,1,C,H,W) single goal frame
                gpx = gpx.to(device).float()
                goal_emb = self.model.encode({"pixels": gpx})["emb"][:, 0]  # (R, D)
            # remaining-horizon (normalized) for the AdaLN-Zero hook; None -> identity.
            horizon_norm = None
            if self.use_horizon and self._steps_left is not None:
                steps = np.maximum(self._steps_left[replan], 1.0)
                hn = np.minimum(steps, self.H_max) / self.H_max
                horizon_norm = torch.tensor(hn, device=device, dtype=torch.float32)

            if self.use_horizon and self._frame_buf is not None:
                # feed buffered HS frames + past-action blocks a_{<t} to rebuild the training context;
                # per-env histories differ in length early, so process each env then re-stack.
                blocks = torch.full((len(replan), self.action_block * self.action_dim), float("nan"))
                for row, i in enumerate(replan):
                    frames = list(self._frame_buf[i])                 # up to HS (C,H,W)
                    px_hist = torch.stack(frames, dim=0).unsqueeze(0).to(device).float()  # (1,T0,C,H,W)
                    pab = None
                    if self.use_history and len(self._past_act_buf[i]) > 0:
                        # a_{<t} = last (T0-1) committed blocks; action_encoder consumes raw blocks (no z-score), as in training.
                        n_past = px_hist.size(1) - 1
                        if n_past > 0:
                            past = list(self._past_act_buf[i])[-n_past:]
                            pab = torch.stack(past, dim=0).unsqueeze(0).to(device).float()  # (1,n_past,Adim)
                    ge = goal_emb[row:row + 1] if goal_emb is not None else None
                    hn = horizon_norm[row:row + 1] if horizon_norm is not None else None
                    blk = self.model.intention_rollout(
                        {"pixels": px_hist}, horizon=1, goal_emb=ge, horizon_norm=hn,
                        history_size=self.history_size, past_action_blocks=pab)[:, 0]  # (1, Adim)
                    blocks[row] = blk[0].cpu()
            else:
                px = info_dict["pixels"][replan]
                blocks = self.model.intention_rollout({"pixels": px}, horizon=1, goal_emb=goal_emb,
                                                      horizon_norm=horizon_norm)[:, 0]   # (R, Adim)
                blocks = blocks.cpu()
            block = blocks.reshape(len(replan), self.action_block, self.action_dim)
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(block[row])
                # record the EMITTED raw action block (full Adim) for the next step's a_{<t}
                if self._past_act_buf is not None:
                    self._past_act_buf[i].append(blocks[row].reshape(-1).clone())
                if self._steps_left is not None:
                    self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)  # one obs-step consumed

        action = torch.full((num_envs, self.action_dim), float("nan"))
        for i in range(num_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


class GCIDMPolicy(BasePolicy):
    """Planning-free goal-conditioned IDM policy (mode=gcidm).

    Each replan:
      z_t   = frozen-LeWM encode(current pixels)['emb'][:,0]
      z_goal= frozen-LeWM encode(info_dict['goal'])['emb'][:,0]
      h     = remaining horizon in OBS-steps, computed from a per-env counter
              (init = horizon0 obs-steps = goal_offset / frameskip), decremented
              one obs-step per replan, clamped >=1, normalized min(h,H_max)/H_max.
      a     = GCIDMHead(z_t, z_goal, h_norm)   -> ONE 25-d raw-action block.
    The block is unstacked into `action_block` env actions and executed before
    re-observing. NO CEM, NO WM rollout. Action un-normalization mirrors BCPolicy
    (inverse_transform via the same StandardScaler), so env actions match.
    """

    def __init__(self, lewm, head, action_block, action_dim, H_max, horizon0,
                 process=None, transform=None, ablate_horizon=False, **kwargs):
        super().__init__(**kwargs)
        self.type = "gcidm_policy"
        self.lewm = lewm.eval()
        self.head = head.eval()
        self.process = process or {}
        self.transform = transform or {}
        self.action_block = int(action_block)
        self.action_dim = int(action_dim)
        self.H_max = int(H_max)
        self.horizon0 = _as_h0(horizon0)
        self.ablate_horizon = bool(ablate_horizon)
        self._action_buffer = None
        self._steps_left = None  # per-env remaining obs-steps

    def set_env(self, env):
        """Allocate per-env action buffers and reset the horizon countdown."""
        self.env = env
        num_envs = getattr(env, "num_envs", 1)
        self._action_buffer = [deque() for _ in range(num_envs)]
        self._steps_left = np.full(num_envs, self.horizon0, dtype=np.float64)

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        """Return this step's raw action for every env, replanning (frozen-encode +
        GCIDMHead forward) only for envs whose action buffer is empty. See the class
        docstring for the per-replan computation."""
        info_dict = self._prepare_info(info_dict)
        num_envs = self.env.num_envs
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(num_envs)]
            self._steps_left = np.full(num_envs, self.horizon0, dtype=np.float64)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(num_envs):
                if flush[i]:
                    self._action_buffer[i].clear()
                    self._steps_left[i] = _h0_at(self.horizon0, i)  # reset countdown for the new episode

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(num_envs, dtype=bool)

        replan = [i for i in range(num_envs) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            device = next(self.head.parameters()).device
            px = info_dict["pixels"][replan]
            assert "goal" in info_dict, "GCIDM policy needs info_dict['goal'] (goal-reaching eval)"
            gpx = info_dict["goal"][replan]
            gpx = gpx[:, -1:] if gpx.ndim == 5 else gpx.unsqueeze(1)  # (R,1,C,H,W) single goal frame
            cpx = px[:, -1:] if px.ndim == 5 else px.unsqueeze(1)     # (R,1,C,H,W) current frame
            z_t = self.lewm.encode({"pixels": cpx.to(device).float()})["emb"][:, 0]   # (R, D)
            z_g = self.lewm.encode({"pixels": gpx.to(device).float()})["emb"][:, 0]   # (R, D)
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=device, dtype=torch.float32)
            if self.ablate_horizon:
                h_norm = torch.zeros_like(h_norm)
            block = self.head(z_t, z_g, h_norm)                       # (R, Adim)
            block = block.reshape(len(replan), self.action_block, self.action_dim).cpu()
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(block[row])
                self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)  # one obs-step consumed

        action = torch.full((num_envs, self.action_dim), float("nan"))
        for i in range(num_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


class LeWAMSplitPolicy(BasePolicy):
    """Eval adapter for LeWAM-Split (`lewam.models.lewam_split.LeWAMSplit`)
    
    Reactive one-step GC policy (`cfg.gip_eval.mode=split_policy`)
      z_t    = model.encode(current pixels)          # (R, D) cls latent
      z_goal = model.encode(goal pixels)             # (R, D)
      h      = remaining horizon in OBS-steps (init horizon0 = goal_offset/frameskip,
              decremented per replan, clamped >=1, normalized min(h,H_max)/H_max)
      a      = model.gc_head(z_t, z_goal, h_norm)    # (R, block_dim) z-scored action block

    The z-scored block is un-z-scored with the model's saved stats (action_mean/std,
    frameskip, action_raw_dim from lewam_gc_config.json), unstacked into `action_block` 
    env actions, and executed before re-observing. 
    """

    def __init__(self, model, cfg, action_block, action_dim, horizon0=None,
                 H_max=50, process=None, transform=None, ablate_horizon=False, **kwargs):
        """Wrap a trained LeWAMSplit model; cfg is its lewam_gc_config.json (action
        z-score stats + frameskip/H_max), used to un-normalize predicted blocks internally."""
        super().__init__(**kwargs)
        self.type = "lewam_split_policy"
        self.model = model.eval()  # BN-in-projector -> eval() = deterministic running stats
        self.cfg = cfg
        self.frameskip = int(cfg["frameskip"])
        self.raw_adim = int(cfg["action_raw_dim"])
        self.block_dim = int(cfg["action_dim"])           # frameskip * raw_adim
        self.action_block = int(action_block)
        self.action_dim = int(action_dim)                 # per-step raw dim = raw_adim
        self.H_max = int(cfg.get("H_max", H_max))
        self.horizon0 = _as_h0(horizon0) if horizon0 is not None else float(self.H_max)
        self.ablate_horizon = bool(ablate_horizon)
        self.transform = transform or {}
        self.process = {}                                 # action un-norm handled internally
        device = next(model.parameters()).device
        self._amean = torch.tensor(cfg["action_mean"], dtype=torch.float32, device=device)  # (raw_adim,)
        self._astd = torch.tensor(cfg["action_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
        self._action_buffer = None
        self._steps_left = None  # per-env remaining obs-steps

    def set_env(self, env):
        """Allocate per-env action buffers and reset the horizon countdown."""
        self.env = env
        num_envs = getattr(env, "num_envs", 1)
        self._action_buffer = [deque() for _ in range(num_envs)]
        self._steps_left = np.full(num_envs, self.horizon0, dtype=np.float64)

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        """Return this step's raw action for every env, replanning (encode + gc_head
        forward) only for envs whose action buffer is empty. See the class docstring
        for the per-replan computation."""
        info_dict = self._prepare_info(info_dict)
        num_envs = self.env.num_envs
        device = next(self.model.parameters()).device
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(num_envs)]
            self._steps_left = np.full(num_envs, self.horizon0, dtype=np.float64)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(num_envs):
                if flush[i]:
                    self._action_buffer[i].clear()
                    self._steps_left[i] = _h0_at(self.horizon0, i)

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(num_envs, dtype=bool)

        replan = [i for i in range(num_envs) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            curr_obs = info_dict["pixels"][replan]
            assert "goal" in info_dict, "LeWAM-Split eval needs info_dict['goal'] (goal-reaching)"
            goal_obs = info_dict["goal"][replan]
            c_obs = curr_obs[:, -1] if curr_obs.ndim == 5 else curr_obs     # (R,C,H,W) current frame
            g_obs = goal_obs[:, -1] if goal_obs.ndim == 5 else goal_obs     # (R,C,H,W) single goal frame
            z_t = self.model.encode(c_obs.to(device).float())   # (R, D)
            z_g = self.model.encode(g_obs.to(device).float())   # (R, D)
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=device, dtype=torch.float32)
            if self.ablate_horizon:
                h_norm = torch.zeros_like(h_norm)
                
            z_blk = self.model.gc_head.point(self.model.gc_head(z_t, z_g, h_norm))   # (R, block_dim) z-scored (mixture mean for gmm)
            raw = (z_blk.reshape(len(replan), self.frameskip, self.raw_adim)
                   * self._astd + self._amean)             # un-z-score per raw dim
            raw = raw.reshape(len(replan), self.action_block, self.action_dim).cpu()
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(raw[row])
                self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)  # one obs-step consumed

        action = torch.full((num_envs, self.action_dim), float("nan"))
        for i in range(num_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()


class CrossAttnPolicy(LeWAMSplitPolicy):
    """Eval adapter for the cross-attention LeWAM. Reactive: each replan encodes the current frame,
    caches it, and cross-attends the flow head over the last `policy_history_len` frame latents plus
    the goal. The head returns raw actions directly; only the executed prefix is un-z-scored and
    buffered. Dynamics head is unused on this path."""

    def __init__(self, model, cfg, *args, **kwargs):
        self.exec_actions = int(kwargs.pop("exec_actions", 0))
        super().__init__(model, cfg, *args, **kwargs)
        self.type = "crossattn_policy"
        self.policy_history_len = int(cfg["policy_history_len"])
        self.goal_conditioning = bool(cfg["goal_conditioning"])
        self.n_tokens = int(cfg["fs"]) * int(cfg["num_chunks"])
        self._lat_buf = None

    def set_env(self, env):
        super().set_env(env)
        self._lat_buf = [[] for _ in range(getattr(env, "num_envs", 1))]

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        info_dict = self._prepare_info(info_dict)
        num_envs = self.env.num_envs
        device = next(self.model.parameters()).device
        hl = self.policy_history_len
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(num_envs)]
            self._steps_left = np.full(num_envs, self.horizon0, dtype=np.float64)
        if self._lat_buf is None:
            self._lat_buf = [[] for _ in range(num_envs)]

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(num_envs):
                if flush[i]:
                    self._action_buffer[i].clear()
                    self._lat_buf[i].clear()
                    self._steps_left[i] = _h0_at(self.horizon0, i)

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(num_envs, dtype=bool)

        # dense raw history: encode the current frame every step (not just at replan), keep the last hl
        active = [i for i in range(num_envs) if not dead[i]]
        if active:
            curr = info_dict["pixels"][active]
            c_obs = curr[:, -1] if curr.ndim == 5 else curr
            z_cur = self.model.encode(c_obs.to(device).float())
            for row, i in enumerate(active):
                self._lat_buf[i].append(z_cur[row])
                if len(self._lat_buf[i]) > hl:
                    self._lat_buf[i] = self._lat_buf[i][-hl:]

        replan = [i for i in range(num_envs) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            assert "goal" in info_dict, "crossattn eval needs info_dict['goal'] (goal-reaching)"
            goal = info_dict["goal"][replan]
            g_obs = goal[:, -1] if goal.ndim == 5 else goal
            z_g = self.model.encode(g_obs.to(device).float())
            R, D = len(replan), z_g.shape[-1]
            frames = torch.zeros(R, hl, D, device=device)
            frame_pad = torch.ones(R, hl, dtype=torch.bool, device=device)
            for row, i in enumerate(replan):
                buf = self._lat_buf[i][-hl:]
                frames[row, hl - len(buf):] = torch.stack(buf)
                frame_pad[row, hl - len(buf):] = False

            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=device, dtype=torch.float32)
            if self.ablate_horizon:
                h_norm = torch.zeros_like(h_norm)
            raw = self.model.policy_head.act(frames, frame_pad, z_g, self.goal_conditioning, h_norm)
            raw = (raw * self._astd + self._amean).cpu()               # (R, N, raw_adim)
            take = self.exec_actions if 0 < self.exec_actions < self.n_tokens else self.action_block
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(raw[row, :take])
                self._steps_left[i] = max(self._steps_left[i] - take / self.action_block, 1.0)

        action = torch.full((num_envs, self.action_dim), float("nan"))
        for i in range(num_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()


class JointFlowPolicy(LeWAMSplitPolicy):
    """Eval adapter for jointflow (lewam.models.jointflow). Reactive and goal-blind: every step
    encodes the current frame into the rolling latent history; a replan samples the joint
    [action chunk; imagined boundary latents] and executes the first `exec_actions` raw actions
    (default one frameskip block). The imagined latents are discarded on this path."""

    def __init__(self, model, cfg, *args, **kwargs):
        self.exec_actions = int(kwargs.pop("exec_actions", 0))
        # flow-noise generator seeded from the eval seed (the flow-head seeding rule):
        # every model.sample / inpaint-noise draw goes through _flow_generator(), never the global RNG
        self._flow_seed = int(kwargs.pop("flow_seed", 0))
        self._flow_gen = None
        # pixel_scale=raw255: the checkpoint trained on raw 0-255 floats (uint8 cache whose
        # frames the GR dataset floated before run_batch's dtype check could normalize), so
        # the eval transform's ImageNet normalization must be reverted before encode.
        self.pixel_scale = str(kwargs.pop("pixel_scale", "norm"))
        assert self.pixel_scale in ("norm", "raw255"), self.pixel_scale
        stats = spt.data.dataset_stats.ImageNet
        self._in_mean = torch.tensor(stats["mean"]).view(1, 3, 1, 1)
        self._in_std = torch.tensor(stats["std"]).view(1, 3, 1, 1)
        super().__init__(model, cfg, *args, **kwargs)
        self.type = "jointflow_policy"
        self.history_len = int(cfg["policy_history_len"])
        self.num_actions = int(cfg["num_actions_pred"])
        self._append_every_step = True
        self._lat_buf = None
        # a multi-view checkpoint lists its cameras; the env serves the extra ones as pixels.<camera>
        # info keys, normalized here exactly like `pixels` (the transform is looked up by key)
        self.view_keys = views.info_keys(views.columns(cfg.get("views")))
        for key in self.view_keys[1:]:
            if "pixels" in self.transform:
                self.transform[key] = self.transform["pixels"]

    def _enc(self, x):
        """Encode eval pixels at the checkpoint's training scale (see pixel_scale above)."""
        if self.pixel_scale == "raw255":
            x = (x * self._in_std.to(x.device) + self._in_mean.to(x.device)) * 255.0
        return self.model.encode(x)

    def set_env(self, env):
        super().set_env(env)
        self._lat_buf = [[] for _ in range(getattr(env, "num_envs", 1))]

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        info_dict = self._prepare_info(info_dict)
        num_envs = self.env.num_envs
        device = next(self.model.parameters()).device
        hl = self.history_len
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(num_envs)]
            self._steps_left = np.full(num_envs, self.horizon0, dtype=np.float64)
        if self._lat_buf is None:
            self._lat_buf = [[] for _ in range(num_envs)]

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(num_envs):
                if flush[i]:
                    self._flush_env(i)

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(num_envs, dtype=bool)

        # history cadence: raw-consecutive models encode EVERY step; anchor-spaced models
        # (GR on the fs-strided cache) encode only at replans, so eval history matches the
        # training spacing in both cases
        replan = [i for i in range(num_envs) if len(self._action_buffer[i]) == 0 and not dead[i]]
        active = ([i for i in range(num_envs) if not dead[i]]
                  if self._append_every_step else replan)
        n_views = len(self.view_keys)
        if active:
            frames = []
            for key in self.view_keys:
                curr = info_dict[key][active]
                frames.append(curr[:, -1] if curr.ndim == 5 else curr)
            c_obs = torch.stack(frames, dim=1)                                   # (R, views, C, H, W)
            z_cur = self._enc(c_obs.flatten(0, 1).to(device).float()).view(len(active), n_views, -1)
            for row, i in enumerate(active):
                self._lat_buf[i].append(z_cur[row])                              # (views, z) per step
                if len(self._lat_buf[i]) > hl:
                    self._lat_buf[i] = self._lat_buf[i][-hl:]

        if replan:
            R, D = len(replan), self.model.z_dim
            history = torch.zeros(R, n_views, hl, D, device=device)              # view-major, as the model reads it
            history_pad = torch.ones(R, hl, dtype=torch.bool, device=device)
            for row, i in enumerate(replan):
                buf = self._lat_buf[i][-hl:]
                history[row, :, hl - len(buf):] = torch.stack(buf, dim=1)
                history_pad[row, hl - len(buf):] = False
            if n_views == 1:
                history = history[:, 0]
            action_chunk = self._propose(info_dict, replan, history, history_pad)
            raw = (action_chunk * self._astd + self._amean).cpu()      # (R, num_actions, raw_adim)
            take = self._n_actions_to_execute()
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(raw[row, :take])
                self._steps_left[i] = max(self._steps_left[i] - take / self.action_block, 1.0)

        action = torch.full((num_envs, self.action_dim), float("nan"))
        for i in range(num_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()

    def _n_actions_to_execute(self):
        return self.exec_actions if 0 < self.exec_actions <= self.num_actions else self.action_block

    def _flush_env(self, i):
        self._action_buffer[i].clear()
        self._lat_buf[i].clear()
        self._steps_left[i] = _h0_at(self.horizon0, i)

    def _flow_generator(self, device):
        dev = torch.device(device)
        if self._flow_gen is None or self._flow_gen.device != dev:
            self._flow_gen = torch.Generator(device=dev)
            self._flow_gen.manual_seed(self._flow_seed)
        return self._flow_gen

    def _propose(self, info_dict, replan, history, history_pad):
        action_chunk, _imagined = self.model.sample(history, history_pad,
                                                    generator=self._flow_generator(history.device))
        return action_chunk


class DPTPolicy(BasePolicy):
    """Eval adapter for a diffusion_policy checkpoint (mode=dp_policy). Receding horizon
    exactly as the DP codebase evals: stack the last n_obs_steps raw frames (plus proprio
    keys when the checkpoint consumes them), predict_action, execute the first
    n_action_steps actions raw -- the checkpoint carries its own normalizer. Pixels must
    arrive UNTRANSFORMED (uint8 HWC); build_policy passes transform={} for this mode."""

    def __init__(self, model, cfg, action_dim, **kwargs):
        super().__init__(**kwargs)
        self.type = "dp_policy"
        self.model = model
        self.obs_keys = set(cfg.task.shape_meta["obs"].keys())
        self.n_obs = int(cfg.n_obs_steps)
        self.n_act = int(cfg.n_action_steps)
        self.action_dim = int(action_dim)
        self._hist = None
        self._action_buffer = None

    def set_env(self, env):
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._hist = [deque(maxlen=self.n_obs) for _ in range(n)]
        self._action_buffer = [deque() for _ in range(n)]

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        info_dict = self._prepare_info(info_dict)
        num_envs = self.env.num_envs
        device = next(self.model.parameters()).device
        if self._hist is None:
            self.set_env(self.env)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(num_envs):
                if flush[i]:
                    self._hist[i].clear()
                    self._action_buffer[i].clear()

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(num_envs, dtype=bool)

        pixels = np.asarray(info_dict["pixels"])
        if pixels.ndim == 5:
            pixels = pixels[:, -1]
        proprio = np.asarray(info_dict["proprio"]) if "proprio" in info_dict else None
        for i in range(num_envs):
            if dead[i]:
                continue
            pr = proprio[i] if proprio is not None else None
            if not self._hist[i]:
                for _ in range(self.n_obs):   # first stack = n_obs repeats, as DP pads at reset
                    self._hist[i].append((pixels[i], pr))
            else:
                self._hist[i].append((pixels[i], pr))

        replan = [i for i in range(num_envs) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            frames = np.stack([np.stack([f for f, _ in self._hist[i]]) for i in replan])
            frames = np.moveaxis(frames.astype(np.float32) / 255.0, -1, 2)   # (R, To, 3, H, W)
            t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device)
            full = {"sideview_image": t(frames)}
            if proprio is not None:
                props = np.stack([np.stack([p for _, p in self._hist[i]])
                                  for i in replan]).astype(np.float32)
                full.update({"robot0_eef_pos": t(props[..., 0:3]),
                             "robot0_eef_quat": t(props[..., 3:7]),
                             "robot0_gripper_qpos": t(props[..., 7:9])})
            obs = {k: v for k, v in full.items() if k in self.obs_keys}
            chunk = self.model.predict_action(obs)["action"].cpu().numpy()
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(chunk[row, :self.n_act])

        action = np.full((num_envs, self.action_dim), np.nan, dtype=np.float32)
        for i in range(num_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        return action.reshape(*self.env.action_space.shape).astype(np.float32)


class JointFlowGCPolicy(JointFlowPolicy):
    """Goal + horizon conditioned jointflow (mode=jointflow_gc): each replan encodes the goal
    frame and passes h_norm from the inherited horizon countdown into the joint sample."""

    def __init__(self, model, cfg, *args, **kwargs):
        assert len(views.columns(cfg.get("views"))) == 1, "goal-conditioned policies run single-view"
        # diagnostic: hand each env another env's goal (roll across the replan batch), so SR
        # measures how much of the policy is actually goal-driven vs goal-blind behavior
        self.shuffle_goal = bool(kwargs.pop("shuffle_goal", False))
        # dynamics-informativeness probe (owner spec 2026-08-27): at every replan log the
        # model's predicted next latent (joint sample + inpaint under the EXECUTED chunk)
        # and, at the following replan, the REAL encoded latent that arrived; with
        # dyn_random_p > 0 a random z-scored chunk replaces the policy's chunk (and is
        # executed) so pred-vs-real is also measured off-policy. dump_diag writes the npz.
        self.dyn_log = bool(kwargs.pop("dyn_log", False))
        self.dyn_random_p = float(kwargs.pop("dyn_random_p", 0.0))
        self._dyn_prev = None
        self._dyn_rows = []
        super().__init__(model, cfg, *args, **kwargs)
        self.type = "jointflow_gc"
        assert getattr(model, "goal_conditioning", False), "checkpoint lacks goal conditioning"
        # goal_terminal ckpts train on the RAW cache (consecutive-frame history, h_norm==0
        # constant); fs-strided GR ckpts use anchor-spaced history + the horizon countdown
        self._goal_terminal = bool(cfg.get("goal_terminal", False))
        self._append_every_step = self._goal_terminal

    def _propose(self, info_dict, replan, history, history_pad):
        assert "goal" in info_dict, "jointflow_gc eval needs info_dict['goal'] (goal-reaching)"
        device = history.device
        goal = info_dict["goal"][replan]
        g_obs = goal[:, -1] if goal.ndim == 5 else goal
        z_goal = self._enc(g_obs.to(device).float())
        if self.shuffle_goal:
            # identity when only one env is replanning (late-episode tail) -- acceptable
            # for the diagnostic since most replans carry the full batch
            z_goal = torch.roll(z_goal, 1, dims=0)
        if self._goal_terminal:
            h_norm = torch.zeros(len(replan), device=device, dtype=torch.float32)
        else:
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=device, dtype=torch.float32)
        action_chunk, z_imag = self.model.sample(history, history_pad,
                                                 z_goal=z_goal, h_norm=h_norm,
                                                 generator=self._flow_generator(history.device))
        if not self.dyn_log:
            return action_chunk
        return self._dyn_log_step(replan, history, history_pad, z_goal, h_norm,
                                  action_chunk, z_imag)

    def _flush_env(self, i):
        super()._flush_env(i)
        if self._dyn_prev is not None:
            self._dyn_prev[i] = None

    def _dyn_log_step(self, replan, history, history_pad, z_goal, h_norm, action_chunk, z_imag):
        """Close the previous replan's prediction against the real latent that arrived,
        then open a new prediction for the chunk about to execute (policy or random)."""
        device = history.device
        R = len(replan)
        assert self._n_actions_to_execute() == self.action_block, \
            "dyn_log needs exec == one action block so the +fs prediction meets the next replan"
        if self._dyn_prev is None:
            self._dyn_prev = [None] * self.env.num_envs
        for row, i in enumerate(replan):
            prev = self._dyn_prev[i]
            if prev is not None:
                z_real = history[row, -1].detach().cpu()
                self._dyn_rows.append(dict(env=i, z_prev=prev["z_prev"], z_real=z_real,
                                           pred_joint=prev["pred_joint"],
                                           pred_inpaint=prev["pred_inpaint"],
                                           z_goal=prev["z_goal"], is_random=prev["is_random"]))
            self._dyn_prev[i] = None
        exec_chunk = action_chunk.clone()
        is_random = torch.zeros(R, dtype=torch.bool)
        if self.dyn_random_p > 0:
            gen = self._flow_generator(device)
            coin = torch.rand(R, device=device, generator=gen) < self.dyn_random_p
            rnd = torch.randn(action_chunk.shape, device=device, generator=gen)
            exec_chunk[coin] = rnd[coin]
            is_random = coin.cpu()
        gen = self._flow_generator(device)
        A, adim = self.model.num_actions, self.model.action_raw_dim
        noise_a = torch.randn(R, A, adim, device=device, generator=gen)
        noise_s = torch.randn(R, self.model.num_states, self.model.z_dim, device=device,
                              generator=gen)
        z_pi = self.model.sample_inpaint(history, history_pad, exec_chunk, noise_a, noise_s,
                                         z_goal=z_goal, h_norm=h_norm)
        for row, i in enumerate(replan):
            self._dyn_prev[i] = dict(
                z_prev=history[row, -1].detach().cpu(),
                pred_joint=(z_imag[row, 0].detach().cpu() if not bool(is_random[row])
                            else torch.full_like(z_imag[row, 0].cpu(), float("nan"))),
                pred_inpaint=z_pi[row, 0].detach().cpu(),
                z_goal=z_goal[row].detach().cpu(), is_random=bool(is_random[row]))
        return exec_chunk

    def dump_diag(self, path):
        import numpy as np
        rows = self._dyn_rows
        if not rows:
            print("[dyn-log] no closed transitions to dump")
            return
        np.savez(path,
                 env=np.array([r["env"] for r in rows]),
                 z_prev=torch.stack([r["z_prev"] for r in rows]).numpy(),
                 z_real=torch.stack([r["z_real"] for r in rows]).numpy(),
                 pred_joint=torch.stack([r["pred_joint"] for r in rows]).numpy(),
                 pred_inpaint=torch.stack([r["pred_inpaint"] for r in rows]).numpy(),
                 z_goal=torch.stack([r["z_goal"] for r in rows]).numpy(),
                 is_random=np.array([r["is_random"] for r in rows]))
        print(f"[dyn-log] {len(rows)} transitions "
              f"({int(sum(r['is_random'] for r in rows))} random) -> {path}")


def _load_official_lewm(path):
    """Load a LeWM checkpoint (the authors' release or one trained by our train.py port) as the
    OFFICIAL stable_worldmodel.wm.lewm.LeWM, predictor included, by instantiating its own
    config.json with the package classes (probe_wm_discrim's retarget: module.* / jepa.JEPA ->
    stable_worldmodel.wm.lewm.*). `path` = the checkpoint folder, or a .pt inside it (that file
    is used; otherwise the last .pt by name). Returns (model.eval(), cfg, weights filename)."""
    import os as _os
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    folder, weights = (_os.path.dirname(path), path) if path.endswith(".pt") else (path, None)
    raw = json.loads(open(_os.path.join(folder, "config.json")).read())

    def retarget(o):
        if isinstance(o, dict):
            o = {k: retarget(v) for k, v in o.items()}
            t = o.get("_target_")
            if t == "module.ARPredictor":
                o["_target_"] = "stable_worldmodel.wm.lewm.module.Predictor"
            elif isinstance(t, str) and t.startswith("module."):
                o["_target_"] = "stable_worldmodel.wm.lewm.module." + t[len("module."):]
            elif t == "jepa.JEPA":
                o["_target_"] = "stable_worldmodel.wm.lewm.LeWM"
            return o
        if isinstance(o, list):
            return [retarget(v) for v in o]
        return o

    cfg = OmegaConf.create(retarget(raw))
    model = instantiate(cfg)
    if weights is None:
        pts = sorted(p for p in _os.listdir(folder) if p.endswith(".pt"))
        weights = _os.path.join(folder, pts[-1])
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    res = model.load_state_dict(sd, strict=False)
    core = [k for k in res.missing_keys if k.startswith(("encoder.", "predictor.", "action_encoder.", "projector."))]
    assert not core, f"LeWM core weights missing: {core[:5]}"
    assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys[:5]}"
    model.requires_grad_(False)
    return model.eval(), cfg, _os.path.basename(weights)


class JointFlowPlanPolicy(JointFlowPolicy):
    """Goal-reaching planner on a trained jointflow (mode=jointflow_plan). Candidate action
    chunks are scored by the final cost ||z_imag_last - z_goal||^2 over the imagined boundary
    latents; the winner's first `exec_actions` are executed.

      plan_mode=best_of_k: plan_k independent joint samples, cheapest wins
      plan_mode=cem: iterative refit of a candidate mean, warm-started from the previous
        replan's plan shifted by the executed prefix. Candidates are evaluated with
        sample_inpaint under noise fixed per replan, so their costs are comparable.
      plan_mode=steer: the prompt-MPC port to flow. A per-replan bias delta in z_dim is
        added to the goal latent fed to the SAMPLER; every candidate is the policy's own
        flow sample under one fixed noise, so the search stays on the behavioral manifold.
        The cost re-imagines each candidate's actions via sample_inpaint under the TRUE
        goal (closing the exploitation channel jointflow's goal-conditioned state branch
        would otherwise open), and iterate 0 (delta=0, the pure policy sample) stays in
        the candidate set. ||delta|| is capped at pm_rho*||z_goal||.
    """

    def __init__(self, model, cfg, *args, **kwargs):
        assert len(views.columns(cfg.get("views"))) == 1, "the planners imagine one view; multi-view is the reactive policy"
        self.plan_mode = str(kwargs.pop("plan_mode", "best_of_k"))
        self.plan_k = int(kwargs.pop("plan_k", 32))
        self.pm_steps = int(kwargs.pop("pm_steps", 20))
        self.pm_lr = float(kwargs.pop("pm_lr", 0.02))
        self.pm_rho = float(kwargs.pop("pm_rho", 0.3))
        # search-verification control: equal-norm random direction instead of the gradient
        self.pm_random = bool(kwargs.pop("pm_random", False))
        # plan_score=inpaint: re-imagine each sampled candidate's outcome via sample_inpaint
        # under SHARED noise (per replan row), so candidate costs differ only through the
        # actions -- removes the joint sampler's state-noise from the ranking (the measured
        # cost SNR of joint scoring is ~0.1: action effect 1.14 vs draw variance 11.7).
        self.plan_score = str(kwargs.pop("plan_score", "joint"))
        assert self.plan_score in ("joint", "inpaint"), self.plan_score
        # plan_rollout H > 1: autoregressive imagination -- each unroll denoises the joint
        # chunk, keeps the FIRST frameskip action block, slides the imagined z (at +1 obs
        # step) into the latent history, H times. Cost scores the FINAL imagined state
        # against the goal; the winner's full H*frameskip raw actions execute, then replan.
        self.plan_rollout = int(kwargs.pop("plan_rollout", 1))
        # control: replace the K policy proposals by N(0,1) block sequences (random shooting on the
        # dynamics, same scoring/selection) -- measures how much the policy contributes to best_of_k
        self.plan_random_candidates = bool(kwargs.pop("plan_random_candidates", False))
        self.cem_iters = int(kwargs.pop("cem_iters", 3))
        self.cem_elites = int(kwargs.pop("cem_elites", 6))
        self.cem_std = float(kwargs.pop("cem_std", 0.5))
        # MoT CEM (Minghao 2026-09-04, "same ckpt: CEM vs grad-based"): cem_init=policy warm-starts the
        # Gaussian at the best of plan_k policy rollouts (same warm start as grad); cem_init=zero is the
        # pure world-model planner (LeWM-style, no policy prior).
        self.cem_init = str(kwargs.pop("cem_init", "policy"))
        assert self.cem_init in ("policy", "zero"), self.cem_init
        # oracle_bok (owner sanity check 2026-08-29): candidate 0 = the replayed demo's expert
        # chunk at the current step, K-1 = wrong chunks ("uniform" over the dataset action box or
        # "shuffle" = other demos' chunks); the dynamics picks; SR then measures whether the
        # world model as verifier recovers the demo's success when the expert IS in the set.
        self.oracle_cands = str(kwargs.pop("oracle_cands", "uniform"))
        self.expert_actions = kwargs.pop("expert_actions", None)   # per env: (T, adim) raw, from the start step
        # plan_mode=grad (MoT only): Adam on the z-scored H-block plan through the differentiable
        # rollout, warm-started from the best of plan_k policy rollouts; warm start = floor.
        self.grad_steps = int(kwargs.pop("grad_steps", 50))
        self.grad_lr = float(kwargs.pop("grad_lr", 0.05))
        self.grad_clip = float(kwargs.pop("grad_clip", 10.0))
        self.grad_tr = float(kwargs.pop("grad_tr", 0.0))                # penalty weight on ||U - U_warm||^2 (0 = off)
        self.grad_action_clip = float(kwargs.pop("grad_action_clip", 3.0))   # |z-scored action| bound (0 = off)
        self.pm_K = int(kwargs.pop("pm_K", 0))                    # SteerMPC on MoT: noise draws per env (0 = plan_k)
        self.pm_chunk = int(kwargs.pop("pm_chunk", 8))            # SteerMPC on MoT: envs per optimization batch (memory)
        # subgoal planning (owner 2026-08-30): cost target = the demo's frame subgoal_every raw
        # steps ahead of the env's current step (frames via policy_kwargs, fs-strided, uint8 HWC);
        # the model stays goal-blind -- the subgoal enters only the planner's cost.
        self.subgoal_every = int(kwargs.pop("subgoal_every", 0))
        self.subgoal_frames = kwargs.pop("subgoal_frames", None)
        # rollout scorers take the cost at each env's GOAL TIME (block H_i = remaining blocks,
        # capped at plan_rollout) instead of after the last imagined block
        self.plan_goal_time = bool(kwargs.pop("plan_goal_time", True))
        # plan_mode=extwm_bok (owner 2026-09-05, dynamics ablation): the SAME K policy proposals as
        # best_of_k (imagined through our dynamics), but GRADED by an external frozen LeWM world
        # model: its encoder on the last ext_wm_hist real frames + the goal frame, its predictor
        # rolled over the candidate action blocks (past executed blocks as the action history),
        # cost = ||z_lewm_terminal - z_lewm_goal||^2. Answers "is the jointly trained world model
        # needed to grade rollouts, or would any world model do?"
        self.ext_wm_path = str(kwargs.pop("ext_wm_path", ""))
        self.ext_wm_action_block_dim = int(kwargs.pop("ext_wm_action_block_dim", 0))
        self.ext_wm_embed_dim = int(kwargs.pop("ext_wm_embed_dim", 192))
        self.ext_wm_hist = int(kwargs.pop("ext_wm_hist", 3))
        self.ext = None; self._ext_frames = None; self._ext_acts = None
        super().__init__(model, cfg, *args, **kwargs)
        if self.plan_mode == "extwm_bok":
            assert self.ext_wm_path, "extwm_bok needs ext_wm_path (a LeWM checkpoint folder with config.json, or a .pt inside one)"
            self.ext, ext_cfg, ext_weights = _load_official_lewm(self.ext_wm_path)
            self.ext_wm_hist = int(ext_cfg.predictor.num_frames)
            abd = int(ext_cfg.action_encoder.input_dim)
            assert self.ext_wm_action_block_dim in (0, abd), \
                f"ext_wm_action_block_dim {self.ext_wm_action_block_dim} != checkpoint's {abd}"
            self.ext_wm_action_block_dim = abd
            print(f"[extwm] external LeWM {ext_weights} (hist {self.ext_wm_hist} frames, action block {abd}) grades the "
                  f"policy's best-of-{self.plan_k} proposals; our dynamics still imagines the proposals", flush=True)
        assert self.plan_mode in ("best_of_k", "cem", "steer", "oracle_bok", "grad", "extwm_bok"), self.plan_mode
        if self.subgoal_every > 0:
            assert self.plan_mode in ("best_of_k", "grad"), "subgoal cost: best_of_k / grad"
            assert self.subgoal_frames is not None, "subgoal planning needs the demo frames (eval_gip passes them)"
            assert self.expert_actions is None, "subgoal and oracle candidate modes both track raw_done; pick one"
            self._raw_done = None
        if self.plan_mode == "oracle_bok":
            assert self.expert_actions is not None, "oracle_bok needs the demo actions (eval_gip passes them)"
            assert self.oracle_cands in ("uniform", "shuffle"), self.oracle_cands
            allv = np.concatenate([np.asarray(a) for a in self.expert_actions], 0)
            self._act_lo, self._act_hi = allv.min(0), allv.max(0)
            self._raw_done = None
            self._oracle_rng = np.random.default_rng(self._flow_seed + 101)
            self._oracle_hits, self._oracle_n = 0, 0
            self._oracle_idx = None                # per env: replans so far in the episode
            self._oracle_ondemo = None             # per env: every pick so far was the expert
            self._oracle_tab = {}                  # (replan_idx, ondemo) -> [hits, n]
        assert self.model.num_states > 0, "planning needs imagined state tokens"
        # MoT (model.rollout_only): the dynamics is a one-block step, so planning IS the
        # autoregressive rollout to the goal time -- plan_rollout defaults to the goal horizon in
        # blocks, one block executes per replan, and the joint model's one-block scorers
        # (plan_score=inpaint, cem, steer) are refused (owner 2026-08-29).
        self.rollout_only = bool(getattr(model, "rollout_only", False))
        if self.plan_mode == "grad" and not getattr(model, "rollout_only", False):
            raise ValueError("plan_mode=grad is implemented on the MoT (rollout) dynamics only")
        if self.rollout_only:
            if self.plan_mode not in ("best_of_k", "oracle_bok", "grad", "steer", "cem", "extwm_bok"):
                raise ValueError(f"MoT plans by rollout only: plan_mode={self.plan_mode!r} has no rollout form")
            if self.plan_score != "joint":
                raise ValueError("MoT has no one-block scorer: drop plan_score (planning is the rollout)")
            if self.plan_rollout <= 1:
                self.plan_rollout = max(1, int(round(float(np.max(np.asarray(self.horizon0, dtype=float))))))
            if self.exec_actions == 0:
                self.exec_actions = self.action_block
            print(f"[plan] MoT rollout planning: {self.plan_rollout} blocks imagined per candidate, "
                  f"execute {self.exec_actions} raw actions per replan", flush=True)
        if self.plan_mode == "steer":
            assert getattr(model, "goal_conditioning", False), \
                "steer needs a goal pathway to bias (goal-conditioned checkpoint)"
        if self.plan_rollout > 1:
            assert self.plan_mode in ("best_of_k", "oracle_bok", "grad", "steer", "cem", "extwm_bok"), \
                "rollout planning: best_of_k / oracle_bok / grad / steer / cem / extwm_bok"
            if self.plan_mode in ("best_of_k", "grad", "steer", "cem", "extwm_bok"):
                # the plan is H*fs raw actions; execute all of it by default, then replan.
                # GR checkpoints are supported: h_norm counts down by one obs step per unroll.
                self.num_actions = self.plan_rollout * self.action_block
                if self.exec_actions == 0:
                    self.exec_actions = self.num_actions
        self.type = f"jointflow_plan_{self.plan_mode}"
        if getattr(model, "goal_conditioning", False):
            # GR checkpoints train on anchor-spaced history -> encode only at replans
            self._append_every_step = False
        self._prev_plan = None

    def set_env(self, env):
        super().set_env(env)
        self._prev_plan = [None] * getattr(env, "num_envs", 1)

    def _flush_env(self, i):
        super()._flush_env(i)
        if self._prev_plan is not None:
            self._prev_plan[i] = None
        if getattr(self, "_raw_done", None) is not None:
            self._raw_done[i] = 0
            if getattr(self, "_oracle_idx", None) is not None:
                self._oracle_idx[i] = 0; self._oracle_ondemo[i] = True

    @staticmethod
    def _final_cost(z_imag, z_goal):
        return ((z_imag[:, -1] - z_goal) ** 2).mean(-1)

    def _subgoal_latents(self, replan, device):
        """Encode, per replanning env, the demo frame subgoal_every raw steps ahead of the env's
        current step (clamped to the demo's last frame). Frames are fs-strided uint8 HWC."""
        if self._raw_done is None:
            self._raw_done = np.zeros(len(self.subgoal_frames), dtype=int)
        fs = self.action_block
        stats = spt.data.dataset_stats.ImageNet
        mean = torch.tensor(stats["mean"]).view(3, 1, 1)
        std = torch.tensor(stats["std"]).view(3, 1, 1)
        frames = []
        for i in replan:
            fr = self.subgoal_frames[i]
            idx = min(int(round((self._raw_done[i] + self.subgoal_every) / fs)), len(fr) - 1)
            x = torch.from_numpy(np.ascontiguousarray(fr[idx])).permute(2, 0, 1).float() / 255.0
            frames.append((x - mean) / std)
        return self._enc(torch.stack(frames).to(device))

    def _propose(self, info_dict, replan, history, history_pad):
        device = history.device
        if self.subgoal_every > 0:
            z_cost = self._subgoal_latents(replan, device)
            z_cond, h_norm = z_cost, None
            if getattr(self.model, "goal_conditioning", False):
                # goal-conditioned ckpts trained with a goal on every sample: condition the
                # policy on the env's real goal (goal_terminal convention, h_norm 0); the
                # subgoal enters only the planner's cost
                assert "goal" in info_dict, \
                    "subgoal planning on a goal-conditioned ckpt needs info_dict['goal']"
                goal = info_dict["goal"][replan]
                g_obs = goal[:, -1] if goal.ndim == 5 else goal
                z_cond = self._enc(g_obs.to(device).float())
                h_norm = torch.zeros(len(replan), device=device, dtype=torch.float32)
            plan = (self._gradient_plan(history, history_pad, z_cond, None, cost_goal=z_cost)
                    if self.plan_mode == "grad"
                    else self._best_of_k(history, history_pad, z_cond, h_norm,
                                         cost_goal=z_cost))
            take = self._n_actions_to_execute()
            for i in replan:
                self._raw_done[i] += take
            if self._prev_plan is not None:
                for row, i in enumerate(replan):
                    self._prev_plan[i] = plan[row]
            return plan
        assert "goal" in info_dict, "jointflow_plan eval needs info_dict['goal'] (goal-reaching)"
        goal = info_dict["goal"][replan]
        g_obs = goal[:, -1] if goal.ndim == 5 else goal
        z_goal = self._enc(g_obs.to(device).float())
        h_norm = steps_t = None
        if getattr(self.model, "goal_conditioning", False):
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=device, dtype=torch.float32)
            steps_t = torch.tensor(steps, device=device, dtype=torch.float32)
        if self.plan_mode == "best_of_k":
            plan = self._best_of_k(history, history_pad, z_goal, h_norm, steps=steps_t)
        elif self.plan_mode == "extwm_bok":
            plan = self._best_of_k_external_grader(info_dict, replan, history, history_pad, z_goal, steps_t, g_obs)
        elif self.plan_mode == "oracle_bok":
            plan = self._best_of_k_oracle(history, history_pad, z_goal, h_norm, replan, steps=steps_t)
        elif self.plan_mode == "grad":
            plan = self._gradient_plan(history, history_pad, z_goal, steps_t)
        elif self.plan_mode == "steer":
            plan = (self._steer_rollout(history, history_pad, z_goal, steps_t) if self.rollout_only
                    else self._steer(history, history_pad, z_goal, h_norm))
        else:
            plan = (self._cem_plan(history, history_pad, z_goal, steps_t) if self.rollout_only
                    else self._cem(history, history_pad, z_goal, h_norm, replan))
        if self._prev_plan is not None:
            for row, i in enumerate(replan):
                self._prev_plan[i] = plan[row]
        return plan

    def _flush_env(self, i):
        super()._flush_env(i)
        if self._ext_frames is not None:
            self._ext_frames[i].clear(); self._ext_acts[i].clear()

    def _best_of_k_external_grader(self, info_dict, replan, history, history_pad, z_goal, steps, g_obs):
        """best_of_k proposals (policy chunks imagined through OUR dynamics, as in _best_of_k's
        rollout path), selected by an EXTERNAL frozen LeWM world model's terminal cost. The LeWM
        sees the last ext_wm_hist real frames (its own encoder), the executed past action blocks
        as action history, and each candidate's H blocks (time-major (fs*adim) blocks, z-scored
        with the dataset stats both models share); cost = MSE(z_lewm_after_H, z_lewm_goal).
        Logs how often the LeWM pick coincides with our dynamics' pick and its rank under our cost."""
        R, K, fs, H = history.shape[0], self.plan_k, self.action_block, self.plan_rollout
        adim = self.model.action_raw_dim
        device = history.device
        num_envs = self.env.num_envs
        if self._ext_frames is None:
            self._ext_frames = [deque(maxlen=self.ext_wm_hist) for _ in range(num_envs)]
            self._ext_acts = [deque(maxlen=max(self.ext_wm_hist - 1, 0)) for _ in range(num_envs)]
        if next(self.ext.parameters()).device != torch.device(device):
            self.ext = self.ext.to(device)
        # current real frame per replanning env -> the LeWM frame history (one frame per replan = one block)
        cur = info_dict["pixels"][replan]
        cur = cur[:, -1] if cur.ndim == 5 else cur
        for row, i in enumerate(replan):
            self._ext_frames[i].append(cur[row].detach().float().cpu())
        with torch.no_grad():
            # 1) proposals + our dynamics' own cost (identical to best_of_k)
            h = history.repeat_interleave(K, 0); p = history_pad.repeat_interleave(K, 0)
            goal_rep = z_goal.repeat_interleave(K, 0)
            steps_rep = steps.repeat_interleave(K, 0) if steps is not None else None
            blocks, z_final = self._imagine_rollout(h, p, goal_rep, steps_rep, n_dyn_steps=H)   # the external grader scores the whole plan
            cost_ours = ((z_final - goal_rep) ** 2).mean(-1).view(R, K)
            # 2) LeWM grading
            Hf = self.ext_wm_hist; A = fs * adim
            frames = []; pasts = []
            for i in replan:
                fr = list(self._ext_frames[i])
                fr = [fr[0]] * (Hf - len(fr)) + fr                                    # pad early episodes with the oldest frame
                frames.append(torch.stack(fr))                                        # (Hf, C, H, W)
                pa = list(self._ext_acts[i])
                pa = [torch.zeros(A)] * ((Hf - 1) - len(pa)) + pa                      # zero (= mean action) padding
                pasts.append(torch.stack(pa) if pa else torch.zeros(0, A))            # (Hf-1, A)
            px = torch.stack(frames).to(device).float()                              # (R, Hf, C, H, W)
            past = torch.stack(pasts).to(device).float()                             # (R, Hf-1, A)
            cand = blocks.reshape(R, K, H, A)                                          # (R, K, H, fs*adim) time-major blocks
            act_seq = torch.cat([past[:, None].expand(R, K, Hf - 1, A), cand], dim=2)  # (R, K, Hf-1+H, A)
            info = {"pixels": px[:, None].expand(R, K, Hf, *px.shape[2:])}
            out = self.ext.rollout(info, act_seq, history_size=Hf)
            term = out["predicted_emb"][:, :, -1]                                      # (R, K, D) after the H blocks
            g_ext = self.ext.encode({"pixels": g_obs.to(device).float()[:, None]})["emb"][:, 0]   # (R, D)
            cost_ext = ((term - g_ext[:, None]) ** 2).mean(-1)                         # (R, K)
            pick = cost_ext.argmin(1)
            rows = torch.arange(R, device=device)
            pick_ours = cost_ours.argmin(1)
            rank = (cost_ours < cost_ours[rows, pick][:, None]).sum(1).float()         # 0 = LeWM's pick is also our best
            st = self.__dict__.setdefault("_extwm_stats", dict(n=0, agree=0, rank=0.0))
            st["n"] += R; st["agree"] += int((pick == pick_ours).sum()); st["rank"] += float(rank.sum())
            if st["n"] % 50 < R:
                print(f"[extwm] LeWM pick == our-dynamics pick on {st['agree']}/{st['n']} replans; "
                      f"mean rank of the LeWM pick under our cost {st['rank'] / st['n']:.2f} (of {K})", flush=True)
            plan = blocks.reshape(R, K, H * fs, adim)[rows, pick]                       # (R, H*fs, adim) z-scored
            # the executed first block becomes LeWM action history for the next replan
            for row, i in enumerate(replan):
                self._ext_acts[i].append(plan[row, :fs].reshape(-1).detach().cpu())
        return plan

    def _goal_step_idx(self, goal_horizon, n_rows_per_env=1):
        """Index of the predicted state at each proposal's goal horizon.

        goal_horizon: dynamics steps left until the goal must be reached, one float per proposal,
        or None. Returns clamp(round(goal_horizon), 1, full plan length) - 1, or None when the
        planning cost is taken at the last predicted state (plan_goal_time off, or no horizon)."""
        n_dyn_steps_full = self.plan_rollout
        if not self.plan_goal_time or goal_horizon is None:
            return None
        goal_step_idx = goal_horizon.round().clamp(min=1.0, max=float(n_dyn_steps_full)).long() - 1
        return goal_step_idx.repeat_interleave(n_rows_per_env, 0) if n_rows_per_env > 1 else goal_step_idx

    def _n_dyn_steps_needed(self, goal_horizon):
        """Dynamics steps to imagine in one replan: the largest goal horizon in the batch, never
        fewer than the action chunks that will be executed. The dynamics is autoregressive, so
        predicted states past the goal horizon cannot change the planning cost. The full plan
        length when the cost is taken at the last step."""
        n_dyn_steps_full = self.plan_rollout
        if not self.plan_goal_time or goal_horizon is None:
            return n_dyn_steps_full
        n_dyn_steps_to_goal = int(self._goal_step_idx(goal_horizon).max().item()) + 1
        n_chunks_executed = (self._n_actions_to_execute() + self.action_block - 1) // self.action_block
        return max(1, min(n_dyn_steps_full, max(n_dyn_steps_to_goal, n_chunks_executed)))

    @staticmethod
    def _pred_state_at(pred_states_per_step, step_idx):
        """Pick one predicted state per proposal: pred_states_per_step (proposals, steps, z_dim)
        -> (proposals, z_dim) at each proposal's step_idx (None = the last step)."""
        if step_idx is None:
            return pred_states_per_step[:, -1]
        return pred_states_per_step[torch.arange(pred_states_per_step.shape[0], device=pred_states_per_step.device), step_idx]

    def _imagine_rollout(self, state_history, history_pad, goal_state, goal_horizon,
                         candidate_actions=None, n_dyn_steps=None):
        """Predict the state trajectory of every proposal (candidate plan) and return the predicted
        state at each proposal's goal horizon, where the planning cost is measured.

        Tensors are stacked over all proposals of all episodes, episode-major (entry = episode *
        n_proposals + proposal). One dynamics step predicts the next state from the state history
        and one action chunk. The chunk comes from candidate_actions[:, step] when proposals are
        given (CEM, gradient and oracle planning; an episode's proposals share one noise draw),
        otherwise from a fresh policy sample on the predicted history (best-of-K). Each predicted
        state is appended to the history; goal-conditioned checkpoints see the goal horizon shrink
        by one per step. Steps predicted: n_dyn_steps, default _n_dyn_steps_needed. The noise for
        every step of a full plan is drawn first, in plan order, so the seeded generator advances
        exactly as for a full rollout however many steps are predicted.
        Returns (act_chunks_per_step (total_proposals, n_dyn_steps, actions_per_chunk, action_dim),
        pred_state_at_goal (total_proposals, z_dim))."""
        actions_per_chunk, n_proposals = self.action_block, self.plan_k
        total_proposals = state_history.shape[0]
        n_episodes = total_proposals // n_proposals
        device = state_history.device
        n_actions_pred, action_dim = self.model.num_actions, self.model.action_raw_dim
        n_states_pred, z_dim = self.model.num_states, self.model.z_dim
        is_goal_conditioned = getattr(self.model, "goal_conditioning", False)
        noise_generator = self._flow_generator(device)
        n_proposal_steps = 0 if candidate_actions is None else candidate_actions.shape[1]
        assert candidate_actions is None or n_proposal_steps == self.plan_rollout, \
            "candidate_actions must cover the whole plan"
        n_chunk_repeats = (n_actions_pred + actions_per_chunk - 1) // actions_per_chunk
        n_dyn_steps = self._n_dyn_steps_needed(goal_horizon) if n_dyn_steps is None else int(n_dyn_steps)
        self._last_n_dyn_steps = n_dyn_steps          # read by the evaluation timer

        noise_per_dyn_step = []
        for step in range(self.plan_rollout):
            if step < n_proposal_steps:
                act_noise = torch.randn(n_episodes, 1, n_actions_pred, action_dim, device=device, generator=noise_generator) \
                    .expand(n_episodes, n_proposals, n_actions_pred, action_dim).reshape(total_proposals, n_actions_pred, action_dim)
                state_noise = torch.randn(n_episodes, 1, n_states_pred, z_dim, device=device, generator=noise_generator) \
                    .expand(n_episodes, n_proposals, n_states_pred, z_dim).reshape(total_proposals, n_states_pred, z_dim)
                noise_per_dyn_step.append((act_noise, state_noise))
            else:
                noise_per_dyn_step.append(self.model.draw_sample_noise(total_proposals, device, generator=noise_generator))

        act_chunks_per_step, pred_states_per_step = [], []
        for step in range(n_dyn_steps):
            act_noise, state_noise = noise_per_dyn_step[step]
            goal_inputs = dict()
            if is_goal_conditioned:
                # goal_terminal checkpoints train with h_norm 0 and get no horizon
                h_norm = torch.zeros(total_proposals, device=device) if goal_horizon is None else \
                    (goal_horizon - step).clamp(min=1.0).clamp(max=float(self.H_max)) / float(self.H_max)
                goal_inputs = dict(z_goal=goal_state, h_norm=h_norm)
            if step < n_proposal_steps:
                curr_act_chunk = candidate_actions[:, step]
                next_act_chunk = candidate_actions[:, step + 1] if step + 1 < n_proposal_steps else curr_act_chunk
                dyn_act_input = torch.cat([curr_act_chunk] + [next_act_chunk] * (n_chunk_repeats - 1), 1)[:, :n_actions_pred]
                if self.rollout_only:
                    pred_states = self.model.predict_state(state_history, history_pad, dyn_act_input,
                                                        noise_state=state_noise, **goal_inputs)
                else:
                    pred_states = self.model.sample_inpaint(state_history, history_pad, dyn_act_input, noise_action=act_noise,
                                                         noise_state=state_noise, **goal_inputs)
            else:
                with torch.no_grad():   # model.sample with the pre-drawn noise
                    sampled, pred_states = self.model._sample_impl(state_history, history_pad, None, goal_inputs.get("z_goal"),
                                                                goal_inputs.get("h_norm"), noise_action=act_noise,
                                                                noise_state=state_noise)
                curr_act_chunk = sampled[:, :actions_per_chunk]
            act_chunks_per_step.append(curr_act_chunk)
            next_pred_state = pred_states[:, :1]
            pred_states_per_step.append(next_pred_state.squeeze(1))
            state_history = torch.cat([state_history[:, 1:], next_pred_state], dim=1)
            history_pad = torch.cat([history_pad[:, 1:], torch.zeros_like(history_pad[:, :1])], dim=1)
        return torch.stack(act_chunks_per_step, 1), self._pred_state_at(torch.stack(pred_states_per_step, 1), self._goal_step_idx(goal_horizon))

    def _best_of_k_oracle(self, history, history_pad, z_goal, h_norm, replan, steps=None):
        """Expert sequence vs K-1 wrong sequences, scored by the dynamics: one block ahead by
        inpaint (plan_rollout 1: chunks of num_actions raw actions, shared noise) or by rollout to
        the goal time (plan_rollout H: sequences of H*fs raw actions, every block imagined from
        the given actions, no policy continuation)."""
        device = history.device
        R, K = history.shape[0], self.plan_k
        adim = self.model.action_raw_dim
        A = self.plan_rollout * self.action_block if self.plan_rollout > 1 else self.model.num_actions
        if self._raw_done is None:
            self._raw_done = np.zeros(len(self.expert_actions), dtype=int)
            self._oracle_idx = np.zeros(len(self.expert_actions), dtype=int)
            self._oracle_ondemo = np.ones(len(self.expert_actions), dtype=bool)
        amean = self._amean.cpu().numpy(); astd = self._astd.cpu().numpy()
        cands = np.zeros((R, K, A, adim), np.float32)
        for row, i in enumerate(replan):
            ea = np.asarray(self.expert_actions[i], np.float32)
            t0 = int(self._raw_done[i])
            chunk = ea[t0:t0 + A]
            if len(chunk) < A:                                    # past the demo's end: hold the last action
                last = ea[-1:] if len(ea) else np.zeros((1, adim), np.float32)
                chunk = np.concatenate([chunk, np.repeat(last, A - len(chunk), 0)], 0)
            cands[row, 0] = chunk
            if self.oracle_cands == "uniform":
                cands[row, 1:] = self._oracle_rng.uniform(self._act_lo, self._act_hi, (K - 1, A, adim))
            else:                                                  # other demos' chunks (on-manifold, wrong state)
                for k in range(1, K):
                    j = int(self._oracle_rng.integers(len(self.expert_actions)))
                    src = np.asarray(self.expert_actions[j], np.float32)
                    if len(src) < A:
                        src = np.concatenate([src, np.repeat(src[-1:], A - len(src), 0)], 0)
                    s0 = int(self._oracle_rng.integers(0, len(src) - A + 1))
                    cands[row, k] = src[s0:s0 + A]
        cz = torch.from_numpy((cands - amean) / astd).to(device).reshape(R * K, A, adim)
        h = history.repeat_interleave(K, 0); p = history_pad.repeat_interleave(K, 0)
        goal_rep = z_goal.repeat_interleave(K, 0)
        gen = self._flow_generator(device)
        Am = self.model.num_actions
        noise_a = torch.randn(R, 1, Am, adim, device=device, generator=gen).expand(R, K, Am, adim).reshape(R * K, Am, adim)
        S, D = self.model.num_states, self.model.z_dim
        noise_s = torch.randn(R, 1, S, D, device=device, generator=gen).expand(R, K, S, D).reshape(R * K, S, D)
        if self.plan_rollout > 1:
            fs = self.action_block
            steps_rep = steps.repeat_interleave(K, 0) if steps is not None else None
            _, z_final = self._imagine_rollout(h, p, goal_rep, steps_rep, candidate_actions=cz.view(R * K, self.plan_rollout, fs, adim))
            cost = ((z_final - goal_rep) ** 2).mean(-1).view(R, K)
        else:
            z_imag = self.model.sample_inpaint(h, p, cz, noise_action=noise_a, noise_state=noise_s,
                                               **self._cond_args(z_goal, h_norm, K))
            cost = self._final_cost(z_imag, goal_rep).view(R, K)
        pick = cost.argmin(1)
        self._oracle_hits += int((pick == 0).sum()); self._oracle_n += R
        take = self._n_actions_to_execute()
        pick_np = pick.cpu().numpy()
        for row, i in enumerate(replan):
            key = (int(self._oracle_idx[i]), bool(self._oracle_ondemo[i]))
            cell = self._oracle_tab.setdefault(key, [0, 0])
            cell[0] += int(pick_np[row] == 0); cell[1] += 1
            self._oracle_idx[i] += 1
            if pick_np[row] != 0:
                self._oracle_ondemo[i] = False
            self._raw_done[i] += take
        if self._oracle_n % 50 < R:
            print(f"[oracle] expert picked {self._oracle_hits}/{self._oracle_n} = "
                  f"{100.0 * self._oracle_hits / max(1, self._oracle_n):.1f}% of replans "
                  f"(cands={self.oracle_cands}, K={K}, score="
                  f"{'rollout%d' % self.plan_rollout if self.plan_rollout > 1 else 'inpaint'})", flush=True)
            tab = sorted(self._oracle_tab.items())
            print("[oracle] by replan index (on-demo = every earlier pick was the expert): " +
                  "  ".join(f"r{k}{'*' if od else ''}={h}/{n}" for (k, od), (h, n) in tab), flush=True)
        return cz.view(R, K, A, adim)[torch.arange(R, device=device), pick]

    def _steer_rollout(self, history, history_pad, z_goal, steps):
        """SteerMPC on MoT over all replanning envs, optimized pm_chunk envs at a time (each env's
        delta is independent; the chunking only bounds the autograd graph)."""
        R = history.shape[0]
        chunk = max(1, self.pm_chunk)
        if getattr(self.model, "state_head", "mse") == "flow":
            chunk = max(1, chunk // 4)          # the state flow doubles the passes per block in the graph
        plans = []
        for lo in range(0, R, chunk):
            hi = min(R, lo + chunk)
            plans.append(self._steer_plan(history[lo:hi], history_pad[lo:hi], z_goal[lo:hi],
                                                   None if steps is None else steps[lo:hi]))
        return torch.cat(plans, 0)

    def _steer_plan(self, history, history_pad, z_goal, steps):
        """SteerMPC on MoT: Adam on a z_dim bias delta added to the goal latent fed to the POLICY,
        through the frozen flow sampler and the dynamics rollout (plan_rollout blocks). K noise
        realizations per env share one delta; the noise is fixed per replan, so delta -> cost is
        deterministic. The state stream never sees the goal (MoT mask), so the dynamics and the
        cost (vs the TRUE goal) are goal-blind; only behaviour selection is steered. Iterate 0
        (delta=0) is the pure roll planner and stays in the set; the best-cost (candidate,
        iterate) plan is returned. ||delta|| is capped at pm_rho*||z_goal||."""
        assert getattr(self.model, "goal_conditioning", False), "steer needs a goal-conditioned policy"
        R, fs, H = history.shape[0], self.action_block, self.plan_rollout
        K = self.pm_K if self.pm_K > 0 else self.plan_k
        A, adim = self.model.num_actions, self.model.action_raw_dim
        S, D = self.model.num_states, self.model.z_dim
        device = history.device
        gen = self._flow_generator(device)
        steps_t = steps if steps is not None else torch.full((R,), float(H), device=device)
        h0 = history.repeat_interleave(K, 0); p0 = history_pad.repeat_interleave(K, 0)
        goal_rep = z_goal.repeat_interleave(K, 0); steps_rep = steps_t.repeat_interleave(K, 0)
        noise_a = [torch.randn(R * K, A, adim, device=device, generator=gen) for _ in range(H)]
        noise_s = [torch.randn(R * K, S, D, device=device, generator=gen) for _ in range(H)]
        gnorm = z_goal.norm(dim=-1, keepdim=True)
        rows = torch.arange(R, device=device)

        gidx = self._goal_step_idx(steps_rep)
        n_dyn_steps = self._n_dyn_steps_needed(steps_rep)   # the noise lists above cover the whole plan: same draws as a full rollout
        self._last_n_dyn_steps = n_dyn_steps

        def rollout(delta):
            zg = goal_rep + delta.repeat_interleave(K, 0)
            h, p, zs, blocks = h0, p0, [], []
            for k in range(n_dyn_steps):
                hk = (steps_rep - k).clamp(min=1.0).clamp(max=float(self.H_max)) / float(self.H_max)
                action, z_imag = self.model._sample_impl(h, p, z_goal=zg, h_norm=hk,
                                                         noise_action=noise_a[k], noise_state=noise_s[k])
                blocks.append(action[:, :fs])
                zs.append(z_imag[:, 0])
                h = torch.cat([h[:, 1:], z_imag[:, :1]], dim=1)
                p = torch.cat([p[:, 1:], torch.zeros_like(p[:, :1])], dim=1)
            z = self._pred_state_at(torch.stack(zs, 1), gidx)                          # latent at the goal time
            cost = ((z - goal_rep) ** 2).mean(-1).view(R, K)              # vs the TRUE goal
            return torch.stack(blocks, 1).view(R, K, n_dyn_steps, fs, adim), cost

        with torch.enable_grad():
            delta = torch.zeros(R, D, device=device, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=self.pm_lr)
            best_plan = best_cost = c0 = None
            for it in range(self.pm_steps + 1):
                blocks, cost = rollout(delta)
                c, pick = cost.detach().min(1)                                # best candidate per env
                cand = blocks.detach()[rows, pick]                            # (R, H, fs, adim)
                if best_cost is None:
                    best_cost, best_plan, c0 = c.clone(), cand.clone(), c.clone()
                else:
                    better = c < best_cost
                    best_plan[better] = cand[better]
                    best_cost = torch.where(better, c, best_cost)
                if it == self.pm_steps:
                    break
                grad = torch.autograd.grad(cost.mean(), delta)[0]
                if self.pm_random:
                    r = torch.randn(R, D, device=device, generator=gen)
                    grad = r * (grad.norm(dim=-1, keepdim=True) / r.norm(dim=-1, keepdim=True).clamp_min(1e-9))
                opt.zero_grad(set_to_none=True)
                delta.grad = grad
                opt.step()
                with torch.no_grad():
                    scale = (self.pm_rho * gnorm / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
                    delta.mul_(scale)
        st = self.__dict__.setdefault("_steer_stats", dict(n=0, improved=0, c0=0.0, cb=0.0, dn=0.0))
        st["n"] += R; st["improved"] += int((best_cost < c0).sum())
        st["c0"] += float(c0.sum()); st["cb"] += float(best_cost.sum())
        st["dn"] += float((delta.detach().norm(dim=-1) / gnorm.squeeze(-1).clamp_min(1e-9)).sum())
        if st["n"] % 50 < R:
            print(f"[steer] {st['improved']}/{st['n']} replans improved on iterate 0 (best of {K}); mean cost "
                  f"{st['c0'] / st['n']:.4f} -> {st['cb'] / st['n']:.4f}; mean ||delta||/||z_goal|| "
                  f"{st['dn'] / st['n']:.3f} (steps {self.pm_steps}, lr {self.pm_lr}, rho {self.pm_rho})", flush=True)
        return best_plan.reshape(R, best_plan.shape[1] * fs, adim)

    def _rollout_cost_differentiable(self, history, history_pad, z_goal, steps, U, noise_s=None,
                           cost_goal=None):
        """Planning cost of a plan U (episodes, steps, chunk_len, action_dim), differentiable in U:
        imagine the plan step by step and measure the distance from the latent at the goal horizon
        (`steps`) to the goal latent (cost_goal when the scoring goal differs from the conditioning
        goal). noise_s: per-step state noise for a flow state head, fixed per replan."""
        h, p = history, history_pad
        gc = getattr(self.model, "goal_conditioning", False)
        zs = []
        for k in range(U.shape[1]):
            cond = dict()
            if gc:
                hk = torch.zeros(h.shape[0], device=h.device) if steps is None else \
                    (steps - k).clamp(min=1.0).clamp(max=float(self.H_max)) / float(self.H_max)
                cond = dict(z_goal=z_goal, h_norm=hk)
            z_imag = self.model.imagine_step(h, p, U[:, k], noise_state=None if noise_s is None else noise_s[k], **cond)
            zs.append(z_imag[:, 0])
            h = torch.cat([h[:, 1:], z_imag[:, :1]], dim=1)
            p = torch.cat([p[:, 1:], torch.zeros_like(p[:, :1])], dim=1)
        z = self._pred_state_at(torch.stack(zs, 1), self._goal_step_idx(steps))
        tgt = cost_goal if cost_goal is not None else z_goal
        return ((z - tgt) ** 2).mean(-1)

    def _gradient_plan(self, history, history_pad, z_goal, steps, cost_goal=None):
        """Gradient planning: start from the best of plan_k policy proposals (best-of-K), then refine
        the whole plan with grad_steps Adam steps on the planning cost, with an optional trust region
        around the start (grad_tr). Returns the lowest-cost plan seen, the start included."""
        R, K, fs, H = history.shape[0], self.plan_k, self.action_block, self.plan_rollout
        device = history.device
        steps_t = steps          # None = goal_terminal ckpts (h_norm 0, cost at the last block)
        with torch.no_grad():
            h = history.repeat_interleave(K, 0); p = history_pad.repeat_interleave(K, 0)
            goal_rep = z_goal.repeat_interleave(K, 0)
            blocks, z_final = self._imagine_rollout(
                h, p, goal_rep, steps_t.repeat_interleave(K, 0) if steps_t is not None else None)
            cg_rep = cost_goal.repeat_interleave(K, 0) if cost_goal is not None else goal_rep
            cost0 = ((z_final - cg_rep) ** 2).mean(-1).view(R, K)
            pick = cost0.argmin(1)
            rows = torch.arange(R, device=device)
            U0 = blocks.view(R, K, blocks.shape[1], fs, -1)[rows, pick].detach()   # (R, n, fs, adim)
            c_warm = cost0[rows, pick].detach()
        if self.grad_steps <= 0:
            return U0.reshape(R, U0.shape[1] * fs, -1)
        S, D = self.model.num_states, self.model.z_dim
        gen = self._flow_generator(device)
        noise_s = [torch.randn(R, S, D, device=device, generator=gen) for _ in range(H)]   # flow state head: fixed per replan
        best_U, best_c = U0.clone(), c_warm.clone()
        U = U0.clone().requires_grad_(True)
        opt = torch.optim.Adam([U], lr=self.grad_lr)
        with torch.enable_grad():
            for it in range(self.grad_steps + 1):
                c = self._rollout_cost_differentiable(history, history_pad, z_goal, steps_t, U, noise_s,
                                            cost_goal=cost_goal)
                with torch.no_grad():
                    better = c < best_c
                    best_c = torch.where(better, c.detach(), best_c)
                    best_U[better] = U.detach()[better]
                if it == self.grad_steps:
                    break
                loss = c.sum()
                if self.grad_tr > 0:
                    loss = loss + self.grad_tr * ((U - U0) ** 2).sum()
                opt.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([U], self.grad_clip)
                opt.step()
                if self.grad_action_clip > 0:
                    with torch.no_grad():
                        U.clamp_(-self.grad_action_clip, self.grad_action_clip)
        # diagnostics: how often and by how much the refinement beats the policy's best rollout
        st = self.__dict__.setdefault("_grad_stats", dict(n=0, improved=0, c_warm=0.0, c_best=0.0, du=0.0))
        st["n"] += R; st["improved"] += int((best_c < c_warm).sum())
        st["c_warm"] += float(c_warm.sum()); st["c_best"] += float(best_c.sum())
        st["du"] += float((best_U - U0).flatten(1).norm(dim=1).sum())
        if st["n"] % 50 < R:
            print(f"[grad] {st['improved']}/{st['n']} replans improved on the warm start; mean cost "
                  f"{st['c_warm'] / st['n']:.4f} -> {st['c_best'] / st['n']:.4f}; mean ||U-U0|| "
                  f"{st['du'] / st['n']:.3f} (steps {self.grad_steps}, lr {self.grad_lr}, tr {self.grad_tr})", flush=True)
        return best_U.detach().reshape(R, best_U.shape[1] * fs, -1)

    def _cem_plan(self, history, history_pad, z_goal, steps, cost_goal=None):
        """Cross-entropy-method planning on the planning cost: cem_iters rounds of plan_k Gaussian
        candidate plans around a running mean (candidate 0 is the mean itself); the cem_elites best
        refit the mean and std. The mean starts at the best policy proposal (cem_init=policy) or at
        zero (cem_init=zero, world model only). Returns the lowest-cost candidate seen."""
        R, P, fs, H = history.shape[0], self.plan_k, self.action_block, self.plan_rollout
        adim = self.model.action_raw_dim
        device = history.device
        steps_t = steps
        h = history.repeat_interleave(P, 0); p = history_pad.repeat_interleave(P, 0)
        goal_rep = z_goal.repeat_interleave(P, 0)
        steps_rep = steps_t.repeat_interleave(P, 0) if steps_t is not None else None
        cg_rep = cost_goal.repeat_interleave(P, 0) if cost_goal is not None else goal_rep
        rows = torch.arange(R, device=device)
        with torch.no_grad():
            if self.cem_init == "policy":
                blocks, z_final = self._imagine_rollout(h, p, goal_rep, steps_rep)
                cost0 = ((z_final - cg_rep) ** 2).mean(-1).view(R, P)
                pick = cost0.argmin(1)
                mean = blocks.view(R, P, blocks.shape[1], fs, adim)[rows, pick].clone()   # (R, n, fs, adim)
                best_plan, best_cost = mean.clone(), cost0[rows, pick].clone()
            else:
                mean = torch.zeros(R, H, fs, adim, device=device)
                best_plan, best_cost = mean.clone(), torch.full((R,), float("inf"), device=device)
            if mean.shape[1] < H:
                # candidates always span the whole plan, so the random draws match a full rollout;
                # the part past the imagined steps is never scored or executed
                unpredicted_steps_pad = mean.new_zeros(R, H - mean.shape[1], fs, adim)
                mean, best_plan = torch.cat([mean, unpredicted_steps_pad], 1), torch.cat([best_plan, unpredicted_steps_pad], 1)
            std = torch.full_like(mean, self.cem_std)
            for _ in range(self.cem_iters):
                cand = mean[:, None] + std[:, None] * torch.randn(R, P, H, fs, adim, device=device,
                                                                  generator=self._flow_generator(device))
                cand[:, 0] = mean
                _, z_final = self._imagine_rollout(h, p, goal_rep, steps_rep, candidate_actions=cand.reshape(R * P, H, fs, adim))
                cost = ((z_final - cg_rep) ** 2).mean(-1).view(R, P)
                iter_cost, iter_pick = cost.min(1)
                better = iter_cost < best_cost
                best_plan[better] = cand[rows, iter_pick][better]
                best_cost = torch.minimum(iter_cost, best_cost)
                elite_idx = cost.topk(min(self.cem_elites, P), dim=1, largest=False).indices
                elites = cand[rows[:, None], elite_idx]
                mean, std = elites.mean(1), elites.std(1, correction=0).clamp_min(0.02)
        st = self.__dict__.setdefault("_cem_stats", dict(n=0, c0=0.0, c=0.0))
        st["n"] += R; st["c"] += float(best_cost.sum())
        if self.cem_init == "policy":
            st["c0"] += float(cost0[rows, pick].sum())
        if st["n"] % 50 < R:
            print(f"[cem] init={self.cem_init} iters={self.cem_iters} elites={self.cem_elites} std0={self.cem_std}: "
                  f"mean best cost {st['c'] / st['n']:.4f}" + (f" (warm start {st['c0'] / st['n']:.4f})" if self.cem_init == "policy" else ""), flush=True)
        return best_plan.reshape(R, best_plan.shape[1] * fs, adim)

    def _cond_args(self, z_goal, h_norm, repeat):
        if not getattr(self.model, "goal_conditioning", False):
            return dict()
        return dict(z_goal=z_goal.repeat_interleave(repeat, 0),
                    h_norm=h_norm.repeat_interleave(repeat, 0) if h_norm is not None else None)

    def _best_of_k(self, history, history_pad, z_goal, h_norm=None, steps=None, cost_goal=None):
        R, K = history.shape[0], self.plan_k
        h = history.repeat_interleave(K, 0)
        p = history_pad.repeat_interleave(K, 0)
        goal_rep = z_goal.repeat_interleave(K, 0)
        if self.plan_rollout <= 1:
            action, z_imag = self.model.sample(h, p, generator=self._flow_generator(h.device),
                                               **self._cond_args(z_goal, h_norm, K))
            if self.plan_score == "inpaint":
                A, adim = self.model.num_actions, self.model.action_raw_dim
                gen = self._flow_generator(h.device)
                noise_a = torch.randn(R, 1, A, adim, device=h.device, generator=gen) \
                    .expand(R, K, A, adim).reshape(R * K, A, adim)
                noise_s = torch.randn(R, 1, self.model.num_states, self.model.z_dim,
                                      device=h.device, generator=gen) \
                    .expand(R, K, self.model.num_states, self.model.z_dim) \
                    .reshape(R * K, self.model.num_states, self.model.z_dim)
                z_imag = self.model.sample_inpaint(h, p, action, noise_action=noise_a,
                                                   noise_state=noise_s,
                                                   **self._cond_args(z_goal, h_norm, K))
            cost = self._final_cost(z_imag, goal_rep).view(R, K)
            pick = cost.argmin(1)
            return action.view(R, K, *action.shape[1:])[torch.arange(R, device=action.device), pick]
        # autoregressive imagination (owner's MoT planner): policy proposes, the first block
        # goes through the dynamics, the imagined z slides into the history, plan_rollout
        # times to the goal time; cost on the final imagined z (see _imagine).
        fs = self.action_block
        gc = getattr(self.model, "goal_conditioning", False)
        steps_rep = steps.repeat_interleave(K, 0) if gc and steps is not None else None
        random_proposals = None
        if self.plan_random_candidates:
            random_proposals = torch.randn(h.shape[0], self.plan_rollout, fs, self.model.action_raw_dim,
                                device=h.device, generator=self._flow_generator(h.device))
        blocks, z_final = self._imagine_rollout(h, p, goal_rep, steps_rep, candidate_actions=random_proposals)
        # cost_goal (subgoal planning on goal-conditioned ckpts): the policy is conditioned on
        # z_goal, the planner scores against cost_goal
        cg_rep = cost_goal.repeat_interleave(K, 0) if cost_goal is not None else goal_rep
        cost = ((z_final - cg_rep) ** 2).mean(-1).view(R, K)
        plan = blocks.reshape(R, K, blocks.shape[1] * fs, -1)
        pick = cost.argmin(1)
        return plan[torch.arange(R, device=plan.device), pick]

    def _steer(self, history, history_pad, z_goal, h_norm):
        """Prompt-MPC on flow: Adam on a z_dim goal bias delta through the frozen sampler.
        One fixed noise realization per replan (sampler noise AND scoring noise), so the
        map delta -> cost is deterministic and the gradient is not noise-dominated; all
        draws come from the eval-seeded generator. Actions always come from the sampler
        under z_goal+delta (on-manifold); the cost re-imagines them via inpaint under the
        TRUE z_goal. Iterate 0 is delta=0; the best-cost iterate's plan executes."""
        R = history.shape[0]
        A, adim = self.model.num_actions, self.model.action_raw_dim
        S, D = self.model.num_states, self.model.z_dim
        device = history.device
        gen = self._flow_generator(device)
        noise_a = torch.randn(R, A, adim, device=device, generator=gen)
        noise_s = torch.randn(R, S, D, device=device, generator=gen)
        in_noise_a = torch.randn(R, A, adim, device=device, generator=gen)
        in_noise_s = torch.randn(R, S, D, device=device, generator=gen)
        gnorm = z_goal.norm(dim=-1, keepdim=True)
        with torch.enable_grad():
            delta = torch.zeros(R, D, device=device, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=self.pm_lr)
            best_plan = best_cost = None
            for k in range(self.pm_steps + 1):
                action, _ = self.model._sample_impl(history, history_pad,
                                                    z_goal=z_goal + delta, h_norm=h_norm,
                                                    noise_action=noise_a, noise_state=noise_s)
                z_imag = self.model._inpaint_impl(history, history_pad, action,
                                                  in_noise_a, in_noise_s,
                                                  z_goal=z_goal, h_norm=h_norm)
                cost = self._final_cost(z_imag, z_goal)
                c = cost.detach()
                if best_cost is None:
                    best_cost, best_plan = c.clone(), action.detach().clone()
                else:
                    better = c < best_cost
                    best_plan[better] = action.detach()[better]
                    best_cost = torch.where(better, c, best_cost)
                if k == self.pm_steps:
                    break
                grad = torch.autograd.grad(cost.mean(), delta)[0]
                if self.pm_random:
                    r = torch.randn(R, D, device=device, generator=gen)
                    grad = r * (grad.norm(dim=-1, keepdim=True)
                                / r.norm(dim=-1, keepdim=True).clamp_min(1e-9))
                opt.zero_grad(set_to_none=True)
                delta.grad = grad
                opt.step()
                with torch.no_grad():
                    scale = (self.pm_rho * gnorm
                             / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
                    delta.mul_(scale)
        return best_plan

    def _cem(self, history, history_pad, z_goal, h_norm, replan):
        R, P = history.shape[0], self.plan_k
        A, adim = self.model.num_actions, self.model.action_raw_dim
        device = history.device
        take = self._n_actions_to_execute()

        mean, _ = self.model.sample(history, history_pad, generator=self._flow_generator(history.device),
                                    **self._cond_args(z_goal, h_norm, 1))
        for row, i in enumerate(replan):
            prev = self._prev_plan[i] if self._prev_plan is not None else None
            if prev is not None:
                mean[row] = torch.cat([prev[take:], prev[-1:].expand(take, adim)])
        std = torch.full_like(mean, self.cem_std)

        noise_action = torch.randn(R, 1, A, adim, device=device, generator=self._flow_generator(device)) \
            .expand(R, P, A, adim).reshape(R * P, A, adim)
        noise_state = torch.randn(R, 1, self.model.num_states, self.model.z_dim, device=device,
                                  generator=self._flow_generator(device)) \
            .expand(R, P, self.model.num_states, self.model.z_dim) \
            .reshape(R * P, self.model.num_states, self.model.z_dim)
        hist = history.repeat_interleave(P, 0)
        pad = history_pad.repeat_interleave(P, 0)
        goal = z_goal.repeat_interleave(P, 0)
        rows = torch.arange(R, device=device)

        best_plan = mean.clone()
        best_cost = torch.full((R,), float("inf"), device=device)
        for _ in range(self.cem_iters):
            cand = mean[:, None] + std[:, None] * torch.randn(R, P, A, adim, device=device,
                                                              generator=self._flow_generator(device))
            cand[:, 0] = mean
            z_imag = self.model.sample_inpaint(hist, pad, cand.reshape(R * P, A, adim),
                                               noise_action, noise_state,
                                               **self._cond_args(z_goal, h_norm, P))
            cost = self._final_cost(z_imag, goal).view(R, P)
            iter_cost, iter_pick = cost.min(1)
            better = iter_cost < best_cost
            best_plan[better] = cand[rows, iter_pick][better]
            best_cost = torch.minimum(iter_cost, best_cost)
            elite_idx = cost.topk(min(self.cem_elites, P), dim=1, largest=False).indices
            elites = cand[rows[:, None], elite_idx]
            mean, std = elites.mean(1), elites.std(1, correction=0).clamp_min(0.02)
        return best_plan


# LeWAM-Unified eval adapter (mode: unified_policy = reactive, full-causal from episode start)
class LeWAMUnifiedPolicy(LeWAMSplitPolicy):
    """Eval adapter for LeWAM-Unified. Full-causal from the episode start, mirroring training (each
    start is a fresh start). Per env it caches the encoded latent of every observed frame (encoder is
    deterministic at eval, so each frame is encoded once), re-runs the causal aggregator over the
    growing latent history, and reads the current position:

      z_t   = model.encode(current frame)          # appended to the per-env latent cache
      c_t   = model.aggregate([z_start..z_t])[-1]  # causal, over the whole history so far
      a     = model.gc_head(c_t, z_goal, h_norm)   # (block_dim) z-scored action block

    The goal is the fixed goal frame (encoded each replan); horizon counts down from
    goal_offset/action_block. 
    If `agg_action_cond`, the per-env cache of executed z-scored action blocks supplies a_{t-1} (null at t_start)
    Lengths are padded per batch and each env's last valid position is read, so unequal history lengths are handled. Un-z-score /
    action-unstack / model.eval() are inherited from LeWAMSplitPolicy."""

    def __init__(self, model, cfg, *args, **kwargs):
        """Wrap a trained LeWAMUnified model on top of LeWAMSplitPolicy's setup;
        ctx_cap/log_latents are documented inline below."""
        # ctx_cap > 0 -> aggregate only the LAST ctx_cap cached latents (i.e. sliding window)
        # 0 = full causal history from episode start (default behaviour).
        self.ctx_cap = int(kwargs.pop("ctx_cap", 0))
        # exec_blocks > 1 -> execute that many chunk blocks per replan (chunk-trained flow head
        # 1 = replan every block, the default
        self.exec_blocks = int(kwargs.pop("exec_blocks", 1))
        # exec_actions < action_block -> execute only the first N raw actions of a block
        self.exec_actions = int(kwargs.pop("exec_actions", 0))
        self.log_latents = bool(kwargs.pop("log_latents", False))
        super().__init__(model, cfg, *args, **kwargs)
        self.type = "lewam_unified_policy"
        self.action_cond = bool(cfg.get("agg_action_cond", False))
        self._lat_log = []       # (call_idx, env_i, z_cur[D], z_goal[D]) tuples when log_latents
        self._call = 0
        self._lat_buf = None     # per-env list of cached latents z_start..z_t (grows per episode)
        self._pblk_buf = None    # per-env list of z-scored blocks that led INTO each frame
        self._last_blk = None    # per-env last z-scored block emitted (the next frame's prev-action)

    def set_env(self, env):
        """Reset the per-env latent/prev-block caches and call log, on top of
        LeWAMSplitPolicy's action-buffer/horizon reset."""
        super().set_env(env)
        num_envs = getattr(env, "num_envs", 1)
        self._lat_buf = [[] for _ in range(num_envs)]
        self._pblk_buf = [[] for _ in range(num_envs)]
        self._last_blk = [None] * num_envs
        self._lat_log = []
        self._call = 0

    def dump_latents(self, path):
        """Save the executed per-obs-step latent trajectory (for the divergence probe)."""
        import numpy as np
        if not self._lat_log:
            np.savez(path, calls=np.zeros(0), envs=np.zeros(0))
            return
        calls = np.array([r[0] for r in self._lat_log], dtype=np.int64)
        envs = np.array([r[1] for r in self._lat_log], dtype=np.int64)
        z_cur = torch.stack([r[2] for r in self._lat_log]).numpy()
        z_goal = torch.stack([r[3] for r in self._lat_log]).numpy()
        np.savez(path, calls=calls, envs=envs, z_cur=z_cur, z_goal=z_goal)

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        info_dict = self._prepare_info(info_dict)
        num_envs = self.env.num_envs
        device = next(self.model.parameters()).device
        self._call += 1
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(num_envs)]
            self._steps_left = np.full(num_envs, self.horizon0, dtype=np.float64)
        if self._lat_buf is None:
            self._lat_buf = [[] for _ in range(num_envs)]
            self._pblk_buf = [[] for _ in range(num_envs)]
            self._last_blk = [None] * num_envs

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(num_envs):
                if flush[i]:
                    self._action_buffer[i].clear()
                    self._lat_buf[i].clear()
                    self._pblk_buf[i].clear()
                    self._last_blk[i] = None
                    self._steps_left[i] = _h0_at(self.horizon0, i)

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(num_envs, dtype=bool)

        replan = [i for i in range(num_envs) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            curr_obs = info_dict["pixels"][replan]
            assert "goal" in info_dict, "LeWAM-Unified eval needs info_dict['goal'] (goal-reaching)"
            goal_obs = info_dict["goal"][replan]
            c_obs = curr_obs[:, -1] if curr_obs.ndim == 5 else curr_obs     # (R,C,H,W) current frame
            g_obs = goal_obs[:, -1] if goal_obs.ndim == 5 else goal_obs     # (R,C,H,W) single goal frame
            # encode the new current frame once and cache it; cache the prev emitted block too
            z_new = self.model.encode(c_obs.to(device).float())             # (R, D)
            z_g = self.model.encode(g_obs.to(device).float())               # (R, D) fixed goal
            for row, i in enumerate(replan):
                self._lat_buf[i].append(z_new[row])
                self._pblk_buf[i].append(self._last_blk[i])        # None at start -> null

            if self.log_latents:
                for row, i in enumerate(replan):
                    self._lat_log.append((self._call, int(i),
                                          z_new[row].detach().cpu(), z_g[row].detach().cpu()))
                    
            lens = [len(self._lat_buf[i]) for i in replan]
            if self.ctx_cap and self.ctx_cap > 0:
                lens = [min(l, self.ctx_cap) for l in lens]
            Lmax = max(lens)
            R, Dd = len(replan), z_new.shape[-1]
            # pad the growing histories to Lmax (pads at the END; causal -> never seen by earlier
            # positions, and we read each env's own last valid index)
            seq = torch.zeros(R, Lmax, Dd, device=device)
            for row, i in enumerate(replan):
                buf = torch.stack(self._lat_buf[i], dim=0)         # (L_i, D)
                if self.ctx_cap and self.ctx_cap > 0:
                    buf = buf[-self.ctx_cap:]                       # keep only the last k latents
                seq[row, : buf.shape[0]] = buf
            a_prev = a_prev_mask = None

            if self.action_cond:
                a_prev = torch.zeros(R, Lmax, self.block_dim, device=device)
                a_prev_mask = torch.zeros(R, Lmax, dtype=torch.bool, device=device)
                for row, i in enumerate(replan):
                    pblk = self._pblk_buf[i]
                    if self.ctx_cap and self.ctx_cap > 0:
                        if len(pblk) > self.ctx_cap:
                            # Truncated window: training always presents a window's first position
                            # with the null action (fresh start), so the capped window must too --
                            # feeding the real pre-window block here is a train/eval mismatch
                            pblk = [None] + pblk[-self.ctx_cap + 1:] if self.ctx_cap > 1 else [None]
                        else:
                            pblk = pblk[-self.ctx_cap:]
                    for k, blk in enumerate(pblk):
                        if blk is not None:
                            a_prev[row, k] = blk.to(device)
                            a_prev_mask[row, k] = True

            c = self.model.aggregate(seq, a_prev, a_prev_mask)        # (R, Lmax, D)
            last_idx = torch.tensor([l - 1 for l in lens], device=device)
            c_last = c[torch.arange(R, device=device), last_idx]      # (R, D) each env's current pos
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=device, dtype=torch.float32)
            if self.ablate_horizon:
                h_norm = torch.zeros_like(h_norm)

            head_out = self.model.gc_head(c_last, z_g, h_norm)
            if self.exec_blocks > 1 and hasattr(self.model.gc_head, "_integrate"):
                chunk = self.model.gc_head._integrate(head_out, K=1)[:, 0]   # (R, H, blk) one ODE draw
                n_exec = min(self.exec_blocks, chunk.shape[1])
                z_blocks = [chunk[:, b] for b in range(n_exec)]              # z-scored blocks, in order
            else:
                z_blocks = [self.model.gc_head.point(head_out)]              # (R, blk)
                n_exec = 1
            for row, i in enumerate(replan):
                self._last_blk[i] = z_blocks[-1][row].detach()     # last EXECUTED block, next frame's prev
            take = self.exec_actions if 0 < self.exec_actions < self.action_block else self.action_block
            for z_blk in z_blocks:
                raw = (z_blk.reshape(len(replan), self.frameskip, self.raw_adim)
                       * self._astd + self._amean)                 # un-z-score per raw dim
                raw = raw.reshape(len(replan), self.action_block, self.action_dim).cpu()
                for row, i in enumerate(replan):
                    self._action_buffer[i].extend(raw[row][:take])
            for row, i in enumerate(replan):
                consumed = float(n_exec) * take / self.action_block
                self._steps_left[i] = max(self._steps_left[i] - consumed, 1.0)  # obs-steps consumed

        action = torch.full((num_envs, self.action_dim), float("nan"))
        for i in range(num_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()


# LeWAM-Unified CEM planner (mode: unified_cem) -- plans with the dynamics head instead of the
# reactive gc_head. Diagnostic: if CEM reaches goals the reactive policy fails, the world model is
# fine and the reactive head/rollout is the weak link (undertraining); if CEM also fails, the
# encoder/dynamics is the problem.
class LeWAMUnifiedCEMPolicy(LeWAMUnifiedPolicy):
    """Receding-horizon CEM over z-scored action BLOCKS, scored by predicted latent distance to the
    goal. Rollout: z_0 = encode(current); z_{h+1} = model.dynamics(z_h, a_h, z_goal); cost = sum_h
    ||z_h - z_goal||^2. CEM refines a per-env Gaussian over [a_0..a_{H-1}] and executes a_0."""

    def __init__(self, model, cfg, *args, **kwargs):
        self.cem_K = int(kwargs.pop("cem_K", 256))       # samples per iter
        self.cem_M = int(kwargs.pop("cem_M", 32))        # "elite" subset per iter
        self.cem_iter = int(kwargs.pop("cem_iter", 4))
        self.cem_H = int(kwargs.pop("cem_H", 5))             # planning horizon (in blocks)
        self.cem_std = float(kwargs.pop("cem_std", 1.0))     # init std (z-scored actions ~ unit var)
        self.cem_warm = bool(kwargs.pop("cem_warm", False))  # seed CEM mean from the reactive gc_head plan
        # cem_state: 'c' feeds dynamics the aggregated context c=Aggr([z..]) it was trained on (rollout
        # re-aggregates the sliding window each step -- the faithful world-model loop); 'z' feeds the raw
        # last latent (the old shortcut, kept as an ablation to measure the gap).
        self.cem_state = str(kwargs.pop("cem_state", "c"))
        # receding-horizon MPC scheme: False (default) executes only a_0 then replans every frame (max
        # feedback); True executes the entire optimized H-block plan before replanning
        self.cem_exec_full = bool(kwargs.pop("cem_exec_full", False))
        # 'cem' (default) = Gaussian sample+refine around the warm-start; 
        # 'policy' = sample K candidate sequences from the policy (works with GMM or Flow head)
        self.cem_propose = str(kwargs.pop("cem_propose", "cem"))
        # 'final' (default) scores only the terminal ||z_H - z_goal||^2
        # 'sum' accumulates ||z_h - z_goal||^2 over every rollout step; not recommended, 
        #     leads to greedy exploitation
        self.cem_cost = str(kwargs.pop("cem_cost", "final"))
        # 'prefix' = Fast-LeWM one-forward parallel refix prediction from the fixed anchor 
        # 'rollout' = classic autoregressive rollout of dynamics head
        # 'auto' = pick mode based on whether model uses a prefix decoder
        _dyn_mode = str(kwargs.pop("cem_dyn_mode", "auto"))
        self.cem_dyn_mode = (("prefix" if getattr(model, "use_prefix", False) else "rollout")
                             if _dyn_mode == "auto" else _dyn_mode)
        # dedicated RNG for CEM sampling for reproducibility
        self._cem_seed = int(kwargs.pop("cem_seed", 0))
        self._cem_gen = None
        super().__init__(model, cfg, *args, **kwargs)
        assert self.ctx_cap and self.ctx_cap > 0, "unified_cem needs ctx_cap>0 (the trained context_len)"
        self.type = f"lewam_unified_cem_{self.cem_state}" + ("_warm" if self.cem_warm else "")

    def _dyn_step(self, ctx, action, z_goal):
        """One-step latent transition, dynamics-agnostic. PrefixDynamics -> the k=1 prefix head (an
        H-invariant one-step model); single-step GoalCondDynamics -> called directly. Lets the
        autoregressive rollout / warm-start loops work for both dynamics types."""
        if getattr(self.model, "use_prefix", False):
            goal_in = z_goal if self.model.dynamics.goal_cond else None
            return self.model.dynamics(ctx, action.unsqueeze(1), goal_in)[:, 0]
        return self.model.dynamics(ctx, action, z_goal)

    def _context_at_head(self, window, win_len):
        """Aggregated world-model context at each batch idx's newest valid position:
        Args:
            window: (batch, ctx_cap, latent_dim) history of latent states
            win_len: (batch,) length of each batch idx's valid history
        Returns:
            model.aggregate(window)[win_len - 1]: (batch, latent_dim)"""
        prev_action = prev_action_mask = None
        if getattr(self, "action_cond", False):
            # the AdaLN aggregator needs a real (non-None) prev-action tensor; feed the null action
            # everywhere (mask all-False -> trained null_action conditioning, the fresh-start default).
            batch, ctx_cap = window.shape[0], window.shape[1]
            prev_action = torch.zeros(batch, ctx_cap, self.block_dim, device=window.device, dtype=window.dtype)
            prev_action_mask = torch.zeros(batch, ctx_cap, dtype=torch.bool, device=window.device)
        context = self.model.aggregate(window, prev_action, prev_action_mask)   # (batch, ctx_cap, latent_dim)
        rows = torch.arange(window.shape[0], device=window.device)
        return context[rows, (win_len - 1).clamp(min=0)]

    def _append_latent(self, window, win_len, latent, ctx_cap):
        """Append `latent` to the sliding window.
        Batch_idx with room place latent at win_len in dim1; full rows shift left (FIFO)
        to keep `window` (batch, ctx_cap, latent_dim) left-aligned. 

        Args:
            window: (batch, ctx_cap, latent_dim) history of latent states
            win_len: (batch,) length of each batch idx's valid history
            latent: (batch, latent_dim) new latent to append
            ctx_cap (int): max context length (caps window.shape[1])
        """
        batch = window.shape[0]
        is_full = (win_len >= ctx_cap)
        window_shifted = torch.cat([window[:, 1:], latent[:, None]], dim=1)   # drop idx0, append latent
        window_placed = window.clone()
        rows = torch.arange(batch, device=window.device)
        window_placed[rows, win_len.clamp(max=ctx_cap - 1)] = latent
        window_next = torch.where(is_full[:, None, None], window_shifted, window_placed)
        return window_next, torch.clamp(win_len + 1, max=ctx_cap)

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        info_dict = self._prepare_info(info_dict)
        n_envs = self.env.num_envs
        device = next(self.model.parameters()).device

        # seed CEM sampling from the eval seed
        if self._cem_gen is None:
            self._cem_gen = torch.Generator(device=device); self._cem_gen.manual_seed(self._cem_seed)
        self._call += 1

        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(n_envs)]
            self._steps_left = np.full(n_envs, self.horizon0, dtype=np.float64)
        if self._lat_buf is None:
            self._lat_buf = [[] for _ in range(n_envs)]
        
        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for env_i in range(n_envs):
                if flush[env_i]:
                    self._action_buffer[env_i].clear()
                    self._lat_buf[env_i].clear()
                    self._steps_left[env_i] = _h0_at(self.horizon0, env_i)

        term = info_dict.get("terminated")
        is_dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n_envs, dtype=bool)
        # only envs whose action buffer has drained (and aren't done) need a fresh plan this step
        replan_envs = [i for i in range(n_envs) if len(self._action_buffer[i]) == 0 and not is_dead[i]]
        if replan_envs:
            cur_px = info_dict["pixels"][replan_envs]
            assert "goal" in info_dict, "LeWAM-Unified CEM needs info_dict['goal']"
            goal_px = info_dict["goal"][replan_envs]
            goal_px = goal_px[:, -1] if goal_px.ndim == 5 else goal_px
            cur_px = cur_px[:, -1] if cur_px.ndim == 5 else cur_px
            z_cur = self.model.encode(cur_px.to(device).float())        # (n_replan, latent_dim) current latent
            z_goal = self.model.encode(goal_px.to(device).float())      # (n_replan, latent_dim) goal latent
            for row, env_i in enumerate(replan_envs):
                self._lat_buf[env_i].append(z_cur[row])                 # cache the REAL latent
            if self.log_latents:                                        # executed real dist-to-goal probe
                for row, env_i in enumerate(replan_envs):
                    self._lat_log.append((self._call, int(env_i),
                                          z_cur[row].detach().cpu(), z_goal[row].detach().cpu()))
            n_replan, latent_dim = z_cur.shape
            n_samples, n_elites = self.cem_K, self.cem_M
            plan_horizon, block_dim = self.cem_H, self.block_dim        # horizon in action-BLOCKS
            ctx_cap = int(self.ctx_cap)                                 # trained context window (=context_len)
            # initial window = last <= `ctx_cap` observed latents per env
            window0 = torch.zeros(n_replan, ctx_cap, latent_dim, device=device)
            win_len0 = torch.empty(n_replan, dtype=torch.long, device=device)
            for row, env_i in enumerate(replan_envs):
                real_latents = torch.stack(self._lat_buf[env_i][-ctx_cap:], dim=0)   # (len, latent_dim), len<=ctx_cap
                window0[row, : real_latents.shape[0]] = real_latents
                win_len0[row] = real_latents.shape[0]
            # goal replicated once per candidate sample: (n_replan*n_samples, latent_dim)
            z_goal_rep = z_goal.unsqueeze(1).expand(n_replan, n_samples, latent_dim).reshape(n_replan * n_samples, latent_dim)
            # propose candidate H-block plans, keep the WM-verified best -> `plan_mean` (n_replan,H,block_dim)
            if self.cem_propose in ("policy", "policy_modes"):
                # policy-proposal MPC: sample n_samples sequences from a probabilistic policy autoregressively,
                # roll each through the dynamics, score, keep the best per env.
                sample_noise = (self.cem_propose == "policy")
                window = window0.unsqueeze(1).expand(n_replan, n_samples, ctx_cap, latent_dim).reshape(n_replan * n_samples, ctx_cap, latent_dim).clone()
                win_len = win_len0.unsqueeze(1).expand(n_replan, n_samples).reshape(n_replan * n_samples).clone()
                steps_left = np.repeat(np.maximum(self._steps_left[replan_envs], 1.0).astype(np.float64), n_samples)
                cost = torch.zeros(n_replan * n_samples, device=device); action_blocks = []
                for _h in range(plan_horizon):
                    horizon_norm = torch.tensor(np.minimum(steps_left, self.H_max) / self.H_max,
                                                device=device, dtype=torch.float32)
                    if self.ablate_horizon:
                        horizon_norm = torch.zeros_like(horizon_norm)
                    context = self._context_at_head(window, win_len)    # (n_replan*n_samples, latent_dim)
                    action_blk = self.model.gc_head.sample(
                        self.model.gc_head(context, z_goal_rep, horizon_norm), 1, 
                        noise=sample_noise, generator=self._cem_gen).squeeze(1)
                    action_blocks.append(action_blk)
                    z_next = self._dyn_step(context, action_blk, z_goal_rep)          # WM imagines next latent (k=1 for prefix)
                    step_cost = ((z_next - z_goal_rep) ** 2).sum(-1)
                    cost = step_cost if self.cem_cost == "final" else cost + step_cost
                    window, win_len = self._append_latent(window, win_len, z_next, ctx_cap)
                    steps_left = np.maximum(steps_left - 1.0, 1.0)
                best_sample = cost.reshape(n_replan, n_samples).argmin(dim=1)         # (n_replan,) WM-verified best
                plan_mean = torch.stack(action_blocks, dim=1).reshape(n_replan, n_samples, plan_horizon, block_dim)[torch.arange(n_replan, device=device), best_sample]
            else:
                # warm-start mean: AR-roll gc_head + dynamics through the same windowed context (batched over replans)
                if self.cem_warm:
                    steps_left = np.maximum(self._steps_left[replan_envs], 1.0).astype(np.float64)
                    window, win_len, warm_blocks = window0.clone(), win_len0.clone(), []
                    for _h in range(plan_horizon):
                        horizon_norm = torch.tensor(np.minimum(steps_left, self.H_max) / self.H_max,
                                                    device=device, dtype=torch.float32)
                        if self.ablate_horizon:
                            horizon_norm = torch.zeros_like(horizon_norm)
                        context = self._context_at_head(window, win_len)    # (n_replan, latent_dim) trained context
                        action_blk = self.model.gc_head.point(self.model.gc_head(context, z_goal, horizon_norm))
                        warm_blocks.append(action_blk)
                        window, win_len = self._append_latent(window, win_len, self._dyn_step(context, action_blk, z_goal), ctx_cap)
                        steps_left = np.maximum(steps_left - 1.0, 1.0)
                    plan_mean = torch.stack(warm_blocks, dim=1)             # (n_replan, plan_horizon, block_dim)
                else:
                    plan_mean = torch.zeros(n_replan, plan_horizon, block_dim, device=device)
                plan_std = torch.full((n_replan, plan_horizon, block_dim), self.cem_std, device=device)
                # window/win_len for every (replan, sample) candidate, seeded from the real window
                window0_rep = window0.unsqueeze(1).expand(n_replan, n_samples, ctx_cap, latent_dim).reshape(n_replan * n_samples, ctx_cap, latent_dim)
                win_len0_rep = win_len0.unsqueeze(1).expand(n_replan, n_samples).reshape(n_replan * n_samples)
                flat_rows = torch.arange(n_replan * n_samples, device=device)
                for _ in range(self.cem_iter):
                    action_seqs = plan_mean.unsqueeze(1) + plan_std.unsqueeze(1) * torch.randn(n_replan, n_samples, plan_horizon, block_dim, device=device, generator=self._cem_gen)
                    action_seqs_flat = action_seqs.reshape(n_replan * n_samples, plan_horizon, block_dim)
                    if self.cem_dyn_mode == "prefix":
                        # Fast-LeWM: one forward predicts all H prefix latents from the fixed anchor
                        # context (no sliding, no latent feedback -> no compounding); score the terminal.
                        anchor_ctx = self._context_at_head(window0_rep, win_len0_rep)   # (n_replan*n_samples, D)
                        goal_in = z_goal_rep if self.model.dynamics.goal_cond else None
                        pred_seq = self.model.dynamics(anchor_ctx, action_seqs_flat, goal_in)   # (RK, H, D)
                        if self.cem_cost == "final":
                            cost = ((pred_seq[:, -1] - z_goal_rep) ** 2).sum(-1)
                        else:
                            cost = ((pred_seq - z_goal_rep.unsqueeze(1)) ** 2).sum(-1).sum(-1)
                    else:
                        # classic autoregressive rollout: feed each predicted latent back, re-aggregate.
                        window, win_len = window0_rep.clone(), win_len0_rep.clone()
                        cost = torch.zeros(n_replan * n_samples, device=device)
                        for h in range(plan_horizon):
                            if self.cem_state == "z":                  # ablation: raw last latent
                                context = window[flat_rows, (win_len - 1).clamp(min=0)]
                            else:                                      # correct: aggregated context
                                context = self._context_at_head(window, win_len)
                            z_next = self._dyn_step(context, action_seqs_flat[:, h], z_goal_rep)  # WM imagines next latent
                            step_cost = ((z_next - z_goal_rep) ** 2).sum(-1)
                            cost = step_cost if self.cem_cost == "final" else cost + step_cost
                            window, win_len = self._append_latent(window, win_len, z_next, ctx_cap)
                    cost = cost.reshape(n_replan, n_samples)
                    elite_idx = cost.argsort(dim=1)[:, :n_elites]       # (n_replan, n_elites) best candidates
                    elite_seqs = torch.gather(action_seqs, 1, elite_idx[:, :, None, None].expand(n_replan, n_elites, plan_horizon, block_dim))
                    plan_mean = elite_seqs.mean(1)
                    plan_std = elite_seqs.std(1).clamp(min=1e-3)
            # execute block 0 only (replan every frame), or the whole H-block plan (LeWM receding MPC)
            n_exec_blocks = self.cem_H if self.cem_exec_full else 1
            for h in range(n_exec_blocks):
                block = plan_mean[:, h]                                 # (n_replan, block_dim) block h of the plan
                raw_action = (block.reshape(n_replan, self.frameskip, self.raw_adim) * self._astd + self._amean)
                raw_action = raw_action.reshape(n_replan, self.action_block, self.action_dim).cpu()
                for row, env_i in enumerate(replan_envs):
                    self._action_buffer[env_i].extend(raw_action[row])
            for row, env_i in enumerate(replan_envs):
                self._steps_left[env_i] = max(self._steps_left[env_i] - float(n_exec_blocks), 1.0)
        action = torch.full((n_envs, self.action_dim), float("nan"))
        for env_i in range(n_envs):
            if not is_dead[env_i]:
                action[env_i] = self._action_buffer[env_i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()



# intuition-seeded planning: opt-in CEM variance floor
from stable_worldmodel.solver.callbacks import Callback as _SWMCallback


class VarFloorCallback(_SWMCallback):
    """Opt-in CEM variance floor for intuition-seeded planning.

    The swm CEM updates ``var = elites.std()`` with NO floor, so over ``n_steps``
    the variance collapses to ~0 and the search dies ON the seed -- the intuition
    prior gets REPLAYED, not refined. This callback clamps the per-dim action std
    to ``>= min_var`` (in place, after each elite fit), so CEM keeps exploring a
    neighborhood around the intuition-seeded mean and DELIBERATION actually happens.

    The full-horizon intuition seed is ALREADY provided by the swm path
    (prepare_init_action -> model.get_action(horizon) -> jepa.intention_rollout);
    this floor is the missing piece that lets that good seed be *refined* rather
    than just collapsed onto. This callback has no effect unless it is added
    with ``min_var > 0`` (the default solver config has no callbacks).
    """

    def __init__(self, min_var: float = 0.1, reduction: str = "none"):
        super().__init__(reduction)
        self.min_var = float(min_var)

    def compute(self, **state):
        v = state.get("var")
        if v is not None and self.min_var > 0:
            v.clamp_(min=self.min_var)  # in-place -> propagates to next CEM iter
        return None  # no per-step recording
