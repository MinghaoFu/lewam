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
    last frame and only one start per episode qualifies. Episodes too short to reach
    the goal are excluded. Sampling is deterministic given cfg.seed.

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
    to_end = bool(cfg.get("gip_eval", {}).get("to_end", False))
    if to_end:
        valid = np.nonzero((step_idx == max_start_per_row) & (max_start_per_row >= 0))[0]
    else:
        valid = np.nonzero(step_idx <= max_start_per_row)[0]

    g = np.random.default_rng(cfg.seed)
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
    # infer the projector width from the checkpoint so older ckpts (wider projector)
    # load into the current ViTEncoder without a config field.
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
        head_type=str(cfg.get("head_type", "mse")), n_mix=int(cfg.get("n_mix", 5)),
        flow_H=int(cfg.get("flow_H", 1)),
        dyn_action_embed_dim=int(cfg.get("dyn_action_embed_dim", 0) or 0),
        use_idm=bool(float(cfg.get("w_idm", 0) or 0) > 0),
        use_prefix=bool(cfg.get("use_prefix", False)),
        prefix_H=int(cfg.get("prefix_H", 5)),
        prefix_depth=int(cfg.get("prefix_depth", 2)),
        prefix_heads=int(cfg.get("prefix_heads", 4)),
        # the head has to be rebuilt with the flags it was trained under -- the load below is
        # strict, so a latent-h run whose flags are dropped here fails on missing keys rather than
        # quietly falling back to horizon conditioning
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
    # end-to-end (from-scratch) model: the encoder was trained jointly with the head,
    # so its weights are in the full-model checkpoint under "encoder.*", not a frozen
    # file. build_frozen_lewm built the arch only; load the trained encoder here.
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
def build_policy(cfg, model, adim, process, transform, goal_offsets=None):
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

    # mode=unified_policy: LeWAM-Unified adapter. `model` is a loaded LeWAMUnified with its config
    # attached as model._unified_cfg (done in eval_gip.py).
    if mode in ("unified_policy", "unified_cem", "unified_grad", "unified_candgrad", "unified_dgoal", "unified_cemdiag"):
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
                      ablate_horizon=bool(ge.get("ablate_horizon", uni_cfg.get("ablate_horizon", False))))
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
        if mode == "unified_cemdiag":
            from lewam.models.grad_mpc import LeWAMUnifiedCEMDiagPolicy
            return LeWAMUnifiedCEMDiagPolicy(
                cem_K=int(ge.get("cem_K", 256)), cem_M=int(ge.get("cem_M", 32)),
                cem_iter=int(ge.get("cem_iter", 4)), cem_H=int(ge.get("cem_H", 5)),
                cem_std=float(ge.get("cem_std", 1.0)),
                cem_warm=bool(ge.get("cem_warm", False)),
                cem_state=str(ge.get("cem_state", "c")),
                cem_exec_full=bool(ge.get("cem_exec_full", False)),
                cem_cost=str(ge.get("cem_cost", "final")),
                cem_seed=int(cfg.seed),
                log_latents=bool(ge.get("dump_latents", "")),
                ctx_cap=int(ge.get("ctx_cap", uni_cfg.get("context_len", 5))), **common)
        if mode == "unified_dgoal":
            from lewam.models.grad_mpc import LeWAMUnifiedDGoalPolicy
            return LeWAMUnifiedDGoalPolicy(
                dg_steps=int(ge.get("dg_steps", 20)), dg_lr=float(ge.get("dg_lr", 0.02)),
                dg_clip=float(ge.get("dg_clip", 5.0)), dg_rho=float(ge.get("dg_rho", 0.3)),
                dg_cost=str(ge.get("dg_cost", "terminal")),
                dg_warm=bool(ge.get("dg_warm", False)), dg_pop=int(ge.get("dg_pop", 1)),
                dg_random=bool(ge.get("dg_random", False)),
                dg_seg=int(ge.get("dg_seg", 0)), dg_k1=int(ge.get("dg_k1", 0)),
                dg_r_rho=float(ge.get("dg_r_rho", 0.1)),
                grad_H_auto=bool(ge.get("grad_H_auto", False)),
                grad_exec_k=int(ge.get("grad_exec_k", 0)),
                grad_H=int(ge.get("grad_H", 5)),
                grad_exec_full=bool(ge.get("grad_exec_full", True)),
                cem_seed=int(cfg.seed),
                log_latents=bool(ge.get("dump_latents", "")),
                ctx_cap=int(ge.get("ctx_cap", uni_cfg.get("context_len", 5))), **common)
        if mode == "unified_candgrad":
            from lewam.models.grad_mpc import LeWAMUnifiedCandGradPolicy
            return LeWAMUnifiedCandGradPolicy(
                cand_snapshots=str(ge.get("cand_snapshots", "5,15,50")),
                cand_pol_K=int(ge.get("cand_pol_K", 3)),
                judge_noise=float(ge.get("judge_noise", 0.3)),
                judge_m=int(ge.get("judge_m", 6)),
                grad_steps=int(ge.get("grad_steps", 50)), grad_lr=float(ge.get("grad_lr", 0.05)),
                grad_H=int(ge.get("grad_H", 5)), grad_clip=float(ge.get("grad_clip", 10.0)),
                grad_dyn_mode=str(ge.get("grad_dyn_mode", "auto")),
                grad_exec_full=bool(ge.get("grad_exec_full", True)),
                cem_seed=int(cfg.seed),
                log_latents=bool(ge.get("dump_latents", "")),
                ctx_cap=int(ge.get("ctx_cap", uni_cfg.get("context_len", 5))), **common)
        if mode == "unified_grad":
            from lewam.models.grad_mpc import LeWAMUnifiedGradPolicy
            return LeWAMUnifiedGradPolicy(
                grad_steps=int(ge.get("grad_steps", 50)), grad_lr=float(ge.get("grad_lr", 0.05)),
                grad_H=int(ge.get("grad_H", 5)), grad_clip=float(ge.get("grad_clip", 10.0)),
                grad_action_clip=ge.get("grad_action_clip", None),
                grad_dyn_mode=str(ge.get("grad_dyn_mode", "auto")),
                grad_exec_full=bool(ge.get("grad_exec_full", True)),
                grad_warm=bool(ge.get("grad_warm", True)),
                grad_noise=float(ge.get("grad_noise", 0.0)),
                grad_select=str(ge.get("grad_select", "best")),
                grad_H_auto=bool(ge.get("grad_H_auto", False)),
                grad_exec_k=int(ge.get("grad_exec_k", 0)),
                grad_cycle=float(ge.get("grad_cycle", 0.0)), grad_dis=float(ge.get("grad_dis", 0.0)),
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
        # log_latents -> record the executed per-obs-step current latent + goal latent for probes
        # (e.g., ompare ctx_cap=1 vs full rollouts).
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

            z_blk = self.model.gc_head.point(self.model.gc_head(c_last, z_g, h_norm))   # (R, blk) z-scored (mixture mean for gmm)
            for row, i in enumerate(replan):
                self._last_blk[i] = z_blk[row].detach()            # z-scored, next frame's prev
            raw = (z_blk.reshape(len(replan), self.frameskip, self.raw_adim)
                   * self._astd + self._amean)                     # un-z-score per raw dim
            raw = raw.reshape(len(replan), self.action_block, self.action_dim).cpu()
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(raw[row])
                self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)  # one obs-step consumed

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
            # ---- propose candidate H-block plans, keep the WM-verified best -> `plan_mean` (n_replan,H,block_dim) ----
            if self.cem_propose in ("policy", "policy_modes"):
                # policy-proposal MPC: sample n_samples sequences from a probabilistic policy autoregressively,
                # roll each through the dynamics, score, keep the best per env.
                # No CEM gaussian sampling -> candidates stay on the policy's action manifold
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
