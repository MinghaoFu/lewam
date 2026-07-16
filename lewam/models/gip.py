"""GIP step-1 evaluation infrastructure.

One place for everything the GIP intention policy needs at eval time:
  - loading a GIP checkpoint (vanilla JEPA + the runtime action modules),
  - the shared env/dataset/process/episode-sampling helpers,
  - the two policies (reactive BC, and the Actionable adapter for guided
    planning), and
  - a single `build_policy` factory dispatched by `gip_eval.mode`.

The success-rate mode is the one config knob, mirroring how `action_pred.enabled`
gates training:  gip_eval.mode = bc | guided | planning
  - bc        : run the intention head directly as the policy (action head only)
  - guided    : intention-guided planning -- the action head warm-starts CEM,
                the world-model state head rolls candidates to the goal (JOINT)
  - planning  : plain world-model CEM planning (state head only; LeWM baseline)
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


# shared env / dataset / process helpers
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
    """Episode ids + start steps, identical sampling to eval.py."""
    col = episode_col(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col), return_index=True)

    lengths = []
    step_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    ep_idx = np.asarray(dataset.get_col_data(col)).reshape(-1)
    for ep_id in ep_indices:
        lengths.append(np.max(step_idx[ep_idx == ep_id]) + 1)
    lengths = np.array(lengths)

    max_start = lengths - cfg.eval.goal_offset_steps - 1
    max_start_by_ep = {e: max_start[i] for i, e in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_by_ep[e] for e in ep_idx])
    valid = np.nonzero(step_idx <= max_start_per_row)[0]

    g = np.random.default_rng(cfg.seed)
    picks = np.sort(valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
    episodes = dataset.get_row_data(picks)[col]
    starts = dataset.get_row_data(picks)["step_idx"]
    if len(episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")
    return episodes.tolist(), starts.tolist()


# model loading
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
        embed_dim = int(config.predictor.input_dim)  # 192 vit-tiny / 384 dinov2-small
    adim = int(config.action_encoder.input_dim)
    model.action_predictor = hydra_instantiate(config.predictor)
    # action-head: rebuild decoder from config.json (cfg.model.action_head); absent => MLP (mse).
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


def load_lewam_seq_model(run_name, which="best"):
    """Load a trained LeWAM-Seq model (lewam.models.lewam_seq.LeWAMSeq) + its config.

    Reads checkpoints/<run_name>/lewam_seq_config.json (arch dims + action z-score
    stats: action_mean/std, frameskip, action_raw_dim) and lewam_seq_best.pt (or
    lewam_seq_latest.pt when which='latest' or best is absent). Returns (model, cfg).
    The trainer only writes best/latest (no per-epoch files)."""
    from lewam.models.lewam_seq import LeWAMSeq

    cache = Path(get_cache_dir(sub_folder="checkpoints"))
    run_dir = cache / run_name
    cfg = json.loads((run_dir / "lewam_seq_config.json").read_text())
    model = LeWAMSeq(
        act_dim=int(cfg["action_dim"]), img_size=224, embed_dim=int(cfg["embed_dim"]),
        n_layers=int(cfg["n_layers"]), n_heads=int(cfg["n_heads"]), mlp_dim=int(cfg["mlp_dim"]),
        num_frames=int(cfg["num_frames"]), head_hidden=int(cfg["head_hidden"]),
    )
    ckpt = run_dir / ("lewam_seq_latest.pt" if which == "latest" else "lewam_seq_best.pt")
    if not ckpt.exists():
        ckpt = run_dir / "lewam_seq_latest.pt"
    assert ckpt.exists(), f"no lewam_seq_*.pt checkpoint in {run_dir}"
    sd = torch.load(ckpt, map_location="cpu")
    res = model.load_state_dict(sd, strict=True)
    print(f"[SEQ] load {run_name} <- {ckpt.name}: nf={cfg['num_frames']} "
          f"action_block={cfg['action_dim']} (raw {cfg['action_raw_dim']}x{cfg['frameskip']}) "
          f"H_max={cfg['H_max']} missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    return model, cfg


def load_lewam_split_model(run_name, which="best"):
    """Load a trained LeWAM-Split model (lewam.models.lewam_split.LeWAMSplit) + its
    config, both written by scripts/train_lewam_gc.py.

    Reads checkpoints/<run_name>/lewam_gc_config.json (arch dims + action z-score
    stats: action_mean/std, frameskip, action_raw_dim) and lewam_gc_best.pt (or
    lewam_gc_latest.pt when which='latest' or best is absent). The checkpoint is the
    full model state_dict (encoder./gc_head./dynamics. prefixes), so it loads strict.
    Returns (model, cfg). This is the direct reactive-GC eval path for the split model;
    it does NOT go through the frozen-LeWM/JEPA rebuild (load_gcidm_model), whose
    projector no longer matches module.ViTEncoder's."""
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
    """Load a trained LeWAM-Unified model (lewam.models.lewam_unified.LeWAMUnified) + its
    config, both written by scripts/train_lewam_unified.py.

    Reads checkpoints/<run_name>/lewam_unified_config.json (arch dims incl. window/agg_depth
    + action z-score stats) and lewam_unified_best.pt (or _latest.pt). The checkpoint is the
    full model state_dict (encoder./aggregator./gc_head./dynamics. prefixes) -> loads strict.
    Returns (model, cfg). This is the direct reactive-GC eval path for the unified model;
    like the split loader it does NOT go through the frozen-LeWM/JEPA rebuild."""
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
        embed_dim=int(cfg["z_dim"]), action_dim=int(cfg["action_dim"]),
        hidden_dim=int(cfg["hidden_dim"]), img_size=224,
        dropout=float(cfg.get("dropout", 0.1)), proj_hidden=proj_hidden,
        window=int(cfg["window"]), agg_depth=int(cfg["agg_depth"]),
        agg_heads=int(cfg.get("agg_heads", 4)),
        agg_residual=bool(cfg.get("agg_residual", True)),
        agg_gate=bool(cfg.get("agg_gate", False)),
        agg_action_cond=bool(cfg.get("agg_action_cond", False)),
        dyn_on_ct=bool(cfg.get("dyn_on_ct", True)),
    )
    res = model.load_state_dict(sd, strict=True)
    print(f"[UNIFIED] load {run_name} <- {ckpt.name}: action_block={cfg['action_dim']} "
          f"(raw {cfg['action_raw_dim']}x{cfg['frameskip']}) H_max={cfg['H_max']} "
          f"window={cfg['window']} agg_depth={cfg['agg_depth']} "
          f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    return model, cfg


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


# reactive BC policy (mode = bc)
class BCPolicy(BasePolicy):
    """Behavioral-cloning policy: run the intention head directly. Each replan
    encodes the current frame (position 0, as trained), proposes one
    frameskip-stacked block via the action head, unstacks it into `action_block`
    raw actions, and executes them before re-observing. Buffer/inverse_transform
    handling mirrors WorldModelPolicy."""

    def __init__(self, model, action_block, action_dim, process=None, transform=None,
                 goal_conditioned=False, H_max=50, horizon0=None, history_size=3, **kw):
        super().__init__(**kw)
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
        self.horizon0 = float(horizon0) if horizon0 is not None else None
        self._steps_left = None  # per-env remaining obs-steps
        # buffer last HS observed frames + last HS-1 emitted action blocks per env so the eval
        # context (z_{t-HS+1..t} + a_{<t}) matches training. Active only for the history-conditioned GC head.
        self.history_size = int(history_size)
        self.use_history = bool(getattr(self.model, "use_action_history", True))
        self._frame_buf = None      # per-env deque of the last HS preprocessed frames
        self._past_act_buf = None   # per-env deque of the last HS-1 emitted raw action blocks

    def set_env(self, env):
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._action_buffer = [deque() for _ in range(n)]
        self._frame_buf = [deque(maxlen=self.history_size) for _ in range(n)]
        self._past_act_buf = [deque(maxlen=max(self.history_size - 1, 0)) for _ in range(n)]
        if self.use_horizon and self.horizon0 is not None:
            self._steps_left = np.full(n, self.horizon0, dtype=np.float64)

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(n)]
        if self.use_horizon and self.horizon0 is not None and self._steps_left is None:
            self._steps_left = np.full(n, self.horizon0, dtype=np.float64)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._action_buffer[i].clear()
                    if self._steps_left is not None:
                        self._steps_left[i] = self.horizon0  # reset countdown for the new episode
                    if self._frame_buf is not None:
                        self._frame_buf[i].clear(); self._past_act_buf[i].clear()  # reset history

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        # push the current obs-frame into each live env's HS buffer every replan (one replan == one obs-step).
        cur_px = info_dict["pixels"]  # (n, 1, C, H, W) preprocessed
        if self.use_horizon and self._frame_buf is not None:
            for i in range(n):
                if not dead[i] and len(self._action_buffer[i]) == 0:
                    fr = cur_px[i]
                    fr = fr[-1] if (hasattr(fr, "ndim") and fr.ndim == 4) else fr  # (C,H,W)
                    self._frame_buf[i].append(fr)

        replan = [i for i in range(n) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            dev = next(self.model.parameters()).device
            # goal-conditioned: encode the (already-transformed) goal frame -> z_goal.
            goal_emb = None
            if self.goal_conditioned and "goal" in info_dict:
                gpx = info_dict["goal"][replan]
                gpx = gpx[:, -1:] if gpx.ndim == 5 else gpx.unsqueeze(1)  # -> (R,1,C,H,W) single goal frame
                gpx = gpx.to(dev).float()
                goal_emb = self.model.encode({"pixels": gpx})["emb"][:, 0]  # (R, D)
            # remaining-horizon (normalized) for the AdaLN-Zero hook; None -> identity.
            horizon_norm = None
            if self.use_horizon and self._steps_left is not None:
                steps = np.maximum(self._steps_left[replan], 1.0)
                hn = np.minimum(steps, self.H_max) / self.H_max
                horizon_norm = torch.tensor(hn, device=dev, dtype=torch.float32)

            if self.use_horizon and self._frame_buf is not None:
                # feed buffered HS frames + past-action blocks a_{<t} to rebuild the training context;
                # per-env histories differ in length early, so process each env then re-stack.
                blocks = torch.full((len(replan), self.action_block * self.action_dim), float("nan"))
                for row, i in enumerate(replan):
                    frames = list(self._frame_buf[i])                 # up to HS (C,H,W)
                    px_hist = torch.stack(frames, dim=0).unsqueeze(0).to(dev).float()  # (1,T0,C,H,W)
                    pab = None
                    if self.use_history and len(self._past_act_buf[i]) > 0:
                        # a_{<t} = last (T0-1) committed blocks; action_encoder consumes raw blocks (no z-score), as in training.
                        n_past = px_hist.size(1) - 1
                        if n_past > 0:
                            past = list(self._past_act_buf[i])[-n_past:]
                            pab = torch.stack(past, dim=0).unsqueeze(0).to(dev).float()  # (1,n_past,Adim)
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

        action = torch.full((n, self.action_dim), float("nan"))
        for i in range(n):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


# GC-IDM: planning-free goal-conditioned IDM on frozen LeWM latents (arXiv 2605.08732). mode=gcidm. NO CEM / NO WM.
def load_gcidm_model(run_name):
    """Load a trained GC-IDM: the FROZEN LeWM encoder + the GCIDMHead.

    The trainer (train_gcidm.py) writes checkpoints/<run_name>/gcidm_config.json
    (head dims, frozen-LeWM weights path, action z-score stats) and
    gcidm_head_best.pt. We rebuild the frozen LeWM exactly like train_gcidm
    (so z = encode({pixels})['emb'][:,0] matches) and the GCIDMHead,
    then load the head weights. Returns (lewm, head, gcfg)."""
    import lewam.models.gcidm as _gcidm
    from train_gcidm import build_frozen_lewm

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
                 process=None, transform=None, ablate_horizon=False, **kw):
        super().__init__(**kw)
        self.type = "gcidm_policy"
        self.lewm = lewm.eval()
        self.head = head.eval()
        self.process = process or {}
        self.transform = transform or {}
        self.action_block = int(action_block)
        self.action_dim = int(action_dim)
        self.H_max = int(H_max)
        self.horizon0 = float(horizon0)
        self.ablate_horizon = bool(ablate_horizon)
        self._action_buffer = None
        self._steps_left = None  # per-env remaining obs-steps

    def set_env(self, env):
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._action_buffer = [deque() for _ in range(n)]
        self._steps_left = np.full(n, self.horizon0, dtype=np.float64)

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(n)]
            self._steps_left = np.full(n, self.horizon0, dtype=np.float64)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._action_buffer[i].clear()
                    self._steps_left[i] = self.horizon0  # reset countdown for the new episode

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        replan = [i for i in range(n) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            dev = next(self.head.parameters()).device
            px = info_dict["pixels"][replan]
            assert "goal" in info_dict, "GCIDM policy needs info_dict['goal'] (goal-reaching eval)"
            gpx = info_dict["goal"][replan]
            gpx = gpx[:, -1:] if gpx.ndim == 5 else gpx.unsqueeze(1)  # (R,1,C,H,W) single goal frame
            cpx = px[:, -1:] if px.ndim == 5 else px.unsqueeze(1)     # (R,1,C,H,W) current frame
            z_t = self.lewm.encode({"pixels": cpx.to(dev).float()})["emb"][:, 0]   # (R, D)
            z_g = self.lewm.encode({"pixels": gpx.to(dev).float()})["emb"][:, 0]   # (R, D)
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=dev, dtype=torch.float32)
            if self.ablate_horizon:
                h_norm = torch.zeros_like(h_norm)
            block = self.head(z_t, z_g, h_norm)                       # (R, Adim)
            block = block.reshape(len(replan), self.action_block, self.action_dim).cpu()
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(block[row])
                self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)  # one obs-step consumed

        action = torch.full((n, self.action_dim), float("nan"))
        for i in range(n):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


# LeWAM-Seq eval adapter (modes: seq_policy = get_action AR/BC; seq_cem = get_cost CEM)
class LeWAMSeqPolicy(BasePolicy):
    """Eval adapter for LeWAM-Seq (lewam.models.lewam_seq.LeWAMSeq).

    Two planners share one sliding history buffer -- the last `num_frames` observed
    obs-frames plus the z-scored action blocks emitted between them (exactly the
    context the model saw in training):
      plan_mode='policy'  ->  model.get_action (AR/BC): one goal-conditioned action
                              block per replan straight from the action head; reactive
                              receding-horizon (replan every obs-step on the real frame).
      plan_mode='cem'     ->  CEM over model.get_cost: sample z-scored action-block
                              sequences, roll them through the dynamics head, keep the
                              lowest-cost elites, execute the best FIRST block (MPC).

    Action normalization uses the model's OWN z-score (action_mean/std, frameskip,
    action_raw_dim from lewam_seq_config.json): past-action blocks fed to the model
    are z-scored, and the action head's z-scored output block is un-z-scored back to
    raw env actions. The policy therefore ignores process['action'] (feeding it a
    process would double-normalize) -- only `transform` (pixels/goal) is applied.

    One replan == one obs-step == one action block == `action_block`(frameskip) env
    steps. The horizon countdown (obs-steps to the goal frame, init horizon0 =
    goal_offset/frameskip, decremented per replan, clamped >=1, fed as min(h,H_max)/H_max)
    mirrors GCIDMPolicy so the head sees the same remaining-horizon signal as training.
    """

    def __init__(self, model, cfg, action_block, action_dim, plan_mode="policy",
                 horizon0=None, H_max=50, process=None, transform=None,
                 goal_conditioned=True, cem_horizon=5, cem_samples=256, cem_iters=3,
                 cem_elites=32, history_size=None, **kw):
        super().__init__(**kw)
        self.type = f"lewam_seq_{plan_mode}"
        self.model = model.eval()  # BN-in-projector -> eval() = deterministic running stats
        self.cfg = cfg
        self.num_frames = int(cfg["num_frames"])
        # eval context window: <=num_frames. history_size=1 -> memoryless reactive (like GCIDM);
        # smaller windows trade the learned history for less trajectory covariate-shift at eval.
        self.hist = int(history_size) if history_size else self.num_frames
        self.frameskip = int(cfg["frameskip"])
        self.raw_adim = int(cfg["action_raw_dim"])
        self.block_dim = int(cfg["action_dim"])           # frameskip * raw_adim
        self.H_max = int(cfg.get("H_max", H_max))
        self.action_block = int(action_block)
        self.action_dim = int(action_dim)                 # per-step raw dim = raw_adim
        self.plan_mode = str(plan_mode)
        self.goal_conditioned = bool(goal_conditioned)
        self.horizon0 = float(horizon0) if horizon0 is not None else float(H_max)
        self.transform = transform or {}
        self.process = {}                                 # action un-norm handled internally
        dev = next(model.parameters()).device
        self._amean = torch.tensor(cfg["action_mean"], dtype=torch.float32, device=dev)  # (raw_adim,)
        self._astd = torch.tensor(cfg["action_std"], dtype=torch.float32, device=dev).clamp_min(1e-6)
        # CEM knobs (plan_mode='cem' only)
        self.cem_horizon = int(cem_horizon); self.cem_samples = int(cem_samples)
        self.cem_iters = int(cem_iters); self.cem_elites = int(cem_elites)
        # per-env state (allocated in set_env)
        self._action_buffer = None
        self._frame_buf = None       # deque(maxlen=num_frames) of the last obs-frames (C,H,W)
        self._pastblk_buf = None     # deque(maxlen=num_frames-1) of z-scored blocks (block_dim,)
        self._steps_left = None

    def set_env(self, env):
        self.env = env
        self._reset_bufs(getattr(env, "num_envs", 1))

    def _reset_bufs(self, n):
        self._action_buffer = [deque() for _ in range(n)]
        self._frame_buf = [deque(maxlen=self.hist) for _ in range(n)]
        self._pastblk_buf = [deque(maxlen=max(self.hist - 1, 0)) for _ in range(n)]
        self._steps_left = np.full(n, self.horizon0, dtype=np.float64)

    def _zscore(self, raw_block):
        x = raw_block.reshape(self.frameskip, self.raw_adim)
        return ((x - self._amean) / self._astd).reshape(-1)

    def _unzscore(self, z_block):
        x = z_block.reshape(self.frameskip, self.raw_adim)
        return (x * self._astd + self._amean).reshape(-1)

    @torch.no_grad()
    def _cem_block(self, px_hist, past_actions, goal_px, h0):
        """CEM over z-scored action-block sequences (train actions ~ N(0,1) -> N(0,1)
        prior). Rolls candidates through model.get_cost, refits on the elites, returns
        the best FIRST block (z-scored)."""
        dev = next(self.model.parameters()).device
        Hz, S, E = self.cem_horizon, self.cem_samples, self.cem_elites
        mean = torch.zeros(Hz, self.block_dim, device=dev)
        std = torch.ones(Hz, self.block_dim, device=dev)
        for _ in range(self.cem_iters):
            cand = mean[None] + std[None] * torch.randn(S, Hz, self.block_dim, device=dev)
            cost = self.model.get_cost(px_hist, past_actions, cand.unsqueeze(0),
                                       goal_px, history_size=self.hist)[0]   # (S,)
            elite = cand[cost.topk(E, largest=False).indices]                      # (E,Hz,bd)
            mean, std = elite.mean(0), elite.std(0).clamp_min(1e-3)
        return mean[0]

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        dev = next(self.model.parameters()).device
        if self._action_buffer is None:
            self._reset_bufs(n)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._action_buffer[i].clear(); self._frame_buf[i].clear()
                    self._pastblk_buf[i].clear(); self._steps_left[i] = self.horizon0

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        # one replan == one obs-step: push the current frame into the sliding window
        cur_px = info_dict["pixels"]  # (n,T,C,H,W) or (n,C,H,W), preprocessed
        for i in range(n):
            if not dead[i] and len(self._action_buffer[i]) == 0:
                fr = cur_px[i]
                fr = fr[-1] if (hasattr(fr, "ndim") and fr.ndim == 4) else fr  # (C,H,W)
                self._frame_buf[i].append(fr.to(dev).float())

        replan = [i for i in range(n) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            gpx = info_dict.get("goal")
            if self.goal_conditioned or self.plan_mode == "cem":
                assert gpx is not None, "LeWAM-Seq eval needs info_dict['goal'] (goal-reaching)"
            for i in replan:
                frames = list(self._frame_buf[i])
                px_hist = torch.stack(frames, dim=0).unsqueeze(0)              # (1,T0,C,H,W)
                T0 = px_hist.size(1)
                past = list(self._pastblk_buf[i])[-(T0 - 1):] if T0 > 1 else []
                past_actions = (torch.stack(past, dim=0).unsqueeze(0).to(dev) if past
                                else px_hist.new_zeros(1, 0, self.block_dim))  # (1,T0-1,bd)
                g = None
                if gpx is not None and (self.goal_conditioned or self.plan_mode == "cem"):
                    gi = gpx[i]
                    gi = gi[-1] if (hasattr(gi, "ndim") and gi.ndim == 4) else gi  # (C,H,W)
                    g = gi.unsqueeze(0).to(dev).float()                          # (1,C,H,W)
                steps = max(self._steps_left[i], 1.0)
                h0 = float(min(steps, self.H_max) / self.H_max)
                if self.plan_mode == "cem":
                    z_blk = self._cem_block(px_hist, past_actions, g, h0)        # (bd,) z-scored
                else:
                    z_blk = self.model.get_action(
                        px_hist, past_actions, horizon=1, z_goal_pixels=g,
                        h_norm0=h0, H_max=self.H_max, history_size=self.hist)[0, 0]
                self._pastblk_buf[i].append(z_blk.detach().to(dev))             # feed next a_{<t}
                raw = self._unzscore(z_blk).reshape(self.action_block, self.action_dim).cpu()
                self._action_buffer[i].extend(raw)
                self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)

        action = torch.full((n, self.action_dim), float("nan"))
        for i in range(n):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()


# LeWAM-Split eval adapter (mode: split_policy = reactive one-step goal-conditioned)
class LeWAMSplitPolicy(BasePolicy):
    """Eval adapter for LeWAM-Split (lewam.models.lewam_split.LeWAMSplit), the split
    goal-conditioned model trained by scripts/train_lewam_gc.py. Reactive one-step GC
    policy (mode=split_policy) -- structurally identical to GCIDMPolicy, but reading the
    split model's OWN encoder + gc_head instead of a frozen LeWM + separate GCIDMHead:

      z_t   = model.encode(current pixels)          # (R, D) cls latent (projector applied)
      z_goal= model.encode(goal pixels)             # (R, D)
      h     = remaining horizon in OBS-steps (init horizon0 = goal_offset/frameskip,
              decremented per replan, clamped >=1, normalized min(h,H_max)/H_max)
      a     = model.gc_head(z_t, z_goal, h_norm)    # (R, block_dim) z-scored action block

    The z-scored block is un-z-scored with the model's OWN stats (action_mean/std,
    frameskip, action_raw_dim from lewam_gc_config.json -- the exact inverse of the
    trainer's per-raw-dim z-score), unstacked into `action_block` env actions, and
    executed before re-observing. It ignores process['action'] (un-norm handled
    internally, so no double-normalization). NO CEM, NO WM rollout. model.eval() is
    forced (BatchNorm in the ViT projector). This is the head-off-raw-z_t reactive
    baseline the seq policy's AR rollout is compared against."""

    def __init__(self, model, cfg, action_block, action_dim, horizon0=None,
                 H_max=50, process=None, transform=None, ablate_horizon=False, **kw):
        super().__init__(**kw)
        self.type = "lewam_split_policy"
        self.model = model.eval()  # BN-in-projector -> eval() = deterministic running stats
        self.cfg = cfg
        self.frameskip = int(cfg["frameskip"])
        self.raw_adim = int(cfg["action_raw_dim"])
        self.block_dim = int(cfg["action_dim"])           # frameskip * raw_adim
        self.action_block = int(action_block)
        self.action_dim = int(action_dim)                 # per-step raw dim = raw_adim
        self.H_max = int(cfg.get("H_max", H_max))
        self.horizon0 = float(horizon0) if horizon0 is not None else float(self.H_max)
        self.ablate_horizon = bool(ablate_horizon)
        self.transform = transform or {}
        self.process = {}                                 # action un-norm handled internally
        dev = next(model.parameters()).device
        self._amean = torch.tensor(cfg["action_mean"], dtype=torch.float32, device=dev)  # (raw_adim,)
        self._astd = torch.tensor(cfg["action_std"], dtype=torch.float32, device=dev).clamp_min(1e-6)
        self._action_buffer = None
        self._steps_left = None  # per-env remaining obs-steps

    def set_env(self, env):
        self.env = env
        n = getattr(env, "num_envs", 1)
        self._action_buffer = [deque() for _ in range(n)]
        self._steps_left = np.full(n, self.horizon0, dtype=np.float64)

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        dev = next(self.model.parameters()).device
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(n)]
            self._steps_left = np.full(n, self.horizon0, dtype=np.float64)

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._action_buffer[i].clear()
                    self._steps_left[i] = self.horizon0  # reset countdown for the new episode

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        replan = [i for i in range(n) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            px = info_dict["pixels"][replan]
            assert "goal" in info_dict, "LeWAM-Split eval needs info_dict['goal'] (goal-reaching)"
            gpx = info_dict["goal"][replan]
            gpx = gpx[:, -1] if gpx.ndim == 5 else gpx  # (R,C,H,W) single goal frame
            cpx = px[:, -1] if px.ndim == 5 else px     # (R,C,H,W) current frame
            z_t = self.model.encode(cpx.to(dev).float())   # (R, D)
            z_g = self.model.encode(gpx.to(dev).float())   # (R, D)
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=dev, dtype=torch.float32)
            if self.ablate_horizon:
                h_norm = torch.zeros_like(h_norm)
            z_blk = self.model.gc_head(z_t, z_g, h_norm)   # (R, block_dim) z-scored
            raw = (z_blk.reshape(len(replan), self.frameskip, self.raw_adim)
                   * self._astd + self._amean)             # un-z-score per raw dim
            raw = raw.reshape(len(replan), self.action_block, self.action_dim).cpu()
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(raw[row])
                self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)  # one obs-step consumed

        action = torch.full((n, self.action_dim), float("nan"))
        for i in range(n):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()


# LeWAM-Unified eval adapter (mode: unified_policy = reactive GC over a state window)
class LeWAMUnifiedPolicy(LeWAMSplitPolicy):
    """Eval adapter for LeWAM-Unified. Structurally the split policy, but instead of a
    single-frame z_t it maintains a per-env deque of the last `window` observed frames
    (one per obs-step / replan), encodes that window, and reads the action off the
    aggregated context latent c_t = z_t + Aggr([z_{t-k..t}]):

      window = [last W observed frames]            # left-padded (repeat oldest) to length W
      z_win  = model.encode(window)                # (R, W, D)
      c_t    = model.aggregate(z_win)              # (R, D) = z_t + temporal correction
      a      = model.gc_head(c_t, z_goal, h_norm)  # (R, block_dim) z-scored action block

    Frame buffering matches training's causal window (clamp to episode start = repeat the
    first observed frame). Everything else (goal encode, horizon countdown, un-z-score,
    action-block unstack, model.eval()) is inherited from LeWAMSplitPolicy unchanged."""

    def __init__(self, model, cfg, *a, **kw):
        super().__init__(model, cfg, *a, **kw)
        self.type = "lewam_unified_policy"
        self.window = int(cfg["window"])
        self.action_cond = bool(cfg.get("agg_action_cond", False))
        self._frame_buf = None   # per-env deque of the last W transformed frames
        self._pblk_buf = None    # per-env deque of the z-scored block that led INTO each frame
        self._last_blk = None    # per-env last z-scored block emitted (the next frame's prev-action)

    def set_env(self, env):
        super().set_env(env)
        n = getattr(env, "num_envs", 1)
        self._frame_buf = [deque(maxlen=self.window) for _ in range(n)]
        self._pblk_buf = [deque(maxlen=self.window) for _ in range(n)]
        self._last_blk = [None] * n

    def _build_window(self, i):
        """Left-pad the env-i frame deque to length W by repeating the oldest frame."""
        buf = list(self._frame_buf[i])
        pad = [buf[0]] * (self.window - len(buf))
        return torch.stack(pad + buf, dim=0)  # (W, C, H, W)

    def _build_prev_action(self, i):
        """Previous-action window aligned to the frame window: a_prev[j] is the z-scored block
        that led into frame j (None -> null via mask; left-pad positions -> null). Mirrors
        training's a_{tau-1} with the episode-start/left-pad positions masked out."""
        buf = list(self._pblk_buf[i])
        pad = self.window - len(buf)
        a_prev, mask = [], []
        zero = torch.zeros(self.block_dim)
        for _ in range(pad):
            a_prev.append(zero); mask.append(False)
        for blk in buf:
            if blk is None:
                a_prev.append(zero); mask.append(False)
            else:
                a_prev.append(blk); mask.append(True)
        return torch.stack(a_prev, dim=0), torch.tensor(mask, dtype=torch.bool)  # (W,blk),(W,)

    @torch.no_grad()
    def get_action(self, info_dict, **kw):
        info_dict = self._prepare_info(info_dict)
        n = self.env.num_envs
        dev = next(self.model.parameters()).device
        if self._action_buffer is None:
            self._action_buffer = [deque() for _ in range(n)]
            self._steps_left = np.full(n, self.horizon0, dtype=np.float64)
        if self._frame_buf is None:
            self._frame_buf = [deque(maxlen=self.window) for _ in range(n)]
            self._pblk_buf = [deque(maxlen=self.window) for _ in range(n)]
            self._last_blk = [None] * n

        flush = info_dict.pop("_needs_flush", None)
        if flush is not None:
            for i in range(n):
                if flush[i]:
                    self._action_buffer[i].clear()
                    self._frame_buf[i].clear()   # new episode: drop the old window
                    self._pblk_buf[i].clear()
                    self._last_blk[i] = None
                    self._steps_left[i] = self.horizon0

        term = info_dict.get("terminated")
        dead = np.asarray(term, dtype=bool) if term is not None else np.zeros(n, dtype=bool)

        replan = [i for i in range(n) if len(self._action_buffer[i]) == 0 and not dead[i]]
        if replan:
            px = info_dict["pixels"][replan]
            assert "goal" in info_dict, "LeWAM-Unified eval needs info_dict['goal'] (goal-reaching)"
            gpx = info_dict["goal"][replan]
            gpx = gpx[:, -1] if gpx.ndim == 5 else gpx  # (R,C,H,W) single goal frame
            cpx = px[:, -1] if px.ndim == 5 else px     # (R,C,H,W) current frame
            # push current frame + the block that led into it (prev emitted block) into the buffers
            for row, i in enumerate(replan):
                self._frame_buf[i].append(cpx[row])
                self._pblk_buf[i].append(self._last_blk[i])  # None at episode start -> null
            windows = torch.stack([self._build_window(i) for i in replan], dim=0)  # (R,W,C,H,W)
            R, Wn = windows.shape[0], windows.shape[1]
            z_win = self.model.encode(windows.reshape(R * Wn, *windows.shape[2:]).to(dev).float())
            z_win = z_win.reshape(R, Wn, -1)
            z_g = self.model.encode(gpx.to(dev).float())   # (R, D)
            steps = np.maximum(self._steps_left[replan], 1.0)
            h_norm = torch.tensor(np.minimum(steps, self.H_max) / self.H_max,
                                  device=dev, dtype=torch.float32)
            if self.ablate_horizon:
                h_norm = torch.zeros_like(h_norm)
            a_prev = a_prev_mask = None
            if self.action_cond:
                pa = [self._build_prev_action(i) for i in replan]
                a_prev = torch.stack([p[0] for p in pa], dim=0).to(dev).float()   # (R,W,blk)
                a_prev_mask = torch.stack([p[1] for p in pa], dim=0).to(dev)      # (R,W)
            # gc_action encapsulates aggregate (+ action-cond) -> arm-agnostic across ablations
            z_blk = self.model.gc_action(z_win, z_g, h_norm, a_prev, a_prev_mask)  # (R,blk) z-scored
            for row, i in enumerate(replan):
                self._last_blk[i] = z_blk[row].detach().cpu()  # z-scored, for the next frame's prev
            raw = (z_blk.reshape(len(replan), self.frameskip, self.raw_adim)
                   * self._astd + self._amean)             # un-z-score per raw dim
            raw = raw.reshape(len(replan), self.action_block, self.action_dim).cpu()
            for row, i in enumerate(replan):
                self._action_buffer[i].extend(raw[row])
                self._steps_left[i] = max(self._steps_left[i] - 1.0, 1.0)  # one obs-step consumed

        action = torch.full((n, self.action_dim), float("nan"))
        for i in range(n):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        return action.reshape(*self.env.action_space.shape).float().numpy()


# policy factory (the one config switch)
def build_policy(cfg, model, adim, process, transform):
    """Dispatch on cfg.gip_eval.mode -> a configured policy."""
    mode = cfg.get("gip_eval", {}).get("mode", "bc")
    goal_conditioned = bool(cfg.get("gip_eval", {}).get("goal_conditioned", False))
    action_block = int(cfg.plan_config.action_block)

    # mode=seq_policy | seq_cem: LeWAM-Seq adapter. `model` is a loaded LeWAMSeq with its
    # config attached as model._seq_cfg (done in eval_gip.py). action_block(=frameskip)
    # splits the model's z-scored action block into per-step raw env actions.
    if mode in ("seq_policy", "seq_cem"):
        ge = cfg.get("gip_eval", {})
        seq_cfg = getattr(model, "_seq_cfg")
        horizon0 = ge.get("horizon0", None)
        if horizon0 is None:
            horizon0 = float(cfg.eval.goal_offset_steps) / float(action_block)
        return LeWAMSeqPolicy(
            model=model, cfg=seq_cfg, action_block=action_block,
            action_dim=adim // action_block, plan_mode=("cem" if mode == "seq_cem" else "policy"),
            horizon0=float(horizon0), H_max=int(ge.get("horizon_H_max", seq_cfg.get("H_max", 50))),
            process=process, transform=transform,
            goal_conditioned=bool(ge.get("goal_conditioned", True)),
            cem_horizon=int(ge.get("cem_horizon", cfg.plan_config.horizon)),
            cem_samples=int(ge.get("cem_samples", 256)), cem_iters=int(ge.get("cem_iters", 3)),
            cem_elites=int(ge.get("cem_elites", 32)),
            history_size=(int(ge["history_size"]) if ge.get("history_size") else None),
        )

    # mode=split_policy: LeWAM-Split adapter. `model` is a loaded LeWAMSplit with its config
    # attached as model._split_cfg (done in eval_gip.py). Reactive one-step GC, no CEM.
    if mode == "split_policy":
        ge = cfg.get("gip_eval", {})
        split_cfg = getattr(model, "_split_cfg")
        horizon0 = ge.get("horizon0", None)
        if horizon0 is None:
            horizon0 = float(cfg.eval.goal_offset_steps) / float(action_block)
        return LeWAMSplitPolicy(
            model=model, cfg=split_cfg, action_block=action_block,
            action_dim=adim // action_block, horizon0=float(horizon0),
            H_max=int(ge.get("horizon_H_max", split_cfg.get("H_max", 50))),
            process=process, transform=transform,
            ablate_horizon=bool(ge.get("ablate_horizon", split_cfg.get("ablate_horizon", False))),
        )

    # mode=unified_policy: LeWAM-Unified adapter. `model` is a loaded LeWAMUnified with its config
    # attached as model._unified_cfg (done in eval_gip.py). Reactive GC over a state window, no CEM.
    if mode == "unified_policy":
        ge = cfg.get("gip_eval", {})
        uni_cfg = getattr(model, "_unified_cfg")
        horizon0 = ge.get("horizon0", None)
        if horizon0 is None:
            horizon0 = float(cfg.eval.goal_offset_steps) / float(action_block)
        return LeWAMUnifiedPolicy(
            model=model, cfg=uni_cfg, action_block=action_block,
            action_dim=adim // action_block, horizon0=float(horizon0),
            H_max=int(ge.get("horizon_H_max", uni_cfg.get("H_max", 50))),
            process=process, transform=transform,
            ablate_horizon=bool(ge.get("ablate_horizon", uni_cfg.get("ablate_horizon", False))),
        )

    # mode=gcidm: the `model` arg is unused; GCIDM loads its OWN frozen-LeWM + head via load_gcidm_model.
    if mode == "gcidm":
        gc = cfg.gip_eval
        lewm, head, gcfg = load_gcidm_model(gc.get("gcidm_run", cfg.policy))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        lewm = lewm.to(dev); head = head.to(dev)
        # horizon0 = goal_offset(raw frames)/frameskip(=action_block) = obs-steps to the goal frame. Override via gip_eval.gcidm_horizon.
        horizon0 = gc.get("gcidm_horizon", None)
        if horizon0 is None:
            horizon0 = float(cfg.eval.goal_offset_steps) / float(action_block)
        return GCIDMPolicy(
            lewm=lewm, head=head, action_block=action_block,
            action_dim=int(gcfg["action_dim"]) // action_block,
            H_max=int(gcfg["H_max"]), horizon0=float(horizon0),
            process=process, transform=transform,
            ablate_horizon=bool(gc.get("ablate_horizon", gcfg.get("ablate_horizon", False))),
        )

    # mode=bc and mode=policy are the SAME direct forward policy; `policy` exposes the goal_conditioned
    # switch (bc == policy with goal_conditioned=false). One forward pass, NO solver.
    if mode in ("bc", "policy"):
        ge = cfg.get("gip_eval", {})
        # horizon0 = goal_offset/frameskip (obs-steps to goal); override via gip_eval.horizon0.
        # H_max MUST match training H_max (action_pred.horizon_H_max, default 50).
        horizon0 = ge.get("horizon0", None)
        if horizon0 is None:
            horizon0 = float(cfg.eval.goal_offset_steps) / float(action_block)
        return BCPolicy(
            model=model,
            action_block=action_block,
            action_dim=adim // action_block,
            process=process,
            transform=transform,
            goal_conditioned=(mode == "policy" and goal_conditioned),
            H_max=int(ge.get("horizon_H_max", 50)),
            horizon0=float(horizon0),
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
        raise ValueError(f"unknown gip_eval.mode={mode!r} (bc|policy|guided|planning)")

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra_instantiate(cfg.solver, model=model)
    return swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )


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
