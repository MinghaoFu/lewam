"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate as hydra_instantiate

# AdaLN-Zero horizon conditioning, reused from gcidm.py (sinusoidal(64)->MLP->(scale,shift)).
from lewam.models.gcidm import _sinusoidal_embedding


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class HorizonModulator(nn.Module):
    """AdaLN-Zero horizon conditioning for OUR history-conditioned forward GC policy.

    Modulates the intention embedding (the action-head output, BEFORE the action
    decoder) by a (scale, shift) derived from the remaining horizon h, EXACTLY the
    AdaLN-Zero machinery gcidm.GCIDMHead uses (sinusoidal(64 freqs)->2-layer MLP->
    per-target (scale,shift), with the (scale,shift) projection ZERO-INITIALIZED so
    at init it is the identity). This is what makes the OURS vs GC-IDM comparison
    fair: both arms inject the remaining horizon through the same AdaLN-Zero hook.

    forward(x, h_norm):
      x      : (B, T, D) intention embedding (or (B, D))
      h_norm : (B,) float in [0,1]  = min(steps_left, H_max)/H_max  (caller owns
               the normalization so train and eval agree, mirroring GCIDMHead).
    AdaLN-Zero: at init scale=shift=0 so x is unchanged (identity at init),
    matching the horizon-agnostic policy until the conditioning is learned.
    """

    def __init__(self, emb_dim, n_freqs=64, cond_dim=128):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.n_freqs = int(n_freqs)
        sin_dim = 2 * self.n_freqs  # 128, matches gcidm
        self.horizon_mlp = nn.Sequential(
            nn.Linear(sin_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.norm = nn.LayerNorm(self.emb_dim, elementwise_affine=False, eps=1e-6)
        self.cond_proj = nn.Linear(cond_dim, 2 * self.emb_dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)  # AdaLN-Zero: identity at init

    def forward(self, x, h_norm):
        # h_norm: (B,) -> cond (B, cond_dim) -> (scale, shift) each (B, emb_dim)
        cond = self.horizon_mlp(_sinusoidal_embedding(h_norm.float(), self.n_freqs))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        if x.dim() == 3:  # (B, T, D): broadcast the per-sample (scale, shift) over T
            scale = scale[:, None, :]
            shift = shift[:, None, :]
        return self.norm(x) * (1 + scale) + shift

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        action_predictor=None,
        action_decoder=None,
        use_action_history=True,
        proprio_encoder=None,
        proprio_pred_proj=None,
        use_proprio=False,
        mt_task_names=None,
        action_head=None,
        horizon_conditioned=False,
        inverse_conditioned=False,
        inverse_action_dim=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        # GIP step 1: intention predictor (None -> vanilla LeWM).
        self.action_predictor = action_predictor
        self.action_decoder = action_decoder
        # state-only intention zeros the past-action stream to the action head (WM state head stays action-conditioned).
        self.use_action_history = use_action_history
        # proprio-as-token: encode proprio into a state token + predict it.
        self.proprio_encoder = proprio_encoder
        self.proprio_pred_proj = proprio_pred_proj
        self.use_proprio = use_proprio
        # multi-task: ordered task list (row i of task_table = mt_task_names[i]).
        self.mt_task_names = list(mt_task_names) if mt_task_names else None
        # action-head spec ({"type": mse|gmm|diffusion, ...}); None => mse (MLP). Saved in config.json so eval rebuilds it.
        self.action_head = action_head
        # optional AdaLN-Zero horizon conditioning on the intention embedding (see HorizonModulator).
        # Flag rides in config.json so load_gip_model rebuilds the SAME module before load_state_dict.
        self.horizon_conditioned = bool(horizon_conditioned)
        self.horizon_modulator = None
        if self.horizon_conditioned:
            # emb_dim from the predictor's pos_embedding; the action predictor mirrors it, so the intention dim matches.
            emb_dim = predictor.pos_embedding.shape[-1]
            self.horizon_modulator = HorizonModulator(emb_dim=emb_dim, n_freqs=64)

        # SMWM (2606.20104) inverse-dynamics anti-collapse head: MLP h([z_tau ; z_{tau+1}]) -> a_hat_tau,
        # train-time only (L_inv in lejepa_forward, action_pred.w_inv>0); NOT in the deployed policy.
        # Flag rides in config.json so load_gip_model rebuilds the SAME head before load_state_dict.
        # Shape 2*D -> 256 -> Adim (D=192 vit-tiny, Adim = frameskip*action_dim).
        self.inverse_conditioned = bool(inverse_conditioned)
        self.inverse_model = None
        if self.inverse_conditioned:
            emb_dim = predictor.pos_embedding.shape[-1]
            adim = int(inverse_action_dim) if inverse_action_dim is not None else int(action_encoder.patch_embed.weight.shape[1])
            self.inverse_model = nn.Sequential(
                nn.Linear(2 * emb_dim, 256),
                nn.GELU(),
                nn.Linear(256, adim),
            )

    def predict_inverse(self, z_t, z_tp1):
        """SMWM inverse head: read the action off a consecutive latent pair.
        z_t, z_tp1: (..., D). Returns a_hat: (..., Adim). Train-time only."""
        return self.inverse_model(torch.cat([z_t, z_tp1], dim=-1))

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            # multi-task eval: env-space actions are padded to the trained block (no-op otherwise)
            info["action"] = self._pad_eval_action(info["action"])
            info["act_emb"] = self.action_encoder(info["action"])

        if self.use_proprio and self.proprio_encoder is not None and "proprio" in info:
            pr = info["proprio"]
            # multi-task eval: pad env proprio to the trained width (no-op when equal)
            ped = self.proprio_encoder.patch_embed.weight.shape[1]
            if pr.shape[-1] < ped:
                pr = F.pad(pr.float(), (0, ped - pr.shape[-1]))
                info["proprio"] = pr
            info["proprio_emb"] = self.proprio_encoder(pr)

        # Multi-task: frozen language-embedding table + learned projection (Newt-style). Conditions the
        # predictors' AdaLN streams via task_vec; NOT added to act_emb, so intention targets stay task-unshifted.
        if getattr(self, "task_proj", None) is not None and "task_id" in info:
            info["task_vec"] = self.task_proj(self.task_table[info["task_id"]])  # (B, D)

        return info

    def predict(self, emb, act_emb, proprio_emb=None, task_vec=None):
        """Predict next state embedding.
        emb: (B, T, D), act_emb: (B, T, A_emb), proprio_emb: (B, T, D) or None,
        task_vec: (B, D) or None -- added to the action-conditioning stream (multi-task).
        Returns next-emb (B,T,D), or (next_emb, next_proprio_emb) if proprio_emb given.
        """
        b = emb.size(0)
        if task_vec is not None:
            act_emb = act_emb + task_vec[:, None, :]
        preds = self.predictor(emb, act_emb, proprio_emb)
        if proprio_emb is None:
            preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
            return rearrange(preds, "(b t) d -> b t d", b=b)
        pix_out, prop_out = preds
        pix = self.pred_proj(rearrange(pix_out, "b t d -> (b t) d"))
        pix = rearrange(pix, "(b t) d -> b t d", b=b)
        prop = self.proprio_pred_proj(rearrange(prop_out, "b t d -> (b t) d"))
        prop = rearrange(prop, "(b t) d -> b t d", b=b)
        return pix, prop

    def predict_intention(self, emb, past_act_emb, detach_decoder=False, proprio_emb=None, task_vec=None, goal_emb=None, decode=True, horizon=None):
        """GIP step 1: autoregressive policy. Predict the next action from the
        latent state history (emb = z_<=t) conditioned on shifted past actions
        (past_act_emb = a_<t, so position t never sees a_t).
        task_vec (B, D): multi-task conditioning, added to the past-action stream
        AFTER the state-only zeroing (so the task signal survives either mode).
        goal_emb (B, D): GOAL conditioning (z_goal) for the goal-conditioned policy
        setting -- threaded the SAME additive way as task_vec; None -> goal-agnostic
        (the original behaviour), so this is a strict superset (forward HISTORY-conditioned
        policy with an OPTIONAL goal, NOT a Markovian IDM).
        horizon (B,): remaining-horizon h ALREADY normalized to [0,1]
        (= min(steps_left, H_max)/H_max, caller owns the normalization, mirroring
        gcidm). Conditions the intention embedding via AdaLN-Zero (self.horizon_modulator).
        None or no modulator gives identity.

        Returns:
          intention: (B, T, D)     -- intention embedding (trained vs act_emb_t)
          action:    (B, T, Adim)  -- decoded raw action  (trained vs a_t)
        """
        b = emb.size(0)
        if not self.use_action_history:
            past_act_emb = torch.zeros_like(past_act_emb)  # state-only intention
        if task_vec is not None:
            past_act_emb = past_act_emb + task_vec[:, None, :]
        if goal_emb is not None:
            past_act_emb = past_act_emb + goal_emb[:, None, :]   # goal-conditioned policy (optional)
        out = self.action_predictor(emb, past_act_emb, proprio_emb)  # (B,T,D) or (pix,prop)
        # proprio position (last token): with causal [pix_t, prop_t] order, the only per-frame token that attended to BOTH (full state).
        intention = out if proprio_emb is None else out[1]
        # AdaLN-Zero horizon conditioning on the intention embedding, BEFORE the decoder,
        # so both intent_loss and the decoded action see it.
        if self.horizon_modulator is not None and horizon is not None:
            intention = self.horizon_modulator(intention, horizon)
        # gmm/diffusion train time: skip sampling; the loss is computed from the intention by action_decoder.loss (see lejepa_forward).
        if not decode:
            return intention, None
        dec_in = intention.detach() if detach_decoder else intention  # detached readout
        action = self.action_decoder(rearrange(dec_in, "b t d -> (b t) d"))
        action = rearrange(action, "(b t) d -> b t d", b=b)
        return intention, action

    def _pad_eval_action(self, a):
        """Multi-task eval: zero-pad env-space action blocks (..., f*d_raw) -> (..., f*d_max),
        matching the training-time per-frame padding layout ((f d), d fastest).
        No-op unless eval harness sets eval_action_pad=(d_raw, f). Idempotent."""
        ap = getattr(self, "eval_action_pad", None)
        if ap is None:
            return a
        d_raw, f = ap
        A = self.action_encoder.patch_embed.weight.shape[1]  # f * d_max
        if a.shape[-1] == A:
            return a
        d_max = A // f
        sh = a.shape[:-1]
        a = a.reshape(*sh, f, d_raw)
        a = F.pad(a, (0, d_max - d_raw))
        return a.reshape(*sh, f * d_max)

    def _unpad_eval_action(self, a):
        """Inverse of _pad_eval_action: slice each frame chunk back to the env's dims."""
        ap = getattr(self, "eval_action_pad", None)
        if ap is None:
            return a
        d_raw, f = ap
        d_max = self.action_encoder.patch_embed.weight.shape[1] // f
        if a.shape[-1] == f * d_raw:
            return a
        sh = a.shape[:-1]
        return a.reshape(*sh, f, d_max)[..., :d_raw].reshape(*sh, f * d_raw)

    def _eval_task_vec(self, b, device):
        """Inference-time task vector (multi-task only). Eval harnesses set
        `model.eval_task = <name or id>` once after load; single-task models
        (no task_proj) or unset eval_task return None -> conditioning unchanged."""
        if getattr(self, "task_proj", None) is None or getattr(self, "eval_task", None) is None:
            return None
        et = self.eval_task
        tid = self.mt_task_names.index(et) if isinstance(et, str) else int(et)
        tv = self.task_proj(self.task_table[tid].to(device))
        return tv[None].expand(b, -1)

    @torch.no_grad()
    def intention_rollout(self, info, horizon, prefix_actions=None, history_size: int = 3, goal_emb=None, horizon_norm=None, past_action_blocks=None):
        """GIP joint actor: the proposal used by intention-guided planning.

        Autoregressively rolls the policy with BOTH heads -- the intention head
        predicts a_t from (z_<=t, a_<t), the state head advances z_t -> z_{t+1}.
        Optional prefix_actions (an already-committed plan) are applied through
        the state head first, then the remaining `horizon` actions are proposed.

        info:           dict with 'pixels' (B, T0, C, H, W)
        prefix_actions: (B, n_prev, Adim) or None
        horizon_norm:   (B,) remaining-horizon h already normalized to [0,1]
                        (= min(steps_left, H_max)/H_max), or a python float, or None.
                        When self.horizon_modulator is set this is fed to the AdaLN-Zero
                        hook at every proposed step; None / no modulator gives identity.
                        The decrement-per-replan bookkeeping lives in the policy
                        (gip.BCPolicy), so train and eval feed h identically.
        returns:        (B, horizon, Adim), Adim = frameskip * action_dim
        Adapted to the Actionable protocol by gip.attach_intention_actor.
        """
        assert self.action_predictor is not None and self.action_decoder is not None, (
            "intention_rollout requires the GIP action modules (action_pred.enabled)"
        )
        HS = history_size
        device = next(self.parameters()).device
        px = info["pixels"]
        px = (px if torch.is_tensor(px) else torch.as_tensor(px)).to(device).float()
        b = px.size(0)
        enc_in = {"pixels": px}
        if self.use_proprio and self.proprio_encoder is not None and "proprio" in info:
            pr = info["proprio"]
            enc_in["proprio"] = (pr if torch.is_tensor(pr) else torch.as_tensor(pr)).to(device).float()
        enc = self.encode(enc_in)
        emb = enc["emb"]                                       # (B, T0, D)
        prop = enc.get("proprio_emb", None)                    # (B, T0, D) or None
        D = emb.size(-1)
        act_hist = torch.zeros(b, 0, D, device=device)         # embeddings of committed actions
        tv = self._eval_task_vec(b, device)                    # (B, D) or None (multi-task)
        # remaining-horizon (normalized) for the AdaLN-Zero hook.
        h_norm = None
        if self.horizon_modulator is not None and horizon_norm is not None:
            if torch.is_tensor(horizon_norm):
                h_norm = horizon_norm.to(device).float().reshape(-1)
                if h_norm.numel() == 1:
                    h_norm = h_norm.expand(b)
            else:
                h_norm = torch.full((b,), float(horizon_norm), device=device)

        def advance(emb, act_full):                            # state head: (z_t, a_t) -> z_{t+1}
            nonlocal prop
            if prop is None:
                return self.predict(emb[:, -HS:], act_full[:, -HS:], task_vec=tv)[:, -1:]
            # proprio-as-token: autoregress the proprio token through imagination
            p_pix, p_prop = self.predict(emb[:, -HS:], act_full[:, -HS:], prop[:, -HS:], task_vec=tv)
            prop = torch.cat([prop, p_prop[:, -1:]], dim=1)
            return p_pix[:, -1:]

        if prefix_actions is not None and prefix_actions.size(1) > 0:
            prefix = self._pad_eval_action(prefix_actions.to(device).float())
            for t in range(prefix.size(1)):
                ae = self.action_encoder(prefix[:, t : t + 1])
                act_hist = torch.cat([act_hist, ae], dim=1)
                emb = torch.cat([emb, advance(emb, act_hist)], dim=1)

        # eval: given the HS-frame history + committed past-action blocks a_{<t}, fill act_hist from those
        # raw blocks WITHOUT advancing the state head (the frames already ARE the observed states),
        # rebuilding the training context (z_{t-HS+1..t} + a_{<t}). None -> act_hist stays empty.
        if past_action_blocks is not None and past_action_blocks.size(1) > 0:
            pab = self._pad_eval_action(past_action_blocks.to(device).float())
            act_hist = self.action_encoder(pab)  # (B, n_past, D)

        actions = []
        for t in range(horizon):
            L = emb.size(1)
            past = torch.cat(                                  # a_<t aligned to emb (leading zeros = context)
                [torch.zeros(b, L - act_hist.size(1), D, device=device), act_hist], dim=1
            )
            pa = past[:, -HS:]
            if not self.use_action_history:
                pa = torch.zeros_like(pa)  # state-only intention
            if tv is not None:
                pa = pa + tv[:, None, :]   # multi-task conditioning (after zeroing)
            if goal_emb is not None:
                pa = pa + goal_emb[:, None, :]   # goal-conditioned policy (optional; same additive hook as tv)
            out = self.action_predictor(emb[:, -HS:], pa, None if prop is None else prop[:, -HS:])
            intention = out if prop is None else out[1]        # proprio position = full state
            # AdaLN-Zero horizon conditioning (same hook as predict_intention/GC-IDM)
            if h_norm is not None:
                intention = self.horizon_modulator(intention, h_norm)
            a = self.action_decoder(intention[:, -1])          # (B, Adim)
            actions.append(a)
            ae = self.action_encoder(a.unsqueeze(1))
            act_hist = torch.cat([act_hist, ae], dim=1)
            if t < horizon - 1:
                emb = torch.cat([emb, advance(emb, act_hist)], dim=1)

        # multi-task eval: slice decoded blocks back to the env's action dims (no-op otherwise)
        return self._unpad_eval_action(torch.stack(actions, dim=1))  # (B, horizon, Adim_env)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        prop = _init.get("proprio_emb", None)
        if prop is not None:
            prop = prop.unsqueeze(1).expand(B, S, -1, -1)
            prop = rearrange(prop, "b s ... -> (b s) ...").clone()
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = self._pad_eval_action(rearrange(act_0, "b s ... -> (b s) ..."))
        act_future = self._pad_eval_action(rearrange(act_future, "b s ... -> (b s) ..."))

        # rollout predictor autoregressively for n_steps
        # (proprio-as-token: the proprio token is autoregressed through imagination)
        HS = history_size
        tv = self._eval_task_vec(emb.size(0), emb.device)  # (BS, D) or None (multi-task)

        def _step(emb, act_emb):
            nonlocal prop
            if prop is None:
                return self.predict(emb[:, -HS:], act_emb[:, -HS:], task_vec=tv)[:, -1:]
            p_pix, p_prop = self.predict(emb[:, -HS:], act_emb[:, -HS:], prop[:, -HS:], task_vec=tv)
            prop = torch.cat([prop, p_prop[:, -1:]], dim=1)
            return p_pix[:, -1:]

        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            pred_emb = _step(emb, act_emb)  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        pred_emb = _step(emb, act_emb)  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost


################
#  Build/Load  #
################
def build_frozen_lewm(weights_path, embed_dim=192, history_size=3, img_size=224, action_block_dim=25):
    """Instantiate the LeWM JEPA (vit-tiny-192) and load the frozen weights.
    Mirrors train.py's init_from path so z = encode({pixels})['emb'][:,0] is the
    exact LeWM latent. Encoder/projector frozen; action_encoder kept for key match."""
    model_cfg = OmegaConf.create({
        "_target_": "lewam.models.jepa.JEPA",
        "use_action_history": True,
        "use_proprio": False,
        "encoder": {
            "_target_": "stable_pretraining.backbone.utils.vit_hf",
            "size": "tiny", "patch_size": 14, "image_size": img_size,
            "pretrained": False, "use_mask_token": False,
        },
        "predictor": {
            "_target_": "lewam.models.lewm.ARPredictor",
            "num_frames": history_size, "input_dim": embed_dim, "hidden_dim": embed_dim,
            "output_dim": embed_dim, "depth": 6, "heads": 16, "mlp_dim": 2048,
            "dim_head": 64, "dropout": 0.1, "emb_dropout": 0.0,
        },
        "action_encoder": {
            "_target_": "lewam.models.module.Embedder", "input_dim": action_block_dim, "emb_dim": embed_dim,
        },
        "projector": {
            "_target_": "lewam.models.lewm.MLP", "input_dim": embed_dim, "output_dim": embed_dim,
            "hidden_dim": 2048, "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
        },
        "pred_proj": {
            "_target_": "lewam.models.lewm.MLP", "input_dim": embed_dim, "output_dim": embed_dim,
            "hidden_dim": 2048, "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True},
        },
    })
    model = hydra_instantiate(model_cfg)
    if weights_path in (None, "self", "scratch"):
        # end-to-end / from-scratch model: the encoder weights live in the caller's
        # full-model checkpoint (loaded by load_gcidm_model), not a separate frozen
        # file. Here we only build the architecture.
        model.requires_grad_(False)
        model.eval()
        model.interpolate_pos_encoding = True
        return model
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # GC-IDM consumes the ENCODER only. projector/pred_proj (SIGReg) and predictor (AR dynamics)
    # are training-time machinery whose layout drifted on this branch (norm-first MLP;
    # ARPredictor grew type_embedding), so current-train.py checkpoints no longer match the
    # layout built here. Phase-1 never touches any of them -- drop instead of failing.
    sd = {k: v for k, v in sd.items() if not k.startswith(("projector.", "pred_proj.", "predictor."))}
    res = model.load_state_dict(sd, strict=False)
    print(f"[gcidm-train] frozen LeWM <- {Path(weights_path).name}: "
          f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys[:5]}"
    model.requires_grad_(False)
    model.eval()
    model.interpolate_pos_encoding = True
    return model