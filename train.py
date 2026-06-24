import stable_worldmodel.data.formats.hdf5  # HDF5 self-registers on import
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback, EMAActionEncoderCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # Multi-task: zero the padded action dims before the action encoder
    # (per-task masks emitted by MultiTaskDataset; absent in single-task).
    if "action_mask" in batch:
        batch["action"] = batch["action"] * batch["action_mask"][:, None, :]

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]
    proprio_emb = output.get("proprio_emb", None)  # (B, T, D) or None
    task_vec = output.get("task_vec", None)        # (B, D) or None (multi-task)

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]
    ctx_proprio = proprio_emb[:, :ctx_len] if proprio_emb is not None else None

    tgt_emb = emb[:, n_preds:] # label
    pred = self.model.predict(ctx_emb, ctx_act, ctx_proprio, task_vec=task_vec) # pred
    pred_emb, proprio_pred = (pred if ctx_proprio is not None else (pred, None))

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # proprio-as-token: predict next proprio in embedding space (DINO-WM z_proprio loss)
    if ctx_proprio is not None:
        tgt_proprio = proprio_emb[:, n_preds:]
        output["z_proprio_loss"] = (proprio_pred - tgt_proprio.detach()).pow(2).mean()
        output["loss"] = output["loss"] + cfg.get("proprio_pred_weight", 1.0) * output["z_proprio_loss"]

    # GIP step 1 (config-gated): additional Intention predictor (autoregressive policy).
    # Predict the next action a_t from z_<=t conditioned on shifted past actions a_<t.
    # Joint with the world-model loss above; default (enabled=false) skips this entirely.
    ap = cfg.get("action_pred", None)
    if ap is not None and ap.get("enabled", False):
        head = ap.get("head", "mse")  # mse (default) | gmm | diffusion
        # causal: prepend a zero/start token so position t only sees a_<t
        past_act = torch.cat([torch.zeros_like(act_emb[:, :1]), act_emb[:, :ctx_len - 1]], dim=1)
        # goal-conditioned policy (config-gated): encode the hindsight goal frame -> z_goal,
        # apply goal_dropout (zero it per-sample w.p. p -> the SAME head also learns the
        # goal-AGNOSTIC policy = the two-setting WAM / BESO classifier-free-guidance). Default
        # off (no "goal" in batch) -> goal_emb stays None -> existing behaviour byte-identical.
        goal_emb = None
        if ap.get("goal_conditioned", False) and "goal" in batch:
            z_goal = self.model.encode({"pixels": batch["goal"].unsqueeze(1)})["emb"][:, 0]  # (B, D)
            p_drop = float(ap.get("goal_dropout", 0.0))
            if p_drop > 0.0:
                keep = (torch.rand(z_goal.size(0), device=z_goal.device) >= p_drop).to(z_goal.dtype)
                z_goal = z_goal * keep[:, None]
            goal_emb = z_goal
        # OURS GC head (config-gated): AdaLN-Zero horizon conditioning. The hindsight
        # goal carries a realized horizon (batch["horizon"], obs-steps) from
        # GoalSamplingDataset; normalize min(h,H_max)/H_max with the SAME H_max the
        # GC-IDM arm uses (default 50) so both arms see identical horizon scaling.
        # Default off (no horizon_modulator / no "horizon" in batch) -> None -> identity.
        horizon_norm = None
        if (getattr(self.model, "horizon_modulator", None) is not None
                and ap.get("horizon_conditioned", False) and "horizon" in batch):
            H_max = float(ap.get("horizon_H_max", 50))
            horizon_norm = torch.clamp(batch["horizon"].to(emb.device).float(), max=H_max) / H_max
        # mse decodes a point estimate; gmm/diffusion skip the decode and get
        # their loss from action_decoder.loss(intention, target) instead.
        intention, pred_act = self.model.predict_intention(
            ctx_emb, past_act, detach_decoder=ap.get("detach_decoder", False),
            proprio_emb=ctx_proprio, task_vec=task_vec, goal_emb=goal_emb,
            decode=(head == "mse"), horizon=horizon_norm)
        # intention target = action embedding of the true action (JEPA-style).
        # ema_target (opt-in, BYOL/I-JEPA): target = EMA(momentum) encoder applied
        # to the SAME raw actions, inherently detached (EMA has no grad). The
        # online act_emb still feeds the prediction + conditioning (past_act).
        if ap.get("ema_target", False) and getattr(self.model, "action_encoder_ema", None) is not None:
            tgt_act_emb = self.model.action_encoder_ema(batch["action"])[:, :ctx_len].detach()
        else:
            tgt_act_emb = ctx_act.detach() if ap.get("detach_target", True) else ctx_act
        output["intent_loss"] = (intention - tgt_act_emb).pow(2).mean()
        # decoded raw-action loss. head=mse: deterministic MSE (byte-identical to
        # the original). head=gmm/diffusion: multimodal mixture-NLL / DDPM loss
        # from the multimodal decoder, conditioned on the intention embedding.
        # multi-task masks average over each task's VALID action dims only.
        tgt_act = batch["action"][:, :ctx_len]
        if head == "mse":
            if "action_mask" in batch:
                m = batch["action_mask"][:, None, :]  # (B,1,A)
                diff2 = (pred_act - tgt_act).pow(2) * m
                output["act_loss"] = diff2.sum() / (m.expand_as(diff2).sum() + 1e-8)
            else:
                output["act_loss"] = (pred_act - tgt_act).pow(2).mean()
        else:
            dec_in = intention.detach() if ap.get("detach_decoder", False) else intention
            flat_emb = dec_in.reshape(-1, dec_in.shape[-1])
            flat_tgt = tgt_act.reshape(-1, tgt_act.shape[-1])
            w = None
            if "action_mask" in batch:
                w = batch["action_mask"][:, None, :].expand(-1, ctx_len, -1).reshape(-1, tgt_act.shape[-1])
            output["act_loss"] = self.model.action_decoder.loss(flat_emb, flat_tgt, w)
        output["loss"] = (
            output["loss"]
            + ap.get("w_intent", 1.0) * output["intent_loss"]
            + ap.get("w_act", 1.0) * output["act_loss"]
        )

        # SMWM (2606.20104, Balestriero) inverse-dynamics ANTI-COLLAPSE term (config-gated;
        # default action_pred.w_inv=0 -> block skipped -> byte-identical). A DENSE inverse
        # regularizer h([z_tau ; z_{tau+1}]) ~= a_tau over EVERY consecutive latent pair in
        # the window forces the action information INTO the latent, which SMWM shows prevents
        # collapse and can REPLACE SIGReg (loss.sigreg.weight=0). The deployed policy
        # (predict_intention / intention_rollout) is UNCHANGED; this is a train-time term only.
        #   inv_mode=dense (default): mean over ALL consecutive pairs in the context window.
        #   inv_mode=last:            SMWM-style single transition (the last context pair).
        #   inv_target=encoded (default): z_{tau+1} = the ENCODED next latent.
        #   inv_target=predicted (A8/cycle): z_{tau+1} = the FDM rollout zhat_{t+1}=predict(z_t,a_t)
        #                                    (read the action off the FDM's own prediction).
        # Multi-task: the inverse target is masked like act_loss (padded dims zeroed).
        _w_inv = float(ap.get("w_inv", 0.0))
        if _w_inv > 0.0 and getattr(self.model, "inverse_model", None) is not None:
            inv_mode = ap.get("inv_mode", "dense")
            inv_target = ap.get("inv_target", "encoded")
            # encoded next-latent pairs, bounded to the CONTEXT-SUPPORTED window: transition
            # tau -> tau+1 is driven by a_tau, and only the first ctx_len actions are supervised
            # / conditioned on by the forward + action objectives. So pair z_tau (tau=0..ctx_len-1)
            # with z_{tau+1} (tau=1..ctx_len) and the action a_tau over the SAME ctx_len window.
            # (emb has T = ctx_len + num_preds >= ctx_len+1 frames, so z_{ctx_len} exists.) This
            # keeps the inverse term aligned for any num_preds; with num_preds=1, emb[:, :ctx_len]
            # / emb[:, 1:ctx_len+1] is exactly the whole window. Avoids training the inverse head
            # on future transitions outside the context (Codex review, 2026-06-23).
            n_pairs = min(ctx_len, emb.size(1) - 1)
            z_t_enc = emb[:, :n_pairs]                  # (B, ctx_len, D)  z_tau
            z_tp1_enc = emb[:, 1:n_pairs + 1]           # (B, ctx_len, D)  z_{tau+1}
            a_tau = batch["action"][:, :n_pairs]        # (B, ctx_len, A)  action at step tau
            if inv_target == "predicted":
                # z_{tau+1} from the FDM's own rollout. pred_emb = predict(ctx_emb, ctx_act)
                # is the FDM step the FDM actually took; pair zhat_{t+1} with the encoded z_t
                # and the action a_t the FDM was conditioned on (read the action off the FDM's
                # own prediction). Aligned over the predicted window (length = pred_emb.size(1)).
                z_t = ctx_emb[:, : pred_emb.size(1)]    # z_t over the predicted window
                z_tp1 = pred_emb                        # zhat_{t+1} (FDM rollout)
                a_inv = batch["action"][:, : z_t.size(1)]
            else:
                z_t, z_tp1, a_inv = z_t_enc, z_tp1_enc, a_tau
            if inv_mode == "last":
                z_t, z_tp1, a_inv = z_t[:, -1:], z_tp1[:, -1:], a_inv[:, -1:]
            a_hat = self.model.predict_inverse(z_t, z_tp1)    # (B, L, A)
            if "action_mask" in batch:
                m = batch["action_mask"][:, None, :]
                diff2 = (a_hat - a_inv).pow(2) * m
                output["inv_loss"] = diff2.sum() / (m.expand_as(diff2).sum() + 1e-8)
            else:
                output["inv_loss"] = (a_hat - a_inv).pow(2).mean()
            output["loss"] = output["loss"] + _w_inv * output["inv_loss"]

        # FDM<->IDM consistency loss (config-gated; default w_cyc=0 -> block skipped,
        # byte-identical). Roll the predictor (FDM) one step with the IDM's predicted
        # action a_hat and require it to reach the true next latent tgt_emb. Couples the
        # forward (predictor) and inverse (intention) heads end-to-end. SCAR/VERA: target
        # stop-gradded (the JEPA next-latent), keep w_cyc small, warm-start the predictor.
        # Defined for head=mse (a_hat is the decoded raw action). The FDM stays anchored by
        # pred_loss (GT-action), so the IDM is pulled toward FDM-consistent actions.
        _w_cyc = float(ap.get("w_cyc", 0.0))
        if _w_cyc > 0.0 and head == "mse" and pred_act is not None:
            a_hat = pred_act
            if "action_mask" in batch:
                a_hat = a_hat * batch["action_mask"][:, None, :]
            a_hat_emb = self.model.action_encoder(a_hat)
            _cyc = self.model.predict(ctx_emb, a_hat_emb, ctx_proprio, task_vec=task_vec)
            cyc_pred = _cyc[0] if ctx_proprio is not None else _cyc
            output["cyc_loss"] = (cyc_pred - tgt_emb.detach()).pow(2).mean()
            output["loss"] = output["loss"] + _w_cyc * output["cyc_loss"]
            _w_anorm = float(ap.get("w_anorm", 0.0))
            if _w_anorm > 0.0:
                output["anorm_loss"] = a_hat.pow(2).mean()
                output["loss"] = output["loss"] + _w_anorm * output["anorm_loss"]
        if ap.get("sigreg_act", False) and getattr(self, "sigreg_act", None) is not None:
            output["sigreg_act_loss"] = self.sigreg_act(act_emb.transpose(0, 1))
            output["loss"] = output["loss"] + lambd * output["sigreg_act_loss"]
        # collapse monitor: mean per-dim std of action embeddings (-> 0 means collapse)
        output["act_emb_std"] = act_emb.detach().reshape(-1, act_emb.shape[-1]).std(dim=0).mean()

    # Proprio-as-alignment (TC-WM InfoNCEAlignmentObjective, config-gated): instead of feeding
    # proprio as a token, align a leading subspace of the latent to the raw proprio via InfoNCE.
    # Train-time only -- alignment_projection is never used at inference, so eval needs no proprio
    # input (a practical edge over proprio-as-token). enabled=false -> no effect.
    pal = cfg.get("proprio_align", None)
    if pal is not None and pal.get("enabled", False) and "proprio" in batch:
        proj = self.model.alignment_projection
        ad = pal.get("align_dim", 64)
        temp = pal.get("temperature", 0.1)
        z_align = emb[:, :, :ad]                                  # (B,T,ad) leading subspace (grad to encoder)
        proprio_tgt = batch["proprio"]                           # (B,T,proprio_dim) normalized
        b, t, _ = z_align.shape
        z_proj = proj(z_align.reshape(b * t, ad))                # (B*T, proprio_dim)
        za = F.normalize(z_proj, dim=1)
        zt = F.normalize(proprio_tgt.reshape(b * t, -1), dim=1)
        logits = (za @ zt.T) / temp                              # (B*T, B*T) in-batch negatives
        labels = torch.arange(logits.shape[0], device=logits.device)
        output["align_loss"] = F.cross_entropy(logits, labels)
        w_reg = pal.get("reg", 1.0e-4) * (proj.weight ** 2).sum()
        output["loss"] = output["loss"] + pal.get("weight", 1.0) * (output["align_loss"] + w_reg)

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k or k.endswith("_std")}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    mt_tasks = dataset_cfg.pop("tasks", None)
    mt_only = dataset_cfg.pop("only_tasks", None)  # finetune: sample-only subset, full table/dims
    mt_alpha = dataset_cfg.pop("sampling_alpha", 0.0)  # task-sampling temperature (MMBench2)
    mt_task_names = None

    if mt_tasks is not None:
        # Multi-task (config-gated): one swm dataset per task, each with its OWN column
        # normalizers; two-step task-balanced sampling + pad/mask in MultiTaskDataset.
        from multitask_dataset import MultiTaskDataset
        img_t = get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)
        subs_tr, subs_va, names, adims, pdims, pkeys = [], [], [], [], [], []
        for tcfg in mt_tasks:
            tname = tcfg.pop("task")
            h5 = tcfg.pop("name")
            pkey = tcfg.pop("proprio_key", "proprio")
            sub_cfg = {**dataset_cfg, **tcfg}
            ds = swm.data.load_dataset(h5, transform=None, cache_dir=cache_dir, **sub_cfg)
            norm_cols = [c for c in sub_cfg["keys_to_load"] if not c.startswith("pixels")]
            if pkey not in norm_cols:  # merged proprio (cube) is not in keys_to_load
                norm_cols.append(pkey)
            ts = [img_t] + [get_column_normalizer(ds, c, c) for c in norm_cols]
            ds.transform = spt.data.transforms.Compose(*ts)
            tr, va = spt.data.random_split(
                ds, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
            )
            subs_tr.append(tr); subs_va.append(va); names.append(tname); pkeys.append(pkey)
            adims.append(ds.get_dim("action")); pdims.append(ds.get_dim(pkey))
            print(f"[MT] {tname}: {h5} adim={adims[-1]} proprio({pkey})={pdims[-1]} n={len(ds)}")
        fs = dataset_cfg["frameskip"]
        train_set = MultiTaskDataset(subs_tr, names, adims, pdims, pkeys, fs,
                                     active_tasks=mt_only, sampling_alpha=mt_alpha)
        val_set = MultiTaskDataset(subs_va, names, adims, pdims, pkeys, fs,
                                   active_tasks=mt_only, sampling_alpha=mt_alpha)
        if mt_only:
            print(f"[MT] FINETUNE sampling restricted to {list(mt_only)} (table/dims stay 7-task)")
        probs = ", ".join(f"{n}={p:.2f}" for n, p in zip(
            [names[i] for i in train_set.active], train_set.task_p))
        print(f"[MT] task sampling alpha={mt_alpha}: {probs}")
        with open_dict(cfg):
            cfg.model.action_encoder.input_dim = fs * max(adims)
            cfg.proprio_dim = max(pdims)
            cfg.model.mt_task_names = names  # task order -> saved config.json -> eval
        dataset = None
        mt_task_names = names
        print(f"[MT] action block={fs}x{max(adims)}  proprio padded to {max(pdims)}")
    else:
        dataset = swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )
        transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

        with open_dict(cfg):
            for col in cfg.data.dataset.keys_to_load:
                if col.startswith("pixels"):
                    continue
                normalizer = get_column_normalizer(dataset, col, col)
                transforms.append(normalizer)

            cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

        transform = spt.data.transforms.Compose(*transforms)
        dataset.transform = transform

        # goal-conditioned policy (config-gated): wrap so each item carries a HINDSIGHT goal
        # (future frame t+h, preprocessed like the window) + horizon. Default off -> base untouched.
        _ap = cfg.get("action_pred", None)
        if _ap is not None and _ap.get("goal_conditioned", False):
            from goal_dataset import GoalSamplingDataset
            dataset = GoalSamplingDataset(dataset, hindsight_max_k=int(_ap.get("hindsight_max_k", 50)))

        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    # OURS GC head (config-gated): set horizon_conditioned in cfg.model BEFORE
    # instantiation so JEPA builds the HorizonModulator (AdaLN-Zero, reused from
    # gcidm) and the flag travels into config.json -> eval rebuilds the same module.
    # Default OFF (action_pred.horizon_conditioned absent/false) -> byte-identical.
    _apc = cfg.get("action_pred", None)
    if _apc is not None and _apc.get("enabled", False) and _apc.get("horizon_conditioned", False):
        with open_dict(cfg):
            cfg.model.horizon_conditioned = True

    # SMWM inverse head (config-gated): when action_pred.w_inv>0, set inverse_conditioned +
    # inverse_action_dim in cfg.model BEFORE instantiation so JEPA builds self.inverse_model
    # (h([z;z'])->a_hat) and the flag rides in config.json -> load_gip_model rebuilds the SAME
    # head before load_state_dict (the inverse_model.* weights are in the ckpt but the head is
    # NOT used at eval; rebuilding it keeps the strict unexpected-keys audit meaningful).
    # Default OFF (w_inv absent/0) -> head never built -> byte-identical to existing arms.
    if _apc is not None and _apc.get("enabled", False) and float(_apc.get("w_inv", 0.0)) > 0.0:
        with open_dict(cfg):
            cfg.model.inverse_conditioned = True
            cfg.model.inverse_action_dim = int(cfg.model.action_encoder.input_dim)

    world_model = hydra.utils.instantiate(cfg.model)

    # proprio-as-token (config-gated): attach proprio encoder + predicted-token projection.
    if cfg.model.get("use_proprio", False):
        from module import Embedder, MLP as _PropMLP
        pdim = cfg.get("proprio_dim", None) or dataset.get_dim("proprio")
        world_model.proprio_encoder = Embedder(input_dim=pdim, emb_dim=cfg.embed_dim)
        world_model.proprio_pred_proj = _PropMLP(cfg.embed_dim, 2048, cfg.embed_dim)
        world_model.use_proprio = True
        print(f"[PROPRIO] proprio-as-token ON  proprio_dim={pdim} -> emb_dim={cfg.embed_dim}")

    # Proprio-as-alignment (config-gated): learnable projection latent_subspace -> proprio,
    # optimized as part of `model`. Never used at inference (eval ignores it as an unexpected key).
    pal = cfg.get("proprio_align", None)
    if pal is not None and pal.get("enabled", False):
        pdim = cfg.get("proprio_dim", None) or dataset.get_dim("proprio")
        ad = pal.get("align_dim", 64)
        world_model.alignment_projection = torch.nn.Linear(ad, pdim)
        print(f"[ALIGN] proprio-as-alignment ON  align_dim={ad} -> proprio_dim={pdim}  "
              f"temp={pal.get('temperature', 0.1)} w={pal.get('weight', 1.0)}")

    # Multi-task (config-gated): frozen CLIP task-embedding table + learned projection
    # (Newt-style conditioning). Saved in the state_dict so eval can rebuild it.
    if mt_task_names is not None:
        import json as _json
        meta = _json.load(open("all_tasks.json"))
        table = torch.tensor(
            [meta[n]["text_embedding"] for n in mt_task_names], dtype=torch.float32
        )
        world_model.task_table = torch.nn.Parameter(table, requires_grad=False)
        world_model.task_proj = torch.nn.Linear(table.shape[1], cfg.embed_dim)
        print(f"[MT] task conditioning ON  {len(mt_task_names)} tasks {mt_task_names}  "
              f"CLIP {table.shape[1]} -> {cfg.embed_dim}")

    # GIP step 1 (config-gated): attach the optional Intention predictor + action decoder.
    # action_predictor reuses the world-model predictor's spec (a fresh ARPredictor instance);
    # action_decoder maps the 192-d intention -> raw (frameskip-stacked) action.
    ap = cfg.get("action_pred", None)
    gip_on = ap is not None and ap.get("enabled", False)
    if gip_on:
        from module import MLP, GMMHead, DiffusionHead
        adim = cfg.model.action_encoder.input_dim  # = frameskip * action_dim (max over tasks in MT)
        world_model.action_predictor = hydra.utils.instantiate(cfg.model.predictor)
        # action-head factory (config-gated). mse = the original deterministic MLP
        # (byte-identical, old ckpts load); gmm/diffusion = multimodal decoders.
        head = ap.get("head", "mse")
        if head == "gmm":
            world_model.action_decoder = GMMHead(cfg.embed_dim, 2048, adim, n_modes=ap.get("n_modes", 5))
            head_spec = {"type": "gmm", "n_modes": int(ap.get("n_modes", 5))}
        elif head == "diffusion":
            world_model.action_decoder = DiffusionHead(cfg.embed_dim, 2048, adim, n_steps=ap.get("diff_steps", 50))
            head_spec = {"type": "diffusion", "n_steps": int(ap.get("diff_steps", 50))}
        else:
            world_model.action_decoder = MLP(cfg.embed_dim, 2048, adim)
            head_spec = {"type": "mse"}
        # persist the head spec into cfg.model so config.json carries it; eval
        # (load_gip_model) rebuilds the matching decoder before load_state_dict.
        with open_dict(cfg):
            cfg.model.action_head = head_spec
        world_model.action_head = head_spec
        print(f"[GIP] Intention predictor ON  Adim={adim}  head={head}  "
              f"w_act={ap.get('w_act',1.0)} w_intent={ap.get('w_intent',1.0)} "
              f"detach_target={ap.get('detach_target',True)} sigreg_act={ap.get('sigreg_act',False)}")
        if ap.get("horizon_conditioned", False):
            print(f"[GIP] OURS horizon conditioning ON  AdaLN-Zero H_max={ap.get('horizon_H_max',50)} "
                  f"(reuses gcidm sinusoidal(64)+AdaLN-Zero; modulator built in JEPA.__init__)")
        # EMA target encoder (opt-in, BYOL/I-JEPA): a frozen deep copy of the online
        # action_encoder, momentum-updated AFTER each optimizer step by
        # EMAActionEncoderCallback. Only the intent_loss TARGET reads it; the online
        # path (prediction + past_act conditioning) is unchanged. Default OFF.
        if ap.get("ema_target", False):
            import copy as _copy
            world_model.action_encoder_ema = _copy.deepcopy(world_model.action_encoder)
            world_model.action_encoder_ema.requires_grad_(False)
            world_model.action_encoder_ema.eval()
            print(f"[GIP] EMA target encoder ON  ema_tau={ap.get('ema_tau', 0.99)} "
                  f"(action_encoder_ema = frozen momentum copy of action_encoder)")

    # optional: initialize encoder/predictor/projector/action_encoder from a pretrained LeWM ckpt
    init_from = cfg.get("init_from", None)
    if init_from:
        sd = torch.load(init_from, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        res = world_model.load_state_dict(sd, strict=False)
        print(f"[GIP] init_from={init_from}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")

    # optional: freeze the WM trunk (everything the WM-only pretrain trained), so only
    # the intention head + decoder learn — linear/head-probe arms for the pretrain-benefit test
    if cfg.get("freeze_wm", False):
        n_frozen = 0
        for mod in (world_model.encoder, world_model.projector, world_model.predictor,
                    world_model.pred_proj, world_model.action_encoder):
            for p in mod.parameters():
                p.requires_grad = False
                n_frozen += p.numel()
        print(f"[GIP] freeze_wm=true: froze {n_frozen/1e6:.1f}M params "
              f"(encoder/projector/predictor/pred_proj/action_encoder)")

    # optional: dump teacher-forced decoded actions vs ground-truth (visualize act_loss).
    # Reuses the exact val loader + normalizers + frameskip; early-exits before training.
    if cfg.get("dump_decode", False):
        import numpy as _np
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        wm = world_model.to(dev).eval()
        cl = cfg.history_size
        dec, tru, msk = [], [], []
        nb = cfg.get("dump_batches", 6)
        with torch.no_grad():
            for bi, batch in enumerate(val):
                if bi >= nb:
                    break
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                batch["action"] = torch.nan_to_num(batch["action"], 0.0)
                if "action_mask" in batch:
                    batch["action"] = batch["action"] * batch["action_mask"][:, None, :]
                out = wm.encode(batch)
                emb, act_emb = out["emb"], out["act_emb"]
                pemb = out.get("proprio_emb", None)
                tvec = out.get("task_vec", None)
                ctx_emb = emb[:, :cl]
                ctx_prop = pemb[:, :cl] if pemb is not None else None
                past_act = torch.cat([torch.zeros_like(act_emb[:, :1]), act_emb[:, :cl - 1]], dim=1)
                _, pred_act = wm.predict_intention(ctx_emb, past_act, proprio_emb=ctx_prop, task_vec=tvec)
                dec.append(pred_act.cpu().numpy())
                tru.append(batch["action"][:, :cl].cpu().numpy())
                if "action_mask" in batch:
                    msk.append(batch["action_mask"].cpu().numpy())
        dec = _np.concatenate(dec); tru = _np.concatenate(tru)
        outp = cfg.get("dump_path", "/tmp/decode_dump.npz")
        _np.savez(outp, decoded=dec, truth=tru,
                  masks=(_np.concatenate(msk) if msk else _np.array([])))
        mse = float(((dec - tru) ** 2).mean())
        var = float(tru.var())
        print(f"[DUMP] wrote {outp} decoded={dec.shape} mse={mse:.4f} "
              f"var={var:.4f} R2={1 - mse / (var + 1e-9):.3f}")
        return

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )
    if gip_on and ap.get("sigreg_act", False):
        world_model.sigreg_act = SIGReg(**cfg.loss.sigreg.kwargs)  # anti-collapse on action embeddings

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model,
        epoch_interval=cfg.get("ckpt_every", 1), full_cfg=cfg,
    )

    callbacks = [object_dump_callback]
    if gip_on and ap.get("ema_target", False):
        callbacks.append(EMAActionEncoderCallback(tau=ap.get("ema_tau", 0.99)))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
