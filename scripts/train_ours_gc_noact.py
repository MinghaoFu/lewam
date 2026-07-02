# External baseline comparison: OURS history-conditioned forward GC head trained on the
# same frozen-latent cache the gcidm arm (arXiv 2605.08732) uses, for a fair OURS-vs-GC-IDM comparison.
"""Fast cached trainer for OUR history-conditioned forward GC policy (cube).

Mirrors train_gcidm.py's two-phase design (precompute frozen latents once, then
train the head on cached tensors -> seconds/epoch, true convergence), but for OUR
head instead of the Markovian GC-IDM:

  Our head = the GIP intention head (jepa.JEPA.predict_intention):
    action_predictor (a fresh ARPredictor, trainable) over the latent HISTORY
    z_{t-HS+1..t} + past-action EMBEDDINGS a_{<t} (the wedge vs gcidm), PLUS the
    goal latent z_goal (additive, goal-conditioned), PLUS the SAME AdaLN-Zero
    horizon (jepa.HorizonModulator, reused from gcidm) -> action_decoder (MLP).

It REUSES the SAME frozen-LeWM latent cache the gcidm arm built
(checkpoints/cube_gcidm/latents_cache.pt: per-ep lat[ep] (n+1,192) = the projected
emb encode(pixels)['emb'][:,0], act[ep] (n,25) z-scored), so OURS and GC-IDM train
on byte-identical frozen features -> maximally fair comparison.

The action_encoder is the FROZEN base action_encoder (so a_<t embeddings match the
base); action_predictor/decoder/horizon_modulator are trained from scratch.

Saves a JEPA checkpoint + config.json that gip.load_gip_model loads verbatim
(config.json carries horizon_conditioned=true, action_head=mse), so eval_gip.py
--config-name cube policy=cube_ours_gc +gip_eval.mode=policy works unchanged.

Run (L40S):
  STABLEWM_HOME=/mnt/minghao_data/.stable-wm MUJOCO_GL=egl HF_HUB_OFFLINE=1 \
  MPLCONFIGDIR=/tmp/mpl_x TMPDIR=/tmp CUDA_VISIBLE_DEVICES=<g> \
    python train_ours_gc.py --epochs 200 --H_max 50 --run_name cube_ours_gc
"""
import argparse, json, os, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import stable_worldmodel as swm
from hydra.utils import instantiate as hydra_instantiate
import lewam.models.jepa as jepa
from lewam.models.module import MLP


def build_jepa(weights, embed_dim=192, history_size=3, img_size=224, action_block_dim=25, use_action_history=True):
    """Instantiate the full JEPA (vit-tiny-192) with horizon_conditioned=True, load the
    frozen base weights, attach a fresh action_predictor/decoder. Returns the JEPA; the
    trunk (encoder/projector/predictor/pred_proj/action_encoder) is loaded from the base,
    action_predictor/decoder/horizon_modulator are fresh (trained here)."""
    cfg = OmegaConf.create({
        "_target_": "jepa.JEPA", "use_action_history": use_action_history, "use_proprio": False,
        "horizon_conditioned": True,
        "encoder": {"_target_": "stable_pretraining.backbone.utils.vit_hf",
                    "size": "tiny", "patch_size": 14, "image_size": img_size,
                    "pretrained": False, "use_mask_token": False},
        "predictor": {"_target_": "module.ARPredictor", "num_frames": history_size,
                      "input_dim": embed_dim, "hidden_dim": embed_dim, "output_dim": embed_dim,
                      "depth": 6, "heads": 16, "mlp_dim": 2048, "dim_head": 64,
                      "dropout": 0.1, "emb_dropout": 0.0},
        "action_encoder": {"_target_": "module.Embedder", "input_dim": action_block_dim, "emb_dim": embed_dim},
        "projector": {"_target_": "module.MLP", "input_dim": embed_dim, "output_dim": embed_dim,
                      "hidden_dim": 2048, "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}},
        "pred_proj": {"_target_": "module.MLP", "input_dim": embed_dim, "output_dim": embed_dim,
                      "hidden_dim": 2048, "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}},
        "action_head": {"type": "mse"},
    })
    model = hydra_instantiate(cfg)
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    res = model.load_state_dict(sd, strict=False)
    print(f"[ours-train] base <- {Path(weights).name}: missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)}")
    assert not res.unexpected_keys, res.unexpected_keys[:5]
    # attach fresh trainable action modules (match train.py's GIP attach)
    model.action_predictor = hydra_instantiate(cfg.predictor)
    model.action_decoder = MLP(embed_dim, 2048, action_block_dim)
    model.action_head = {"type": "mse"}
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--H_max", type=int, default=50)
    ap.add_argument("--history_size", type=int, default=3)
    ap.add_argument("--w_act", type=float, default=1.0)
    ap.add_argument("--w_intent", type=float, default=1.0)
    ap.add_argument("--detach_target", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--run_name", type=str, default="cube_ours_gc")
    ap.add_argument("--cache_run", type=str, default="cube_gcidm")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--use_action_history", type=int, default=1, help="0 = zero the past-action stream (state-only HS-frame + goal + horizon)")
    ap.add_argument("--train_split", type=float, default=0.9)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HS = args.history_size

    cache = Path(swm.data.utils.get_cache_dir())
    weights = args.weights or str(cache / "decoders" / "cube_ours_lewm_weights.pt")
    ck = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
    run_dir = ck / args.run_name; run_dir.mkdir(parents=True, exist_ok=True)

    # reuse the gcidm frozen-latent cache (same frozen features)
    cache_path = ck / args.cache_run / "latents_cache.pt"
    assert cache_path.exists(), f"missing latent cache {cache_path} (run train_gcidm precompute first)"
    blob = torch.load(cache_path, map_location="cpu")
    lat_list, act_list = blob["lat"], blob["act"]
    act_mean, act_std = blob["act_mean"], blob["act_std"]
    print(f"[ours-train] loaded cache {cache_path.name}: n_eps={len(lat_list)} "
          f"emb_dim={lat_list[0].shape[-1]} adim={act_list[0].shape[-1]}", flush=True)
    action_block_dim = int(act_list[0].shape[-1])  # 25

    model, cfg = build_jepa(weights, embed_dim=192, history_size=HS,
                            action_block_dim=action_block_dim,
                            use_action_history=bool(args.use_action_history))
    model = model.to(device)
    # freeze the trunk (encoder/projector/predictor/pred_proj) + action_encoder; train the head.
    for mod in (model.encoder, model.projector, model.predictor, model.pred_proj, model.action_encoder):
        for p in mod.parameters():
            p.requires_grad = False
    model.encoder.eval(); model.projector.eval(); model.predictor.eval(); model.pred_proj.eval()
    model.action_encoder.eval()
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in train_params)
    print(f"[ours-train] trainable params={n_tr/1e6:.2f}M "
          f"(action_predictor + action_decoder + horizon_modulator)", flush=True)

    # build flat sample index: (ep, t) with t>=HS-1 (full history) and t<n_obs
    # store latents/actions per episode on device; sample windows vectorized per batch.
    lat_dev = [z.float().to(device) for z in lat_list]
    act_dev = [a.float().to(device) for a in act_list]
    samples = []  # (ep, t, maxh)
    for ep in range(len(act_dev)):
        n_obs = act_dev[ep].shape[0]
        n_lat = lat_dev[ep].shape[0]
        last = n_lat - 1
        n_valid = min(n_obs, n_lat - 1)
        for t in range(HS - 1, n_valid):     # need z[t-HS+1..t] and a[t-HS+1..t-1]
            maxh = last - t
            if maxh >= 1:
                samples.append((ep, t, maxh))
    samples = np.array(samples, dtype=np.int64)
    g = np.random.default_rng(args.seed); g.shuffle(samples)
    n_val = int(round((1 - args.train_split) * len(samples)))
    val_s = samples[:n_val]; train_s = samples[n_val:]
    print(f"[ours-train] windows n={len(samples)} train={len(train_s)} val={len(val_s)} "
          f"HS={HS} H_max={args.H_max}", flush=True)

    # save config.json (cfg.model) so gip.load_gip_model rebuilds the SAME JEPA + modulator
    cfg_model = OmegaConf.to_container(cfg, resolve=True)
    (run_dir / "config.json").write_text(json.dumps(cfg_model, indent=2))

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def make_batch(batch_samples):
        """Build (z_hist (B,HS,D), past_act (B,HS,Adim), z_goal (B,D), h_norm (B,), a_t (B,Adim))."""
        B = len(batch_samples)
        D = lat_dev[0].shape[-1]
        z_hist = torch.zeros(B, HS, D, device=device)
        a_hist = torch.zeros(B, HS, action_block_dim, device=device)  # raw actions a_{t-HS+1..t}
        z_goal = torch.zeros(B, D, device=device)
        a_t = torch.zeros(B, action_block_dim, device=device)
        h = np.empty(B, dtype=np.float32)
        for i, (ep, t, maxh) in enumerate(batch_samples):
            z_hist[i] = lat_dev[ep][t - HS + 1: t + 1]
            a_hist[i] = act_dev[ep][t - HS + 1: t + 1]          # a_{t-HS+1..t}
            hh = int(min(np.random.randint(1, args.H_max + 1), maxh)); hh = max(hh, 1)
            z_goal[i] = lat_dev[ep][t + hh]
            a_t[i] = act_dev[ep][t]
            h[i] = hh
        h_norm = torch.tensor(np.minimum(h, args.H_max) / args.H_max, device=device)
        return z_hist, a_hist, z_goal, h_norm, a_t

    def run_epoch(pool, train_mode):
        model.action_predictor.train(train_mode)
        model.action_decoder.train(train_mode)
        model.horizon_modulator.train(train_mode)
        torch.set_grad_enabled(train_mode)
        order = np.random.permutation(len(pool)) if train_mode else np.arange(len(pool))
        bs = args.batch_size
        tot_act, tot_int, cnt = 0.0, 0.0, 0
        for i in range(0, len(order) - (bs if train_mode else 0) + (0 if train_mode else 1), bs):
            bidx = order[i:i + bs]
            if len(bidx) == 0: continue
            bs_samples = pool[bidx]
            z_hist, a_hist, z_goal, h_norm, a_t = make_batch(bs_samples)
            # past_act_emb: causal a_<t -> shift so position t sees a_{<t} (a_{t-HS+1..t-1}),
            # with a leading zero (matches train.py: cat[zeros[:, :1], act_emb[:, :HS-1]]).
            act_emb = model.action_encoder(a_hist)               # (B,HS,D) embeddings of a_{t-HS+1..t}
            past_act = torch.cat([torch.zeros_like(act_emb[:, :1]), act_emb[:, :HS - 1]], dim=1)
            intention, pred_act = model.predict_intention(
                z_hist, past_act, detach_decoder=False, goal_emb=z_goal,
                decode=True, horizon=h_norm)
            # targets: intent target = act_emb of the TRUE current action a_t (position t = last),
            # decoded target = the raw z-scored a_t. Both at the last history position.
            tgt_emb = act_emb[:, HS - 1:HS]                      # (B,1,D) emb of a_t
            if args.detach_target: tgt_emb = tgt_emb.detach()
            int_loss = (intention[:, HS - 1:HS] - tgt_emb).pow(2).mean()
            act_loss = (pred_act[:, HS - 1] - a_t).pow(2).mean()
            loss = args.w_intent * int_loss + args.w_act * act_loss
            if train_mode:
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
            tot_act += act_loss.item() * len(bidx); tot_int += int_loss.item() * len(bidx); cnt += len(bidx)
        torch.set_grad_enabled(True)
        return tot_act / max(cnt, 1), tot_int / max(cnt, 1)

    def save_state(filename):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(model, run_name=args.run_name, config=OmegaConf.create(cfg_model),
                        filename=filename)

    # Early-stopping on val_act: BEST-val-act checkpoint -> weights_epoch_1.pt
    # (load_gip_model picks highest epoch #); LATEST -> weights_epoch_0.pt (diagnostic, not picked).
    # cube val_act plateaus ~epoch 14-15, so T_max should land lr~0 there.
    best_val = float("inf"); best_epoch = -1
    for ep in range(args.epochs):
        t0 = time.time()
        tr_a, tr_i = run_epoch(train_s, True); sched.step()
        va_a, va_i = run_epoch(val_s, False)
        dt = time.time() - t0
        is_best = va_a < best_val
        print(f"[ours-train] epoch {ep+1}/{args.epochs}  train_act={tr_a:.5f} train_int={tr_i:.5f}  "
              f"val_act={va_a:.5f} val_int={va_i:.5f}  lr={sched.get_last_lr()[0]:.2e}  {dt:.1f}s"
              f"{'  <-BEST' if is_best else ''}", flush=True)
        if is_best:
            best_val = va_a; best_epoch = ep + 1
            save_state("weights_epoch_1.pt")    # BEST (load_gip_model loads this)
        save_state("weights_epoch_0.pt")         # LATEST (diagnostic; lower epoch# -> not picked)
    print(f"[ours-train] DONE best_val_act={best_val:.5f} @epoch {best_epoch}  saved -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
