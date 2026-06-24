"""Decoder-probe for LeWM (le-wm-repro copy).

Adapted from exx world-forge `scripts/decode_lewm.py`. This is a COPY into our
le-wm-repro workspace; exx's original is left untouched. Two changes vs the
source:

  1. Decoder-TRAINING hyperparameters are read from `config/decode/cnn.yaml`
     (the `decode.*` group) instead of being hardcoded. The runtime PATH args
     (--lance, --ckpt-dir, --tag) stay on the CLI; they are per-run.
  2. The frozen-LeWM loader matches OUR checkpoint layout: `config.json` whose
     top-level `_target_` is `jepa.JEPA` (instantiated directly, no `cfg.model`
     wrapper) plus the latest `weights_epoch_*.pt` in the checkpoint dir. The
     embed dim is read from that config and asserted against `decode.embed_dim`.

Trains a CNN image decoder on top of the frozen LeWM encoder CLS embedding
(D=192) to (a) reconstruct input frames from the encoded CLS embedding and
(b) decode LeWM-predicted-future latents into predicted frames.

Frozen: the whole JEPA world model (encoder + projector + predictor +
action_encoder + pred_proj). Trained: only the CNN decoder.

`decode.enabled` is the master switch:
  - true  (default): train a fresh decoder, plateau-stop, save best as
                     `decoder_conv_best.pt`, then render.
  - false:           load an existing `decoder_conv_best.pt` (or --decoder),
                     SKIP the training loop entirely, then render.

Encoder normalization: uint8 [0,255] -> /255 -> ImageNet Normalize, at 224.
Decode target: uint8 [0,255] -> /255 -> resize to 128 ([0,1], no ImageNet).
Decoder output: sigmoid -> [0,1] compared (MSE) to the [0,1] 128 target.
"""

import argparse
import io
import json
from pathlib import Path

import hydra
import lance
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

import stable_pretraining as spt  # noqa: F401  (registers backbone)
from stable_pretraining.backbone.decoders import build_image_decoder


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEC_SIZE = 128
ENC_SIZE = 224
HISTORY = 3
BEST_NAME = "decoder_conv_best.pt"
DEFAULT_DECODE_CFG = str(
    Path(__file__).resolve().parent / "config" / "decode" / "cnn.yaml"
)


def log(msg):
    print(f"[decode_lewm] {msg}", flush=True)


def _latest_weights(ckpt_dir):
    """Newest weights_epoch_<N>.pt in the checkpoint dir (by epoch number)."""
    cands = sorted(
        Path(ckpt_dir).glob("weights_epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not cands:
        raise FileNotFoundError(f"no weights_epoch_*.pt in {ckpt_dir}")
    return cands[-1]


def _ckpt_embed_dim(cfg):
    """Read the LeWM latent dim from our config.json (predictor.input_dim,
    falling back to action_encoder.emb_dim). Returns None if not present."""
    for path in (("predictor", "input_dim"), ("action_encoder", "emb_dim")):
        node = cfg
        ok = True
        for k in path:
            if OmegaConf.is_config(node) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok and node is not None:
            return int(node)
    return None


def load_lewm(ckpt_dir, device):
    """Load the frozen LeWM world model from OUR checkpoint layout: a top-level
    `jepa.JEPA` config.json + the latest weights_epoch_*.pt. (UNCHANGED in
    spirit from exx: hydra.utils.instantiate of the config + load_state_dict;
    only the config nesting and weights filename are adapted to our repo.)"""
    ckpt_dir = Path(ckpt_dir)
    cfg = OmegaConf.create(json.load(open(ckpt_dir / "config.json")))
    model = hydra.utils.instantiate(cfg)
    wpath = _latest_weights(ckpt_dir)
    sd = torch.load(wpath, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    log(
        f"loaded {ckpt_dir.name} ({wpath.name}): "
        f"missing={list(missing)} unexpected={list(unexpected)}"
    )
    return model, cfg


def decode_jpegs(blobs):
    return np.stack(
        [np.array(Image.open(io.BytesIO(b)).convert("RGB")) for b in blobs]
    )


def to_unit(imgs_uint8):
    """(N,H,W,3) uint8 -> (N,3,H,W) float in [0,1]."""
    t = torch.from_numpy(imgs_uint8).permute(0, 3, 1, 2).float() / 255.0
    return t


def enc_normalize(unit):
    """[0,1] (N,3,H,W) -> ImageNet-normalized at 224."""
    if unit.shape[-1] != ENC_SIZE:
        unit = F.interpolate(unit, size=ENC_SIZE, mode="bilinear", align_corners=False)
    return (unit - IMAGENET_MEAN.to(unit.device)) / IMAGENET_STD.to(unit.device)


def dec_target(unit, size):
    """[0,1] (N,3,H,W) -> [0,1] resized to `size`."""
    return F.interpolate(unit, size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def encode_cls(model, unit_imgs, device, bs=128):
    """unit_imgs [0,1] (N,3,H,W) -> CLS emb (N,D) detached."""
    out = []
    for i in range(0, unit_imgs.shape[0], bs):
        chunk = unit_imgs[i : i + bs].to(device)
        xn = enc_normalize(chunk)
        info = {"pixels": xn.unsqueeze(1)}  # (B,T=1,C,H,W)
        info = model.encode(info)
        out.append(info["emb"][:, 0].float().cpu())  # (B,D)
    return torch.cat(out, 0)


def load_episode(ds, ep_id):
    cols = ["pixels", "action", "step_idx"]
    tbl = ds.to_table(columns=cols, filter=f"episode_idx = {ep_id}").to_pydict()
    order = np.argsort(tbl["step_idx"])
    imgs = decode_jpegs([tbl["pixels"][i] for i in order])
    acts = np.array([tbl["action"][i] for i in order], dtype=np.float32)
    return imgs, acts


def main():
    # Runtime PATH args stay on the CLI (per-run). Everything else (the decoder
    # build + training hyperparameters) comes from config/decode/cnn.yaml.
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_DECODE_CFG,
                    help="decode yaml (decoder build + training knobs)")
    ap.add_argument("--task", required=True, help="e.g. robomimic_lift (for output naming)")
    ap.add_argument("--lance", required=True, help="absolute path to the .lance dataset dir")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", required=True,
                    help="absolute path to the frozen-LeWM checkpoint dir "
                         "(holds config.json + weights_epoch_*.pt)")
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="output dir for decoder ckpt + viz (default: <ckpt-dir>/decode)")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--decoder", default=None,
                    help="decoder ckpt to load when decode.enabled=false "
                         "(default: <out-dir>/decoder_conv_best.pt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-frames", dest="max_train_frames", type=int, default=20000)
    ap.add_argument("--log-every", dest="log_every", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--n-recon", dest="n_recon", type=int, default=6)
    args = ap.parse_args()

    # ---- decoder-training knobs from yaml (the `decode.*` group) ----
    yc = OmegaConf.load(args.config)
    dc = yc.decode
    enabled = bool(dc.enabled)
    kind = str(dc.kind)
    cfg_embed = int(dc.embed_dim)
    image_shape = [int(x) for x in OmegaConf.to_container(dc.image_shape)]
    max_epochs = int(dc.max_epochs)
    min_epochs = int(dc.min_epochs)
    lr = float(dc.lr)
    plateau_patience = int(dc.plateau_patience)
    eval_every = int(dc.eval_every)
    dec_c, dec_h = int(image_shape[0]), int(image_shape[1])

    log(
        f"config={args.config} decode.enabled={enabled} kind={kind} task={args.task} "
        f"max_epochs={max_epochs} min_epochs={min_epochs} lr={lr} "
        f"eval_every={eval_every} plateau_patience={plateau_patience}"
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt_dir) / "decode"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt_cfg = load_lewm(args.ckpt_dir, device)

    # Prefer embed_dim from the checkpoint config.json; assert it matches yaml.
    ckpt_embed = _ckpt_embed_dim(ckpt_cfg)
    if ckpt_embed is not None:
        assert ckpt_embed == cfg_embed, (
            f"embed_dim mismatch: checkpoint config.json={ckpt_embed} "
            f"!= decode.embed_dim={cfg_embed} (fix config/decode/cnn.yaml)"
        )
        D = ckpt_embed
        log(f"embed_dim D={D} (from checkpoint config.json; matches decode.embed_dim)")
    else:
        D = cfg_embed
        log(f"embed_dim D={D} (checkpoint config.json had none; using decode.embed_dim)")

    ds = lance.dataset(args.lance)
    ep_ids = np.unique(ds.to_table(columns=["episode_idx"]).to_pydict()["episode_idx"])
    ep_ids = np.sort(ep_ids)
    log(f"dataset {args.lance}: {ds.count_rows()} frames, {len(ep_ids)} episodes")

    # held-out: last few episodes for viz; rest for decoder training
    n_hold = min(5, max(1, len(ep_ids) // 10))
    hold_eps = ep_ids[-n_hold:]
    train_eps = ep_ids[:-n_hold]
    log(f"train episodes={len(train_eps)} held-out viz episodes={list(hold_eps)}")

    sfx = ("_" + args.tag) if args.tag else ""
    best_val = float("inf")
    best_epoch = -1
    val_hist = []
    last_epoch_mse = None
    epochs_run = 0

    decoder = build_image_decoder(
        embed_dim=D, image_shape=(dec_c, dec_h, dec_h), kind=kind,
    ).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    log(f"decoder params={n_params:,} kind={kind} image_shape={(dec_c, dec_h, dec_h)}")

    if enabled:
        # ---- gather training frames (uint8) from train episodes ----
        frame_filter = "episode_idx IN ({})".format(
            ",".join(str(int(e)) for e in train_eps)
        )
        tbl = ds.to_table(columns=["pixels"], filter=frame_filter).to_pydict()
        blobs = tbl["pixels"]
        if len(blobs) > args.max_train_frames:
            idx = np.random.choice(len(blobs), args.max_train_frames, replace=False)
            blobs = [blobs[i] for i in idx]
        log(f"training decoder on {len(blobs)} frames")

        imgs = decode_jpegs(blobs)
        unit = to_unit(imgs)  # (N,3,224,224) [0,1]

        # pre-encode all CLS embeddings (encoder frozen) and precompute targets
        log("pre-encoding CLS embeddings for all training frames ...")
        z_all = encode_cls(model, unit, device, bs=256)  # (N,D)
        tgt_all = dec_target(unit, dec_h)  # (N,3,128,128) [0,1] on cpu

        # ---- held-out eval set (frames from held-out episodes) ----
        val_filter = "episode_idx IN ({})".format(
            ",".join(str(int(e)) for e in hold_eps)
        )
        val_blobs = ds.to_table(
            columns=["pixels"], filter=val_filter
        ).to_pydict()["pixels"]
        if len(val_blobs) > 2000:
            vidx = np.random.choice(len(val_blobs), 2000, replace=False)
            val_blobs = [val_blobs[i] for i in vidx]
        val_unit = to_unit(decode_jpegs(val_blobs))
        val_z = encode_cls(model, val_unit, device, bs=256)
        val_tgt = dec_target(val_unit, dec_h)
        log(f"held-out eval set: {val_z.shape[0]} frames")

        @torch.no_grad()
        def eval_heldout(dec, bs=256):
            dec.eval()
            tot, cnt = 0.0, 0
            for i in range(0, val_z.shape[0], bs):
                zz = val_z[i:i + bs].to(device)
                tt = val_tgt[i:i + bs].to(device)
                xx = torch.sigmoid(dec(zz))
                tot += F.mse_loss(xx, tt, reduction="sum").item()
                cnt += tt.numel()
            dec.train()
            return tot / cnt

        opt = torch.optim.Adam(decoder.parameters(), lr=lr)
        BS = 64
        N = z_all.shape[0]
        steps_per_epoch = (N + BS - 1) // BS
        # convergence: train up to max_epochs, plateau-stop on held-out recon.
        # cosine horizon = max_epochs so LR doesn't hit 0 before we may stop.
        plan_epochs = max(min_epochs, max_epochs)
        total_steps = plan_epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        log(f"cosine LR: T_max={total_steps} (plan_epochs={plan_epochs})")
        step = 0
        for ep in range(plan_epochs):
            perm = torch.randperm(N)
            running = 0.0
            nb = 0
            for i in range(0, N, BS):
                sel = perm[i : i + BS]
                z = z_all[sel].to(device)
                tgt = tgt_all[sel].to(device)
                xhat = torch.sigmoid(decoder(z))
                loss = F.mse_loss(xhat, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                running += loss.item()
                nb += 1
                step += 1
                if step % args.log_every == 0:
                    lr_now = opt.param_groups[0]["lr"]
                    log(f"epoch {ep} step {step} mse={loss.item():.5f} lr={lr_now:.2e}")
            last_epoch_mse = running / max(1, nb)
            epochs_run = ep + 1

            if (ep + 1) % eval_every == 0 or ep == plan_epochs - 1:
                val_mse = eval_heldout(decoder)
                val_hist.append((ep, val_mse))
                tag_best = ""
                if val_mse < best_val:
                    best_val = val_mse
                    best_epoch = ep
                    torch.save(decoder.state_dict(), out_dir / BEST_NAME)
                    tag_best = " [BEST]"
                log(f"== epoch {ep} train_mse={last_epoch_mse:.5f} "
                    f"heldout_mse={val_mse:.5f} best={best_val:.5f}@{best_epoch}{tag_best} ==")
                # plateau: no held-out improvement over the last `plateau_patience`
                # evals, only after the `min_epochs` minimum has elapsed.
                recent = [v for (_, v) in val_hist[-(plateau_patience + 1):]]
                if (ep + 1) >= min_epochs and len(recent) > plateau_patience:
                    window_best = min(recent[:-1])
                    if min(recent) >= window_best:
                        log(f"plateau: held-out flat over last {plateau_patience} "
                            f"evals; stopping at epoch {ep}")
                        break
            else:
                log(f"== epoch {ep} train_mse={last_epoch_mse:.5f} ==")

        log(f"best held-out recon MSE={best_val:.6f} at epoch {best_epoch} -> {BEST_NAME}")
        # render from the BEST checkpoint
        if (out_dir / BEST_NAME).exists():
            decoder.load_state_dict(torch.load(out_dir / BEST_NAME, map_location="cpu"))
            log("loaded BEST checkpoint for visualization")
    else:
        # decode.enabled=false: load an existing decoder and SKIP training.
        dec_ckpt = Path(args.decoder) if args.decoder else out_dir / BEST_NAME
        if not dec_ckpt.exists():
            ap.error(
                f"decode.enabled=false but decoder checkpoint not found: {dec_ckpt} "
                f"(pass --decoder or train one first with decode.enabled=true)"
            )
        decoder.load_state_dict(torch.load(dec_ckpt, map_location="cpu"))
        log(f"decode.enabled=false: loaded decoder {dec_ckpt} (D={D}); skipping training")

    decoder.eval()

    # ---- (a) recon_grid.png on held-out frames ----
    rec_imgs, _ = load_episode(ds, int(hold_eps[0]))
    n_recon = int(args.n_recon)
    sel_idx = np.linspace(0, len(rec_imgs) - 1, n_recon).astype(int)
    ru = to_unit(rec_imgs[sel_idx])  # (n_recon,3,224,224)
    with torch.no_grad():
        z = encode_cls(model, ru, device)
        xhat = torch.sigmoid(decoder(z.to(device))).cpu()  # (n_recon,3,128,128)
    tgt = dec_target(ru, dec_h)
    final_recon_mse = F.mse_loss(xhat, tgt).item()
    log(f"held-out recon MSE ({dec_h} target) = {final_recon_mse:.5f}")

    fig, axes = plt.subplots(n_recon, 2, figsize=(4, 2 * n_recon))
    for r in range(n_recon):
        axes[r, 0].imshow(tgt[r].permute(1, 2, 0).numpy())
        axes[r, 0].set_title("input" if r == 0 else "", fontsize=9)
        axes[r, 0].axis("off")
        axes[r, 1].imshow(xhat[r].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[r, 1].set_title("decoded recon" if r == 0 else "", fontsize=9)
        axes[r, 1].axis("off")
    fig.suptitle(f"{args.task}: input vs CLS-decode (recon MSE={final_recon_mse:.4f})")
    fig.tight_layout()
    recon_name = "recon_grid" + sfx + ".png"
    strip_name = "predict_strip" + sfx + ".png"
    summ_name = "summary" + sfx + ".json"
    fig.savefig(out_dir / recon_name, dpi=120)
    plt.close(fig)
    log("saved " + str(out_dir / recon_name))

    # ---- (b) predict_strip.png: rollout GT actions, decode predicted future ----
    ep_imgs, acts = load_episode(ds, int(hold_eps[0]))
    H = HISTORY
    horizon = min(args.horizon, len(ep_imgs) - H - 1)
    T = H + horizon
    eu = to_unit(ep_imgs[:T])  # (T,3,224,224)
    en = enc_normalize(eu).to(device)
    pix = en[:H].unsqueeze(0).unsqueeze(0)  # (1,1,H,C,H,W)
    act_seq = (
        torch.from_numpy(acts[:T]).to(device).unsqueeze(0).unsqueeze(0)
    )  # (1,1,T,A)
    info = {"pixels": pix}
    with torch.no_grad():
        info = model.rollout(info, act_seq, history_size=H)
        pe = info["predicted_emb"][0, 0]  # (H+horizon+1, D)
        fut_z = pe[H:T]  # (horizon, D) predicted latents for future frames
        fut_xhat = torch.sigmoid(decoder(fut_z.to(device))).cpu()  # (horizon,3,128,128)
    gt_future = dec_target(to_unit(ep_imgs[H:T]), dec_h)  # (horizon,3,128,128) [0,1]
    pred_track_mse = F.mse_loss(fut_xhat, gt_future).item()
    log(f"decoded-prediction vs GT-future MSE = {pred_track_mse:.5f}")

    cols = horizon
    fig, axes = plt.subplots(2, cols, figsize=(2 * cols, 4.4))
    if cols == 1:
        axes = axes.reshape(2, 1)
    for c in range(cols):
        axes[0, c].imshow(gt_future[c].permute(1, 2, 0).numpy())
        axes[0, c].set_title(f"GT t+{c+1}", fontsize=9)
        axes[0, c].axis("off")
        axes[1, c].imshow(fut_xhat[c].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, c].set_title(f"pred t+{c+1}", fontsize=9)
        axes[1, c].axis("off")
    fig.suptitle(
        f"{args.task}: GT future (top) vs decoded LeWM prediction (bottom) "
        f"| pred MSE={pred_track_mse:.4f}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / strip_name, dpi=120)
    plt.close(fig)
    log("saved " + str(out_dir / strip_name))

    summary = {
        "task": args.task,
        "tag": args.tag,
        "decode_enabled": bool(enabled),
        "D": int(D),
        "kind": kind,
        "image_shape": [dec_c, dec_h, dec_h],
        "final_train_mean_mse": (float(last_epoch_mse)
                                 if last_epoch_mse is not None else None),
        "best_heldout_recon_mse": (float(best_val)
                                   if best_val != float("inf") else None),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "heldout_recon_mse": float(final_recon_mse),
        "decoded_prediction_vs_gt_mse": float(pred_track_mse),
        "val_history": [[int(e), float(v)] for (e, v) in val_hist],
        "horizon": int(horizon),
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        "artifacts": {
            "decoder_best": str(out_dir / BEST_NAME),
            "recon_grid": str(out_dir / recon_name),
            "predict_strip": str(out_dir / strip_name),
        },
    }
    with open(out_dir / summ_name, "w") as f:
        json.dump(summary, f, indent=2)
    log("SUMMARY: " + json.dumps(summary))


if __name__ == "__main__":
    main()
