"""Feature-dimension interaction analysis on trained MoT snapshots (owner design 2026-08-31).

Central question: when a tiny step optimizes L_P (action loss) and a tiny step optimizes L_D
(state loss), do they move the encoder REPRESENTATION in the same useful directions, different
useful directions, or opposite useful directions? Measured through the encoder Jacobian via
virtual parameter steps, not raw dL/dz.

Per (arm, snapshot), on one fixed probe set of training windows (identical across arms/snaps):

  1. spectrum          eigenbasis u_k / eigenvalues lam_k of Cov(z_t) over the probe set,
                       effective rank, top-k variance shares, displacement Var(u_k^T(z_next-z_t)).
  2. usage (diagnostic) E[(u_k^T dL/dz)^2] at the latent, encoder held fixed, for
                       P (action loss wrt z_t), D_in (state loss wrt z_t input path),
                       D_tar (state loss wrt state_target), S (SIGReg wrt z_t).
  3. ablation (causal)  direction k's variation replaced by its probe mean in ALL latents fed to
                       the trunk (z_history, state_target, SIGReg input); same RNG draws ->
                       dL_P(k), dL_D(k), dL_S(k). Top --abl_top directions individually, the
                       rest in bands. This is the necessity measure A_k.
  4. induced representation update (centerpiece): encoder-parameter gradients g_l for
                       l in {P, D_full, D_in, S} on the same batch (D_tar = D_full - D_in;
                       the target-detached second loss call under identical RNG gives the exact
                       split, trainer probe convention). Virtual step theta' = theta - eps*g_l,
                       eps calibrated so rms(dz)/rms(z) ~= --dz_target; re-encode the SAME fixed
                       probe frames -> dz_l(x); project on u_k -> v_{l,k}(x). Report per-direction
                       uncentered correlation a_k(l1,l2) = E[v1 v2]/sqrt(E[v1^2]E[v2^2]),
                       amplitude fractions, global cosines, and a linearity check (eps/2).
                       SIGReg's update is measured like the others (counterfactual on noreg arms),
                       never assumed analytic.
  5. conflict index    conflict_k = sqrt(A_P,k * A_D,k) * max(0, -a_k): opposite pushes only
                       count where BOTH losses demonstrably need the direction.

All loss evaluations per (arm,snap) run under forked, re-seeded RNG so tau/noise/SIGReg
projections are identical draws across evaluations; differences are pure signal. Everything
fp32, model.eval(). One npz per (arm,snap); resumable (existing npz skipped without --force).

Usage (GPU job):
  python3 experimental/feature_dim_probe.py --sync_root <ckpts/jointflow_tc> \
      --arms mot_nm,mot_sm,mot_nf,mot_sf --snaps ep15,ep30,ep45,ep60,ep75,ep90,ep105,final \
      --out <ckpts/jointflow_tc>/fdp
Smoke (CPU, synthetic windows, real weights):
  python3 experimental/feature_dim_probe.py --smoke --arms mot_nm --snaps ep15 --out /tmp/fdp
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_crossattn import load_cache  # noqa: E402
from train_lewam_unified import _IMG_MEAN, _IMG_STD  # noqa: E402
from lewam.models.module import SIGReg  # noqa: E402
from lewam.models.motflow import build_model as build_motflow  # noqa: E402

LOSS_NAMES = ["P", "Dfull", "Din", "Dtar", "S"]
PAIRS = [("P", "Dfull"), ("P", "Din"), ("P", "Dtar"), ("P", "S"),
         ("Dtar", "S"), ("Din", "Dtar"), ("Dfull", "S")]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync_root", default="/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/"
                                           "lewam/ckpts/jointflow_tc")
    ap.add_argument("--arms", default="mot_nm,mot_sm,mot_nf,mot_sf")
    ap.add_argument("--snaps", default="ep15,ep30,ep45,ep60,ep75,ep90,ep105,final")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_probe", type=int, default=2048,
                    help="windows for the spectrum / eigenbasis")
    ap.add_argument("--n_grad", type=int, default=512,
                    help="windows for parameter gradients, usage, ablation")
    ap.add_argument("--n_eval", type=int, default=1024,
                    help="frames on which the induced dz is evaluated")
    ap.add_argument("--grad_chunk", type=int, default=64)
    ap.add_argument("--enc_chunk", type=int, default=256)
    ap.add_argument("--abl_top", type=int, default=32)
    ap.add_argument("--abl_band", type=int, default=32)
    ap.add_argument("--dz_target", type=float, default=1e-3,
                    help="calibrated rms(dz)/rms(z) for the virtual step")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--deadline_min", type=int, default=0,
                    help="soft deadline: start no new combo after this many minutes (0 = off)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic probe windows (no frames cache), tiny sizes")
    return ap.parse_args()


def load_arm_model(sync_root, arm, snap, device):
    d = Path(sync_root) / f"tc_toolhang_probe_{arm}_s0"
    cfg = json.loads((d / "jointflow_config.json").read_text())
    cfg.setdefault("action_dim", int(cfg["fs"]) * int(cfg["action_raw_dim"]))
    assert cfg["model"] == "motflow", cfg["model"]
    assert not cfg["goal_conditioning"] and not cfg["state_residual"] \
        and not cfg["state_ema_target"], "analysis assumes plain TC MoT (no goal/residual/ema)"
    model = build_motflow(cfg)
    ckpt = d / ("jointflow_best.pt" if snap == "final" else f"snap_{snap}.pt")
    sd = torch.load(ckpt, map_location="cpu")
    res = model.load_state_dict(sd, strict=False)
    assert set(res.missing_keys) <= {"state_mu", "state_sd"} and not res.unexpected_keys, \
        f"{arm}/{snap}: missing={res.missing_keys} unexpected={res.unexpected_keys}"
    return model.to(device).float().eval(), cfg


def build_probe_windows(cfg, n_probe, seed, smoke):
    """The trainer's exact dataset over the TRAIN split, n_probe windows drawn with a fixed
    generator so every arm/snapshot (and every resumed run) sees identical windows."""
    if smoke:
        g = torch.Generator().manual_seed(seed)
        L, S, A, isz = cfg["policy_history_len"], cfg["num_states_pred"], \
            cfg["num_actions_pred"], cfg["img_size"]
        return dict(
            history_frames=torch.randint(0, 255, (n_probe, L, 3, isz, isz), generator=g,
                                         dtype=torch.uint8),
            history_pad=torch.zeros(n_probe, L),
            action_target=torch.randn(n_probe, A, cfg["action_raw_dim"], generator=g),
            action_valid=torch.ones(n_probe, A),
            state_frames=torch.randint(0, 255, (n_probe, S, 3, isz, isz), generator=g,
                                       dtype=torch.uint8),
            state_valid=torch.ones(n_probe, S))
    from train_jointflow import JointFlowDataset
    ca = argparse.Namespace(frames_cache=cfg["frames_cache"], dataset_name=cfg["dataset_name"],
                            frameskip=cfg["frameskip"], img_size=cfg["img_size"], cache_mmap=True)
    frames, a_frame, t_gidx, ep_base, frames_to_terminal, _ = load_cache(ca)
    gen = torch.Generator().manual_seed(cfg["seed"])            # the trainer's split, verbatim
    perm = torch.randperm(t_gidx.shape[0], generator=gen)
    perm = perm[frames_to_terminal[perm] >= cfg["frameskip"]]
    n_val = int(round((1 - cfg["train_split"]) * perm.numel()))
    train_idx = perm[n_val:]
    pick = torch.randperm(train_idx.numel(),
                          generator=torch.Generator().manual_seed(seed))[:n_probe]
    ds = JointFlowDataset(frames, a_frame, t_gidx, ep_base, frames_to_terminal, train_idx[pick],
                          cfg["policy_history_len"], cfg["num_actions_pred"], cfg["frameskip"],
                          cfg["num_states_pred"])
    rows = [ds[i] for i in range(len(ds))]
    hf, hp, at, av, sf, sv, _goal, _h = (torch.stack([r[j] for r in rows]) for j in range(8))
    return dict(history_frames=hf, history_pad=hp, action_target=at.float(), action_valid=av,
                state_frames=sf, state_valid=sv)


def encode_windows(model, pw, sl, device, mean, std, chunk, grad=False):
    """Mirror run_batch: one encoder call over (history frames ++ state frames) per chunk.
    Returns z_hist (n,L,zd), z_state (n,S,zd)."""
    hf, sf = pw["history_frames"][sl], pw["state_frames"][sl]
    n, L, S = hf.shape[0], hf.shape[1], sf.shape[1]
    zh, zs = [], []
    ctx = torch.enable_grad if grad else torch.no_grad
    with ctx():
        for i in range(0, n, chunk):
            h = hf[i:i + chunk].to(device)
            s = sf[i:i + chunk].to(device)
            b = h.shape[0]
            px = torch.cat([h.reshape(b * L, *h.shape[2:]), s.reshape(b * S, *s.shape[2:])])
            px = (px.float() / 255.0 - mean) / std
            z = model.encode(px)
            zh.append(z[:b * L].reshape(b, L, -1))
            zs.append(z[b * L:].reshape(b, S, -1))
    return torch.cat(zh), torch.cat(zs)


def encode_frames(encoder, frames_u8, device, mean, std, chunk, params=None):
    """Plain frame encoding, optionally through functional_call with shifted params."""
    out = []
    with torch.no_grad():
        for i in range(0, frames_u8.shape[0], chunk):
            px = (frames_u8[i:i + chunk].to(device).float() / 255.0 - mean) / std
            out.append(functional_call(encoder, params, (px,)) if params is not None
                       else encoder(px))
    return torch.cat(out)


def losses_on(model, sigreg, zh, pad, at, av, st, sv, seed, device, detach_target=False):
    """One (loss_action, loss_state, loss_sigreg) evaluation under forked, re-seeded RNG, so
    every call with the same seed draws identical tau/noise/projections."""
    devs = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(seed)
        if devs:
            torch.cuda.manual_seed_all(seed)
        la, ls = model.loss(zh, pad, at, av, st.detach() if detach_target else st, sv)
        zd = zh.shape[-1]
        zcat = torch.cat([zh[:, -1], st.reshape(-1, zd)]).unsqueeze(0)
        lg = sigreg(zcat)
    return la, ls, lg


def ablate(z, U_cols, m_cols):
    """Replace the variation of the given directions with their probe means: keeps the mean
    component, removes the information."""
    proj = z @ U_cols - m_cols                       # (..., n_cols)
    return z - proj @ U_cols.T


def usage_spectrum(rows, U):
    """rows (n, zd) of per-sample latent gradients -> mean squared projection per direction."""
    return (rows @ U).pow(2).mean(0)


def analyze_combo(args, model, cfg, pw, out_npz, arm, snap, device, mean, std, git_sha):
    t0 = time.time()
    zd = cfg["z_dim"]
    sigreg = SIGReg().to(device)
    seed = args.seed
    meta = dict(arm=arm, snap=snap, w_reg=cfg["w_reg"], state_head=cfg["mot_state_head"],
                n_probe=args.n_probe, n_grad=args.n_grad, n_eval=args.n_eval, seed=seed,
                dz_target=args.dz_target, git_sha=git_sha, time=time.strftime("%F %T"))
    res = {}

    # ---- 1. spectrum -------------------------------------------------------------------
    zh_all, zs_all = encode_windows(model, pw, slice(0, args.n_probe), device, mean, std,
                                    args.enc_chunk)
    z_t = zh_all[:, -1]
    z_next = zs_all[:, 0]
    mu_t = z_t.mean(0)
    C = torch.cov(z_t.T)
    lam, U = torch.linalg.eigh(C)
    order = torch.argsort(lam, descending=True)
    lam, U = lam[order].clamp_min(0), U[:, order]                       # (zd,), (zd, zd) cols
    tot = lam.sum().clamp_min(1e-12)
    res["lam"] = lam
    res["U_top"] = U[:, :64]
    res["eff_rank"] = (tot ** 2 / lam.pow(2).sum().clamp_min(1e-12)).reshape(1)
    res["top_share"] = torch.stack([lam[:k].sum() / tot for k in (1, 8, 32, 64)])
    res["var_disp"] = ((z_next - z_t) @ U).var(0)
    pooled = torch.cat([zh_all.reshape(-1, zd), zs_all.reshape(-1, zd)])
    m_proj = pooled.mean(0) @ U                                          # ablation centers

    # ---- 2. usage (diagnostic) ---------------------------------------------------------
    gsl = slice(0, args.n_grad)
    zh_g, zs_g = encode_windows(model, pw, gsl, device, mean, std, args.enc_chunk)
    zh_leaf = zh_g.detach().requires_grad_(True)
    st_leaf = zs_g.detach().requires_grad_(True)
    pad = pw["history_pad"][gsl].to(device)
    at = pw["action_target"][gsl].to(device)
    av = pw["action_valid"][gsl].to(device)
    sv = pw["state_valid"][gsl].to(device)
    la, ls, lg = losses_on(model, sigreg, zh_leaf, pad, at, av, st_leaf, sv, seed, device)
    gP = torch.autograd.grad(la, zh_leaf, retain_graph=True)[0]
    gD_zh, gD_st = torch.autograd.grad(ls, [zh_leaf, st_leaf], retain_graph=True)
    gS_zh, _gS_st = torch.autograd.grad(lg, [zh_leaf, st_leaf])
    res["usage_P"] = usage_spectrum(gP[:, -1], U)
    res["usage_Din"] = usage_spectrum(gD_zh[:, -1], U)
    res["usage_Dtar"] = usage_spectrum(gD_st[:, 0], U)
    res["usage_S"] = usage_spectrum(gS_zh[:, -1], U)
    base_losses = dict(P=la.item(), D=ls.item(), S=lg.item())
    meta["base_losses"] = base_losses

    # ---- 3. ablation (causal necessity) ------------------------------------------------
    groups = [[k] for k in range(args.abl_top)]
    groups += [list(range(a, min(a + args.abl_band, zd)))
               for a in range(args.abl_top, zd, args.abl_band)]
    dl = {n: [] for n in ("P", "D", "S")}
    zh_a, st_a = zh_g.detach(), zs_g.detach()
    with torch.no_grad():
        la0, ls0, lg0 = losses_on(model, sigreg, zh_a, pad, at, av, st_a, sv, seed, device)
        for cols in groups:
            Uc, mc = U[:, cols], m_proj[cols]
            la_d, ls_d, lg_d = losses_on(model, sigreg, ablate(zh_a, Uc, mc), pad, at, av,
                                         ablate(st_a, Uc, mc), sv, seed, device)
            dl["P"].append(la_d - la0)
            dl["D"].append(ls_d - ls0)
            dl["S"].append(lg_d - lg0)
    res["abl_groups_lo"] = torch.tensor([g[0] for g in groups])
    res["abl_groups_hi"] = torch.tensor([g[-1] for g in groups])
    for n in ("P", "D", "S"):
        res[f"abl_dL_{n}"] = torch.stack(dl[n])
    meta["abl_base"] = dict(P=la0.item(), D=ls0.item(), S=lg0.item())

    # ---- 4. induced representation updates ---------------------------------------------
    enc = model.encoder
    enc_names = [n for n, _ in enc.named_parameters()]
    base_params = dict(enc.named_parameters())
    buffers = dict(enc.named_buffers())
    acc = {n: [torch.zeros_like(base_params[k]) for k in enc_names]
           for n in ("P", "Dfull", "Din", "S")}
    nc = 0
    for i in range(0, args.n_grad, args.grad_chunk):
        csl = slice(i, min(i + args.grad_chunk, args.n_grad))
        zh_c, zs_c = encode_windows(model, pw, csl, device, mean, std, args.grad_chunk,
                                    grad=True)
        w = zh_c.shape[0] / args.n_grad
        sc = seed + 1000 + nc
        la1, ls1, lg1 = losses_on(model, sigreg, zh_c, pw["history_pad"][csl].to(device),
                                  pw["action_target"][csl].to(device),
                                  pw["action_valid"][csl].to(device), zs_c,
                                  pw["state_valid"][csl].to(device), sc, device)
        _la2, ls2, _ = losses_on(model, sigreg, zh_c, pw["history_pad"][csl].to(device),
                                 pw["action_target"][csl].to(device),
                                 pw["action_valid"][csl].to(device), zs_c,
                                 pw["state_valid"][csl].to(device), sc, device,
                                 detach_target=True)
        assert abs(la1.item() - _la2.item()) < 1e-5, "RNG fork broke: action losses differ"
        enc_plist = [p for _, p in enc.named_parameters()]
        for name, loss in (("P", la1), ("Dfull", ls1), ("Din", ls2), ("S", lg1)):
            gs = torch.autograd.grad(loss, enc_plist, retain_graph=True, allow_unused=True)
            for a, g in zip(acc[name], gs):
                if g is not None:
                    a.add_(g, alpha=w)
        del la1, ls1, lg1, _la2, ls2, zh_c, zs_c
        nc += 1
    acc["Dtar"] = [f - i for f, i in zip(acc["Dfull"], acc["Din"])]

    ev_frames = pw["history_frames"][:args.n_eval, -1]                  # current frames
    Z0 = encode_frames(enc, ev_frames, device, mean, std, args.enc_chunk)
    z_rms = Z0.pow(2).mean().sqrt().item()
    theta_norm = torch.sqrt(sum(p.pow(2).sum() for p in base_params.values())).item()

    def step_encode(g_list, eps):
        shifted = {k: base_params[k] - eps * g for k, g in zip(enc_names, g_list)}
        shifted.update(buffers)
        return encode_frames(enc, ev_frames, device, mean, std, args.enc_chunk, params=shifted)

    V, dz_norm, eps_used, g_norm = {}, {}, {}, {}
    for name in LOSS_NAMES:
        g_list = acc[name]
        gn = torch.sqrt(sum(g.pow(2).sum() for g in g_list)).item()
        g_norm[name] = gn
        if gn < 1e-20:
            V[name] = torch.zeros(args.n_eval, zd, device=device)
            dz_norm[name] = 0.0
            eps_used[name] = 0.0
            continue
        eps0 = 1e-4 * theta_norm / gn                                    # trial step
        dz0 = step_encode(g_list, eps0) - Z0
        rel0 = (dz0.pow(2).mean().sqrt() / max(z_rms, 1e-12)).item()
        eps = eps0 * args.dz_target / max(rel0, 1e-12)
        dz = step_encode(g_list, eps) - Z0
        if name == "P":                                                  # linearity check
            dz_h = step_encode(g_list, eps / 2) - Z0
            cos = torch.nn.functional.cosine_similarity(
                dz.flatten(), dz_h.flatten(), dim=0).item()
            ratio = (dz.norm() / dz_h.norm().clamp_min(1e-12)).item()
            meta["linearity"] = dict(cos=cos, ratio=ratio)
        V[name] = dz @ U
        dz_norm[name] = dz.pow(2).mean().sqrt().item()
        eps_used[name] = eps
    meta["g_norm"] = g_norm
    meta["eps_used"] = eps_used
    meta["dz_rel"] = {k: v / max(z_rms, 1e-12) for k, v in dz_norm.items()}

    for name in LOSS_NAMES:
        amp = V[name].pow(2).mean(0)
        res[f"amp_{name}"] = amp
        res[f"ampfrac_{name}"] = amp / amp.sum().clamp_min(1e-20)
    for n1, n2 in PAIRS:
        num = (V[n1] * V[n2]).mean(0)
        den = (V[n1].pow(2).mean(0) * V[n2].pow(2).mean(0)).sqrt().clamp_min(1e-20)
        res[f"a_{n1}_{n2}"] = num / den
        meta[f"gcos_{n1}_{n2}"] = torch.nn.functional.cosine_similarity(
            V[n1].flatten(), V[n2].flatten(), dim=0).item()

    # ---- 5. conflict index (top individually-ablated directions) -----------------------
    k = args.abl_top
    A_P = (res["abl_dL_P"][:k].clamp_min(0) / max(base_losses["P"], 1e-12))
    A_D = (res["abl_dL_D"][:k].clamp_min(0) / max(base_losses["D"], 1e-12))
    for dv in ("Dfull", "Dtar"):
        res[f"conflict_{dv}"] = (A_P * A_D).sqrt() * (-res[f"a_P_{dv}"][:k]).clamp_min(0)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, meta=json.dumps(meta),
             **{key: v.detach().cpu().numpy() for key, v in res.items()})
    er = res["eff_rank"].item()
    print(f"[fdp] {arm} {snap} done {time.time()-t0:.0f}s eff_rank={er:.1f} "
          f"top8={res['top_share'][1].item():.2f} "
          f"gcos(P,Dfull)={meta['gcos_P_Dfull']:+.3f} gcos(P,Dtar)={meta['gcos_P_Dtar']:+.3f} "
          f"gcos(P,S)={meta['gcos_P_S']:+.3f} lin={meta.get('linearity', {})}", flush=True)


def main():
    args = parse_args()
    if args.smoke:
        args.n_probe, args.n_grad, args.n_eval = 48, 16, 32
        args.grad_chunk, args.enc_chunk, args.abl_top, args.abl_band = 8, 16, 4, 128
        torch.set_num_threads(16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = _IMG_MEAN.to(device), _IMG_STD.to(device)
    git_sha = "unknown"
    try:
        import subprocess
        git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True, cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except Exception:
        pass
    out = Path(args.out)
    arms, snaps = args.arms.split(","), args.snaps.split(",")
    cfg0 = json.loads((Path(args.sync_root) / f"tc_toolhang_probe_{arms[0]}_s0" /
                       "jointflow_config.json").read_text())
    pw = build_probe_windows(cfg0, args.n_probe, args.seed, args.smoke)
    print(f"[fdp] probe windows built n={args.n_probe} device={device.type}", flush=True)
    t_start = time.time()
    for arm in arms:
        for snap in snaps:
            out_npz = out / f"fdp_{arm}_{snap}.npz"
            if out_npz.exists() and not args.force:
                print(f"[fdp] skip {arm} {snap} (exists)", flush=True)
                continue
            if args.deadline_min and (time.time() - t_start) / 60 >= args.deadline_min:
                print(f"[fdp] DEADLINE_SKIP {arm} {snap}", flush=True)
                continue
            model, cfg = load_arm_model(args.sync_root, arm, snap, device)
            analyze_combo(args, model, cfg, pw, out_npz, arm, snap, device, mean, std, git_sha)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print("[fdp] ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
