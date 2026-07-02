"""Inference SPEED / FLOPs benchmark for the LeWAM control methods.

Measures per-ACTION-decision wall-clock (mean +/- std over >=100 reps, after
warmup) and FLOPs/action for the four control methods on the SAME box / model /
GPU / dtype, using the real trained `can` base (single robomimic env):

  1. LeWAM GC head (ours, planning-free)  -- one forward pass:
       encode(HS=3 frames) -> action_predictor (6-layer ARPredictor) ->
       horizon_modulator (AdaLN-Zero) -> action_decoder (MLP) -> action block.
       This is one step of jepa.JEPA.intention_rollout.
  2. gcidm (Markovian, planning-free)      -- encode(1 frame) -> GCIDMHead -> action.
  3. CEM planning (le-wm solver)           -- the full CEMSolver.solve() per action:
       num_samples * n_steps WM rollouts of horizon H (predict()), via get_cost.
  4. Diffusion Policy / LDP (diffusion head) -- K reverse-diffusion steps through
       the DiffusionHead conditioned on the intention embedding.

Run from the le-wm-repro dir (so jepa/module/gip/gcidm import).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import time
import statistics
import contextlib

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate as hydra_instantiate

import lewam.models.jepa as jepa  # noqa
from lewam.models.module import MLP, DiffusionHead
from lewam.models.gcidm import GCIDMHead

CKPT = "/mnt/minghao_data/.stable-wm/checkpoints"
GC_DIR = f"{CKPT}/can_gc_ours"
GCIDM_DIR = f"{CKPT}/can_gcidm"

DEVICE = "cuda"
DTYPE = torch.float32          # eval default (model weights are fp32; CEM uses model dtype)
IMG = 224
HS = 3                         # history_size for the GC head (training ctx_len)
N_WARMUP = 20
N_REPS = 120                   # >= 100 reps for mean +/- std

# CEM config (config/eval/solver/cem.yaml + config/eval/robomimic.yaml)
CEM_NUM_SAMPLES = 300
CEM_N_STEPS = 30
CEM_HORIZON = 5                # plan_config.horizon
CEM_ACTION_BLOCK = 5           # plan_config.action_block (frameskip)
DP_N_STEPS = 50                # DiffusionHead default n_steps (DP/LDP ~50-100)


def synced_time(fn, n_warmup, n_reps):
    """Warm up then time fn() n_reps times with CUDA syncs. Returns (mean_ms, std_ms, raw_ms)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(times), statistics.pstdev(times), times


def count_flops(module, inputs):
    """fvcore FLOPs (multiply-adds counted as fvcore does -> MACs ~ FLOPs/2; we
    report the fvcore total and ALSO 2x as 'FLOPs' where MAC convention matters).
    Returns fvcore 'flops' total (which is really MACs in fvcore's accounting)."""
    from fvcore.nn import FlopCountAnalysis
    module.eval()
    with torch.no_grad():
        fca = FlopCountAnalysis(module, inputs)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = fca.total()
    return total  # fvcore total() = number of multiply-accumulates (MACs)


# --------------------------------------------------------------------------- #
# Build the real trained model                                                 #
# --------------------------------------------------------------------------- #
def build_gc_model():
    cfg = OmegaConf.create(json.loads(open(f"{GC_DIR}/config.json").read()))
    model = hydra_instantiate(cfg)
    embed_dim = int(cfg.predictor.input_dim)
    adim = int(cfg.action_encoder.input_dim)
    # rebuild the GIP runtime modules exactly as gip.load_gip_model does
    model.action_predictor = hydra_instantiate(cfg.predictor)
    model.action_decoder = MLP(embed_dim, 2048, adim)      # action_head.type == mse
    sd = torch.load(f"{GC_DIR}/weights_epoch_1.pt", map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("alignment_projection")}
    res = model.load_state_dict(sd, strict=False)
    print(f"[build] GC model loaded: adim={adim} embed_dim={embed_dim} "
          f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)} "
          f"horizon_conditioned={model.horizon_conditioned}")
    assert not res.unexpected_keys, res.unexpected_keys
    return model.to(DEVICE).eval(), embed_dim, adim


def build_gcidm_head():
    gcfg = json.loads(open(f"{GCIDM_DIR}/gcidm_config.json").read())
    head = GCIDMHead(emb_dim=int(gcfg["emb_dim"]), action_dim=int(gcfg["action_dim"]),
                     hidden_dim=int(gcfg["hidden_dim"]), n_freqs=int(gcfg["n_freqs"]),
                     dropout=float(gcfg["dropout"]))
    sd = torch.load(f"{GCIDM_DIR}/gcidm_head_best.pt", map_location="cpu")
    res = head.load_state_dict(sd, strict=True)
    print(f"[build] GCIDM head loaded: emb_dim={gcfg['emb_dim']} action_dim={gcfg['action_dim']} "
          f"hidden={gcfg['hidden_dim']}")
    return head.to(DEVICE).eval(), gcfg


def main():
    torch.manual_seed(0)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"=== GPU: {gpu_name} | dtype={DTYPE} | warmup={N_WARMUP} reps={N_REPS} ===")

    model, D, ADIM = build_gc_model()
    gcidm, gcfg = build_gcidm_head()
    model.interpolate_pos_encoding = True

    # a DiffusionHead with the same input/output dims as the MSE decoder (intention->action)
    diff = DiffusionHead(D, 2048, ADIM, n_steps=DP_N_STEPS).to(DEVICE).eval()

    nparams = lambda m: sum(p.numel() for p in m.parameters())
    print(f"[params] encoder={nparams(model.encoder)/1e6:.2f}M "
          f"action_predictor={nparams(model.action_predictor)/1e6:.2f}M "
          f"action_decoder(MLP)={nparams(model.action_decoder)/1e6:.3f}M "
          f"horizon_modulator={nparams(model.horizon_modulator)/1e6:.3f}M "
          f"predictor(WM)={nparams(model.predictor)/1e6:.2f}M "
          f"gcidm_head={nparams(gcidm)/1e6:.2f}M diffusion_head={nparams(diff)/1e6:.3f}M")

    B = 1
    px_hs = torch.randn(B, HS, 3, IMG, IMG, device=DEVICE, dtype=DTYPE)   # HS-frame history
    px_1 = torch.randn(B, 1, 3, IMG, IMG, device=DEVICE, dtype=DTYPE)     # single frame
    h_norm = torch.full((B,), 0.5, device=DEVICE, dtype=DTYPE)
    results = {}

    # ----------------------------------------------------------------------- #
    # 1. LeWAM GC head (ours) -- one forward pass, mirrors HistoryBCPolicy.get_action
    #    (eval.histbc): per decision the policy encodes ONLY the 1 NEW frame, appends
    #    its latent to a maxlen-HS deque, then runs action_predictor over the HS-frame
    #    latent history -> horizon_modulator (AdaLN-Zero) -> action_decoder -> block.
    #    The older HS-1 latents are cached, so the real per-action cost is 1 encode + head.
    # ----------------------------------------------------------------------- #
    # warm a cached HS-1 frame-latent history (encoded once, reused like the deque)
    with torch.no_grad():
        _hist = model.encode({"pixels": px_hs[:, :HS - 1]})["emb"].detach()  # (B, HS-1, D)
    @torch.no_grad()
    def lewam_gc():
        z = model.encode({"pixels": px_1})["emb"]                 # encode 1 NEW frame -> (B,1,D)
        emb = torch.cat([_hist, z], dim=1)                        # (B, HS, D) deque
        past = torch.zeros(B, HS, D, device=DEVICE, dtype=DTYPE)
        out = model.action_predictor(emb[:, -HS:], past, None)    # (B, HS, D)
        intention = model.horizon_modulator(out, h_norm)          # AdaLN-Zero (horizon_conditioned)
        a = model.action_decoder(intention[:, -1])                # (B, ADIM)
        return a
    m, s, _ = synced_time(lewam_gc, N_WARMUP, N_REPS)
    results["LeWAM-GC"] = (m, s)
    print(f"[time] LeWAM-GC : {m:.3f} +/- {s:.3f} ms/action  (1 encode + predictor + head)")

    # ----------------------------------------------------------------------- #
    # 2. gcidm (Markovian) -- encode 1 frame + encode goal + GCIDMHead
    #    (the eval policy encodes z_t and z_goal once per decision; we count both
    #     encodes + the head, matching GCIDMRobomimicPolicy.get_action)
    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def gcidm_step():
        z_t = model.encode({"pixels": px_1})["emb"][:, 0]    # (B, D)
        z_g = model.encode({"pixels": px_1})["emb"][:, 0]    # (B, D) goal
        return gcidm(z_t, z_g, h_norm)                       # (B, ADIM)
    m, s, _ = synced_time(gcidm_step, N_WARMUP, N_REPS)
    results["gcidm"] = (m, s)
    print(f"[time] gcidm    : {m:.3f} +/- {s:.3f} ms/action  (2 encodes + head)")

    # gcidm head-only (excludes the encode, to isolate the head cost)
    z_t_fix = model.encode({"pixels": px_1})["emb"][:, 0].detach()
    z_g_fix = z_t_fix.clone()
    @torch.no_grad()
    def gcidm_headonly():
        return gcidm(z_t_fix, z_g_fix, h_norm)
    mh, sh, _ = synced_time(gcidm_headonly, N_WARMUP, N_REPS)
    print(f"[time] gcidm head-only: {mh:.4f} +/- {sh:.4f} ms")

    # ----------------------------------------------------------------------- #
    # 3. CEM planning -- the full per-action solve(): num_samples * n_steps WM
    #    rollouts of horizon H. We time the actual CEMSolver.solve() on get_cost.
    # ----------------------------------------------------------------------- #
    # Build a minimal info_dict matching what the robomimic eval passes to the solver:
    #   pixels (B, HS, C, H, W) init frames, goal (B, C, H, W), action placeholder.
    # get_cost reads info["goal"] and the init pixels, rolls predict() over horizon.
    from stable_worldmodel.solver.cem import CEMSolver

    class _Cfg:
        action_block = CEM_ACTION_BLOCK
        horizon = CEM_HORIZON

    # action_space.shape: (n_envs, action_raw_dim) ; flattened action_dim = raw * action_block
    action_raw = int(gcfg["action_raw_dim"])   # 7 for can
    import numpy as np
    from gymnasium.spaces import Box
    aspace = Box(low=-1, high=1, shape=(1, action_raw), dtype=np.float32)

    solver = CEMSolver(model=model, batch_size=1, num_samples=CEM_NUM_SAMPLES,
                       var_scale=1.0, n_steps=CEM_N_STEPS, topk=30, device=DEVICE, seed=42)
    solver.configure(action_space=aspace, n_envs=1, config=_Cfg())

    # info_dict for one env, shaped to the solver->get_cost->rollout contract:
    #   the solver expands each tensor v -> v.unsqueeze(1).expand(B, num_samples, *v.shape[1:]),
    #   so input pixels (B, T0, C, H, W) -> (B, S, T0, C, H, W) which rollout consumes
    #   (H = pixels.size(2) = T0 init frames). get_cost does goal[:,0] -> (B, T0, C, H, W)
    #   then encodes (needs 5-D). LeWM planning uses T0=1 (current frame).
    T0 = 1
    info_dict = {
        "pixels": torch.randn(1, T0, 3, IMG, IMG, device=DEVICE, dtype=DTYPE),
        "goal": torch.randn(1, T0, 3, IMG, IMG, device=DEVICE, dtype=DTYPE),
        "action": torch.zeros(1, T0, action_raw * CEM_ACTION_BLOCK, device=DEVICE, dtype=DTYPE),
    }

    def _make_info():
        return {k: v.clone() for k, v in info_dict.items()}

    @torch.no_grad()
    def cem_solve():
        out = solver.solve(_make_info())
        return out
    # CEM is slow -> fewer reps but still >= the requested floor where feasible.
    CEM_WARMUP, CEM_REPS = 3, 12
    # silence the per-call "CEM solve time" print by capturing stdout
    import io, sys
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m, s, raw = synced_time(cem_solve, CEM_WARMUP, CEM_REPS)
    results["CEM"] = (m, s)
    # also surface one real captured solver-print line for the report
    solver_lines = [l for l in buf.getvalue().splitlines() if "CEM solve time" in l]
    print(f"[time] CEM      : {m:.1f} +/- {s:.1f} ms/action  "
          f"(num_samples={CEM_NUM_SAMPLES} n_steps={CEM_N_STEPS} H={CEM_HORIZON}, reps={CEM_REPS})")
    if solver_lines:
        print(f"[time] CEM solver self-report (sample): {solver_lines[-1]}")

    # ----------------------------------------------------------------------- #
    # 4. Diffusion Policy / LDP head -- K reverse-diffusion steps. We add the
    #    encode(HS) + action_predictor cost (to produce the conditioning intention),
    #    then DP denoising. Report both the full DP-action and the head-only K-step.
    # ----------------------------------------------------------------------- #
    @torch.no_grad()
    def dp_step():
        z = model.encode({"pixels": px_1})["emb"]                 # 1 NEW frame (cached history)
        emb = torch.cat([_hist, z], dim=1)
        past = torch.zeros(B, HS, D, device=DEVICE, dtype=DTYPE)
        out = model.action_predictor(emb[:, -HS:], past, None)
        intention = model.horizon_modulator(out, h_norm)[:, -1]   # (B, D) conditioning
        return diff(intention)                                    # K reverse-diffusion steps
    m, s, _ = synced_time(dp_step, N_WARMUP, N_REPS)
    results["DP/LDP"] = (m, s)
    print(f"[time] DP/LDP   : {m:.3f} +/- {s:.3f} ms/action  (n_steps={DP_N_STEPS} denoising)")

    cond_fix = model.encode({"pixels": px_hs})["emb"]
    cond_fix = model.action_predictor(cond_fix[:, -HS:], torch.zeros(B, HS, D, device=DEVICE), None)
    cond_fix = model.horizon_modulator(cond_fix, h_norm)[:, -1].detach()
    @torch.no_grad()
    def dp_headonly():
        return diff(cond_fix)
    mdh, sdh, _ = synced_time(dp_headonly, N_WARMUP, N_REPS)
    print(f"[time] DP head-only ({DP_N_STEPS} steps): {mdh:.3f} +/- {sdh:.3f} ms")

    # ----------------------------------------------------------------------- #
    # FLOPs (fvcore total() = MACs). Build per-module FLOPs and compose.        #
    # ----------------------------------------------------------------------- #
    print("\n=== FLOPs (fvcore total = MACs; FLOPs ~ 2x MACs) ===")

    # encoder: vit-tiny on one 224 frame. wrap so FlopCountAnalysis sees a clean module.
    class _EncWrap(torch.nn.Module):
        def __init__(self, enc): super().__init__(); self.enc = enc
        def forward(self, px):  # px (N,3,224,224)
            return self.enc(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    enc_wrap = _EncWrap(model.encoder).to(DEVICE).eval()
    enc_macs = count_flops(enc_wrap, (torch.randn(1, 3, IMG, IMG, device=DEVICE),))

    # projector MLP (per token)
    proj_macs = count_flops(model.projector, (torch.randn(1, D, device=DEVICE),))

    # action_predictor (ARPredictor): inputs (x (B,HS,D), c (B,HS,D), proprio=None)
    class _APWrap(torch.nn.Module):
        def __init__(self, ap): super().__init__(); self.ap = ap
        def forward(self, x, c): return self.ap(x, c, None)
    ap_wrap = _APWrap(model.action_predictor).to(DEVICE).eval()
    ap_macs = count_flops(ap_wrap, (torch.randn(1, HS, D, device=DEVICE),
                                    torch.randn(1, HS, D, device=DEVICE)))

    # WM predictor (same ARPredictor class as action_predictor) for CEM rollout:
    pred_macs = ap_macs  # same architecture (depth 6 ARPredictor), HS-token forward
    pred_proj_macs = count_flops(model.pred_proj, (torch.randn(1, D, device=DEVICE),))

    # horizon_modulator
    hm_macs = count_flops(model.horizon_modulator,
                          (torch.randn(1, HS, D, device=DEVICE), h_norm))
    # action_decoder MLP
    dec_macs = count_flops(model.action_decoder, (torch.randn(1, D, device=DEVICE),))
    # action_encoder Embedder (used per rolled action in the WM rollout / GC autoregress)
    aenc_macs = count_flops(model.action_encoder,
                            (torch.randn(1, 1, ADIM, device=DEVICE),))
    # gcidm head
    gc_macs = count_flops(gcidm, (torch.randn(1, D, device=DEVICE),
                                  torch.randn(1, D, device=DEVICE), h_norm))
    # diffusion head one eps step (net forward); reverse loop runs n_steps of these
    class _EpsWrap(torch.nn.Module):
        def __init__(self, d): super().__init__(); self.d = d
        def forward(self, x_t, t, cond): return self.d._eps(x_t, t, cond)
    eps_wrap = _EpsWrap(diff).to(DEVICE).eval()
    eps_macs = count_flops(eps_wrap, (torch.randn(1, ADIM, device=DEVICE),
                                      torch.zeros(1, dtype=torch.long, device=DEVICE),
                                      torch.randn(1, D, device=DEVICE)))

    print(f"[macs] encoder(1 frame)      = {enc_macs/1e6:.2f} M")
    print(f"[macs] projector             = {proj_macs/1e6:.3f} M")
    print(f"[macs] action_predictor(HS)  = {ap_macs/1e6:.2f} M")
    print(f"[macs] WM predictor(HS)      = {pred_macs/1e6:.2f} M (+pred_proj {pred_proj_macs/1e6:.3f}M)")
    print(f"[macs] horizon_modulator     = {hm_macs/1e6:.3f} M")
    print(f"[macs] action_decoder(MLP)   = {dec_macs/1e6:.3f} M")
    print(f"[macs] action_encoder        = {aenc_macs/1e6:.3f} M")
    print(f"[macs] gcidm head            = {gc_macs/1e6:.3f} M")
    print(f"[macs] diffusion eps step    = {eps_macs/1e6:.3f} M")

    # ---- compose per-action MACs ----
    enc_full = enc_macs + proj_macs   # encode one frame -> latent

    # 1. LeWAM-GC: encode 1 NEW frame (HS-1 latents cached in deque) + action_predictor
    #    over HS tokens + horizon_modulator + decoder
    macs_lewam = 1 * enc_full + ap_macs + hm_macs + dec_macs
    # 2. gcidm: 2 encodes (z_t, z_goal) + head
    macs_gcidm = 2 * enc_full + gc_macs
    # 3. CEM: per solve() the init frame (T0) and goal (T0) are each encoded ONCE
    #    (rollout encodes _init at S-index 0 then expands; get_cost encodes goal once),
    #    then for EACH of num_samples candidates the WM is rolled over the horizon:
    #    rollout runs (horizon - T0) inner predict steps + 1 final predict = horizon
    #    predict() calls, each = ARPredictor(HS tokens) + pred_proj + an action_encoder
    #    on the running action block. This whole get_cost is re-run for n_steps CEM
    #    iterations (fresh candidate population each iter).
    T0 = 1
    n_predict = CEM_HORIZON            # (horizon - T0) inner + 1 final = horizon
    per_rollout = n_predict * (pred_macs + pred_proj_macs) + CEM_HORIZON * aenc_macs
    macs_cem = (CEM_N_STEPS * (CEM_NUM_SAMPLES * per_rollout + 2 * T0 * enc_full))
    # 4. DP/LDP: encode 1 NEW frame + action_predictor + horizon_modulator (conditioning) +
    #    n_steps diffusion eps forwards (our intention conditions a DP-style decoder)
    macs_dp = 1 * enc_full + ap_macs + hm_macs + DP_N_STEPS * eps_macs

    macs = {"LeWAM-GC": macs_lewam, "gcidm": macs_gcidm, "CEM": macs_cem, "DP/LDP": macs_dp}

    # ----------------------------------------------------------------------- #
    # REPORT TABLE                                                             #
    # ----------------------------------------------------------------------- #
    cem_ms = results["CEM"][0]
    print("\n================ SPEED / FLOPs TABLE (can, " + gpu_name + ", fp32) ================")
    print(f"{'method':<12}{'ms/action (mean+/-std)':<26}{'GFLOPs/action':<16}"
          f"{'x vs CEM (time)':<18}{'x vs CEM (FLOPs)':<16}")
    for name in ["LeWAM-GC", "gcidm", "CEM", "DP/LDP"]:
        mm, ss = results[name]
        gflops = 2 * macs[name] / 1e9   # FLOPs = 2 * MACs
        spd_t = cem_ms / mm
        spd_f = (2 * macs["CEM"] / 1e9) / gflops
        print(f"{name:<12}{f'{mm:.3f} +/- {ss:.3f}':<26}{f'{gflops:.4f}':<16}"
              f"{f'{spd_t:.1f}x':<18}{f'{spd_f:.1f}x':<16}")

    # headline speedups
    lg = results["LeWAM-GC"][0]; gc = results["gcidm"][0]; dp = results["DP/LDP"][0]
    print("\n--- headline speedups (wall-clock) ---")
    print(f"LeWAM-GC is {cem_ms/lg:.1f}x faster than CEM, {dp/lg:.1f}x faster than DP/LDP")
    print(f"gcidm    is {cem_ms/gc:.1f}x faster than CEM, {dp/gc:.1f}x faster than DP/LDP")
    print("\n--- headline speedups (FLOPs) ---")
    g = {k: 2 * v / 1e9 for k, v in macs.items()}
    print(f"LeWAM-GC uses {g['CEM']/g['LeWAM-GC']:.1f}x fewer FLOPs than CEM, "
          f"{g['DP/LDP']/g['LeWAM-GC']:.1f}x fewer than DP/LDP")
    print(f"gcidm    uses {g['CEM']/g['gcidm']:.1f}x fewer FLOPs than CEM")
    print("\n[config] CEM: num_samples=%d n_steps=%d horizon=%d action_block=%d | "
          "DP: n_steps=%d | batch=1 | dtype=fp32 | %s"
          % (CEM_NUM_SAMPLES, CEM_N_STEPS, CEM_HORIZON, CEM_ACTION_BLOCK, DP_N_STEPS, gpu_name))


if __name__ == "__main__":
    main()
