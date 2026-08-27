"""World-model action discrimination (owner question 2026-08-27): can a learned latent dynamics
tell the expert's action from wrong ones, and is that a JEPA-family property?

Same test on three world models of pusht: the official LeWM checkpoint (JEPA, MSE predictor),
LeWAM-unified (JEPA-style MSE dynamics on the aggregated context), and jointflow (state flow).

Per dataset anchor t: history frames per the model's convention, the goal frame at
t + goal_offset_obs*fs, the real next frame z(t+fs), and the expert's action block a* (fs steps).
Candidate blocks, all in the model's own z-scored action space:
  expert       a*
  zero         the all-zero block (stay put)
  neg          -a* (move the opposite way)
  shuf_other   a block taken from a random OTHER anchor (expert marginal, wrong state)
  shuf_ep      a block from the same episode at a random other time (wrong phase)
  pert{s}      a* + s * N(0, I) per step and dim, s in --sigmas (z-scored units)
  uniform      each step ~ U[min, max] per raw dim (the dataset's action box)
For each candidate the model predicts z_next; we record its distance to the goal and to the
REAL next latent. Two unit-free scores per candidate type: the fraction of (anchor, candidate)
pairs where the expert's prediction is closer to the goal than the candidate's (goal_acc) and
where it is closer to the real next latent (truth_acc); 0.5 = coin flip. Mean gaps are also
saved in raw latent units and in units of the model's own mean real one-step progress.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py  # noqa: E402

_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--lewm_dir", default="", help="folder with config.json + weights_epoch_*.pt")
    ap.add_argument("--uni", default="", help="comma list of name=dir (lewam_unified_*.pt + config)")
    ap.add_argument("--jf", default="", help="comma list of name=dir (jointflow_best.pt + config)")
    ap.add_argument("--swm_home", default="/tmp/wm_discrim_home")
    ap.add_argument("--n_anchors", type=int, default=200)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--goal_offset_obs", type=int, default=10)
    ap.add_argument("--sigmas", default="0.25,0.5,1,2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="/tmp/wm_discrim.json")
    ap.add_argument("--dump_raw", default="")
    return ap.parse_args()


def to_px(frames_u8):
    px = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).float() / 255.0
    return (px - _IMG_MEAN) / _IMG_STD


# ----------------------------------------------------------------------------- adapters
class LeWMAdapter:
    """Official LeWM: 3-frame history at fs spacing, one z-scored 10-dim action block per frame;
    predictor over (emb, act_emb) returns the next embedding for the last frame."""
    name = "lewm"

    def __init__(self, folder, fs, amean, astd):
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        raw = json.loads(open(os.path.join(folder, "config.json")).read())
        # the official checkpoint was built from le-wm's jepa.py / module.py, which live in
        # stable_worldmodel.wm.lewm (lewm.LeWM, module.*); our lewam/models forks differ in shape
        def retarget(o):
            if isinstance(o, dict):
                o = {k: retarget(v) for k, v in o.items()}
                t = o.get("_target_")
                if t == "module.ARPredictor":            # renamed in the package, same kwargs/forward
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
        pts = sorted([p for p in os.listdir(folder) if p.endswith(".pt")])
        sd = torch.load(os.path.join(folder, pts[-1]), map_location="cpu")
        res = model.load_state_dict(sd, strict=False)
        print(f"[lewm] loaded {pts[-1]} missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
              flush=True)
        self.m = model.eval()
        self.fs = fs
        self.n_hist = int(cfg.predictor.num_frames)
        self.hist_offsets = [-(self.n_hist - 1 - i) * fs for i in range(self.n_hist)]   # e.g. [-10,-5,0]
        self.amean, self.astd = amean, astd
        self.n_blocks_needed = self.n_hist           # one action block per history frame

    @torch.no_grad()
    def encode(self, px):
        cls = self.m.encoder(px, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        return self.m.projector(cls)

    @torch.no_grad()
    def predict(self, z_hist, blocks_z):
        """z_hist (N, n_hist, D); blocks_z (N, n_hist, fs, adim) z-scored, last block = candidate."""
        N = z_hist.shape[0]
        act = blocks_z.reshape(N, self.n_hist, -1)
        act_emb = self.m.action_encoder(act)
        preds = self.m.predictor(z_hist, act_emb)
        preds = self.m.pred_proj(preds.reshape(N * self.n_hist, -1)).reshape(N, self.n_hist, -1)
        return preds[:, -1]


class UnifiedAdapter:
    """LeWAM-unified: context_len frames at fs spacing -> aggregate -> c_t; dynamics(c_t, block)."""

    def __init__(self, name, run_dir, swm_home):
        os.makedirs(os.path.join(swm_home, "checkpoints"), exist_ok=True)
        link = os.path.join(swm_home, "checkpoints", name)
        if not os.path.exists(link):
            os.symlink(run_dir, link)
        from lewam.models.gip import load_lewam_unified_model
        model, cfg = load_lewam_unified_model(name)
        self.name = f"uni:{name}"
        self.m = model.eval()
        self.cfg = cfg
        self.fs = int(cfg["frameskip"])
        ctx = int(cfg["context_len"])
        self.hist_offsets = [-(ctx - 1 - i) * self.fs for i in range(ctx)]
        self.amean = np.asarray(cfg["action_mean"], np.float32)
        self.astd = np.asarray(cfg["action_std"], np.float32)
        self.action_cond = bool(cfg.get("agg_action_cond", False))
        self.block_dim = int(cfg["action_dim"])
        self.n_blocks_needed = 1

    @torch.no_grad()
    def encode(self, px):
        return self.m.encode(px)

    @torch.no_grad()
    def predict(self, z_hist, blocks_z):
        N, ctx = z_hist.shape[:2]
        pa = pm = None
        if self.action_cond:
            pa = torch.zeros(N, ctx, self.block_dim, dtype=z_hist.dtype)
            pm = torch.zeros(N, ctx, dtype=torch.bool)
        c = self.m.aggregate(z_hist, pa, pm)[:, -1]
        a = blocks_z[:, -1].reshape(N, -1)
        dyn = self.m.dynamics
        if dyn.__class__.__name__ == "PrefixDynamics":
            return dyn(c, a[:, None, :], None)[:, 0]
        return dyn(c, a, torch.zeros_like(c))


class JointFlowAdapter:
    """jointflow / twinflow: history_len frames at fs spacing; the state flow under a clean
    10-token plan = [candidate block; expert's next block], fixed noise per anchor."""

    def __init__(self, name, run_dir, swm_home):
        os.makedirs(os.path.join(swm_home, "checkpoints"), exist_ok=True)
        link = os.path.join(swm_home, "checkpoints", name)
        if not os.path.exists(link):
            os.symlink(run_dir, link)
        from lewam.models.gip import load_jointflow_model
        model, cfg = load_jointflow_model(name)
        self.name = f"jf:{name}"
        self.m = model.eval()
        self.cfg = cfg
        self.fs = int(cfg["fs"])
        hl = int(cfg["policy_history_len"])
        self.hist_offsets = [-(hl - 1 - i) * self.fs for i in range(hl)]
        self.amean = np.asarray(cfg["action_mean"], np.float32)
        self.astd = np.asarray(cfg["action_std"], np.float32)
        self.n_blocks_needed = 1
        self.n_chunks = int(model.num_actions) // self.fs      # 2 for a10 fs5
        self.gen = torch.Generator().manual_seed(12345)
        self.next_block_z = None                                # set per anchor by the driver

    @torch.no_grad()
    def encode(self, px):
        return self.m.encode(px)

    @torch.no_grad()
    def predict(self, z_hist, blocks_z):
        N = z_hist.shape[0]
        cand = blocks_z[:, -1]                                  # (N, fs, adim)
        plan = [cand] + [self.next_block_z.expand(N, -1, -1)] * (self.n_chunks - 1)
        plan = torch.cat(plan, 1)                               # (N, num_actions, adim)
        pad = torch.zeros(N, z_hist.shape[1], dtype=torch.bool)
        noise_a = torch.randn(1, self.m.num_actions, self.m.action_raw_dim, generator=self.gen).expand(N, -1, -1)
        noise_s = torch.randn(1, self.m.num_states, self.m.z_dim, generator=self.gen).expand(N, -1, -1)
        return self.m.sample_inpaint(z_hist, pad, plan, noise_a, noise_s)[:, 0]


# ----------------------------------------------------------------------------- driver
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.environ["STABLEWM_HOME"] = args.swm_home
    sigmas = [float(s) for s in args.sigmas.split(",") if s]

    f = h5py.File(args.h5, "r")
    ep_off = np.asarray(f["ep_offset"][:]).reshape(-1)
    ep_len = np.asarray(f["ep_len"][:]).reshape(-1)
    A = np.asarray(f["action"][:], np.float32)                 # (N, adim) raw
    amean, astd = A.mean(0), A.std(0) + 1e-6
    amin, amax = A.min(0), A.max(0)
    adim = A.shape[1]

    adapters = []
    if args.lewm_dir:
        adapters.append(LeWMAdapter(args.lewm_dir, fs=5, amean=amean, astd=astd))
    for item in [s for s in args.uni.split(",") if s]:
        n, d = item.split("=")
        adapters.append(UnifiedAdapter(n, d, args.swm_home))
    for item in [s for s in args.jf.split(",") if s]:
        n, d = item.split("=")
        adapters.append(JointFlowAdapter(n, d, args.swm_home))
    assert adapters, "no models given"
    fs = adapters[0].fs
    assert all(a.fs == fs for a in adapters), "all models must share the frameskip"
    max_back = max(-min(a.hist_offsets) for a in adapters) + fs * max(a.n_blocks_needed - 1 for a in adapters)
    goff = args.goal_offset_obs * fs

    # anchors: t with room for the deepest history, the real next frame and the goal
    cand = [(e, int(ep_off[e]), int(ep_len[e])) for e in range(len(ep_len))
            if int(ep_len[e]) > max_back + goff + 1]
    picks = []
    for _ in range(args.n_anchors):
        e, lo, L = cand[int(rng.integers(len(cand)))]
        t = lo + int(rng.integers(max_back, L - goff - 1))
        picks.append((e, lo, L, t))
    K = args.k

    def block(t0):
        return A[t0:t0 + fs]                                    # raw (fs, adim)

    # candidate blocks in RAW space per anchor (shared across models; z-scored per model later)
    types = ["expert", "zero", "neg", "shuf_other", "shuf_ep"] + [f"pert{s:g}" for s in sigmas] + ["uniform"]
    raw_cands = {ty: [] for ty in types}
    for i, (e, lo, L, t) in enumerate(picks):
        a_star = block(t)
        raw_cands["expert"].append(a_star[None])
        raw_cands["zero"].append(np.zeros_like(a_star)[None])
        raw_cands["neg"].append((-a_star)[None])
        others = rng.choice([j for j in range(len(picks)) if j != i], K, replace=len(picks) - 1 < K)
        raw_cands["shuf_other"].append(np.stack([block(picks[j][3]) for j in others]))
        ts = rng.integers(lo, lo + L - fs, K)
        raw_cands["shuf_ep"].append(np.stack([block(int(x)) for x in ts]))
        for s in sigmas:
            eps = rng.standard_normal((K, fs, adim)).astype(np.float32)
            raw_cands[f"pert{s:g}"].append(a_star[None] + s * eps * astd[None, None])
        raw_cands["uniform"].append(rng.uniform(amin, amax, (K, fs, adim)).astype(np.float32))
    raw_cands = {ty: np.stack(v) for ty, v in raw_cands.items()}   # (n_anchors, n_c, fs, adim)

    results = {}
    raw_dump = {}
    for ad in adapters:
        zs = lambda x: (torch.from_numpy(np.asarray(x, np.float32)) - torch.from_numpy(np.asarray(ad.amean, np.float32))) \
            / torch.from_numpy(np.asarray(ad.astd, np.float32))
        # frames needed: history, real next, goal
        rows = set()
        for (e, lo, L, t) in picks:
            for o in ad.hist_offsets:
                rows.add(t + o)
            rows.add(t + fs); rows.add(t + goff)
        rows = np.array(sorted(rows))
        z_of = {}
        for s in range(0, len(rows), args.batch):
            r = rows[s:s + args.batch]
            z = ad.encode(to_px(f["pixels"][r]))
            for rr, zz in zip(r, z):
                z_of[int(rr)] = zz.float()
        cost_goal = {ty: np.zeros((len(picks), raw_cands[ty].shape[1]), np.float32) for ty in types}
        cost_true = {ty: np.zeros_like(cost_goal[ty]) for ty in types}
        cost_now = np.zeros(len(picks), np.float32)
        cost_real = np.zeros(len(picks), np.float32)
        for i, (e, lo, L, t) in enumerate(picks):
            z_hist = torch.stack([z_of[t + o] for o in ad.hist_offsets])[None]      # (1, H, D)
            z_goal = z_of[t + goff]; z_true = z_of[t + fs]
            cost_now[i] = float((z_hist[0, -1] - z_goal).norm())
            cost_real[i] = float((z_true - z_goal).norm())
            # history action blocks (models that condition on one block per history frame)
            hist_blocks = [zs(block(t + o)) for o in ad.hist_offsets[:-1]]           # (fs, adim) each
            if isinstance(ad, JointFlowAdapter):
                ad.next_block_z = zs(block(t + fs))[None]
            for ty in types:
                c = torch.from_numpy(raw_cands[ty][i])                              # (n_c, fs, adim) raw
                cz = zs(c)
                n_c = cz.shape[0]
                if ad.n_blocks_needed > 1:
                    hb = torch.stack(hist_blocks)[None].expand(n_c, -1, -1, -1)      # (n_c, H-1, fs, adim)
                    blocks = torch.cat([hb, cz[:, None]], 1)
                else:
                    blocks = cz[:, None]
                zn = ad.predict(z_hist.expand(n_c, -1, -1), blocks)
                cost_goal[ty][i] = (zn - z_goal).norm(dim=1).numpy()
                cost_true[ty][i] = (zn - z_true).norm(dim=1).numpy()
        prog = float(np.mean(cost_now - cost_real))                                  # real one-step progress
        summ = {"n_anchors": len(picks), "k": K, "cost_now_mean": float(cost_now.mean()),
                "cost_real_next_mean": float(cost_real.mean()), "real_progress_mean": prog,
                "expert_pred_err_mean": float(cost_true["expert"].mean()),
                "expert_pred_cost_mean": float(cost_goal["expert"].mean()), "types": {}}
        ce_g, ce_t = cost_goal["expert"], cost_true["expert"]                        # (n, 1)
        for ty in types:
            if ty == "expert":
                continue
            gap_g = cost_goal[ty] - ce_g; gap_t = cost_true[ty] - ce_t
            summ["types"][ty] = {
                "goal_gap_mean": float(gap_g.mean()), "goal_gap_per_progress": float(gap_g.mean() / (prog + 1e-8)),
                "goal_acc": float((gap_g > 0).mean()),
                "truth_gap_mean": float(gap_t.mean()), "truth_gap_per_err": float(gap_t.mean() / (ce_t.mean() + 1e-8)),
                "truth_acc": float((gap_t > 0).mean()),
            }
        results[ad.name] = summ
        print(f"[wm-discrim] {ad.name}: now={summ['cost_now_mean']:.3f} real_next={summ['cost_real_next_mean']:.3f} "
              f"progress={prog:.3f} expert_pred_err={summ['expert_pred_err_mean']:.3f} "
              f"expert_pred_cost={summ['expert_pred_cost_mean']:.3f}", flush=True)
        for ty, r in summ["types"].items():
            print(f"[wm-discrim] {ad.name} {ty:>10}: goal_acc={r['goal_acc']:.3f} goal_gap={r['goal_gap_mean']:+.4f} "
                  f"({r['goal_gap_per_progress']:+.2f}x progress) | truth_acc={r['truth_acc']:.3f} "
                  f"truth_gap={r['truth_gap_mean']:+.4f} ({r['truth_gap_per_err']:+.2f}x err)", flush=True)
        if args.dump_raw:
            raw_dump[ad.name] = {"cost_goal": cost_goal, "cost_true": cost_true,
                                 "cost_now": cost_now, "cost_real": cost_real}
    json.dump(results, open(args.out, "w"), indent=1)
    if args.dump_raw:
        flat = {}
        for name, d in raw_dump.items():
            tag = name.replace(":", "_")
            flat[f"{tag}__cost_now"] = d["cost_now"]; flat[f"{tag}__cost_real"] = d["cost_real"]
            for ty in types:
                flat[f"{tag}__goal__{ty}"] = d["cost_goal"][ty]; flat[f"{tag}__true__{ty}"] = d["cost_true"][ty]
        np.savez(args.dump_raw, **flat)
    print(f"[wm-discrim] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
