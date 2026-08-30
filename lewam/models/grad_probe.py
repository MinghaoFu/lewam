"""Encoder gradient-geometry probe (owner design 2026-08-30).

Every `every` training steps, on the SAME batch, measure the gradients w.r.t. the shared encoder
of the four objectives: P (policy flow loss), Din (dynamics loss, target detached: the input
path), Dtar (= D_full - Din: the online-target path, the JEPA channel), S (SIGReg, computed as a
counterfactual even on no-reg arms). Per encoder layer bucket (stem, layer1-4, head) and total:

  ||g_i||, all pairwise cos(g_i, g_j), the coherence ||EMA(g_i)|| / EMA(||g_i||),
  cos between the EMA'd gradient vectors (the primary geometry estimate; per-batch cosines in
  ~1e7 dims are noise-dominated), and the same for Adam-preconditioned gradients
  g / (sqrt(v_hat) + eps) (secondary: what the optimizer does vs what the objectives want).

Rows are appended to a jsonl. The caller guarantees the loss tensors share one graph rooted at
the encoder outputs and were drawn under a forked, seeded RNG (so Din uses the SAME tau/noise as
D_full and the SIGReg projections are shared within the measurement)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

PAIRS = [("P", "S"), ("P", "Dtar"), ("P", "Din"), ("Dtar", "S"), ("Din", "S"), ("Din", "Dtar")]


def _bucket_of(name):
    if name.startswith(("trunk.0", "trunk.1", "trunk.2", "trunk.3")):
        return "stem"
    for i, layer in ((4, "layer1"), (5, "layer2"), (6, "layer3"), (7, "layer4")):
        if name.startswith(f"trunk.{i}"):
            return layer
    return "head"


def _cos(a, b):
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


class GradProbe:
    def __init__(self, encoder, out_path, every=250, ema_steps=64, precond=True):
        self.every = int(every)
        self.alpha = 2.0 / (float(ema_steps) + 1.0)
        self.precond = bool(precond)
        self.out_path = Path(out_path)
        self.optimizer = None                       # set by the trainer after the optimizer exists
        named = [(n, p) for n, p in encoder.named_parameters() if p.requires_grad]
        self.params = [p for _, p in named]
        self.slices, self.buckets, off = [], {}, 0
        for n, p in named:
            k = p.numel()
            self.slices.append(slice(off, off + k))
            self.buckets.setdefault(_bucket_of(n), []).append(len(self.slices) - 1)
            off += k
        self.dim = off
        self.bucket_names = sorted(self.buckets) + ["total"]
        self.ema_g = {}                             # (loss, kind) -> flat cpu tensor
        self.ema_n = {}                             # (loss, kind, bucket) -> float
        print(f"[grad-probe] {len(self.params)} encoder tensors, {off / 1e6:.2f}M params, "
              f"buckets {sorted((b, len(i)) for b, i in self.buckets.items())}, every {self.every}", flush=True)

    def _flat(self, loss):
        gs = torch.autograd.grad(loss, self.params, retain_graph=True, allow_unused=True)
        out = torch.zeros(self.dim, device=self.params[0].device)
        for g, sl in zip(gs, self.slices):
            if g is not None:
                out[sl] = g.reshape(-1).float()
        return out

    def _precondition(self, flat):
        """g / (sqrt(v_hat) + eps) with the live Adam second moment (bias-corrected)."""
        out = flat.clone()
        for p, sl in zip(self.params, self.slices):
            st = self.optimizer.state.get(p) if self.optimizer is not None else None
            if st and "exp_avg_sq" in st and st.get("step", 0):
                step = st["step"].item() if torch.is_tensor(st["step"]) else float(st["step"])
                beta2 = 0.999
                for grp in self.optimizer.param_groups:
                    if any(p is q for q in grp["params"]):
                        beta2 = grp["betas"][1]
                        break
                v_hat = st["exp_avg_sq"].reshape(-1).float() / (1.0 - beta2 ** step)
                out[sl] = out[sl] / (v_hat.sqrt() + 1e-8)
        return out

    def _mask(self, flat, bucket):
        if bucket == "total":
            return flat
        m = torch.zeros_like(flat)
        for i in self.buckets[bucket]:
            m[self.slices[i]] = flat[self.slices[i]]
        return m

    @torch.no_grad()
    def _account(self, flats, kind, step, row):
        # update the EMAs first (EMA is linear, so the per-bucket EMA vector = the masked total EMA)
        cpu = {k: v.cpu() for k, v in flats.items()}
        for k, v in cpu.items():
            key = (k, kind)
            if key not in self.ema_g:
                self.ema_g[key] = v.clone()
            else:
                self.ema_g[key].mul_(1 - self.alpha).add_(v, alpha=self.alpha)
        for b in self.bucket_names:
            gb = {k: self._mask(v, b) for k, v in cpu.items()}
            eb = {k: self._mask(self.ema_g[(k, kind)], b) for k in cpu}
            for k, v in gb.items():
                nkey = (k, kind, b)
                n = float(v.norm())
                row[f"{kind}/{b}/norm_{k}"] = n
                self.ema_n[nkey] = (1 - self.alpha) * self.ema_n.get(nkey, n) + self.alpha * n
                row[f"{kind}/{b}/coh_{k}"] = float(eb[k].norm()) / max(self.ema_n[nkey], 1e-12)
            for a, c in PAIRS:
                row[f"{kind}/{b}/cos_{a}_{c}"] = _cos(gb[a], gb[c])
                row[f"{kind}/{b}/emacos_{a}_{c}"] = _cos(eb[a], eb[c])
        return row

    def measure(self, loss_P, loss_D, loss_Din, loss_S, step, scalars=None):
        flats = {"P": self._flat(loss_P), "Din": self._flat(loss_Din), "S": self._flat(loss_S)}
        g_full = self._flat(loss_D)
        flats["Dtar"] = g_full - flats["Din"]
        row = {"step": int(step)}
        if scalars:
            row.update({k: float(v) for k, v in scalars.items()})
        nP = float(flats["P"].norm())
        row["raw/total/norm_Dfull"] = float(g_full.norm())
        row["r_D"] = row["raw/total/norm_Dfull"] / max(nP, 1e-12)                       # lambda_D = 1
        row["r_S"] = float(row.get("lam_S", 0.0)) * float(flats["S"].norm()) / max(nP, 1e-12)
        self._account(flats, "raw", step, row)
        if self.precond and self.optimizer is not None:
            pre = {k: self._precondition(v) for k, v in flats.items()}
            self._account(pre, "pre", step, row)
        with self.out_path.open("a") as f:
            f.write(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}) + "\n")
        return row
