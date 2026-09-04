"""PCGrad (Yu et al. 2020) over the model's full parameter vector, for the jointflow losses.

Per step each task loss gets its own backward pass; gradients are flattened over ALL
parameters (a parameter a task does not touch contributes zeros, so the projection acts only
where tasks share weights: encoder + MoT trunk). mode="sym" is the original symmetric
surgery -- every task's gradient is projected away from each conflicting other's, in a
random task order per step, then summed. mode="protect_p" leaves the policy gradient
untouched and projects only the other tasks away from it (owner's asymmetric variant).
Stats per step: which pairs conflicted and their cosines, so the epoch log can show when
the surgery is active (the early-training hypothesis predicts high conflict rate early).
"""
import random

import torch


class PCGrad:
    def __init__(self, params, mode="sym", protect="P", match_params=None):
        assert mode in ("sym", "protect_p", "match_s"), mode
        self.params = [p for p in params if p.requires_grad]
        self.mode = mode
        self.protect = protect
        # mode="match_s" (owner 2026-09-04, gradient-MAGNITUDE hypothesis): no projection; the
        # SIGReg gradient is rescaled every step so its norm on the shared encoder equals the
        # policy gradient's norm there (probes: ||g_S|| ~ 100x ||g_P|| on the reacher encoder).
        # match_params = the encoder parameters the norms are measured on.
        ids = {id(p) for p in (match_params or [])}
        self.match_mask = torch.cat([torch.full((p.numel(),), id(p) in ids, dtype=torch.bool)
                                     for p in self.params]) if ids else None
        self.reset_stats()

    def reset_stats(self):
        self.n_steps = 0
        self.n_conflict = {}      # pair -> steps with negative dot
        self.cos_sum = {}         # pair -> summed cosine (pre-surgery)
        self.scale_sum = 0.0      # match_s: summed S rescale factor

    def _flat_grad(self, loss, retain):
        grads = torch.autograd.grad(loss, self.params, retain_graph=retain, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                          for g, p in zip(grads, self.params)])

    def backward(self, losses):
        """losses: ordered dict name -> scalar loss (graph shared). Writes the surgered sum
        into .grad of every parameter (replaces any existing .grad)."""
        names = list(losses.keys())
        flats = {}
        for i, k in enumerate(names):
            flats[k] = self._flat_grad(losses[k], retain=(i < len(names) - 1))
        self.n_steps += 1
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                ka, kb = names[a], names[b]
                ga, gb = flats[ka], flats[kb]
                dot = torch.dot(ga, gb)
                cos = (dot / (ga.norm() * gb.norm() + 1e-12)).item()
                key = f"{ka}{kb}"
                self.cos_sum[key] = self.cos_sum.get(key, 0.0) + cos
                if dot < 0:
                    self.n_conflict[key] = self.n_conflict.get(key, 0) + 1
        out = {k: flats[k].clone() for k in names}
        if self.mode == "match_s":
            if "S" in flats and "P" in flats:
                m = self.match_mask.to(flats["S"].device) if self.match_mask is not None else \
                    torch.ones_like(flats["S"], dtype=torch.bool)
                scale = (flats["P"][m].norm() / (flats["S"][m].norm() + 1e-12)).clamp(max=1.0).item()
                out["S"] = flats["S"] * scale
                self.scale_sum += scale
        elif self.mode == "sym":
            for k in names:
                others = [o for o in names if o != k]
                random.shuffle(others)
                for o in others:
                    g_o = flats[o]
                    dot = torch.dot(out[k], g_o)
                    if dot < 0:
                        out[k] = out[k] - (dot / (g_o.norm() ** 2 + 1e-12)) * g_o
        else:  # protect_p: only the non-protected tasks are projected, away from the protected one
            g_p = flats[self.protect]
            for k in names:
                if k == self.protect:
                    continue
                dot = torch.dot(out[k], g_p)
                if dot < 0:
                    out[k] = out[k] - (dot / (g_p.norm() ** 2 + 1e-12)) * g_p
        total = sum(out[k] for k in names)
        offset = 0
        for p in self.params:
            n = p.numel()
            p.grad = total[offset:offset + n].view_as(p).clone()
            offset += n

    def stats(self):
        n = max(self.n_steps, 1)
        return {f"conflict_rate_{k}": v / n for k, v in self.n_conflict.items()} | \
               {f"cos_{k}": v / n for k, v in self.cos_sum.items()} | \
               ({"s_scale": self.scale_sum / n} if self.mode == "match_s" else {})
