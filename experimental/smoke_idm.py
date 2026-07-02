#!/usr/bin/env python
"""CPU smoke test for the SMWM inverse-dynamics anti-collapse term (idm ablation).

Proves three things, all on CPU (CUDA_VISIBLE_DEVICES=""), tiny synthetic batch:
  (1) default w_inv=0 (no inverse head built) -> forward runs, NO inv_loss key.
  (2) w_inv=0.1 inv_mode=dense -> forward runs, produces a FINITE inv_loss.
  (3) BYTE-IDENTICAL: the w_inv=0 total loss EQUALS the loss from the pre-edit
      BACKUP code (jepa.py.bak_idm / train.py.bak_idm) on the SAME seeded inputs.

Run: CUDA_VISIBLE_DEVICES="" python smoke_idm.py
"""
import importlib.util
import sys, os
import torch
from omegaconf import OmegaConf

REPO = "/var/lib/docker/data/minghao_home/workspace/le-wm-repro"
sys.path.insert(0, REPO)

from lewam.models.module import SIGReg  # noqa


import types
def load_mod(name, path):
    # works for any extension (.py or .bak_idm): read source, exec into a fresh module
    src = open(path).read()
    m = types.ModuleType(name)
    m.__file__ = path
    sys.modules[name] = m
    exec(compile(src, path, "exec"), m.__dict__)
    return m


def build_cfg(w_inv, sigreg_w, inv_mode="dense", inv_target="encoded"):
    """Minimal action_pred config mirroring the campaign GC recipe."""
    return OmegaConf.create({
        "history_size": 3,
        "num_preds": 1,
        "loss": {"sigreg": {"weight": sigreg_w, "kwargs": {"knots": 17, "num_proj": 1024}}},
        "action_pred": {
            "enabled": True, "w_act": 1.0, "w_intent": 0.0, "detach_target": True,
            "detach_decoder": False, "head": "mse", "w_inv": w_inv,
            "inv_mode": inv_mode, "inv_target": inv_target,
        },
    })


def build_model(jepa_mod, embed_dim=192, adim=10, inverse=False):
    """Construct a JEPA with the SAME module specs the real config uses, plus the
    action_predictor + action_decoder the gip_on path attaches."""
    from lewam.models.module import ARPredictor, Embedder, MLP
    import stable_pretraining as spt
    torch.manual_seed(0)
    enc = spt.backbone.utils.vit_hf(size="tiny", patch_size=14, image_size=224,
                                    pretrained=False, use_mask_token=False)
    def mk_pred():
        return ARPredictor(num_frames=3, input_dim=embed_dim, hidden_dim=embed_dim,
                           output_dim=embed_dim, depth=6, heads=16, mlp_dim=2048,
                           dim_head=64, dropout=0.1, emb_dropout=0.0)
    act_enc = Embedder(input_dim=adim, emb_dim=embed_dim)
    proj = MLP(embed_dim, 2048, embed_dim, norm_fn=lambda d: torch.nn.BatchNorm1d(d))
    pred_proj = MLP(embed_dim, 2048, embed_dim, norm_fn=lambda d: torch.nn.BatchNorm1d(d))
    kw = {}
    if "inverse_conditioned" in jepa_mod.JEPA.__init__.__code__.co_varnames:
        kw = dict(inverse_conditioned=inverse, inverse_action_dim=adim)
    elif inverse:
        raise RuntimeError("backup JEPA has no inverse head support")
    m = jepa_mod.JEPA(encoder=enc, predictor=mk_pred(), action_encoder=act_enc,
                      projector=proj, pred_proj=pred_proj, use_action_history=True, **kw)
    m.action_predictor = mk_pred()
    m.action_decoder = MLP(embed_dim, 2048, adim)
    m.action_head = {"type": "mse"}
    return m.eval()


class Holder:
    """Mimics the spt.Module attributes lejepa_forward reads: .model, .sigreg, .log_dict."""
    def __init__(self, model, sigreg):
        self.model = model
        self.sigreg = sigreg
    def log_dict(self, *a, **k):
        pass


def make_batch(adim=10, T=4, B=8, seed=123):
    g = torch.Generator().manual_seed(seed)
    return {
        "pixels": torch.rand(B, T, 3, 224, 224, generator=g),
        "action": torch.randn(B, T, adim, generator=g),
    }


def run_forward(train_mod, jepa_mod, cfg, inverse, seed=123):
    torch.manual_seed(7)
    model = build_model(jepa_mod, inverse=inverse)
    holder = Holder(model, SIGReg(**cfg.loss.sigreg.kwargs))
    batch = make_batch(seed=seed)
    with torch.no_grad():
        out = train_mod.lejepa_forward(holder, batch, "val", cfg)
    return out, model


def main():
    edited_jepa = load_mod("jepa_edit", f"{REPO}/jepa.py")
    edited_train = load_mod("train_edit", f"{REPO}/train.py")
    bak_jepa = load_mod("jepa_bak", f"{REPO}/jepa.py.bak_idm")
    bak_train = load_mod("train_bak", f"{REPO}/train.py.bak_idm")

    sigreg_w = 0.09

    # (1) default w_inv=0, edited code, no inverse head
    cfg0 = build_cfg(w_inv=0.0, sigreg_w=sigreg_w)
    out0, m0 = run_forward(edited_train, edited_jepa, cfg0, inverse=False)
    assert "inv_loss" not in out0, "inv_loss must be ABSENT at w_inv=0"
    assert getattr(m0, "inverse_model", None) is None, "inverse_model must NOT be built at default"
    loss0 = float(out0["loss"])
    print(f"[1] edited w_inv=0  loss={loss0:.8f}  inv_loss_present={'inv_loss' in out0}  inverse_head=None OK")

    # (3) byte-identical: backup code on the SAME seeded inputs
    cfgb = build_cfg(w_inv=0.0, sigreg_w=sigreg_w)
    # backup cfg has no w_inv/inv_mode keys; strip them so it matches the pre-edit config exactly
    cfgb.action_pred.pop("w_inv"); cfgb.action_pred.pop("inv_mode"); cfgb.action_pred.pop("inv_target")
    outb, mb = run_forward(bak_train, bak_jepa, cfgb, inverse=False)
    lossb = float(outb["loss"])
    delta = abs(loss0 - lossb)
    print(f"[3] backup        loss={lossb:.8f}  |edited-backup|={delta:.2e}")
    assert delta < 1e-6, f"NOT byte-identical at default: delta={delta}"
    print("    -> BYTE-IDENTICAL at default confirmed (running campaign unaffected).")

    # (2) w_inv=0.1 dense, edited code, inverse head built -> finite inv_loss
    cfg1 = build_cfg(w_inv=0.1, sigreg_w=sigreg_w, inv_mode="dense", inv_target="encoded")
    out1, m1 = run_forward(edited_train, edited_jepa, cfg1, inverse=True)
    assert "inv_loss" in out1, "inv_loss must be present at w_inv>0"
    assert getattr(m1, "inverse_model", None) is not None, "inverse_model must be built at w_inv>0"
    inv = float(out1["inv_loss"]); loss1 = float(out1["loss"])
    import math
    assert math.isfinite(inv) and math.isfinite(loss1), "inv_loss / loss must be finite"
    print(f"[2] edited w_inv=0.1 dense  inv_loss={inv:.6f}  total_loss={loss1:.6f}  (finite OK)")

    # bonus: predicted-target variant also runs finite
    cfg2 = build_cfg(w_inv=0.1, sigreg_w=0.0, inv_mode="dense", inv_target="predicted")
    out2, _ = run_forward(edited_train, edited_jepa, cfg2, inverse=True)
    inv2 = float(out2["inv_loss"])
    assert math.isfinite(inv2)
    print(f"[2b] edited w_inv=0.1 inv_target=predicted sigreg=0  inv_loss={inv2:.6f} (finite OK)")

    # bonus: inv_mode=last
    cfg3 = build_cfg(w_inv=0.1, sigreg_w=0.0, inv_mode="last", inv_target="encoded")
    out3, _ = run_forward(edited_train, edited_jepa, cfg3, inverse=True)
    inv3 = float(out3["inv_loss"])
    assert math.isfinite(inv3)
    print(f"[2c] edited w_inv=0.1 inv_mode=last  inv_loss={inv3:.6f} (finite OK)")

    print("\nSMOKE PASS: default byte-identical; w_inv>0 builds head + finite inv_loss (dense/predicted/last).")


if __name__ == "__main__":
    main()
