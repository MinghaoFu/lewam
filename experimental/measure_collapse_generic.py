#!/usr/bin/env python
"""Generic latent-collapse readout for ANY checkpoint run (idm ablation).

Loads a run via gip.load_gip_model, builds a fixed val batch from the run's OWN
config (config.yaml -> cfg.data.dataset), runs model.encode, and reports:
  z_std    = emb.reshape(-1,D).std(dim=0).mean()      (-> 0 means latent collapse)
  act_std  = act_emb per-dim std
  erank    = effective rank (entropy of normalized singular values) of the latent
  inv_loss = (if the run has an inverse_model) the dense inverse MSE on this batch

Usage: measure_collapse_generic.py <run_name> [--epoch N] [--bs 256]
Prints one PARSE line:  PARSE run=<run> z_std=.. act_std=.. erank=.. inv_loss=.. D=.. n=..
"""
import os, sys, argparse, math

from pathlib import Path
import torch
from omegaconf import OmegaConf

import stable_worldmodel.data.formats.hdf5  # self-registers HDF5
import stable_pretraining as spt
import stable_worldmodel as swm
import lewam.models.gip as gip
from lewam.utils import get_column_normalizer, get_img_preprocessor

CKROOT = Path(os.environ.get("STABLEWM_HOME", "/mnt/minghao_data/.stable-wm")) / "checkpoints"


def build_val_batch(run_name, bs=256):
    # prefer full_config.yaml; fall back to config.yaml
    rd = CKROOT / run_name
    cfgp = rd / "full_config.yaml"
    if not cfgp.exists():
        cfgp = rd / "config.yaml"
    cfg = OmegaConf.load(cfgp)
    dcfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dcfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    ds = swm.data.load_dataset(name, transform=None, cache_dir=cache_dir, **dcfg)
    tfs = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    for col in cfg.data.dataset.keys_to_load:
        if str(col).startswith("pixels"):
            continue
        tfs.append(get_column_normalizer(ds, col, col))
    ds.transform = spt.data.transforms.Compose(*tfs)
    rnd = torch.Generator().manual_seed(int(cfg.seed))
    tr, va = spt.data.random_split(ds, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd)
    loader = torch.utils.data.DataLoader(va, batch_size=bs, shuffle=False, drop_last=False)
    return next(iter(loader)), cfg


@torch.no_grad()
def measure(run_name, epoch, bs, device):
    batch, cfg = build_val_batch(run_name, bs=bs)
    model, adim = gip.load_gip_model(run_name, epoch=epoch)
    model = model.to(device).eval()
    info = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    info["action"] = torch.nan_to_num(info["action"], 0.0)
    out = model.encode(info)
    emb = out["emb"]
    D = emb.shape[-1]
    flat = emb.detach().reshape(-1, D).float()
    z_std = flat.std(dim=0).mean().item()
    act_std = None
    if out.get("act_emb", None) is not None:
        ae = out["act_emb"]
        act_std = ae.detach().reshape(-1, ae.shape[-1]).std(dim=0).mean().item()
    c = flat - flat.mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(c)
        p = (s / s.sum()).clamp_min(1e-12)
        erank = torch.exp(-(p * p.log()).sum()).item()
    except Exception:
        erank = float("nan")
    # inverse-head readout (only if the trained run carries one)
    inv = None
    if getattr(model, "inverse_model", None) is not None:
        z_t, z_tp1 = emb[:, :-1], emb[:, 1:]
        a_tau = info["action"][:, : z_t.size(1)]
        a_hat = model.predict_inverse(z_t, z_tp1)
        inv = (a_hat - a_tau).pow(2).mean().item()
    return dict(z_std=z_std, act_std=act_std, erank=erank, inv=inv, D=D, n=flat.shape[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        r = measure(args.run, args.epoch, args.bs, device)
        astd = "%.5f" % r["act_std"] if r["act_std"] is not None else "NA"
        inv = "%.5f" % r["inv"] if r["inv"] is not None else "NA"
        print("PARSE run=%s z_std=%.5f act_std=%s erank=%.2f inv_loss=%s D=%d n=%d"
              % (args.run, r["z_std"], astd, r["erank"], inv, r["D"], r["n"]))
    except Exception as e:
        import traceback; traceback.print_exc()
        print("PARSE run=%s z_std=NA act_std=NA erank=NA inv_loss=NA D=0 n=0 ERROR=%s"
              % (args.run, type(e).__name__))
