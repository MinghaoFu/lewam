"""Dynamics-informativeness metrics (owner spec 2026-08-27) from JointFlowGCPolicy dyn_log
dumps: (1) variance of REAL next latents vs variance of PREDICTED ones (collapsed
predictions = uninformative), (2) pred-vs-real gap relative to the real transition size,
for the policy's own chunks and for random executed chunks, (3) agreement between the
model's goal cost on the prediction and the true goal cost on the real latent.
"""
import argparse
import json

import numpy as np


def _stats(d, mask, key):
    pred = d[key][mask]
    ok = np.isfinite(pred).all(1)
    pred, zr, zp, zg = pred[ok], d["z_real"][mask][ok], d["z_prev"][mask][ok], d["z_goal"][mask][ok]
    if len(pred) == 0:
        return None
    trans = np.linalg.norm(zr - zp, axis=1)
    gap = np.linalg.norm(pred - zr, axis=1)
    null_gap = trans                                   # predicting "no change"
    c_pred = np.linalg.norm(pred - zg, axis=1)
    c_real = np.linalg.norm(zr - zg, axis=1)
    c_prev = np.linalg.norm(zp - zg, axis=1)           # persistence cost baseline
    var_real = float(zr.var(0).mean())
    var_pred = float(pred.var(0).mean())
    return dict(
        n=int(len(pred)),
        var_pred_over_real=round(var_pred / max(var_real, 1e-12), 4),
        gap_over_transition_median=round(float(np.median(gap / np.clip(trans, 1e-8, None))), 4),
        gap_over_transition_mean=round(float(np.mean(gap / np.clip(trans, 1e-8, None))), 4),
        gap_over_null_median=round(float(np.median(gap / np.clip(null_gap, 1e-8, None))), 4),
        cost_corr_pearson=round(float(np.corrcoef(c_pred, c_real)[0, 1]), 4),
        cost_corr_persistence=round(float(np.corrcoef(c_prev, c_real)[0, 1]), 4),
        cost_corr_spearman=round(float(np.corrcoef(np.argsort(np.argsort(c_pred)),
                                                   np.argsort(np.argsort(c_real)))[0, 1]), 4),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, nargs="+")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    results = {}
    for path in args.npz:
        d = np.load(path)
        is_r = d["is_random"].astype(bool)
        res = {
            "on_policy_inpaint": _stats(d, ~is_r, "pred_inpaint"),
            "on_policy_joint": _stats(d, ~is_r, "pred_joint"),
            "random_inpaint": _stats(d, is_r, "pred_inpaint") if is_r.any() else None,
        }
        results[path] = res
        for k, v in res.items():
            print(f"[dyn-inf] {path} {k}: {v}", flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
