"""
Linear probes on the latents of LeWM and LeWAM physical state (proprio, object), action (IDM),
and effective rank.

NOTE: this script is largely experimental and likely requires further edits to work on more
datasets. The easiest way to add a dataset is updating `STATE_TARGET_OVERRIDES`

Usage:
  python scripts/probe_latents.py --dataset_path </path/to/.h5> [--lewam_runs <run ...>] [--lewm_dirs <dir ...>] \
      [--frameskip 5] [--batch_size 256] [--n_train 4000] [--n_test 1000] [--seed 0] --output_path </path/to/.json>
"""
import argparse
import json
import h5py
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

from lewam.eval.loaders import load_lewam, load_lewm
from lewam.train.datasets import read_episode_meta
from lewam.train.utils import IMAGENET_MEAN, IMAGENET_STD

# dataset_name: {state_type: [(target_name, column_name, column slice)]}
STATE_TARGET_OVERRIDES = {
    "pusht": {  # [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]
        "proprio": [
            ("agent_pos", "state", slice(0, 2)),
            ("agent_vel", "state", slice(5, 7))
        ],
        "object": [
            ("block_pos", "state", slice(2, 4)),
            ("block_angle", "state", slice(4, 5))
        ],
    },
}


def is_angle(name: str) -> bool:
    return name.lower().endswith(("yaw", "angle", "theta"))


def is_velocity(name: str) -> bool:
    return "_vel" in name.lower()


def get_state_targets(
    h5_file: h5py.File, dataset: str, rows: np.ndarray, frameskip: int
) -> dict[tuple[str, str], torch.Tensor]:
    """
    Read the probe targets of a dataset at `rows`, by state type (proprio, object)

    Args:
        h5_file (h5py.File): the dataset
        dataset (str): dataset name, the key into STATE_TARGET_OVERRIDES. Without an entry, `proprio_*` columns
            are proprio targets and `privileged_*` columns without "target" in the name are object targets
        rows (np.ndarray): h5 rows to read
        frameskip (int): velocity targets are read at `rows + frameskip`, the second frame of the pair they are
            probed from

    Returns:
        targets (dict): (state_type, target_name) -> (len(rows), d) tensor, with state_type "proprio" or "object"
            (e.g., ("proprio", "agent_pos")); angle targets are sin/cos encoded, so d doubles
    """
    if dataset in STATE_TARGET_OVERRIDES:
        state_meta = STATE_TARGET_OVERRIDES[dataset]
    else:
        state_meta = {"proprio": [], "object": []}
        for col in h5_file.keys():
            lowered = col.lower()
            if lowered.startswith("proprio"):
                state_meta["proprio"].append((col, col, slice(None)))
            elif lowered.startswith("privileged") and "target" not in lowered:
                state_meta["object"].append((col, col, slice(None)))

        if not state_meta["proprio"] and not state_meta["object"]:
            raise SystemExit(
                f"[PROBE] no proprio_*/privileged_* columns found and no override for '{dataset}'. "
                f"Add a STATE_TARGET_OVERRIDES entry. Columns present: {list(h5_file.keys())}"
            )

    targets = {}
    for state_type, items in state_meta.items():
        for tgt_name, col, col_slice in items:
            target_rows = rows + frameskip if is_velocity(tgt_name) else rows
            values = np.asarray(h5_file[col][:])[target_rows].reshape(len(rows), -1)[:, col_slice]
            target = torch.from_numpy(np.ascontiguousarray(values)).float()
            if is_angle(tgt_name):  # sin/cos so wrap-around isn't penalised
                target = torch.cat([target.sin(), target.cos()], dim=1)

            targets[(state_type, tgt_name)] = target
    return targets


def encode_frames(
    model: nn.Module, h5_file: h5py.File, rows: np.ndarray, device: str, batch_size: int = 256
) -> torch.Tensor:
    """Encode the frames at `rows`, reading each unique row once. (N, V, H, W, 3)"""
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    latents = []
    with torch.no_grad():
        for start in range(0, len(unique_rows), batch_size):
            frames = np.asarray(h5_file["pixels"][unique_rows[start:start + batch_size]])
            if frames.ndim == 5:   # multi-view -> primary view
                frames = frames[:, 0]

            pixels = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
            pixels = pixels.to(device)
            embeds = model.encode(pixels)

            latents.append(embeds.float().cpu())

    return torch.cat(latents)[inverse]


def ridge_r2(
    X_train: torch.Tensor, Y_train: torch.Tensor, X_test: torch.Tensor, Y_test: torch.Tensor, w_reg: float = 1e-3
) -> float:
    """Test R^2 of closed-form ridge with z-scored targets, averaged over target columns"""
    X_train = torch.cat([X_train, torch.ones(len(X_train), 1)], 1)
    X_test = torch.cat([X_test, torch.ones(len(X_test), 1)], 1)
    d = X_train.shape[1]

    mean, std = Y_train.mean(0), Y_train.std(0).clamp_min(1e-8)
    Y_train_norm = (Y_train - mean) / std
    Y_test_norm = (Y_test - mean) / std

    A = X_train.T @ X_train
    lam = w_reg * torch.trace(A) / d
    W = torch.linalg.solve(A + lam * torch.eye(d), X_train.T @ Y_train_norm)

    # NOTE: for low std, R^2 is not reliable; we noticed this in some variables e.g., block angle in Cube
    ss_res = ((Y_test_norm - X_test @ W) ** 2).mean(0)
    ss_tot = (Y_test_norm).var(0, correction=0).clamp_min(1e-8)

    # TODO: return raw, non-avg scores as option?
    return float((1 - ss_res / ss_tot).mean())


def effective_rank(z: torch.Tensor) -> float:
    """Exponential of the spectral entropy of the latent covariance"""
    z_center = z - z.mean(0)
    eigenvalues = torch.linalg.eigvalsh((z_center.T @ z_center) / (len(z_center) - 1)).clamp_min(0)
    spectrum = eigenvalues / (eigenvalues.sum() + 1e-12)
    entropy = -(spectrum * torch.log(spectrum + 1e-20)).sum()
    return float(torch.exp(entropy))


def state_type_mean(per_target: dict[str, dict], state_type: str) -> float | None:
    values = [target["r2"] for target in per_target.values() if target["state_type"] == state_type]
    return round(np.mean(values), 4) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="path to h5 dataset to probe")
    parser.add_argument("--lewam_runs", nargs="+", default=[], help="LeWAM run names under $STABLEWM_HOME/checkpoints")
    parser.add_argument("--lewm_dirs", nargs="+", default=[], help="LeWM checkpoint folders, each with `config.json` and a `.pt`")
    parser.add_argument("--frameskip", type=int, default=5, help="env steps between the probed frames")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size to use when encoding frames")
    parser.add_argument("--n_train", type=int, default=4000, help="frames to fit each probe on")
    parser.add_argument("--n_test", type=int, default=1000, help="frames to score each probe on")
    parser.add_argument("--seed", type=int, default=0, help="seed of the frame draw and the train/test split")
    parser.add_argument("--output_path", required=True, help="path of json file to write the results to")
    args = parser.parse_args()

    if not (args.lewam_runs or args.lewm_dirs):
        parser.error("pass --lewam_runs and/or --lewm_dirs")

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Path(args.dataset_path).stem
    h5_file = h5py.File(args.dataset_path, "r")
    ep_offsets, ep_len = read_episode_meta(args.dataset_path)
    fs = args.frameskip
    n_samples = args.n_train + args.n_test

    # load models
    models = []
    for run_name in args.lewam_runs:
        model, _ = load_lewam(run_name)
        models.append((
            f"lewam:{run_name}",
            model.to(device).eval()
        ))
    for ckpt_dir in args.lewm_dirs:
        model, _ = load_lewm(ckpt_dir, encoder_only=True)
        models.append((
            f"lewm:{Path(ckpt_dir).name}",
            model.to(device).eval()
        ))

    # sample at row t needs z[t], z[t+fs] in the same episode
    candidates = rng.permutation(int(ep_offsets[-1] + ep_len[-1]))
    episode = np.searchsorted(ep_offsets, candidates, side="right") - 1
    in_episode = candidates + fs < ep_offsets[episode] + ep_len[episode]
    samples = np.sort(candidates[in_episode][:n_samples])
    assert len(samples) == n_samples, f"only {len(samples)} valid samples found (<{n_samples}); lower --n_train, --n_test"

    is_train = np.zeros(n_samples, bool)
    is_train[rng.permutation(n_samples)[:args.n_train]] = True

    state_targets = get_state_targets(h5_file, dataset, samples, fs)

    action = np.asarray(h5_file["action"][:]).astype(np.float32)
    action_targets = torch.from_numpy(
        np.stack([action[t:t+fs].reshape(-1) for t in samples])
    ).float()

    # compute results
    results = {}
    for model_name, model in models:
        z_t, z_next = encode_frames(
            model, h5_file, np.concatenate([samples, samples + fs]), device, args.batch_size
        ).chunk(2)
        z_pair = torch.cat([z_t, z_next], 1)

        # probe state
        per_target = {}
        for (state_type, name), Y in state_targets.items():
            X = z_pair if is_velocity(name) else z_t
            r2 = ridge_r2(X[is_train], Y[is_train], X[~is_train], Y[~is_train])
            per_target[name] = {"state_type": state_type, "r2": round(r2, 4)}

        proprio_mean, object_mean = state_type_mean(per_target, "proprio"), state_type_mean(per_target, "object")
        overall = round(np.mean([target["r2"] for target in per_target.values()]), 4)

        # probe action
        action_idm_r2 = ridge_r2(z_pair[is_train], action_targets[is_train], z_pair[~is_train], action_targets[~is_train])
        action_pred_r2 = ridge_r2(z_t[is_train], action_targets[is_train], z_t[~is_train], action_targets[~is_train])

        # record results for model
        result = {
            "effective_rank": round(effective_rank(z_t), 3),
            "action_idm_r2": round(action_idm_r2, 4),
            "action_pred_r2": round(action_pred_r2, 4),
            "state_proprio_mean": proprio_mean,
            "state_object_mean": object_mean,
            "state_overall_mean": overall,
            "targets": per_target,
        }
        results[model_name] = result

        print(f"[{dataset}] {model_name}: eff_rank={result['effective_rank']} idm={result['action_idm_r2']} "
              f"act_single={result['action_pred_r2']} | proprio_mean={proprio_mean} "
              f"object_mean={object_mean} overall={overall}", flush=True)

        for name, target in per_target.items():
            print(f"{target['state_type']:8s} {name:26s} r2={target['r2']:.3f}"
                  f"{' (pair)' if is_velocity(name) else ''}", flush=True)

    # save results
    Path(args.output_path).write_text(
        json.dumps({
            "dataset": dataset, "fs": fs, "n_train": args.n_train,
            "n_test": args.n_test, "results": results
        }, indent=1)
    )
    print(f"[PROBE] wrote probing results to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
