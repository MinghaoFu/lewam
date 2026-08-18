"""Convert a worldforge flat h5 (swm layout) into a robomimic-format hdf5 that
diffusion_policy's RobomimicReplayImageDataset can read.

Input columns (flat, row-per-frame): pixels [N,H,W,3] uint8, action [N,adim],
proprio [N,9] = eef_pos(3) + eef_quat(4) + gripper_qpos(2) (quat layout verified:
|q|=1 at dims 3:7), ep_len/ep_offset [n_eps].

Output: data/demo_<i>/{obs/{<image_key>, robot0_eef_pos, robot0_eef_quat,
robot0_gripper_qpos}, actions}, attrs num_samples per demo, and data.attrs
env_args copied from --env_meta_h5 (a robomimic file whose data attrs carry the
env metadata; the 6KB image_384_v15.hdf5 stub works).
"""
import argparse, json
import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", required=True, help="flat worldforge h5")
p.add_argument("--out", required=True, help="robomimic-format hdf5 to write")
p.add_argument("--env_meta_h5", required=True, help="robomimic h5 carrying data.attrs['env_args']")
p.add_argument("--image_key", default="sideview_image")
p.add_argument("--max_eps", type=int, default=0, help="0 = all episodes")
args = p.parse_args()

with h5py.File(args.env_meta_h5, "r") as m:
    env_args = m["data"].attrs["env_args"]
    json.loads(env_args)  # must be valid JSON

src = h5py.File(args.src, "r")
ep_offset = np.asarray(src["ep_offset"][:]).reshape(-1)
ep_len = np.asarray(src["ep_len"][:]).reshape(-1)
n_eps = len(ep_len) if not args.max_eps else min(args.max_eps, len(ep_len))
total = 0

with h5py.File(args.out, "w") as out:
    data = out.create_group("data")
    data.attrs["env_args"] = env_args
    for i in range(n_eps):
        o, L = int(ep_offset[i]), int(ep_len[i])
        g = data.create_group(f"demo_{i}")
        g.attrs["num_samples"] = L
        obs = g.create_group("obs")
        obs.create_dataset(args.image_key, data=src["pixels"][o:o + L],
                           dtype=np.uint8, chunks=(1, *src["pixels"].shape[1:]),
                           compression="gzip", compression_opts=1)
        prop = src["proprio"][o:o + L]
        obs.create_dataset("robot0_eef_pos", data=prop[:, 0:3])
        obs.create_dataset("robot0_eef_quat", data=prop[:, 3:7])
        obs.create_dataset("robot0_gripper_qpos", data=prop[:, 7:9])
        g.create_dataset("actions", data=src["action"][o:o + L])
        g.create_dataset("rewards", data=np.zeros(L, np.float64))
        d = np.zeros(L, np.float64); d[-1] = 1.0
        g.create_dataset("dones", data=d)
        total += L
        if (i + 1) % 50 == 0:
            print(f"[convert] {i+1}/{n_eps} eps, {total} frames", flush=True)
    data.attrs["total"] = total
print(f"[convert] DONE eps={n_eps} frames={total} -> {args.out}", flush=True)

# verify: byte-identical images + fields for 2 random episodes
rng = np.random.default_rng(0)
with h5py.File(args.out, "r") as out:
    for i in rng.choice(n_eps, size=min(2, n_eps), replace=False):
        o, L = int(ep_offset[i]), int(ep_len[i])
        g = out[f"data/demo_{i}"]
        assert g.attrs["num_samples"] == L
        assert np.array_equal(g[f"obs/{args.image_key}"][:], src["pixels"][o:o + L])
        assert np.array_equal(g["actions"][:], src["action"][o:o + L])
        assert np.array_equal(g["obs/robot0_eef_quat"][:], src["proprio"][o:o + L, 3:7])
print("[convert] VERIFY_OK", flush=True)
