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
try:
    import hdf5plugin  # noqa: F401  registers external HDF5 filters (wf8 h5 pixels need it)
except ImportError:
    pass
import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", required=True, help="flat worldforge h5")
p.add_argument("--out", required=True, help="robomimic-format hdf5 to write")
p.add_argument("--env_meta_h5", default="",
               help="robomimic h5 carrying data.attrs['env_args']")
p.add_argument("--env_args_json", default="",
               help="verbatim env_args JSON instead of --env_meta_h5; inert under the dummy "
                    "env runner (DP's replay dataset reads shapes from the task yaml)")
p.add_argument("--image_key", default="sideview_image")
p.add_argument("--no_proprio", action="store_true",
               help="skip the robot0_* proprio datasets (noprop task yamls; also required "
                    "when the src has no flat 'proprio' column, e.g. cube)")
p.add_argument("--truncate_at_success", default="",
               help="bool column name: truncate each episode at its first True (drop the "
                    "post-success tail; cube's is ~69%% of every episode)")
p.add_argument("--max_eps", type=int, default=0, help="0 = all episodes")
args = p.parse_args()

assert bool(args.env_meta_h5) != bool(args.env_args_json), \
    "exactly one of --env_meta_h5 / --env_args_json"
if args.env_meta_h5:
    with h5py.File(args.env_meta_h5, "r") as m:
        env_args = m["data"].attrs["env_args"]
        json.loads(env_args)  # must be valid JSON
else:
    json.loads(args.env_args_json)
    env_args = args.env_args_json

src = h5py.File(args.src, "r")
ep_offset = np.asarray(src["ep_offset"][:]).reshape(-1)
ep_len = np.asarray(src["ep_len"][:]).reshape(-1)
n_eps = len(ep_len) if not args.max_eps else min(args.max_eps, len(ep_len))
write_prop = (not args.no_proprio) and ("proprio" in src)


def eff_len(i):
    """Episode length after optional first-success truncation (>= 2 frames kept)."""
    o, L = int(ep_offset[i]), int(ep_len[i])
    if not args.truncate_at_success:
        return L
    s = np.asarray(src[args.truncate_at_success][o:o + L]).reshape(-1).astype(bool)
    return max(int(np.argmax(s)) + 1, 2) if s.any() else L


total = 0
with h5py.File(args.out, "w") as out:
    data = out.create_group("data")
    data.attrs["env_args"] = env_args
    for i in range(n_eps):
        o, L = int(ep_offset[i]), eff_len(i)
        g = data.create_group(f"demo_{i}")
        g.attrs["num_samples"] = L
        obs = g.create_group("obs")
        obs.create_dataset(args.image_key, data=src["pixels"][o:o + L],
                           dtype=np.uint8, chunks=(1, *src["pixels"].shape[1:]),
                           compression="gzip", compression_opts=1)
        if write_prop:
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
print(f"[convert] DONE eps={n_eps} frames={total} prop={write_prop} "
      f"trunc={args.truncate_at_success or 'no'} -> {args.out}", flush=True)

# verify: byte-identical images + fields for 2 random episodes
rng = np.random.default_rng(0)
with h5py.File(args.out, "r") as out:
    for i in rng.choice(n_eps, size=min(2, n_eps), replace=False):
        o, L = int(ep_offset[i]), eff_len(i)
        g = out[f"data/demo_{i}"]
        assert g.attrs["num_samples"] == L
        assert np.array_equal(g[f"obs/{args.image_key}"][:], src["pixels"][o:o + L])
        assert np.array_equal(g["actions"][:], src["action"][o:o + L])
        if write_prop:
            assert np.array_equal(g["obs/robot0_eef_quat"][:], src["proprio"][o:o + L, 3:7])
print("[convert] VERIFY_OK", flush=True)
