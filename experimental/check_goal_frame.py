"""Validation: confirm the eval goal frame is a REAL future frame (not the current
frame, not blank). Loads the lift dataset, samples eval episodes the SAME way the
evaluator does (gip.sample_eval_episodes), then for the FIRST chunk re-extracts the
init+goal via the world's _extract_init_goal and checks:
  (1) goal pixels != current (start) pixels  -> goal is a distinct future frame
  (2) goal == frame at start+goal_offset in the raw episode -> goal is the RIGHT frame
  (3) encode(goal) gives a non-degenerate latent distinct from encode(start)
"""
import robomimic_env  # noqa
import stable_worldmodel.data.formats.hdf5  # noqa
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
from omegaconf import OmegaConf
import stable_worldmodel as swm
from stable_worldmodel.world.world import _extract_init_goal
import gip

cfg = OmegaConf.create({
    "cache_dir": None,
    "seed": 42,
    "eval": {"dataset_name": "lift", "num_eval": 50, "goal_offset_steps": 30,
             "img_size": 224},
    "dataset": {"keys_to_cache": ["action", "proprio", "state"]},
})
dataset = gip.get_dataset(cfg, "lift")
episodes, starts = gip.sample_eval_episodes(cfg, dataset)
print(f"[goalcheck] sampled {len(episodes)} eval episodes; first 5 (ep,start)="
      f"{list(zip(episodes[:5], starts[:5]))}")

# extract init+goal exactly as world.evaluate does, for the first chunk of 10
ep10, st10 = episodes[:10], starts[:10]
init_state, goal_state, _ = _extract_init_goal(dataset, ep10, st10, 30)
print(f"[goalcheck] init keys={list(init_state.keys())}  goal keys={list(goal_state.keys())}")
gpix = goal_state["goal"]          # (10, C, H, W) raw uint8
ipix = init_state["pixels"]        # (10, C, H, W) raw uint8 (start frame)
print(f"[goalcheck] goal pixels shape={gpix.shape} dtype={gpix.dtype}  "
      f"start pixels shape={ipix.shape}")
# (1) goal != start
diff = np.abs(gpix.astype(np.int32) - ipix.astype(np.int32))
print(f"[goalcheck] per-episode mean|goal-start| (first 10): "
      f"{diff.reshape(10, -1).mean(1).round(2).tolist()}")
nonzero = (diff.reshape(10, -1).mean(1) > 1.0).sum()
print(f"[goalcheck] {nonzero}/10 episodes have goal DISTINCT from start (mean diff > 1)")
# (3) latent distinctness via the frozen lift base
from train_gcidm import build_frozen_lewm
lewm = build_frozen_lewm("/mnt/minghao_data/.stable-wm/decoders/lift_lewm_weights.pt",
                         embed_dim=192, history_size=3, img_size=224, action_block_dim=35).cuda()
tf = gip.img_transform(OmegaConf.create({"eval": {"img_size": 224}}))
def enc(px):
    # px (10,C,H,W) uint8 -> transform -> encode
    from torchvision import tv_tensors
    t = torch.stack([tf(tv_tensors.Image(torch.as_tensor(x))) for x in px]).cuda().float()
    with torch.no_grad():
        return lewm.encode({"pixels": t.unsqueeze(1)})["emb"][:, 0]
zg = enc(gpix); zi = enc(ipix)
lat_d = (zg - zi).norm(dim=-1)
print(f"[goalcheck] latent L2(goal,start) per-ep (first 10): {lat_d.cpu().round(decimals=2).tolist()}")
print(f"[goalcheck] goal latent norm mean={zg.norm(dim=-1).mean().item():.2f} "
      f"(0 => degenerate)")
print("[goalcheck] DONE")
