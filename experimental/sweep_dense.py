"""Dense CEM-iteration sweep, single model load. For n_steps = 1..NMAX (every
integer), for both planning and guided, run the env eval and record success
rate. Loads the model/world/dataset ONCE and only re-runs world.evaluate per
point, so the per-point cost is just the planning+rollout (cheap), not a fresh
process. Writes a CSV incrementally so the curve can be plotted as it fills.
"""
import stable_worldmodel.data.formats.hdf5  # self-register
import os
os.environ["MUJOCO_GL"] = "egl"
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import stable_worldmodel as swm
import lewam.models.gip as gip

NMAX = int(os.environ.get("NMAX", "40"))
OUT = os.environ.get("OUT", "/mnt/data_nvme1/minghao.fu/le-wm-repro-logs/sweep_dense.csv")


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": gip.img_transform(cfg), "goal": gip.img_transform(cfg)}
    dataset = gip.get_dataset(cfg, cfg.eval.dataset_name)
    process = gip.build_process(cfg, dataset)
    episodes, starts = gip.sample_eval_episodes(cfg, dataset)
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)

    model, adim = gip.load_gip_model(cfg.policy)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    f = open(OUT, "w")
    f.write("n_steps,mode,success_rate\n")
    f.flush()

    def do(mode):
        OmegaConf.update(cfg, "gip_eval", {"mode": mode}, force_add=True)
        for ns in range(1, NMAX + 1):
            cfg.solver.n_steps = int(ns)
            policy = gip.build_policy(cfg, model, adim, process, transform)
            world.set_policy(policy)
            m = world.evaluate(
                dataset=dataset, start_steps=starts, goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget, episodes_idx=episodes,
                callables=callables, video=None,
            )
            sr = float(m["success_rate"])
            print(f"[dense] {mode} n_steps={ns} -> {sr}", flush=True)
            f.write(f"{ns},{mode},{sr}\n"); f.flush()

    do("planning")   # model not Actionable yet -> plain CEM
    do("guided")     # build_policy attaches the intention actor
    f.write("DENSE_DONE\n"); f.close()
    print("DENSE SWEEP DONE")


if __name__ == "__main__":
    run()
