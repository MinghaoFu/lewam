"""Build the migration manifest for docs/HANDOFF.md: every artifact needed to reproduce the reported numbers and
continue the work, as absolute paths that exist right now, with sizes, grouped in tiers. Writes
docs/handoff_manifest.txt (tab-separated: tier, bytes, path; '#' lines are comments) and prints the totals and any
path that is missing. Directories are listed as one line and copied whole; their size is the sum of their files.
usage: handoff_manifest.py"""
import glob
import json
import os
from pathlib import Path

import h5py

HROOT = "/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam"
CK = f"{HROOT}/ckpts"
OUT = Path(__file__).resolve().parents[1] / "docs" / "handoff_manifest.txt"

TC_RUNS = ["tc_toolhang_fx_mnm192_s42", "tc_toolhang_fx_msig192_s42", "tc_toolhang_fx_mfl192_s42",
           "tc_toolhang_fx_mflsig192_s42", "tc_toolhang_fx_mvsig192_s42",
           "tc_drawer_fx_mnm192_s42", "tc_drawer_fx_msig192_s42", "tc_drawer_fx_mvsig192_s42",
           "tc_transport_fx_mnm192_s42", "tc_transport_fx_msig192_s42", "tc_transport_fx_mvsig192_s42",
           "tc_twodoor2_fx_flowvsig192_s42", "tc_twodoor2_fx_msevsig192_s42",
           "tc_drawer_fx_mvsig192_3v_s42"]
GR_CELLS = {"pusht": ["fx_nm192", "fx_vsig192", "fx_sig192", "fx_fl192", "fx_flsig192"],
            "tworoom": ["fx_nm192", "fx_vsig192"], "pointmaze_large": ["fx_nm192", "fx_vsig192"],
            "cube": ["fx_nm192", "fx_vsig192"], "reacher_policy": ["fx_nm192", "fx_vsig192", "fx_sig192"],
            "toolhang": ["fx_nm192", "fx_vsig192"], "drawer": ["fx_nm192", "fx_vsig192"],
            "transport": ["fx_nm192", "fx_vsig192"]}
CODE_SHAS = ["67e20e1", "5cca99f", "b05a7f8", "42560ab", "2a5acc5", "f4fb884", "2b5805f", "ed314d6", "75f0fa7",
             "85b5df9", "c461482"]
DATASETS = ["pusht", "tworoom", "pointmaze_large", "cube", "reacher_policy", "toolhang", "toolhang_eih", "drawer",
            "drawer_3view", "transport", "transport_3view", "twodoor2"]
CACHES = [("preload_cache_u8/pusht_expert_train", "pusht_expert_train_fs5_i224"),
          ("preload_cache_u8/reacher_policy", "reacher_policy_fs5_i224"),
          ("preload_cache/tworoom", "tworoom_fs5_i224"),
          ("preload_cache/pointmaze_large", "pointmaze_large_fs5_i224"),
          ("preload_cache/cube_single_expert", "cube_single_expert_fs5_i224"),
          ("preload_cache/tool_hang", "tool_hang_fs5_i224_raw"), ("preload_cache/tool_hang", "tool_hang_fs5_i224"),
          ("preload_cache/tool_hang", "tool_hang_fs5_i224_raw.pixels_r0eih"),
          ("preload_cache/drawer_cleanup_fixed", "drawer_cleanup_fixed_fs5_i224_raw"),
          ("preload_cache/drawer_cleanup_fixed", "drawer_cleanup_fixed_fs5_i224"),
          ("preload_cache/drawer_cleanup_fixed", "drawer_cleanup_fixed_fs5_i224_raw.pixels_r0eih"),
          ("preload_cache/drawer_cleanup_fixed", "drawer_cleanup_fixed_fs5_i224_raw.pixels_r1eih"),
          ("preload_cache/transport", "transport_fs5_i224_raw"), ("preload_cache/transport", "transport_fs5_i224"),
          ("preload_cache/transport", "transport_fs5_i224_raw.pixels_r0eih"),
          ("preload_cache/transport", "transport_fs5_i224_raw.pixels_r1eih"),
          ("preload_cache/twodoor2", "twodoor2_fs5_i224_raw")]


def dir_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size      # symlinks count as links, not targets
            except OSError:
                pass
    return total


def size_of(path):
    if os.path.isdir(path):
        return dir_bytes(path)
    return os.stat(path).st_size


rows, missing = [], []


def add(tier, path, note=""):
    if not os.path.exists(path):
        missing.append((tier, path, note))
        return
    rows.append((tier, size_of(path), path, note))


def add_glob(tier, pattern, note=""):
    hits = sorted(glob.glob(pattern))
    if not hits:
        missing.append((tier, pattern, note))
    for hit in hits:
        if os.path.isfile(hit):
            rows.append((tier, os.stat(hit).st_size, hit, note))


# tier 1: checkpoints and logs behind every reported number (run dirs copied whole: best, latest, full, snapshots, logs)
for run in TC_RUNS:
    add("1-results", f"{CK}/jointflow_tc/{run}", "TC arm run dir (jointflow_best.pt = the row)")
for cell, arms in GR_CELLS.items():
    for arm in arms:
        add("1-results", f"{CK}/jointflow_gr_{cell}_{arm}/{arm}_s42", f"GR arm run dir ({cell})")
        add_glob("1-results", f"{CK}/jointflow_gr_{cell}_{arm}/*.log", "chained eval + heartbeat logs")
add_glob("1-results", f"{CK}/jointflow_tc/*.log", "TC eval logs: hb_, ev_, evtask_, evshard_, evexp_")
add("1-results", f"{CK}/jf_grev", "planner ladders, E7 grid, controls (hb + ev logs)")
add("1-results", f"{CK}/lewm_grid", "LeWM E7 rows")
for cell in ["toolhang", "drawer", "transport"]:
    add("1-results", f"{CK}/wf8_dp/{cell}_eval", "DP-T eval logs")
add("1-results", f"{CK}/wf8_uni/toolhang_dp_noprop/snap_ep120.ckpt", "DP-T toolhang row")
add("1-results", f"{CK}/wf8_dp/drawer/latest.ckpt", "DP-T drawer row (epoch 120)")
add("1-results", f"{CK}/wf8_dp/transport/epoch=0120-train_loss=0.0476.ckpt", "DP-T transport row")
add("1-results", f"{CK}/wf8_dp/cube/latest.ckpt", "DP-T cube (no row; optional)")
add("1-results", f"{CK}/official_baselines/tool_hang/eval", "B1 eval logs of the published checkpoints (the weights are public downloads: robomimic model zoo BC-RNN tool_hang, diffusion_policy tool_hang train_0; not migrated)")
add("1-results", f"{CK}/probes_e3", "E3 outputs")
add("1-results", f"{CK}/probes_e4", "E4 outputs")
add("1-results", f"{CK}/dp_tc", "DP-C runs trained by train_dp.py (in progress)")
add("1-results", f"{HROOT}/code/lewm_main_eval/hf_release_native", "LeWM authors' checkpoints (E7 row)")
# tier 1 code: tarballs, entries, external repos
for sha in CODE_SHAS:
    add("1-code", f"{HROOT}/code/lewam_jointflow_{sha}.tar.gz", "code tarball a pod ran")
for name in ["dp_repo.tar.gz", "dp_env.tar.gz", "wf8_sim_src.tar.gz", "sim_src.tgz", "lewm_official_code.tgz",
             "lewm_official.tgz", "lewam_baselines_code.tgz"]:
    add("1-code", f"{HROOT}/code/{name}", "external code / env")
add_glob("1-code", f"{HROOT}/code/*.sh", "entry scripts")
add_glob("1-code", f"{HROOT}/code/*.py", "helper scripts copied to pods")
add("1-code", f"{HROOT}/code/dp_task_yamls", "DP-T task yamls")
add("1-code", f"{HROOT}/code/lewam_scripts", "converter copies used by the DP-T entries")
add("1-code", f"{HROOT}/code/lewm_main_eval/stable-worldmodel", "LeWM eval stack")
# tier 2: datasets
for name in DATASETS:
    add("2-data", f"{HROOT}/wf8/train/{name}.h5", "training file")
with h5py.File(f"{HROOT}/wf8/train/drawer.h5", "r") as f:
    link = f.get("pixels", getlink=True)
    if isinstance(link, h5py.ExternalLink):
        add("2-data", link.filename if os.path.isabs(link.filename) else f"{HROOT}/wf8/train/{link.filename}",
            "drawer.h5 external-link target")
add("2-data", f"{HROOT}/wf8/train/_views", "external-link joins for the multi-view caches")
add("2-data", f"{HROOT}/wf8/eval", "eval views (links) and symlinks")
add("2-data", f"{HROOT}/wf8/env", "env metadata: model xml h5s, robomimic env args")
add("2-data", f"{HROOT}/wf8/README.md", "dataset provenance and pod-side names")
# tier 3: caches used by the reported rows and the multi-view runs
for sub, tag in CACHES:
    for ext in ["frames.npy", "aux.npz", "meta.json"]:
        add("3-caches", f"{HROOT}/{sub}/{tag}.{ext}", "preload cache")
# devbox
add("4-devbox", "/home/tiger/lewam_project/jobs", "Merlin YAMLs, submit guard, entry copies")
add("4-devbox", "/home/tiger/.claude/projects/-home-tiger-lewam/memory", "session memory (rules, ops, results notes)")
add("4-devbox", "/home/tiger/.job_name_map.tsv", "coded job name -> real name -> job id")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as fh:
    fh.write("# LeWAM migration manifest (2026-09-11). tier<TAB>bytes<TAB>path<TAB>note. Directories are copied whole.\n")
    fh.write("# The repo itself is git: git clone git@github.com:MinghaoFu/lewam.git (branch lewam-jointflow).\n")
    for tier, size, path, note in rows:
        fh.write(f"{tier}\t{size}\t{path}\t{note}\n")
totals = {}
for tier, size, _path, _note in rows:
    totals[tier] = totals.get(tier, 0) + size
print(json.dumps({t: f"{v / 1e9:.1f} GB" for t, v in sorted(totals.items())}, indent=1))
print("entries:", len(rows), "| missing:", missing)
