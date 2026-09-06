"""Board collector (protocol item 3, design verified 2026-09-06): mean +- sample std over the eval seeds.

Reads the heartbeat logs the entries write beside the checkpoints (`hb_*.log` under ckpts/jointflow_gr_*/ and
ckpts/jointflow_tc/), keeps the arms whose name starts with --arm_prefix (fx_ = trained on 67e20e1 or later; the
invalidated board never matches), takes the LAST line per eval seed (a resubmit after a kill overrides), and
reduces every (cell, arm, mode) to per-seed values, n, mean and sample std (ddof=1). The [timing-json] lines the
eval script prints (protocol item 5) are picked up from the persisted eval logs (`ev_*.log`) and averaged over seeds.

  python3 scripts/collect_board.py [--ckpts ROOT] [--arm_prefix fx_] [--out_json board.json] [--out_md board.md]

Result line format (from the entries):  [stamp] [who] JFGC fx_nm192 evseed_42 success_rate: 62.0
GR entries put the ARM in that slot (the cell is in the directory name, jointflow_gr_<cell>_<arm>); the TC entries
put the CELL there (the arm is in the log file name, hb_ev_tc_<cell>_<arm>_s<seed>.log)."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

RESULT_LINE = re.compile(r"^\[(?P<stamp>[^\]]+)\] \[(?P<who>[^\]]+)\] (?P<tag>JF[A-Z]+|DP[A-Z]*) "
                         r"(?P<arm_or_cell>\S+) evseed_(?P<seed>\d+) success_rate: (?P<value>[0-9.]+)")
TIMING_LINE = "[timing-json] "
TC_CELLS = ("toolhang", "transport", "drawer", "cube")
MODE_LABEL = {"JFGC": "reactive (goal-conditioned)", "JFROLL": "best-of-K", "JFGRAD": "gradient",
              "JFGRADTR": "gradient-TR", "JFSTEER": "SteerMPC", "JFCEM": "CEM", "JFROLLRAND": "random-candidates",
              "JFTC": "reactive TC (goal-blind)", "JFTCGC": "goal-conditioned TC", "DP": "diffusion policy"}
TIMING_FIELDS = ("t_block_amortized_mean_s", "t_call_mean_s", "n_replanned_mean", "t_episode_mean_s",
                 "t_episode_std_s", "replans_per_env_mean", "wall_total_s")


def tc_arm_from_log_name(log_name, cell):
    """hb_ev_tc_<cell>_<arm>_s<seed>.log / hb_tc_<cell>_<arm>_s<seed>.log / ev_tc_<cell>_<arm>_s<seed>_e<es>.log -> arm"""
    name = re.sub(r"^(hb_ev_|hb_|ev_)", "", Path(log_name).stem)
    name = re.sub(r"_e\d+$", "", name)
    name = re.sub(r"[_-]s\d+$", "", name)
    prefix = f"tc_{cell}_"
    return name[len(prefix):] if name.startswith(prefix) else None


def gr_cell_from_dir(dir_name, arm):
    """jointflow_gr_<cell>_<arm> -> cell (the arm is known from the result line, so the split is exact)"""
    prefix, suffix = "jointflow_gr_", f"_{arm}"
    if dir_name.startswith(prefix) and dir_name.endswith(suffix) and len(dir_name) > len(prefix) + len(suffix):
        return dir_name[len(prefix):-len(suffix)]
    return None


def tag_from_timing(timing, in_tc_dir):
    mode, plan_mode = timing.get("mode"), timing.get("plan_mode")
    if mode == "jointflow_gc":
        return "JFTCGC" if in_tc_dir else "JFGC"
    if mode == "jointflow_policy":
        return "JFTC"
    if mode == "dp_policy":
        return "DP"
    if mode == "jointflow_plan":
        if plan_mode == "best_of_k":
            return "JFROLLRAND" if timing.get("plan_random_candidates") else "JFROLL"
        if plan_mode == "grad":
            return "JFGRADTR" if float(timing.get("grad_tr") or 0) > 0 else "JFGRAD"
        return {"steer": "JFSTEER", "cem": "JFCEM"}.get(plan_mode, f"JF_{plan_mode}")
    return f"{mode}"


def collect(ckpts_root, arm_prefix):
    results = {}          # (cell, arm, tag) -> {seed: value}
    sources = {}          # (cell, arm, tag) -> [file:line]
    timings = {}          # (cell, arm, tag) -> [timing dict]
    gr_dir_identity = {}  # GR dir -> (cell, arm), learned from its result lines
    grev_identity = {}    # standalone-eval (jf_grev) "<cell>_<tag>" -> (cell, arm), learned from its result lines
    log_dirs = sorted(ckpts_root.glob("jointflow_gr_*")) + [ckpts_root / "jointflow_tc", ckpts_root / "jf_grev"]
    for log_dir in log_dirs:
        if not log_dir.is_dir():
            continue
        in_tc_dir = log_dir.name == "jointflow_tc"
        in_grev_dir = log_dir.name == "jf_grev"
        for log_path in sorted(log_dir.glob("hb_*.log")):
            for line_no, line in enumerate(log_path.read_text(errors="replace").splitlines(), 1):
                match = RESULT_LINE.match(line)
                if not match:
                    continue
                tag, seed, value = match["tag"], int(match["seed"]), float(match["value"])
                if in_grev_dir:
                    # standalone GR eval entry: who = grev-<cell>-<ckpt tag>, ckpt tag = <arm>_s<train seed>
                    who = match["who"].split("-")
                    cell = who[1] if len(who) == 3 and who[0] == "grev" else None
                    ckpt_tag = match["arm_or_cell"]
                    arm = re.sub(r"_s\d+$", "", ckpt_tag)
                    if cell is not None:
                        grev_identity[f"{cell}_{ckpt_tag}"] = (cell, arm)
                elif in_tc_dir:
                    cell = match["arm_or_cell"]
                    arm = tc_arm_from_log_name(log_path.name, cell)
                else:
                    arm = match["arm_or_cell"]
                    cell = gr_cell_from_dir(log_dir.name, arm)
                    if cell is not None:
                        gr_dir_identity[log_dir.name] = (cell, arm)
                if arm is None or cell is None or not arm.startswith(arm_prefix):
                    continue
                key = (cell, arm, tag)
                results.setdefault(key, {})[seed] = value          # last line per seed wins
                sources.setdefault(key, []).append(f"{log_path.name}:{line_no}")
        for log_path in sorted(log_dir.glob("ev_*.log")):
            for line in log_path.read_text(errors="replace").splitlines():
                if not line.startswith(TIMING_LINE):
                    continue
                try:
                    timing = json.loads(line[len(TIMING_LINE):])
                except json.JSONDecodeError:
                    continue
                if in_grev_dir:
                    # ev_<cell>_<ckpt tag>_<mode>_e<seed>.log: match the longest known "<cell>_<ckpt tag>" prefix
                    known = sorted(grev_identity, key=len, reverse=True)
                    key_hit = next((k for k in known if log_path.name.startswith(f"ev_{k}_")), None)
                    cell, arm = grev_identity.get(key_hit, (None, None))
                elif in_tc_dir:
                    cell = next((c for c in TC_CELLS if log_path.name.startswith(f"ev_tc_{c}_")), None)
                    arm = tc_arm_from_log_name(log_path.name, cell) if cell else None
                else:
                    cell, arm = gr_dir_identity.get(log_dir.name, (None, None))
                if arm is None or cell is None or not arm.startswith(arm_prefix):
                    continue
                timings.setdefault((cell, arm, tag_from_timing(timing, in_tc_dir)), []).append(timing)
    board = {}
    for key in sorted(results):
        per_seed = results[key]
        values = np.array([per_seed[seed] for seed in sorted(per_seed)], dtype=float)
        entry = dict(cell=key[0], arm=key[1], tag=key[2], mode=MODE_LABEL.get(key[2], key[2]),
                     n=int(len(values)), mean=float(values.mean()),
                     std=(float(values.std(ddof=1)) if len(values) > 1 else float("nan")),
                     per_seed={str(seed): per_seed[seed] for seed in sorted(per_seed)}, sources=sources[key])
        if key in timings:
            entry["timing"] = {field: float(np.mean([t[field] for t in timings[key] if field in t]))
                               for field in TIMING_FIELDS if any(field in t for t in timings[key])}
            entry["timing"]["n_timing_logs"] = len(timings[key])
        board["|".join(key)] = entry
    return board


def markdown_table(board):
    lines = ["| cell | arm | mode | mean ± std (n) | per-seed | block ms (amortized) | call ms (batch) | episode s |",
             "|---|---|---|---|---|---|---|---|"]
    for entry in board.values():
        timing = entry.get("timing", {})
        fmt_ms = lambda field: f"{1e3 * timing[field]:.1f}" if field in timing else "–"
        per_seed = ", ".join(f"{seed}: {value:g}" for seed, value in entry["per_seed"].items())
        std = f"{entry['std']:.1f}" if np.isfinite(entry["std"]) else "n/a"
        episode = (f"{timing['t_episode_mean_s']:.2f} ± {timing['t_episode_std_s']:.2f}"
                   if "t_episode_mean_s" in timing else "–")
        lines.append(f"| {entry['cell']} | {entry['arm']} | {entry['mode']} | {entry['mean']:.1f} ± {std} ({entry['n']}) "
                     f"| {per_seed} | {fmt_ms('t_block_amortized_mean_s')} | {fmt_ms('t_call_mean_s')} | {episode} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpts", default="/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/ckpts")
    parser.add_argument("--arm_prefix", default="fx_", help="only arms whose name starts with this (fx_ = post-fix)")
    parser.add_argument("--out_json", default="board.json")
    parser.add_argument("--out_md", default="board.md")
    args = parser.parse_args()
    board = collect(Path(args.ckpts), args.arm_prefix)
    Path(args.out_json).write_text(json.dumps(board, indent=1))
    table = (f"Std = sample std over the eval seeds (ddof = 1); n = seeds. Arms with prefix '{args.arm_prefix}'.\n\n"
             + markdown_table(board))
    Path(args.out_md).write_text(table + "\n")
    print(table)
    print(f"\n{len(board)} rows -> {args.out_json}, {args.out_md}")


if __name__ == "__main__":
    main()
