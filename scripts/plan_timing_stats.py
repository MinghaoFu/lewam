"""Planning-time statistics from the per-call timing records the evaluation harnesses emit
(`[timing-record] {...}` lines in an eval log, or the timing_records_*.jsonl files).

Each record is one policy call that planned: the envs planned, their plan-cycle index within the
episode (0 = the first plan), the imagined dynamics length, the whole call's seconds and the plan
phase's seconds. With `--successes <json>` (a per-env success list from the eval) the per-episode
totals are also reported for successful episodes only.

usage: plan_timing_stats.py <log or jsonl> [more ...] [--phase plan|call] [--successes file.json]
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np


def load_records(path):
    records = []
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("[timing-record] "):
                records.append(json.loads(line[len("[timing-record] "):]))
            elif line.startswith("{"):
                records.append(json.loads(line))
    return [r for r in records if r.get("envs_planned")]


def episode_totals(records, seconds_key):
    """Per-env sum of its amortized share of every call it was planned in, and its cycle count."""
    total, cycles = defaultdict(float), defaultdict(int)
    for r in records:
        share = r[seconds_key] / len(r["envs_planned"])
        for env in r["envs_planned"]:
            total[env] += share
            cycles[env] += 1
    return total, cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--phase", choices=["plan", "call"], default="plan",
                    help="plan = proposal to finalized chunk; call = the whole policy call including encoding")
    ap.add_argument("--successes", default=None, help="json list of per-env success flags, same env order")
    args = ap.parse_args()
    key = "t_plan_s" if args.phase == "plan" else "t_call_s"
    for path in args.paths:
        records = load_records(path)
        if not records:
            print(f"{path}: no records")
            continue
        by_cycle = defaultdict(list)
        for r in records:
            for cycle in set(r["cycle_idx"]):
                by_cycle[cycle].append((r[key], len(r["envs_planned"]), r["imagined_blocks"]))
        print(f"== {path}: {len(records)} plan calls, phase = {args.phase}")
        print("cycle  calls  envs/call  imagined blocks  seconds per call (mean +- std)  amortized per env")
        for cycle in sorted(by_cycle):
            rows = np.array(by_cycle[cycle], dtype=float)
            print(f"{cycle:5d}  {len(rows):5d}  {rows[:, 1].mean():9.1f}  {rows[:, 2].mean():15.1f}  "
                  f"{rows[:, 0].mean():8.3f} +- {rows[:, 0].std(ddof=1) if len(rows) > 1 else 0:.3f}          "
                  f"{(rows[:, 0] / rows[:, 1]).mean():.4f}")
        total, cycles = episode_totals(records, key)
        envs = sorted(total)
        totals = np.array([total[e] for e in envs])
        print(f"per episode (amortized {args.phase} seconds): mean {totals.mean():.3f} +- {totals.std(ddof=1):.3f} "
              f"over {len(envs)} envs, {np.mean([cycles[e] for e in envs]):.2f} cycles each")
        if args.successes:
            success = json.load(open(args.successes))
            ok = [e for e in envs if success[e]]
            if ok:
                sel = np.array([total[e] for e in ok])
                print(f"successful episodes only ({len(ok)}): mean {sel.mean():.3f} +- {sel.std(ddof=1) if len(ok) > 1 else 0:.3f}")


if __name__ == "__main__":
    main()
