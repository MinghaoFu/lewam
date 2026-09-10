"""Planning-time report over a grid of evaluation logs: for every (harness, arm, planner, goal horizon)
the plan time per replan cycle, the first plan alone, and the total planning time of a successful
episode, each in two regimes.

Batched runs (num_envs > 1): the harness plans every live env in one call, so a cycle's seconds are
the batched cost of up to 50 plans and an episode's total is its amortized share (seconds / envs
planned in that call): a throughput number; reported as mean +- std over eval seeds. Batch-1 runs
(num_envs == 1, i.e. eval.num_eval=1 on a few seeds): every record is one plan, so the seconds are
the time a plan takes alone; reported as mean +- std over those single-episode runs. The first
call of any run (a fresh process) also carries the CUDA warm-up, so a second table gives the warm
plan time by imagined length (our planners' cost depends on the imagined length, LeWM's on its
fixed horizon) from every record after the first call of its process.

usage: plan_timing_report.py <eval logs ...> [--phase plan|call] [--json out.json]
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np

from plan_timing_stats import episode_totals, load_records

ARM_NAMES = {"fx_nm192": "noreg", "fx_vsig192": "SIGReg"}


def load_run(path):
    """One eval log -> its timing summary line, plan records and per-episode success flags."""
    summary, text = None, open(path, errors="ignore").read()
    for line in text.splitlines():
        if line.startswith("[timing-json] "):
            summary = json.loads(line[len("[timing-json] "):])
    if summary is None:
        return None
    success = summary.get("episode_successes")
    if not success:
        # older logs: the flags only appear in the printed metrics dict, possibly across several lines
        match = re.search(r"'episode_successes': array\(\[(.*?)\]", text, re.S)
        success = [tok == "True" for tok in re.findall(r"True|False", match.group(1))] if match else None
    return dict(path=path, summary=summary, records=load_records(path), success=success)


def run_key(summary):
    """(harness, arm, planner, goal horizon in raw steps) of a run."""
    if summary.get("planner") == "cem":
        return ("LeWM", "release", "CEM", int(summary["goal_offset_steps"]))
    policy = summary["policy"]
    arm = next((name for tag, name in ARM_NAMES.items() if tag in policy), policy)
    planner = {"best_of_k": "best-of-K", "grad": "gradient-TR" if summary.get("grad_tr") else "gradient"}.get(
        summary.get("plan_mode"), str(summary.get("plan_mode")))
    horizon = int(summary["horizon_blocks"]) * int(summary["action_block"])
    return ("LeWAM", arm, planner, horizon)


def run_stats(run, key):
    """Per replan cycle: seconds and envs planned of the (one) call per cycle; per episode: amortized
    totals, all and successful only (with one env the amortized total is the episode's own time)."""
    by_cycle, by_cycle_envs = defaultdict(list), defaultdict(list)
    for r in run["records"]:
        for cycle in set(r["cycle_idx"]):
            by_cycle[cycle].append(r[key])
            by_cycle_envs[cycle].append(len(r["envs_planned"]))
    per_cycle = {cycle: float(np.mean(v)) for cycle, v in by_cycle.items()}
    per_cycle_envs = {cycle: float(np.mean(v)) for cycle, v in by_cycle_envs.items()}
    total, _ = episode_totals(run["records"], key)
    envs = sorted(total)
    success = run["success"]
    ok = [e for e in envs if success is not None and e < len(success) and success[e]]
    return per_cycle, per_cycle_envs, [total[e] for e in envs], [total[e] for e in ok], success is not None


def warm_by_length(run, key):
    """(seconds, envs planned) of every plan call after the first one of the process, keyed by the
    imagined length."""
    by_length = defaultdict(list)
    for r in run["records"]:
        if r["call_idx"] > 0 and r.get("imagined_blocks") is not None:
            by_length[int(r["imagined_blocks"])].append((r[key], len(r["envs_planned"])))
    return by_length


def mean_std_envs(pairs):
    """mean_std of the seconds plus the mean envs planned per call."""
    stat = mean_std([p[0] for p in pairs])
    if stat is not None:
        stat["envs"] = float(np.mean([p[1] for p in pairs]))
    return stat


def mean_std(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    return dict(mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                n=int(len(values)))


def fmt(stat, digits=3):
    """mean +- std (n, and the mean envs planned per call when it is not 1)."""
    if stat is None:
        return "-"
    envs = stat.get("envs")
    tail = f", {envs:.0f} envs" if envs is not None and envs != 1 else ""
    return f"{stat['mean']:.{digits}f} +- {stat['std']:.{digits}f} (n={stat['n']}{tail})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--phase", choices=["plan", "call"], default="plan",
                    help="plan = proposal to finalized chunk; call = the whole policy call including encoding")
    ap.add_argument("--json", default=None, help="write every statistic to this json file")
    args = ap.parse_args()
    key = "t_plan_s" if args.phase == "plan" else "t_call_s"

    groups = defaultdict(list)
    for path in args.paths:
        run = load_run(path)
        if run is None:
            print(f"{path}: no timing summary line, skipped")
            continue
        groups[run_key(run["summary"])].append(run)

    report = {}
    for group in sorted(groups):
        batched_cycles, batched_totals, batched_ok, batched_seeds, success_rates = defaultdict(list), [], [], [], []
        single_cycles, single_totals, single_ok, single_runs = defaultdict(list), [], [], 0
        batched_warm, single_warm, flags_missing = defaultdict(list), defaultdict(list), 0
        for run in groups[group]:
            per_cycle, per_cycle_envs, totals, ok, has_flags = run_stats(run, key)
            flags_missing += int(not has_flags)
            warm = warm_by_length(run, key)
            if int(run["summary"].get("num_envs", 0)) == 1:
                single_runs += 1
                for cycle, seconds in per_cycle.items():
                    single_cycles[cycle].append((seconds, 1))
                single_totals.extend(totals)
                single_ok.extend(ok)
                for length, pairs in warm.items():
                    single_warm[length].extend(pairs)
                continue
            batched_seeds.append(int(run["summary"]["seed"]))
            success_rates.append(float(run["summary"].get("success_rate", float("nan"))))
            for cycle, seconds in per_cycle.items():
                batched_cycles[cycle].append((seconds, per_cycle_envs[cycle]))
            batched_totals.append(float(np.mean(totals)))
            if ok:
                batched_ok.append(float(np.mean(ok)))
            for length, pairs in warm.items():
                batched_warm[length].extend(pairs)
        report["|".join(map(str, group))] = dict(
            seeds=sorted(batched_seeds), success_rate=mean_std(success_rates),
            batched_per_cycle={c: mean_std_envs(v) for c, v in sorted(batched_cycles.items())},
            batched_episode_all=mean_std(batched_totals), batched_episode_success=mean_std(batched_ok),
            batched_warm_by_length={n: mean_std_envs(v) for n, v in sorted(batched_warm.items())},
            runs_without_success_flags=flags_missing, batch1_runs=single_runs,
            batch1_per_cycle={c: mean_std_envs(v) for c, v in sorted(single_cycles.items())},
            batch1_episode_all=mean_std(single_totals), batch1_episode_success=mean_std(single_ok),
            batch1_warm_by_length={n: mean_std_envs(v) for n, v in sorted(single_warm.items())})

    print(f"phase = {args.phase}; seconds; batched = mean +- std over seeds, batch 1 = over single-episode runs\n")
    print("| harness | arm | planner | H | seeds | success % | batched cycle 0 | batched cycles 1.. | "
          "batched episode (success) | batch-1 cycle 0 | batch-1 cycles 1.. | batch-1 episode (success) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, stats in report.items():
        harness, arm, planner, horizon = name.split("|")
        later = lambda table: ", ".join(fmt(table[c]) for c in sorted(table) if c != 0) or "-"
        print(f"| {harness} | {arm} | {planner} | {horizon} | {stats['seeds']} | {fmt(stats['success_rate'], 1)} | "
              f"{fmt(stats['batched_per_cycle'].get(0))} | {later(stats['batched_per_cycle'])} | "
              f"{fmt(stats['batched_episode_success'])} | {fmt(stats['batch1_per_cycle'].get(0))} | "
              f"{later(stats['batch1_per_cycle'])} | {fmt(stats['batch1_episode_success'])} |")

    print("\nwarm plan seconds by imagined length (records after the first call of each process)\n")
    print("| harness | arm | planner | H | batched: blocks -> seconds | batch-1: blocks -> seconds |")
    print("|---|---|---|---|---|---|")
    for name, stats in report.items():
        harness, arm, planner, horizon = name.split("|")
        by_length = lambda table: ", ".join(f"{n}: {fmt(table[n])}" for n in sorted(table)) or "-"
        print(f"| {harness} | {arm} | {planner} | {horizon} | {by_length(stats['batched_warm_by_length'])} | "
              f"{by_length(stats['batch1_warm_by_length'])} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
