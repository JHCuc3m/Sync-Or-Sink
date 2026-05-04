#!/usr/bin/env python3
"""Summarize WikiSQL live PPO results into final-report artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def std(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def stderr(xs: list[float]) -> float:
    return std(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def load_runs(input_dir: Path, include_all_runs: bool) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(input_dir.glob("*/logs.json")):
        payload = json.loads(path.read_text())
        args = payload.get("args", {})
        logs = payload.get("logs", [])
        if not logs:
            continue
        if not include_all_runs:
            if int(args.get("updates", 0)) != 200:
                continue
            if float(args.get("lr", 0.0)) != 1e-5:
                continue
        run_name = path.parent.name
        condition = args.get("condition") or infer_condition(run_name)
        lag = int(args.get("lag", infer_lag(run_name)))
        seed = int(args.get("seed", infer_seed(run_name)))
        runs.append(
            {
                "path": path,
                "run_name": run_name,
                "condition": condition,
                "lag": lag,
                "seed": seed,
                "args": args,
                "logs": logs,
            }
        )
    return runs


def infer_condition(name: str) -> str:
    if "stale_mismatch" in name:
        return "stale_mismatch"
    if "rescored" in name:
        return "rescored"
    if "mismatch" in name:
        return "mismatch"
    if "stale" in name:
        return "stale"
    return "sync"


def infer_lag(name: str) -> int:
    match = re.search(r"_lag(\d+)_", name)
    return int(match.group(1)) if match else 0


def infer_seed(name: str) -> int:
    match = re.search(r"_seed(\d+)", name)
    return int(match.group(1)) if match else 0


def last_eval(logs: list[dict[str, Any]]) -> dict[str, float]:
    eval_logs = [x for x in logs if "eval_pass_at_1" in x]
    item = eval_logs[-1] if eval_logs else logs[-1]
    return {
        "eval_reward_mean": float(item.get("eval_reward_mean", float("nan"))),
        "eval_pass_at_1": float(item.get("eval_pass_at_1", float("nan"))),
        "eval_valid_sql": float(item.get("eval_valid_sql", float("nan"))),
        "eval_correct_result": float(item.get("eval_correct_result", float("nan"))),
    }


def steady_metrics(logs: list[dict[str, Any]], window: int) -> dict[str, float]:
    tail = logs[-window:] if len(logs) >= window else logs

    def collect(key: str) -> list[float]:
        return [float(x[key]) for x in tail if key in x and x[key] is not None]

    return {
        "clip_fraction": mean(collect("clip_fraction")),
        "ratio_variance": mean(collect("ratio_variance")),
        "post_update_clip_fraction": mean(collect("post_update_clip_fraction")),
        "post_update_ratio_variance": mean(collect("post_update_ratio_variance")),
        "abs_logprob_movement": mean(collect("abs_logprob_movement")),
        "post_update_abs_logprob_movement": mean(collect("post_update_abs_logprob_movement")),
        "post_update_abs_pre_update_movement": mean(collect("post_update_abs_pre_update_movement")),
        "entropy": mean(collect("entropy")),
        "reward_mean": mean(collect("reward_mean")),
        "pass_at_1": mean(collect("pass_at_1")),
    }


def summarize_runs(runs: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        row = {
            "run_name": run["run_name"],
            "condition": run["condition"],
            "lag": run["lag"],
            "seed": run["seed"],
            **steady_metrics(run["logs"], window),
            **last_eval(run["logs"]),
        }
        rows.append(row)
    return rows


def grouped(rows: list[dict[str, Any]], condition: str, lag: int | None = None) -> list[dict[str, Any]]:
    out = [r for r in rows if r["condition"] == condition]
    if lag is not None:
        out = [r for r in out if r["lag"] == lag]
    return out


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and not math.isnan(float(r[key]))]
        result[f"{key}_mean"] = mean(vals)
        result[f"{key}_std"] = std(vals)
        result[f"{key}_stderr"] = stderr(vals)
    return result


def make_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    sync_rows = grouped(rows, "sync", 0)
    stale_lags = sorted({r["lag"] for r in rows if r["condition"] == "stale"})

    h2_x = [0] + stale_lags
    h2_clip = []
    h2_clip_err = []
    h2_var = []
    h2_var_err = []
    for lag in h2_x:
        source = sync_rows if lag == 0 else grouped(rows, "stale", lag)
        clip_vals = [r["clip_fraction"] for r in source]
        var_vals = [r["ratio_variance"] for r in source]
        h2_clip.append(mean(clip_vals))
        h2_clip_err.append(stderr(clip_vals))
        h2_var.append(mean(var_vals))
        h2_var_err.append(stderr(var_vals))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].errorbar(h2_x, h2_clip, yerr=h2_clip_err, marker="o", linewidth=2, capsize=3)
    axes[0].set_title("H2: Pre-update clipping rises with lag")
    axes[0].set_xlabel("Actor lag L")
    axes[0].set_ylabel("Pre-update clip fraction")
    axes[0].grid(True, alpha=0.25)
    axes[1].errorbar(h2_x, h2_var, yerr=h2_var_err, marker="o", linewidth=2, capsize=3, color="#b44")
    axes[1].set_title("H2: Ratio variance rises with lag")
    axes[1].set_xlabel("Actor lag L")
    axes[1].set_ylabel("Pre-update ratio variance")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "wikisql_h2_lag_sweep.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for condition, label in [("sync", "Sync"), ("mismatch", "Mismatch"), ("rescored", "Rescored")]:
        selected = [r for r in rows if r["condition"] == condition and r["lag"] == 0]
        if not selected:
            continue
        by_update = defaultdict(list)
        by_update_move = defaultdict(list)
        for r in selected:
            run = next(x for x in RUNS_CACHE if x["run_name"] == r["run_name"])
            for item in run["logs"]:
                by_update[int(item["update"])].append(float(item.get("ratio_variance", 0.0)))
                by_update_move[int(item["update"])].append(float(item.get("post_update_abs_logprob_movement", 0.0)))
        xs = sorted(by_update)
        axes[0].plot(xs, [mean(by_update[x]) for x in xs], label=label, linewidth=1.8)
        axes[1].plot(xs, [mean(by_update_move[x]) for x in xs], label=label, linewidth=1.8)
    axes[0].set_title("H1: Pre-update ratio variance")
    axes[0].set_xlabel("PPO update")
    axes[0].set_ylabel("Ratio variance")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("H1: Logprob movement")
    axes[1].set_xlabel("PPO update")
    axes[1].set_ylabel("Abs logprob movement")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "wikisql_h1_mismatch_rescoring.png", dpi=220)
    plt.close(fig)

    h3_lags = [0, 2, 4, 8]
    stale_pass = []
    stale_err = []
    combined_pass = []
    combined_err = []
    sync_pass = mean([r["eval_pass_at_1"] for r in sync_rows])
    sync_err = stderr([r["eval_pass_at_1"] for r in sync_rows])
    for lag in h3_lags:
        if lag == 0:
            stale_source = sync_rows
            combined_source = sync_rows
        else:
            stale_source = grouped(rows, "stale", lag)
            combined_source = grouped(rows, "stale_mismatch", lag)
        stale_vals = [r["eval_pass_at_1"] for r in stale_source]
        combined_vals = [r["eval_pass_at_1"] for r in combined_source]
        stale_pass.append(mean(stale_vals))
        stale_err.append(stderr(stale_vals))
        combined_pass.append(mean(combined_vals))
        combined_err.append(stderr(combined_vals))

    width = 0.35
    xs = list(range(len(h3_lags)))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar([x - width / 2 for x in xs], stale_pass, width, yerr=stale_err, capsize=3, label="Stale only")
    ax.bar([x + width / 2 for x in xs], combined_pass, width, yerr=combined_err, capsize=3, label="Stale + mismatch")
    ax.axhline(sync_pass, color="black", linestyle="--", linewidth=1.2, label=f"Sync mean ({sync_pass:.3f})")
    if sync_err:
        ax.fill_between([-0.6, len(xs) - 0.4], sync_pass - sync_err, sync_pass + sync_err, color="black", alpha=0.08)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in h3_lags])
    ax.set_xlabel("Actor lag L")
    ax.set_ylabel("Final eval pass@1")
    ax.set_title("H3: Task degradation is clearest for combined drift")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "wikisql_h3_pass_at_1.png", dpi=220)
    plt.close(fig)


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    keys = [
        "eval_reward_mean",
        "eval_pass_at_1",
        "clip_fraction",
        "ratio_variance",
        "post_update_clip_fraction",
        "post_update_ratio_variance",
        "post_update_abs_logprob_movement",
    ]
    groups = []
    for condition in ["sync", "mismatch", "rescored", "stale", "stale_mismatch"]:
        lags = sorted({r["lag"] for r in rows if r["condition"] == condition})
        for lag in lags:
            source = grouped(rows, condition, lag)
            agg = aggregate(source, keys)
            groups.append((condition, lag, agg))

    lines = [
        "# WikiSQL Live PPO Final Results",
        "",
        "These results are generated from the upstream `final_experiments` branch logs.",
        "Metrics are averaged over the last training window for stability metrics and the final evaluation point for task metrics.",
        "",
        "## Aggregate Table",
        "",
        "| Condition | Lag | Seeds | Eval Reward | Eval Pass@1 | Pre Clip | Pre Ratio Var | Post Clip | Post Ratio Var | Post Abs Logprob Move |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, lag, agg in groups:
        lines.append(
            "| "
            f"{condition} | {lag} | {agg['n']} | "
            f"{agg['eval_reward_mean_mean']:.3f} | "
            f"{agg['eval_pass_at_1_mean']:.3f} | "
            f"{agg['clip_fraction_mean']:.5f} | "
            f"{agg['ratio_variance_mean']:.7f} | "
            f"{agg['post_update_clip_fraction_mean']:.5f} | "
            f"{agg['post_update_ratio_variance_mean']:.7f} | "
            f"{agg['post_update_abs_logprob_movement_mean']:.5f} |"
        )

    lines.extend(
        [
            "",
            "## Report-Ready Interpretation",
            "",
            "- H1 is supported in the live WikiSQL setting: fp16 actor mismatch creates nonzero pre-update ratio variance, while learner-side rescoring removes that pre-update variance.",
            "- H2 is strongly supported: stale actors create monotonic increases in pre-update clip fraction and pre-update ratio variance as lag increases.",
            "- H3 is partially supported: stale-only runs do not reduce pass@1 below the sync baseline, but combined stale+mismatch runs at larger lags fall below sync.",
            "- The task-level effect is weaker than the stability effect, so the final report should emphasize stability degradation as the clearest empirical finding.",
            "",
            "## Generated Figures",
            "",
            "- `outputs/figures/wikisql_h2_lag_sweep.png`",
            "- `outputs/figures/wikisql_h1_mismatch_rescoring.png`",
            "- `outputs/figures/wikisql_h3_pass_at_1.png`",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


RUNS_CACHE: list[dict[str, Any]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="outputs/live_ppo")
    parser.add_argument("--figure_dir", default="outputs/figures")
    parser.add_argument("--summary_json", default="outputs/logs/wikisql_live_summary.json")
    parser.add_argument("--summary_md", default="docs/final/wikisql_live_results.md")
    parser.add_argument("--steady_window", type=int, default=50)
    parser.add_argument("--include_all_runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global RUNS_CACHE
    RUNS_CACHE = load_runs(Path(args.input_dir), args.include_all_runs)
    rows = summarize_runs(RUNS_CACHE, args.steady_window)

    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    make_plots(rows, Path(args.figure_dir))
    write_markdown(rows, Path(args.summary_md))
    print(f"Loaded {len(RUNS_CACHE)} runs")
    print(f"Wrote {args.summary_json}")
    print(f"Wrote {args.summary_md}")
    print(f"Wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
