import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG_DIR, LOG_DIR, ensure_output_dirs, save_json


def load_logs(experiment_name: str):
    path = LOG_DIR / f"{experiment_name}_logs.json"
    with path.open() as handle:
        return json.load(handle)


def summarize_run(experiment_name: str, rows):
    final = rows[-1]
    eval_rows = [row for row in rows if "eval_pass_at_1" in row]
    final_eval = eval_rows[-1] if eval_rows else {}

    return {
        "experiment_name": experiment_name,
        "num_steps": len(rows),
        "final_reward_mean": final["reward_mean"],
        "final_train_pass_rate": final["train_pass_rate"],
        "final_clip_fraction": final["clip_fraction"],
        "mean_clip_fraction": sum(row["clip_fraction"] for row in rows) / len(rows),
        "final_policy_kl": final["policy_approx_kl"],
        "mean_policy_kl_abs": sum(abs(row["policy_approx_kl"]) for row in rows) / len(rows),
        "max_lag_updates": max(row["lag_updates"] for row in rows),
        "mean_logprob_delta": sum(row["logprob_delta_mean"] for row in rows) / len(rows),
        "max_logprob_delta_abs": max(row["logprob_delta_max_abs"] for row in rows),
        "final_eval_reward_mean": final_eval.get("eval_reward_mean"),
        "final_eval_pass_at_1": final_eval.get("eval_pass_at_1"),
        "best_eval_pass_at_1": max((row.get("eval_pass_at_1", 0.0) for row in rows), default=0.0),
        "old_logprob_source": final["old_logprob_source"],
        "clip_eps_end": final["clip_eps_used"],
    }


def plot_runs(experiment_names, runs, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("reward_mean", "Reward Mean", axes[0, 0]),
        ("clip_fraction", "Clip Fraction", axes[0, 1]),
        ("policy_approx_kl", "Policy KL Proxy", axes[1, 0]),
        ("eval_pass_at_1", "Eval pass@1", axes[1, 1]),
    ]

    for experiment_name in experiment_names:
        rows = runs[experiment_name]
        steps = [row["step"] for row in rows]
        for metric_name, title, ax in metric_specs:
            values = [row.get(metric_name) for row in rows]
            if all(value is None for value in values):
                continue
            filtered_steps = [step for step, value in zip(steps, values) if value is not None]
            filtered_values = [value for value in values if value is not None]
            ax.plot(filtered_steps, filtered_values, linewidth=2, label=experiment_name)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_ylim(bottom=0.0)
    axes[1, 1].set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    ensure_output_dirs()
    experiment_names = args.experiments
    runs = {name: load_logs(name) for name in experiment_names}
    summary = [summarize_run(name, runs[name]) for name in experiment_names]

    summary_path = LOG_DIR / f"{args.output_prefix}_summary.json"
    save_json(summary_path, summary)

    figure_path = FIG_DIR / f"{args.output_prefix}_comparison.png"
    plot_runs(experiment_names, runs, figure_path)

    print(f"Saved summary to {summary_path}")
    print(f"Saved figure to {figure_path}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment names whose *_logs.json files should be compared",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="ppo_formal_compare",
        help="Prefix for generated summary JSON and comparison figure",
    )
    main(parser.parse_args())
