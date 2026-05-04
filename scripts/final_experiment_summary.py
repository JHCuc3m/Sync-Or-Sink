import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG_DIR, LOG_DIR, ensure_output_dirs, save_json


def load_rows(experiment_name):
    path = LOG_DIR / f"{experiment_name}_logs.json"
    with path.open() as handle:
        return json.load(handle)


def summarize(experiment_name):
    rows = load_rows(experiment_name)
    final = rows[-1]
    eval_rows = [row for row in rows if "eval_pass_at_1" in row]
    final_eval = eval_rows[-1] if eval_rows else {}

    return {
        "experiment_name": experiment_name,
        "num_steps": len(rows),
        "max_lag_updates": max(row["lag_updates"] for row in rows),
        "mean_clip_fraction": mean(row["clip_fraction"] for row in rows),
        "mean_abs_policy_kl_proxy": mean(abs(row["policy_approx_kl"]) for row in rows),
        "mean_entropy": mean(row["entropy"] for row in rows),
        "mean_importance_clipped_fraction": mean(row["importance_clipped_fraction"] for row in rows),
        "mean_logprob_delta": mean(row["logprob_delta_mean"] for row in rows),
        "max_logprob_delta_abs": max(row["logprob_delta_max_abs"] for row in rows),
        "final_reward_mean": final["reward_mean"],
        "final_train_pass_rate": final["train_pass_rate"],
        "final_eval_reward_mean": final_eval.get("eval_reward_mean"),
        "final_eval_pass_at_1": final_eval.get("eval_pass_at_1"),
        "best_eval_pass_at_1": max((row.get("eval_pass_at_1", 0.0) for row in rows), default=0.0),
        "old_logprob_source": final["old_logprob_source"],
        "clip_eps_end": final["clip_eps_used"],
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def format_float(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key)
            cells.append(format_float(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_lag(summary_rows, output_path):
    ordered = sorted(summary_rows, key=lambda row: row["max_lag_updates"])
    lags = [row["max_lag_updates"] for row in ordered]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    specs = [
        ("mean_clip_fraction", "Mean Clip Fraction"),
        ("mean_abs_policy_kl_proxy", "Mean |KL Proxy|"),
        ("final_eval_reward_mean", "Final Eval Reward"),
    ]
    for ax, (key, title) in zip(axes, specs):
        values = [row[key] for row in ordered]
        ax.plot(lags, values, marker="o", linewidth=2)
        ax.set_xlabel("Max lag updates")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_mitigation(summary_rows, output_path):
    labels = [row["experiment_name"].replace("ppo_final_", "") for row in summary_rows]
    metrics = [
        ("mean_clip_fraction", "Mean Clip"),
        ("mean_abs_policy_kl_proxy", "Mean |KL|"),
        ("final_eval_reward_mean", "Eval Reward"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, title) in zip(axes, metrics):
        values = [row[key] for row in summary_rows]
        ax.bar(labels, values, color="#4C78A8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def classify_structured_json_failure(sample):
    response = sample.get("response", "")
    answer = sample.get("answer")
    parity = sample.get("parity", "")
    stripped = response.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "invalid_json", "No JSON object found"

    candidate = stripped[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return "invalid_json", "JSON parse error"

    if candidate != stripped:
        return "extra_text", "Valid JSON object appears with surrounding text"
    if parsed.get("answer") != answer:
        return "wrong_answer", "Format is valid but answer is incorrect"
    if parsed.get("parity") != parity:
        return "wrong_parity", "Answer is correct but parity is incorrect"
    return "success", "Exact structured response"


def collect_failure_examples(experiment_names):
    examples = []
    for name in experiment_names:
        rows = load_rows(name)
        eval_rows = [row for row in rows if row.get("samples")]
        if not eval_rows:
            continue
        for sample in eval_rows[-1]["samples"]:
            failure_type, explanation = classify_structured_json_failure(sample)
            examples.append(
                {
                    "experiment_name": name,
                    "failure_type": failure_type,
                    "prompt": sample.get("prompt", "").replace("\n", " "),
                    "response": sample.get("response", "").replace("\n", " "),
                    "answer": sample.get("answer"),
                    "parity": sample.get("parity", ""),
                    "explanation": explanation,
                }
            )
            break
    return examples


def write_markdown(path, lag_rows, mitigation_rows, failures):
    columns = [
        ("Experiment", "experiment_name"),
        ("Max Lag", "max_lag_updates"),
        ("Mean Clip", "mean_clip_fraction"),
        ("Mean |KL|", "mean_abs_policy_kl_proxy"),
        ("Eval Reward", "final_eval_reward_mean"),
        ("Pass@1", "final_eval_pass_at_1"),
    ]
    mitigation_columns = columns + [
        ("Mean Drift", "mean_logprob_delta"),
        ("Max |Drift|", "max_logprob_delta_abs"),
        ("Old Logprob Source", "old_logprob_source"),
    ]

    failure_columns = [
        ("Experiment", "experiment_name"),
        ("Failure Type", "failure_type"),
        ("Response", "response"),
        ("Expected", "answer"),
        ("Parity", "parity"),
        ("Explanation", "explanation"),
    ]

    lines = [
        "# Final PPO Experiment Results",
        "",
        "## Lag Sweep",
        "",
        markdown_table(lag_rows, columns),
        "",
        "## Mitigation Ablation",
        "",
        markdown_table(mitigation_rows, mitigation_columns),
        "",
        "## Qualitative Failure Examples",
        "",
        markdown_table(failures, failure_columns),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main(args):
    ensure_output_dirs()
    lag_rows = [summarize(name) for name in args.lag_experiments]
    mitigation_rows = [summarize(name) for name in args.mitigation_experiments]
    failures = collect_failure_examples(args.failure_experiments)

    payload = {
        "lag_sweep": lag_rows,
        "mitigation_ablation": mitigation_rows,
        "failure_examples": failures,
    }
    save_json(LOG_DIR / f"{args.output_prefix}_summary.json", payload)
    plot_lag(lag_rows, FIG_DIR / f"{args.output_prefix}_lag_sweep.png")
    plot_mitigation(mitigation_rows, FIG_DIR / f"{args.output_prefix}_mitigation_ablation.png")
    write_markdown(Path(args.markdown_out), lag_rows, mitigation_rows, failures)

    print(f"Saved summary to {LOG_DIR / f'{args.output_prefix}_summary.json'}")
    print(f"Saved lag plot to {FIG_DIR / f'{args.output_prefix}_lag_sweep.png'}")
    print(f"Saved mitigation plot to {FIG_DIR / f'{args.output_prefix}_mitigation_ablation.png'}")
    print(f"Saved markdown to {args.markdown_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag_experiments", nargs="+", required=True)
    parser.add_argument("--mitigation_experiments", nargs="+", required=True)
    parser.add_argument("--failure_experiments", nargs="+", required=True)
    parser.add_argument("--output_prefix", default="ppo_final")
    parser.add_argument("--markdown_out", default="docs/final/final_experiment_results.md")
    main(parser.parse_args())
