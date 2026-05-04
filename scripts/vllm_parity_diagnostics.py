#!/usr/bin/env python3
"""Create diagnostics for static vLLM-vs-HF parity results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[idx]


def summarize_deltas(deltas: list[float], clip_eps: float) -> dict[str, Any]:
    abs_deltas = [abs(x) for x in deltas]
    lower = math.log(1.0 - clip_eps)
    upper = math.log(1.0 + clip_eps)
    crossings = [x < lower or x > upper for x in deltas]
    ratio_errors = [math.exp(x) for x in deltas]
    return {
        "num_tokens": len(deltas),
        "mean_delta": statistics.fmean(deltas) if deltas else float("nan"),
        "std_delta": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "mean_abs_delta": statistics.fmean(abs_deltas) if abs_deltas else float("nan"),
        "p50_abs_delta": percentile(abs_deltas, 0.50),
        "p90_abs_delta": percentile(abs_deltas, 0.90),
        "p95_abs_delta": percentile(abs_deltas, 0.95),
        "p99_abs_delta": percentile(abs_deltas, 0.99),
        "max_abs_delta": max(abs_deltas) if abs_deltas else float("nan"),
        "clip_eps": clip_eps,
        "clip_lower_log_boundary": lower,
        "clip_upper_log_boundary": upper,
        "clip_crossing_fraction": sum(crossings) / len(crossings) if crossings else float("nan"),
        "mean_ratio_error": statistics.fmean(ratio_errors) if ratio_errors else float("nan"),
        "p99_ratio_error": percentile(ratio_errors, 0.99),
    }


def load_rows(parity_path: Path, generations_path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parity = json.loads(parity_path.read_text())
    generation_records = []
    if generations_path and generations_path.exists():
        generation_records = json.loads(generations_path.read_text()).get("records", [])

    rows = []
    sequence_rows = []
    for prompt_id, record in enumerate(parity["records"]):
        gen_record = generation_records[prompt_id] if prompt_id < len(generation_records) else {}
        gen_token_records = gen_record.get("token_records", [])
        token_texts = [x.get("token") for x in gen_token_records]
        deltas = []
        generated_prefix = ""
        for position, token_record in enumerate(record["token_records"]):
            token_text = token_texts[position] if position < len(token_texts) else None
            delta = float(token_record["delta"])
            deltas.append(delta)
            prefix_snippet = generated_prefix[-80:].replace("\n", "\\n")
            if token_text is not None:
                generated_prefix += token_text
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "position": position,
                    "token_id": token_record["token_id"],
                    "token_text": token_text,
                    "vllm_logprob": float(token_record["vllm_logprob"]),
                    "hf_logprob": float(token_record["hf_logprob"]),
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "prefix_snippet": prefix_snippet,
                    "prompt": record["prompt"],
                }
            )
        sequence_rows.append(
            {
                "prompt_id": prompt_id,
                "num_tokens": len(deltas),
                "sum_delta": sum(deltas),
                "sum_abs_delta": sum(abs(x) for x in deltas),
                "mean_abs_delta": statistics.fmean(abs(x) for x in deltas) if deltas else float("nan"),
                "text_preview": record.get("text", "")[:200].replace("\n", "\\n"),
            }
        )
    return rows, sequence_rows


def write_outliers(path: Path, rows: list[dict[str, Any]], n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "prompt_id",
        "position",
        "token_id",
        "token_text",
        "vllm_logprob",
        "hf_logprob",
        "delta",
        "abs_delta",
        "prefix_snippet",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(sorted(rows, key=lambda x: x["abs_delta"], reverse=True)[:n], start=1):
            writer.writerow({key: row.get(key) if key != "rank" else rank for key in fields})


def make_plots(rows: list[dict[str, Any]], figure_dir: Path, clip_eps: float) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    abs_deltas = sorted(row["abs_delta"] for row in rows)
    lower_abs = abs(math.log(1.0 - clip_eps))
    upper_abs = abs(math.log(1.0 + clip_eps))

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.hist(abs_deltas, bins=80, color="#3f6f8f", alpha=0.85)
    ax.axvline(upper_abs, color="#b44", linestyle="--", linewidth=1.2, label=r"$\log(1+\epsilon)$")
    ax.axvline(lower_abs, color="#7a3", linestyle="--", linewidth=1.2, label=r"$|\log(1-\epsilon)|$")
    ax.set_xlabel(r"$|\Delta \log p|$")
    ax.set_ylabel("Token count")
    ax.set_title("vLLM-HF absolute logprob deltas")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "vllm_hf_abs_delta_histogram.png", dpi=220)
    plt.close(fig)

    y = [(i + 1) / len(abs_deltas) for i in range(len(abs_deltas))]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(abs_deltas, y, linewidth=2, color="#3f6f8f")
    ax.axvline(upper_abs, color="#b44", linestyle="--", linewidth=1.2, label=r"$\log(1+\epsilon)$")
    ax.axvline(lower_abs, color="#7a3", linestyle="--", linewidth=1.2, label=r"$|\log(1-\epsilon)|$")
    ax.set_xlabel(r"$|\Delta \log p|$")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("vLLM-HF delta CDF")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "vllm_hf_abs_delta_cdf.png", dpi=220)
    plt.close(fig)


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    sequence_summary: dict[str, Any],
    outlier_path: Path,
    figure_dir: Path,
) -> None:
    lines = [
        "# vLLM-HF Backend Parity Diagnostics",
        "",
        "These diagnostics validate the static vLLM actor / HuggingFace learner parity result before using it as H1 evidence.",
        "",
        "## Token-Level Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Compared completion tokens | {summary['num_tokens']:,} |",
        f"| Mean delta | {summary['mean_delta']:.4f} |",
        f"| Mean absolute delta | {summary['mean_abs_delta']:.4f} |",
        f"| p95 absolute delta | {summary['p95_abs_delta']:.4f} |",
        f"| p99 absolute delta | {summary['p99_abs_delta']:.4f} |",
        f"| Max absolute delta | {summary['max_abs_delta']:.4f} |",
        f"| PPO clip-boundary crossing fraction | {summary['clip_crossing_fraction']:.4f} |",
        f"| p99 ratio error | {summary['p99_ratio_error']:.4f} |",
        "",
        "## Sequence-Level Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Sequences | {sequence_summary['num_sequences']} |",
        f"| Mean summed delta | {sequence_summary['mean_sum_delta']:.4f} |",
        f"| p95 absolute summed delta | {sequence_summary['p95_abs_sum_delta']:.4f} |",
        f"| Max absolute summed delta | {sequence_summary['max_abs_sum_delta']:.4f} |",
        "",
        "## Artifacts",
        "",
        f"- Outlier table: `{outlier_path}`",
        f"- Histogram: `{figure_dir / 'vllm_hf_abs_delta_histogram.png'}`",
        f"- CDF: `{figure_dir / 'vllm_hf_abs_delta_cdf.png'}`",
        "",
        "## Interpretation",
        "",
        "The current parity result is completion-token-only: the vLLM API generated responses and returned sampled-token logprobs, and HF recomputed logprobs on the same generated completion text. The large tail remains after token-text alignment checks, so the result is appropriate as static backend-parity evidence. It should not be described as full vLLM-based PPO training.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity_json", default="outputs/vllm_backend_probe/llama32_3b_instruct/vllm_hf_parity.json")
    parser.add_argument("--generations_json", default="outputs/vllm_backend_probe/llama32_3b_instruct/vllm_generations.json")
    parser.add_argument("--stats_json", default="outputs/logs/vllm_hf_parity_diagnostics.json")
    parser.add_argument("--outliers_csv", default="outputs/logs/vllm_hf_parity_outliers.csv")
    parser.add_argument("--figure_dir", default="outputs/figures")
    parser.add_argument("--markdown", default="docs/final/vllm_hf_backend_diagnostics.md")
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--top_n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, sequence_rows = load_rows(Path(args.parity_json), Path(args.generations_json))
    deltas = [row["delta"] for row in rows]
    summary = summarize_deltas(deltas, args.clip_eps)
    abs_sum_deltas = [abs(row["sum_delta"]) for row in sequence_rows]
    sequence_summary = {
        "num_sequences": len(sequence_rows),
        "mean_sum_delta": statistics.fmean(row["sum_delta"] for row in sequence_rows),
        "mean_abs_sum_delta": statistics.fmean(abs_sum_deltas),
        "p95_abs_sum_delta": percentile(abs_sum_deltas, 0.95),
        "p99_abs_sum_delta": percentile(abs_sum_deltas, 0.99),
        "max_abs_sum_delta": max(abs_sum_deltas),
    }
    out = {
        "token_summary": summary,
        "sequence_summary": sequence_summary,
        "top_outliers": sorted(rows, key=lambda x: x["abs_delta"], reverse=True)[: args.top_n],
    }
    Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    write_outliers(Path(args.outliers_csv), rows, args.top_n)
    make_plots(rows, Path(args.figure_dir), args.clip_eps)
    write_markdown(Path(args.markdown), summary, sequence_summary, Path(args.outliers_csv), Path(args.figure_dir))
    print(json.dumps({"token_summary": summary, "sequence_summary": sequence_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
