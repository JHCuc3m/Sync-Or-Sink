"""
Logprob Parity Test (Task 4)
Compares token-level log-probabilities between two model instances:
  - Model A: base GPT-2 (simulates inference/actor backend)
  - Model B: same weights, but reloaded independently (simulates training backend)

In practice the mismatch arises between vLLM and HuggingFace; here we
simulate it using dtype or minor config differences to quantify baseline
numerical drift.

Outputs:
  - outputs/figures/logprob_drift_histogram.png
  - outputs/logs/logprob_parity_stats.json
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_ID    = "gpt2"
NUM_SAMPLES = 200
MAX_LEN     = 128


def get_token_logprobs(model, tokenizer, texts, device, max_len=MAX_LEN):
    """Return per-token log-probs for each text as a flat numpy array."""
    all_logprobs = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding=False,
            ).to(device)

            input_ids = enc["input_ids"]  # (1, T)
            if input_ids.shape[1] < 2:
                continue

            outputs = model(**enc)
            logits  = outputs.logits  # (1, T, V)

            # Shift: predict token t+1 from position t
            shift_logits = logits[:, :-1, :]       # (1, T-1, V)
            shift_labels = input_ids[:, 1:]        # (1, T-1)

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_lp  = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            all_logprobs.append(token_lp.cpu().float().numpy().flatten())

    return np.concatenate(all_logprobs) if all_logprobs else np.array([])


def load_texts(num_samples=NUM_SAMPLES):
    """Load short text samples from HH-RLHF chosen responses."""
    ds = load_dataset("Anthropic/hh-rlhf", split="test")
    ds = ds.select(range(min(num_samples, len(ds))))
    texts = [ex["chosen"] for ex in ds]
    return texts


def plot_histogram(deltas, out_path, mean, std):
    plt.figure(figsize=(7, 4))
    plt.hist(deltas, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    plt.axvline(mean, color="red",    linestyle="--", linewidth=1.5, label=f"mean={mean:.4f}")
    plt.axvline(mean + std, color="orange", linestyle=":",  linewidth=1.2, label=f"±1σ={std:.4f}")
    plt.axvline(mean - std, color="orange", linestyle=":",  linewidth=1.2)
    plt.xlabel("Δ log-prob  (Model A − Model B)")
    plt.ylabel("Token count")
    plt.title("Logprob Parity: Δ between Two GPT-2 Instances\n(fp32 vs fp16 — backend mismatch simulation)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Histogram saved to {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Model A: fp32 (simulates HuggingFace training backend)
    print("Loading Model A (fp32)...")
    model_a = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)

    # Model B: fp16 (simulates vLLM / quantized inference backend)
    # This is the primary source of logprob mismatch in real pipelines.
    if device == "cuda":
        print("Loading Model B (fp16)...")
        model_b = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    else:
        # On CPU float16 is not supported; use a second fp32 load to show zero drift
        print("Loading Model B (fp32, CPU fallback)...")
        model_b = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)

    print(f"Loading {NUM_SAMPLES} text samples...")
    texts = load_texts(NUM_SAMPLES)

    print("Computing logprobs for Model A...")
    lp_a = get_token_logprobs(model_a, tokenizer, texts, device)

    print("Computing logprobs for Model B...")
    lp_b = get_token_logprobs(model_b, tokenizer, texts, device)

    # Align lengths (they should match but guard against truncation edge cases)
    n     = min(len(lp_a), len(lp_b))
    lp_a  = lp_a[:n]
    lp_b  = lp_b[:n]
    delta = lp_a - lp_b

    mean  = float(np.mean(delta))
    std   = float(np.std(delta))
    abs_mean = float(np.mean(np.abs(delta)))
    max_abs  = float(np.max(np.abs(delta)))

    print(f"\n=== Logprob Parity Stats ({n} tokens) ===")
    print(f"  Mean Δ:       {mean:.6f}")
    print(f"  Std  Δ:       {std:.6f}")
    print(f"  Mean |Δ|:     {abs_mean:.6f}")
    print(f"  Max  |Δ|:     {max_abs:.6f}")

    stats = {
        "model_a": f"{MODEL_ID} fp32",
        "model_b": f"{MODEL_ID} {'fp16' if device == 'cuda' else 'fp32'}",
        "num_tokens": n,
        "mean_delta": mean,
        "std_delta": std,
        "mean_abs_delta": abs_mean,
        "max_abs_delta": max_abs,
    }
    log_path = os.path.join(LOG_DIR, "logprob_parity_stats.json")
    with open(log_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {log_path}")

    fig_path = os.path.join(FIG_DIR, "logprob_drift_histogram.png")
    plot_histogram(delta, fig_path, mean, std)


if __name__ == "__main__":
    main()
