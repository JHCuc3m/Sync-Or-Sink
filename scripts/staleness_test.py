"""
Simulated Staleness Test (Task 5)

Simulates policy lag L by:
  1. Fine-tuning GPT-2 with SFT on HH-RLHF chosen responses for TOTAL_STEPS,
     saving checkpoints at steps 0, 25, 50, 75, 100.
  2. For each (checkpoint_L, final_policy) pair, computing token-level PPO ratios:
       r_t = exp(log pi_final(a_t|s_t) - log pi_L(a_t|s_t))
  3. Measuring: ratio variance, approx KL, clip fraction vs lag L.

Outputs:
  - outputs/figures/staleness_ratio_variance.png
  - outputs/figures/staleness_kl_clip.png
  - outputs/logs/staleness_stats.json
"""

import os
import copy
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common import CKPT_DIR, FIG_DIR, LOG_DIR, ensure_output_dirs, get_device, save_json, set_seed, write_config

MODEL_ID     = "gpt2"
TOTAL_STEPS  = 100          # total SFT steps
SAVE_STEPS   = {0, 25, 50, 75, 100}   # checkpoints saved during training
CLIP_EPS     = 0.2          # PPO clip threshold
MAX_LEN      = 128
BATCH        = 4
LR           = 2e-5
NUM_EVAL     = 200          # tokens to evaluate ratios on


# ── Data ───────────────────────────────────────────────────────────────────────

def load_sft_data(num_samples=1000):
    """SFT on chosen responses from HH-RLHF."""
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    ds = ds.select(range(num_samples))
    return [ex["chosen"] for ex in ds]


def tokenize_texts(texts, tokenizer, max_len=MAX_LEN):
    out = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=max_len, padding="max_length"
    )
    return out


# ── SFT training loop ──────────────────────────────────────────────────────────

def sft_step(model, batch, device):
    """One SFT gradient step (next-token prediction loss)."""
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss    = outputs.loss
    return loss


# ── Logprob computation ────────────────────────────────────────────────────────

def get_token_logprobs(model, tokenizer, texts, device, max_len=MAX_LEN):
    """Return flat array of per-token log-probs."""
    model.eval()
    all_lp = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_len, padding=False).to(device)
            ids = enc["input_ids"]
            if ids.shape[1] < 2:
                continue
            logits    = model(**enc).logits                          # (1, T, V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            shift_lp  = log_probs[:, :-1, :]                        # (1, T-1, V)
            shift_ids = ids[:, 1:]                                   # (1, T-1)
            token_lp  = shift_lp.gather(2, shift_ids.unsqueeze(-1)).squeeze(-1)
            all_lp.append(token_lp.cpu().float().numpy().flatten())
    return np.concatenate(all_lp) if all_lp else np.array([])


# ── Ratio statistics ───────────────────────────────────────────────────────────

def compute_ratio_stats(lp_current, lp_old, eps=CLIP_EPS):
    """Given two log-prob arrays, compute PPO ratio stats."""
    log_ratio = lp_current - lp_old
    ratios     = np.exp(log_ratio)

    variance      = float(np.var(ratios))
    approx_kl     = float(np.mean(log_ratio))       # E[log r] ≈ KL (first-order)
    clip_fraction = float(np.mean(np.abs(ratios - 1.0) > eps))
    mean_ratio    = float(np.mean(ratios))

    return dict(variance=variance, approx_kl=approx_kl,
                clip_fraction=clip_fraction, mean_ratio=mean_ratio)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_ratio_variance(lag_values, variances, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(lag_values, variances, "o-", linewidth=2, markersize=7, color="#E05C5C")
    plt.xlabel("Policy lag L (gradient updates)")
    plt.ylabel("Ratio variance  Var(r_t)")
    plt.title("PPO Ratio Variance vs. Staleness Lag\n(GPT-2 SFT, HH-RLHF)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Ratio variance plot saved to {out_path}")


def plot_kl_clip(lag_values, kls, clips, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(lag_values, kls, "s-", linewidth=2, markersize=7, color="#4C9BE8")
    ax1.set_xlabel("Policy lag L (gradient updates)")
    ax1.set_ylabel("Approx KL  E[log r]")
    ax1.set_title("Approx KL vs. Staleness Lag")
    ax1.grid(True, alpha=0.3)

    ax2.plot(lag_values, [c * 100 for c in clips], "^-",
             linewidth=2, markersize=7, color="#3DAA6E")
    ax2.axhline(y=20, color="red", linestyle="--", linewidth=1.2,
                label="20% clip threshold")
    ax2.set_xlabel("Policy lag L (gradient updates)")
    ax2.set_ylabel("Clip fraction (%)")
    ax2.set_title("PPO Clip Fraction vs. Staleness Lag")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Staleness Effects on PPO Stability (GPT-2 SFT, ε=0.2)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"KL/clip plot saved to {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    ensure_output_dirs()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def ckpt_path(step):
        return os.path.join(str(CKPT_DIR), f"step_{step:04d}.pt")

    save_steps = {0, *args.save_steps}
    all_saved = all(os.path.exists(ckpt_path(s)) for s in save_steps)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32).to(device)

    config_payload = vars(args).copy()
    config_payload["device"] = device
    write_config("staleness_test", config_payload)

    if all_saved:
        # ── Fast path: load from disk, skip SFT ───────────────────────────────
        print("All checkpoints found on disk — skipping SFT training.")
        checkpoints = {}
        for step in save_steps:
            checkpoints[step] = torch.load(ckpt_path(step), map_location=device)
            print(f"  Loaded checkpoint step {step} from {ckpt_path(step)}")
        final_state = checkpoints[args.total_steps]
    else:
        # ── SFT training ──────────────────────────────────────────────────────
        checkpoints = {}

        # Step 0 = base model
        checkpoints[0] = copy.deepcopy(model.state_dict())
        torch.save(checkpoints[0], ckpt_path(0))
        print(f"Checkpoint saved: step 0 → {ckpt_path(0)}")

        print("Loading SFT data...")
        texts = load_sft_data(num_samples=args.num_train_samples)
        enc   = tokenize_texts(texts, tokenizer, max_len=args.max_len)

        dataset = torch.utils.data.TensorDataset(
            enc["input_ids"], enc["attention_mask"]
        )
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)
        loader_iter = iter(loader)

        optimizer = AdamW(model.parameters(), lr=LR)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=10, num_training_steps=args.total_steps
        )

        print(f"Starting SFT training for {args.total_steps} steps...")
        model.train()
        for step in range(1, args.total_steps + 1):
            try:
                batch_ids, batch_mask = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch_ids, batch_mask = next(loader_iter)

            batch = {"input_ids": batch_ids, "attention_mask": batch_mask}
            loss  = sft_step(model, batch, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step in save_steps and step > 0:
                checkpoints[step] = copy.deepcopy(model.state_dict())
                torch.save(checkpoints[step], ckpt_path(step))
                print(f"  Checkpoint saved: step {step} → {ckpt_path(step)}  "
                      f"(loss={loss.item():.4f})")

            if step % 25 == 0:
                print(f"  Step {step}/{args.total_steps}  loss={loss.item():.4f}")

        # Final model = "current policy" (step total_steps)
        final_state = copy.deepcopy(model.state_dict())
        checkpoints[args.total_steps] = final_state

    # ── Evaluation texts ──────────────────────────────────────────────────────
    print("\nLoading evaluation texts...")
    eval_texts = load_dataset("Anthropic/hh-rlhf", split="test")
    eval_texts = [ex["chosen"] for ex in eval_texts.select(range(args.num_eval_samples))]

    # ── Compute logprobs for final policy ─────────────────────────────────────
    model.load_state_dict(final_state)
    model.eval()
    print(f"Computing logprobs for final policy (step {args.total_steps})...")
    lp_final = get_token_logprobs(model, tokenizer, eval_texts, device, max_len=args.max_len)

    # ── Compute stats per lag ─────────────────────────────────────────────────
    ckpt_for_lag = {lag: args.total_steps - lag for lag in sorted(args.lags)}

    all_stats = []
    lag_plot, var_plot, kl_plot, clip_plot = [], [], [], []

    for lag, ckpt_step in sorted(ckpt_for_lag.items()):
        if ckpt_step not in checkpoints:
            raise ValueError(f"Missing checkpoint for lag {lag}: expected step {ckpt_step}.")
        model.load_state_dict(checkpoints[ckpt_step])
        print(f"Computing logprobs for lag L={lag} (checkpoint step {ckpt_step})...")
        lp_old = get_token_logprobs(model, tokenizer, eval_texts, device, max_len=args.max_len)

        n = min(len(lp_final), len(lp_old))
        stats = compute_ratio_stats(lp_final[:n], lp_old[:n], eps=args.clip_eps)
        stats["lag"]        = lag
        stats["ckpt_step"]  = ckpt_step
        stats["num_tokens"] = n
        all_stats.append(stats)

        print(f"  lag={lag}  var={stats['variance']:.5f}  "
              f"KL={stats['approx_kl']:.5f}  clip={stats['clip_fraction']*100:.1f}%")

        lag_plot.append(lag)
        var_plot.append(stats["variance"])
        kl_plot.append(stats["approx_kl"])
        clip_plot.append(stats["clip_fraction"])

    # ── Save & plot ───────────────────────────────────────────────────────────
    log_path = os.path.join(str(LOG_DIR), "staleness_stats.json")
    save_json(LOG_DIR / "staleness_stats.json", all_stats)
    print(f"\nStats saved to {log_path}")

    plot_ratio_variance(lag_plot, var_plot,
                        os.path.join(str(FIG_DIR), "staleness_ratio_variance.png"))
    plot_kl_clip(lag_plot, kl_plot, clip_plot,
                 os.path.join(str(FIG_DIR), "staleness_kl_clip.png"))

    print("\n=== Staleness Test Complete ===")
    for s in all_stats:
        print(f"  L={s['lag']:3d}  var={s['variance']:.5f}  "
              f"KL={s['approx_kl']:.5f}  clip={s['clip_fraction']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=MODEL_ID,
                        help="HF model id")
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS,
                        help="Total SFT steps")
    parser.add_argument("--save_steps", type=int, nargs="+", default=[25, 50, 75, 100],
                        help="Checkpoint steps to save; step 0 is always included")
    parser.add_argument("--lags", type=int, nargs="+", default=[0, 25, 50, 75, 100],
                        help="Lag values to evaluate; each lag must map to a saved checkpoint")
    parser.add_argument("--clip_eps", type=float, default=CLIP_EPS,
                        help="PPO clipping threshold used for the report metric")
    parser.add_argument("--max_len", type=int, default=MAX_LEN,
                        help="Maximum tokens per sample")
    parser.add_argument("--num_train_samples", type=int, default=1000,
                        help="Training samples for the SFT proxy run")
    parser.add_argument("--num_eval_samples", type=int, default=NUM_EVAL,
                        help="Evaluation samples for ratio computation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    main(parser.parse_args())
