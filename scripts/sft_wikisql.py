"""
SFT Warm-Start for WikiSQL Text-to-SQL

Fine-tunes GPT-2 on (prompt → gold SQL) pairs from WikiSQL to provide
a warm-start checkpoint for PPO training. Without SFT, PPO struggles to
discover valid SQL syntax from sparse execution reward.

Usage:
    python scripts/sft_wikisql.py --max_steps 500 --eval_every 100

Outputs:
    - outputs/checkpoints/wikisql_sft/checkpoint.pt (model state dict)
    - outputs/checkpoints/wikisql_sft/config.json (training config)
    - outputs/logs/sft_wikisql_logs.json (training logs)
    - outputs/figures/sft_wikisql_loss.png (loss curve)
"""

import argparse
import json
import os
import sys
import random

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from wikisql_utils import WikiSQLDataset, TABLE_NAME


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints", "wikisql_sft")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

MODEL_ID = "gpt2"


class SQLDataset(Dataset):
    """PyTorch Dataset for SFT on WikiSQL."""

    def __init__(self, examples, tokenizer, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        # Combine prompt and gold SQL as the training sequence
        # Format: <prompt>SELECT col0 FROM t WHERE...
        full_text = example.prompt + " " + example.gold_sql

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # For causal LM, labels = input_ids (shifted internally by the model)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, tokenizer, eval_examples, device, max_new_tokens=64, num_samples=50):
    """
    Evaluate model on SQL generation.

    Returns:
        dict with valid_sql_rate, exact_match_rate, sample outputs
    """
    model.eval()
    valid_count = 0
    exact_count = 0
    samples = []

    eval_subset = eval_examples[:num_samples]

    with torch.no_grad():
        for example in eval_subset:
            # Encode prompt
            inputs = tokenizer(
                example.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=192,
            ).to(device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for eval
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode only the generated part
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Extract SQL (take first line, remove trailing content)
            generated_sql = generated_text.split("\n")[0].strip()
            if generated_sql.endswith(";"):
                generated_sql = generated_sql[:-1]

            # Check validity (basic check: starts with SELECT, contains FROM)
            is_valid = (
                generated_sql.upper().startswith("SELECT") and
                "FROM" in generated_sql.upper()
            )
            if is_valid:
                valid_count += 1

            # Check exact match (normalized)
            gold_normalized = example.gold_sql.lower().strip()
            gen_normalized = generated_sql.lower().strip()
            is_exact = gold_normalized == gen_normalized
            if is_exact:
                exact_count += 1

            if len(samples) < 5:
                samples.append({
                    "question": example.question,
                    "gold_sql": example.gold_sql,
                    "generated_sql": generated_sql,
                    "is_valid": is_valid,
                    "is_exact": is_exact,
                })

    return {
        "valid_sql_rate": valid_count / len(eval_subset),
        "exact_match_rate": exact_count / len(eval_subset),
        "num_evaluated": len(eval_subset),
        "samples": samples,
    }


def plot_training_curves(logs, out_path):
    """Plot training loss and validation metrics."""
    steps = [log["step"] for log in logs]
    losses = [log["loss"] for log in logs]

    # Find eval points
    eval_logs = [log for log in logs if "valid_sql_rate" in log]
    eval_steps = [log["step"] for log in eval_logs]
    valid_rates = [log["valid_sql_rate"] for log in eval_logs]
    exact_rates = [log["exact_match_rate"] for log in eval_logs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(steps, losses, linewidth=1.5, alpha=0.7)
    # Smoothed loss
    if len(losses) > 10:
        window = min(20, len(losses) // 5)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[0].plot(steps[window-1:], smoothed, linewidth=2, color='red', label='Smoothed')
        axes[0].legend()
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("SFT Training Loss")
    axes[0].grid(True, alpha=0.3)

    # Validation metrics
    if eval_steps:
        axes[1].plot(eval_steps, [r * 100 for r in valid_rates], 'o-',
                     linewidth=2, markersize=6, label="Valid SQL %")
        axes[1].plot(eval_steps, [r * 100 for r in exact_rates], 's-',
                     linewidth=2, markersize=6, label="Exact Match %")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Rate (%)")
        axes[1].set_title("Validation Metrics")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 105)

    fig.suptitle("WikiSQL SFT Training", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {out_path}")


def main(args):
    set_seed(args.seed)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: max_steps={args.max_steps}, batch_size={args.batch_size}, lr={args.lr}")

    # Load tokenizer and model
    print(f"\nLoading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
    ).to(device)

    # Load WikiSQL data
    print(f"\nLoading WikiSQL dataset...")
    train_dataset = WikiSQLDataset(
        split="train",
        max_examples=args.num_train_examples,
        seed=args.seed,
        few_shot=True,
        show_col_mapping=True,
    )
    print(f"Train examples: {len(train_dataset)}")

    eval_dataset = WikiSQLDataset(
        split="dev",
        max_examples=args.num_eval_examples,
        seed=args.seed + 1000,
        few_shot=True,
        show_col_mapping=True,
    )
    print(f"Eval examples: {len(eval_dataset)}")

    # Create PyTorch dataset and dataloader
    train_data = SQLDataset(train_dataset.examples, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    train_iter = iter(train_loader)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # Training loop
    print(f"\nStarting SFT training for {args.max_steps} steps...")
    logs = []
    model.train()

    for step in range(1, args.max_steps + 1):
        # Get batch (cycle through data)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Log
        log_entry = {
            "step": step,
            "loss": float(loss.item()),
            "lr": float(scheduler.get_last_lr()[0]),
        }

        # Evaluate periodically
        if step % args.eval_every == 0 or step == 1:
            eval_results = evaluate_model(
                model, tokenizer, eval_dataset.examples, device,
                max_new_tokens=args.max_new_tokens,
                num_samples=args.eval_samples,
            )
            log_entry.update(eval_results)

            print(
                f"Step {step:4d}/{args.max_steps} | "
                f"loss={loss.item():.4f} | "
                f"valid_sql={eval_results['valid_sql_rate']*100:.1f}% | "
                f"exact={eval_results['exact_match_rate']*100:.1f}%"
            )

            # Print sample generations
            if args.verbose and eval_results["samples"]:
                print("  Sample generations:")
                for i, sample in enumerate(eval_results["samples"][:2]):
                    print(f"    [{i+1}] Q: {sample['question'][:60]}...")
                    print(f"        Gold: {sample['gold_sql']}")
                    print(f"        Gen:  {sample['generated_sql']}")
                    print(f"        Valid={sample['is_valid']}, Exact={sample['is_exact']}")

        elif step % args.log_every == 0:
            print(f"Step {step:4d}/{args.max_steps} | loss={loss.item():.4f}")

        logs.append(log_entry)
        model.train()

    # Save checkpoint
    checkpoint_path = os.path.join(CKPT_DIR, "checkpoint.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")

    # Save config
    config = {
        "model_id": args.model_id,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "num_train_examples": args.num_train_examples,
        "seed": args.seed,
        "table_name": TABLE_NAME,
    }
    config_path = os.path.join(CKPT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # Save logs
    log_path = os.path.join(LOG_DIR, "sft_wikisql_logs.json")
    with open(log_path, "w") as f:
        json.dump({"config": config, "logs": logs}, f, indent=2)
    print(f"Logs saved to {log_path}")

    # Plot training curves
    plot_path = os.path.join(FIG_DIR, "sft_wikisql_loss.png")
    plot_training_curves(logs, plot_path)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_eval = evaluate_model(
        model, tokenizer, eval_dataset.examples, device,
        max_new_tokens=args.max_new_tokens,
        num_samples=min(100, len(eval_dataset)),
    )
    print(f"Valid SQL rate: {final_eval['valid_sql_rate']*100:.1f}%")
    print(f"Exact match rate: {final_eval['exact_match_rate']*100:.1f}%")

    print("\nSample outputs:")
    for i, sample in enumerate(final_eval["samples"]):
        print(f"\n[{i+1}] Question: {sample['question']}")
        print(f"    Gold SQL: {sample['gold_sql']}")
        print(f"    Generated: {sample['generated_sql']}")
        print(f"    Valid={sample['is_valid']}, Exact={sample['is_exact']}")

    print("\n=== SFT Training Complete ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Use with: python scripts/live_ppo_rlvr.py --checkpoint {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT warm-start on WikiSQL")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_train_examples", type=int, default=2000)
    parser.add_argument("--num_eval_examples", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
