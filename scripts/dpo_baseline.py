"""
DPO Baseline Training Script
Task 3: Baseline Experiment for Sync-or-Sink milestone.

Uses GPT-2 + Anthropic HH-RLHF dataset for a minimal DPO run.
Saves training logs and plots loss curve.
"""

import os
import json
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_ID  = "gpt2"
DATASET   = "Anthropic/hh-rlhf"
MAX_LEN   = 512
BETA      = 0.1     # DPO temperature
EPOCHS    = 1
LR        = 1e-4
BATCH     = 4
GRAD_ACC  = 4
MAX_STEPS = 200     # cap for milestone speed


def load_hh_rlhf(split="train", num_samples=2000):
    """Load HH-RLHF and reformat to DPO format: prompt / chosen / rejected."""
    ds = load_dataset(DATASET, split=split)
    ds = ds.select(range(min(num_samples, len(ds))))

    def reformat(example):
        # HH-RLHF has 'chosen' and 'rejected' as full conversation strings.
        # DPO trainer expects: prompt, chosen, rejected.
        # We split on the last '\n\nAssistant:' to get prompt vs response.
        chosen   = example["chosen"]
        rejected = example["rejected"]

        # Find the split point (last assistant turn marker)
        marker = "\n\nAssistant:"
        split_idx = chosen.rfind(marker)
        if split_idx == -1:
            prompt   = ""
            chosen_r = chosen
        else:
            prompt   = chosen[:split_idx + len(marker)]
            chosen_r = chosen[split_idx + len(marker):]

        split_idx2 = rejected.rfind(marker)
        rejected_r = rejected[split_idx2 + len(marker):] if split_idx2 != -1 else rejected

        return {"prompt": prompt, "chosen": chosen_r, "rejected": rejected_r}

    ds = ds.map(reformat, remove_columns=ds.column_names)
    # Drop rows where any field is empty
    ds = ds.filter(lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0)
    return ds


def build_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    return model, tokenizer


def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],   # GPT-2 attention projection
    )


def plot_loss(log_history, out_path):
    steps, losses = [], []
    for entry in log_history:
        if "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

    if not losses:
        print("No loss values found in log history.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("DPO Loss")
    plt.title("DPO Baseline — Training Loss (GPT-2, HH-RLHF)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {out_path}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading dataset...")
    train_ds = load_hh_rlhf("train", num_samples=args.num_samples)
    eval_ds  = load_hh_rlhf("test",  num_samples=200)
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}")

    print("Building model...")
    model, tokenizer = build_model_and_tokenizer(MODEL_ID)

    training_args = DPOConfig(
        output_dir=os.path.join(LOG_DIR, "dpo_baseline"),
        num_train_epochs=EPOCHS,
        max_steps=args.max_steps,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        beta=BETA,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and device == "cuda",
        report_to="none",
        max_length=MAX_LEN,
        max_prompt_length=MAX_LEN // 2,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=get_lora_config(),
    )

    print("Starting DPO training...")
    train_result = trainer.train()

    # Save logs
    log_path = os.path.join(LOG_DIR, "dpo_baseline_logs.json")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Logs saved to {log_path}")

    # Plot
    fig_path = os.path.join(FIG_DIR, "dpo_loss_curve.png")
    plot_loss(trainer.state.log_history, fig_path)

    # Summary
    print("\n=== DPO Baseline Summary ===")
    print(f"  Steps run:     {train_result.global_step}")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    print(f"  Log file:      {log_path}")
    print(f"  Loss plot:     {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Number of training samples to use")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS,
                        help="Max training steps (set 0 to use full epochs)")
    args = parser.parse_args()
    if args.max_steps == 0:
        args.max_steps = -1
    main(args)
