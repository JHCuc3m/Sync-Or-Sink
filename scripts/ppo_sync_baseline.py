import argparse
import copy
import json
import random
import re
from collections import deque
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from common import FIG_DIR, LOG_DIR, ensure_output_dirs, get_device, save_json, set_seed, write_config


DEFAULT_MODEL_ID = "gpt2"


@dataclass
class ArithmeticExample:
    prompt: str
    answer: int
    task_name: str
    expression: str
    parity: str = ""


def build_simple_dataset(num_samples: int, seed: int):
    rng = random.Random(seed)
    examples = []
    operations = ["+", "-"]
    for _ in range(num_samples):
        left = rng.randint(0, 20)
        right = rng.randint(0, 20)
        op = rng.choice(operations)
        answer = left + right if op == "+" else left - right
        prompt = (
            "Solve the arithmetic problem. "
            "Return only the final integer answer.\n"
            f"{left} {op} {right} ="
        )
        examples.append(
            ArithmeticExample(
                prompt=prompt,
                answer=answer,
                task_name="simple_arithmetic",
                expression=f"{left} {op} {right}",
            )
        )
    return examples


def sample_hard_expression(rng):
    profile = rng.choice(["carry_add", "borrow_sub", "three_term", "mixed_mul"])
    if profile == "carry_add":
        while True:
            left = rng.randint(15, 99)
            right = rng.randint(15, 99)
            if left % 10 + right % 10 >= 10:
                return f"{left} + {right}", left + right

    if profile == "borrow_sub":
        while True:
            left = rng.randint(40, 120)
            right = rng.randint(15, left)
            if left % 10 < right % 10:
                return f"{left} - {right}", left - right

    if profile == "three_term":
        first = rng.randint(10, 99)
        second = rng.randint(10, 99)
        third = rng.randint(10, 99)
        op1 = rng.choice(["+", "-"])
        op2 = rng.choice(["+", "-"])
        partial = first + second if op1 == "+" else first - second
        answer = partial + third if op2 == "+" else partial - third
        return f"({first} {op1} {second}) {op2} {third}", answer

    multiplier = rng.randint(2, 12)
    multiplicand = rng.randint(2, 12)
    offset = rng.randint(10, 99)
    op = rng.choice(["+", "-"])
    product = multiplier * multiplicand
    answer = product + offset if op == "+" else product - offset
    return f"({multiplier} * {multiplicand}) {op} {offset}", answer


def build_hard_dataset(num_samples: int, seed: int):
    rng = random.Random(seed)
    examples = []
    for _ in range(num_samples):
        expression, answer = sample_hard_expression(rng)
        prompt = (
            "Solve the arithmetic problem exactly. "
            "The expression may require carry, borrow, or multiple steps. "
            "Return only the final integer answer.\n"
            f"{expression} ="
        )
        examples.append(
            ArithmeticExample(
                prompt=prompt,
                answer=answer,
                task_name="hard_arithmetic",
                expression=expression,
            )
        )
    return examples


def build_structured_json_dataset(num_samples: int, seed: int):
    rng = random.Random(seed)
    examples = []
    for _ in range(num_samples):
        left = rng.randint(0, 60)
        right = rng.randint(0, 60)
        op = rng.choice(["+", "-"])
        answer = left + right if op == "+" else left - right
        parity = "even" if answer % 2 == 0 else "odd"
        expression = f"{left} {op} {right}"
        prompt = (
            "Return exactly one JSON object with keys \"answer\" and \"parity\". "
            "The answer value must be an integer and parity must be \"even\" or \"odd\". "
            "Do not include any extra text.\n"
            f"Compute: {expression}"
        )
        examples.append(
            ArithmeticExample(
                prompt=prompt,
                answer=answer,
                task_name="structured_json",
                expression=expression,
                parity=parity,
            )
        )
    return examples


def build_dataset(num_samples: int, seed: int, task_name: str):
    if task_name == "simple_arithmetic":
        return build_simple_dataset(num_samples, seed)
    if task_name == "hard_arithmetic":
        return build_hard_dataset(num_samples, seed)
    if task_name == "structured_json":
        return build_structured_json_dataset(num_samples, seed)
    raise ValueError(f"Unsupported task: {task_name}")


def parse_prediction(text: str):
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    return int(match.group(0))


def extract_json_object(text: str):
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, False
    candidate = stripped[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None, False
    return parsed, candidate == stripped


def structured_json_reward(prediction_text: str, target: int, target_parity: str):
    parsed, no_extra_text = extract_json_object(prediction_text)
    if not isinstance(parsed, dict):
        return -0.25, False

    reward = 0.4
    answer_ok = isinstance(parsed.get("answer"), int) and parsed["answer"] == target
    parity_ok = parsed.get("parity") == target_parity
    if answer_ok:
        reward += 0.3
    if parity_ok:
        reward += 0.2
    if no_extra_text:
        reward += 0.1
    return reward, bool(answer_ok and parity_ok and no_extra_text)


def reward_fn(prediction_text: str, target: int, task_name: str, target_parity: str = ""):
    if task_name == "structured_json":
        return structured_json_reward(prediction_text, target, target_parity)

    pred = parse_prediction(prediction_text)
    if pred is None:
        return -0.25, False
    if pred == target:
        return 1.0, True
    error = abs(pred - target)
    if task_name == "hard_arithmetic":
        relative_error = error / max(1, abs(target))
        same_sign = float((pred < 0) == (target < 0))
        same_last_digit = float(abs(pred) % 10 == abs(target) % 10)
        same_num_digits = float(len(str(abs(pred))) == len(str(abs(target))))
        shaped = 0.30 - 0.40 * relative_error
        shaped += 0.10 * same_last_digit + 0.05 * same_sign + 0.05 * same_num_digits
        return max(-0.25, min(0.55, shaped)), False
    shaped = max(0.0, 0.25 - 0.05 * error)
    return shaped, False


def resolve_torch_dtype(dtype_name: str, device: str):
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        if device != "cuda":
            return torch.float32
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )


def build_model_and_tokenizer(model_id: str, use_lora: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    policy = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    if use_lora:
        policy = get_peft_model(policy, get_lora_config())

    reference = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    return policy, reference, tokenizer


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def sync_actor_models(policy, actor_model, actor_rescore_model, actor_dtype, device):
    actor_model.load_state_dict(policy.state_dict())
    actor_model.to(device=device)
    actor_model.to(dtype=actor_dtype)
    freeze_model(actor_model)

    if actor_rescore_model is not None:
        actor_rescore_model.load_state_dict(policy.state_dict())
        actor_rescore_model.to(device=device, dtype=torch.float32)
        freeze_model(actor_rescore_model)


def generate_responses(model, tokenizer, prompts, device, max_prompt_tokens, max_new_tokens, temperature, top_p, do_sample):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    ).to(device)

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        generated = model.generate(
            **inputs,
            **generation_kwargs,
        )

    response_ids = generated[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(response_ids, skip_special_tokens=True)


def compute_response_stats(model, tokenizer, prompts, responses, device, max_seq_len):
    full_texts = [prompt + response for prompt, response in zip(prompts, responses)]
    prompt_enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )
    full_enc = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    ).to(device)

    prompt_lens = prompt_enc["attention_mask"].sum(dim=1).to(device)
    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].bool()

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

    seq_len = input_ids.shape[1]
    full_lens = attention_mask.sum(dim=1)
    pad_lens = seq_len - full_lens
    response_start = pad_lens + prompt_lens - 1
    positions = torch.arange(seq_len - 1, device=device).unsqueeze(0)
    response_mask = positions >= response_start.unsqueeze(1)
    response_mask &= shift_mask

    token_counts = response_mask.sum(dim=1).clamp_min(1)
    seq_logprobs = (token_logprobs * response_mask).sum(dim=1)
    mean_entropy = (token_entropy * response_mask).sum(dim=1) / token_counts
    return seq_logprobs, mean_entropy


def evaluate(policy, tokenizer, eval_examples, args, device):
    prompts = [ex.prompt for ex in eval_examples]
    responses = generate_responses(
        policy,
        tokenizer,
        prompts,
        device,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=False,
    )

    rewards = []
    successes = []
    for example, response in zip(eval_examples, responses):
        reward, success = reward_fn(response, example.answer, example.task_name, example.parity)
        rewards.append(reward)
        successes.append(float(success))

    return {
        "eval_reward_mean": sum(rewards) / len(rewards),
        "eval_pass_at_1": sum(successes) / len(successes),
        "samples": [
            {
                "prompt": eval_examples[idx].prompt,
                "response": responses[idx],
                "answer": eval_examples[idx].answer,
                "parity": eval_examples[idx].parity,
            }
            for idx in range(min(5, len(eval_examples)))
        ],
    }


def plot_training(log_history, out_path):
    steps = [entry["step"] for entry in log_history]
    rewards = [entry["reward_mean"] for entry in log_history]
    success = [entry["train_pass_rate"] for entry in log_history]
    clips = [entry["clip_fraction"] for entry in log_history]
    lag = [entry["lag_updates"] for entry in log_history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(steps, rewards, color="#4C9BE8", linewidth=2)
    axes[0, 0].set_title("Reward Mean")

    axes[0, 1].plot(steps, success, color="#3DAA6E", linewidth=2)
    axes[0, 1].set_title("Train Pass Rate")
    axes[0, 1].set_ylim(0.0, 1.0)

    axes[1, 0].plot(steps, clips, color="#E05C5C", linewidth=2)
    axes[1, 0].set_title("Clip Fraction")
    axes[1, 0].set_ylim(0.0, 1.0)

    axes[1, 1].plot(steps, lag, color="#7A5AF8", linewidth=2)
    axes[1, 1].set_title("Lag Updates")

    for ax in axes.flat:
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_rollout_batch(actor_model, actor_rescore_model, tokenizer, train_examples, rng, args, device, actor_version, actor_synced_step, learner_step):
    batch = rng.sample(train_examples, args.batch_size)
    prompts = [example.prompt for example in batch]
    answers = [example.answer for example in batch]

    responses = generate_responses(
        actor_model,
        tokenizer,
        prompts,
        device,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    rewards = []
    successes = []
    for example, answer, response in zip(batch, answers, responses):
        reward, success = reward_fn(response, answer, example.task_name, example.parity)
        rewards.append(reward)
        successes.append(float(success))

    actor_logprobs, _ = compute_response_stats(
        actor_model,
        tokenizer,
        prompts,
        responses,
        device,
        args.max_prompt_tokens + args.max_new_tokens,
    )

    if actor_rescore_model is not None:
        rescored_logprobs, _ = compute_response_stats(
            actor_rescore_model,
            tokenizer,
            prompts,
            responses,
            device,
            args.max_prompt_tokens + args.max_new_tokens,
        )
        logprob_delta = (actor_logprobs.detach() - rescored_logprobs.detach()).cpu()
        if args.rescore_old_logprobs:
            old_logprobs = rescored_logprobs.detach()
            old_logprob_source = "fp32_rescore"
        else:
            old_logprobs = actor_logprobs.detach()
            old_logprob_source = "actor_runtime"
    else:
        old_logprobs = actor_logprobs.detach()
        logprob_delta = torch.zeros_like(actor_logprobs.detach()).cpu()
        old_logprob_source = "actor_runtime"

    return {
        "prompts": prompts,
        "responses": responses,
        "answers": answers,
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
        "successes": successes,
        "old_logprobs": old_logprobs,
        "actor_logprobs": actor_logprobs.detach(),
        "actor_version": actor_version,
        "actor_synced_step": actor_synced_step,
        "rollout_step": learner_step,
        "lag_updates": learner_step - actor_synced_step,
        "logprob_delta_mean": float(logprob_delta.mean().item()),
        "logprob_delta_max_abs": float(logprob_delta.abs().max().item()),
        "old_logprob_source": old_logprob_source,
    }


def sample_training_batch(replay_buffer, sampling_mode, rng):
    if sampling_mode == "latest":
        return replay_buffer.pop()
    if sampling_mode == "oldest":
        return replay_buffer.popleft()
    if sampling_mode == "random":
        index = rng.randrange(len(replay_buffer))
        return replay_buffer.pop() if index == len(replay_buffer) - 1 else replay_buffer[index]
    raise ValueError(f"Unsupported buffer sampling mode: {sampling_mode}")


def remove_random_batch(replay_buffer, batch):
    for index, candidate in enumerate(replay_buffer):
        if candidate is batch:
            del replay_buffer[index]
            return


def maybe_adapt_clip(current_clip_eps, policy_kl, args):
    if not args.adaptive_clip:
        return current_clip_eps

    if policy_kl > args.target_policy_kl * 1.5:
        return max(args.min_clip_eps, current_clip_eps * 0.8)
    if policy_kl < args.target_policy_kl / 1.5:
        return min(args.max_clip_eps, current_clip_eps * 1.05)
    return current_clip_eps


def train_on_batch(policy, reference, tokenizer, train_batch, optimizer, scheduler, args, device, current_clip_eps):
    rewards = train_batch["rewards"]
    advantages = rewards - rewards.mean()
    if advantages.std(unbiased=False) > 1e-6:
        advantages = advantages / advantages.std(unbiased=False)

    prompts = train_batch["prompts"]
    responses = train_batch["responses"]
    old_logprobs = train_batch["old_logprobs"]

    with torch.no_grad():
        ref_logprobs, _ = compute_response_stats(
            reference,
            tokenizer,
            prompts,
            responses,
            device,
            args.max_prompt_tokens + args.max_new_tokens,
        )

    metrics = {
        "clip_fraction": 0.0,
        "importance_clipped_fraction": 0.0,
        "policy_approx_kl": 0.0,
        "ref_kl_proxy": 0.0,
        "entropy": 0.0,
    }

    for _ in range(args.ppo_epochs):
        current_logprobs, entropy = compute_response_stats(
            policy,
            tokenizer,
            prompts,
            responses,
            device,
            args.max_prompt_tokens + args.max_new_tokens,
        )

        log_ratio = current_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        if args.importance_cap is not None:
            ratio_for_loss = torch.clamp(ratio, max=args.importance_cap)
            importance_clipped_fraction = (ratio > args.importance_cap).float().mean().item()
        else:
            ratio_for_loss = ratio
            importance_clipped_fraction = 0.0

        unclipped = ratio_for_loss * advantages
        clipped = torch.clamp(
            ratio_for_loss,
            1.0 - current_clip_eps,
            1.0 + current_clip_eps,
        ) * advantages

        policy_loss = -torch.min(unclipped, clipped).mean()
        ref_kl = (current_logprobs - ref_logprobs).mean()
        entropy_bonus = entropy.mean()
        loss = policy_loss + args.kl_beta * ref_kl - args.entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        metrics = {
            "clip_fraction": (torch.abs(ratio - 1.0) > current_clip_eps).float().mean().item(),
            "importance_clipped_fraction": importance_clipped_fraction,
            "policy_approx_kl": (old_logprobs - current_logprobs).mean().item(),
            "ref_kl_proxy": ref_kl.item(),
            "entropy": entropy_bonus.item(),
        }

    return metrics


def main(args):
    ensure_output_dirs()
    set_seed(args.seed)
    device = get_device()
    actor_dtype = resolve_torch_dtype(args.actor_dtype, device)
    print(f"Using device: {device}")
    print(f"Actor dtype: {args.actor_dtype if actor_dtype != torch.float32 or args.actor_dtype == 'fp32' else 'fp32(cpu-fallback)'}")

    config_payload = vars(args).copy()
    config_payload["device"] = device
    config_payload["resolved_actor_dtype"] = str(actor_dtype)
    write_config(args.experiment_name, config_payload)

    train_examples = build_dataset(args.train_samples, args.seed, args.task_name)
    eval_examples = build_dataset(args.eval_samples, args.seed + 1, args.task_name)
    policy, reference, tokenizer = build_model_and_tokenizer(args.model_id, use_lora=not args.no_lora)
    policy.to(device)
    reference.to(device)
    freeze_model(reference)

    actor_model = copy.deepcopy(policy).to(device)
    need_actor_rescore = args.rescore_old_logprobs or actor_dtype != torch.float32
    actor_rescore_model = copy.deepcopy(policy).to(device) if need_actor_rescore else None
    sync_actor_models(policy, actor_model, actor_rescore_model, actor_dtype, device)

    optimizer = AdamW(policy.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, args.steps // 10),
        num_training_steps=args.steps * args.ppo_epochs,
    )

    trainable_params = sum(param.numel() for param in policy.parameters() if param.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    rng = random.Random(args.seed)
    log_history = []
    latest_eval = {}
    replay_buffer = deque(maxlen=args.replay_buffer_size)
    actor_version = 0
    actor_synced_step = 0
    current_clip_eps = args.clip_eps

    for step in range(1, args.steps + 1):
        for _ in range(args.rollouts_per_step):
            rollout_batch = build_rollout_batch(
                actor_model,
                actor_rescore_model,
                tokenizer,
                train_examples,
                rng,
                args,
                device,
                actor_version,
                actor_synced_step,
                step - 1,
            )
            replay_buffer.append(rollout_batch)

        if len(replay_buffer) < args.min_buffer_size:
            continue

        train_batch = sample_training_batch(replay_buffer, args.buffer_sampling, rng)
        if args.buffer_sampling == "random":
            remove_random_batch(replay_buffer, train_batch)

        learner_metrics = train_on_batch(
            policy,
            reference,
            tokenizer,
            train_batch,
            optimizer,
            scheduler,
            args,
            device,
            current_clip_eps,
        )
        current_clip_eps = maybe_adapt_clip(current_clip_eps, learner_metrics["policy_approx_kl"], args)

        if step % args.actor_refresh_interval == 0:
            actor_version += 1
            actor_synced_step = step
            sync_actor_models(policy, actor_model, actor_rescore_model, actor_dtype, device)

        log_entry = {
            "step": step,
            "reward_mean": float(train_batch["rewards"].mean().item()),
            "reward_std": float(train_batch["rewards"].std(unbiased=False).item()),
            "train_pass_rate": float(sum(train_batch["successes"]) / len(train_batch["successes"])),
            "clip_fraction": learner_metrics["clip_fraction"],
            "importance_clipped_fraction": learner_metrics["importance_clipped_fraction"],
            "policy_approx_kl": learner_metrics["policy_approx_kl"],
            "ref_kl_proxy": learner_metrics["ref_kl_proxy"],
            "entropy": learner_metrics["entropy"],
            "lag_updates": train_batch["lag_updates"],
            "actor_version": train_batch["actor_version"],
            "actor_synced_step": train_batch["actor_synced_step"],
            "replay_buffer_depth_after_pop": len(replay_buffer),
            "clip_eps_used": current_clip_eps,
            "old_logprob_source": train_batch["old_logprob_source"],
            "logprob_delta_mean": train_batch["logprob_delta_mean"],
            "logprob_delta_max_abs": train_batch["logprob_delta_max_abs"],
            "sample_prompt": train_batch["prompts"][0],
            "sample_response": train_batch["responses"][0],
            "sample_answer": train_batch["answers"][0],
            "task_name": args.task_name,
        }

        if step % args.eval_every == 0 or step == args.steps:
            latest_eval = evaluate(policy, tokenizer, eval_examples, args, device)
            log_entry.update(latest_eval)

        log_history.append(log_entry)
        print(
            f"step={step:03d} reward={log_entry['reward_mean']:.3f} "
            f"pass={log_entry['train_pass_rate']:.3f} clip={log_entry['clip_fraction']:.3f} "
            f"policy_kl={log_entry['policy_approx_kl']:.3f} lag={log_entry['lag_updates']} "
            f"delta={log_entry['logprob_delta_mean']:.4f}"
        )

    save_json(LOG_DIR / f"{args.experiment_name}_logs.json", log_history)

    summary = {
        "final_step": args.steps,
        "final_train_reward": log_history[-1]["reward_mean"] if log_history else None,
        "final_train_pass_rate": log_history[-1]["train_pass_rate"] if log_history else None,
        "final_eval_reward": latest_eval.get("eval_reward_mean"),
        "final_eval_pass_at_1": latest_eval.get("eval_pass_at_1"),
        "trainable_params": trainable_params,
        "actor_refresh_interval": args.actor_refresh_interval,
        "actor_dtype": args.actor_dtype,
        "replay_buffer_size": args.replay_buffer_size,
        "task_name": args.task_name,
        "rescore_old_logprobs": args.rescore_old_logprobs,
        "adaptive_clip": args.adaptive_clip,
        "importance_cap": args.importance_cap,
    }
    save_json(LOG_DIR / f"{args.experiment_name}_summary.json", summary)
    plot_training(log_history, FIG_DIR / f"{args.experiment_name}_training_curves.png")

    print("\n=== PPO Live Baseline Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        help="HF model id")
    parser.add_argument("--experiment_name", type=str, default="ppo_sync_baseline",
                        help="Prefix for config, logs, summary, and figures")
    parser.add_argument("--task_name", type=str, default="simple_arithmetic",
                        choices=["simple_arithmetic", "hard_arithmetic", "structured_json"],
                        help="Synthetic verifiable task to train on")
    parser.add_argument("--train_samples", type=int, default=256,
                        help="Synthetic train prompt count")
    parser.add_argument("--eval_samples", type=int, default=64,
                        help="Synthetic eval prompt count")
    parser.add_argument("--steps", type=int, default=60,
                        help="Number of learner updates")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Rollout batch size")
    parser.add_argument("--rollouts_per_step", type=int, default=1,
                        help="Rollout batches collected before each learner update")
    parser.add_argument("--min_buffer_size", type=int, default=1,
                        help="Minimum buffer size required before training starts")
    parser.add_argument("--replay_buffer_size", type=int, default=1,
                        help="Replay buffer capacity")
    parser.add_argument("--buffer_sampling", type=str, default="latest",
                        choices=["latest", "oldest", "random"],
                        help="How to choose a batch from the replay buffer")
    parser.add_argument("--actor_refresh_interval", type=int, default=1,
                        help="Sync actor weights from learner every K learner updates")
    parser.add_argument("--actor_dtype", type=str, default="fp32",
                        choices=["fp32", "fp16"],
                        help="Actor rollout dtype")
    parser.add_argument("--rescore_old_logprobs", action="store_true",
                        help="Mitigation: recompute actor old logprobs in fp32 before PPO updates")
    parser.add_argument("--ppo_epochs", type=int, default=2,
                        help="Optimization passes per batch")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Optimizer learning rate")
    parser.add_argument("--clip_eps", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--adaptive_clip", action="store_true",
                        help="Mitigation: adapt clip epsilon using observed policy KL")
    parser.add_argument("--target_policy_kl", type=float, default=0.02,
                        help="Target policy KL for adaptive clipping")
    parser.add_argument("--min_clip_eps", type=float, default=0.05,
                        help="Lower bound for adaptive clip epsilon")
    parser.add_argument("--max_clip_eps", type=float, default=0.3,
                        help="Upper bound for adaptive clip epsilon")
    parser.add_argument("--importance_cap", type=float, default=None,
                        help="Mitigation: cap importance ratios before PPO loss")
    parser.add_argument("--kl_beta", type=float, default=0.02,
                        help="Reference KL penalty weight")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy bonus coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--max_prompt_tokens", type=int, default=48,
                        help="Maximum prompt length")
    parser.add_argument("--max_new_tokens", type=int, default=8,
                        help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling")
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Evaluation interval")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA and fine-tune the full model")
    main(parser.parse_args())
