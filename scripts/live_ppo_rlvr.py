"""
Live PPO/RLVR harness for structured JSON extraction tasks.

This script is intentionally self-contained so we can run controlled PPO drift
experiments without first building a full distributed actor/learner stack.

Supported conditions:
  - sync: actor and learner use the same fp32 policy snapshot.
  - stale: actor generates/scored rollouts from a lagged policy snapshot.
  - mismatch: actor old logprobs are computed in fp16, learner updates in fp32.
  - rescored: fp16 actor generates rollouts, but old logprobs are rescored in fp32.
  - stale_mismatch: stale actor plus fp16 actor-side old logprobs.

Outputs:
  - outputs/live_ppo/<run_name>/logs.json
  - outputs/live_ppo/<run_name>/metrics.png
"""

import argparse
import copy
import json
import os
import random
from collections import deque
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")
DEFAULT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "live_ppo")

MODEL_ID = "gpt2"
CLIP_EPS = 0.2
DEFAULT_MAX_NEW_TOKENS = 48


@dataclass
class JsonExample:
    prompt: str
    target: dict
    target_json: str
    schema: str


PERSON_NAMES = [
    "Alice",
    "Ben",
    "Carla",
    "Derek",
    "Elena",
    "Farah",
    "Grace",
    "Hector",
]
PERSON_CITIES = [
    "Boston",
    "Atlanta",
    "Seattle",
    "Chicago",
    "Denver",
    "Austin",
    "Miami",
    "Phoenix",
]
PRODUCTS = [
    ("notebook", "stationery"),
    ("headphones", "electronics"),
    ("mug", "kitchen"),
    ("backpack", "travel"),
    ("lamp", "home"),
]
EVENTS = [
    ("seminar", "Monday", "Room 101"),
    ("workshop", "Tuesday", "Lab B"),
    ("meeting", "Friday", "Conference Hall"),
    ("demo", "Thursday", "Studio 3"),
]


def compact_json(payload):
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def build_prompt(text, keys, few_shot=True):
    key_str = ", ".join(keys)
    prefix = (
        "You are an extraction system.\n"
        "Return exactly one compact JSON object with exactly the requested keys, then stop.\n"
    )
    if few_shot:
        prefix += (
            "Example input:\nName: Alice. Age: 24. City: Boston.\n"
            "Example keys: name, age, city\n"
            'Example output:\n{"name":"Alice","age":24,"city":"Boston"}\n\n'
        )
    return f"{prefix}Now extract:\n{text}\nKeys: {key_str}\nOutput:"


def make_person_example(level, few_shot=True):
    name = random.choice(PERSON_NAMES)
    age = random.randint(18, 67)
    city = random.choice(PERSON_CITIES)
    target = {"name": name, "age": age, "city": city}

    if level == "easy_json":
        text = f"Name: {name}. Age: {age}. City: {city}."
    else:
        templates = [
            f"{name} is {age} years old and lives in {city}.",
            f"City: {city}. The person's name is {name}. Age: {age}.",
            f"Profile record - age {age}; city {city}; name {name}.",
            f"{name}, age {age}, recently moved to {city}.",
        ]
        text = random.choice(templates)

    keys = ["name", "age", "city"]
    return JsonExample(
        prompt=build_prompt(text, keys, few_shot=few_shot),
        target=target,
        target_json=compact_json(target),
        schema="person",
    )


def make_product_example(few_shot=True):
    item, category = random.choice(PRODUCTS)
    price = random.randint(5, 90)
    stock = random.randint(0, 40)
    target = {"item": item, "category": category, "price": price, "stock": stock}
    templates = [
        f"Product: {item}. Category: {category}. Price: ${price}. Stock: {stock}.",
        f"The {category} item is a {item}; it costs ${price} and has {stock} units.",
        f"Inventory says {stock} {item} units remain in {category}, price {price} dollars.",
    ]
    keys = ["item", "category", "price", "stock"]
    return JsonExample(
        prompt=build_prompt(random.choice(templates), keys, few_shot=few_shot),
        target=target,
        target_json=compact_json(target),
        schema="product",
    )


def make_event_example(few_shot=True):
    title, day, location = random.choice(EVENTS)
    target = {"event": title, "day": day, "location": location}
    templates = [
        f"Event: {title}. Day: {day}. Location: {location}.",
        f"The {title} will happen on {day} in {location}.",
        f"Schedule update - location {location}; event {title}; day {day}.",
    ]
    keys = ["event", "day", "location"]
    return JsonExample(
        prompt=build_prompt(random.choice(templates), keys, few_shot=few_shot),
        target=target,
        target_json=compact_json(target),
        schema="event",
    )


def sample_json_example(task, few_shot=True):
    if task == "easy_json":
        return make_person_example("easy_json", few_shot=few_shot)
    if task == "medium_json":
        return make_person_example("medium_json", few_shot=few_shot)
    if task == "mixed_json":
        maker = random.choice([make_person_example, make_product_example, make_event_example])
        if maker == make_person_example:
            return maker("medium_json", few_shot=few_shot)
        return maker(few_shot=few_shot)
    raise ValueError(f"Unknown task: {task}")


def build_eval_examples(task, num_examples, seed, few_shot=True):
    """Create a fixed eval set without disturbing the training RNG stream."""
    random_state = random.getstate()
    random.seed(seed)
    examples = [sample_json_example(task, few_shot=few_shot) for _ in range(num_examples)]
    random.setstate(random_state)
    return examples


def extract_json_candidate(response):
    response = response.strip()
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end < start:
        return response, False
    candidate = response[start : end + 1]
    no_extra_text = response == candidate
    return candidate, no_extra_text


def truncate_after_first_complete_json(response):
    response = response.strip()
    start = response.find("{")
    if start == -1:
        return response

    try:
        _, end = json.JSONDecoder().raw_decode(response[start:])
    except json.JSONDecodeError:
        return response

    return response[: start + end].strip()


def score_json_response(response, target, reward_mode="shaped"):
    candidate, no_extra_text = extract_json_candidate(response)
    details = {
        "valid_json": 0.0,
        "no_extra_text": 0.0,
        "keys": 0.0,
        "types": 0.0,
        "values": 0.0,
        "exact": 0.0,
    }

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return 0.0, details

    if not isinstance(parsed, dict):
        return (0.0 if reward_mode == "simple" else 0.05), details

    details["valid_json"] = 1.0
    details["no_extra_text"] = float(no_extra_text)

    target_keys = set(target.keys())
    parsed_keys = set(parsed.keys())
    details["keys"] = len(target_keys & parsed_keys) / max(1, len(target_keys))

    type_hits = 0
    value_hits = 0
    for key, expected in target.items():
        if key not in parsed:
            continue
        observed = parsed[key]
        if isinstance(observed, type(expected)):
            type_hits += 1
        if observed == expected:
            value_hits += 1
        elif isinstance(expected, int) and str(observed).isdigit() and int(observed) == expected:
            value_hits += 1

    details["types"] = type_hits / max(1, len(target))
    details["values"] = value_hits / max(1, len(target))
    details["exact"] = float(parsed == target and no_extra_text)

    if reward_mode == "simple":
        if details["exact"]:
            reward = 1.0
        elif parsed == target:
            reward = 0.5
        elif details["valid_json"] and details["keys"] == 1.0:
            reward = 0.2
        else:
            reward = 0.0
        return float(reward), details

    reward = (
        0.15 * details["valid_json"]
        + 0.25 * details["no_extra_text"]
        + 0.15 * details["keys"]
        + 0.10 * details["types"]
        + 0.25 * details["values"]
        + 0.10 * details["exact"]
    )
    return float(reward), details


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state_to_cpu(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def load_state(model, state):
    model.load_state_dict(state, strict=True)


def build_model(model_id, device, dtype=torch.float32):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.config.pad_token_id = model.config.eos_token_id
    return model


def generate_one(model, tokenizer, prompt, device, args, do_sample=True):
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
    prompt_len = enc["input_ids"].shape[1]
    generation_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    with torch.no_grad():
        generated = model.generate(
            **enc,
            **generation_kwargs,
        )
    full_ids = generated[0].detach().cpu()
    response_ids = full_ids[prompt_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    if args.stop_after_json:
        response = truncate_after_first_complete_json(response)
        full_ids, prompt_len = encode_prompt_response(tokenizer, prompt, response)
    return full_ids, prompt_len, response


def response_logprobs_and_entropy(model, full_ids, response_start, device, require_grad=False):
    if len(full_ids) <= response_start:
        return None, None

    input_ids = full_ids.unsqueeze(0).to(device)
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        start = max(response_start - 1, 0)
        selected_logprobs = token_logprobs[:, start:].squeeze(0)

        probs = torch.exp(log_probs[:, start:, :])
        entropy = -(probs * log_probs[:, start:, :]).sum(dim=-1).squeeze(0)

    return selected_logprobs, entropy


def normalize_advantages(rewards, device):
    values = torch.tensor(rewards, dtype=torch.float32, device=device)
    if values.numel() == 1:
        return values - values.mean()
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-8)


def rollout_batch(learner_policy, actor, actor_fp16, tokenizer, actor_state, device, args):
    load_state(actor, actor_state)
    actor.eval()

    if actor_fp16 is not None:
        load_state(actor_fp16, actor_state)
        actor_fp16.eval()

    records = []
    rewards = []
    reward_details = []

    for _ in range(args.batch_size):
        example = sample_json_example(args.task, few_shot=not args.no_few_shot)
        generation_source = actor_fp16 if args.condition in {"mismatch", "rescored", "stale_mismatch"} else actor
        full_ids, response_start, response = generate_one(
            generation_source, tokenizer, example.prompt, device, args
        )

        old_source = actor_fp16 if args.condition in {"mismatch", "stale_mismatch"} else actor

        old_logprobs, _ = response_logprobs_and_entropy(
            old_source, full_ids, response_start, device, require_grad=False
        )
        pre_update_logprobs, _ = response_logprobs_and_entropy(
            learner_policy, full_ids, response_start, device, require_grad=False
        )
        if old_logprobs is None or pre_update_logprobs is None:
            continue

        reward, details = score_json_response(response, example.target, reward_mode=args.reward_mode)
        rewards.append(reward)
        reward_details.append(details)
        records.append(
            {
                "prompt": example.prompt,
                "target": example.target,
                "target_json": example.target_json,
                "schema": example.schema,
                "response": response,
                "full_ids": full_ids,
                "response_start": response_start,
                "old_logprobs": old_logprobs.detach().float().cpu(),
                "pre_update_logprobs": pre_update_logprobs.detach().float().cpu(),
            }
        )

    return records, rewards, reward_details


def encode_prompt_response(tokenizer, prompt, response):
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    full_ids = torch.cat([prompt_ids, response_ids], dim=0)
    return full_ids, int(prompt_ids.numel())


def build_fixed_response_records(policy, tokenizer, device, args):
    fixed_cases = [
        ("Alice", 24, "Boston"),
        ("Ben", 31, "Seattle"),
        ("Grace", 66, "Miami"),
        ("Elena", 20, "Atlanta"),
    ]
    records = []
    rewards = []

    for name, age, city in fixed_cases:
        text = f"Name: {name}. Age: {age}. City: {city}."
        target = {"name": name, "age": age, "city": city}
        prompt = build_prompt(text, ["name", "age", "city"], few_shot=not args.no_few_shot)
        candidates = [
            ("good", compact_json(target), 1.0),
            ("bad", f"The answer is {name} from {city}.", 0.0),
        ]

        for label, response, reward in candidates:
            full_ids, response_start = encode_prompt_response(tokenizer, prompt, response)
            old_logprobs, _ = response_logprobs_and_entropy(
                policy, full_ids, response_start, device, require_grad=False
            )
            if old_logprobs is None:
                continue
            records.append(
                {
                    "prompt": prompt,
                    "target": target,
                    "target_json": compact_json(target),
                    "schema": "fixed_person",
                    "response": response,
                    "response_label": label,
                    "full_ids": full_ids,
                    "response_start": response_start,
                    "old_logprobs": old_logprobs.detach().float().cpu(),
                    "pre_update_logprobs": old_logprobs.detach().float().cpu(),
                }
            )
            rewards.append(reward)

    return records, rewards


def sequence_logprob_sums(policy, records, device):
    sums = []
    policy.eval()
    for record in records:
        logprobs, _ = response_logprobs_and_entropy(
            policy, record["full_ids"], record["response_start"], device, require_grad=False
        )
        sums.append(float(logprobs.detach().float().sum().cpu()) if logprobs is not None else 0.0)
    return sums


def run_fixed_response_debug(policy, optimizer, tokenizer, device, args):
    records, rewards = build_fixed_response_records(policy, tokenizer, device, args)
    before_sums = sequence_logprob_sums(policy, records, device)
    metrics = ppo_update(policy, optimizer, records, rewards, device, args)
    after_sums = sequence_logprob_sums(policy, records, device)

    rows = []
    good_deltas = []
    bad_deltas = []
    for record, reward, before, after in zip(records, rewards, before_sums, after_sums):
        delta = after - before
        label = record["response_label"]
        if label == "good":
            good_deltas.append(delta)
        else:
            bad_deltas.append(delta)
        rows.append(
            {
                "label": label,
                "reward": reward,
                "target_json": record["target_json"],
                "response": record["response"],
                "logprob_before": before,
                "logprob_after": after,
                "logprob_delta": delta,
            }
        )

    summary = {
        "num_records": len(records),
        "good_delta_mean": float(np.mean(good_deltas)) if good_deltas else 0.0,
        "bad_delta_mean": float(np.mean(bad_deltas)) if bad_deltas else 0.0,
        "relative_delta_mean": (
            float(np.mean(good_deltas) - np.mean(bad_deltas)) if good_deltas and bad_deltas else 0.0
        ),
        "relative_passed": bool(good_deltas and bad_deltas and np.mean(good_deltas) > np.mean(bad_deltas)),
        "strict_passed": bool(good_deltas and bad_deltas and np.mean(good_deltas) > 0.0 and np.mean(bad_deltas) < 0.0),
    }
    summary["passed"] = summary["strict_passed"] or summary["relative_passed"]
    return {"summary": summary, "metrics": metrics, "records": rows}


def clone_trainable_params_to_cpu(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def param_delta_norm(model, before_params):
    delta_sq = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in before_params:
            continue
        delta = param.detach().cpu() - before_params[name]
        delta_sq += float(delta.float().pow(2).sum())
    return float(delta_sq**0.5)


def compute_post_update_stats(policy, records, device, args):
    ratios = []
    clip_flags = []
    old_movement = []
    pre_update_movement = []

    policy.eval()
    for record in records:
        post_logprobs, _ = response_logprobs_and_entropy(
            policy,
            record["full_ids"],
            record["response_start"],
            device,
            require_grad=False,
        )
        if post_logprobs is None:
            continue

        old_logprobs = record["old_logprobs"].to(device)
        pre_update_logprobs = record["pre_update_logprobs"].to(device)
        n = min(post_logprobs.numel(), old_logprobs.numel(), pre_update_logprobs.numel())
        if n == 0:
            continue

        post_logprobs = post_logprobs[:n]
        old_logprobs = old_logprobs[:n]
        pre_update_logprobs = pre_update_logprobs[:n]

        ratio = torch.exp(torch.clamp(post_logprobs - old_logprobs, min=-20.0, max=20.0))
        ratios.extend(ratio.detach().float().cpu().tolist())
        clip_flags.extend((torch.abs(ratio - 1.0) > args.clip_eps).float().cpu().tolist())
        old_movement.extend((post_logprobs - old_logprobs).detach().float().cpu().tolist())
        pre_update_movement.extend((post_logprobs - pre_update_logprobs).detach().float().cpu().tolist())

    return {
        "post_update_ratio_mean": float(np.mean(ratios)) if ratios else 1.0,
        "post_update_ratio_variance": float(np.var(ratios)) if ratios else 0.0,
        "post_update_clip_fraction": float(np.mean(clip_flags)) if clip_flags else 0.0,
        "post_update_logprob_movement": float(np.mean(old_movement)) if old_movement else 0.0,
        "post_update_abs_logprob_movement": float(np.mean(np.abs(old_movement))) if old_movement else 0.0,
        "post_update_abs_pre_update_movement": (
            float(np.mean(np.abs(pre_update_movement))) if pre_update_movement else 0.0
        ),
    }


def ppo_update(policy, optimizer, records, rewards, device, args):
    advantages = normalize_advantages(rewards, device)
    epoch_losses = []
    all_ratios = []
    all_clip_flags = []
    all_movement = []
    all_entropy = []
    param_before = clone_trainable_params_to_cpu(policy)
    grad_norm_before_clip = 0.0

    policy.eval()
    for _ in range(args.ppo_epochs):
        optimizer.zero_grad()
        losses = []

        for i, record in enumerate(records):
            new_logprobs, entropy = response_logprobs_and_entropy(
                policy,
                record["full_ids"],
                record["response_start"],
                device,
                require_grad=True,
            )
            if new_logprobs is None:
                continue

            old_logprobs = record["old_logprobs"].to(device)
            pre_update_logprobs = record["pre_update_logprobs"].to(device)
            n = min(new_logprobs.numel(), old_logprobs.numel(), pre_update_logprobs.numel())
            if n == 0:
                continue

            new_logprobs = new_logprobs[:n]
            old_logprobs = old_logprobs[:n]
            pre_update_logprobs = pre_update_logprobs[:n]
            adv = advantages[i]

            log_ratio = new_logprobs - old_logprobs
            ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
            policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
            movement_delta = new_logprobs - pre_update_logprobs
            movement_penalty = torch.mean(movement_delta.pow(2))
            entropy_bonus = entropy[:n].mean()
            loss = policy_loss + args.movement_coef * movement_penalty - args.entropy_coef * entropy_bonus
            losses.append(loss)

            with torch.no_grad():
                all_ratios.extend(ratio.detach().float().cpu().tolist())
                all_clip_flags.extend((torch.abs(ratio - 1.0) > args.clip_eps).float().cpu().tolist())
                all_movement.extend(movement_delta.detach().float().cpu().tolist())
                all_entropy.extend(entropy[:n].detach().float().cpu().tolist())

        if not losses:
            return {
                "loss": 0.0,
                "advantage_std": float(advantages.detach().float().std(unbiased=False).cpu()),
                "advantage_abs_mean": float(advantages.detach().float().abs().mean().cpu()),
                "grad_norm_before_clip": 0.0,
                "param_delta_norm": 0.0,
                "clip_fraction": 0.0,
                "ratio_variance": 0.0,
                "ratio_mean": 1.0,
                "logprob_movement": 0.0,
                "abs_logprob_movement": 0.0,
                "entropy": 0.0,
                "post_update_ratio_mean": 1.0,
                "post_update_ratio_variance": 0.0,
                "post_update_clip_fraction": 0.0,
                "post_update_logprob_movement": 0.0,
                "post_update_abs_logprob_movement": 0.0,
                "post_update_abs_pre_update_movement": 0.0,
            }

        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        grad_norm_before_clip = float(grad_norm.detach().cpu())
        optimizer.step()
        epoch_losses.append(float(batch_loss.detach().cpu()))

    post_update_stats = compute_post_update_stats(policy, records, device, args)
    return {
        "loss": float(np.mean(epoch_losses)),
        "advantage_std": float(advantages.detach().float().std(unbiased=False).cpu()),
        "advantage_abs_mean": float(advantages.detach().float().abs().mean().cpu()),
        "grad_norm_before_clip": grad_norm_before_clip,
        "param_delta_norm": param_delta_norm(policy, param_before),
        "clip_fraction": float(np.mean(all_clip_flags)) if all_clip_flags else 0.0,
        "ratio_variance": float(np.var(all_ratios)) if all_ratios else 0.0,
        "ratio_mean": float(np.mean(all_ratios)) if all_ratios else 1.0,
        "logprob_movement": float(np.mean(all_movement)) if all_movement else 0.0,
        "abs_logprob_movement": float(np.mean(np.abs(all_movement))) if all_movement else 0.0,
        "entropy": float(np.mean(all_entropy)) if all_entropy else 0.0,
        **post_update_stats,
    }


def pick_actor_state(snapshot_queue, args):
    # Sync uses the latest pre-update policy snapshot, which is standard PPO:
    # collect rollouts with pi_old, then update the learner. Stale conditions
    # use a snapshot from L learner updates before that normal rollout policy.
    if args.condition in {"stale", "stale_mismatch"} and args.lag > 0:
        return snapshot_queue[0]
    return snapshot_queue[-1]


def plot_metrics(logs, out_path, title):
    if not logs:
        return
    steps = [row["update"] for row in logs]
    reward = [row["reward_mean"] for row in logs]
    clip = [row["clip_fraction"] for row in logs]
    movement = [
        row.get("post_update_abs_logprob_movement", row["abs_logprob_movement"])
        for row in logs
    ]
    success = [row["pass_at_1"] for row in logs]
    eval_rows = [row for row in logs if "eval_pass_at_1" in row]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(steps, reward, linewidth=1.8)
    if eval_rows:
        axes[0, 0].plot(
            [row["update"] for row in eval_rows],
            [row["eval_reward_mean"] for row in eval_rows],
            linestyle="--",
            linewidth=1.5,
            label="eval",
        )
        axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Reward")
    axes[0, 0].set_xlabel("Update")

    axes[0, 1].plot(steps, success, linewidth=1.8)
    if eval_rows:
        axes[0, 1].plot(
            [row["update"] for row in eval_rows],
            [row["eval_pass_at_1"] for row in eval_rows],
            linestyle="--",
            linewidth=1.5,
            label="eval",
        )
        axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("Pass@1")
    axes[0, 1].set_xlabel("Update")

    axes[1, 0].plot(steps, clip, linewidth=1.8)
    axes[1, 0].set_title("Clip Fraction")
    axes[1, 0].set_xlabel("Update")

    axes[1, 1].plot(steps, movement, linewidth=1.8)
    axes[1, 1].set_title("Mean |Logprob Movement|")
    axes[1, 1].set_xlabel("Update")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_reward_details(details):
    if not details:
        return {}
    keys = details[0].keys()
    return {key: float(np.mean([row[key] for row in details])) for key in keys}


def evaluate_policy(policy, tokenizer, eval_examples, device, args):
    policy.eval()
    rewards = []
    details = []
    samples = []

    for example in eval_examples:
        _, _, response = generate_one(policy, tokenizer, example.prompt, device, args, do_sample=False)
        reward, reward_detail = score_json_response(response, example.target, reward_mode=args.reward_mode)
        rewards.append(reward)
        details.append(reward_detail)
        if len(samples) < args.num_eval_samples_to_log:
            samples.append(
                {
                    "schema": example.schema,
                    "target_json": example.target_json,
                    "response": response,
                    "reward": reward,
                }
            )

    detail_means = summarize_reward_details(details)
    return {
        "eval_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "eval_reward_std": float(np.std(rewards)) if rewards else 0.0,
        "eval_pass_at_1": float(np.mean([row["exact"] for row in details])) if details else 0.0,
        **{f"eval_{key}": value for key, value in detail_means.items()},
        "eval_samples": samples,
    }


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Condition: {args.condition}  Task: {args.task}  Seed: {args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading learner policy...")
    policy = build_model(args.model_id, device, dtype=torch.float32)
    policy.config.use_cache = False
    print("Loading actor policy...")
    actor = build_model(args.model_id, device, dtype=torch.float32)

    actor_fp16 = None
    if args.condition in {"mismatch", "rescored", "stale_mismatch"}:
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Loading mismatch actor ({dtype})...")
        actor_fp16 = build_model(args.model_id, device, dtype=dtype)

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    # For GPT-2-scale controlled experiments only. Larger models should store
    # fewer snapshots or use adapter-only checkpoints.
    snapshot_queue = deque(maxlen=max(1, args.lag + 1))
    snapshot_queue.append(clone_state_to_cpu(policy))

    run_name = f"live_ppo_{args.condition}_{args.task}_seed{args.seed}"
    if args.run_name:
        run_name = args.run_name
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "logs.json")
    fig_path = os.path.join(run_dir, "metrics.png")

    if args.debug_fixed_responses:
        debug_payload = {
            "args": vars(args),
            **run_fixed_response_debug(policy, optimizer, tokenizer, device, args),
        }
        with open(log_path, "w") as f:
            json.dump(debug_payload, f, indent=2)
        summary = debug_payload["summary"]
        print(
            "Fixed-response debug: "
            f"good_delta={summary['good_delta_mean']:.6f} "
            f"bad_delta={summary['bad_delta_mean']:.6f} "
            f"relative={summary['relative_delta_mean']:.6f} "
            f"passed={summary['passed']}"
        )
        print(f"Logs saved to {log_path}")
        return

    eval_examples = build_eval_examples(
        args.task,
        args.eval_batch_size,
        seed=args.seed + 10000,
        few_shot=not args.no_few_shot,
    )

    logs = []
    for update in range(1, args.updates + 1):
        while len(snapshot_queue) < snapshot_queue.maxlen:
            snapshot_queue.append(copy.deepcopy(snapshot_queue[-1]))
        actor_state = pick_actor_state(snapshot_queue, args)

        records, rewards, details = rollout_batch(policy, actor, actor_fp16, tokenizer, actor_state, device, args)
        if not records:
            print(f"Update {update}: no valid rollouts, skipping")
            continue

        metrics = ppo_update(policy, optimizer, records, rewards, device, args)
        reward_detail_means = summarize_reward_details(details)
        pass_at_1 = float(np.mean([row["exact"] for row in details])) if details else 0.0
        reward_mean = float(np.mean(rewards))

        row = {
            "update": update,
            "condition": args.condition,
            "task": args.task,
            "lag": args.lag,
            "seed": args.seed,
            "sample_schema": records[0]["schema"],
            "sample_target_json": records[0]["target_json"],
            "sample_response": records[0]["response"],
            "reward_mean": reward_mean,
            "reward_std": float(np.std(rewards)),
            "pass_at_1": pass_at_1,
            "num_rollouts": len(records),
            **metrics,
            **{f"reward_{key}": value for key, value in reward_detail_means.items()},
        }

        if args.eval_every > 0 and (update == 1 or update % args.eval_every == 0):
            row.update(evaluate_policy(policy, tokenizer, eval_examples, device, args))

        logs.append(row)

        snapshot_queue.append(clone_state_to_cpu(policy))

        if update % args.log_every == 0 or update == 1:
            print(
                f"Update {update:04d}/{args.updates} "
                f"reward={reward_mean:.3f} pass@1={pass_at_1:.3f} "
                f"clip={row['post_update_clip_fraction']:.3f} "
                f"|post_move|={row['post_update_abs_logprob_movement']:.4f} "
                f"grad={row['grad_norm_before_clip']:.4f} "
                f"loss={row['loss']:.4f}"
            )

    payload = {
        "args": vars(args),
        "logs": logs,
    }
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Logs saved to {log_path}")

    plot_metrics(logs, fig_path, title=f"{args.condition} / {args.task} / seed {args.seed}")
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument(
        "--condition",
        type=str,
        default="sync",
        choices=["sync", "stale", "mismatch", "rescored", "stale_mismatch"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="easy_json",
        choices=["easy_json", "medium_json", "mixed_json"],
    )
    parser.add_argument("--lag", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip_eps", type=float, default=CLIP_EPS)
    parser.add_argument("--movement_coef", type=float, default=0.02)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--stop_after_json", action="store_true")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_eval_samples_to_log", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--reward_mode", type=str, default="shaped", choices=["shaped", "simple"])
    parser.add_argument("--debug_fixed_responses", action="store_true")
    parser.add_argument("--no_few_shot", action="store_true")
    main(parser.parse_args())
