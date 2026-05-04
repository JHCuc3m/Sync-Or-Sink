#!/usr/bin/env python3
"""Static vLLM-vs-HF backend probes for the final report.

This script is intentionally independent from the PPO training loop. It tests
whether an actor-side vLLM backend and a learner-side HuggingFace/PyTorch
backend agree on fixed generated token sequences, and optionally runs a
zero-shot structured JSON sanity check.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")


@dataclass
class JsonExample:
    prompt: str
    answer: int
    parity: str


def build_arithmetic_examples(n: int, prompt_style: str = "current") -> list[JsonExample]:
    examples: list[JsonExample] = []
    for i in range(n):
        a = 11 + (7 * i) % 89
        b = 13 + (11 * i) % 87
        op = "+" if i % 2 == 0 else "-"
        answer = a + b if op == "+" else a - b
        parity = "even" if answer % 2 == 0 else "odd"
        if prompt_style == "strict":
            prompt = (
                "Return only a valid JSON object. Do not use Markdown. "
                "Do not include explanations, code, or extra text. "
                "The object must have exactly two keys: \"answer\" and \"parity\". "
                f"Compute: {a} {op} {b}."
            )
        elif prompt_style == "one_shot":
            prompt = (
                "Return only a valid JSON object with exactly two keys: \"answer\" and \"parity\". "
                "Do not use Markdown or extra text.\n"
                "Example:\n"
                "Compute: 2 + 4.\n"
                "{\"answer\": 6, \"parity\": \"even\"}\n"
                f"Compute: {a} {op} {b}."
            )
        else:
            prompt = (
                "Return exactly one JSON object with keys \"answer\" and \"parity\". "
                "Use no markdown and no extra text. "
                f"Compute: {a} {op} {b}."
            )
        examples.append(JsonExample(prompt=prompt, answer=answer, parity=parity))
    return examples


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def score_structured_json(text: str, target_answer: int, target_parity: str) -> dict[str, Any]:
    stripped = text.strip()
    match = re.search(r"\{.*?\}", stripped, flags=re.DOTALL)
    obj = None
    if match:
        try:
            parsed = json.loads(match.group(0))
            obj = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            obj = None
    valid_json = obj is not None
    correct_answer = False
    correct_parity = False
    no_extra_text = False

    if obj is not None:
        try:
            correct_answer = int(obj.get("answer")) == target_answer
        except Exception:
            correct_answer = False
        correct_parity = str(obj.get("parity", "")).strip().lower() == target_parity
        no_extra_text = match is not None and match.start() == 0 and match.end() == len(stripped)

    reward = 0.0
    if valid_json:
        reward += 0.4
    else:
        reward -= 0.25
    if correct_answer:
        reward += 0.3
    if correct_parity:
        reward += 0.2
    if no_extra_text:
        reward += 0.1

    return {
        "valid_json": valid_json,
        "correct_answer": correct_answer,
        "correct_parity": correct_parity,
        "no_extra_text": no_extra_text,
        "pass_at_1": bool(valid_json and correct_answer and correct_parity and no_extra_text),
        "reward": reward,
    }


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        prompts = [
            line.strip()
            for line in Path(args.prompt_file).read_text().splitlines()
            if line.strip()
        ]
        return prompts[: args.num_prompts]

    examples = build_arithmetic_examples(args.num_prompts, args.prompt_style)
    return [ex.prompt for ex in examples]


def get_vllm_logprob(entry: Any, token_id: int) -> float | None:
    """Extract the chosen-token logprob across vLLM API versions."""
    if entry is None:
        return None
    candidates = []
    if isinstance(entry, dict):
        candidates.extend([entry.get(token_id), entry.get(str(token_id))])
        if len(entry) == 1:
            candidates.extend(entry.values())
    else:
        candidates.append(entry)

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (float, int)):
            return float(candidate)
        value = getattr(candidate, "logprob", None)
        if value is not None:
            return float(value)
        if isinstance(candidate, dict) and "logprob" in candidate:
            return float(candidate["logprob"])
    return None


def summarize_deltas(deltas: list[float], clip_eps: float) -> dict[str, Any]:
    abs_deltas = [abs(x) for x in deltas]
    ratio_errors = [math.exp(x) for x in deltas]
    lower = math.log(1.0 - clip_eps)
    upper = math.log(1.0 + clip_eps)
    crossing = [x < lower or x > upper for x in deltas]

    if not deltas:
        return {
            "num_tokens": 0,
            "clip_eps": clip_eps,
            "error": "no comparable tokens",
        }

    sorted_abs = sorted(abs_deltas)

    def percentile(p: float) -> float:
        if not sorted_abs:
            return float("nan")
        idx = min(len(sorted_abs) - 1, max(0, math.ceil(p * len(sorted_abs)) - 1))
        return sorted_abs[idx]

    return {
        "num_tokens": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "std_delta": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "mean_abs_delta": statistics.fmean(abs_deltas),
        "p95_abs_delta": percentile(0.95),
        "p99_abs_delta": percentile(0.99),
        "max_abs_delta": max(abs_deltas),
        "mean_ratio_error": statistics.fmean(ratio_errors),
        "p99_ratio_error": sorted(ratio_errors)[min(len(ratio_errors) - 1, math.ceil(0.99 * len(ratio_errors)) - 1)],
        "clip_eps": clip_eps,
        "clip_crossing_fraction": sum(crossing) / len(crossing),
    }


def run_vllm_generation(args: argparse.Namespace, prompts: list[str]):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        dtype=args.vllm_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        seed=args.seed,
    )
    return llm.generate(prompts, sampling)


def generated_token_ids(output: Any) -> list[int]:
    completion = output.outputs[0]
    token_ids = getattr(completion, "token_ids", None)
    if token_ids is None:
        token_ids = getattr(completion, "token_ids", [])
    return list(token_ids)


def prompt_token_ids(output: Any, tokenizer: Any, prompt: str) -> list[int]:
    ids = getattr(output, "prompt_token_ids", None)
    if ids is not None:
        return list(ids)
    return tokenizer.encode(prompt, add_special_tokens=False)


def serialize_vllm_outputs(args: argparse.Namespace, prompts: list[str], outputs: list[Any]) -> dict[str, Any]:
    records = []
    for prompt, output in zip(prompts, outputs):
        completion = output.outputs[0]
        gen_ids = generated_token_ids(output)
        p_ids = getattr(output, "prompt_token_ids", None)
        token_records = []
        for j, token_id in enumerate(gen_ids):
            vllm_entry = None
            if getattr(completion, "logprobs", None) is not None and j < len(completion.logprobs):
                vllm_entry = completion.logprobs[j]
            token_records.append(
                {
                    "token_id": token_id,
                    "vllm_logprob": get_vllm_logprob(vllm_entry, token_id),
                }
            )
        records.append(
            {
                "prompt": prompt,
                "prompt_token_ids": list(p_ids) if p_ids is not None else None,
                "text": completion.text,
                "generated_token_ids": gen_ids,
                "token_records": token_records,
            }
        )
    return {
        "metadata": {
            "model": args.model,
            "tokenizer": args.tokenizer or args.model,
            "num_prompts": len(prompts),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "logprobs": args.logprobs,
        },
        "records": records,
    }


def run_vllm_only(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args)
    outputs = run_vllm_generation(args, prompts)
    payload = serialize_vllm_outputs(args, prompts, outputs)
    write_json(Path(args.output_dir) / "vllm_generations.json", payload)

    if args.score_json_sanity:
        examples = build_arithmetic_examples(args.num_prompts, args.prompt_style)
        records = []
        rewards = []
        valid = 0
        correct_answer = 0
        correct_parity = 0
        passes = 0
        for ex, record in zip(examples, payload["records"]):
            score = score_structured_json(record["text"], ex.answer, ex.parity)
            rewards.append(score["reward"])
            valid += int(score["valid_json"])
            correct_answer += int(score["correct_answer"])
            correct_parity += int(score["correct_parity"])
            passes += int(score["pass_at_1"])
            records.append(
                {
                    "prompt": ex.prompt,
                    "target_answer": ex.answer,
                    "target_parity": ex.parity,
                    "output": record["text"],
                    "score": score,
                }
            )
        n = len(examples)
        json_payload = {
            "summary": {
                "mode": "json_sanity",
                "model": args.model,
                "num_prompts": n,
                "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
                "valid_json_rate": valid / n if n else 0.0,
                "answer_accuracy": correct_answer / n if n else 0.0,
                "parity_accuracy": correct_parity / n if n else 0.0,
                "pass_at_1": passes / n if n else 0.0,
            },
            "records": records[: args.record_prompts],
        }
        write_json(Path(args.output_dir) / "vllm_json_sanity.json", json_payload)
        payload["json_sanity_summary"] = json_payload["summary"]

    return payload


def api_base(args: argparse.Namespace) -> str:
    if args.api_base:
        return args.api_base.rstrip("/")
    port = os.environ.get("port_vllm") or os.environ.get("VLLM_PORT") or "8000"
    return f"http://127.0.0.1:{port}/v1"


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def run_api_generate(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args)
    base = api_base(args)
    records = []
    for prompt in prompts:
        response = post_json(
            f"{base}/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "logprobs": args.logprobs,
                "seed": args.seed,
                "return_token_ids": True,
            },
        )
        choice = response["choices"][0]
        logprobs = choice.get("logprobs") or {}
        token_logprobs = logprobs.get("token_logprobs") or []
        tokens = logprobs.get("tokens") or []
        token_records = []
        for token_text, token_logprob in zip(tokens, token_logprobs):
            token_records.append(
                {
                    "token": token_text,
                    "token_id": None,
                    "vllm_logprob": token_logprob,
                }
            )
        records.append(
            {
                "prompt": prompt,
                "prompt_token_ids": choice.get("prompt_token_ids"),
                "text": choice.get("text", ""),
                "generated_token_ids": choice.get("token_ids"),
                "token_records": token_records,
                "finish_reason": choice.get("finish_reason"),
            }
        )

    payload = {
        "metadata": {
            "mode": "api_generate",
            "api_base": base,
            "model": args.model,
            "tokenizer": args.tokenizer or args.model,
            "num_prompts": len(prompts),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "logprobs": args.logprobs,
        },
        "records": records,
    }
    write_json(Path(args.output_dir) / "vllm_generations.json", payload)

    if args.score_json_sanity:
        examples = build_arithmetic_examples(args.num_prompts, args.prompt_style)
        rewards = []
        valid = 0
        correct_answer = 0
        correct_parity = 0
        passes = 0
        sanity_records = []
        for ex, record in zip(examples, records):
            score = score_structured_json(record["text"], ex.answer, ex.parity)
            rewards.append(score["reward"])
            valid += int(score["valid_json"])
            correct_answer += int(score["correct_answer"])
            correct_parity += int(score["correct_parity"])
            passes += int(score["pass_at_1"])
            sanity_records.append(
                {
                    "prompt": ex.prompt,
                    "target_answer": ex.answer,
                    "target_parity": ex.parity,
                    "output": record["text"],
                    "score": score,
                }
            )
        n = len(examples)
        json_payload = {
            "summary": {
                "mode": "json_sanity",
                "model": args.model,
                "num_prompts": n,
                "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
                "valid_json_rate": valid / n if n else 0.0,
                "answer_accuracy": correct_answer / n if n else 0.0,
                "parity_accuracy": correct_parity / n if n else 0.0,
                "pass_at_1": passes / n if n else 0.0,
            },
            "records": sanity_records[: args.record_prompts],
        }
        write_json(Path(args.output_dir) / "vllm_json_sanity.json", json_payload)
        payload["json_sanity_summary"] = json_payload["summary"]

    return payload


def run_hf_recompute(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    generation_path = Path(args.generation_file)
    payload = json.loads(generation_path.read_text())
    records_in = payload["records"]

    hf_model_name = args.hf_model or args.model or payload.get("metadata", {}).get("model")
    tokenizer_name = args.tokenizer or hf_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=getattr(torch, args.hf_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    deltas: list[float] = []
    records_out: list[dict[str, Any]] = []
    missing_vllm_logprobs = 0
    token_text_matches = 0
    token_text_compared = 0
    token_count_mismatches = 0

    with torch.no_grad():
        for record in records_in:
            prompt = record["prompt"]
            gen_ids = record.get("generated_token_ids")
            if gen_ids is None:
                gen_ids = tokenizer.encode(record["text"], add_special_tokens=False)
            gen_ids = list(gen_ids)
            p_ids = record.get("prompt_token_ids")
            if p_ids is None:
                p_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not gen_ids:
                continue
            api_token_records = record.get("token_records", [])
            if api_token_records and len(gen_ids) != len(api_token_records):
                token_count_mismatches += 1
            all_ids = list(p_ids) + gen_ids
            model_device = next(model.parameters()).device
            input_ids = torch.tensor([all_ids], dtype=torch.long, device=model_device)
            logits = model(input_ids).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            token_records = []
            for j, token_id in enumerate(gen_ids):
                source_pos = len(p_ids) + j - 1
                if source_pos < 0:
                    continue
                hf_logprob = float(log_probs[0, source_pos, token_id].cpu())
                vllm_logprob = None
                if j < len(api_token_records):
                    vllm_logprob = api_token_records[j].get("vllm_logprob")
                    api_token = api_token_records[j].get("token")
                    if api_token is not None:
                        token_text_compared += 1
                        token_text_matches += int(tokenizer.decode([token_id]) == api_token)
                if vllm_logprob is None:
                    missing_vllm_logprobs += 1
                    continue
                delta = float(vllm_logprob) - hf_logprob
                deltas.append(delta)
                token_records.append(
                    {
                        "token_id": token_id,
                        "vllm_logprob": float(vllm_logprob),
                        "hf_logprob": hf_logprob,
                        "delta": delta,
                    }
                )
            records_out.append(
                {
                    "prompt": prompt,
                    "text": record["text"],
                    "num_generated_tokens": len(gen_ids),
                    "tokens_compared": len(token_records),
                    "token_records": token_records[: args.record_tokens_per_prompt],
                }
            )

    summary = summarize_deltas(deltas, args.clip_eps)
    summary.update(
        {
            "mode": "hf_recompute",
            "model": payload.get("metadata", {}).get("model", args.model),
            "hf_model": hf_model_name,
            "generation_file": str(generation_path),
            "num_prompts": len(records_in),
            "missing_vllm_logprobs": missing_vllm_logprobs,
            "token_count_mismatched_prompts": token_count_mismatches,
            "token_text_match_fraction": token_text_matches / token_text_compared if token_text_compared else None,
            "token_text_compared": token_text_compared,
        }
    )
    out = {"summary": summary, "records": records_out[: args.record_prompts]}
    write_json(Path(args.output_dir) / "vllm_hf_parity.json", out)
    return out


def run_parity(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(args)
    outputs = run_vllm_generation(args, prompts)

    hf_model_name = args.hf_model or args.model
    tokenizer_name = args.tokenizer or hf_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=getattr(torch, args.hf_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    deltas: list[float] = []
    records: list[dict[str, Any]] = []
    missing_vllm_logprobs = 0

    with torch.no_grad():
        for prompt, output in zip(prompts, outputs):
            completion = output.outputs[0]
            gen_ids = generated_token_ids(output)
            p_ids = prompt_token_ids(output, tokenizer, prompt)
            if not gen_ids:
                continue

            all_ids = p_ids + gen_ids
            model_device = next(model.parameters()).device
            input_ids = torch.tensor([all_ids], dtype=torch.long, device=model_device)
            logits = model(input_ids).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            token_records = []
            for j, token_id in enumerate(gen_ids):
                source_pos = len(p_ids) + j - 1
                if source_pos < 0:
                    continue
                hf_logprob = float(log_probs[0, source_pos, token_id].cpu())
                vllm_entry = None
                if getattr(completion, "logprobs", None) is not None and j < len(completion.logprobs):
                    vllm_entry = completion.logprobs[j]
                vllm_logprob = get_vllm_logprob(vllm_entry, token_id)
                if vllm_logprob is None:
                    missing_vllm_logprobs += 1
                    continue
                delta = vllm_logprob - hf_logprob
                deltas.append(delta)
                token_records.append(
                    {
                        "token_id": token_id,
                        "vllm_logprob": vllm_logprob,
                        "hf_logprob": hf_logprob,
                        "delta": delta,
                    }
                )

            records.append(
                {
                    "prompt": prompt,
                    "text": completion.text,
                    "num_generated_tokens": len(gen_ids),
                    "tokens_compared": len(token_records),
                    "token_records": token_records[: args.record_tokens_per_prompt],
                }
            )

    summary = summarize_deltas(deltas, args.clip_eps)
    summary.update(
        {
            "mode": "parity",
            "model": args.model,
            "hf_model": hf_model_name,
            "num_prompts": len(prompts),
            "missing_vllm_logprobs": missing_vllm_logprobs,
        }
    )
    payload = {"summary": summary, "records": records[: args.record_prompts]}
    write_json(Path(args.output_dir) / "vllm_hf_parity.json", payload)
    return payload


def run_json_sanity(args: argparse.Namespace) -> dict[str, Any]:
    examples = build_arithmetic_examples(args.num_prompts, args.prompt_style)
    outputs = run_vllm_generation(args, [ex.prompt for ex in examples])

    records = []
    rewards = []
    valid = 0
    correct_answer = 0
    correct_parity = 0
    passes = 0

    for ex, output in zip(examples, outputs):
        text = output.outputs[0].text
        score = score_structured_json(text, ex.answer, ex.parity)
        rewards.append(score["reward"])
        valid += int(score["valid_json"])
        correct_answer += int(score["correct_answer"])
        correct_parity += int(score["correct_parity"])
        passes += int(score["pass_at_1"])
        records.append(
            {
                "prompt": ex.prompt,
                "target_answer": ex.answer,
                "target_parity": ex.parity,
                "output": text,
                "score": score,
            }
        )

    n = len(examples)
    summary = {
        "mode": "json_sanity",
        "model": args.model,
        "num_prompts": n,
        "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
        "valid_json_rate": valid / n if n else 0.0,
        "answer_accuracy": correct_answer / n if n else 0.0,
        "parity_accuracy": correct_parity / n if n else 0.0,
        "pass_at_1": passes / n if n else 0.0,
    }
    payload = {"summary": summary, "records": records[: args.record_prompts]}
    write_json(Path(args.output_dir) / "vllm_json_sanity.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["parity", "json_sanity", "both", "vllm_generate", "api_generate", "hf_recompute"],
        default="both",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf_model", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--prompt_style", choices=["current", "strict", "one_shot"], default="current")
    parser.add_argument("--num_prompts", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vllm_dtype", default="auto")
    parser.add_argument("--hf_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--record_prompts", type=int, default=20)
    parser.add_argument("--record_tokens_per_prompt", type=int, default=16)
    parser.add_argument("--generation_file", default="outputs/vllm_backend_probe/vllm_generations.json")
    parser.add_argument("--score_json_sanity", action="store_true")
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--output_dir", default="outputs/vllm_backend_probe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {}
    if args.mode == "vllm_generate":
        payload = run_vllm_only(args)
        results["vllm_generate"] = {
            "num_prompts": len(payload["records"]),
            "output": str(Path(args.output_dir) / "vllm_generations.json"),
        }
        if "json_sanity_summary" in payload:
            results["json_sanity"] = payload["json_sanity_summary"]
    elif args.mode == "api_generate":
        payload = run_api_generate(args)
        results["api_generate"] = {
            "num_prompts": len(payload["records"]),
            "output": str(Path(args.output_dir) / "vllm_generations.json"),
            "api_base": api_base(args),
        }
        if "json_sanity_summary" in payload:
            results["json_sanity"] = payload["json_sanity_summary"]
    elif args.mode == "hf_recompute":
        results["parity"] = run_hf_recompute(args)["summary"]
    elif args.mode in {"parity", "both"}:
        results["parity"] = run_parity(args)["summary"]
    if args.mode in {"json_sanity", "both"}:
        results["json_sanity"] = run_json_sanity(args)["summary"]
    write_json(Path(args.output_dir) / "summary.json", results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
