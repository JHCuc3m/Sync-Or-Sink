import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
FIG_DIR = OUTPUT_DIR / "figures"
CKPT_DIR = OUTPUT_DIR / "checkpoints"


def ensure_output_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def split_hh_prompt_and_response(text: str):
    marker = "\n\nAssistant:"
    split_idx = text.rfind(marker)
    if split_idx == -1:
        return "", text
    prompt = text[: split_idx + len(marker)]
    response = text[split_idx + len(marker):]
    return prompt, response


def load_hh_preferences(split="train", num_samples=2000):
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    ds = ds.select(range(min(num_samples, len(ds))))

    def reformat(example):
        chosen_prompt, chosen_response = split_hh_prompt_and_response(example["chosen"])
        _, rejected_response = split_hh_prompt_and_response(example["rejected"])
        return {
            "prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }

    ds = ds.map(reformat, remove_columns=ds.column_names)
    return ds.filter(
        lambda row: len(row["prompt"]) > 0
        and len(row["chosen"]) > 0
        and len(row["rejected"]) > 0
    )


def write_config(experiment_name: str, config: dict):
    ensure_output_dirs()
    save_json(LOG_DIR / f"{experiment_name}_config.json", config)
