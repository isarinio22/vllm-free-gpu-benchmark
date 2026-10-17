"""Build the fixed prompt set: 200 Dolly-15k prompts in Qwen's chat format.

Run once before the benchmarks:

    python src/prompts.py

Every method then reads the same data/prompts.json, so they all see
byte-identical prompt strings.
"""

from __future__ import annotations

import json
import random
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from common import MAX_PROMPT_TOKENS, MODEL, NUM_PROMPTS, PROMPTS_PATH, SEED, write_json

DATASET_REPO = "databricks/databricks-dolly-15k"
DATASET_FILE = "databricks-dolly-15k.jsonl"


def load_dolly() -> list[dict[str, str]]:
    """Download (or reuse the cached copy of) Dolly-15k and parse its JSONL rows."""
    path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_user_message(row: dict[str, str]) -> str:
    """The instruction, followed by its context paragraph when there is one."""
    if row["context"].strip():
        return f"{row['instruction']}\n\n{row['context']}"
    return row["instruction"]


def build_prompts(tokenizer: PreTrainedTokenizerBase) -> list[dict[str, Any]]:
    """Shuffle Dolly with a fixed seed and keep the first NUM_PROMPTS prompts
    that fit in MAX_PROMPT_TOKENS once formatted with the chat template."""
    rows = load_dolly()
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)

    prompts: list[dict[str, Any]] = []
    for idx in order:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": to_user_message(rows[idx])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        num_tokens = len(tokenizer(text)["input_ids"])
        if num_tokens > MAX_PROMPT_TOKENS:
            continue
        prompts.append({"id": idx, "category": rows[idx]["category"], "num_tokens": num_tokens, "prompt": text})
        if len(prompts) == NUM_PROMPTS:
            break
    return prompts


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = build_prompts(tokenizer)
    write_json(PROMPTS_PATH, prompts)
    lengths = [p["num_tokens"] for p in prompts]
    print(
        f"Wrote {len(prompts)} prompts to {PROMPTS_PATH} "
        f"(prompt tokens: min {min(lengths)}, mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)})"
    )


if __name__ == "__main__":
    main()
