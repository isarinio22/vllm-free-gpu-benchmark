"""Turn the raw JSON results into the CSVs that plot.py and the README use.

    python src/summarize.py

Reads results/raw/*.json, writes results/offline_throughput.csv and
results/load_test.csv, and prints both so they can be copied off the
Colab runtime.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from common import RAW_DIR, RESULTS_DIR, read_json

OFFLINE_METHODS = ["hf_naive", "hf_batched", "vllm"]
OFFLINE_CSV = RESULTS_DIR / "offline_throughput.csv"
LOAD_TEST_CSV = RESULTS_DIR / "load_test.csv"


def offline_rows() -> list[dict[str, Any]]:
    """One row per offline method that has a result file."""
    rows = []
    for method in OFFLINE_METHODS:
        path = RAW_DIR / f"offline_{method}.json"
        if not path.exists():
            print(f"skipping {method}: {path} not found")
            continue
        result = read_json(path)
        stats, memory = result["summary"], result["memory"]
        rows.append({
            "method": method,
            "num_prompts": result["num_prompts"],
            "batch_size": result["batch_size"],
            "timed_runs": stats["n"],
            "mean_tok_s": stats["mean"],
            "ci95_low": stats["ci95_low"],
            "ci95_high": stats["ci95_high"],
            "median_tok_s": stats["median"],
            "peak_mem_gib": memory.get("peak_allocated_gib"),  # HF only
            "kv_cache_tokens": memory.get("kv_cache_tokens"),  # vLLM only
            "kv_cache_gib": memory.get("kv_cache_gib"),  # vLLM only
        })
    return rows


def load_test_rows() -> list[dict[str, Any]]:
    """One row per concurrency level, without the per-request detail."""
    path = RAW_DIR / "load_test.json"
    if not path.exists():
        print(f"skipping load test: {path} not found")
        return []
    columns = ["concurrency", "requests", "errors", "output_tok_s", "p50_latency_s", "p95_latency_s", "p50_ttft_s"]
    return [{key: level[key] for key in columns} for level in read_json(path)["levels"]]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, rounding floats so the file stays readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})


def main() -> None:
    for path, rows in ((OFFLINE_CSV, offline_rows()), (LOAD_TEST_CSV, load_test_rows())):
        if not rows:
            continue
        write_csv(path, rows)
        print(f"\n==> {path.relative_to(RESULTS_DIR.parent)} <==")
        print(path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
