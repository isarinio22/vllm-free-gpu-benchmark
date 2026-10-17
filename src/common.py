"""Shared settings and small helpers used by every benchmark script."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128
NUM_PROMPTS = 200
NAIVE_SUBSET = 32  # hf_naive runs one prompt at a time, so it only gets the first 32 prompts
SEED = 42
MAX_PROMPT_TOKENS = 512
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.9  # vLLM's default, stated explicitly because the README explains it
WARMUP_RUNS = 1
TIMED_RUNS = 5

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = ROOT / "data" / "prompts.json"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"


def load_prompts() -> list[str]:
    """Return the fixed, chat-formatted prompt strings written by src/prompts.py."""
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"{PROMPTS_PATH} not found. Run `python src/prompts.py` first.")
    return [row["prompt"] for row in read_json(PROMPTS_PATH)]


def summarize_runs(values: list[float]) -> dict[str, float]:
    """Mean with a t-based 95% confidence interval, plus the median.

    With n=5 runs a bootstrap interval is unreliable, so we use the Student t
    distribution: mean ± t(0.975, n-1) · s / √n.
    """
    from scipy import stats

    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    half_width = float(stats.t.ppf(0.975, n - 1)) * std / n**0.5 if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "median": statistics.median(values),
    }


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """Write data as pretty-printed JSON, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def env_info() -> dict[str, Any]:
    """Library versions and GPU name, so every result file records what produced it.

    The GPU name comes from nvidia-smi rather than torch.cuda, so calling this
    never creates a CUDA context in the current process.
    """
    info: dict[str, Any] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "vllm"):
        try:
            info[package] = version(package)
        except PackageNotFoundError:
            info[package] = None
    try:
        info["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["gpu"] = None
    return info
