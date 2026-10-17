"""Experiment 2: online serving load test against `vllm serve`.

Starts vLLM's OpenAI-compatible server in the background, waits for /health,
then sends streaming /v1/completions requests at several concurrency levels.
It records time to first token (TTFT) and total latency for each request and
writes results/raw/load_test.json.

    python src/load_test.py                 # concurrency 1, 4, 16, 64
    python src/load_test.py --levels 1 4    # a quicker run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import httpx
import numpy as np

from common import MAX_MODEL_LEN, MAX_NEW_TOKENS, MODEL, RAW_DIR, env_info, load_prompts, write_json

SERVER_LOG = RAW_DIR / "vllm_server.log"
STARTUP_TIMEOUT_S = 15 * 60  # a cold start on a T4 includes torch.compile and CUDA graph capture
REQUEST_TIMEOUT_S = 300
MIN_REQUESTS = 64  # each level sends max(64, 4 x concurrency) requests
WARMUP_REQUESTS = 8


@dataclass
class RequestResult:
    """Timing for one streamed request."""

    ttft_s: float | None
    latency_s: float
    output_tokens: int
    error: str | None = None


def server_command(port: int, attention_backend: str | None) -> list[str]:
    """The `vllm serve` command line used for the load test."""
    cmd = [
        "vllm", "serve", MODEL,
        "--dtype", "half",  # the T4 has no bf16 support
        "--max-model-len", str(MAX_MODEL_LEN),
        "--no-enable-prefix-caching",  # prompts repeat across levels; cache hits would inflate the numbers
        "--generation-config", "vllm",  # ignore Qwen's sampling defaults (repetition_penalty=1.1, ...)
        "--port", str(port),
    ]
    if attention_backend:
        cmd += ["--attention-backend", attention_backend]
    return cmd


@contextmanager
def vllm_server(cmd: list[str], port: int) -> Iterator[str]:
    """Run the server for the duration of the with-block and yield its base URL."""
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SERVER_LOG, "w") as log:
        # A new session puts the server and its engine subprocess in their own
        # process group, so we can stop all of them together.
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            base_url = f"http://localhost:{port}"
            wait_until_healthy(proc, base_url)
            yield base_url
        finally:
            stop_server(proc)


def wait_until_healthy(proc: subprocess.Popen, base_url: str) -> None:
    """Poll GET /health until it returns 200; fail fast if the server process dies."""
    print(f"Starting vLLM server, log: {SERVER_LOG}", flush=True)
    start = time.monotonic()
    while time.monotonic() - start < STARTUP_TIMEOUT_S:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {proc.returncode}. End of log:\n{log_tail()}")
        try:
            if httpx.get(f"{base_url}/health", timeout=2).status_code == 200:
                print(f"Server ready after {time.monotonic() - start:.0f}s", flush=True)
                return
        except httpx.TransportError:
            pass  # not listening yet
        time.sleep(2)
    raise TimeoutError(f"vllm serve was not healthy after {STARTUP_TIMEOUT_S}s. End of log:\n{log_tail()}")


def stop_server(proc: subprocess.Popen) -> None:
    """SIGTERM the server's process group, then SIGKILL whatever is left so no GPU memory leaks."""
    for sig, wait_s in ((signal.SIGTERM, 60), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return  # the whole group is already gone
        try:
            proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            pass


def log_tail(lines: int = 40) -> str:
    """The last lines of the server log, for error messages."""
    return "\n".join(SERVER_LOG.read_text(errors="replace").splitlines()[-lines:])


async def one_request(client: httpx.AsyncClient, prompt: str) -> RequestResult:
    """Send one streaming completion and time it.

    return_token_ids makes vLLM put the new token ids in every stream chunk, so
    output tokens are counted from real ids, not assumed.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,  # always generate exactly MAX_NEW_TOKENS tokens
        "stream": True,
        "return_token_ids": True,
    }
    start = time.perf_counter()
    ttft: float | None = None
    tokens = 0
    try:
        async with client.stream("POST", "/v1/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                for choice in json.loads(line[len("data: ") :])["choices"]:
                    new_ids = choice.get("token_ids") or []
                    if new_ids and ttft is None:
                        ttft = time.perf_counter() - start
                    tokens += len(new_ids)
    except httpx.HTTPError as exc:
        return RequestResult(ttft, time.perf_counter() - start, tokens, error=repr(exc))
    return RequestResult(ttft, time.perf_counter() - start, tokens)


async def run_level(
    base_url: str, prompts: list[str], concurrency: int, num_requests: int
) -> tuple[list[RequestResult], float]:
    """Send num_requests requests with at most `concurrency` in flight; return results and wall time."""
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=REQUEST_TIMEOUT_S, limits=limits) as client:

        async def limited(prompt: str) -> RequestResult:
            async with semaphore:
                return await one_request(client, prompt)

        start = time.perf_counter()
        results = await asyncio.gather(*(limited(prompts[i % len(prompts)]) for i in range(num_requests)))
        wall_s = time.perf_counter() - start
    return list(results), wall_s


def summarize_level(concurrency: int, results: list[RequestResult], wall_s: float) -> dict[str, Any]:
    """Throughput and latency percentiles for one concurrency level."""
    ok = [r for r in results if r.error is None]
    if not ok:
        raise RuntimeError(f"All requests failed at concurrency {concurrency}, e.g. {results[0].error}")
    latencies = [r.latency_s for r in ok]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    output_tokens = sum(r.output_tokens for r in ok)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "outputs_not_full_length": sum(r.output_tokens != MAX_NEW_TOKENS for r in ok),
        "wall_s": wall_s,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / wall_s,
        "p50_latency_s": float(np.percentile(latencies, 50)),
        "p95_latency_s": float(np.percentile(latencies, 95)),
        "p50_ttft_s": float(np.percentile(ttfts, 50)),
    }


async def run_all(base_url: str, prompts: list[str], levels: list[int]) -> list[dict[str, Any]]:
    """Warm up the server, then run every concurrency level in order."""
    print(f"Warm-up: {WARMUP_REQUESTS} requests (discarded)", flush=True)
    await run_level(base_url, prompts, WARMUP_REQUESTS, WARMUP_REQUESTS)

    summaries = []
    for concurrency in levels:
        num_requests = max(MIN_REQUESTS, 4 * concurrency)
        results, wall_s = await run_level(base_url, prompts, concurrency, num_requests)
        summary = summarize_level(concurrency, results, wall_s)
        print(
            f"concurrency {concurrency:>3}: {summary['output_tok_s']:7.1f} tok/s, "
            f"p50 latency {summary['p50_latency_s']:.2f}s, p95 {summary['p95_latency_s']:.2f}s, "
            f"p50 TTFT {summary['p50_ttft_s'] * 1000:.0f}ms, errors {summary['errors']}",
            flush=True,
        )
        summaries.append({**summary, "per_request": [asdict(r) for r in results]})
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 16, 64], help="concurrency levels")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--attention-backend", default=None, help="e.g. TRITON_ATTN if FlashInfer fails on the T4")
    args = parser.parse_args()

    prompts = load_prompts()
    cmd = server_command(args.port, args.attention_backend)
    with vllm_server(cmd, args.port) as base_url:
        levels = asyncio.run(run_all(base_url, prompts, args.levels))

    out = RAW_DIR / "load_test.json"
    write_json(out, {
        "model": MODEL,
        "max_new_tokens": MAX_NEW_TOKENS,
        "server_command": " ".join(cmd),
        "levels": levels,
        "env": env_info(),
    })
    print(f"-> {out}")


if __name__ == "__main__":
    main()
