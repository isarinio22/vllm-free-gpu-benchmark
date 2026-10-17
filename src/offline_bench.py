"""Experiment 1: offline generation throughput, HF transformers vs vLLM.

    python src/offline_bench.py --method hf_naive     # one prompt at a time, first 32 prompts
    python src/offline_bench.py --method hf_batched   # static batches of 32, all 200 prompts
    python src/offline_bench.py --method vllm         # one LLM.generate call on all 200 prompts

Run each method in its own process (vLLM preallocates most of the GPU memory).
Each one does 1 discarded warm-up run and 5 timed runs, then writes
results/raw/offline_<method>.json.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import torch

from common import (
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    MODEL,
    NAIVE_SUBSET,
    RAW_DIR,
    SEED,
    TIMED_RUNS,
    WARMUP_RUNS,
    env_info,
    load_prompts,
    summarize_runs,
    write_json,
)

GIB = 1024**3

# One benchmark run: returns the number of generated tokens per prompt,
# plus the text of the first output as a sanity check.
RunFn = Callable[[], tuple[list[int], str]]


def sync() -> None:
    """Wait for queued GPU work in this process before reading the clock.

    For HF the model runs in this process, so this matters. vLLM runs the model
    in a separate engine process, and LLM.generate() only returns once every
    request has finished, so if this process has no CUDA context there is
    nothing to wait for.
    """
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def time_runs(run: RunFn) -> dict[str, Any]:
    """Run WARMUP_RUNS discarded runs, then TIMED_RUNS timed runs; return tok/s per run and stats."""
    for i in range(WARMUP_RUNS):
        print(f"warm-up {i + 1}/{WARMUP_RUNS} ...", flush=True)
        run()

    seconds: list[float] = []
    tokens_per_s: list[float] = []
    for i in range(TIMED_RUNS):
        sync()
        start = time.perf_counter()
        counts, sample = run()
        sync()
        elapsed = time.perf_counter() - start

        seconds.append(elapsed)
        tokens_per_s.append(sum(counts) / elapsed)
        print(f"run {i + 1}/{TIMED_RUNS}: {sum(counts)} tokens in {elapsed:.2f}s = {tokens_per_s[-1]:.1f} tok/s", flush=True)

    not_full = sum(c != MAX_NEW_TOKENS for c in counts)
    if not_full:
        print(f"WARNING: {not_full} of {len(counts)} outputs did not have exactly {MAX_NEW_TOKENS} tokens")
    return {
        "output_tokens_per_run": sum(counts),
        "outputs_not_full_length": not_full,
        "seconds": seconds,
        "tokens_per_s": tokens_per_s,
        "summary": summarize_runs(tokens_per_s),
        "sample_output": sample,
    }


def bench_hf(prompts: list[str], batch_size: int) -> dict[str, Any]:
    """Benchmark HF transformers generate() with static, left-padded batches."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    weights_gib = torch.cuda.memory_allocated() / GIB

    # Plain greedy decoding. We build the config ourselves because Qwen's
    # generation_config.json turns on sampling and repetition_penalty=1.1.
    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MAX_NEW_TOKENS,  # blocks EOS, so every output is exactly 128 tokens
        repetition_penalty=1.0,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    @torch.inference_mode()
    def run() -> tuple[list[int], str]:
        counts: list[int] = []
        first_text = ""
        for start in range(0, len(prompts), batch_size):
            inputs = tokenizer(prompts[start : start + batch_size], return_tensors="pt", padding=True).to("cuda")
            output = model.generate(**inputs, generation_config=gen_config, use_model_defaults=False)
            new_ids = output[:, inputs["input_ids"].shape[1] :]
            # Count real generated tokens; anything after an EOS would be padding.
            counts += (new_ids != tokenizer.pad_token_id).sum(dim=1).tolist()
            if start == 0:
                first_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)
        return counts, first_text

    torch.cuda.reset_peak_memory_stats()
    result = time_runs(run)
    result["memory"] = {
        "weights_gib": weights_gib,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
    }
    return result


def bench_vllm(prompts: list[str], attention_backend: str | None) -> dict[str, Any]:
    """Benchmark vLLM offline inference: all prompts in a single LLM.generate() call."""
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="half",  # the T4 has no bf16 support
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=False,  # timed runs repeat the same prompts; don't reuse KV from earlier runs
        generation_config="vllm",  # ignore Qwen's sampling defaults (repetition_penalty=1.1, ...)
        attention_backend=attention_backend,  # None = let vLLM choose (FlashInfer on a T4)
        seed=SEED,
    )
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, ignore_eos=True)

    def run() -> tuple[list[int], str]:
        outputs = llm.generate(prompts, params, use_tqdm=False)
        return [len(o.outputs[0].token_ids) for o in outputs], outputs[0].outputs[0].text

    result = time_runs(run)

    # vLLM reserves GPU_MEMORY_UTILIZATION of VRAM up front by design, so a
    # "peak memory" number would say nothing. What matters is how much of that
    # reservation became KV cache: the number below matches vLLM's
    # "GPU KV cache size: N tokens" log line.
    cache = llm.llm_engine.vllm_config.cache_config
    kv_cache_tokens = cache.num_gpu_blocks * cache.block_size
    cfg = AutoConfig.from_pretrained(MODEL)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2  # K and V, 2 bytes each (fp16)
    result["memory"] = {
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "kv_cache_tokens": kv_cache_tokens,
        "kv_cache_gib": kv_cache_tokens * bytes_per_token / GIB,
        "kv_bytes_per_token": bytes_per_token,
        "block_size": cache.block_size,
        "max_full_length_sequences": kv_cache_tokens / MAX_MODEL_LEN,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=["hf_naive", "hf_batched", "vllm"])
    parser.add_argument("--batch-size", type=int, default=32, help="hf_batched only (default: 32)")
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="vllm only, e.g. TRITON_ATTN if the default FlashInfer backend fails on the T4",
    )
    parser.add_argument("--out", type=Path, default=None, help="default: results/raw/offline_<method>.json")
    args = parser.parse_args()

    prompts = load_prompts()
    if args.method == "hf_naive":
        prompts, batch_size = prompts[:NAIVE_SUBSET], 1
        result = bench_hf(prompts, batch_size)
    elif args.method == "hf_batched":
        batch_size = args.batch_size
        result = bench_hf(prompts, batch_size)
    else:
        batch_size = None  # vLLM schedules its own batches
        result = bench_vllm(prompts, args.attention_backend)

    out = args.out or RAW_DIR / f"offline_{args.method}.json"
    write_json(out, {
        "method": args.method,
        "model": MODEL,
        "num_prompts": len(prompts),
        "batch_size": batch_size,
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        **result,
        "env": env_info(),
    })
    s = result["summary"]
    print(
        f"{args.method}: {s['mean']:.1f} tok/s mean "
        f"(95% CI {s['ci95_low']:.1f}-{s['ci95_high']:.1f}, median {s['median']:.1f}) -> {out}"
    )


if __name__ == "__main__":  # required: vLLM may start worker processes that re-import this module
    main()
