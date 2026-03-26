"""Benchmark inference latency overhead from soft steering.

Measures:
  - Inference latency overhead: < 8% increase target
  - Invariant check overhead: < 2ms per 100 tokens target

Usage:
    python scripts/benchmark_latency.py \
        --dms-result outputs/dms/dms_result.pkl \
        --model Qwen/Qwen3-14B \
        --output-dir outputs/latency
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
import numpy as np

from rad_cot.models.model_loader import load_model_and_tokenizer
from rad_cot.steering.dms import DMSResult
from rad_cot.steering.soft_steering import RADCoTSteering
from rad_cot.utils.config import ModelConfig
from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)

BENCHMARK_PROMPTS = [
    "Explain the concept of recursion in computer science with examples.",
    "What are the key differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis step by step.",
    "How does a neural network learn through backpropagation?",
    "Explain the theory of general relativity in simple terms.",
]


def benchmark_generation(
    model, tokenizer, prompts, max_new_tokens=500, n_runs=3
) -> list[float]:
    """Benchmark generation latency. Returns list of wall times in ms."""
    device = next(model.parameters()).device
    times = []

    for _ in range(n_runs):
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1000)

    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark latency overhead")
    parser.add_argument("--dms-result", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--output-dir", type=str, default="outputs/latency")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dms_result, "rb") as f:
        dms_result: DMSResult = pickle.load(f)

    model, tokenizer = load_model_and_tokenizer(ModelConfig(name=args.model))

    # Warmup
    logger.info("Warming up...")
    inputs = tokenizer("Hello", return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    # Vanilla benchmark
    logger.info("Benchmarking vanilla generation...")
    vanilla_times = benchmark_generation(
        model, tokenizer, BENCHMARK_PROMPTS, args.max_new_tokens, args.n_runs
    )

    # Steered benchmark
    logger.info("Benchmarking steered generation...")
    steering = RADCoTSteering(model, dms_result, alpha=0.3)
    steering.attach()
    steered_times = benchmark_generation(
        model, tokenizer, BENCHMARK_PROMPTS, args.max_new_tokens, args.n_runs
    )
    stats = steering.get_stats()
    steering.detach()

    vanilla_mean = np.mean(vanilla_times)
    steered_mean = np.mean(steered_times)
    overhead_pct = (steered_mean - vanilla_mean) / vanilla_mean * 100

    # Correction overhead per 100 tokens
    total_tokens = stats.n_tokens_generated
    correction_per_100 = stats.correction_time_ms / max(total_tokens, 1) * 100

    results = {
        "vanilla_mean_ms": vanilla_mean,
        "vanilla_std_ms": float(np.std(vanilla_times)),
        "steered_mean_ms": steered_mean,
        "steered_std_ms": float(np.std(steered_times)),
        "overhead_pct": overhead_pct,
        "target_overhead_pct": 8.0,
        "passes_latency_target": overhead_pct < 8.0,
        "correction_per_100_tokens_ms": correction_per_100,
        "target_correction_ms": 2.0,
        "passes_correction_target": correction_per_100 < 2.0,
        "n_corrections": stats.n_corrections,
        "n_tokens": total_tokens,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("LATENCY BENCHMARK RESULTS")
    logger.info(f"  Vanilla:  {vanilla_mean:.1f}ms +/- {np.std(vanilla_times):.1f}ms")
    logger.info(f"  Steered:  {steered_mean:.1f}ms +/- {np.std(steered_times):.1f}ms")
    logger.info(f"  Overhead: {overhead_pct:.2f}% (target < 8%)")
    logger.info(f"  Correction/100tok: {correction_per_100:.3f}ms (target < 2ms)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
