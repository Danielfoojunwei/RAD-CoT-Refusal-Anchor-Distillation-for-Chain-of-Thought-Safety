"""Experiment 3 — Reasoning Quality Preservation.

Verifies soft steering does not degrade reasoning. Evaluates all alpha
conditions on GSM8K (8-shot), MATH L4-5 (4-shot), HumanEval (0-shot pass@1).

Usage:
    python scripts/exp3_reasoning_quality.py \
        --dms-result outputs/dms/dms_result.pkl \
        --model Qwen/Qwen3-14B \
        --output-dir outputs/exp3
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from rad_cot.evaluation.benchmarks import (
    evaluate_gsm8k,
    evaluate_humaneval,
    evaluate_math,
)
from rad_cot.models.model_loader import load_model_and_tokenizer
from rad_cot.steering.dms import DMSResult
from rad_cot.steering.soft_steering import RADCoTSteering
from rad_cot.utils.config import ModelConfig
from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Exp 3: Reasoning Quality Preservation")
    parser.add_argument("--dms-result", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--output-dir", type=str, default="outputs/exp3")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DMS result
    with open(args.dms_result, "rb") as f:
        dms_result: DMSResult = pickle.load(f)

    # Load model
    model, tokenizer = load_model_and_tokenizer(ModelConfig(name=args.model))

    benchmarks = {
        "gsm8k": {"fn": evaluate_gsm8k, "kwargs": {"num_fewshot": 8}},
        "math": {"fn": evaluate_math, "kwargs": {"num_fewshot": 4}},
        "humaneval": {"fn": evaluate_humaneval, "kwargs": {}},
    }

    all_results = {}

    # Vanilla baseline
    logger.info("Evaluating vanilla baseline...")
    vanilla_results = {}
    for bench_name, bench_cfg in benchmarks.items():
        result = bench_cfg["fn"](
            args.model,
            output_dir=str(output_dir / "vanilla" / bench_name),
            **bench_cfg["kwargs"],
        )
        vanilla_results[bench_name] = result.score
        logger.info(f"  Vanilla {bench_name}: {result.score:.4f}")
    all_results["vanilla"] = vanilla_results

    # Steered conditions
    for alpha in args.alphas:
        logger.info(f"\nEvaluating alpha={alpha}...")
        steering = RADCoTSteering(model, dms_result, alpha=alpha)
        steering.attach()

        alpha_results = {}
        for bench_name, bench_cfg in benchmarks.items():
            result = bench_cfg["fn"](
                args.model,
                output_dir=str(output_dir / f"alpha_{alpha}" / bench_name),
                **bench_cfg["kwargs"],
            )
            alpha_results[bench_name] = result.score
            drop = vanilla_results[bench_name] - result.score
            logger.info(f"  alpha={alpha} {bench_name}: {result.score:.4f} (drop={drop:.4f})")

        steering.detach()
        all_results[f"alpha_{alpha}"] = alpha_results

    # Summary with degradation analysis
    summary = {
        "model": args.model,
        "conditions": all_results,
        "degradation": {},
    }
    for alpha in args.alphas:
        key = f"alpha_{alpha}"
        deg = {}
        for bench in benchmarks:
            deg[bench] = vanilla_results[bench] - all_results[key][bench]
        deg["max_degradation"] = max(deg.values())
        deg["passes_3pct_threshold"] = all(v < 0.03 for v in deg.values() if isinstance(v, float))
        summary["degradation"][key] = deg

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 3 SUMMARY")
    for cond, scores in all_results.items():
        logger.info(f"  {cond}: {scores}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
