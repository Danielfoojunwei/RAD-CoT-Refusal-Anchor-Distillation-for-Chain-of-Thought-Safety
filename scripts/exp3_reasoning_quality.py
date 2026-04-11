"""Experiment 3 — Reasoning Quality Preservation.

Verifies soft steering does not degrade reasoning. Evaluates all alpha
conditions on GSM8K (8-shot), MATH L4-5 (4-shot), HumanEval (0-shot pass@1).

CRITICAL FIX: Uses in-process evaluation that passes the actual model object
(with steering hooks attached) to the benchmark functions. The previous
subprocess-based approach launched lm_eval as a separate process which loaded
a fresh model without hooks, meaning steered conditions were never actually
tested.

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
    evaluate_gsm8k_inprocess,
    evaluate_humaneval_inprocess,
    evaluate_math_inprocess,
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
    parser.add_argument("--gsm8k-path", type=str, default="data/gsm8k_test.json")
    parser.add_argument("--math-path", type=str, default="data/math_l4l5.json")
    parser.add_argument("--humaneval-path", type=str, default="data/humaneval.json")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per benchmark (None = full dataset)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DMS result
    with open(args.dms_result, "rb") as f:
        dms_result: DMSResult = pickle.load(f)

    # Load model — this single model object is used for ALL conditions
    model, tokenizer = load_model_and_tokenizer(ModelConfig(name=args.model))

    # In-process benchmark functions that take the model OBJECT (not a string)
    benchmarks = {
        "gsm8k": {
            "fn": evaluate_gsm8k_inprocess,
            "kwargs": {"dataset_path": args.gsm8k_path, "max_samples": args.max_samples},
        },
        "math": {
            "fn": evaluate_math_inprocess,
            "kwargs": {"dataset_path": args.math_path, "max_samples": args.max_samples},
        },
        "humaneval": {
            "fn": evaluate_humaneval_inprocess,
            "kwargs": {"dataset_path": args.humaneval_path, "max_samples": args.max_samples},
        },
    }

    all_results = {}

    # Vanilla baseline — no hooks attached
    logger.info("Evaluating vanilla baseline (no steering hooks)...")
    vanilla_results = {}
    for bench_name, bench_cfg in benchmarks.items():
        result = bench_cfg["fn"](
            model=model,
            tokenizer=tokenizer,
            **bench_cfg["kwargs"],
        )
        vanilla_results[bench_name] = result.score
        logger.info(f"  Vanilla {bench_name}: {result.score:.4f} ({result.n_samples} samples)")
    all_results["vanilla"] = vanilla_results

    # Steered conditions — hooks ARE attached to the same model object
    for alpha in args.alphas:
        logger.info(f"\nEvaluating alpha={alpha} (steering hooks active)...")
        steering = RADCoTSteering(model, dms_result, alpha=alpha)
        steering.attach()

        alpha_results = {}
        for bench_name, bench_cfg in benchmarks.items():
            result = bench_cfg["fn"](
                model=model,
                tokenizer=tokenizer,
                **bench_cfg["kwargs"],
            )
            alpha_results[bench_name] = result.score
            drop = vanilla_results[bench_name] - result.score
            logger.info(
                f"  alpha={alpha} {bench_name}: {result.score:.4f} "
                f"(drop={drop:.4f}, {result.n_samples} samples)"
            )

        # Log correction stats to verify hooks were active
        stats = steering.get_stats()
        logger.info(
            f"  Steering stats: {stats.n_corrections} corrections, "
            f"{stats.invariant_violations} violations, "
            f"{stats.correction_time_ms:.1f}ms correction time"
        )
        if stats.n_tokens_generated > 0 and stats.n_corrections == 0:
            logger.warning(
                "  WARNING: Zero corrections during evaluation. "
                "Steering may not be active or threshold may be too low."
            )

        steering.detach()
        all_results[f"alpha_{alpha}"] = alpha_results

    # Summary with degradation analysis
    summary = {
        "model": args.model,
        "conditions": all_results,
        "degradation": {},
        "note": (
            "All conditions evaluated in-process with the same model object. "
            "Steered conditions have hooks attached to the model during evaluation, "
            "ensuring that steering is actually active (unlike subprocess-based lm_eval)."
        ),
    }
    for alpha in args.alphas:
        key = f"alpha_{alpha}"
        deg = {}
        for bench in benchmarks:
            deg[bench] = vanilla_results[bench] - all_results[key][bench]
        deg["max_degradation"] = max(v for v in deg.values() if isinstance(v, float))
        deg["passes_3pct_threshold"] = all(
            v < 0.03 for v in deg.values() if isinstance(v, float)
        )
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
