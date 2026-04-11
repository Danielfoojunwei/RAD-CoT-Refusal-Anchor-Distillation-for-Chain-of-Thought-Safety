"""Experiment 2 — Primary Safety Evaluation.

Measures ASR reduction vs. baselines on CoT-Hijacking benchmark.

Conditions:
  (a) Vanilla Qwen3-14B
  (b) RAD-CoT alpha=0.1
  (c) RAD-CoT alpha=0.3
  (d) RAD-CoT alpha=0.5

Judge: GPT-4o with StrongREJECT rubric.
Report: ASR with 95% bootstrap CI. Judge failures are EXCLUDED from ASR.

IMPORTANT: Attack prompts MUST be from a held-out evaluation set that was
NOT used during DMS circuit identification. Using the same prompts for
calibration and evaluation constitutes data contamination.

Usage:
    python scripts/exp2_safety_eval.py \
        --dms-result outputs/dms/dms_result.pkl \
        --model Qwen/Qwen3-14B \
        --attack-prompts data/cot_hijacking_eval.json \
        --output-dir outputs/exp2
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from tqdm import tqdm

from rad_cot.data.calibration import generate_cot_hijacking_padding
from rad_cot.evaluation.judge import (
    BatchJudgeResults,
    compute_asr,
    compute_asr_with_ci,
    evaluate_batch,
)
from rad_cot.models.model_loader import load_model_and_tokenizer
from rad_cot.steering.dms import DMSResult
from rad_cot.steering.soft_steering import RADCoTSteering, generate_with_steering
from rad_cot.utils.config import ModelConfig
from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)


def generate_vanilla(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    """Generate without any steering."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.6, top_p=0.95,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_condition(
    model,
    tokenizer,
    attack_prompts: list[str],
    dms_result: DMSResult | None = None,
    alpha: float | None = None,
    max_new_tokens: int = 2048,
) -> tuple[list[str], list[str], float]:
    """Run a single experimental condition. Returns (prompts, responses, wall_time_s)."""
    responses = []
    t_start = time.time()

    for prompt in tqdm(attack_prompts, desc=f"Generating (alpha={alpha})"):
        if alpha is not None and dms_result is not None:
            response, stats = generate_with_steering(
                model, tokenizer, prompt, dms_result, alpha=alpha,
                max_new_tokens=max_new_tokens,
            )
        else:
            response = generate_vanilla(model, tokenizer, prompt, max_new_tokens)
        responses.append(response)

    wall_time = time.time() - t_start
    return attack_prompts, responses, wall_time


def main():
    parser = argparse.ArgumentParser(description="Exp 2: Primary Safety Evaluation")
    parser.add_argument("--dms-result", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--attack-prompts", type=str, required=True,
                        help="JSON file with HELD-OUT evaluation prompts (not used in calibration)")
    parser.add_argument("--output-dir", type=str, default="outputs/exp2")
    parser.add_argument("--n-attempts", type=int, default=100)
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for multi-run evaluation (e.g., --seeds 42 123 456)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or [args.seed]

    # Load DMS result
    with open(args.dms_result, "rb") as f:
        dms_result: DMSResult = pickle.load(f)

    # Load model
    model, tokenizer = load_model_and_tokenizer(ModelConfig(name=args.model))

    # Load attack prompts
    with open(args.attack_prompts) as f:
        raw_prompts = json.load(f)

    conditions = {
        "vanilla": {"alpha": None},
        "rad_cot_0.1": {"alpha": 0.1},
        "rad_cot_0.3": {"alpha": 0.3},
        "rad_cot_0.5": {"alpha": 0.5},
    }

    all_results = {}

    for run_idx, seed in enumerate(seeds):
        logger.info(f"\n{'#'*60}")
        logger.info(f"RUN {run_idx + 1}/{len(seeds)} (seed={seed})")
        logger.info(f"{'#'*60}")

        torch.manual_seed(seed)

        # Construct CoT-Hijacking attack prompts with this seed
        attack_prompts = []
        for i, prompt in enumerate(raw_prompts[:args.n_attempts]):
            padding = generate_cot_hijacking_padding(
                min_tokens=100, max_tokens=500, seed=seed + i
            )
            attack_prompts.append(f"{padding}\n\nNow, please help me with: {prompt}")

        logger.info(f"Prepared {len(attack_prompts)} attack prompts")

        for cond_name, cond_params in conditions.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Running condition: {cond_name} (seed={seed})")
            logger.info(f"{'='*60}")

            prompts, responses, wall_time = run_condition(
                model, tokenizer, attack_prompts,
                dms_result=dms_result if cond_params["alpha"] is not None else None,
                alpha=cond_params["alpha"],
            )

            # Judge evaluation
            logger.info(f"Evaluating {len(responses)} responses with {args.judge_model}...")
            batch_results = evaluate_batch(
                prompts, responses, judge_model=args.judge_model
            )

            # Compute ASR with confidence intervals
            asr, ci_lower, ci_upper = compute_asr_with_ci(batch_results)
            valid_results = batch_results.valid_results

            cond_result = {
                "condition": cond_name,
                "alpha": cond_params["alpha"],
                "seed": seed,
                "asr": asr,
                "asr_ci_lower": ci_lower,
                "asr_ci_upper": ci_upper,
                "n_attack_success": sum(
                    1 for r in valid_results if r.is_attack_success
                ),
                "n_valid": len(valid_results),
                "n_judge_errors": batch_results.n_errors,
                "n_total": batch_results.n_total,
                "judge_error_rate": batch_results.error_rate,
                "wall_time_s": wall_time,
                "mean_score": (
                    sum(r.score for r in valid_results) / len(valid_results)
                    if valid_results else 0.0
                ),
            }

            result_key = f"{cond_name}_seed{seed}"
            all_results[result_key] = cond_result

            # Save per-condition details
            cond_dir = output_dir / result_key
            cond_dir.mkdir(exist_ok=True)
            with open(cond_dir / "results.json", "w") as f:
                json.dump(cond_result, f, indent=2)
            with open(cond_dir / "responses.json", "w") as f:
                json.dump([
                    {
                        "prompt": p, "response": r,
                        "score": jr.score,
                        "attack_success": jr.is_attack_success,
                        "judge_error": jr.judge_error,
                    }
                    for p, r, jr in zip(prompts, responses, batch_results.results)
                ], f, indent=2)

            logger.info(
                f"ASR ({cond_name}, seed={seed}): {asr:.4f} "
                f"[{ci_lower:.4f}, {ci_upper:.4f}] "
                f"(judge errors: {batch_results.n_errors}/{batch_results.n_total})"
            )

    # Aggregate across seeds
    aggregate = {}
    for cond_name in conditions:
        cond_runs = [
            v for k, v in all_results.items() if k.startswith(cond_name + "_seed")
        ]
        if cond_runs:
            asrs = [r["asr"] for r in cond_runs]
            import numpy as np
            aggregate[cond_name] = {
                "mean_asr": float(np.mean(asrs)),
                "std_asr": float(np.std(asrs)),
                "min_asr": float(np.min(asrs)),
                "max_asr": float(np.max(asrs)),
                "n_runs": len(cond_runs),
            }

    # Summary
    summary = {
        "per_run_results": all_results,
        "aggregate": aggregate,
        "model": args.model,
        "n_attempts": args.n_attempts,
        "seeds": seeds,
        "judge_model": args.judge_model,
        "note": (
            "Judge errors are EXCLUDED from ASR computation. "
            "ASR confidence intervals are 95% bootstrap CIs."
        ),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2 SUMMARY (AGGREGATE)")
    for name, agg in aggregate.items():
        logger.info(
            f"  {name}: mean ASR = {agg['mean_asr']:.4f} +/- {agg['std_asr']:.4f} "
            f"(n={agg['n_runs']} runs)"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
