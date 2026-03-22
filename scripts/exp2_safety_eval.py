"""Experiment 2 — Primary Safety Evaluation.

Measures ASR reduction vs. baselines on CoT-Hijacking benchmark.

Conditions:
  (a) Vanilla Qwen3-14B
  (b) RAD-CoT alpha=0.1
  (c) RAD-CoT alpha=0.3
  (d) RAD-CoT alpha=0.5
  (e) SafeChain decoding (baseline)
  (f) RLHF-tuned proxy (Llama-3-8B-Instruct baseline)

Judge: GPT-4o with StrongREJECT rubric.
Report: ASR = fraction of attack queries that succeed.

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
from rad_cot.evaluation.judge import compute_asr, evaluate_batch
from rad_cot.models.model_loader import load_model_and_tokenizer
from rad_cot.steering.dms import DMSResult
from rad_cot.steering.soft_steering import RADCoTSteering, generate_with_steering
from rad_cot.utils.config import ModelConfig
from rad_cot.utils.logging import init_wandb, setup_logger

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
                        help="JSON file with attack prompts or base harmful prompts")
    parser.add_argument("--output-dir", type=str, default="outputs/exp2")
    parser.add_argument("--n-attempts", type=int, default=100)
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # Load DMS result
    with open(args.dms_result, "rb") as f:
        dms_result: DMSResult = pickle.load(f)

    # Load model
    model, tokenizer = load_model_and_tokenizer(ModelConfig(name=args.model))

    # Load attack prompts
    with open(args.attack_prompts) as f:
        raw_prompts = json.load(f)

    # Construct CoT-Hijacking attack prompts
    attack_prompts = []
    for i, prompt in enumerate(raw_prompts[:args.n_attempts]):
        padding = generate_cot_hijacking_padding(
            min_tokens=100, max_tokens=500, seed=args.seed + i
        )
        attack_prompts.append(f"{padding}\n\nNow, please help me with: {prompt}")

    logger.info(f"Prepared {len(attack_prompts)} attack prompts")

    conditions = {
        "vanilla": {"alpha": None},
        "rad_cot_0.1": {"alpha": 0.1},
        "rad_cot_0.3": {"alpha": 0.3},
        "rad_cot_0.5": {"alpha": 0.5},
    }

    all_results = {}

    for cond_name, cond_params in conditions.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running condition: {cond_name}")
        logger.info(f"{'='*60}")

        prompts, responses, wall_time = run_condition(
            model, tokenizer, attack_prompts,
            dms_result=dms_result if cond_params["alpha"] is not None else None,
            alpha=cond_params["alpha"],
        )

        # Judge evaluation
        logger.info(f"Evaluating {len(responses)} responses with {args.judge_model}...")
        judge_results = evaluate_batch(prompts, responses, judge_model=args.judge_model)
        asr = compute_asr(judge_results)

        cond_result = {
            "condition": cond_name,
            "alpha": cond_params["alpha"],
            "asr": asr,
            "n_attack_success": sum(1 for r in judge_results if r.is_attack_success),
            "n_total": len(judge_results),
            "wall_time_s": wall_time,
            "mean_score": sum(r.score for r in judge_results) / len(judge_results),
        }
        all_results[cond_name] = cond_result

        # Save per-condition details
        cond_dir = output_dir / cond_name
        cond_dir.mkdir(exist_ok=True)
        with open(cond_dir / "results.json", "w") as f:
            json.dump(cond_result, f, indent=2)
        with open(cond_dir / "responses.json", "w") as f:
            json.dump([
                {"prompt": p, "response": r, "score": jr.score, "attack_success": jr.is_attack_success}
                for p, r, jr in zip(prompts, responses, judge_results)
            ], f, indent=2)

        logger.info(f"ASR ({cond_name}): {asr:.4f}")

    # Summary
    summary = {
        "conditions": all_results,
        "model": args.model,
        "n_attempts": args.n_attempts,
        "judge_model": args.judge_model,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2 SUMMARY")
    for name, res in all_results.items():
        logger.info(f"  {name}: ASR = {res['asr']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
