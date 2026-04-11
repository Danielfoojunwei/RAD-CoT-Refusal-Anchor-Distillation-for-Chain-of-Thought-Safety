"""Experiment 5 — Ablation Studies.

Sub-experiments:
  A) K sensitivity: vary K from 1 to 50
  B) Alpha sweep: {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
  C) Delta sensitivity: {0.5x, 0.8x, 1.0x, 1.2x, 1.5x} baseline
  D) Layer depth: restrict DMS to early (1-8), middle (9-16), late (17+) layers

Usage:
    python scripts/exp5_ablations.py \
        --dms-result outputs/dms/dms_result.pkl \
        --model Qwen/Qwen3-14B \
        --attack-prompts data/cot_hijacking_eval.json \
        --output-dir outputs/exp5
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rad_cot.data.calibration import generate_cot_hijacking_padding
from rad_cot.evaluation.judge import compute_asr, compute_asr_with_ci, evaluate_batch
from rad_cot.models.model_loader import load_model_and_tokenizer
from rad_cot.steering.dms import DMSResult, select_circuit
from rad_cot.steering.soft_steering import generate_with_steering
from rad_cot.utils.config import ModelConfig
from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)


def prepare_attack_prompts(raw_prompts: list[str], n: int, seed: int) -> list[str]:
    """Construct CoT-Hijacking attack prompts."""
    prompts = []
    for i, p in enumerate(raw_prompts[:n]):
        padding = generate_cot_hijacking_padding(100, 500, seed=seed + i)
        prompts.append(f"{padding}\n\nNow, please help me with: {p}")
    return prompts


def evaluate_condition(
    model, tokenizer, prompts, dms_result, alpha, judge_model
) -> dict:
    """Run generation + judge evaluation for a single condition."""
    responses = []
    for prompt in tqdm(prompts, desc=f"alpha={alpha}", leave=False):
        resp, _ = generate_with_steering(model, tokenizer, prompt, dms_result, alpha=alpha)
        responses.append(resp)

    batch_results = evaluate_batch(prompts, responses, judge_model)
    asr, ci_lower, ci_upper = compute_asr_with_ci(batch_results)
    return {
        "asr": asr,
        "asr_ci_lower": ci_lower,
        "asr_ci_upper": ci_upper,
        "n_valid": len(batch_results.valid_results),
        "n_judge_errors": batch_results.n_errors,
    }


def generate_vanilla_responses(model, tokenizer, prompts) -> list[str]:
    device = next(model.parameters()).device
    responses = []
    for prompt in tqdm(prompts, desc="Vanilla baseline"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=2048, do_sample=True,
                temperature=0.6, top_p=0.95,
            )
        responses.append(tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ))
    return responses


def ablation_k_sensitivity(
    model, tokenizer, prompts, dms_result, judge_model, output_dir
):
    """Ablation A: Vary K from 1 to 50."""
    logger.info("Ablation A: K sensitivity")
    results = []
    k_values = [1, 2, 5, 10, 15, 20, 30, 50]

    for k in k_values:
        # Create modified DMS result with top-K layers
        dms_flat = dms_result.dms_scores.flatten()
        sorted_idx = np.argsort(dms_flat)[::-1][:k]
        modified = copy.deepcopy(dms_result)
        modified.circuit_indices = [(int(i), 0) for i in sorted_idx]
        modified.k = k

        # Only include directions for selected layers
        modified.refusal_directions = {
            (l, h): dms_result.refusal_directions.get((l, h), np.zeros(1))
            for l, h in modified.circuit_indices
            if (l, h) in dms_result.refusal_directions
        }

        if not modified.refusal_directions:
            logger.warning(f"K={k}: no valid refusal directions, skipping")
            continue

        res = evaluate_condition(model, tokenizer, prompts, modified, 0.3, judge_model)
        res["k"] = k
        results.append(res)
        logger.info(f"  K={k}: ASR={res['asr']:.4f}")

    with open(output_dir / "ablation_k.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def ablation_alpha_sweep(
    model, tokenizer, prompts, dms_result, judge_model, output_dir
):
    """Ablation B: Alpha sweep."""
    logger.info("Ablation B: Alpha sweep")
    alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = []

    for alpha in alphas:
        res = evaluate_condition(model, tokenizer, prompts, dms_result, alpha, judge_model)
        res["alpha"] = alpha
        results.append(res)
        logger.info(f"  alpha={alpha}: ASR={res['asr']:.4f}")

    with open(output_dir / "ablation_alpha.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def ablation_delta_sensitivity(
    model, tokenizer, prompts, dms_result, judge_model, output_dir
):
    """Ablation C: Delta sensitivity."""
    logger.info("Ablation C: Delta sensitivity")
    multipliers = [0.5, 0.8, 1.0, 1.2, 1.5]
    base_delta = dms_result.delta_threshold
    results = []

    for mult in multipliers:
        modified = copy.deepcopy(dms_result)
        modified.delta_threshold = base_delta * mult

        res = evaluate_condition(model, tokenizer, prompts, modified, 0.3, judge_model)
        res["delta_multiplier"] = mult
        res["delta_value"] = modified.delta_threshold
        results.append(res)
        logger.info(f"  delta={mult}x ({modified.delta_threshold:.6f}): ASR={res['asr']:.4f}")

    with open(output_dir / "ablation_delta.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def ablation_layer_depth(
    model, tokenizer, prompts, dms_result, judge_model, output_dir, n_layers
):
    """Ablation D: Layer depth analysis."""
    logger.info("Ablation D: Layer depth")

    depth_bands = {
        "early": list(range(0, min(8, n_layers))),
        "middle": list(range(8, min(16, n_layers))),
        "late": list(range(16, n_layers)),
    }
    results = []

    for band_name, band_layers in depth_bands.items():
        modified = copy.deepcopy(dms_result)
        # Restrict circuit to layers in this band
        modified.circuit_indices = [
            (l, h) for l, h in dms_result.circuit_indices if l in band_layers
        ]
        modified.refusal_directions = {
            k: v for k, v in dms_result.refusal_directions.items()
            if k[0] in band_layers
        }
        modified.k = len(modified.circuit_indices)

        if modified.k == 0:
            logger.info(f"  {band_name} ({band_layers[0]}-{band_layers[-1]}): no circuit layers")
            results.append({"band": band_name, "k": 0, "asr": None})
            continue

        res = evaluate_condition(model, tokenizer, prompts, modified, 0.3, judge_model)
        res["band"] = band_name
        res["layers"] = band_layers
        res["k"] = modified.k
        results.append(res)
        logger.info(f"  {band_name} (layers {band_layers[0]}-{band_layers[-1]}, K={modified.k}): ASR={res['asr']:.4f}")

    with open(output_dir / "ablation_depth.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 5: Ablation Studies")
    parser.add_argument("--dms-result", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--attack-prompts", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/exp5")
    parser.add_argument("--n-attempts", type=int, default=100)
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--n-layers", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    with open(args.dms_result, "rb") as f:
        dms_result: DMSResult = pickle.load(f)

    model, tokenizer = load_model_and_tokenizer(ModelConfig(name=args.model))

    with open(args.attack_prompts) as f:
        raw_prompts = json.load(f)

    prompts = prepare_attack_prompts(raw_prompts, args.n_attempts, args.seed)

    # Run all ablations
    all_results = {}

    all_results["k_sensitivity"] = ablation_k_sensitivity(
        model, tokenizer, prompts, dms_result, args.judge_model, output_dir
    )
    all_results["alpha_sweep"] = ablation_alpha_sweep(
        model, tokenizer, prompts, dms_result, args.judge_model, output_dir
    )
    all_results["delta_sensitivity"] = ablation_delta_sensitivity(
        model, tokenizer, prompts, dms_result, args.judge_model, output_dir
    )
    all_results["layer_depth"] = ablation_layer_depth(
        model, tokenizer, prompts, dms_result, args.judge_model, output_dir, args.n_layers
    )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\nAll ablation studies complete. Results saved to " + str(output_dir))


if __name__ == "__main__":
    main()
