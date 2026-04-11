"""Script: Phase 1 — DMS Circuit Identification (Offline, One-Time Per Model).

Usage:
    python scripts/run_dms_identification.py \
        --model Qwen/Qwen3-14B \
        --advbench-path data/advbench.csv \
        --benign-path data/flan_benign.json \
        --output-dir outputs/dms \
        --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from rad_cot.data.calibration import (
    build_calibration_dataset,
    load_advbench,
    load_flan_benign,
    split_calibration_evaluation,
)
from rad_cot.models.model_loader import get_model_info, load_model_and_tokenizer
from rad_cot.steering.dms import run_dms_identification
from rad_cot.utils.config import ModelConfig, load_config
from rad_cot.utils.logging import init_wandb, setup_logger

logger = setup_logger(__name__)


def find_refusal_token_ids(tokenizer) -> list[int]:
    """Find token IDs for common refusal tokens."""
    refusal_tokens = ["Sorry", "sorry", "cannot", "Cannot", "can't", "Can't", "refuse", "Refuse"]
    ids = []
    for token in refusal_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            ids.append(token_ids[0])
    return list(set(ids))


def main():
    parser = argparse.ArgumentParser(description="RAD-CoT Phase 1: DMS Circuit Identification")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--advbench-path", type=str, required=True)
    parser.add_argument("--benign-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/dms")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.model.name = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    init_wandb(cfg.wandb_project, cfg.wandb_entity, config=vars(args))

    # Load model
    model_cfg = ModelConfig(name=args.model)
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    model_info = get_model_info(args.model)

    # Load data
    harmful_prompts = load_advbench(args.advbench_path)
    benign_prompts = load_flan_benign(args.benign_path, n=cfg.dms.n_benign)
    logger.info(f"Loaded {len(harmful_prompts)} harmful, {len(benign_prompts)} benign prompts")

    # CRITICAL: Split into calibration and evaluation sets to prevent data contamination
    cal_prompts, eval_prompts = split_calibration_evaluation(
        harmful_prompts, calibration_fraction=0.70, seed=args.seed
    )

    # Save the evaluation prompts for use by exp2_safety_eval.py
    eval_path = output_dir / "held_out_eval_prompts.json"
    with open(eval_path, "w") as f:
        json.dump(eval_prompts, f, indent=2)
    logger.info(f"Saved {len(eval_prompts)} held-out evaluation prompts to {eval_path}")
    logger.info(f"Using {len(cal_prompts)} prompts for calibration (disjoint from evaluation)")

    # Build calibration dataset (using ONLY calibration prompts)
    cal_dir = output_dir / "calibration"
    cal_dataset = build_calibration_dataset(
        model=model,
        tokenizer=tokenizer,
        harmful_prompts=cal_prompts,  # ONLY calibration prompts, never evaluation
        benign_prompts=benign_prompts,
        n_refuse=cfg.dms.n_refuse,
        n_comply=cfg.dms.n_comply,
        n_benign=cfg.dms.n_benign,
        min_cot_length=cfg.attack.min_cot_length,
        min_padding_tokens=cfg.attack.min_padding_tokens,
        max_padding_tokens=cfg.attack.max_padding_tokens,
        seed=args.seed,
    )
    cal_dataset.save(cal_dir)
    logger.info(f"Calibration dataset saved to {cal_dir}")

    # Find refusal token IDs
    refusal_token_ids = find_refusal_token_ids(tokenizer)
    logger.info(f"Refusal token IDs: {refusal_token_ids}")

    # Run DMS identification
    dms_result = run_dms_identification(
        model=model,
        tokenizer=tokenizer,
        refuse_prompts=cal_dataset.d_refuse,
        comply_prompts=cal_dataset.d_comply,
        refusal_token_ids=refusal_token_ids,
        n_layers=model_info["n_layers"],
        n_heads=model_info["n_heads"],
        d_head=model_info["d_head"],
        dms_mass_threshold=cfg.dms.dms_mass_threshold,
        delta_fraction=cfg.dms.delta_fraction,
        n_patching_prompts=cfg.dms.n_patching_prompts,
        batch_size=cfg.compute.batch_size,
        max_seq_len=cfg.compute.max_seq_len,
    )

    # Save results
    with open(output_dir / "dms_result.pkl", "wb") as f:
        pickle.dump(dms_result, f)

    np.save(output_dir / "dms_scores.npy", dms_result.dms_scores)
    np.save(output_dir / "delta_scores.npy", dms_result.delta_scores)
    np.save(output_dir / "ce_scores.npy", dms_result.ce_scores)

    summary = {
        "model": args.model,
        "k": dms_result.k,
        "circuit_indices": dms_result.circuit_indices,
        "delta_threshold": dms_result.delta_threshold,
        "top_layers_by_dms": [
            int(i) for i in np.argsort(dms_result.dms_scores.flatten())[::-1][:10]
        ],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"DMS identification complete. K={dms_result.k} layers selected.")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
