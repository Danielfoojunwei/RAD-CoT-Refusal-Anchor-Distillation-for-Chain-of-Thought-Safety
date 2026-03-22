"""Download all required datasets for RAD-CoT experiments.

Downloads:
  - AdvBench (520 harmful prompts)
  - HarmBench (320 behaviors)
  - GSM8K test set
  - MATH dataset
  - HumanEval
  - FLAN-v2 benign subset

Usage:
    python scripts/download_datasets.py --output-dir data
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)


def download_advbench(output_dir: Path):
    """Download AdvBench harmful behaviors dataset.

    Tries multiple sources. AdvBench is often gated; falls back to
    alternative repos or instructs user to provide manually.
    """
    from datasets import load_dataset

    logger.info("Downloading AdvBench...")
    sources = [
        ("walledai/AdvBench", "prompt"),
        ("thu-coai/AISafetyLab_Datasets/AdvBench", "goal"),
    ]
    for repo, key in sources:
        try:
            ds = load_dataset(repo, split="train")
            prompts = [row[key] for row in ds]
            output_path = output_dir / "advbench.json"
            with open(output_path, "w") as f:
                json.dump(prompts, f, indent=2)
            logger.info(f"AdvBench: {len(prompts)} prompts saved to {output_path}")
            return
        except Exception:
            continue
    raise RuntimeError(
        "AdvBench requires gated access. Set HF_TOKEN or place advbench.json in data/ manually. "
        "See: https://huggingface.co/datasets/walledai/AdvBench"
    )


def download_harmbench(output_dir: Path):
    """Download HarmBench behaviors dataset."""
    from datasets import load_dataset

    logger.info("Downloading HarmBench...")
    sources = [
        ("walledai/HarmBench", "prompt"),
        ("harmbench/behaviors", "Behavior"),
    ]
    for repo, key in sources:
        try:
            ds = load_dataset(repo, split="train")
            behaviors = [row[key] for row in ds]
            output_path = output_dir / "harmbench.json"
            with open(output_path, "w") as f:
                json.dump(behaviors, f, indent=2)
            logger.info(f"HarmBench: {len(behaviors)} behaviors saved to {output_path}")
            return
        except Exception:
            continue
    raise RuntimeError(
        "HarmBench requires gated access. Set HF_TOKEN or place harmbench.json in data/ manually. "
        "See: https://huggingface.co/datasets/walledai/HarmBench"
    )


def download_gsm8k(output_dir: Path):
    """Download GSM8K test set."""
    from datasets import load_dataset

    logger.info("Downloading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    data = [{"question": row["question"], "answer": row["answer"]} for row in ds]
    output_path = output_dir / "gsm8k_test.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"GSM8K: {len(data)} problems saved to {output_path}")


def download_math(output_dir: Path):
    """Download MATH dataset (Levels 4-5)."""
    from datasets import load_dataset

    logger.info("Downloading MATH dataset...")
    ds = load_dataset("lighteval/MATH-Hard", split="test")
    # Filter for levels 4 and 5
    data = [
        {"problem": row["problem"], "solution": row["solution"], "level": row["level"]}
        for row in ds
        if row["level"] in ["Level 4", "Level 5"]
    ]
    output_path = output_dir / "math_l4l5.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"MATH L4-5: {len(data)} problems saved to {output_path}")


def download_humaneval(output_dir: Path):
    """Download HumanEval dataset."""
    from datasets import load_dataset

    logger.info("Downloading HumanEval...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    data = [
        {"task_id": row["task_id"], "prompt": row["prompt"], "canonical_solution": row["canonical_solution"],
         "test": row["test"], "entry_point": row["entry_point"]}
        for row in ds
    ]
    output_path = output_dir / "humaneval.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"HumanEval: {len(data)} problems saved to {output_path}")


def download_flan_benign(output_dir: Path, n: int = 500):
    """Download benign prompts from FLAN-v2 for false-positive calibration."""
    from datasets import load_dataset

    logger.info("Downloading FLAN-v2 benign subset...")
    ds = load_dataset("Muennighoff/flan", split="train", streaming=True)
    prompts = []
    for row in ds:
        if len(prompts) >= n:
            break
        instruction = row.get("inputs", row.get("instruction", ""))
        if instruction and len(instruction) > 20:
            prompts.append(instruction)

    output_path = output_dir / "flan_benign.json"
    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=2)
    logger.info(f"FLAN benign: {len(prompts)} prompts saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download all RAD-CoT datasets")
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaders = [
        ("AdvBench", download_advbench),
        ("HarmBench", download_harmbench),
        ("GSM8K", download_gsm8k),
        ("MATH", download_math),
        ("HumanEval", download_humaneval),
        ("FLAN benign", download_flan_benign),
    ]

    results = {}
    for name, fn in downloaders:
        try:
            fn(output_dir)
            results[name] = "success"
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            results[name] = f"failed: {e}"

    logger.info("\nDownload summary:")
    for name, status in results.items():
        logger.info(f"  {name}: {status}")


if __name__ == "__main__":
    main()
