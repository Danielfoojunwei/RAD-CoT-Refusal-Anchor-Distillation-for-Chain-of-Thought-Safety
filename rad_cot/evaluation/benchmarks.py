"""Reasoning benchmark evaluation wrappers.

Wraps lm-eval-harness for GSM8K, MATH, and HumanEval evaluation
to measure reasoning quality preservation under soft steering.
"""

from __future__ import annotations

import subprocess
import json
from dataclasses import dataclass
from pathlib import Path

from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a reasoning benchmark evaluation."""

    benchmark: str
    score: float
    n_samples: int
    details: dict


def run_lm_eval(
    model_name_or_path: str,
    tasks: list[str],
    num_fewshot: int | None = None,
    batch_size: int = 4,
    output_dir: str = "results",
    extra_args: list[str] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run lm-eval-harness evaluation.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        tasks: List of task names (e.g., ["gsm8k", "hendrycks_math"]).
        num_fewshot: Number of few-shot examples (overrides task default).
        batch_size: Batch size for evaluation.
        output_dir: Directory to save results.
        extra_args: Additional CLI arguments for lm_eval.

    Returns:
        Dictionary mapping task names to BenchmarkResult.
    """
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name_or_path},dtype=bfloat16",
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
        "--output_path", output_dir,
    ]

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Running lm-eval: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"lm-eval failed: {result.stderr}")
        raise RuntimeError(f"lm-eval failed with return code {result.returncode}")

    # Parse results
    results = {}
    output_path = Path(output_dir)
    for results_file in output_path.glob("**/results.json"):
        with open(results_file) as f:
            data = json.load(f)

        for task_name, task_results in data.get("results", {}).items():
            acc_key = next(
                (k for k in ["acc,none", "exact_match,none", "pass@1,none", "acc"]
                 if k in task_results),
                None,
            )
            score = task_results.get(acc_key, 0.0) if acc_key else 0.0
            n = task_results.get("n", 0)
            results[task_name] = BenchmarkResult(
                benchmark=task_name,
                score=score,
                n_samples=n,
                details=task_results,
            )

    return results


def evaluate_gsm8k(
    model_name_or_path: str,
    num_fewshot: int = 8,
    output_dir: str = "results/gsm8k",
) -> BenchmarkResult:
    """Evaluate on GSM8K (8-shot)."""
    results = run_lm_eval(
        model_name_or_path, ["gsm8k"], num_fewshot=num_fewshot, output_dir=output_dir
    )
    return results.get("gsm8k", BenchmarkResult("gsm8k", 0.0, 0, {}))


def evaluate_math(
    model_name_or_path: str,
    num_fewshot: int = 4,
    output_dir: str = "results/math",
) -> BenchmarkResult:
    """Evaluate on MATH Levels 4-5 (4-shot)."""
    results = run_lm_eval(
        model_name_or_path,
        ["hendrycks_math"],
        num_fewshot=num_fewshot,
        output_dir=output_dir,
    )
    return results.get("hendrycks_math", BenchmarkResult("hendrycks_math", 0.0, 0, {}))


def evaluate_humaneval(
    model_name_or_path: str,
    output_dir: str = "results/humaneval",
) -> BenchmarkResult:
    """Evaluate on HumanEval (0-shot, pass@1)."""
    results = run_lm_eval(
        model_name_or_path,
        ["humaneval"],
        num_fewshot=0,
        output_dir=output_dir,
        extra_args=["--predict_only"],
    )
    return results.get("humaneval", BenchmarkResult("humaneval", 0.0, 0, {}))


def evaluate_benign_refusal_rate(
    model,
    tokenizer,
    benign_prompts: list[str],
    max_new_tokens: int = 512,
) -> float:
    """Measure false positive rate: fraction of benign prompts incorrectly refused."""
    import torch
    from rad_cot.data.calibration import detect_refusal

    device = next(model.parameters()).device
    n_refused = 0

    for prompt in benign_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        if detect_refusal(generated):
            n_refused += 1

    rate = n_refused / len(benign_prompts) if benign_prompts else 0.0
    logger.info(f"Benign refusal rate: {rate:.4f} ({n_refused}/{len(benign_prompts)})")
    return rate
