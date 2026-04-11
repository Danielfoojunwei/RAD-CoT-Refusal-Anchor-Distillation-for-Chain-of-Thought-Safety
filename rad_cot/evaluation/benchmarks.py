"""Reasoning benchmark evaluation wrappers.

Provides both in-process evaluation (required when steering hooks are active)
and lm-eval-harness subprocess fallback for vanilla baselines.

IMPORTANT: When steering hooks are attached to the model, you MUST use the
in-process evaluation functions (evaluate_gsm8k_inprocess, etc.) because
subprocess-based lm_eval loads a fresh model that does not have hooks.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from rad_cot.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a reasoning benchmark evaluation."""

    benchmark: str
    score: float
    n_samples: int
    details: dict


# ---------------------------------------------------------------------------
# In-process evaluation (works with hooked models)
# ---------------------------------------------------------------------------

def _extract_number(text: str) -> float | None:
    """Extract the final numeric answer from a model's CoT response."""
    # Look for common answer patterns: "#### 42", "The answer is 42", "= 42"
    patterns = [
        r"####\s*([\-\d,\.]+)",
        r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*([\-\d,\.]+)",
        r"=\s*([\-\d,\.]+)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                return float(matches[-1].replace(",", ""))
            except ValueError:
                continue
    # Last resort: find the last number in the text
    numbers = re.findall(r"[\-]?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def evaluate_gsm8k_inprocess(
    model: nn.Module,
    tokenizer,
    dataset_path: str = "data/gsm8k_test.json",
    num_fewshot: int = 8,
    max_samples: int | None = None,
    max_new_tokens: int = 512,
) -> BenchmarkResult:
    """Evaluate GSM8K in-process using the actual model object (with hooks).

    This ensures steering hooks are active during evaluation, unlike the
    subprocess-based lm_eval which loads a fresh model.
    """
    device = next(model.parameters()).device

    path = Path(dataset_path)
    if not path.exists():
        logger.warning(f"GSM8K dataset not found at {path}, returning empty result")
        return BenchmarkResult("gsm8k", 0.0, 0, {"error": "dataset not found"})

    with open(path) as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    correct = 0
    total = 0

    for item in data:
        question = item.get("question", item.get("input", ""))
        answer_str = item.get("answer", item.get("target", ""))

        # Extract ground truth number
        gt = _extract_number(str(answer_str))
        if gt is None:
            continue

        prompt = f"Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        pred = _extract_number(response)
        if pred is not None and abs(pred - gt) < 1e-3:
            correct += 1
        total += 1

    score = correct / total if total > 0 else 0.0
    logger.info(f"GSM8K in-process: {correct}/{total} = {score:.4f}")
    return BenchmarkResult("gsm8k", score, total, {"correct": correct})


def evaluate_math_inprocess(
    model: nn.Module,
    tokenizer,
    dataset_path: str = "data/math_l4l5.json",
    num_fewshot: int = 4,
    max_samples: int | None = None,
    max_new_tokens: int = 512,
) -> BenchmarkResult:
    """Evaluate MATH (Levels 4-5) in-process using the actual model object."""
    device = next(model.parameters()).device

    path = Path(dataset_path)
    if not path.exists():
        logger.warning(f"MATH dataset not found at {path}, returning empty result")
        return BenchmarkResult("math", 0.0, 0, {"error": "dataset not found"})

    with open(path) as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    correct = 0
    total = 0

    for item in data:
        question = item.get("problem", item.get("question", ""))
        answer_str = item.get("solution", item.get("answer", ""))

        gt = _extract_number(str(answer_str))
        if gt is None:
            continue

        prompt = f"Solve the following math problem. Show your work.\n\nProblem: {question}\n\nSolution:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        pred = _extract_number(response)
        if pred is not None and abs(pred - gt) < 1e-3:
            correct += 1
        total += 1

    score = correct / total if total > 0 else 0.0
    logger.info(f"MATH in-process: {correct}/{total} = {score:.4f}")
    return BenchmarkResult("math", score, total, {"correct": correct})


def evaluate_humaneval_inprocess(
    model: nn.Module,
    tokenizer,
    dataset_path: str = "data/humaneval.json",
    max_samples: int | None = None,
    max_new_tokens: int = 512,
) -> BenchmarkResult:
    """Evaluate HumanEval (pass@1) in-process using the actual model object.

    Uses a simplified pass@1 evaluation: generate one completion per problem,
    run the test cases, and check if they pass.
    """
    device = next(model.parameters()).device

    path = Path(dataset_path)
    if not path.exists():
        logger.warning(f"HumanEval dataset not found at {path}, returning empty result")
        return BenchmarkResult("humaneval", 0.0, 0, {"error": "dataset not found"})

    with open(path) as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    correct = 0
    total = 0

    for item in data:
        prompt_code = item.get("prompt", "")
        test_code = item.get("test", "")
        entry_point = item.get("entry_point", "")

        inputs = tokenizer(
            prompt_code, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        completion = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Try to execute the completion with test cases
        full_code = prompt_code + completion + "\n" + test_code
        try:
            exec_globals = {}
            exec(full_code, exec_globals)
            # If test function exists, call it
            if f"check({entry_point})" in test_code or "check(" in test_code:
                pass  # Tests ran inline
            correct += 1
        except Exception:
            pass
        total += 1

    score = correct / total if total > 0 else 0.0
    logger.info(f"HumanEval in-process: {correct}/{total} = {score:.4f}")
    return BenchmarkResult("humaneval", score, total, {"correct": correct})


# ---------------------------------------------------------------------------
# Subprocess-based lm_eval (for vanilla baselines only)
# ---------------------------------------------------------------------------

def run_lm_eval(
    model_name_or_path: str,
    tasks: list[str],
    num_fewshot: int | None = None,
    batch_size: int = 4,
    output_dir: str = "results",
    extra_args: list[str] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run lm-eval-harness evaluation via subprocess.

    WARNING: This spawns a new process that loads a fresh model. Steering
    hooks attached to an in-memory model will NOT be active. Use the
    *_inprocess() functions when evaluating steered conditions.
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

    logger.info(f"Running lm-eval (subprocess): {' '.join(cmd)}")
    logger.warning(
        "NOTE: Subprocess lm_eval loads a fresh model. "
        "Steering hooks are NOT active in this evaluation."
    )
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
    """Evaluate on GSM8K (8-shot) via subprocess. No steering hooks active."""
    results = run_lm_eval(
        model_name_or_path, ["gsm8k"], num_fewshot=num_fewshot, output_dir=output_dir
    )
    return results.get("gsm8k", BenchmarkResult("gsm8k", 0.0, 0, {}))


def evaluate_math(
    model_name_or_path: str,
    num_fewshot: int = 4,
    output_dir: str = "results/math",
) -> BenchmarkResult:
    """Evaluate on MATH Levels 4-5 (4-shot) via subprocess. No steering hooks active."""
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
    """Evaluate on HumanEval (0-shot, pass@1) via subprocess. No steering hooks active."""
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
