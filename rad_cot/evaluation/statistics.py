"""Statistical utilities for rigorous evaluation reporting.

Provides bootstrap confidence intervals, significance testing, and
multi-seed aggregation for all RAD-CoT experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricWithCI:
    """A metric value with confidence interval."""

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    confidence: float = 0.95

    def __str__(self) -> str:
        return (
            f"{self.mean:.4f} +/- {self.std:.4f} "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"(n={self.n_samples}, {self.confidence:.0%} CI)"
        )


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> MetricWithCI:
    """Compute bootstrap confidence interval for the mean of values.

    Args:
        values: 1D array of observations.
        confidence: Confidence level (default 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        MetricWithCI with mean, std, and CI bounds.
    """
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return MetricWithCI(0.0, 0.0, 0.0, 0.0, 0, confidence)

    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return MetricWithCI(
        mean=float(values.mean()),
        std=float(values.std()),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_samples=n,
        confidence=confidence,
    )


def clopper_pearson_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute exact Clopper-Pearson confidence interval for a proportion.

    More appropriate than bootstrap for small sample sizes (n < 30).
    """
    from scipy import stats

    if total == 0:
        return 0.0, 0.0

    alpha = 1 - confidence
    lower = stats.beta.ppf(alpha / 2, successes, total - successes + 1) if successes > 0 else 0.0
    upper = stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes) if successes < total else 1.0

    return float(lower), float(upper)


def aggregate_multi_seed(
    per_seed_values: list[float],
    confidence: float = 0.95,
) -> MetricWithCI:
    """Aggregate results across multiple random seeds.

    Args:
        per_seed_values: List of metric values, one per seed/run.
        confidence: Confidence level.

    Returns:
        MetricWithCI with mean, std, and CI.
    """
    values = np.asarray(per_seed_values)
    n = len(values)
    if n == 0:
        return MetricWithCI(0.0, 0.0, 0.0, 0.0, 0, confidence)

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0

    # Use t-distribution for small n
    if n > 1:
        from scipy import stats
        t_val = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
        margin = t_val * std / np.sqrt(n)
        ci_lower = mean - margin
        ci_upper = mean + margin
    else:
        ci_lower = mean
        ci_upper = mean

    return MetricWithCI(mean, std, ci_lower, ci_upper, n, confidence)


def paired_significance_test(
    values_a: list[float],
    values_b: list[float],
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test for comparing two conditions.

    Use when comparing the same prompts under different conditions
    (e.g., vanilla vs. steered on the same evaluation set).

    Args:
        values_a: Per-sample metric for condition A.
        values_b: Per-sample metric for condition B.
        alternative: "two-sided", "greater", or "less".

    Returns:
        (statistic, p_value).
    """
    from scipy import stats

    a = np.asarray(values_a)
    b = np.asarray(values_b)

    if len(a) != len(b):
        raise ValueError("Arrays must have the same length for paired test")

    if len(a) < 10:
        # Too few samples for meaningful test
        return float("nan"), float("nan")

    stat, p_value = stats.wilcoxon(a, b, alternative=alternative)
    return float(stat), float(p_value)
