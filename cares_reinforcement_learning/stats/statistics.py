from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats

from cares_reinforcement_learning.stats.models import MetricDirection

Statistic = Callable[[npt.NDArray[np.float64]], float]


def interquartile_mean(values: Sequence[float] | npt.NDArray[np.float64]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("IQM requires at least one value.")
    return float(stats.trim_mean(array, proportiontocut=0.25))


def bootstrap_bca_ci(
    values: Sequence[float] | npt.NDArray[np.float64],
    statistic: Statistic,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """BCa interval from independent observations."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Bootstrap CI requires at least one value.")
    estimate = float(statistic(array))
    if array.size == 1 or np.all(array == array[0]):
        return estimate, estimate

    result = stats.bootstrap(
        (array,),
        statistic,
        n_resamples=samples,
        confidence_level=confidence,
        method="BCa",
        rng=rng,
        vectorized=False,
    )
    low = float(result.confidence_interval.low)
    high = float(result.confidence_interval.high)
    if not np.isfinite([low, high]).all():
        raise ValueError(
            "BCa interval is undefined for this sample/statistic. "
            "Use more independent observations rather than silently changing methods."
        )
    return low, high


def probability_of_improvement(
    baseline: npt.NDArray[np.float64],
    candidate: npt.NDArray[np.float64],
    direction: MetricDirection,
) -> float:
    """Probability that a random candidate observation beats a random baseline."""
    if baseline.size == 0 or candidate.size == 0:
        raise ValueError("Probability of improvement requires non-empty samples.")
    differences = candidate[:, None] - baseline[None, :]
    if direction == "lower":
        differences = -differences
    wins = np.count_nonzero(differences > 0)
    ties = np.count_nonzero(differences == 0)
    return float((wins + 0.5 * ties) / differences.size)


def signed_cliffs_delta(
    baseline: npt.NDArray[np.float64],
    candidate: npt.NDArray[np.float64],
    direction: MetricDirection,
) -> float:
    """Cliff's delta oriented so positive means candidate is better."""
    return 2.0 * probability_of_improvement(baseline, candidate, direction) - 1.0


def holm_correction(p_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("Holm correction requires a finite one-dimensional array.")
    order = np.argsort(p_values)
    corrected = np.empty_like(p_values)
    running = 0.0
    count = p_values.size
    for position, index in enumerate(order):
        running = max(running, (count - position) * p_values[index])
        corrected[index] = min(1.0, running)
    return corrected


def paired_test(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> tuple[float, float]:
    if a.shape != b.shape:
        raise ValueError("Paired samples must have identical shapes.")
    if np.allclose(a, b):
        return 0.0, 1.0
    result = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def independent_test(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> tuple[float, float]:
    result = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def friedman_test(matrix: npt.NDArray[np.float64]) -> tuple[float, float]:
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 3:
        return float("nan"), float("nan")
    result = stats.friedmanchisquare(
        *[matrix[:, index] for index in range(matrix.shape[1])]
    )
    return float(result.statistic), float(result.pvalue)


def nemenyi_critical_difference(
    n_algorithms: int,
    n_tasks: int,
    alpha: float = 0.05,
) -> float:
    """Critical difference for the two-sided Nemenyi average-rank comparison.

    SciPy's studentized-range quantile is divided by sqrt(2), matching the
    q_alpha values conventionally used for the Nemenyi test.
    """
    if n_algorithms < 2:
        raise ValueError("Nemenyi comparison requires at least two algorithms.")
    if n_tasks < 2:
        raise ValueError("Nemenyi comparison requires at least two tasks.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")

    q_alpha = float(
        stats.studentized_range.ppf(1.0 - alpha, n_algorithms, np.inf) / np.sqrt(2.0)
    )
    return q_alpha * np.sqrt(n_algorithms * (n_algorithms + 1.0) / (6.0 * n_tasks))
