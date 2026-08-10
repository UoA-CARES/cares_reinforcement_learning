from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from cares_reinforcement_learning.reporting.analysis.models import MetricDirection

Statistic = Callable[[npt.NDArray[np.float64]], float]


def _one_dimensional_finite_array(
    values: Sequence[float] | npt.NDArray[np.float64],
    *,
    name: str,
) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} requires a finite one-dimensional non-empty array.")
    return array


def child_rng(base_seed: int, *keys: Hashable) -> np.random.Generator:
    """Create a deterministic RNG whose output is independent of call order."""
    digest = hashlib.sha256(str(int(base_seed)).encode("utf-8"))
    for key in keys:
        digest.update(b"\0")
        digest.update(repr(key).encode("utf-8"))
    seed = int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)
    return np.random.default_rng(seed)


def interquartile_mean(
    values: Sequence[float] | npt.NDArray[np.float64],
) -> float:
    array = _one_dimensional_finite_array(values, name="IQM")
    return float(stats.trim_mean(array, proportiontocut=0.25))


def bootstrap_bca_ci(
    values: Sequence[float] | npt.NDArray[np.float64],
    statistic: Statistic,
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Calculate a BCa interval from independent observations."""
    array = _one_dimensional_finite_array(values, name="Bootstrap CI")

    if samples < 1:
        raise ValueError("Bootstrap samples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap confidence must be in (0, 1).")

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


def percentile_interval(
    bootstrap_replicates: Sequence[float] | npt.NDArray[np.float64],
    confidence: float,
) -> tuple[float, float]:
    """Return a two-sided percentile interval from bootstrap replicates."""
    replicates = _one_dimensional_finite_array(
        bootstrap_replicates,
        name="Percentile interval",
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be in (0, 1).")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(replicates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def probability_of_improvement(
    baseline: npt.NDArray[np.float64],
    candidate: npt.NDArray[np.float64],
    direction: MetricDirection,
) -> float:
    """Probability that a random candidate observation beats a random baseline."""
    baseline = _one_dimensional_finite_array(
        baseline,
        name="Probability baseline",
    )
    candidate = _one_dimensional_finite_array(
        candidate,
        name="Probability candidate",
    )
    differences = candidate[:, None] - baseline[None, :]
    if direction == "lower":
        differences = -differences
    wins = np.count_nonzero(differences > 0)
    ties = np.count_nonzero(differences == 0)
    return float((wins + 0.5 * ties) / differences.size)


def probability_for_algorithm(
    pairwise_rows: pd.DataFrame,
    algorithm: str,
    *,
    algorithm_a_column: str = "algorithm_a",
    algorithm_b_column: str = "algorithm_b",
    probability_b_column: str = "probability_b_better",
) -> npt.NDArray[np.float64]:
    """Orient all relevant pairwise probabilities towards one algorithm."""
    relevant = pairwise_rows[
        (pairwise_rows[algorithm_a_column] == algorithm)
        | (pairwise_rows[algorithm_b_column] == algorithm)
    ]
    if relevant.empty:
        raise ValueError(f"No pairwise comparisons found for {algorithm!r}.")

    probabilities = np.where(
        relevant[algorithm_b_column].to_numpy() == algorithm,
        relevant[probability_b_column].to_numpy(dtype=np.float64),
        1.0 - relevant[probability_b_column].to_numpy(dtype=np.float64),
    )
    return np.asarray(probabilities, dtype=np.float64)


def pairwise_probability(
    pairwise_rows: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    algorithm_a_column: str = "algorithm_a",
    algorithm_b_column: str = "algorithm_b",
    probability_b_column: str = "probability_b_better",
) -> float:
    """Return P(candidate better than baseline) from either stored orientation."""
    matches = pairwise_rows[
        (
            (pairwise_rows[algorithm_a_column] == baseline)
            & (pairwise_rows[algorithm_b_column] == candidate)
        )
        | (
            (pairwise_rows[algorithm_a_column] == candidate)
            & (pairwise_rows[algorithm_b_column] == baseline)
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one comparison for {candidate!r} vs {baseline!r}; "
            f"found {len(matches)}."
        )

    row = matches.iloc[0]
    probability_b = float(row[probability_b_column])
    return (
        probability_b
        if str(row[algorithm_b_column]) == candidate
        else 1.0 - probability_b
    )


def orient_probability_interval(
    probability_a: float,
    low_a: float,
    high_a: float,
    *,
    reference_is_a: bool,
) -> tuple[float, float, float]:
    """Orient a probability and confidence interval towards a reference method."""
    if reference_is_a:
        return probability_a, low_a, high_a
    return 1.0 - probability_a, 1.0 - high_a, 1.0 - low_a


def _holm_correction(
    p_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Holm correction requires a finite one-dimensional array.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("P-values must lie in [0, 1].")
    if values.size == 0:
        return values.copy()

    order = np.argsort(values)
    corrected = np.empty_like(values)
    running = 0.0
    count = values.size
    for position, index in enumerate(order):
        running = max(running, (count - position) * values[index])
        corrected[index] = min(1.0, running)
    return corrected


def add_holm_correction(
    frame: pd.DataFrame,
    *,
    family_columns: Sequence[str],
    significance_level: float,
    p_value_column: str = "p_value",
    adjusted_column: str = "p_value_holm",
    significant_column: str = "significant_holm",
) -> pd.DataFrame:
    """Apply Holm correction independently within each comparison family."""
    if not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must be in (0, 1).")

    result = frame.copy()
    result[adjusted_column] = np.nan
    result[significant_column] = False
    if result.empty:
        return result

    for indices in result.groupby(list(family_columns), sort=False).groups.values():
        index_list = list(indices)
        adjusted = _holm_correction(
            result.loc[index_list, p_value_column].to_numpy(dtype=np.float64)
        )
        result.loc[index_list, adjusted_column] = adjusted
        result.loc[index_list, significant_column] = adjusted < significance_level
    return result


def require_single_direction(
    frame: pd.DataFrame,
    *,
    context: str,
    column: str = "direction",
) -> MetricDirection:
    values = frame[column].dropna().astype(str).unique()
    if values.size != 1 or values[0] not in {"higher", "lower"}:
        raise ValueError(f"Inconsistent metric direction for {context}.")
    return values[0]  # type: ignore[return-value]


def paired_test(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> tuple[float, float]:
    a = _one_dimensional_finite_array(a, name="Paired sample A")
    b = _one_dimensional_finite_array(b, name="Paired sample B")
    if a.shape != b.shape:
        raise ValueError("Paired samples must have identical shapes.")
    if np.allclose(a, b):
        return 0.0, 1.0
    result = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def independent_test(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> tuple[float, float]:
    a = _one_dimensional_finite_array(a, name="Independent sample A")
    b = _one_dimensional_finite_array(b, name="Independent sample B")
    result = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def friedman_test(matrix: npt.NDArray[np.float64]) -> tuple[float, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 3:
        return float("nan"), float("nan")
    if not np.isfinite(matrix).all():
        raise ValueError("Friedman test matrix must be finite.")
    result = stats.friedmanchisquare(
        *[matrix[:, index] for index in range(matrix.shape[1])]
    )
    return float(result.statistic), float(result.pvalue)


def nemenyi_critical_difference(
    n_algorithms: int,
    n_tasks: int,
    alpha: float = 0.05,
) -> float:
    """Critical difference for the two-sided Nemenyi average-rank comparison."""
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
