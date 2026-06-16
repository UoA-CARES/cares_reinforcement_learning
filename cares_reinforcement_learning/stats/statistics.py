from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from .data import MetricDirection


def interquartile_mean(values: Sequence[float]) -> float:
    return float(stats.trim_mean(values, proportiontocut=0.25))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    samples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)

    if array.size == 0:
        return float("nan"), float("nan")

    if array.size == 1:
        value = float(statistic(array.tolist()))
        return value, value

    def scipy_statistic(sample: Any) -> float:
        sample_array = np.asarray(sample, dtype=np.float64)
        return float(statistic(sample_array.tolist()))

    result = stats.bootstrap(
        data=(array,),
        statistic=scipy_statistic,
        n_resamples=samples,
        confidence_level=confidence,
        method="percentile",
        rng=rng,
    )

    return (
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def probability_of_improvement(
    baseline: npt.NDArray[np.float64],
    candidate: npt.NDArray[np.float64],
    direction: MetricDirection,
) -> float:

    if baseline.size == 0 or candidate.size == 0:
        return float("nan")

    wins = 0.0
    total = 0
    for candidate_value in candidate:
        for baseline_value in baseline:
            if direction == "higher":
                if candidate_value > baseline_value:
                    wins += 1.0
                elif candidate_value == baseline_value:
                    wins += 0.5
            else:
                if candidate_value < baseline_value:
                    wins += 1.0
                elif candidate_value == baseline_value:
                    wins += 0.5
            total += 1
    return wins / total


def signed_cliffs_delta(
    baseline: npt.NDArray[np.float64],
    candidate: npt.NDArray[np.float64],
    direction: MetricDirection,
) -> float:
    """Cliff's delta, signed so positive favours the candidate algorithm."""
    if baseline.size == 0 or candidate.size == 0:
        return float("nan")

    more = 0
    less = 0

    for candidate_value in candidate:
        for baseline_value in baseline:
            if candidate_value > baseline_value:
                more += 1
            elif candidate_value < baseline_value:
                less += 1

    delta = (more - less) / float(candidate.size * baseline.size)

    if direction == "lower":
        delta *= -1.0

    return float(delta)


def holm_correction(p_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply Holm-Bonferroni correction for multiple comparisons."""

    corrected = np.empty_like(p_values, dtype=np.float64)
    order = np.argsort(p_values)

    previous = 0.0
    n_tests = len(p_values)

    for rank, index in enumerate(order):
        adjusted = (n_tests - rank) * p_values[index]
        adjusted = max(previous, adjusted)

        corrected[index] = min(1.0, adjusted)
        previous = corrected[index]

    return corrected


def friedman_rank_test(
    task_summary: pd.DataFrame,
    evaluation_metric: str,
    performance_metric: str = "auc",
) -> dict[str, float]:
    selected = task_summary[
        (task_summary["evaluation_metric"] == evaluation_metric)
        & (task_summary["performance_metric"] == performance_metric)
    ].copy()

    rank_matrix = selected.pivot(
        index="task",
        columns="algorithm",
        values="rank",
    )

    if rank_matrix.isna().any().any():  # type: ignore[no-untyped-call]
        raise ValueError(
            "Cross-task rank matrix contains missing values. "
            "Every algorithm must have results for every task."
        )

    if rank_matrix.shape[0] < 2 or rank_matrix.shape[1] < 2:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_tasks": float(rank_matrix.shape[0]),
            "n_algorithms": float(rank_matrix.shape[1]),
        }

    result = stats.friedmanchisquare(
        *[
            np.asarray(rank_matrix[column], dtype=np.float64)
            for column in rank_matrix.columns
        ]
    )

    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_tasks": float(rank_matrix.shape[0]),
        "n_algorithms": float(rank_matrix.shape[1]),
    }
