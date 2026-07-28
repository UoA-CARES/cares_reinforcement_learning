from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cares_reinforcement_learning.stats.cross_task_analysis import (
    _friedman_and_nemenyi,
    _percentile_interval,
    _probability_a_better,
    _stratified_pairwise_probability_ci,
    _task_superiority,
)
from cares_reinforcement_learning.stats.models import AnalysisOptions


def test_percentile_interval_matches_numpy_quantiles() -> None:
    replicates = np.arange(1.0, 101.0)

    low, high = _percentile_interval(replicates, confidence=0.90)
    expected = np.quantile(replicates, [0.05, 0.95])

    assert low == pytest.approx(expected[0])
    assert high == pytest.approx(expected[1])


@pytest.mark.parametrize(
    "replicates",
    [
        np.array([]),
        np.array([[1.0, 2.0]]),
        np.array([1.0, np.nan]),
    ],
)
def test_percentile_interval_rejects_invalid_replicates(
    replicates: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        _percentile_interval(replicates, confidence=0.95)


def test_task_superiority_averages_probabilities_against_all_opponents() -> None:
    pairwise = pd.DataFrame(
        [
            {
                "task": "task_1",
                "evaluation_metric": "episode_reward",
                "performance_metric": "auc",
                "algorithm_a": "A",
                "algorithm_b": "B",
                "probability_b_better": 0.25,
            },
            {
                "task": "task_1",
                "evaluation_metric": "episode_reward",
                "performance_metric": "auc",
                "algorithm_a": "A",
                "algorithm_b": "C",
                "probability_b_better": 0.75,
            },
            {
                "task": "task_1",
                "evaluation_metric": "episode_reward",
                "performance_metric": "auc",
                "algorithm_a": "B",
                "algorithm_b": "C",
                "probability_b_better": 0.60,
            },
        ]
    )

    result = _task_superiority(pairwise).set_index("algorithm")

    # A better than B = 0.75; A better than C = 0.25.
    assert result.loc["A", "task_superiority"] == pytest.approx(0.50)
    # B better than A = 0.25; B better than C = 0.40.
    assert result.loc["B", "task_superiority"] == pytest.approx(0.325)
    # C better than A = 0.75; C better than B = 0.60.
    assert result.loc["C", "task_superiority"] == pytest.approx(0.675)
    assert (result["opponents"] == 2).all()


def test_probability_a_better_handles_direct_and_reverse_rows() -> None:
    direct = pd.DataFrame(
        [
            {
                "algorithm_a": "A",
                "algorithm_b": "B",
                "probability_b_better": 0.30,
            }
        ]
    )
    reverse = pd.DataFrame(
        [
            {
                "algorithm_a": "B",
                "algorithm_b": "A",
                "probability_b_better": 0.70,
            }
        ]
    )

    assert _probability_a_better(direct, "A", "B") == pytest.approx(0.70)
    assert _probability_a_better(reverse, "A", "B") == pytest.approx(0.70)


def test_probability_a_better_rejects_missing_comparison() -> None:
    rows = pd.DataFrame(columns=["algorithm_a", "algorithm_b", "probability_b_better"])

    with pytest.raises(ValueError, match="Missing pairwise comparison"):
        _probability_a_better(rows, "A", "B")


def test_stratified_probability_ci_is_exact_for_complete_separation() -> None:
    rows: list[dict[str, object]] = []
    for task in ("task_1", "task_2"):
        for seed, value in enumerate((10.0, 11.0, 12.0), start=1):
            rows.append(
                {
                    "task": task,
                    "evaluation_metric": "episode_reward",
                    "algorithm": "A",
                    "seed": seed,
                    "auc": value,
                }
            )
        for seed, value in enumerate((1.0, 2.0, 3.0), start=1):
            rows.append(
                {
                    "task": task,
                    "evaluation_metric": "episode_reward",
                    "algorithm": "B",
                    "seed": seed,
                    "auc": value,
                }
            )

    seed_metrics = pd.DataFrame(rows)
    options = AnalysisOptions(
        bootstrap_samples=200,
        bootstrap_confidence=0.95,
        random_seed=9,
        generate_statistical_figures=False,
    )

    low, high = _stratified_pairwise_probability_ci(
        seed_metrics=seed_metrics,
        tasks=["task_1", "task_2"],
        evaluation_metric="episode_reward",
        performance_metric="auc",
        direction="higher",
        algorithm_a="A",
        algorithm_b="B",
        options=options,
        rng=np.random.default_rng(options.random_seed),
    )

    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_friedman_is_skipped_for_only_two_algorithms() -> None:
    summaries = pd.DataFrame(
        [
            {
                "task": task,
                "evaluation_metric": "episode_reward",
                "performance_metric": "auc",
                "algorithm": algorithm,
                "rank": rank,
            }
            for task in ("task_1", "task_2", "task_3")
            for algorithm, rank in (("A", 1.0), ("B", 2.0))
        ]
    )

    friedman, nemenyi, critical_differences = _friedman_and_nemenyi(
        summaries,
        AnalysisOptions(generate_statistical_figures=False),
    )

    assert friedman.empty
    assert nemenyi.empty
    assert critical_differences == {}


def test_significant_friedman_produces_nemenyi_comparisons() -> None:
    summaries = pd.DataFrame(
        [
            {
                "task": f"task_{task_index}",
                "evaluation_metric": "episode_reward",
                "performance_metric": "auc",
                "algorithm": algorithm,
                "rank": rank,
            }
            for task_index in range(1, 11)
            for algorithm, rank in (
                ("A", 1.0),
                ("B", 2.0),
                ("C", 3.0),
            )
        ]
    )
    options = AnalysisOptions(
        significance_level=0.05,
        generate_statistical_figures=False,
    )

    friedman, nemenyi, critical_differences = _friedman_and_nemenyi(
        summaries,
        options,
    )

    assert len(friedman) == 1
    assert bool(friedman.iloc[0]["significant"])
    assert len(nemenyi) == 3
    assert (
        "episode_reward",
        "auc",
    ) in critical_differences
    assert (nemenyi["critical_difference"] > 0.0).all()


def test_friedman_requires_complete_task_algorithm_matrix() -> None:
    summaries = pd.DataFrame(
        [
            {
                "task": "task_1",
                "evaluation_metric": "episode_reward",
                "performance_metric": "auc",
                "algorithm": "A",
                "rank": 1.0,
            },
            {
                "task": "task_1",
                "evaluation_metric": "episode_reward",
                "performance_metric": "auc",
                "algorithm": "B",
                "rank": 2.0,
            },
            {
                "task": "task_1",
                "evaluation_metric": "episode_reward",
                "performance_metric": "auc",
                "algorithm": "C",
                "rank": 3.0,
            },
            {
                "task": "task_2",
                "evaluation_metric": "episode_reward",
                "performance_metric": "auc",
                "algorithm": "A",
                "rank": 1.0,
            },
            {
                "task": "task_2",
                "evaluation_metric": "episode_reward",
                "performance_metric": "auc",
                "algorithm": "B",
                "rank": 2.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="one rank per task and algorithm"):
        _friedman_and_nemenyi(
            summaries,
            AnalysisOptions(generate_statistical_figures=False),
        )
