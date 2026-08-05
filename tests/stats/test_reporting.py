from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cares_reinforcement_learning.reporting.analysis import statistics


def test_child_rng_is_deterministic_for_same_key_sequence() -> None:
    rng_a = statistics.child_rng(7, "task", "metric", 1)
    rng_b = statistics.child_rng(7, "task", "metric", 1)

    sample_a = rng_a.normal(size=5)
    sample_b = rng_b.normal(size=5)

    np.testing.assert_allclose(sample_a, sample_b)


def test_probability_of_improvement_higher_with_ties() -> None:
    baseline = np.asarray([1.0, 2.0], dtype=np.float64)
    candidate = np.asarray([2.0, 3.0], dtype=np.float64)

    probability = statistics.probability_of_improvement(
        baseline,
        candidate,
        "higher",
    )

    # Pair grid: (2 vs 1 win), (2 vs 2 tie), (3 vs 1 win), (3 vs 2 win)
    assert probability == pytest.approx((3.0 + 0.5) / 4.0)


def test_probability_of_improvement_lower_orientation() -> None:
    baseline = np.asarray([5.0, 6.0], dtype=np.float64)
    candidate = np.asarray([3.0, 4.0], dtype=np.float64)

    probability = statistics.probability_of_improvement(
        baseline,
        candidate,
        "lower",
    )

    assert probability == pytest.approx(1.0)


def test_probability_for_algorithm_orients_probabilities() -> None:
    pairwise = pd.DataFrame(
        {
            "algorithm_a": ["algo_a", "algo_c"],
            "algorithm_b": ["algo_b", "algo_a"],
            "probability_b_better": [0.80, 0.40],
        }
    )

    probabilities = statistics.probability_for_algorithm(pairwise, "algo_a")

    # First row has algo_a as baseline => use 1 - p_b
    # Second row has algo_a as candidate => use p_b directly
    np.testing.assert_allclose(np.sort(probabilities), np.asarray([0.20, 0.40]))


def test_probability_for_algorithm_raises_when_missing() -> None:
    pairwise = pd.DataFrame(
        {
            "algorithm_a": ["algo_a"],
            "algorithm_b": ["algo_b"],
            "probability_b_better": [0.80],
        }
    )

    with pytest.raises(ValueError, match="No pairwise comparisons found"):
        statistics.probability_for_algorithm(pairwise, "algo_x")


def test_pairwise_probability_handles_both_row_orientations() -> None:
    frame_ab = pd.DataFrame(
        {
            "algorithm_a": ["algo_a"],
            "algorithm_b": ["algo_b"],
            "probability_b_better": [0.75],
        }
    )
    frame_ba = pd.DataFrame(
        {
            "algorithm_a": ["algo_b"],
            "algorithm_b": ["algo_a"],
            "probability_b_better": [0.25],
        }
    )

    assert statistics.pairwise_probability(
        frame_ab, "algo_b", "algo_a"
    ) == pytest.approx(0.75)
    assert statistics.pairwise_probability(
        frame_ba, "algo_b", "algo_a"
    ) == pytest.approx(0.75)


def test_pairwise_probability_raises_when_not_exactly_one_match() -> None:
    frame = pd.DataFrame(
        {
            "algorithm_a": ["algo_a", "algo_a"],
            "algorithm_b": ["algo_b", "algo_b"],
            "probability_b_better": [0.75, 0.65],
        }
    )

    with pytest.raises(ValueError, match="Expected one comparison"):
        statistics.pairwise_probability(frame, "algo_b", "algo_a")


def test_orient_probability_interval_flips_bounds() -> None:
    probability, low, high = statistics.orient_probability_interval(
        0.30,
        0.20,
        0.40,
        reference_is_a=False,
    )

    assert probability == pytest.approx(0.70)
    assert low == pytest.approx(0.60)
    assert high == pytest.approx(0.80)


def test_add_holm_correction_applies_per_family() -> None:
    frame = pd.DataFrame(
        {
            "evaluation_metric": ["reward", "reward", "reward", "loss", "loss"],
            "performance_metric": ["auc", "auc", "auc", "auc", "auc"],
            "p_value": [0.01, 0.02, 0.04, 0.01, 0.04],
        }
    )

    result = statistics.add_holm_correction(
        frame,
        family_columns=["evaluation_metric", "performance_metric"],
        significance_level=0.05,
    )

    reward_adjusted = result[result["evaluation_metric"] == "reward"][
        "p_value_holm"
    ].to_numpy(dtype=np.float64)
    loss_adjusted = result[result["evaluation_metric"] == "loss"][
        "p_value_holm"
    ].to_numpy(dtype=np.float64)

    # reward family m=3 -> adjusted should be [0.03, 0.04, 0.04]
    np.testing.assert_allclose(np.sort(reward_adjusted), np.asarray([0.03, 0.04, 0.04]))
    # loss family m=2 -> adjusted should be [0.02, 0.04]
    np.testing.assert_allclose(np.sort(loss_adjusted), np.asarray([0.02, 0.04]))


def test_require_single_direction_accepts_consistent_direction() -> None:
    frame = pd.DataFrame({"direction": ["higher", "higher", "higher"]})

    assert statistics.require_single_direction(frame, context="reward") == "higher"


def test_require_single_direction_raises_on_mixed_direction() -> None:
    frame = pd.DataFrame({"direction": ["higher", "lower"]})

    with pytest.raises(ValueError, match="Inconsistent metric direction"):
        statistics.require_single_direction(frame, context="reward")


def test_bootstrap_bca_ci_returns_point_interval_for_constant_sample() -> None:
    values = np.asarray([5.0, 5.0, 5.0], dtype=np.float64)

    low, high = statistics.bootstrap_bca_ci(
        values,
        np.mean,
        samples=100,
        confidence=0.95,
        rng=statistics.child_rng(123, "bootstrap"),
    )

    assert low == pytest.approx(5.0)
    assert high == pytest.approx(5.0)


def test_percentile_interval_matches_quantiles() -> None:
    replicates = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

    low, high = statistics.percentile_interval(replicates, confidence=0.80)

    # alpha=0.1 => q10 and q90
    assert low == pytest.approx(np.quantile(replicates, 0.1))
    assert high == pytest.approx(np.quantile(replicates, 0.9))
