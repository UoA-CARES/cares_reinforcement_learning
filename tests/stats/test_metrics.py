from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cares_reinforcement_learning.stats.metrics import (
    aggregate_evaluation_curve,
    compute_curve_metrics,
    trapezoidal_auc,
    window_auc,
)


def test_constant_curve_raw_auc():
    steps = np.array([10.0, 20.0, 30.0])
    values = np.array([5.0, 5.0, 5.0])
    assert trapezoidal_auc(steps, values) == 100.0


def test_linear_curve_matches_analytical_area():
    steps = np.array([0.0, 1.0, 3.0])
    values = 2.0 * steps + 1.0
    assert trapezoidal_auc(steps, values) == 12.0


def test_window_auc_interpolates_exact_boundary():
    steps = np.array([0.0, 10.0])
    values = np.array([0.0, 10.0])
    assert window_auc(steps, values, 0.0, 2.5) == 3.125


def test_default_windows_are_raw_areas():
    steps = np.array([0.0, 10.0])
    values = np.array([2.0, 2.0])
    result = compute_curve_metrics(steps, values, 0.25, 0.10)
    assert result == {"auc": 20.0, "early_window_auc": 5.0, "final_window_auc": 2.0}


def test_aggregate_evaluation_curve_averages_episodes_per_step() -> None:
    frame = pd.DataFrame(
        {
            "total_steps": [20, 10, 20, 10],
            "episode_reward": [6.0, 1.0, 10.0, 3.0],
        }
    )

    steps, values = aggregate_evaluation_curve(frame, "episode_reward")

    np.testing.assert_array_equal(steps, np.array([10.0, 20.0]))
    np.testing.assert_allclose(values, np.array([2.0, 8.0]))


def test_aggregate_evaluation_curve_requires_two_distinct_steps() -> None:
    frame = pd.DataFrame(
        {
            "total_steps": [10, 10],
            "episode_reward": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="two distinct evaluation steps"):
        aggregate_evaluation_curve(frame, "episode_reward")


def test_trapezoidal_auc_handles_irregular_spacing() -> None:
    steps = np.array([0.0, 1.0, 4.0])
    values = np.array([0.0, 2.0, 2.0])

    # First trapezoid: 1.0. Second trapezoid: 6.0.
    assert trapezoidal_auc(steps, values) == pytest.approx(7.0)


def test_trapezoidal_auc_handles_negative_values() -> None:
    steps = np.array([0.0, 2.0])
    values = np.array([-3.0, -1.0])

    assert trapezoidal_auc(steps, values) == pytest.approx(-4.0)


def test_window_auc_interpolates_both_boundaries() -> None:
    steps = np.array([0.0, 10.0, 20.0])
    values = np.array([0.0, 10.0, 0.0])

    # At x=5 and x=15 the interpolated value is 5.
    # The area is two trapezoids, each with area 37.5.
    assert window_auc(steps, values, 5.0, 15.0) == pytest.approx(75.0)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1.0, 5.0),
        (5.0, 11.0),
        (5.0, 5.0),
    ],
)
def test_window_auc_rejects_invalid_windows(start: float, end: float) -> None:
    steps = np.array([0.0, 10.0])
    values = np.array([0.0, 10.0])

    with pytest.raises(ValueError, match="outside the observed evaluation interval"):
        window_auc(steps, values, start, end)


def test_full_fraction_windows_equal_full_auc() -> None:
    steps = np.array([0.0, 2.0, 5.0])
    values = np.array([1.0, 3.0, 2.0])

    result = compute_curve_metrics(
        steps,
        values,
        early_fraction=1.0,
        final_fraction=1.0,
    )

    assert result["early_window_auc"] == pytest.approx(result["auc"])
    assert result["final_window_auc"] == pytest.approx(result["auc"])


def test_auc_scales_linearly_with_metric_values() -> None:
    steps = np.array([0.0, 1.0, 3.0])
    values = np.array([2.0, -1.0, 4.0])
    scale = 3.5

    assert trapezoidal_auc(steps, scale * values) == pytest.approx(
        scale * trapezoidal_auc(steps, values)
    )
