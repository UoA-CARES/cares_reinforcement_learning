import numpy as np

from cares_reinforcement_learning.stats.metrics import (
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
