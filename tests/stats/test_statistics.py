import numpy as np

from cares_rl_statistics.statistics import (
    holm_correction,
    probability_of_improvement,
    signed_cliffs_delta,
)


def test_probability_and_delta_orientation():
    baseline = np.array([1.0, 2.0])
    candidate = np.array([3.0, 4.0])
    assert probability_of_improvement(baseline, candidate, "higher") == 1.0
    assert signed_cliffs_delta(baseline, candidate, "higher") == 1.0
    assert probability_of_improvement(baseline, candidate, "lower") == 0.0


def test_holm_is_monotone_and_bounded():
    corrected = holm_correction(np.array([0.01, 0.04, 0.03]))
    assert np.all((corrected >= 0.0) & (corrected <= 1.0))
    assert np.allclose(corrected, np.array([0.03, 0.06, 0.06]))


from cares_rl_statistics.statistics import bootstrap_bca_ci, interquartile_mean


def test_bca_interval_contains_observed_iqm():
    values = np.array([1.0, 2.0, 3.0, 4.0, 9.0])
    estimate = interquartile_mean(values)
    low, high = bootstrap_bca_ci(
        values, interquartile_mean, 2_000, 0.95, np.random.default_rng(7)
    )
    assert low <= estimate <= high


def test_constant_sample_has_exact_bootstrap_interval():
    values = np.array([4.0, 4.0, 4.0])
    assert bootstrap_bca_ci(values, np.mean, 100, 0.95, np.random.default_rng(0)) == (
        4.0,
        4.0,
    )
