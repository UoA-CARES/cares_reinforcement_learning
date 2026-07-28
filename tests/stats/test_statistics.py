import numpy as np
import pytest
from scipy import stats

from cares_reinforcement_learning.stats.statistics import (
    bootstrap_bca_ci,
    friedman_test,
    holm_correction,
    independent_test,
    interquartile_mean,
    nemenyi_critical_difference,
    paired_test,
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


def test_iqm_matches_scipy_trim_mean() -> None:
    values = np.array([9.0, 1.0, 4.0, 8.0, 2.0, 3.0, 7.0, 6.0])

    expected = stats.trim_mean(values, proportiontocut=0.25)

    assert interquartile_mean(values) == pytest.approx(expected)


def test_iqm_is_permutation_invariant() -> None:
    values = np.array([1.0, 2.0, 5.0, 9.0, 10.0, 20.0])
    shuffled = values[[3, 0, 5, 2, 1, 4]]

    assert interquartile_mean(values) == pytest.approx(interquartile_mean(shuffled))


def test_iqm_is_translation_equivariant() -> None:
    values = np.array([-2.0, 0.0, 1.0, 4.0, 8.0])
    offset = 12.5

    assert interquartile_mean(values + offset) == pytest.approx(
        interquartile_mean(values) + offset
    )


def test_iqm_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        interquartile_mean(np.array([], dtype=np.float64))


def test_probability_of_improvement_counts_ties_as_half() -> None:
    baseline = np.array([1.0, 2.0])
    candidate = np.array([2.0, 3.0])

    # Pairings: win, win, tie, win -> (3 + 0.5) / 4.
    assert probability_of_improvement(
        baseline,
        candidate,
        "higher",
    ) == pytest.approx(0.875)


def test_probability_of_improvement_is_complementary() -> None:
    a = np.array([0.0, 2.0, 4.0])
    b = np.array([1.0, 2.0, 5.0])

    p_b_better = probability_of_improvement(a, b, "higher")
    p_a_better = probability_of_improvement(b, a, "higher")

    assert p_a_better + p_b_better == pytest.approx(1.0)


def test_lower_direction_reverses_probability() -> None:
    baseline = np.array([10.0, 12.0])
    candidate = np.array([1.0, 2.0])

    assert probability_of_improvement(baseline, candidate, "lower") == 1.0
    assert probability_of_improvement(baseline, candidate, "higher") == 0.0


def test_probability_of_improvement_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="non-empty samples"):
        probability_of_improvement(
            np.array([], dtype=np.float64),
            np.array([1.0]),
            "higher",
        )


def test_cliffs_delta_identity_and_antisymmetry() -> None:
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])

    probability = probability_of_improvement(a, b, "higher")
    delta_ab = signed_cliffs_delta(a, b, "higher")
    delta_ba = signed_cliffs_delta(b, a, "higher")

    assert delta_ab == pytest.approx(2.0 * probability - 1.0)
    assert delta_ab == pytest.approx(-delta_ba)
    assert -1.0 <= delta_ab <= 1.0


def test_holm_matches_known_step_down_result() -> None:
    raw = np.array([0.01, 0.04, 0.03, 0.20])

    corrected = holm_correction(raw)

    np.testing.assert_allclose(corrected, np.array([0.04, 0.09, 0.09, 0.20]))
    assert np.all(corrected >= raw)
    assert np.all((0.0 <= corrected) & (corrected <= 1.0))


def test_holm_singleton_is_unchanged() -> None:
    np.testing.assert_array_equal(
        holm_correction(np.array([0.37])),
        np.array([0.37]),
    )


@pytest.mark.parametrize(
    "values",
    [
        np.array([[0.1, 0.2]]),
        np.array([0.1, np.nan]),
        np.array([0.1, np.inf]),
    ],
)
def test_holm_rejects_invalid_arrays(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite one-dimensional"):
        holm_correction(values)


def test_bootstrap_bca_is_reproducible_with_fixed_seed() -> None:
    values = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])

    first = bootstrap_bca_ci(
        values,
        np.mean,
        2_000,
        0.95,
        np.random.default_rng(123),
    )
    second = bootstrap_bca_ci(
        values,
        np.mean,
        2_000,
        0.95,
        np.random.default_rng(123),
    )

    assert first == pytest.approx(second)


def test_single_observation_has_exact_bootstrap_interval() -> None:
    values = np.array([7.0])

    assert bootstrap_bca_ci(
        values,
        np.mean,
        100,
        0.95,
        np.random.default_rng(0),
    ) == (7.0, 7.0)


def test_paired_test_matches_scipy() -> None:
    a = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    b = np.array([1.5, 1.0, 4.0, 7.0, 9.0])

    expected = stats.wilcoxon(
        a,
        b,
        alternative="two-sided",
        zero_method="wilcox",
    )
    statistic, p_value = paired_test(a, b)

    assert statistic == pytest.approx(expected.statistic)
    assert p_value == pytest.approx(expected.pvalue)


def test_paired_test_identical_values_returns_neutral_result() -> None:
    values = np.array([1.0, 2.0, 3.0])

    assert paired_test(values, values.copy()) == (0.0, 1.0)


def test_paired_test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        paired_test(np.array([1.0, 2.0]), np.array([1.0]))


def test_independent_test_matches_scipy() -> None:
    a = np.array([1.0, 2.0, 2.0, 4.0])
    b = np.array([3.0, 3.0, 5.0, 6.0])

    expected = stats.mannwhitneyu(
        a,
        b,
        alternative="two-sided",
        method="auto",
    )
    statistic, p_value = independent_test(a, b)

    assert statistic == pytest.approx(expected.statistic)
    assert p_value == pytest.approx(expected.pvalue)


def test_friedman_returns_nan_when_design_is_too_small() -> None:
    statistic, p_value = friedman_test(np.ones((3, 2)))

    assert np.isnan(statistic)
    assert np.isnan(p_value)


def test_friedman_matches_scipy() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
        ]
    )

    expected = stats.friedmanchisquare(
        matrix[:, 0],
        matrix[:, 1],
        matrix[:, 2],
    )
    statistic, p_value = friedman_test(matrix)

    assert statistic == pytest.approx(expected.statistic)
    assert p_value == pytest.approx(expected.pvalue)


def test_nemenyi_matches_explicit_formula() -> None:
    n_algorithms = 4
    n_tasks = 12
    alpha = 0.05

    q_alpha = stats.studentized_range.ppf(
        1.0 - alpha,
        n_algorithms,
        np.inf,
    ) / np.sqrt(2.0)
    expected = q_alpha * np.sqrt(n_algorithms * (n_algorithms + 1.0) / (6.0 * n_tasks))

    assert nemenyi_critical_difference(
        n_algorithms,
        n_tasks,
        alpha,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("n_algorithms", "n_tasks", "alpha"),
    [
        (1, 10, 0.05),
        (3, 1, 0.05),
        (3, 10, 0.0),
        (3, 10, 1.0),
    ],
)
def test_nemenyi_rejects_invalid_arguments(
    n_algorithms: int,
    n_tasks: int,
    alpha: float,
) -> None:
    with pytest.raises(ValueError):
        nemenyi_critical_difference(n_algorithms, n_tasks, alpha)
