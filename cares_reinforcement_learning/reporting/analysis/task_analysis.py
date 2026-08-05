from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

import cares_reinforcement_learning.reporting.analysis.metrics as metrics
import cares_reinforcement_learning.reporting.analysis.validation as validation_module
from cares_reinforcement_learning.reporting.analysis import statistics
from cares_reinforcement_learning.reporting.analysis.models import (
    AnalysisOptions,
    ComparisonDesign,
    MetricDirection,
    MetricSpec,
    TaskAnalysisResult,
    ValidationResult,
)
from cares_reinforcement_learning.reporting.models import LoadedRun


def _seed_metrics(
    runs: Sequence[LoadedRun],
    metric_specs: Sequence[MetricSpec],
    options: AnalysisOptions,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in runs:
        for seed, seed_run in sorted(run.seeds.items()):
            for spec in metric_specs:
                prepared = metrics.ensure_required_columns(
                    seed_run.eval_data,
                    [options.evaluation_step_column, spec.column],
                    context=f"{run.comparison_name} seed {seed}",
                )
                steps, values = metrics.aggregate_evaluation_curve(
                    prepared,
                    spec.column,
                    options.evaluation_step_column,
                )
                rows.append(
                    {
                        "algorithm": run.comparison_name,
                        "algorithm_family": run.algorithm,
                        "variant_parameters": dict(run.variant_parameters),
                        "seed": seed,
                        "evaluation_metric": spec.column,
                        "direction": spec.direction,
                        "n_eval_steps": int(steps.size),
                        "first_step": float(steps[0]),
                        "last_step": float(steps[-1]),
                        **metrics.compute_curve_metrics(
                            steps,
                            values,
                            options.early_window_fraction,
                            options.final_window_fraction,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _algorithm_summary(
    seed_metrics: pd.DataFrame,
    options: AnalysisOptions,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = seed_metrics.groupby(
        ["algorithm", "evaluation_metric", "direction"],
        sort=False,
    )

    for (algorithm, metric, direction), group in groups:
        for performance_metric in metrics.PERFORMANCE_METRICS:
            values = np.asarray(group[performance_metric], dtype=np.float64)
            rng_key = ("task", algorithm, metric, performance_metric)
            mean_ci = statistics.bootstrap_bca_ci(
                values,
                np.mean,
                options.bootstrap_samples,
                options.bootstrap_confidence,
                statistics.child_rng(options.random_seed, *rng_key, "mean"),
            )
            iqm_ci = statistics.bootstrap_bca_ci(
                values,
                statistics.interquartile_mean,
                options.bootstrap_samples,
                options.bootstrap_confidence,
                statistics.child_rng(options.random_seed, *rng_key, "iqm"),
            )
            rows.append(
                {
                    "algorithm": algorithm,
                    "evaluation_metric": metric,
                    "direction": direction,
                    "performance_metric": performance_metric,
                    "n_seeds": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "mean_ci_low": mean_ci[0],
                    "mean_ci_high": mean_ci[1],
                    "iqm": statistics.interquartile_mean(values),
                    "iqm_ci_low": iqm_ci[0],
                    "iqm_ci_high": iqm_ci[1],
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "bootstrap_method": "BCa",
                    "bootstrap_resampling_unit": "seed",
                    "bootstrap_samples": options.bootstrap_samples,
                    "bootstrap_confidence": options.bootstrap_confidence,
                }
            )

    result = pd.DataFrame(rows)
    ranked: list[pd.DataFrame] = []
    for key, group in result.groupby(metrics.METRIC_GROUP_COLUMNS, sort=False):
        current = group.copy()
        direction = statistics.require_single_direction(
            current,
            context=f"{key[0]!r}/{key[1]!r}",
        )
        current["rank"] = current["iqm"].rank(
            method="average",
            ascending=direction == "lower",
        )
        ranked.append(current)
    return pd.concat(ranked, ignore_index=True)


def _comparison_samples(
    a_group: pd.DataFrame,
    b_group: pd.DataFrame,
    performance_metric: str,
    design: ComparisonDesign,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], str, int]:
    if design is ComparisonDesign.PAIRED:
        merged = a_group[["seed", performance_metric]].merge(
            b_group[["seed", performance_metric]],
            on="seed",
            suffixes=("_a", "_b"),
        )
        a_values: npt.NDArray[np.float64] = np.asarray(
            merged[f"{performance_metric}_a"], dtype=np.float64
        )
        b_values: npt.NDArray[np.float64] = np.asarray(
            merged[f"{performance_metric}_b"], dtype=np.float64
        )
        return (
            a_values,
            b_values,
            "wilcoxon_signed_rank",
            int(merged.shape[0]),
        )

    a_values = np.asarray(a_group[performance_metric], dtype=np.float64)
    b_values = np.asarray(b_group[performance_metric], dtype=np.float64)
    return (
        a_values,
        b_values,
        "mann_whitney_u",
        0,
    )


def _pairwise(
    seed_metrics: pd.DataFrame,
    design: ComparisonDesign,
    options: AnalysisOptions,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (metric, direction), metric_group in seed_metrics.groupby(
        ["evaluation_metric", "direction"],
        sort=False,
    ):
        metric_direction = cast(MetricDirection, direction)
        current_group = metric_group.copy()
        current_group["algorithm"] = current_group["algorithm"].astype(str)
        algorithms: list[str] = list(dict.fromkeys(current_group["algorithm"]))
        algorithm_groups = {
            algorithm: current_group[current_group["algorithm"] == algorithm]
            for algorithm in algorithms
        }

        for performance_metric in metrics.PERFORMANCE_METRICS:
            for algorithm_a, algorithm_b in itertools.combinations(algorithms, 2):
                a, b, test_name, common_seeds = _comparison_samples(
                    algorithm_groups[algorithm_a],
                    algorithm_groups[algorithm_b],
                    performance_metric,
                    design,
                )
                test_statistic, p_value = (
                    statistics.paired_test(a, b)
                    if design is ComparisonDesign.PAIRED
                    else statistics.independent_test(a, b)
                )
                probability = statistics.probability_of_improvement(
                    a,
                    b,
                    metric_direction,
                )
                rows.append(
                    {
                        "evaluation_metric": metric,
                        "direction": direction,
                        "performance_metric": performance_metric,
                        "algorithm_a": algorithm_a,
                        "algorithm_b": algorithm_b,
                        "comparison_design": design.value,
                        "test": test_name,
                        "n_a": int(a.size),
                        "n_b": int(b.size),
                        "n_common_seeds": common_seeds,
                        "test_statistic": test_statistic,
                        "p_value": p_value,
                        "probability_b_better": probability,
                        "cliffs_delta_b_vs_a": 2.0 * probability - 1.0,
                        "pairwise_input": "observed_seed_metrics",
                    }
                )

    return statistics.add_holm_correction(
        pd.DataFrame(rows),
        family_columns=metrics.METRIC_GROUP_COLUMNS,
        significance_level=options.significance_level,
    )


def _task_summary(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (metric, performance_metric), group in pairwise.groupby(
        metrics.METRIC_GROUP_COLUMNS,
        sort=False,
    ):
        algorithms = sorted(
            set(group["algorithm_a"].astype(str)).union(
                group["algorithm_b"].astype(str)
            )
        )
        ranks = summary[
            (summary["evaluation_metric"] == metric)
            & (summary["performance_metric"] == performance_metric)
        ].set_index("algorithm")["rank"]

        for algorithm in algorithms:
            probabilities = statistics.probability_for_algorithm(group, algorithm)
            rows.append(
                {
                    "algorithm": algorithm,
                    "evaluation_metric": metric,
                    "performance_metric": performance_metric,
                    "iqm_rank": float(ranks.loc[algorithm]),
                    "task_superiority": float(np.mean(probabilities)),
                    "opponents": int(probabilities.size),
                }
            )
    return pd.DataFrame(rows)


def run_task_analysis(
    runs: Sequence[LoadedRun],
    metric_specs: Sequence[MetricSpec] = (MetricSpec("episode_reward", "higher"),),
    options: AnalysisOptions | None = None,
) -> tuple[TaskAnalysisResult, ValidationResult]:
    if options is None:
        options = AnalysisOptions()

    validation = validation_module.validate_runs(runs, metric_specs, options)

    seed_metrics = _seed_metrics(runs, metric_specs, options)
    algorithm_summary = _algorithm_summary(seed_metrics, options)
    pairwise = _pairwise(
        seed_metrics,
        validation.comparison_design,
        options,
    )
    task_summary = _task_summary(algorithm_summary, pairwise)

    result = TaskAnalysisResult(
        seed_metrics=seed_metrics,
        algorithm_summary=algorithm_summary,
        pairwise=pairwise,
        task_summary=task_summary,
    )
    return result, validation
