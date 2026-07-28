from __future__ import annotations

import itertools
import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from cares_reinforcement_learning.stats import reporting, statistics
from cares_reinforcement_learning.stats.io import load_algorithm_run
from cares_reinforcement_learning.stats.metrics import (
    PERFORMANCE_METRICS,
    aggregate_evaluation_curve,
    compute_curve_metrics,
)
from cares_reinforcement_learning.stats.models import (
    AnalysisOptions,
    ComparisonDesign,
    DiscoveredRun,
    MetricSpec,
)
from cares_reinforcement_learning.stats.validation import validate_runs


def _seed_metrics(
    runs, metric_specs: Sequence[MetricSpec], options: AnalysisOptions
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in runs:
        for seed, seed_run in sorted(run.seeds.items()):
            for spec in metric_specs:
                steps, values = aggregate_evaluation_curve(
                    seed_run.eval_data, spec.column
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
                        **compute_curve_metrics(
                            steps,
                            values,
                            options.early_window_fraction,
                            options.final_window_fraction,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _algorithm_summary(
    seed_metrics: pd.DataFrame, options: AnalysisOptions
) -> pd.DataFrame:
    rng = np.random.default_rng(options.random_seed)
    rows: list[dict[str, object]] = []
    groups = seed_metrics.groupby(
        ["algorithm", "evaluation_metric", "direction"], sort=False
    )
    for (algorithm, metric, direction), group in groups:
        for performance_metric in PERFORMANCE_METRICS:
            values = group[performance_metric].to_numpy(dtype=np.float64)
            mean_ci = statistics.bootstrap_bca_ci(
                values,
                np.mean,
                options.bootstrap_samples,
                options.bootstrap_confidence,
                rng,
            )
            iqm_ci = statistics.bootstrap_bca_ci(
                values,
                statistics.interquartile_mean,
                options.bootstrap_samples,
                options.bootstrap_confidence,
                rng,
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
    for _, group in result.groupby(
        ["evaluation_metric", "performance_metric"], sort=False
    ):
        current = group.copy()
        ascending = current["direction"].iloc[0] == "lower"
        current["rank"] = current["iqm"].rank(method="average", ascending=ascending)
        ranked.append(current)
    return pd.concat(ranked, ignore_index=True)


def _pairwise(seed_metrics: pd.DataFrame, design: ComparisonDesign) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    groups = seed_metrics.groupby(["evaluation_metric", "direction"], sort=False)

    for (metric, direction), metric_group in groups:
        algorithms = list(dict.fromkeys(metric_group["algorithm"]))
        for performance_metric in PERFORMANCE_METRICS:
            for algorithm_a, algorithm_b in itertools.combinations(algorithms, 2):

                a_group = metric_group[metric_group["algorithm"] == algorithm_a]
                b_group = metric_group[metric_group["algorithm"] == algorithm_b]

                if design is ComparisonDesign.PAIRED:
                    merged = a_group[["seed", performance_metric]].merge(
                        b_group[["seed", performance_metric]],
                        on="seed",
                        suffixes=("_a", "_b"),
                    )
                    a = merged[f"{performance_metric}_a"].to_numpy(dtype=np.float64)
                    b = merged[f"{performance_metric}_b"].to_numpy(dtype=np.float64)
                    test_statistic, p_value = statistics.paired_test(a, b)
                    test_name = "wilcoxon_signed_rank"
                    common_seeds = int(merged.shape[0])
                else:
                    a = a_group[performance_metric].to_numpy(dtype=np.float64)
                    b = b_group[performance_metric].to_numpy(dtype=np.float64)
                    test_statistic, p_value = statistics.independent_test(a, b)
                    test_name = "mann_whitney_u"
                    common_seeds = 0

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
                        "probability_b_better": statistics.probability_of_improvement(
                            a, b, direction
                        ),
                        "cliffs_delta_b_vs_a": statistics.signed_cliffs_delta(
                            a, b, direction
                        ),
                        "pairwise_input": "observed_seed_metrics",
                    }
                )

    result = pd.DataFrame(rows)
    result["p_value_holm"] = np.nan

    families = result.groupby(
        ["evaluation_metric", "performance_metric"], sort=False
    ).groups

    for indices in families.values():
        indices = list(indices)
        result.loc[indices, "p_value_holm"] = statistics.holm_correction(
            result.loc[indices, "p_value"].to_numpy(dtype=np.float64)
        )
    result["significant_holm_0_05"] = result["p_value_holm"] < 0.05

    return result


def _task_summary(summary: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = pairwise.groupby(["evaluation_metric", "performance_metric"], sort=False)
    for (metric, performance_metric), group in groups:
        algorithms = sorted(set(group["algorithm_a"]).union(group["algorithm_b"]))
        ranks = summary[
            (summary["evaluation_metric"] == metric)
            & (summary["performance_metric"] == performance_metric)
        ].set_index("algorithm")["rank"]
        for algorithm in algorithms:
            probabilities: list[float] = []
            for row in group.itertuples(index=False):
                if row.algorithm_b == algorithm:
                    probabilities.append(float(row.probability_b_better))
                elif row.algorithm_a == algorithm:
                    probabilities.append(1.0 - float(row.probability_b_better))
            rows.append(
                {
                    "algorithm": algorithm,
                    "evaluation_metric": metric,
                    "performance_metric": performance_metric,
                    "iqm_rank": float(ranks.loc[algorithm]),
                    "task_superiority": float(np.mean(probabilities)),
                    "opponents": len(probabilities),
                }
            )
    return pd.DataFrame(rows)


def run_task_analysis(
    discovered_runs: Sequence[DiscoveredRun],
    output_dir: str | pathlib.Path,
    metric_specs: Sequence[MetricSpec] = (MetricSpec("episode_reward", "higher"),),
    options: AnalysisOptions = AnalysisOptions(),
) -> dict[str, pd.DataFrame]:
    runs = [load_algorithm_run(discovered) for discovered in discovered_runs]
    validation = validate_runs(runs, metric_specs, options)
    seed_metrics = _seed_metrics(runs, metric_specs, options)
    algorithm_summary = _algorithm_summary(seed_metrics, options)
    pairwise = _pairwise(seed_metrics, validation.comparison_design)
    task_summary = _task_summary(algorithm_summary, pairwise)

    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(output / "seed_metrics.csv", index=False)
    algorithm_summary.to_csv(output / "algorithm_summary.csv", index=False)
    pairwise.to_csv(output / "pairwise_comparisons.csv", index=False)
    task_summary.to_csv(output / "task_summary_all_metrics.csv", index=False)
    reporting.write_task_outputs(
        output,
        algorithm_summary,
        pairwise,
        task_summary,
        options,
        validation.comparison_design.value,
        validation.warnings,
    )
    return {
        "seed_metrics": seed_metrics,
        "algorithm_summary": algorithm_summary,
        "pairwise": pairwise,
        "task_summary": task_summary,
    }
