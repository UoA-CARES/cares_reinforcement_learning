from __future__ import annotations

import itertools
import pathlib
from collections.abc import Mapping

import numpy as np
import pandas as pd

from cares_reinforcement_learning.stats import reporting, statistics
from cares_reinforcement_learning.stats.models import AnalysisOptions
from cares_reinforcement_learning.stats.statistical_plots import (
    write_cross_task_figures,
)

GROUP_COLUMNS = ["evaluation_metric", "performance_metric"]


def _load_task(
    task: str, directory: pathlib.Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(directory / "algorithm_summary.csv")
    pairwise = pd.read_csv(directory / "pairwise_comparisons.csv")
    seed_metrics = pd.read_csv(directory / "seed_metrics.csv")
    summary.insert(0, "task", task)
    pairwise.insert(0, "task", task)
    seed_metrics.insert(0, "task", task)
    return summary, pairwise, seed_metrics


def _percentile_interval(
    bootstrap_replicates: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    """Return a two-sided percentile bootstrap confidence interval."""
    replicates = np.asarray(bootstrap_replicates, dtype=np.float64)
    if replicates.ndim != 1 or replicates.size == 0:
        raise ValueError(
            "Percentile interval requires one-dimensional bootstrap replicates."
        )
    if not np.isfinite(replicates).all():
        raise ValueError("Bootstrap replicates must be finite.")

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(replicates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _task_seed_values(
    seed_metrics: pd.DataFrame,
    task: str,
    evaluation_metric: str,
    performance_metric: str,
    algorithm: str,
) -> np.ndarray:
    values = seed_metrics[
        (seed_metrics["task"] == task)
        & (seed_metrics["evaluation_metric"] == evaluation_metric)
        & (seed_metrics["algorithm"] == algorithm)
    ][performance_metric].to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError(
            f"Missing seed metrics for {task!r}, {evaluation_metric!r}, "
            f"{performance_metric!r}, {algorithm!r}."
        )
    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite seed metrics for {task!r}, {evaluation_metric!r}, "
            f"{performance_metric!r}, {algorithm!r}."
        )
    return values


def _stratified_pairwise_probability_ci(
    seed_metrics: pd.DataFrame,
    tasks: list[str],
    evaluation_metric: str,
    performance_metric: str,
    direction: str,
    algorithm_a: str,
    algorithm_b: str,
    options: AnalysisOptions,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile CI from seed resampling within each fixed benchmark task.

    The benchmark task set is held fixed. In each bootstrap replicate, runs are
    sampled independently with replacement for both algorithms within every task,
    the task-level probability of improvement is recomputed, and those task-level
    probabilities are averaged.
    """
    task_samples: list[tuple[np.ndarray, np.ndarray]] = []
    for task in tasks:
        values_a = _task_seed_values(
            seed_metrics,
            task,
            evaluation_metric,
            performance_metric,
            algorithm_a,
        )
        values_b = _task_seed_values(
            seed_metrics,
            task,
            evaluation_metric,
            performance_metric,
            algorithm_b,
        )
        task_samples.append((values_a, values_b))

    if not task_samples:
        raise ValueError("At least one task is required for a stratified bootstrap.")

    replicates = np.empty(options.bootstrap_samples, dtype=np.float64)
    for replicate_index in range(options.bootstrap_samples):
        probabilities = np.empty(len(task_samples), dtype=np.float64)
        for task_index, (values_a, values_b) in enumerate(task_samples):
            sampled_a = rng.choice(values_a, size=values_a.size, replace=True)
            sampled_b = rng.choice(values_b, size=values_b.size, replace=True)
            probabilities[task_index] = statistics.probability_of_improvement(
                sampled_b,
                sampled_a,
                direction,
            )
        replicates[replicate_index] = float(np.mean(probabilities))

    return _percentile_interval(replicates, options.bootstrap_confidence)


def _stratified_algorithm_superiority_ci(
    seed_metrics: pd.DataFrame,
    tasks: list[str],
    evaluation_metric: str,
    performance_metric: str,
    direction: str,
    algorithm: str,
    opponents: list[str],
    options: AnalysisOptions,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile CI for roster-average superiority using fixed task strata.

    Each replicate holds the task roster fixed, resamples runs independently
    within every task and algorithm, recomputes the candidate's probability of
    improvement against every opponent, then averages over opponents and tasks.
    """
    if not opponents:
        raise ValueError("Mean superiority requires at least one opponent.")

    task_samples: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for task in tasks:
        candidate = _task_seed_values(
            seed_metrics,
            task,
            evaluation_metric,
            performance_metric,
            algorithm,
        )
        baselines = [
            _task_seed_values(
                seed_metrics,
                task,
                evaluation_metric,
                performance_metric,
                opponent,
            )
            for opponent in opponents
        ]
        task_samples.append((candidate, baselines))

    if not task_samples:
        raise ValueError("At least one task is required for a stratified bootstrap.")

    replicates = np.empty(options.bootstrap_samples, dtype=np.float64)
    for replicate_index in range(options.bootstrap_samples):
        task_values = np.empty(len(task_samples), dtype=np.float64)
        for task_index, (candidate, baselines) in enumerate(task_samples):
            sampled_candidate = rng.choice(
                candidate,
                size=candidate.size,
                replace=True,
            )
            opponent_probabilities = np.empty(len(baselines), dtype=np.float64)
            for opponent_index, baseline in enumerate(baselines):
                sampled_baseline = rng.choice(
                    baseline,
                    size=baseline.size,
                    replace=True,
                )
                opponent_probabilities[opponent_index] = (
                    statistics.probability_of_improvement(
                        sampled_baseline,
                        sampled_candidate,
                        direction,
                    )
                )
            task_values[task_index] = float(np.mean(opponent_probabilities))
        replicates[replicate_index] = float(np.mean(task_values))

    return _percentile_interval(replicates, options.bootstrap_confidence)


def _task_superiority(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = pairwise.groupby(["task", *GROUP_COLUMNS], sort=False)
    for (task, metric, performance_metric), group in groups:
        algorithms = sorted(set(group["algorithm_a"]).union(group["algorithm_b"]))
        for algorithm in algorithms:
            probabilities: list[float] = []
            for row in group.itertuples(index=False):
                if row.algorithm_b == algorithm:
                    probabilities.append(float(row.probability_b_better))
                elif row.algorithm_a == algorithm:
                    probabilities.append(1.0 - float(row.probability_b_better))
            rows.append(
                {
                    "task": task,
                    "evaluation_metric": metric,
                    "performance_metric": performance_metric,
                    "algorithm": algorithm,
                    "task_superiority": float(np.mean(probabilities)),
                    "opponents": len(probabilities),
                }
            )
    return pd.DataFrame(rows)


def _benchmark_summary(
    summaries: pd.DataFrame,
    task_superiority: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    options: AnalysisOptions,
) -> pd.DataFrame:
    rng = np.random.default_rng(options.random_seed)
    rows: list[dict[str, object]] = []
    groups = summaries.groupby([*GROUP_COLUMNS, "algorithm"], sort=False)

    for (evaluation_metric, performance_metric, algorithm), algorithm_rows in groups:
        algorithm_rows = algorithm_rows.sort_values("task")
        ranks = algorithm_rows["rank"].to_numpy(dtype=np.float64)
        superiority_rows = task_superiority[
            (task_superiority["evaluation_metric"] == evaluation_metric)
            & (task_superiority["performance_metric"] == performance_metric)
            & (task_superiority["algorithm"] == algorithm)
        ].sort_values("task")
        superiority = superiority_rows["task_superiority"].to_numpy(dtype=np.float64)
        if ranks.size != superiority.size:
            raise ValueError(
                f"Task-count mismatch for {algorithm!r}, {evaluation_metric!r}, "
                f"{performance_metric!r}."
            )

        rank_ci = statistics.bootstrap_bca_ci(
            ranks, np.mean, options.bootstrap_samples, options.bootstrap_confidence, rng
        )
        direction_values = algorithm_rows["direction"].dropna().unique()
        if direction_values.size != 1:
            raise ValueError(
                f"Inconsistent metric direction for {evaluation_metric!r}/"
                f"{performance_metric!r}."
            )
        direction = str(direction_values[0])
        tasks = algorithm_rows["task"].tolist()
        all_algorithms = sorted(
            summaries[
                (summaries["evaluation_metric"] == evaluation_metric)
                & (summaries["performance_metric"] == performance_metric)
            ]["algorithm"].unique()
        )
        opponents = [name for name in all_algorithms if name != algorithm]
        superiority_ci = _stratified_algorithm_superiority_ci(
            seed_metrics,
            tasks,
            evaluation_metric,
            performance_metric,
            direction,
            algorithm,
            opponents,
            options,
            rng,
        )
        q1, q3 = np.quantile(ranks, [0.25, 0.75])
        n_tasks = int(ranks.size)
        top_counts = {k: int(np.count_nonzero(ranks <= float(k))) for k in (1, 2, 3)}

        rows.append(
            {
                "evaluation_metric": evaluation_metric,
                "performance_metric": performance_metric,
                "algorithm": algorithm,
                "n_tasks": n_tasks,
                "mean_superiority": float(np.mean(superiority)),
                "superiority_ci_low": float(superiority_ci[0]),
                "superiority_ci_high": float(superiority_ci[1]),
                "average_rank": float(np.mean(ranks)),
                "average_rank_ci_low": float(rank_ci[0]),
                "average_rank_ci_high": float(rank_ci[1]),
                "median_rank": float(np.median(ranks)),
                "rank_std": float(np.std(ranks, ddof=1)) if n_tasks > 1 else 0.0,
                "rank_q1": float(q1),
                "rank_q3": float(q3),
                "rank_iqr": float(q3 - q1),
                "top_1_count": top_counts[1],
                "top_1_rate": float(top_counts[1] / n_tasks),
                "top_2_count": top_counts[2],
                "top_2_rate": float(top_counts[2] / n_tasks),
                "top_3_count": top_counts[3],
                "top_3_rate": float(top_counts[3] / n_tasks),
                "rank_ci_bootstrap_method": "BCa",
                "rank_ci_resampling_unit": "task",
                "superiority_ci_bootstrap_method": "stratified_percentile",
                "superiority_ci_resampling_unit": "seed_within_fixed_task",
                "bootstrap_samples": options.bootstrap_samples,
                "bootstrap_confidence": options.bootstrap_confidence,
            }
        )
    return pd.DataFrame(rows)


def _probability_a_better(
    task_rows: pd.DataFrame,
    algorithm_a: str,
    algorithm_b: str,
) -> float:
    direct = task_rows[
        (task_rows["algorithm_a"] == algorithm_a)
        & (task_rows["algorithm_b"] == algorithm_b)
    ]
    if len(direct) == 1:
        return 1.0 - float(direct.iloc[0]["probability_b_better"])
    reverse = task_rows[
        (task_rows["algorithm_a"] == algorithm_b)
        & (task_rows["algorithm_b"] == algorithm_a)
    ]
    if len(reverse) == 1:
        return float(reverse.iloc[0]["probability_b_better"])
    raise ValueError(
        f"Missing pairwise comparison for {algorithm_a!r} vs {algorithm_b!r}."
    )


def _cross_task_pairwise(
    summaries: pd.DataFrame,
    pairwise: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    options: AnalysisOptions,
) -> pd.DataFrame:
    rng = np.random.default_rng(options.random_seed)
    rows: list[dict[str, object]] = []

    for (evaluation_metric, performance_metric), summary_group in summaries.groupby(
        GROUP_COLUMNS, sort=False
    ):
        pairwise_group = pairwise[
            (pairwise["evaluation_metric"] == evaluation_metric)
            & (pairwise["performance_metric"] == performance_metric)
        ]
        algorithms = sorted(summary_group["algorithm"].unique())
        rank_table = summary_group.pivot(
            index="task", columns="algorithm", values="rank"
        ).sort_index()

        for algorithm_a, algorithm_b in itertools.combinations(algorithms, 2):
            probabilities = np.asarray(
                [
                    _probability_a_better(
                        pairwise_group[pairwise_group["task"] == task],
                        algorithm_a,
                        algorithm_b,
                    )
                    for task in rank_table.index
                ],
                dtype=np.float64,
            )
            direction_values = summary_group["direction"].dropna().unique()
            if direction_values.size != 1:
                raise ValueError(
                    f"Inconsistent metric direction for {evaluation_metric!r}/"
                    f"{performance_metric!r}."
                )
            probability_ci = _stratified_pairwise_probability_ci(
                seed_metrics,
                rank_table.index.tolist(),
                evaluation_metric,
                performance_metric,
                str(direction_values[0]),
                algorithm_a,
                algorithm_b,
                options,
                rng,
            )
            ranks_a = rank_table[algorithm_a].to_numpy(dtype=np.float64)
            ranks_b = rank_table[algorithm_b].to_numpy(dtype=np.float64)
            rank_difference = ranks_a - ranks_b
            test_statistic, p_value = statistics.paired_test(ranks_a, ranks_b)
            wins_a = int(np.count_nonzero(ranks_a < ranks_b))
            wins_b = int(np.count_nonzero(ranks_b < ranks_a))
            ties = int(np.count_nonzero(np.isclose(ranks_a, ranks_b)))
            n_tasks = int(ranks_a.size)

            rows.append(
                {
                    "evaluation_metric": evaluation_metric,
                    "performance_metric": performance_metric,
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "n_tasks": n_tasks,
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                    "ties": ties,
                    "win_rate_a": float(wins_a / n_tasks),
                    "win_rate_b": float(wins_b / n_tasks),
                    "tie_rate": float(ties / n_tasks),
                    "mean_probability_a_better": float(np.mean(probabilities)),
                    "probability_a_better_ci_low": float(probability_ci[0]),
                    "probability_a_better_ci_high": float(probability_ci[1]),
                    "mean_rank_difference_a_minus_b": float(np.mean(rank_difference)),
                    "median_rank_difference_a_minus_b": float(
                        np.median(rank_difference)
                    ),
                    "test": "wilcoxon_signed_rank_on_task_ranks",
                    "test_statistic": test_statistic,
                    "p_value": p_value,
                    "pairwise_input": "paired_task_level_ranks",
                    "probability_ci_bootstrap_method": "stratified_percentile",
                    "probability_ci_resampling_unit": "seed_within_fixed_task",
                    "bootstrap_samples": options.bootstrap_samples,
                    "bootstrap_confidence": options.bootstrap_confidence,
                }
            )

    result = pd.DataFrame(rows)
    result["p_value_holm"] = np.nan
    result["significant_holm"] = False
    for _, indices in result.groupby(GROUP_COLUMNS, sort=False).groups.items():
        adjusted = statistics.holm_correction(
            result.loc[indices, "p_value"].to_numpy(dtype=np.float64)
        )
        result.loc[indices, "p_value_holm"] = adjusted
        result.loc[indices, "significant_holm"] = adjusted < options.significance_level
    return result


def _friedman_and_nemenyi(
    summaries: pd.DataFrame,
    options: AnalysisOptions,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], float]]:
    friedman_rows: list[dict[str, object]] = []
    nemenyi_rows: list[dict[str, object]] = []
    critical_differences: dict[tuple[str, str], float] = {}

    for (evaluation_metric, performance_metric), group in summaries.groupby(
        GROUP_COLUMNS, sort=False
    ):
        matrix = group.pivot(index="task", columns="algorithm", values="rank")
        if matrix.isna().any().any():
            raise ValueError(
                "Friedman test requires one rank per task and algorithm for "
                f"{evaluation_metric!r}/{performance_metric!r}."
            )
        matrix = matrix.sort_index().sort_index(axis=1)
        n_tasks, n_algorithms = matrix.shape
        if n_tasks < 2 or n_algorithms < 3:
            continue

        statistic, p_value = statistics.friedman_test(matrix.to_numpy(dtype=np.float64))
        significant = bool(p_value < options.significance_level)
        friedman_rows.append(
            {
                "evaluation_metric": evaluation_metric,
                "performance_metric": performance_metric,
                "test_statistic": statistic,
                "degrees_of_freedom": n_algorithms - 1,
                "p_value": p_value,
                "significant": significant,
                "n_tasks": n_tasks,
                "n_algorithms": n_algorithms,
                "test_input": "task_level_ranks",
            }
        )

        if significant:
            cd = statistics.nemenyi_critical_difference(
                n_algorithms, n_tasks, options.significance_level
            )
            critical_differences[(evaluation_metric, performance_metric)] = cd
            average_ranks = matrix.mean(axis=0)
            for algorithm_a, algorithm_b in itertools.combinations(matrix.columns, 2):
                difference = abs(
                    float(average_ranks[algorithm_a] - average_ranks[algorithm_b])
                )
                nemenyi_rows.append(
                    {
                        "evaluation_metric": evaluation_metric,
                        "performance_metric": performance_metric,
                        "algorithm_a": algorithm_a,
                        "algorithm_b": algorithm_b,
                        "average_rank_a": float(average_ranks[algorithm_a]),
                        "average_rank_b": float(average_ranks[algorithm_b]),
                        "absolute_rank_difference": difference,
                        "critical_difference": cd,
                        "significant": bool(difference > cd),
                        "alpha": options.significance_level,
                    }
                )

    return pd.DataFrame(friedman_rows), pd.DataFrame(nemenyi_rows), critical_differences


def run_cross_task_analysis(
    task_output_dirs: Mapping[str, str | pathlib.Path],
    output_dir: str | pathlib.Path,
    options: AnalysisOptions = AnalysisOptions(),
) -> dict[str, pd.DataFrame]:
    loaded = [
        _load_task(task, pathlib.Path(path)) for task, path in task_output_dirs.items()
    ]
    summaries = pd.concat([summary for summary, _, _ in loaded], ignore_index=True)
    pairwise = pd.concat(
        [comparisons for _, comparisons, _ in loaded], ignore_index=True
    )
    seed_metrics = pd.concat([metrics for _, _, metrics in loaded], ignore_index=True)

    algorithms_by_group_task = summaries.groupby([*GROUP_COLUMNS, "task"])[
        "algorithm"
    ].apply(set)

    for group_key, group_sets in algorithms_by_group_task.groupby(level=GROUP_COLUMNS):
        task_sets = {task: algorithms for (*_, task), algorithms in group_sets.items()}

        unique_sets = {frozenset(algorithms) for algorithms in task_sets.values()}
        if len(unique_sets) <= 1:
            continue

        expected = sorted(set.union(*task_sets.values()))

        details = "\n".join(
            (
                f"  {task}: "
                f"{sorted(algorithms)}"
                f"{'' if algorithms == set(expected) else f' (missing: {sorted(set(expected) - algorithms)})'}"
            )
            for task, algorithms in sorted(task_sets.items())
        )

        raise ValueError(
            "Every task must contain the same algorithms within each metric group.\n"
            f"Group: {group_key!r}\n"
            f"Expected algorithms: {expected}\n"
            f"Task contents:\n{details}"
        )

    task_superiority = _task_superiority(pairwise)
    benchmark = _benchmark_summary(summaries, task_superiority, seed_metrics, options)
    cross_task_pairwise = _cross_task_pairwise(
        summaries, pairwise, seed_metrics, options
    )
    friedman, nemenyi, critical_differences = _friedman_and_nemenyi(summaries, options)

    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(output / "task_algorithm_summaries.csv", index=False)
    pairwise.to_csv(output / "task_pairwise_comparisons.csv", index=False)
    seed_metrics.to_csv(output / "task_seed_metrics.csv", index=False)
    task_superiority.to_csv(output / "task_superiority.csv", index=False)
    benchmark.to_csv(output / "benchmark_summary.csv", index=False)
    cross_task_pairwise.to_csv(output / "cross_task_pairwise.csv", index=False)
    friedman.to_csv(output / "friedman_tests.csv", index=False)
    nemenyi.to_csv(output / "nemenyi_posthoc.csv", index=False)

    reporting.write_cross_task_outputs(
        output, benchmark, cross_task_pairwise, friedman, nemenyi, options
    )
    if options.generate_statistical_figures:
        write_cross_task_figures(
            output,
            summaries,
            benchmark,
            cross_task_pairwise,
            critical_differences,
            options.figure_dpi,
        )

    return {
        "benchmark_summary": benchmark,
        "cross_task_pairwise": cross_task_pairwise,
        "friedman_tests": friedman,
        "nemenyi_posthoc": nemenyi,
        "task_superiority": task_superiority,
    }
