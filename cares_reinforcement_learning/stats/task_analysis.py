from __future__ import annotations

import dataclasses
import json
import logging
import math
import pathlib
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Iterable, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import integrate, stats

from . import (
    reporting,
    statistics,
    utils,
    validation,
)
from .data import AlgorithmRun, MetricDirection, MetricSpec, SeedMetricSummary, SeedRun
from .validation import AnalysisOptions, ValidationPolicy

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PairwiseSamples:
    values_a: npt.NDArray[np.float64]
    values_b: npt.NDArray[np.float64]
    paired: bool
    n_common_seeds: int


@dataclasses.dataclass(frozen=True)
class TestResult:
    test_name: str
    statistic: float
    p_value: float


def _log_validation_issues(issues: Sequence[validation.ValidationIssue]) -> None:
    if not issues:
        LOGGER.info("Validation completed with no issues.")
        return

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    info_count = sum(1 for issue in issues if issue.severity == "info")
    LOGGER.info(
        "Validation completed with %d issue(s): %d error(s), %d warning(s), %d info.",
        len(issues),
        error_count,
        warning_count,
        info_count,
    )

    for issue in issues:
        log_message = f"[{issue.check_name}] {issue.message} Recommendation: {issue.recommendation}"
        if issue.severity == "error":
            LOGGER.error(log_message)
        elif issue.severity == "warning":
            LOGGER.warning(log_message)
        else:
            LOGGER.info(log_message)


def _unique_non_null_as_str(series: pd.Series, *, column_name: str) -> list[str]:
    values: list[str] = []
    for index, raw_value in series.items():
        if raw_value is None:
            raise ValueError(
                f"Malformed value in input: column {column_name!r}, row {index}, value is null."
            )
        if isinstance(raw_value, float) and math.isnan(raw_value):
            raise ValueError(
                f"Malformed value in input: column {column_name!r}, row {index}, value is null."
            )
        if not isinstance(raw_value, str):
            raise ValueError(
                f"Malformed value in input: column {column_name!r}, row {index}, "
                f"value {raw_value!r} is not a string."
            )
        value = raw_value.strip()
        if not value:
            raise ValueError(
                f"Malformed value in input: column {column_name!r}, row {index}, value is empty."
            )
        values.append(value)

    return list(dict.fromkeys(values))


def _iter_algorithm_metric_groups(
    seed_summary: pd.DataFrame,
) -> list[tuple[str, str, str, pd.DataFrame]]:
    groups: list[tuple[str, str, str, pd.DataFrame]] = []
    for algorithm in _unique_non_null_as_str(
        seed_summary["algorithm"], column_name="algorithm"
    ):
        algorithm_group = seed_summary[seed_summary["algorithm"] == algorithm]
        for metric in _unique_non_null_as_str(
            algorithm_group["evaluation_metric"], column_name="evaluation_metric"
        ):
            group = algorithm_group[algorithm_group["evaluation_metric"] == metric]
            direction_values = _unique_non_null_as_str(
                group["direction"], column_name="direction"
            )
            if len(direction_values) != 1:
                raise ValueError(
                    "Malformed value in input: metric group has multiple "
                    f"directions for algorithm {algorithm!r}, metric {metric!r}: {direction_values}."
                )
            direction = direction_values[0]
            groups.append((algorithm, metric, direction, group))
    return groups


def _extract_scalar_values(group: pd.DataFrame, scalar: str) -> npt.NDArray[np.float64]:
    values: list[float] = []
    for index, raw_value in group[scalar].items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Malformed value in input: "
                f"column {scalar!r}, row {index}, value {raw_value!r} is not numeric."
            ) from exc
        if math.isnan(value):
            raise ValueError(
                "Malformed value in input: "
                f"column {scalar!r}, row {index}, value is NaN."
            )
        values.append(value)

    if not values:
        raise ValueError(
            f"Malformed value in input: column {scalar!r} has no numeric values."
        )

    return np.asarray(values, dtype=np.float64)


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return cast(dict[str, Any], data)


def _parse_seed_from_directory(path: pathlib.Path) -> int | None:
    """Parse CARES RL numeric seed directories.

    CARES RL logs use numeric seed directories, e.g.:

        <log_path>/10/data/eval.csv

    Non-numeric directories are ignored.
    """

    try:
        return int(path.name)
    except ValueError:
        return None


def _load_algorithm_run(algorithm: str, log_path: str | pathlib.Path) -> AlgorithmRun:
    root = pathlib.Path(log_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Log path does not exist: {root}")

    configs: dict[str, Mapping[str, Any]] = {}
    for config_path in sorted(root.glob("*_config.json")):
        configs[config_path.name] = _load_json(config_path)

    seeds: dict[int, SeedRun] = {}
    for seed_dir in sorted(root.iterdir()):
        if not seed_dir.is_dir():
            continue
        seed = _parse_seed_from_directory(seed_dir)
        if seed is None:
            LOGGER.warning(
                "Skipping non-numeric seed directory for %s: %s", algorithm, seed_dir
            )
            continue

        eval_path = seed_dir / "data" / "eval.csv"
        if not eval_path.exists():
            LOGGER.warning(
                "Missing eval.csv for %s seed %s: %s", algorithm, seed, eval_path
            )
            continue

        eval_data = pd.read_csv(eval_path)
        seeds[seed] = SeedRun(
            algorithm=algorithm,
            seed=seed,
            log_path=root,
            eval_path=eval_path,
            eval_data=eval_data,
        )

    if not seeds:
        raise ValueError(f"No seed directories with data/eval.csv found in {root}")

    return AlgorithmRun(
        algorithm=algorithm,
        log_path=root,
        configs=configs,
        seeds=seeds,
    )


def _build_scalar_summary_row(
    *,
    algorithm: str,
    evaluation_metric: str,
    direction: str,
    performance_metric: str,
    values_array: npt.NDArray[np.float64],
    options: AnalysisOptions,
    rng: np.random.Generator,
) -> dict[str, object]:
    mean_ci = statistics.bootstrap_ci(
        values_array.tolist(),
        np.mean,
        options.bootstrap_samples,
        options.bootstrap_confidence,
        rng,
    )
    iqm_ci = statistics.bootstrap_ci(
        values_array.tolist(),
        statistics.interquartile_mean,
        options.bootstrap_samples,
        options.bootstrap_confidence,
        rng,
    )

    mean = float(np.mean(values_array))
    std = float(np.std(values_array, ddof=1)) if values_array.size > 1 else 0.0
    median = float(np.median(values_array))
    iqm = statistics.interquartile_mean(values_array.tolist())

    return {
        "algorithm": algorithm,
        "evaluation_metric": evaluation_metric,
        "performance_metric": performance_metric,
        "direction": direction,
        "n_seeds": int(values_array.size),
        "mean": mean,
        "std": std,
        "median": median,
        "iqm": iqm,
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "mean_ci_lower": mean_ci[0],
        "mean_ci_upper": mean_ci[1],
        "iqm_ci_lower": iqm_ci[0],
        "iqm_ci_upper": iqm_ci[1],
    }


def _add_algorithm_ranks(algorithm_summary: pd.DataFrame) -> pd.DataFrame:
    ranked = algorithm_summary.copy()

    grouped = cast(
        Iterable[tuple[tuple[str, str], pd.DataFrame]],
        ranked.groupby(["evaluation_metric", "performance_metric"]),  # type: ignore[no-untyped-call]
    )

    for (_, _), group in grouped:
        direction = str(group["direction"].iloc[0])
        ascending = direction == "lower"

        ranks = group["iqm"].rank(
            method="min",
            ascending=ascending,
        )

        ranked.loc[group.index, "rank"] = ranks

    ranked["rank"] = ranked["rank"].astype(int)

    return ranked


def _compute_auc(
    steps: npt.NDArray[np.float64], values: npt.NDArray[np.float64]
) -> float:
    if values.size == 1:
        return float(values[0])

    if steps.size != values.size:
        raise ValueError(
            f"Cannot compute AUC: mismatched step and value lengths: {steps.size} steps, {values.size} values."
        )

    auc_integral = float(integrate.trapezoid(values, x=steps))

    duration = float(steps[-1] - steps[0])
    if duration <= 0:
        normalised_auc = float(np.mean(values))
    else:
        normalised_auc = auc_integral / duration

    return normalised_auc


def _parse_numeric_column_to_array(
    column: Any, *, column_name: str, source_name: str
) -> npt.NDArray[np.float64]:
    values: list[float] = []
    for index, raw_value in enumerate(column):
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed value in {source_name}: column {column_name!r}, row {index}, value {raw_value!r} is not numeric."
            ) from exc
        if math.isnan(value):
            raise ValueError(
                f"Malformed value in {source_name}: column {column_name!r}, row {index}, "
                "value is NaN."
            )
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _summarise_seed_metric(
    seed_run: SeedRun,
    metric: MetricSpec,
    options: AnalysisOptions,
) -> SeedMetricSummary:
    # Step 1: select and validate the metric column for this seed.
    frame = seed_run.eval_data.copy()
    if metric.name not in frame.columns:
        raise KeyError(f"Metric {metric.name!r} not found in {seed_run.eval_path}")

    # Step 2: coerce metric values to finite float array.
    values = _parse_numeric_column_to_array(
        frame[metric.name], column_name=metric.name, source_name=str(seed_run.eval_path)
    )
    if values.size == 0:
        raise ValueError(
            f"Metric {metric.name!r} has no valid numeric values in {seed_run.eval_path}"
        )

    # Step 3: resolve x-axis steps and align with metric values.
    step_column = utils.find_step_column(frame)
    steps = _parse_numeric_column_to_array(
        frame[step_column],
        column_name=step_column,
        source_name=str(seed_run.eval_path),
    )
    if steps.size != values.size:
        raise ValueError(
            f"Mismatched metric and step lengths in {seed_run.eval_path}: "
            f"metric column {metric.name!r} has {values.size} values but step "
            f"column {step_column!r} has {steps.size}."
        )

    # Step 4: compute the final-evaluation window used for terminal statistics.
    window_size = max(
        options.final_window_min_points,
        int(math.ceil(values.size * options.final_window_fraction)),
    )
    window_size = min(window_size, values.size)
    final_window = values[-window_size:]

    # Step 5: compute core scalar diagnostics from the full trace.
    final_window_mean = float(np.mean(final_window))
    final_window_std = (
        float(np.std(final_window, ddof=1)) if final_window.size > 1 else 0.0
    )

    diagnostic_best_value = float(
        np.max(values) if metric.direction == "higher" else np.min(values)
    )
    auc = _compute_auc(steps, values)

    early_cutoff = max(1, int(math.ceil(values.size * 0.25)))
    early_auc_25 = _compute_auc(steps[:early_cutoff], values[:early_cutoff])

    # Step 6: package all per-seed metric diagnostics into one summary record.
    return SeedMetricSummary(
        algorithm=seed_run.algorithm,
        seed=seed_run.seed,
        evaluation_metric=metric.name,
        direction=metric.direction,
        n_eval_points=int(values.size),
        first_step=float(steps[0]),
        last_step=float(steps[-1]),
        final_window_mean=final_window_mean,
        auc=auc,
        early_auc_25=early_auc_25,
        final_window_std=final_window_std,
        diagnostic_best_value=diagnostic_best_value,
    )


def _build_seed_summary_table(
    runs: Sequence[AlgorithmRun],
    metric_specs: Sequence[MetricSpec],
    options: AnalysisOptions,
) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for run in runs:
        for seed_run in run.seeds.values():
            for metric in metric_specs:
                summary = _summarise_seed_metric(seed_run, metric, options)
                summaries.append(dataclasses.asdict(summary))
    return pd.DataFrame(summaries)


def _build_algorithm_summary_table(
    seed_summary: pd.DataFrame,
    options: AnalysisOptions,
) -> pd.DataFrame:
    rng = np.random.default_rng(options.random_seed)
    rows: list[dict[str, object]] = []

    performance_metrics = ["auc", "early_auc_25", "final_window_mean"]
    for algorithm, evaluation_metric, direction, group in _iter_algorithm_metric_groups(
        seed_summary
    ):
        for performance_metric in performance_metrics:
            values_array = _extract_scalar_values(group, performance_metric)
            rows.append(
                _build_scalar_summary_row(
                    algorithm=algorithm,
                    evaluation_metric=evaluation_metric,
                    direction=direction,
                    performance_metric=performance_metric,
                    values_array=values_array,
                    options=options,
                    rng=rng,
                )
            )

    result = pd.DataFrame(rows)
    return _add_algorithm_ranks(result)


def _load_and_validate_runs(
    run_paths: Mapping[str, pathlib.Path],
    policy: ValidationPolicy,
    options: AnalysisOptions,
    output_dir: pathlib.Path,
) -> list[AlgorithmRun]:
    LOGGER.info("Loading %d algorithm run(s).", len(run_paths))
    for algorithm, path in run_paths.items():
        LOGGER.info("Run input: %s -> %s", algorithm, path)

    runs = [
        _load_algorithm_run(algorithm, path) for algorithm, path in run_paths.items()
    ]
    LOGGER.info("Loaded %d algorithm run(s).", len(runs))

    validation_report_path = output_dir / "validation_report.txt"
    try:
        issues = validation.validate_runs(runs, policy, options)
    except ValueError as exc:
        validation_report_path.write_text(
            "Validation failed in strict mode.\n"
            "No summary or pairwise output files were generated.\n\n"
            f"Details:\n{exc}\n",
            encoding="utf-8",
        )
        LOGGER.error(
            "Validation failed before output generation. See %s",
            validation_report_path,
        )
        raise ValueError(
            f"Validation failed before output generation. See {validation_report_path}"
        ) from exc

    reporting.save_validation_report(issues, validation_report_path)
    LOGGER.info("Wrote validation report to %s", validation_report_path)
    _log_validation_issues(issues)
    return runs


def _write_core_outputs(
    output_dir: pathlib.Path,
    seed_summary: pd.DataFrame,
    algorithm_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    seed_summary_path = output_dir / "summary_by_seed.csv"
    algorithm_summary_path = output_dir / "summary_by_algorithm.csv"
    pairwise_path = output_dir / "pairwise_comparisons.csv"

    seed_summary.to_csv(seed_summary_path, index=False)
    algorithm_summary.to_csv(algorithm_summary_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)

    LOGGER.info("Wrote %s", seed_summary_path)
    LOGGER.info("Wrote %s", algorithm_summary_path)
    LOGGER.info("Wrote %s", pairwise_path)


def _write_task_publication_outputs(
    output_dir: pathlib.Path,
    algorithm_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    publication_evaluation_metric: str,
    publication_performance_metric: str,
    baseline_algorithm: str | None,
) -> None:
    publication_summary = reporting.build_publication_summary_table(
        algorithm_summary=algorithm_summary,
        primary_metric=publication_evaluation_metric,
        performance_metric=publication_performance_metric,
    )

    publication_pairwise = reporting.build_publication_pairwise_table(
        pairwise=pairwise,
        primary_metric=publication_evaluation_metric,
        performance_metric=publication_performance_metric,
        baseline_algorithm=baseline_algorithm,
    )

    publication_summary.to_csv(output_dir / "publication_summary.csv", index=False)
    publication_pairwise.to_csv(output_dir / "publication_pairwise.csv", index=False)

    reporting.save_latex_table(
        publication_summary,
        output_dir / "publication_summary.tex",
        caption=(
            f"{publication_performance_metric} summary for "
            f"{publication_evaluation_metric}."
        ),
        label=(
            f"tab:{publication_evaluation_metric}_"
            f"{publication_performance_metric}_summary"
        ),
    )

    reporting.save_latex_table(
        publication_pairwise,
        output_dir / "publication_pairwise.tex",
        caption=(
            f"Pairwise {publication_performance_metric} comparisons for "
            f"{publication_evaluation_metric}."
        ),
        label=(
            f"tab:{publication_evaluation_metric}_"
            f"{publication_performance_metric}_pairwise"
        ),
    )

    LOGGER.info("Wrote %s", output_dir / "publication_summary.csv")
    LOGGER.info("Wrote %s", output_dir / "publication_pairwise.csv")
    LOGGER.info("Wrote %s", output_dir / "publication_summary.tex")
    LOGGER.info("Wrote %s", output_dir / "publication_pairwise.tex")


def _write_task_commentary(
    output_dir: pathlib.Path,
    pairwise: pd.DataFrame,
    publication_evaluation_metric: str,
    publication_performance_metric: str,
) -> None:
    commentary = reporting.generate_result_commentary(
        pairwise,
        primary_metric=publication_evaluation_metric,
        performance_metric=publication_performance_metric,
    )
    commentary_path = output_dir / "result_commentary.md"
    commentary_path.write_text(commentary, encoding="utf-8")
    LOGGER.info("Wrote %s", commentary_path)


def _unique_algorithms(frame: pd.DataFrame) -> list[str]:
    return sorted(str(value) for value in frame["algorithm"].unique())


def _generate_algorithm_pairs(algorithms: Sequence[str]) -> list[tuple[str, str]]:
    return list(combinations(algorithms, 2))


def _extract_algorithm_values(
    group: pd.DataFrame,
    algorithm: str,
    performance_metric: str,
) -> pd.DataFrame:
    values = group[group["algorithm"] == algorithm][["seed", performance_metric]]

    if values.empty:
        raise ValueError(
            f"No values found for algorithm {algorithm!r}, "
            f"performance metric {performance_metric!r}."
        )

    return values


def _extract_pairwise_samples(
    evaluation_metric_group: pd.DataFrame,
    algorithm_a: str,
    algorithm_b: str,
    performance_metric: str,
) -> PairwiseSamples:
    algorithm_a_values = _extract_algorithm_values(
        evaluation_metric_group,
        algorithm_a,
        performance_metric,
    )
    algorithm_b_values = _extract_algorithm_values(
        evaluation_metric_group,
        algorithm_b,
        performance_metric,
    )

    common_seeds = sorted(
        set(algorithm_a_values["seed"]) & set(algorithm_b_values["seed"])
    )
    paired = len(common_seeds) == len(algorithm_a_values) == len(algorithm_b_values)

    if paired and common_seeds:
        values_a = np.asarray(
            algorithm_a_values.set_index("seed").loc[
                common_seeds,
                performance_metric,
            ],
            dtype=np.float64,
        )
        values_b = np.asarray(
            algorithm_b_values.set_index("seed").loc[
                common_seeds,
                performance_metric,
            ],
            dtype=np.float64,
        )
    else:
        values_a = np.asarray(
            algorithm_a_values[performance_metric],
            dtype=np.float64,
        )
        values_b = np.asarray(
            algorithm_b_values[performance_metric],
            dtype=np.float64,
        )

    return PairwiseSamples(
        values_a=values_a,
        values_b=values_b,
        paired=paired,
        n_common_seeds=len(common_seeds),
    )


def _run_pairwise_test(samples: PairwiseSamples) -> TestResult:
    if samples.paired:
        differences = samples.values_b - samples.values_a

        if np.allclose(differences, 0.0):
            return TestResult(
                test_name="all_equal_no_test",
                statistic=0.0,
                p_value=1.0,
            )

        wilcoxon_result = stats.wilcoxon(
            samples.values_b,
            samples.values_a,
            zero_method="wilcox",
        )

        return TestResult(
            test_name="wilcoxon_signed_rank",
            statistic=float(wilcoxon_result.statistic),
            p_value=float(wilcoxon_result.pvalue),
        )

    mann_whitney_result = stats.mannwhitneyu(
        samples.values_b,
        samples.values_a,
        alternative="two-sided",
    )

    return TestResult(
        test_name="mann_whitney_u",
        statistic=float(mann_whitney_result.statistic),
        p_value=float(mann_whitney_result.pvalue),
    )


def _compare_pairwise(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    performance_metrics = ["auc", "early_auc_25", "final_window_mean"]

    eval_grouped = cast(
        Iterable[tuple[str, pd.DataFrame]],
        seed_summary.groupby("evaluation_metric"),  # type: ignore[no-untyped-call]
    )

    for evaluation_metric, evaluation_metric_group in eval_grouped:
        direction = cast(
            MetricDirection,
            str(evaluation_metric_group["direction"].iloc[0]),
        )
        if direction not in {"higher", "lower"}:
            raise ValueError(f"Unknown metric direction: {direction}")

        algorithms = _unique_algorithms(evaluation_metric_group)
        algorithm_pairs = _generate_algorithm_pairs(algorithms)

        for algorithm_a, algorithm_b in algorithm_pairs:
            for performance_metric in performance_metrics:
                samples = _extract_pairwise_samples(
                    evaluation_metric_group,
                    algorithm_a,
                    algorithm_b,
                    performance_metric,
                )
                test_result = _run_pairwise_test(samples)

                mean_a = float(np.mean(samples.values_a))
                mean_b = float(np.mean(samples.values_b))

                mean_difference_b_minus_a = mean_b - mean_a

                probability_b_improves_over_a = statistics.probability_of_improvement(
                    samples.values_a,
                    samples.values_b,
                    direction,
                )

                cliffs_delta_b_over_a = statistics.signed_cliffs_delta(
                    samples.values_a,
                    samples.values_b,
                    direction,
                )

                rows.append(
                    {
                        "evaluation_metric": evaluation_metric,
                        "performance_metric": performance_metric,
                        "algorithm_a": algorithm_a,
                        "algorithm_b": algorithm_b,
                        "direction": direction,
                        "paired": samples.paired,
                        "test": test_result.test_name,
                        "statistic": float(test_result.statistic),
                        "p_value": float(test_result.p_value),
                        "algorithm_a_mean": mean_a,
                        "algorithm_b_mean": mean_b,
                        "mean_difference_b_minus_a": mean_difference_b_minus_a,
                        "probability_b_improves_over_a": probability_b_improves_over_a,
                        "cliffs_delta_b_over_a": cliffs_delta_b_over_a,
                        "n_algorithm_a": int(samples.values_a.size),
                        "n_algorithm_b": int(samples.values_b.size),
                        "n_common_seeds": samples.n_common_seeds,
                    }
                )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["p_value_holm"] = statistics.holm_correction(
            np.asarray(result["p_value"], dtype=np.float64)
        )

    return result


def run_task_analysis(
    task_name: str,
    run_paths: Mapping[str, pathlib.Path],
    output_dir: pathlib.Path,
    metric_specs: Sequence[MetricSpec],
    options: AnalysisOptions,
    publication_evaluation_metric: str | None = None,
    publication_performance_metric: str = "auc",
    baseline_algorithm: str | None = None,
    policy: ValidationPolicy | None = None,
) -> pathlib.Path:
    if not metric_specs:
        raise ValueError("At least one metric spec is required.")

    publication_evaluation_metric = (
        publication_evaluation_metric or metric_specs[0].name
    )

    policy = policy or ValidationPolicy()

    task_output_dir = output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting task analysis for %s.", task_name)
    LOGGER.info("Output directory: %s", task_output_dir)

    LOGGER.info("Stage 1/5: load and validate inputs")
    runs = _load_and_validate_runs(run_paths, policy, options, task_output_dir)

    LOGGER.info("Stage 2/5: compute per-seed summaries")
    seed_summary = _build_seed_summary_table(runs, metric_specs, options)

    LOGGER.info("Stage 3/5: aggregate summaries and compute pairwise tests")
    algorithm_summary = _build_algorithm_summary_table(seed_summary, options)
    pairwise = _compare_pairwise(seed_summary)

    LOGGER.info("Stage 4/5: write machine-readable outputs")
    _write_core_outputs(task_output_dir, seed_summary, algorithm_summary, pairwise)

    LOGGER.info("Stage 5/5: write publication outputs and commentary")
    _write_task_publication_outputs(
        task_output_dir,
        algorithm_summary,
        pairwise,
        publication_evaluation_metric,
        publication_performance_metric,
        baseline_algorithm,
    )
    _write_task_commentary(
        task_output_dir,
        pairwise,
        publication_evaluation_metric,
        publication_performance_metric,
    )

    LOGGER.info("Task analysis completed successfully: %s", task_name)

    return task_output_dir
