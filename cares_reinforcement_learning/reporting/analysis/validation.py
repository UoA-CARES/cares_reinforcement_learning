from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

import cares_reinforcement_learning.reporting.analysis.metrics as metrics_module
from cares_reinforcement_learning.reporting.analysis.models import (
    AnalysisOptions,
    ComparisonDesign,
    MetricSpec,
    ValidationResult,
)
from cares_reinforcement_learning.reporting.models import LoadedRun

MATCHED_CONFIG_FIELDS: dict[str, tuple[str, ...]] = {
    "environment": (
        "domain",
        "task",
        "gym",
        "state_std",
        "action_std",
        "frames_to_stack",
        "frame_width",
        "frame_height",
        "grey_scale",
    ),
    "training": (
        "number_steps_per_evaluation",
        "number_eval_episodes",
    ),
    "algorithm": ("max_steps_training",),
}


def _validate_inputs(
    runs: Sequence[LoadedRun],
    metrics: Sequence[MetricSpec],
) -> None:
    if len(runs) < 2:
        raise ValueError("At least two comparison conditions are required.")
    if len({run.comparison_name for run in runs}) != len(runs):
        raise ValueError("Comparison names must be unique.")
    if not metrics:
        raise ValueError("At least one evaluation metric is required.")
    if len({metric.column for metric in metrics}) != len(metrics):
        raise ValueError("Evaluation metric columns must be unique.")


def _validate_matching_configuration(runs: Sequence[LoadedRun]) -> None:
    for config_name, fields in MATCHED_CONFIG_FIELDS.items():
        for field in fields:
            values = {
                run.comparison_name: getattr(
                    getattr(run.configuration, config_name),
                    field,
                    None,
                )
                for run in runs
            }
            if len({repr(value) for value in values.values()}) != 1:
                raise ValueError(
                    "Required experiment setting differs for "
                    f"{config_name}.{field}: {values}"
                )


def _validate_frame(
    seed_name: str,
    frame: pd.DataFrame,
    metrics: Sequence[MetricSpec],
    step_column: str,
) -> None:
    prepared = metrics_module.ensure_required_columns(
        frame,
        [step_column, *(metric.column for metric in metrics)],
        context=seed_name,
    )

    steps = pd.to_numeric(prepared[step_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(steps).all():
        raise ValueError(f"{seed_name} contains non-finite {step_column!r} values.")

    for metric in metrics:
        values = pd.to_numeric(prepared[metric.column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(
                f"{seed_name} contains non-numeric or non-finite values "
                f"in {metric.column!r}."
            )
        metrics_module.aggregate_evaluation_curve(prepared, metric.column, step_column)


def _validate_evaluation_protocol(
    runs: Sequence[LoadedRun],
    metrics: Sequence[MetricSpec],
    step_column: str,
) -> None:
    reference_steps: np.ndarray | None = None

    for run in runs:
        expected_episodes = run.configuration.training.number_eval_episodes
        for seed, seed_run in sorted(run.seeds.items()):
            name = f"{run.comparison_name} seed {seed}"
            if seed_run.eval_data is None:
                raise ValueError(f"{name} is missing required data/eval.csv.")

            _validate_frame(name, seed_run.eval_data, metrics, step_column)
            counts = (
                seed_run.eval_data.groupby(step_column, sort=True).size().to_numpy()
            )
            if not np.all(counts == expected_episodes):
                raise ValueError(
                    f"{name} does not contain exactly {expected_episodes} "
                    "evaluation episodes at every step. "
                    f"Observed counts: {np.unique(counts).tolist()}"
                )

            steps = np.sort(seed_run.eval_data[step_column].unique()).astype(np.float64)
            if reference_steps is None:
                reference_steps = steps
            elif not np.array_equal(reference_steps, steps):
                raise ValueError(
                    f"The complete evaluation step grid differs for {name}."
                )


def _comparison_design(
    runs: Sequence[LoadedRun],
    options: AnalysisOptions,
) -> ValidationResult:
    seed_sets = {run.comparison_name: set(run.seeds) for run in runs}
    matched = len({frozenset(value) for value in seed_sets.values()}) == 1
    if matched:
        return ValidationResult(ComparisonDesign.PAIRED, ())

    if not options.allow_unmatched_seeds:
        raise ValueError(
            f"Seed IDs/counts differ across comparison conditions: {seed_sets}. "
            "Set allow_unmatched_seeds=True only for an explicitly independent "
            "comparison."
        )

    return ValidationResult(
        ComparisonDesign.INDEPENDENT,
        (
            "Seed IDs or counts differ. Pairwise tests use independent "
            "Mann–Whitney U comparisons; all other experiment compatibility "
            "checks remain strict.",
        ),
    )


def validate_runs(
    runs: Sequence[LoadedRun],
    metrics: Sequence[MetricSpec],
    options: AnalysisOptions,
) -> ValidationResult:
    _validate_inputs(runs, metrics)
    _validate_matching_configuration(runs)
    _validate_evaluation_protocol(
        runs,
        metrics,
        options.evaluation_step_column,
    )
    return _comparison_design(runs, options)
