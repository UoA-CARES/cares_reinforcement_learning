from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from cares_reinforcement_learning.stats.metrics import aggregate_evaluation_curve
from cares_reinforcement_learning.stats.models import (
    AlgorithmRun,
    AnalysisOptions,
    ComparisonDesign,
    MetricSpec,
    ValidationResult,
)

ENV_MATCH_KEYS = (
    "domain",
    "task",
    "gym",
    "state_std",
    "action_std",
    "frames_to_stack",
    "frame_width",
    "frame_height",
    "grey_scale",
)
TRAIN_MATCH_KEYS = ("number_steps_per_evaluation", "number_eval_episodes")
ALG_MATCH_KEYS = ("max_steps_training",)


def _matching_values(
    runs: Sequence[AlgorithmRun], config_name: str, keys: Sequence[str]
) -> None:
    for key in keys:
        values = {
            run.comparison_name: getattr(
                getattr(run.configuration, config_name), key, None
            )
            for run in runs
        }
        if len({repr(value) for value in values.values()}) != 1:
            raise ValueError(
                f"Required experiment setting differs for {config_name}.{key}: {values}"
            )


def _validate_frame(
    seed_name: str, frame: pd.DataFrame, metrics: Sequence[MetricSpec]
) -> None:
    required = {"total_steps", *[metric.column for metric in metrics]}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{seed_name} is missing required eval.csv columns: {missing}")
    steps = pd.to_numeric(frame["total_steps"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(steps).all():
        raise ValueError(f"{seed_name} contains non-finite total_steps values.")
    for metric in metrics:
        values = pd.to_numeric(frame[metric.column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(
                f"{seed_name} contains non-numeric or non-finite values in {metric.column!r}."
            )
        aggregate_evaluation_curve(frame, metric.column)


def validate_runs(
    runs: Sequence[AlgorithmRun],
    metrics: Sequence[MetricSpec],
    options: AnalysisOptions,
) -> ValidationResult:
    if len(runs) < 2:
        raise ValueError("At least two comparison conditions are required.")
    if len({run.comparison_name for run in runs}) != len(runs):
        raise ValueError("Comparison names must be unique.")
    if not metrics:
        raise ValueError("At least one evaluation metric is required.")
    if len({metric.column for metric in metrics}) != len(metrics):
        raise ValueError("Evaluation metric columns must be unique.")

    _matching_values(runs, "environment", ENV_MATCH_KEYS)
    _matching_values(runs, "training", TRAIN_MATCH_KEYS)
    _matching_values(runs, "algorithm", ALG_MATCH_KEYS)

    reference_steps: np.ndarray | None = None
    reference_episodes: np.ndarray | None = None
    for run in runs:
        expected_episodes = run.configuration.training.number_eval_episodes
        for seed, seed_run in sorted(run.seeds.items()):
            name = f"{run.comparison_name} seed {seed}"
            _validate_frame(name, seed_run.eval_data, metrics)
            counts = (
                seed_run.eval_data.groupby("total_steps", sort=True).size().to_numpy()
            )
            if not np.all(counts == expected_episodes):
                raise ValueError(
                    f"{name} does not contain exactly {expected_episodes} evaluation episodes at every step. "
                    f"Observed counts: {np.unique(counts).tolist()}"
                )
            steps = np.sort(seed_run.eval_data["total_steps"].unique()).astype(
                np.float64
            )
            if reference_steps is None:
                reference_steps = steps
                reference_episodes = counts
            elif not np.array_equal(reference_steps, steps):
                raise ValueError(
                    f"The complete evaluation step grid differs for {name}."
                )
            elif not np.array_equal(reference_episodes, counts):
                raise ValueError(
                    f"The number of evaluation episodes per step differs for {name}."
                )

    seed_sets = {run.comparison_name: set(run.seeds) for run in runs}
    matched = len({frozenset(value) for value in seed_sets.values()}) == 1
    if matched:
        return ValidationResult(ComparisonDesign.PAIRED, ())
    if not options.allow_unmatched_seeds:
        raise ValueError(
            f"Seed IDs/counts differ across comparison conditions: {seed_sets}. "
            "Set allow_unmatched_seeds=True only for an explicitly independent comparison."
        )
    warning = (
        "Seed IDs or counts differ. Pairwise tests use independent Mann–Whitney U comparisons; "
        "all other experiment compatibility checks remain strict."
    )
    return ValidationResult(ComparisonDesign.INDEPENDENT, (warning,))
