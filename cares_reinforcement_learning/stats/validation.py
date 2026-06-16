from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from . import utils
from .data import AlgorithmRun, SeedRun

ValidationSeverity = Literal["error", "warning", "info"]


class ValidationMode(str, enum.Enum):
    STRICT = "strict"
    EXPLORATORY = "exploratory"


@dataclasses.dataclass(frozen=True)
class ValidationIssue:
    severity: ValidationSeverity
    check_name: str
    message: str
    recommendation: str
    override_flag: str | None = None


@dataclasses.dataclass(frozen=True)
class ValidationPolicy:
    """Defines which config values must match for publication-quality comparison."""

    required_match_keys: Mapping[str, Sequence[str]] = dataclasses.field(
        default_factory=lambda: {
            "env_config.json": [
                "domain",
                "task",
                "gym",
            ],
            "train_config.json": [
                "number_steps_per_evaluation",
                "number_eval_episodes",
            ],
        }
    )
    ignored_keys: Mapping[str, Sequence[str]] = dataclasses.field(
        default_factory=lambda: {
            "env_config.json": [
                "display",
                "save_train_checkpoints",
                "record_video_fps",
                "frames_to_stack",
                "frame_width",
                "frame_height",
                "grey_scale",
            ],
            "train_config.json": [
                "record_eval_video",
                "checkpoint_interval",
                "max_workers",
            ],
        }
    )


@dataclasses.dataclass(frozen=True)
class AnalysisOptions:
    final_window_fraction: float = 0.05
    final_window_min_points: int = 1
    bootstrap_samples: int = 10_000
    bootstrap_confidence: float = 0.95
    random_seed: int = 0
    validation_mode: ValidationMode = ValidationMode.STRICT
    allow_unmatched_seeds: bool = False
    allow_different_seed_counts: bool = False
    allow_config_mismatch: bool = False
    allow_step_mismatch: bool = False
    allow_missing_metrics: bool = False

    def __post_init__(self) -> None:
        if self.validation_mode != ValidationMode.EXPLORATORY:
            return

        # In exploratory mode, all validation mismatches are downgraded to warnings.
        object.__setattr__(self, "allow_unmatched_seeds", True)
        object.__setattr__(self, "allow_different_seed_counts", True)
        object.__setattr__(self, "allow_config_mismatch", True)
        object.__setattr__(self, "allow_step_mismatch", True)
        object.__setattr__(self, "allow_missing_metrics", True)


def _validate_seed_sets(
    runs: Sequence[AlgorithmRun],
    options: AnalysisOptions,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seed_sets = {run.algorithm: set(run.seeds) for run in runs}
    all_seed_sets = list(seed_sets.values())

    if len({frozenset(seed_set) for seed_set in all_seed_sets}) > 1:
        severity: ValidationSeverity = (
            "warning" if options.allow_unmatched_seeds else "error"
        )
        issues.append(
            ValidationIssue(
                severity=severity,
                check_name="seed_ids_match",
                message=f"Seed IDs differ across algorithms: {seed_sets}",
                recommendation=(
                    "Use matched seed IDs for publication-quality paired comparison. "
                    "Use --allow-unmatched-seeds only for exploratory/reference analysis."
                ),
                override_flag="--allow-unmatched-seeds",
            )
        )

    seed_counts = {run.algorithm: len(run.seeds) for run in runs}
    if len(set(seed_counts.values())) > 1:
        severity = "warning" if options.allow_different_seed_counts else "error"
        issues.append(
            ValidationIssue(
                severity=severity,
                check_name="seed_counts_match",
                message=f"Number of seeds differs across algorithms: {seed_counts}",
                recommendation=(
                    "Use the same number of seeds for each algorithm. "
                    "Use --allow-different-seed-counts only for exploratory/reference analysis."
                ),
                override_flag="--allow-different-seed-counts",
            )
        )

    return issues


def _validate_required_config_keys(
    runs: Sequence[AlgorithmRun],
    policy: ValidationPolicy,
    options: AnalysisOptions,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for config_name, keys in policy.required_match_keys.items():
        reference_run = runs[0]
        reference_config = reference_run.configs.get(config_name, {})

        for key in keys:
            reference_value = reference_config.get(key, None)
            values = {
                run.algorithm: run.configs.get(config_name, {}).get(key, None)
                for run in runs
            }
            if any(value != reference_value for value in values.values()):
                severity: ValidationSeverity = (
                    "warning" if options.allow_config_mismatch else "error"
                )
                issues.append(
                    ValidationIssue(
                        severity=severity,
                        check_name="required_config_keys_match",
                        message=(
                            f"Required config key differs: {config_name}:{key}. "
                            f"Values: {values}"
                        ),
                        recommendation=(
                            "Publication comparisons should use the same task and evaluation protocol. "
                            "Use --allow-config-mismatch only if this difference is intentional."
                        ),
                        override_flag="--allow-config-mismatch",
                    )
                )

    return issues


def _format_step_signature(signature: tuple[float, float, int]) -> str:
    start_step, end_step, num_evaluations = signature

    return (
        f"start={start_step:.0f}, "
        f"end={end_step:.0f}, "
        f"evaluations={num_evaluations}"
    )


def _extract_eval_step_signature(
    seed_run: SeedRun,
) -> tuple[float, float, int]:
    step_column = utils.find_step_column(seed_run.eval_data)

    steps = np.asarray(
        seed_run.eval_data[step_column],
        dtype=np.float64,
    )

    if steps.size == 0:
        raise ValueError(f"No evaluation steps found in {seed_run.eval_path}.")

    return (
        float(steps[0]),
        float(steps[-1]),
        int(steps.size),
    )


def _validate_eval_steps(
    runs: Sequence[AlgorithmRun],
    options: AnalysisOptions,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    reference_name: str | None = None
    reference_signature: tuple[float, float, int] | None = None
    mismatches: list[str] = []

    for run in runs:
        for seed, seed_run in sorted(run.seeds.items()):
            signature = _extract_eval_step_signature(seed_run)
            name = f"{run.algorithm} seed {seed}"

            if reference_signature is None:
                reference_name = name
                reference_signature = signature
                continue

            if signature != reference_signature:
                mismatches.append(f"{name}: {_format_step_signature(signature)}")

    if not mismatches:
        return issues

    if reference_name is None or reference_signature is None:
        raise RuntimeError(
            "Internal validation error: missing reference evaluation step signature."
        )

    severity: ValidationSeverity = "warning" if options.allow_step_mismatch else "error"

    message = (
        "Evaluation step grid mismatch detected.\n\n"
        "Reference:\n"
        f"  {reference_name}: {_format_step_signature(reference_signature)}\n\n"
        "Mismatches:\n" + "\n".join(f"  {line}" for line in mismatches)
    )

    issues.append(
        ValidationIssue(
            severity=severity,
            check_name="eval_step_grid_match",
            message=message,
            recommendation=(
                "Compare algorithms using the same evaluation cadence, number of "
                "evaluation points, and final training step for every seed. "
                "Use --allow-step-mismatch only for exploratory/reference analysis."
            ),
            override_flag="--allow-step-mismatch",
        )
    )

    return issues


def _validate_metric_columns(
    runs: Sequence[AlgorithmRun],
    options: AnalysisOptions,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    columns_by_algorithm: dict[str, set[str]] = {}

    for run in runs:
        first_seed = next(iter(run.seeds.values()))
        numeric_columns = {
            column
            for column in first_seed.eval_data.columns
            if pd.api.types.is_numeric_dtype(first_seed.eval_data[column])
        }
        columns_by_algorithm[run.algorithm] = numeric_columns

    if len({frozenset(columns) for columns in columns_by_algorithm.values()}) > 1:
        severity: ValidationSeverity = (
            "warning" if options.allow_missing_metrics else "error"
        )
        issues.append(
            ValidationIssue(
                severity=severity,
                check_name="metric_columns_match",
                message="Numeric metric columns differ across algorithms.",
                recommendation=(
                    "Use the intersection of common metrics or pass --allow-missing-metrics "
                    "for exploratory analysis."
                ),
                override_flag="--allow-missing-metrics",
            )
        )

    return issues


def validate_runs(
    runs: Sequence[AlgorithmRun],
    policy: ValidationPolicy,
    options: AnalysisOptions,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues.extend(_validate_seed_sets(runs, options))
    issues.extend(_validate_required_config_keys(runs, policy, options))
    issues.extend(_validate_eval_steps(runs, options))
    issues.extend(_validate_metric_columns(runs, options))

    blocking = [issue for issue in issues if issue.severity == "error"]
    if blocking and options.validation_mode == ValidationMode.STRICT:
        message = "\n".join(
            f"[{issue.check_name}] {issue.message}\n"
            f"Recommendation: {issue.recommendation}"
            for issue in blocking
        )
        raise ValueError(f"Validation failed in strict mode:\n{message}")

    return issues
