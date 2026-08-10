from __future__ import annotations

import dataclasses
import enum
import pathlib
from typing import Literal

import pandas as pd

MetricDirection = Literal["higher", "lower"]


class ComparisonDesign(str, enum.Enum):
    PAIRED = "paired"
    INDEPENDENT = "independent"


@dataclasses.dataclass(frozen=True)
class MetricSpec:
    column: str
    direction: MetricDirection = "higher"


@dataclasses.dataclass(frozen=True)
class AnalysisOptions:
    early_window_fraction: float = 0.25
    final_window_fraction: float = 0.10
    bootstrap_samples: int = 10_000
    bootstrap_confidence: float = 0.95
    random_seed: int = 0
    allow_unmatched_seeds: bool = False
    primary_performance_metric: str = "auc"
    significance_level: float = 0.05
    evaluation_step_column: str = "total_steps"

    def __post_init__(self) -> None:
        if not 0.0 < self.early_window_fraction <= 1.0:
            raise ValueError("early_window_fraction must be in (0, 1].")
        if not 0.0 < self.final_window_fraction <= 1.0:
            raise ValueError("final_window_fraction must be in (0, 1].")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive.")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must be in (0, 1).")
        if self.primary_performance_metric not in {
            "auc",
            "early_window_auc",
            "final_window_auc",
        }:
            raise ValueError("Unsupported primary_performance_metric.")
        if not 0.0 < self.significance_level < 1.0:
            raise ValueError("significance_level must be in (0, 1).")
        if not self.evaluation_step_column.strip():
            raise ValueError("evaluation_step_column must not be empty.")


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    comparison_design: ComparisonDesign
    warnings: tuple[str, ...]


def _write_csv_outputs(
    outputs: dict[str, pd.DataFrame],
    output: str | pathlib.Path,
) -> None:
    directory = pathlib.Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(directory / filename, index=False)


@dataclasses.dataclass(frozen=True)
class TaskAnalysisResult:
    seed_metrics: pd.DataFrame
    algorithm_summary: pd.DataFrame
    pairwise: pd.DataFrame
    task_summary: pd.DataFrame

    def csv_outputs(self) -> dict[str, pd.DataFrame]:
        return {
            "seed_metrics.csv": self.seed_metrics,
            "algorithm_summary.csv": self.algorithm_summary,
            "pairwise_comparisons.csv": self.pairwise,
            "task_summary_all_metrics.csv": self.task_summary,
        }

    def write_csv(self, output: str | pathlib.Path) -> None:
        _write_csv_outputs(self.csv_outputs(), output)


@dataclasses.dataclass(frozen=True)
class BenchmarkAnalysisResult:
    benchmark_summary: pd.DataFrame
    cross_task_pairwise: pd.DataFrame
    friedman_tests: pd.DataFrame
    nemenyi_posthoc: pd.DataFrame
    task_superiority: pd.DataFrame
    task_algorithm_summaries: pd.DataFrame
    task_pairwise_comparisons: pd.DataFrame
    task_seed_metrics: pd.DataFrame

    def csv_outputs(self) -> dict[str, pd.DataFrame]:
        return {
            "task_algorithm_summaries.csv": self.task_algorithm_summaries,
            "task_pairwise_comparisons.csv": self.task_pairwise_comparisons,
            "task_seed_metrics.csv": self.task_seed_metrics,
            "task_superiority.csv": self.task_superiority,
            "benchmark_summary.csv": self.benchmark_summary,
            "cross_task_pairwise.csv": self.cross_task_pairwise,
            "friedman_tests.csv": self.friedman_tests,
            "nemenyi_posthoc.csv": self.nemenyi_posthoc,
        }

    def write_csv(self, output: str | pathlib.Path) -> None:
        _write_csv_outputs(self.csv_outputs(), output)
