from __future__ import annotations

import dataclasses
import enum
import pathlib
from collections.abc import Mapping
from typing import Any, Literal

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
class ComparisonIdentity:
    """Identity of one condition included in a statistical comparison."""

    algorithm: str
    parameters: tuple[tuple[str, Any], ...] = ()

    @property
    def comparison_name(self) -> str:
        if not self.parameters:
            return self.algorithm

        parameter_text = ", ".join(
            f"{name}={_format_parameter_value(value)}"
            for name, value in self.parameters
        )
        return f"{self.algorithm} [{parameter_text}]"

    @property
    def variant_parameters(self) -> Mapping[str, Any]:
        return dict(self.parameters)


def _format_parameter_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    if value is None:
        return "null"
    return str(value)


@dataclasses.dataclass(frozen=True)
class DiscoveredRun:
    """A result directory and the comparison identity derived from its configs."""

    comparison_name: str
    algorithm: str
    variant_parameters: Mapping[str, Any]
    root: pathlib.Path


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
    generate_statistical_figures: bool = True
    figure_dpi: int = 300

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
        if self.figure_dpi < 72:
            raise ValueError("figure_dpi must be at least 72.")


@dataclasses.dataclass(frozen=True)
class SeedRun:
    comparison_name: str
    seed: int
    root: pathlib.Path
    eval_path: pathlib.Path
    eval_data: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class AlgorithmRun:
    comparison_name: str
    algorithm: str
    variant_parameters: Mapping[str, Any]
    root: pathlib.Path
    alg_config: Mapping[str, Any]
    env_config: Mapping[str, Any]
    train_config: Mapping[str, Any]
    seeds: Mapping[int, SeedRun]


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    comparison_design: ComparisonDesign
    warnings: tuple[str, ...]
