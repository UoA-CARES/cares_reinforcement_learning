from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

MetricDirection = Literal["higher", "lower"]


@dataclasses.dataclass(frozen=True)
class SeedRun:
    algorithm: str
    seed: int
    log_path: pathlib.Path
    eval_path: pathlib.Path
    eval_data: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class AlgorithmRun:
    algorithm: str
    log_path: pathlib.Path
    configs: Mapping[str, Mapping[str, Any]]
    seeds: Mapping[int, SeedRun]


@dataclasses.dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: MetricDirection = "higher"


@dataclasses.dataclass(frozen=True)
class SeedMetricSummary:
    algorithm: str
    seed: int
    evaluation_metric: str
    direction: MetricDirection

    n_eval_points: int
    first_step: float
    last_step: float

    auc: float
    early_auc_25: float

    final_window_mean: float
    final_window_std: float
    diagnostic_best_value: float
