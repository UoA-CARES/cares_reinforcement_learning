from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Any

import pandas as pd

from cares_reinforcement_learning.algorithm.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from cares_reinforcement_learning.envs.configurations import GymEnvironmentConfig


def _format_parameter_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    if value is None:
        return "null"
    return str(value)


@dataclasses.dataclass(frozen=True)
class ComparisonIdentity:
    algorithm: str
    parameters: tuple[tuple[str, Any], ...] = ()

    @property
    def comparison_name(self) -> str:
        if not self.parameters:
            return self.algorithm
        text = ", ".join(
            f"{name}={_format_parameter_value(value)}"
            for name, value in self.parameters
        )
        return f"{self.algorithm} [{text}]"

    @property
    def variant_parameters(self) -> Mapping[str, Any]:
        return dict(self.parameters)


@dataclasses.dataclass(frozen=True)
class RunConfiguration:
    algorithm: AlgorithmConfig
    environment: GymEnvironmentConfig
    training: TrainingConfig


@dataclasses.dataclass(frozen=True)
class DiscoveredRun:
    comparison_name: str
    algorithm: str
    variant_parameters: Mapping[str, Any]
    root: pathlib.Path
    configuration: RunConfiguration


@dataclasses.dataclass(frozen=True)
class SeedData:
    seed: int
    root: pathlib.Path
    train_path: pathlib.Path | None
    eval_path: pathlib.Path | None
    train_data: pd.DataFrame | None
    eval_data: pd.DataFrame | None


@dataclasses.dataclass(frozen=True)
class LoadedRun:
    discovered: DiscoveredRun
    seeds: Mapping[int, SeedData]

    @property
    def comparison_name(self) -> str:
        return self.discovered.comparison_name

    @property
    def algorithm(self) -> str:
        return self.discovered.algorithm

    @property
    def variant_parameters(self) -> Mapping[str, Any]:
        return self.discovered.variant_parameters

    @property
    def root(self) -> pathlib.Path:
        return self.discovered.root

    @property
    def configuration(self) -> RunConfiguration:
        return self.discovered.configuration

    def to_plot_run(self) -> PlotRun:
        return PlotRun(
            name=self.comparison_name,
            train_frames={
                seed_number: seed.train_data
                for seed_number, seed in self.seeds.items()
                if seed.train_data is not None
            },
            eval_frames={
                seed_number: seed.eval_data
                for seed_number, seed in self.seeds.items()
                if seed.eval_data is not None
            },
        )


@dataclasses.dataclass(frozen=True)
class LoadedTask:
    name: str
    runs: tuple[LoadedRun, ...]

    def to_plot_task(self) -> PlotTask:
        return PlotTask(
            name=self.name,
            runs=tuple(run.to_plot_run() for run in self.runs),
        )


@dataclasses.dataclass(frozen=True)
class PlotRun:
    name: str
    train_frames: Mapping[int, pd.DataFrame]
    eval_frames: Mapping[int, pd.DataFrame]


@dataclasses.dataclass(frozen=True)
class PlotTask:
    name: str
    runs: tuple[PlotRun, ...]
