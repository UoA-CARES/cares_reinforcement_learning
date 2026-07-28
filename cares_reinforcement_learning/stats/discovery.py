from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from cares_reinforcement_learning.stats.io import (
    REQUIRED_CONFIG_FILES,
    load_run_configuration,
)
from cares_reinforcement_learning.stats.models import (
    ComparisonIdentity,
    DiscoveredRun,
    RunConfiguration,
)


def _is_result_run(path: pathlib.Path) -> bool:
    return path.is_dir() and all(
        (path / filename).is_file() for filename in REQUIRED_CONFIG_FILES.values()
    )


def _config_sources(
    configuration: RunConfiguration,
) -> dict[str, BaseModel]:
    return {
        "alg_config": configuration.algorithm,
        "env_config": configuration.environment,
        "train_config": configuration.training,
    }


def _resolve_config_value(
    configuration: RunConfiguration,
    path: str,
) -> Any:
    source, separator, field_path = path.partition(".")

    if not separator or not field_path:
        valid_sources = ", ".join(REQUIRED_CONFIG_FILES)
        raise ValueError(
            f"Invalid configuration path {path!r}. Paths must begin with one "
            f"of {valid_sources} and include a field name."
        )

    sources = _config_sources(configuration)
    if source not in sources:
        valid_sources = ", ".join(REQUIRED_CONFIG_FILES)
        raise ValueError(
            f"Unknown configuration source {source!r} in {path!r}. "
            f"Expected one of: {valid_sources}."
        )

    value: Any = sources[source]
    traversed = source
    for field in field_path.split("."):
        traversed = f"{traversed}.{field}"

        if isinstance(value, BaseModel):
            if field not in value.model_fields:
                raise ValueError(
                    f"Configuration path {path!r} was not found; "
                    f"{traversed!r} is not a field of "
                    f"{type(value).__name__}."
                )
            value = getattr(value, field)
        elif isinstance(value, Mapping):
            if field not in value:
                raise ValueError(
                    f"Configuration path {path!r} was not found; "
                    f"mapping key {traversed!r} is missing."
                )
            value = value[field]
        else:
            raise ValueError(
                f"Configuration path {path!r} was not found; "
                f"{traversed.rsplit('.', 1)[0]!r} resolves to "
                f"{type(value).__name__}, not a nested configuration."
            )

    if isinstance(value, (BaseModel, Mapping, list, tuple, set)):
        raise ValueError(
            f"Configuration path {path!r} resolves to a non-scalar "
            f"{type(value).__name__}; comparison parameters must be scalar."
        )

    return value


def _build_discovered_run(
    run_directory: pathlib.Path,
    comparison_parameters: Sequence[str],
) -> DiscoveredRun:
    configuration = load_run_configuration(run_directory)
    algorithm = configuration.algorithm.algorithm.strip()

    parameters = tuple(
        (path.rsplit(".", 1)[-1], _resolve_config_value(configuration, path))
        for path in comparison_parameters
    )

    parameter_names = [name for name, _ in parameters]
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError(
            "Comparison parameter names must be unique in generated labels. "
            "Use paths whose final field names differ."
        )

    identity = ComparisonIdentity(algorithm=algorithm, parameters=parameters)
    return DiscoveredRun(
        comparison_name=identity.comparison_name,
        algorithm=identity.algorithm,
        variant_parameters=identity.variant_parameters,
        root=run_directory,
        configuration=configuration,
    )


def _discover_comparisons(
    task_directory: pathlib.Path,
    comparison_parameters: Sequence[str],
) -> tuple[DiscoveredRun, ...]:
    comparisons: list[DiscoveredRun] = []
    seen: dict[str, pathlib.Path] = {}

    for run_directory in sorted(task_directory.iterdir(), key=lambda path: path.name):
        if not _is_result_run(run_directory):
            continue

        discovered = _build_discovered_run(run_directory, comparison_parameters)
        previous = seen.get(discovered.comparison_name)
        if previous is not None:
            reason = (
                "The selected comparison parameters do not uniquely identify "
                "the result conditions. Include every changed parameter."
                if comparison_parameters
                else "Use --comparison-parameter for an ablation study that "
                "contains multiple runs of the same algorithm."
            )
            raise ValueError(
                f"Task {task_directory.name!r} contains multiple result "
                f"directories for comparison {discovered.comparison_name!r}: "
                f"{previous} and {run_directory}. {reason}"
            )

        seen[discovered.comparison_name] = run_directory
        comparisons.append(discovered)

    return tuple(comparisons)


def _require_multiple_comparisons(
    task_name: str,
    comparisons: tuple[DiscoveredRun, ...],
) -> None:
    if len(comparisons) < 2:
        raise ValueError(
            f"Task {task_name!r} contains only {len(comparisons)} valid "
            "comparison condition. At least two conditions are required."
        )


def discover_tasks(
    root: str | pathlib.Path,
    comparison_parameters: Sequence[str] = (),
) -> dict[str, tuple[DiscoveredRun, ...]]:
    """Discover one task or a directory containing multiple tasks."""
    root_path = pathlib.Path(root).expanduser().resolve()

    if not root_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root_path}")

    parameters = tuple(comparison_parameters)
    if len(parameters) != len(set(parameters)):
        raise ValueError("Each comparison parameter may be supplied only once.")

    direct_comparisons = _discover_comparisons(root_path, parameters)
    if direct_comparisons:
        _require_multiple_comparisons(root_path.name, direct_comparisons)
        return {root_path.name: direct_comparisons}

    tasks: dict[str, tuple[DiscoveredRun, ...]] = {}
    for task_directory in sorted(root_path.iterdir(), key=lambda path: path.name):
        if not task_directory.is_dir():
            continue

        comparisons = _discover_comparisons(task_directory, parameters)
        if not comparisons:
            continue

        _require_multiple_comparisons(task_directory.name, comparisons)
        tasks[task_directory.name] = comparisons

    if not tasks:
        raise ValueError(
            f"No valid CARES RL task directories were found in {root_path}. "
            "Expected result-run directories directly inside the input "
            "directory, or task directories containing result runs."
        )

    return tasks
