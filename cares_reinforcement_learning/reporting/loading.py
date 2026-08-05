from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypeVar, cast

import pandas as pd
from pydantic import BaseModel, ValidationError

from cares_reinforcement_learning.algorithm.configurations import (
    AlgorithmConfig,
    TrainingConfig,
)
from cares_reinforcement_learning.envs.configurations import (
    GymEnvironmentConfig,
)
from cares_reinforcement_learning.reporting.models import (
    ComparisonIdentity,
    DiscoveredRun,
    LoadedRun,
    LoadedTask,
    RunConfiguration,
    SeedData,
)

REQUIRED_CONFIG_FILES: dict[str, str] = {
    "alg_config": "alg_config.json",
    "env_config": "env_config.json",
    "train_config": "train_config.json",
}

ConfigModel = TypeVar("ConfigModel", bound=BaseModel)


def _load_json_object(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return cast(dict[str, Any], value)


def _all_subclasses(model_type: type[ConfigModel]) -> Iterator[type[ConfigModel]]:
    """Yield every currently imported concrete subclass recursively."""
    for subclass in model_type.__subclasses__():
        yield subclass
        yield from _all_subclasses(subclass)


def _algorithm_registry() -> dict[str, type[AlgorithmConfig]]:
    registry: dict[str, type[AlgorithmConfig]] = {}

    for config_type in _all_subclasses(AlgorithmConfig):
        field = config_type.model_fields.get("algorithm")
        if field is None:
            continue

        default = field.default
        if isinstance(default, str) and default:
            previous = registry.get(default)
            if previous is not None and previous is not config_type:
                raise RuntimeError(
                    f"Algorithm configuration {default!r} is defined by both "
                    f"{previous.__name__} and {config_type.__name__}."
                )
            registry[default] = config_type

    return registry


def _environment_registry() -> dict[str, type[GymEnvironmentConfig]]:
    registry: dict[str, type[GymEnvironmentConfig]] = {}

    for config_type in _all_subclasses(GymEnvironmentConfig):
        gym = getattr(config_type, "gym", None)
        if not isinstance(gym, str) or not gym:
            continue

        previous = registry.get(gym)
        if previous is not None and previous is not config_type:
            raise RuntimeError(
                f"Environment configuration {gym!r} is defined by both "
                f"{previous.__name__} and {config_type.__name__}."
            )
        registry[gym] = config_type

    return registry


def _validate_model(
    config_type: type[ConfigModel],
    data: dict[str, Any],
    path: pathlib.Path,
) -> ConfigModel:
    try:
        return config_type.model_validate(data)
    except ValidationError as error:
        raise ValueError(
            f"Configuration validation failed for {path} using "
            f"{config_type.__name__}:\n{error}"
        ) from error


def _load_algorithm_config(path: pathlib.Path) -> AlgorithmConfig:
    data = _load_json_object(path)
    algorithm = data.get("algorithm")

    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError(f"Missing valid string field 'algorithm' in {path}")

    algorithm = algorithm.strip()
    registry = _algorithm_registry()
    config_type = registry.get(algorithm)

    if config_type is None:
        raise ValueError(
            f"Unsupported algorithm {algorithm!r} in {path}. "
            f"Known algorithms: {sorted(registry)}"
        )

    return _validate_model(config_type, data, path)


def _load_environment_config(path: pathlib.Path) -> GymEnvironmentConfig:
    data = _load_json_object(path)
    gym = data.get("gym")

    if not isinstance(gym, str) or not gym.strip():
        raise ValueError(f"Missing valid string field 'gym' in {path}")

    gym = gym.strip()
    registry = _environment_registry()
    config_type = registry.get(gym)

    if config_type is None:
        raise ValueError(
            f"Unsupported environment type {gym!r} in {path}. "
            f"Known environment types: {sorted(registry)}"
        )

    return _validate_model(config_type, data, path)


def _load_training_config(path: pathlib.Path) -> TrainingConfig:
    data = _load_json_object(path)
    return _validate_model(TrainingConfig, data, path)


def _load_run_configuration(root: pathlib.Path) -> RunConfiguration:
    """Load and validate all three CARES RL configuration files for a run."""
    return RunConfiguration(
        algorithm=_load_algorithm_config(root / REQUIRED_CONFIG_FILES["alg_config"]),
        environment=_load_environment_config(
            root / REQUIRED_CONFIG_FILES["env_config"]
        ),
        training=_load_training_config(root / REQUIRED_CONFIG_FILES["train_config"]),
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
                return None
            value = value[field]
        else:
            raise ValueError(
                f"Configuration path {path!r} was not found; "
                f"{traversed.rsplit('.', 1)[0]!r} resolves to "
                f"{type(value).__name__}, not a nested configuration."
            )

    if isinstance(value, (BaseModel, Mapping, set)):
        raise ValueError(
            f"Configuration path {path!r} resolves to a non-scalar "
            f"{type(value).__name__}; comparison parameters must be scalar "
            "or a sequence of scalar values."
        )

    if isinstance(value, (list, tuple)):
        if any(
            isinstance(item, (BaseModel, Mapping, list, tuple, set)) for item in value
        ):
            raise ValueError(
                f"Configuration path {path!r} contains nested non-scalar values."
            )
        return tuple(value)

    return value


def _build_discovered_run(
    run_directory: pathlib.Path,
    comparison_parameters: Sequence[str],
) -> DiscoveredRun:
    configuration = _load_run_configuration(run_directory)
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


def _discover_tasks(
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
        return {root_path.name: direct_comparisons}

    tasks: dict[str, tuple[DiscoveredRun, ...]] = {}
    for task_directory in sorted(root_path.iterdir(), key=lambda path: path.name):
        if not task_directory.is_dir():
            continue

        comparisons = _discover_comparisons(task_directory, parameters)
        if not comparisons:
            continue

        tasks[task_directory.name] = comparisons

    if not tasks:
        raise ValueError(
            f"No valid CARES RL task directories were found in {root_path}. "
            "Expected result-run directories directly inside the input "
            "directory, or task directories containing result runs."
        )

    return tasks


def _discover_task(
    root: pathlib.Path,
    comparison_parameters: Sequence[str] = (),
) -> tuple[str, tuple[DiscoveredRun, ...]]:
    """Discover one task directory using the existing statistics layout."""
    tasks = _discover_tasks(root, comparison_parameters=comparison_parameters)
    if len(tasks) != 1:
        names = ", ".join(sorted(tasks))
        raise ValueError(
            f"--task expected one task directory, but discovered {len(tasks)} "
            f"tasks: {names}. Use --tasks for a root containing multiple tasks."
        )
    return next(iter(tasks.items()))


def _discover_runs(
    roots: Sequence[pathlib.Path],
    comparison_parameters: Sequence[str] = (),
) -> tuple[DiscoveredRun, ...]:
    """Build plotting runs from explicitly supplied algorithm result folders."""
    parameters = tuple(comparison_parameters)
    if len(parameters) != len(set(parameters)):
        raise ValueError("Each comparison parameter may be supplied only once.")

    discovered: list[DiscoveredRun] = []
    seen_names: dict[str, pathlib.Path] = {}
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Input directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {root}")
        if not _is_result_run(root):
            expected = ", ".join(REQUIRED_CONFIG_FILES.values())
            raise ValueError(
                f"Arbitrary data directory {root} is not a CARES RL result run. "
                f"Expected configuration files: {expected}."
            )

        run = _build_discovered_run(root, parameters)
        previous = seen_names.get(run.comparison_name)
        if previous is not None:
            raise ValueError(
                f"Multiple supplied runs resolve to comparison label "
                f"{run.comparison_name!r}: {previous} and {root}. Add the "
                "changed fields with --comparison-parameter."
            )
        seen_names[run.comparison_name] = root
        discovered.append(run)

    if not discovered:
        raise ValueError("At least one arbitrary algorithm directory is required.")
    return tuple(discovered)


def _infer_task_name(runs: Sequence[DiscoveredRun]) -> str:
    """Infer a readable title when arbitrary runs share one environment."""
    names: list[str] = []
    for run in runs:
        environment = run.configuration.environment
        data = environment.model_dump()
        domain = str(data.get("domain") or "").strip()
        task = str(data.get("task") or "").strip()
        if task:
            names.append(f"{domain}-{task}" if domain else task)
            continue

        env_name = str(data.get("env_name") or data.get("gym") or "").strip()
        if env_name:
            names.append(env_name)

    unique = tuple(dict.fromkeys(names))
    return unique[0] if len(unique) == 1 else "comparison"


def _optional_csv(
    seed_root: pathlib.Path, name: str
) -> tuple[pathlib.Path | None, pd.DataFrame | None]:
    path = seed_root / "data" / f"{name}.csv"
    if not path.is_file():
        return None, None
    return path, pd.read_csv(path)


def _load_run(discovered: DiscoveredRun) -> LoadedRun:
    """Load every available train/eval log for a discovered run."""
    root = discovered.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Algorithm run directory does not exist: {root}")

    configured_algorithm = discovered.configuration.algorithm.algorithm.strip()
    if configured_algorithm != discovered.algorithm:
        raise ValueError(
            f"Discovered algorithm {discovered.algorithm!r} does not match "
            f"alg_config.json value {configured_algorithm!r} in {root}."
        )

    seeds: dict[int, SeedData] = {}
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue

        try:
            seed = int(child.name)
        except ValueError:
            continue

        train_path, train_data = _optional_csv(child, "train")
        eval_path, eval_data = _optional_csv(child, "eval")
        if train_data is None and eval_data is None:
            continue

        seeds[seed] = SeedData(
            seed=seed,
            root=child,
            train_path=train_path,
            eval_path=eval_path,
            train_data=train_data,
            eval_data=eval_data,
        )

    if not seeds:
        raise ValueError(
            "No numeric seed directories containing data/train.csv or "
            f"data/eval.csv were found in {root}."
        )

    return LoadedRun(discovered=discovered, seeds=seeds)


def _load_task_from_discovered(
    name: str,
    discovered_runs: Sequence[DiscoveredRun],
) -> LoadedTask:
    return LoadedTask(
        name=name,
        runs=tuple(_load_run(run) for run in discovered_runs),
    )


def load_tasks(
    root: str | pathlib.Path,
    comparison_parameters: Sequence[str] = (),
) -> dict[str, LoadedTask]:
    """Discover and fully load one task root or a multi-task root."""
    discovered_tasks = _discover_tasks(
        root,
        comparison_parameters=comparison_parameters,
    )
    return {
        task_name: _load_task_from_discovered(task_name, discovered_runs)
        for task_name, discovered_runs in discovered_tasks.items()
    }


def load_task(
    root: str | pathlib.Path,
    comparison_parameters: Sequence[str] = (),
) -> LoadedTask:
    """Discover and fully load exactly one task directory."""
    task_name, discovered_runs = _discover_task(
        pathlib.Path(root),
        comparison_parameters=comparison_parameters,
    )
    return _load_task_from_discovered(task_name, discovered_runs)


def load_runs(
    roots: Sequence[pathlib.Path],
    comparison_parameters: Sequence[str] = (),
) -> LoadedTask:
    """Discover explicit run directories and fully load them as one task."""
    discovered_runs = _discover_runs(
        roots,
        comparison_parameters=comparison_parameters,
    )
    task_name = _infer_task_name(discovered_runs)
    return _load_task_from_discovered(task_name, discovered_runs)
