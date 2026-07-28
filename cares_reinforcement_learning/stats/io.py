from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator
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
from cares_reinforcement_learning.stats.models import (
    AlgorithmRun,
    DiscoveredRun,
    RunConfiguration,
    SeedRun,
)

REQUIRED_CONFIG_FILES: dict[str, str] = {
    "alg_config": "alg_config.json",
    "env_config": "env_config.json",
    "train_config": "train_config.json",
}

ConfigModel = TypeVar("ConfigModel", bound=BaseModel)


def load_json_object(path: pathlib.Path) -> dict[str, Any]:
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


def load_algorithm_config(path: pathlib.Path) -> AlgorithmConfig:
    data = load_json_object(path)
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


def load_environment_config(path: pathlib.Path) -> GymEnvironmentConfig:
    data = load_json_object(path)
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


def load_training_config(path: pathlib.Path) -> TrainingConfig:
    data = load_json_object(path)
    return _validate_model(TrainingConfig, data, path)


def load_run_configuration(root: pathlib.Path) -> RunConfiguration:
    """Load and validate all three CARES RL configuration files for a run."""
    return RunConfiguration(
        algorithm=load_algorithm_config(root / REQUIRED_CONFIG_FILES["alg_config"]),
        environment=load_environment_config(root / REQUIRED_CONFIG_FILES["env_config"]),
        training=load_training_config(root / REQUIRED_CONFIG_FILES["train_config"]),
    )


def _seed_eval_path(seed_dir: pathlib.Path) -> pathlib.Path:
    path = seed_dir / "data" / "eval.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation log is missing: {path}")
    return path


def load_algorithm_run(discovered: DiscoveredRun) -> AlgorithmRun:
    root = discovered.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Algorithm run directory does not exist: {root}")

    configuration = discovered.configuration

    configured_algorithm = configuration.algorithm.algorithm.strip()
    if configured_algorithm != discovered.algorithm:
        raise ValueError(
            f"Discovered algorithm {discovered.algorithm!r} does not match "
            f"alg_config.json value {configured_algorithm!r} in {root}."
        )

    seeds: dict[int, SeedRun] = {}
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue

        try:
            seed = int(child.name)
        except ValueError:
            continue

        eval_path = _seed_eval_path(child)
        seeds[seed] = SeedRun(
            comparison_name=discovered.comparison_name,
            seed=seed,
            root=child,
            eval_path=eval_path,
            eval_data=pd.read_csv(eval_path),
        )

    if not seeds:
        raise ValueError(
            "No numeric seed directories containing data/eval.csv were found "
            f"in {root}."
        )

    return AlgorithmRun(
        comparison_name=discovered.comparison_name,
        algorithm=discovered.algorithm,
        variant_parameters=dict(discovered.variant_parameters),
        root=root,
        configuration=configuration,
        seeds=seeds,
    )
