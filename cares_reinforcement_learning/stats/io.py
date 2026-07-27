from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping
from typing import Any, cast

import pandas as pd

from cares_reinforcement_learning.stats.models import (
    AlgorithmRun,
    DiscoveredRun,
    SeedRun,
)

REQUIRED_CONFIG_FILES: dict[str, str] = {
    "alg_config": "alg_config.json",
    "env_config": "env_config.json",
    "train_config": "train_config.json",
}


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


def load_run_configs(root: pathlib.Path) -> dict[str, dict[str, Any]]:
    return {
        source: load_json_object(root / filename)
        for source, filename in REQUIRED_CONFIG_FILES.items()
    }


def _seed_eval_path(seed_dir: pathlib.Path) -> pathlib.Path:
    path = seed_dir / "data" / "eval.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation log is missing: {path}")
    return path


def load_algorithm_run(discovered: DiscoveredRun) -> AlgorithmRun:
    root = discovered.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Algorithm run directory does not exist: {root}")

    configs = load_run_configs(root)
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
        alg_config=cast(Mapping[str, Any], configs["alg_config"]),
        env_config=cast(Mapping[str, Any], configs["env_config"]),
        train_config=cast(Mapping[str, Any], configs["train_config"]),
        seeds=seeds,
    )
