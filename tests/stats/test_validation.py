from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cares_reinforcement_learning.stats.io import (
    load_algorithm_run,
    load_run_configuration,
)
from cares_reinforcement_learning.stats.models import (
    AnalysisOptions,
    ComparisonDesign,
    DiscoveredRun,
    MetricSpec,
)
from cares_reinforcement_learning.stats.validation import validate_runs


def _make_run(
    root: Path,
    comparison_name: str,
    algorithm: str,
    seeds: list[int],
) -> DiscoveredRun:
    path = root / comparison_name
    path.mkdir()

    (path / "alg_config.json").write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "max_steps_training": 20,
            }
        ),
        encoding="utf-8",
    )

    (path / "env_config.json").write_text(
        json.dumps(
            {
                "domain": "control_suite",
                "task": "test_task",
                "gym": "dmcs",
            }
        ),
        encoding="utf-8",
    )

    (path / "train_config.json").write_text(
        json.dumps(
            {
                "number_steps_per_evaluation": 10,
                "number_eval_episodes": 2,
            }
        ),
        encoding="utf-8",
    )

    for seed in seeds:
        data_dir = path / str(seed) / "data"
        data_dir.mkdir(parents=True)

        pd.DataFrame(
            {
                "total_steps": [10, 10, 20, 20],
                "episode_reward": [1.0, 2.0, 3.0, 4.0],
            }
        ).to_csv(
            data_dir / "eval.csv",
            index=False,
        )

    return DiscoveredRun(
        comparison_name=comparison_name,
        algorithm=algorithm,
        variant_parameters={},
        root=path,
        configuration=load_run_configuration(path),
    )


def test_loaded_run_keeps_typed_configuration(tmp_path: Path) -> None:
    discovered = _make_run(tmp_path, "A", "SAC", [1, 2])
    run = load_algorithm_run(discovered)

    assert run.configuration is discovered.configuration
    assert run.configuration.algorithm.algorithm == "SAC"
    assert run.configuration.environment.gym == "dmcs"
    assert run.configuration.training.number_eval_episodes == 2


def test_seed_mismatch_is_the_only_optional_relaxation(
    tmp_path: Path,
) -> None:
    a = load_algorithm_run(_make_run(tmp_path, "A", "SAC", [1, 2]))
    b = load_algorithm_run(_make_run(tmp_path, "B", "TD3", [3]))

    metrics = [MetricSpec("episode_reward")]

    with pytest.raises(
        ValueError,
        match="Seed IDs/counts differ",
    ):
        validate_runs(
            [a, b],
            metrics,
            AnalysisOptions(),
        )

    result = validate_runs(
        [a, b],
        metrics,
        AnalysisOptions(
            allow_unmatched_seeds=True,
        ),
    )

    assert result.comparison_design is ComparisonDesign.INDEPENDENT


def test_step_grid_mismatch_is_never_relaxed(
    tmp_path: Path,
) -> None:
    a_discovered = _make_run(tmp_path, "A", "SAC", [1])
    b_discovered = _make_run(tmp_path, "B", "TD3", [1])

    b_eval_path = b_discovered.root / "1" / "data" / "eval.csv"

    frame = pd.read_csv(b_eval_path)
    frame.loc[
        frame["total_steps"] == 20,
        "total_steps",
    ] = 30
    frame.to_csv(
        b_eval_path,
        index=False,
    )

    a = load_algorithm_run(a_discovered)
    b = load_algorithm_run(b_discovered)

    with pytest.raises(
        ValueError,
        match="complete evaluation step grid",
    ):
        validate_runs(
            [a, b],
            [MetricSpec("episode_reward")],
            AnalysisOptions(
                allow_unmatched_seeds=True,
            ),
        )
