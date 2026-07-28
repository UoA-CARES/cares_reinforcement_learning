import json
from pathlib import Path

import pandas as pd
import pytest

from cares_reinforcement_learning.algorithm.configurations import (
    PPOConfig,
    SACConfig,
    TD3Config,
)
from cares_reinforcement_learning.envs.configurations import DMCSConfig
from cares_reinforcement_learning.stats.discovery import discover_tasks
from cares_reinforcement_learning.stats.models import RunConfiguration


def _algorithm(root: Path, task: str, algorithm: str) -> None:
    path = root / task / algorithm
    path.mkdir(parents=True)

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
                "gym": "dmcs",
                "domain": "control_suite",
                "task": task,
            }
        ),
        encoding="utf-8",
    )
    (path / "train_config.json").write_text(
        json.dumps(
            {
                "number_steps_per_evaluation": 10,
                "number_eval_episodes": 1,
            }
        ),
        encoding="utf-8",
    )

    seed = path / "10" / "data"
    seed.mkdir(parents=True)

    pd.DataFrame(
        {
            "total_steps": [10, 20],
            "episode_reward": [1.0, 2.0],
        }
    ).to_csv(seed / "eval.csv", index=False)


def test_discovers_multiple_tasks_with_typed_configurations(
    tmp_path: Path,
) -> None:
    for task in ("ball_in_cup", "reacher"):
        for algorithm in ("SAC", "TD3"):
            _algorithm(tmp_path, task, algorithm)

    tasks = discover_tasks(tmp_path)

    assert list(tasks) == ["ball_in_cup", "reacher"]

    runs = tasks["ball_in_cup"]
    assert [run.comparison_name for run in runs] == ["SAC", "TD3"]
    assert all(isinstance(run.configuration, RunConfiguration) for run in runs)
    assert isinstance(runs[0].configuration.algorithm, SACConfig)
    assert isinstance(runs[1].configuration.algorithm, TD3Config)
    assert all(isinstance(run.configuration.environment, DMCSConfig) for run in runs)
    assert all(
        run.configuration.training.number_steps_per_evaluation == 10 for run in runs
    )


def test_discovers_single_task(tmp_path: Path) -> None:
    task = tmp_path / "swingup"
    for algorithm in ("PPO", "SAC"):
        _algorithm(tmp_path, "swingup", algorithm)

    tasks = discover_tasks(task)

    assert list(tasks) == ["swingup"]
    assert isinstance(
        tasks["swingup"][0].configuration.algorithm,
        PPOConfig,
    )
    assert isinstance(
        tasks["swingup"][1].configuration.algorithm,
        SACConfig,
    )


def test_rejects_task_with_one_algorithm(tmp_path: Path) -> None:
    _algorithm(tmp_path, "reacher", "SAC")

    with pytest.raises(
        ValueError,
        match="At least two conditions are required",
    ):
        discover_tasks(tmp_path)


def test_rejects_invalid_algorithm_configuration(tmp_path: Path) -> None:
    _algorithm(tmp_path, "reacher", "SAC")
    config_path = tmp_path / "reacher" / "SAC" / "alg_config.json"
    config_path.write_text(
        json.dumps({"algorithm": "NOT_AN_ALGORITHM"}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported algorithm",
    ):
        discover_tasks(tmp_path)


def test_rejects_invalid_environment_configuration(tmp_path: Path) -> None:
    _algorithm(tmp_path, "reacher", "SAC")
    config_path = tmp_path / "reacher" / "SAC" / "env_config.json"
    config_path.write_text(
        json.dumps({"gym": "not_a_gym", "task": "reacher"}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported environment type",
    ):
        discover_tasks(tmp_path)
