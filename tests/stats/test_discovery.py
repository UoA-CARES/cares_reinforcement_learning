import json
from pathlib import Path

import pandas as pd
import pytest

from cares_rl_statistics.discovery import discover_tasks


def _algorithm(root: Path, task: str, algorithm: str) -> None:
    path = root / task / algorithm
    path.mkdir(parents=True)
    for name in ("alg_config.json", "env_config.json", "train_config.json"):
        (path / name).write_text(json.dumps({}), encoding="utf-8")
    seed = path / "10" / "data"
    seed.mkdir(parents=True)
    pd.DataFrame({"step": [1], "episode_reward": [1]}).to_csv(
        seed / "eval.csv", index=False
    )


def test_discovers_multiple_tasks(tmp_path: Path) -> None:
    for task in ("ball_in_cup", "reacher"):
        for algorithm in ("SAC", "TD3"):
            _algorithm(tmp_path, task, algorithm)
    tasks = discover_tasks(tmp_path)
    assert list(tasks) == ["ball_in_cup", "reacher"]
    assert list(tasks["ball_in_cup"]) == ["SAC", "TD3"]


def test_discovers_single_task(tmp_path: Path) -> None:
    task = tmp_path / "swingup"
    for algorithm in ("PPO", "SAC"):
        _algorithm(tmp_path, "swingup", algorithm)
    tasks = discover_tasks(task)
    assert list(tasks) == ["swingup"]


def test_rejects_task_with_one_algorithm(tmp_path: Path) -> None:
    _algorithm(tmp_path, "reacher", "SAC")
    with pytest.raises(ValueError, match="fewer than two algorithms"):
        discover_tasks(tmp_path)
