from pathlib import Path

import pandas as pd
import pytest

from cares_rl_statistics.io import load_algorithm_run
from cares_rl_statistics.models import AnalysisOptions, ComparisonDesign, MetricSpec
from cares_rl_statistics.validation import validate_runs


def _make_run(root: Path, name: str, seeds: list[int]) -> Path:
    path = root / name
    path.mkdir()
    (path / "alg_config.json").write_text('{"max_steps_training": 20}')
    (path / "env_config.json").write_text('{"domain":"x","task":"y","gym":"z"}')
    (path / "train_config.json").write_text(
        '{"number_steps_per_evaluation":10,"number_eval_episodes":2}'
    )
    for seed in seeds:
        data_dir = path / str(seed) / "data"
        data_dir.mkdir(parents=True)
        pd.DataFrame(
            {"total_steps": [10, 10, 20, 20], "episode_reward": [1.0, 2.0, 3.0, 4.0]}
        ).to_csv(data_dir / "eval.csv", index=False)
    return path


def test_seed_mismatch_is_the_only_optional_relaxation(tmp_path: Path):
    a = load_algorithm_run("A", _make_run(tmp_path, "a", [1, 2]))
    b = load_algorithm_run("B", _make_run(tmp_path, "b", [3]))
    metrics = [MetricSpec("episode_reward")]
    with pytest.raises(ValueError, match="Seed IDs/counts differ"):
        validate_runs([a, b], metrics, AnalysisOptions())
    result = validate_runs([a, b], metrics, AnalysisOptions(allow_unmatched_seeds=True))
    assert result.comparison_design is ComparisonDesign.INDEPENDENT


def test_step_grid_mismatch_is_never_relaxed(tmp_path: Path):
    a_path = _make_run(tmp_path, "a", [1])
    b_path = _make_run(tmp_path, "b", [1])
    frame = pd.read_csv(b_path / "1" / "data" / "eval.csv")
    frame.loc[frame["total_steps"] == 20, "total_steps"] = 30
    frame.to_csv(b_path / "1" / "data" / "eval.csv", index=False)
    with pytest.raises(ValueError, match="complete evaluation step grid"):
        validate_runs(
            [load_algorithm_run("A", a_path), load_algorithm_run("B", b_path)],
            [MetricSpec("episode_reward")],
            AnalysisOptions(allow_unmatched_seeds=True),
        )
