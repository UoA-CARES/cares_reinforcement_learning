import json
from pathlib import Path

import pandas as pd

from cares_reinforcement_learning.stats.io import load_run_configuration
from cares_reinforcement_learning.stats.models import (
    AnalysisOptions,
    DiscoveredRun,
)
from cares_reinforcement_learning.stats.task_analysis import run_task_analysis


def _algorithm(
    root: Path,
    comparison_name: str,
    algorithm: str,
    seeds: list[int],
    offset: float,
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
        data = path / str(seed) / "data"
        data.mkdir(parents=True)
        pd.DataFrame(
            {
                "total_steps": [10, 10, 20, 20],
                "episode_reward": [
                    1 + offset + seed,
                    2 + offset + seed,
                    3 + offset + seed,
                    4 + offset + seed,
                ],
            }
        ).to_csv(data / "eval.csv", index=False)

    return DiscoveredRun(
        comparison_name=comparison_name,
        algorithm=algorithm,
        variant_parameters={},
        root=path,
        configuration=load_run_configuration(path),
    )


def test_unmatched_seed_analysis_is_independent_and_auditable(
    tmp_path: Path,
) -> None:
    a = _algorithm(tmp_path, "A", "SAC", [1, 2, 3], 0.0)
    b = _algorithm(tmp_path, "B", "TD3", [10, 20, 30, 40], 1.0)

    result = run_task_analysis(
        [a, b],
        tmp_path / "output",
        options=AnalysisOptions(
            allow_unmatched_seeds=True,
            bootstrap_samples=50,
            random_seed=2,
        ),
    )

    pairwise = result["pairwise"]
    assert set(pairwise["comparison_design"]) == {"independent"}
    assert set(pairwise["test"]) == {"mann_whitney_u"}
    assert set(pairwise["pairwise_input"]) == {"observed_seed_metrics"}
    assert set(result["algorithm_summary"]["bootstrap_method"]) == {"BCa"}
    assert set(result["algorithm_summary"]["bootstrap_resampling_unit"]) == {"seed"}


def test_publication_outputs_follow_statistical_hierarchy(
    tmp_path: Path,
) -> None:
    a = _algorithm(tmp_path, "A", "SAC", [1, 2, 3], 0.0)
    b = _algorithm(tmp_path, "B", "TD3", [1, 2, 3], 1.0)
    output = tmp_path / "output"

    result = run_task_analysis(
        [a, b],
        output,
        options=AnalysisOptions(
            bootstrap_samples=50,
            random_seed=2,
        ),
    )

    expected = {
        "task_performance.csv",
        "task_performance.tex",
        "pairwise_statistics.csv",
        "pairwise_statistics.tex",
        "task_summary.csv",
        "task_summary.tex",
        "methodology.md",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    assert set(result["task_summary"]["opponents"]) == {1}

    auc = result["task_summary"][result["task_summary"]["performance_metric"] == "auc"]
    superiority = sorted(auc["task_superiority"])
    assert superiority[0] < 0.5 < superiority[1]
    assert abs(sum(superiority) - 1.0) < 1e-12
