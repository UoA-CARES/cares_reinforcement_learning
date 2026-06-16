from __future__ import annotations

import dataclasses
import logging
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from . import (
    reporting,
    statistics,
)
from .validation import AnalysisOptions

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class TaskAnalysis:
    task_name: str
    output_dir: pathlib.Path
    algorithm_summary: pd.DataFrame
    pairwise_comparisons: pd.DataFrame


def _with_task_column(frame: pd.DataFrame, task_name: str) -> pd.DataFrame:
    copied = frame.copy()
    copied["task"] = task_name
    columns = ["task"] + [column for column in copied.columns if column != "task"]
    return copied.loc[:, columns]


def _groupby_str(
    frame: pd.DataFrame,
    column: str,
) -> Iterable[tuple[str, pd.DataFrame]]:
    return cast(
        Iterable[tuple[str, pd.DataFrame]],
        frame.groupby(column),  # type: ignore[no-untyped-call]
    )


def _groupby_str_pair(
    frame: pd.DataFrame,
    columns: list[str],
) -> Iterable[tuple[tuple[str, str], pd.DataFrame]]:
    return cast(
        Iterable[tuple[tuple[str, str], pd.DataFrame]],
        frame.groupby(columns),  # type: ignore[no-untyped-call]
    )


def _load_task_analysis(task_name: str, output_dir: pathlib.Path) -> TaskAnalysis:
    return TaskAnalysis(
        task_name=task_name,
        output_dir=output_dir,
        algorithm_summary=pd.read_csv(output_dir / "summary_by_algorithm.csv"),
        pairwise_comparisons=pd.read_csv(output_dir / "pairwise_comparisons.csv"),
    )


def _combine_task_summaries(tasks: Sequence[TaskAnalysis]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for task in tasks:
        frame = _with_task_column(task.algorithm_summary, task.task_name)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _combine_task_pairwise(tasks: Sequence[TaskAnalysis]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for task in tasks:
        frame = _with_task_column(task.pairwise_comparisons, task.task_name)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _build_average_rank_table(
    task_summary: pd.DataFrame,
    evaluation_metric: str,
    performance_metric: str = "auc",
) -> pd.DataFrame:
    selected = task_summary[
        (task_summary["evaluation_metric"] == evaluation_metric)
        & (task_summary["performance_metric"] == performance_metric)
    ].copy()

    rows: list[dict[str, object]] = []
    for algorithm, group in _groupby_str(selected, "algorithm"):
        ranks = np.asarray(group["rank"], dtype=np.float64)

        rows.append(
            {
                "algorithm": algorithm,
                "n_tasks": int(len(group)),
                "average_rank": float(np.mean(ranks)),
                "median_rank": float(np.median(ranks)),
                "best_rank_count": int(np.sum(ranks == 1)),
                "worst_rank": int(np.max(ranks)),
            }
        )

    return pd.DataFrame(rows).sort_values("average_rank")


def _build_pairwise_win_rate_table(
    task_pairwise: pd.DataFrame,
    evaluation_metric: str,
    performance_metric: str = "auc",
) -> pd.DataFrame:
    selected = task_pairwise[
        (task_pairwise["evaluation_metric"] == evaluation_metric)
        & (task_pairwise["performance_metric"] == performance_metric)
    ].copy()

    algorithms = sorted(
        set(selected["algorithm_a"]).union(set(selected["algorithm_b"]))
    )

    wins = {algorithm: 0 for algorithm in algorithms}
    losses = {algorithm: 0 for algorithm in algorithms}
    ties = {algorithm: 0 for algorithm in algorithms}

    for _, row in selected.iterrows():
        algorithm_a = row["algorithm_a"]
        algorithm_b = row["algorithm_b"]
        direction = row["direction"]
        diff = float(row["mean_difference_b_minus_a"])

        if np.isclose(diff, 0.0):
            ties[algorithm_a] += 1
            ties[algorithm_b] += 1
            continue

        b_wins = diff > 0 if direction == "higher" else diff < 0

        if b_wins:
            wins[algorithm_b] += 1
            losses[algorithm_a] += 1
        else:
            wins[algorithm_a] += 1
            losses[algorithm_b] += 1

    rows: list[dict[str, object]] = []
    for algorithm in algorithms:
        total = wins[algorithm] + losses[algorithm] + ties[algorithm]
        rows.append(
            {
                "algorithm": algorithm,
                "wins": wins[algorithm],
                "losses": losses[algorithm],
                "ties": ties[algorithm],
                "total_comparisons": total,
                "win_rate": wins[algorithm] / total if total else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values("win_rate", ascending=False)


def _build_cross_task_superiority_table(
    task_pairwise: pd.DataFrame,
    evaluation_metric: str,
    performance_metric: str,
    options: AnalysisOptions,
) -> pd.DataFrame:
    selected = task_pairwise[
        (task_pairwise["evaluation_metric"] == evaluation_metric)
        & (task_pairwise["performance_metric"] == performance_metric)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No pairwise results found for evaluation metric {evaluation_metric!r} "
            f"and performance metric {performance_metric!r}."
        )

    algorithms = sorted(
        set(selected["algorithm_a"]).union(set(selected["algorithm_b"]))
    )

    superiority_scores: dict[str, list[float]] = {
        algorithm: [] for algorithm in algorithms
    }

    for _, row in selected.iterrows():
        algorithm_a = str(row["algorithm_a"])
        algorithm_b = str(row["algorithm_b"])

        probability_b_over_a = float(row["probability_b_improves_over_a"])

        superiority_scores[algorithm_b].append(probability_b_over_a)
        superiority_scores[algorithm_a].append(1.0 - probability_b_over_a)

    rng = np.random.default_rng(options.random_seed)

    rows: list[dict[str, object]] = []

    for algorithm, scores in superiority_scores.items():
        values = np.asarray(scores, dtype=np.float64)

        if values.size == 0:
            raise ValueError(
                f"No superiority scores found for algorithm {algorithm!r}."
            )

        ci_lower, ci_upper = statistics.bootstrap_ci(
            values.tolist(),
            np.mean,
            samples=options.bootstrap_samples,
            confidence=options.bootstrap_confidence,
            rng=rng,
        )

        rows.append(
            {
                "algorithm": algorithm,
                "mean_superiority": float(np.mean(values)),
                "median_superiority": float(np.median(values)),
                "std_superiority": (
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                ),
                "min_superiority": float(np.min(values)),
                "max_superiority": float(np.max(values)),
                "superiority_ci_lower": ci_lower,
                "superiority_ci_upper": ci_upper,
                "num_comparisons": int(values.size),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("mean_superiority", ascending=False)
        .reset_index(drop=True)
    )


def _build_cross_task_probability_table(
    task_pairwise: pd.DataFrame,
    evaluation_metric: str,
    performance_metric: str = "auc",
) -> pd.DataFrame:
    selected = task_pairwise[
        (task_pairwise["evaluation_metric"] == evaluation_metric)
        & (task_pairwise["performance_metric"] == performance_metric)
    ].copy()

    rows: list[dict[str, object]] = []

    for (algorithm_a, algorithm_b), group in _groupby_str_pair(
        selected, ["algorithm_a", "algorithm_b"]
    ):
        probabilities = np.asarray(
            group["probability_b_improves_over_a"],
            dtype=np.float64,
        )

        rows.append(
            {
                "algorithm_a": algorithm_a,
                "algorithm_b": algorithm_b,
                "n_tasks": int(len(group)),
                "mean_probability_b_improves_over_a": float(np.mean(probabilities)),
                "median_probability_b_improves_over_a": float(np.median(probabilities)),
            }
        )

    return pd.DataFrame(rows)


def _write_cross_task_core_outputs(
    output_dir: pathlib.Path,
    task_summary: pd.DataFrame,
    task_pairwise: pd.DataFrame,
    average_ranks: pd.DataFrame,
    win_rates: pd.DataFrame,
    probability_table: pd.DataFrame,
    superiority_table: pd.DataFrame,
    friedman: dict[str, float],
) -> None:
    task_summary.to_csv(output_dir / "per_task_algorithm_summary.csv", index=False)
    LOGGER.info("Wrote %s", output_dir / "per_task_algorithm_summary.csv")
    task_pairwise.to_csv(output_dir / "per_task_pairwise_comparisons.csv", index=False)
    LOGGER.info("Wrote %s", output_dir / "per_task_pairwise_comparisons.csv")
    average_ranks.to_csv(output_dir / "cross_task_average_ranks.csv", index=False)
    LOGGER.info("Wrote %s", output_dir / "cross_task_average_ranks.csv")
    win_rates.to_csv(output_dir / "cross_task_win_rates.csv", index=False)
    LOGGER.info("Wrote %s", output_dir / "cross_task_win_rates.csv")
    probability_table.to_csv(
        output_dir / "cross_task_probability_of_improvement.csv", index=False
    )
    LOGGER.info("Wrote %s", output_dir / "cross_task_probability_of_improvement.csv")
    superiority_table.to_csv(
        output_dir / "cross_task_superiority_scores.csv", index=False
    )
    LOGGER.info("Wrote %s", output_dir / "cross_task_superiority_scores.csv")
    pd.DataFrame([friedman]).to_csv(output_dir / "friedman_test.csv", index=False)
    LOGGER.info("Wrote %s", output_dir / "friedman_test.csv")


def _write_cross_task_publication_outputs(
    output_dir: pathlib.Path,
    average_ranks: pd.DataFrame,
    win_rates: pd.DataFrame,
    evaluation_metric: str,
    performance_metric: str,
    superiority_table: pd.DataFrame,
) -> None:
    publication_table = reporting.build_cross_task_publication_table(
        average_ranks,
        win_rates,
        superiority_table,
    )

    publication_table.to_csv(
        output_dir / "cross_task_publication_summary.csv",
        index=False,
    )
    LOGGER.info("Wrote %s", output_dir / "cross_task_publication_summary.csv")
    publication_table.to_latex(
        output_dir / "cross_task_publication_summary.tex",
        index=False,
        escape=False,
        caption=(
            f"Cross-task benchmark summary for {evaluation_metric} "
            f"using {performance_metric}."
        ),
        label=f"tab:{evaluation_metric}_{performance_metric}_cross_task",
    )
    LOGGER.info("Wrote %s", output_dir / "cross_task_publication_summary.tex")


def _write_cross_task_commentary(
    output_dir: pathlib.Path,
    average_ranks: pd.DataFrame,
    win_rates: pd.DataFrame,
    superiority_table: pd.DataFrame,
    friedman: dict[str, float],
) -> None:
    commentary = reporting.generate_cross_task_commentary(
        average_ranks,
        win_rates,
        superiority_table,
        friedman,
    )

    commentary_path = output_dir / "cross_task_commentary.md"
    commentary_path.write_text(commentary, encoding="utf-8")
    LOGGER.info("Wrote %s", commentary_path)


def run_cross_task_analysis(
    task_outputs: Mapping[str, pathlib.Path],
    output_dir: pathlib.Path,
    evaluation_metric: str,
    options: AnalysisOptions,
    performance_metric: str = "auc",
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting cross-task analysis.")
    LOGGER.info("Output directory: %s", output_dir)

    LOGGER.info("Stage 1/4: load per-task outputs")
    tasks = [
        _load_task_analysis(task_name, pathlib.Path(path))
        for task_name, path in task_outputs.items()
    ]

    LOGGER.info("Stage 2/4: combine and aggregate per-task results")
    task_summary = _combine_task_summaries(tasks)
    task_pairwise = _combine_task_pairwise(tasks)

    if task_summary.empty:
        error_message = (
            f"No per-task algorithm summaries were loaded for evaluation metric "
            f"{evaluation_metric!r} and performance metric {performance_metric!r}."
        )
        LOGGER.error(error_message)
        raise ValueError(error_message)

    if task_pairwise.empty:
        error_message = (
            f"No per-task pairwise comparisons were loaded for evaluation metric "
            f"{evaluation_metric!r} and performance metric {performance_metric!r}."
        )
        LOGGER.error(error_message)
        raise ValueError(error_message)

    average_ranks = _build_average_rank_table(
        task_summary,
        evaluation_metric=evaluation_metric,
        performance_metric=performance_metric,
    )
    win_rates = _build_pairwise_win_rate_table(
        task_pairwise,
        evaluation_metric=evaluation_metric,
        performance_metric=performance_metric,
    )
    probability_table = _build_cross_task_probability_table(
        task_pairwise,
        evaluation_metric=evaluation_metric,
        performance_metric=performance_metric,
    )
    superiority_table = _build_cross_task_superiority_table(
        task_pairwise=task_pairwise,
        evaluation_metric=evaluation_metric,
        performance_metric=performance_metric,
        options=options,
    )

    friedman = statistics.friedman_rank_test(
        task_summary,
        evaluation_metric=evaluation_metric,
        performance_metric=performance_metric,
    )

    LOGGER.info("Stage 3/4: write machine-readable outputs")
    _write_cross_task_core_outputs(
        output_dir,
        task_summary,
        task_pairwise,
        average_ranks,
        win_rates,
        probability_table,
        superiority_table,
        friedman,
    )

    LOGGER.info("Stage 4/4: write publication outputs and commentary")
    _write_cross_task_publication_outputs(
        output_dir,
        average_ranks,
        win_rates,
        evaluation_metric,
        performance_metric,
        superiority_table,
    )

    _write_cross_task_commentary(
        output_dir, average_ranks, win_rates, superiority_table, friedman
    )

    LOGGER.info("Cross-task analysis completed successfully.")

    return output_dir
