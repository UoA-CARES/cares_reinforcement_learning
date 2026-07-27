from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP_COLUMNS = ["evaluation_metric", "performance_metric"]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "metric"


def _save(fig: plt.Figure, path: pathlib.Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_dominance(
    pairwise: pd.DataFrame, output: pathlib.Path, dpi: int
) -> None:
    algorithms = sorted(set(pairwise["algorithm_a"]).union(pairwise["algorithm_b"]))
    matrix = np.full((len(algorithms), len(algorithms)), 0.5, dtype=float)
    index = {algorithm: position for position, algorithm in enumerate(algorithms)}
    for row in pairwise.itertuples(index=False):
        a, b = index[row.algorithm_a], index[row.algorithm_b]
        matrix[a, b] = float(row.mean_probability_a_better)
        matrix[b, a] = 1.0 - float(row.mean_probability_a_better)
    fig, ax = plt.subplots(
        figsize=(max(5.5, 0.85 * len(algorithms)), max(4.8, 0.75 * len(algorithms)))
    )
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(algorithms)), algorithms, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(algorithms)), algorithms)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Algorithm")
    ax.set_title("Mean probability of improvement across tasks")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="P(row algorithm better)")
    _save(fig, output / "pairwise_dominance.png", dpi)


def plot_critical_difference(
    benchmark: pd.DataFrame, critical_difference: float, output: pathlib.Path, dpi: int
) -> None:
    ordered = benchmark.sort_values("average_rank")
    ranks = ordered["average_rank"].to_numpy(float)
    algorithms = ordered["algorithm"].tolist()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.05 * len(algorithms)), 3.8))
    ax.scatter(ranks, np.zeros_like(ranks), zorder=3)
    for rank, algorithm in zip(ranks, algorithms, strict=True):
        ax.plot([rank, rank], [0.0, 0.2], linewidth=1.0)
        ax.text(rank, 0.24, algorithm, rotation=45, ha="left", va="bottom")
    level = -0.15
    for start in range(len(algorithms)):
        end = start
        while (
            end + 1 < len(algorithms)
            and ranks[end + 1] - ranks[start] <= critical_difference
        ):
            end += 1
        if end > start:
            ax.plot([ranks[start], ranks[end]], [level, level], linewidth=3.0)
            level -= 0.08
    ax.set_xlim(
        min(1.0, float(ranks.min()) - 0.25),
        max(float(len(algorithms)), float(ranks.max()) + 0.25),
    )
    ax.set_ylim(level - 0.08, 0.9)
    ax.set_yticks([])
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title(
        f"Nemenyi critical-difference diagram (CD = {critical_difference:.3f})"
    )
    ax.grid(axis="x", alpha=0.25)
    _save(fig, output / "critical_difference.png", dpi)


def write_cross_task_figures(
    output: pathlib.Path,
    summaries: pd.DataFrame,
    benchmark: pd.DataFrame,
    pairwise: pd.DataFrame,
    critical_differences: dict[tuple[str, str], float],
    dpi: int,
) -> None:
    figures_root = output / "figures"
    for (evaluation_metric, performance_metric), summary_group in summaries.groupby(
        GROUP_COLUMNS, sort=False
    ):
        group_output = (
            figures_root / f"{_slug(evaluation_metric)}__{_slug(performance_metric)}"
        )
        group_output.mkdir(parents=True, exist_ok=True)
        benchmark_group = benchmark[
            (benchmark["evaluation_metric"] == evaluation_metric)
            & (benchmark["performance_metric"] == performance_metric)
        ]
        pairwise_group = pairwise[
            (pairwise["evaluation_metric"] == evaluation_metric)
            & (pairwise["performance_metric"] == performance_metric)
        ]
        plot_pairwise_dominance(pairwise_group, group_output, dpi)
        cd = critical_differences.get((evaluation_metric, performance_metric))
        if cd is not None:
            plot_critical_difference(benchmark_group, cd, group_output, dpi)
