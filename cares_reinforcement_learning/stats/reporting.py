from __future__ import annotations

import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .validation import ValidationIssue


def filter_pairwise_against_baseline(
    pairwise: pd.DataFrame,
    baseline_algorithm: str,
) -> pd.DataFrame:
    mask = (pairwise["algorithm_a"] == baseline_algorithm) | (
        pairwise["algorithm_b"] == baseline_algorithm
    )

    return pairwise[mask].copy()


def classify_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "insufficient evidence"
    if p_value < 0.001:
        return "very strong statistical evidence"
    if p_value < 0.01:
        return "strong statistical evidence"
    if p_value < 0.05:
        return "moderate statistical evidence"
    return "no clear statistical evidence"


def classify_cliffs_delta(delta: float) -> str:
    abs_delta = abs(delta)
    if not np.isfinite(abs_delta):
        return "unknown effect size"
    if abs_delta < 0.147:
        return "negligible effect"
    if abs_delta < 0.33:
        return "small effect"
    if abs_delta < 0.474:
        return "medium effect"
    return "large effect"


def format_mean_std(mean: float, std: float, precision: int = 2) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{precision}f} $\\\\pm$ {std:.{precision}f}"


def format_ci(value: float, lower: float, upper: float, precision: int = 2) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{precision}f} [{lower:.{precision}f}, {upper:.{precision}f}]"


def _bold_best_value(
    values: pd.Series,
    direction: str,
) -> set[int]:
    numeric = np.asarray(values, dtype=np.float64)

    if direction == "higher":
        best = numeric.max()
    else:
        best = numeric.min()

    return set(values.index[numeric == best])


def build_publication_summary_table(
    algorithm_summary: pd.DataFrame,
    primary_metric: str,
    performance_metric: str = "auc",
    precision: int = 3,
) -> pd.DataFrame:
    selected = algorithm_summary[
        (algorithm_summary["evaluation_metric"] == primary_metric)
        & (algorithm_summary["performance_metric"] == performance_metric)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No summary results found for primary metric {primary_metric!r} "
            f"and performance metric {performance_metric!r}."
        )

    direction = str(selected["direction"].iloc[0])

    best_rows = _bold_best_value(selected["iqm"], direction)

    rows: list[dict[str, str]] = []

    for index, row in selected.iterrows():
        iqm_text = (
            f"{row['iqm']:.{precision}f} "
            f"[{row['iqm_ci_lower']:.{precision}f}, "
            f"{row['iqm_ci_upper']:.{precision}f}]"
        )

        if index in best_rows:
            iqm_text = f"\\textbf{{{iqm_text}}}"

        rows.append(
            {
                "Algorithm": str(row["algorithm"]),
                "Seeds": str(int(row["n_seeds"])),
                "Mean $\\pm$ Std": (
                    f"{row['mean']:.{precision}f} "
                    f"$\\pm$ "
                    f"{row['std']:.{precision}f}"
                ),
                "Median": f"{row['median']:.{precision}f}",
                "IQM [95\\% CI]": iqm_text,
            }
        )

    return pd.DataFrame(rows)


def build_cross_task_publication_table(
    average_ranks: pd.DataFrame,
    win_rates: pd.DataFrame,
    superiority_table: pd.DataFrame,
    precision: int = 3,
) -> pd.DataFrame:
    merged = average_ranks.merge(win_rates, on="algorithm", how="inner").merge(
        superiority_table, on="algorithm", how="inner"
    )

    if merged.empty:
        raise ValueError("No cross-task publication rows could be built.")

    best_rank = float(np.min(np.asarray(merged["average_rank"], dtype=np.float64)))
    best_superiority = float(
        np.max(np.asarray(merged["mean_superiority"], dtype=np.float64))
    )

    rows: list[dict[str, str]] = []

    for _, row in merged.sort_values("average_rank").iterrows():
        rank_text = f"{row['average_rank']:.{precision}f}"
        if np.isclose(float(row["average_rank"]), best_rank):
            rank_text = f"\\textbf{{{rank_text}}}"

        superiority_text = (
            f"{row['mean_superiority']:.{precision}f} "
            f"[{row['superiority_ci_lower']:.{precision}f}, "
            f"{row['superiority_ci_upper']:.{precision}f}]"
        )
        if np.isclose(float(row["mean_superiority"]), best_superiority):
            superiority_text = f"\\textbf{{{superiority_text}}}"

        rows.append(
            {
                "Algorithm": str(row["algorithm"]),
                "Average Rank": rank_text,
                "Mean Superiority [95\\% CI]": superiority_text,
                "Win Rate": f"{row['win_rate']:.{precision}f}",
                "Best-Rank Count": str(int(row["best_rank_count"])),
                "Tasks": str(int(row["n_tasks"])),
            }
        )

    return pd.DataFrame(rows)


def build_publication_pairwise_table(
    pairwise: pd.DataFrame,
    primary_metric: str,
    performance_metric: str = "auc",
    baseline_algorithm: str | None = None,
    precision: int = 3,
) -> pd.DataFrame:
    selected = pairwise[
        (pairwise["evaluation_metric"] == primary_metric)
        & (pairwise["performance_metric"] == performance_metric)
    ].copy()

    if baseline_algorithm is not None:
        selected = filter_pairwise_against_baseline(selected, baseline_algorithm)

    rows: list[dict[str, str]] = []
    for _, row in selected.iterrows():
        algorithm_a = row["algorithm_a"]
        algorithm_b = row["algorithm_b"]

        rows.append(
            {
                "Comparison": f"{algorithm_b} vs {algorithm_a}",
                "Mean A": f"{row['algorithm_a_mean']:.{precision}f}",
                "Mean B": f"{row['algorithm_b_mean']:.{precision}f}",
                "$\\Delta$ B-A": (f"{row['mean_difference_b_minus_a']:.{precision}f}"),
                "$P(B>A)$": (f"{row['probability_b_improves_over_a']:.{precision}f}"),
                "Cliff's $\\delta$": (f"{row['cliffs_delta_b_over_a']:.{precision}f}"),
                "$p_{Holm}$": f"{row['p_value_holm']:.{precision}f}",
            }
        )

    return pd.DataFrame(rows)


def save_latex_table(
    table: pd.DataFrame, output_path: pathlib.Path, caption: str, label: str
) -> None:
    latex = table.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
    )
    output_path.write_text(latex, encoding="utf-8")


def generate_result_commentary(
    pairwise: pd.DataFrame,
    primary_metric: str,
    performance_metric: str = "auc",
) -> str:
    selected = pairwise[
        (pairwise["evaluation_metric"] == primary_metric)
        & (pairwise["performance_metric"] == performance_metric)
    ].copy()

    if selected.empty:
        return "No pairwise comparison results were available for the selected metric."

    lines = [
        f"Primary evaluation metric: {primary_metric}.",
        f"Performance metric: {performance_metric}.",
        "",
    ]

    for _, row in selected.iterrows():
        algorithm_a = row["algorithm_a"]
        algorithm_b = row["algorithm_b"]

        p_value = float(row["p_value_holm"])
        delta = float(row["cliffs_delta_b_over_a"])
        probability = float(row["probability_b_improves_over_a"])
        mean_difference = float(row["mean_difference_b_minus_a"])
        direction = row["direction"]

        favours_b = (
            mean_difference > 0 if direction == "higher" else mean_difference < 0
        )
        favoured = algorithm_b if favours_b else algorithm_a

        lines.append(
            f"- {algorithm_b} vs {algorithm_a}: the observed mean difference favours "
            f"{favoured}. The Holm-corrected p-value is {p_value:.4f}, indicating "
            f"{classify_p_value(p_value)}. Cliff's delta B>A is {delta:.3f}, which is "
            f"classified as a {classify_cliffs_delta(delta)}. The estimated probability "
            f"that {algorithm_b} improves over {algorithm_a} is {probability:.3f}."
        )

    return "\n".join(lines)


def save_validation_report(
    issues: Sequence[ValidationIssue], output_path: pathlib.Path
) -> None:
    lines: list[str] = []
    if not issues:
        lines.append("No validation issues detected.")
    else:
        for issue in issues:
            lines.extend(
                [
                    f"[{issue.severity.upper()}] {issue.check_name}",
                    issue.message,
                    f"Recommendation: {issue.recommendation}",
                    (
                        f"Override flag: {issue.override_flag}"
                        if issue.override_flag
                        else ""
                    ),
                    "",
                ]
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_cross_task_commentary(
    average_ranks: pd.DataFrame,
    win_rates: pd.DataFrame,
    superiority_table: pd.DataFrame,
    friedman: dict[str, float],
) -> str:
    ranked = average_ranks.sort_values("average_rank")
    best_rank_row = ranked.iloc[0]

    superior = superiority_table.sort_values(
        "mean_superiority",
        ascending=False,
    )
    best_superiority_row = superior.iloc[0]

    lines = [
        "Cross-task summary.",
        "",
        (
            f"- {best_rank_row['algorithm']} achieved the best average rank "
            f"({best_rank_row['average_rank']:.3f}) across "
            f"{int(best_rank_row['n_tasks'])} tasks."
        ),
        (
            f"- {best_superiority_row['algorithm']} achieved the highest mean "
            f"superiority score ({best_superiority_row['mean_superiority']:.3f} "
            f"[{best_superiority_row['superiority_ci_lower']:.3f}, "
            f"{best_superiority_row['superiority_ci_upper']:.3f}])."
        ),
    ]

    best_win_row = win_rates.sort_values("win_rate", ascending=False).iloc[0]
    lines.append(
        f"- {best_win_row['algorithm']} achieved the highest pairwise win rate "
        f"({best_win_row['win_rate']:.3f}; {int(best_win_row['wins'])} wins, "
        f"{int(best_win_row['losses'])} losses, {int(best_win_row['ties'])} ties)."
    )

    p_value = float(friedman["p_value"])
    if np.isfinite(p_value):
        if p_value < 0.05:
            lines.append(
                f"- The Friedman test indicates statistically detectable differences "
                f"between algorithm ranks across tasks (p={p_value:.4f})."
            )
        else:
            lines.append(
                f"- The Friedman test does not indicate clear statistical differences "
                f"between algorithm ranks across tasks (p={p_value:.4f})."
            )

    return "\n".join(lines)
