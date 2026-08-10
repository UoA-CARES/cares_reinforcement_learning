from __future__ import annotations

import datetime as dt
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from cares_reinforcement_learning.reporting.analysis import statistics
from cares_reinforcement_learning.reporting.analysis.models import (
    AnalysisOptions,
    BenchmarkAnalysisResult,
    TaskAnalysisResult,
)


def _latex(frame: pd.DataFrame, path: pathlib.Path, caption: str, label: str) -> None:
    path.write_text(
        frame.to_latex(index=False, escape=True, caption=caption, label=label),
        encoding="utf-8",
    )


def _available_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _prepare_task_publication_frames(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    task_summary: pd.DataFrame,
    options: AnalysisOptions,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = options.primary_performance_metric
    performance = summary[summary["performance_metric"] == primary].copy()
    performance = performance.sort_values(["evaluation_metric", "rank"])[
        _available_columns(
            performance,
            [
                "algorithm",
                "evaluation_metric",
                "iqm",
                "iqm_ci_low",
                "iqm_ci_high",
                "mean",
                "std",
                "minimum",
                "maximum",
                "n_seeds",
            ],
        )
    ]

    significance_column = (
        "significant_holm"
        if "significant_holm" in pairwise.columns
        else "significant_holm_0_05"
    )
    comparison_columns = _available_columns(
        pairwise,
        [
            "evaluation_metric",
            "algorithm_a",
            "algorithm_b",
            "comparison_design",
            "test",
            "n_a",
            "n_b",
            "test_statistic",
            "p_value",
            "p_value_holm",
            significance_column,
            "probability_b_better",
            "cliffs_delta_b_vs_a",
        ],
    )
    comparisons = pairwise[pairwise["performance_metric"] == primary].copy()[
        comparison_columns
    ]

    overview = (
        task_summary[task_summary["performance_metric"] == primary]
        .copy()[
            _available_columns(
                task_summary,
                [
                    "algorithm",
                    "evaluation_metric",
                    "iqm_rank",
                    "task_superiority",
                    "opponents",
                ],
            )
        ]
        .sort_values(["evaluation_metric", "iqm_rank"])
    )

    return performance, comparisons, overview


def _prepare_cross_task_publication_frames(
    benchmark: pd.DataFrame,
    pairwise: pd.DataFrame,
    options: AnalysisOptions,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = options.primary_performance_metric
    publication_columns = _available_columns(
        benchmark,
        [
            "algorithm",
            "evaluation_metric",
            "mean_superiority",
            "superiority_ci_low",
            "superiority_ci_high",
            "average_rank",
            "average_rank_ci_low",
            "average_rank_ci_high",
            "median_rank",
            "rank_std",
            "rank_iqr",
            "top_1_count",
            "top_1_rate",
            "top_2_count",
            "top_2_rate",
            "top_3_count",
            "top_3_rate",
            "n_tasks",
        ],
    )
    benchmark_publication = benchmark[
        benchmark["performance_metric"] == primary
    ].copy()[publication_columns]
    benchmark_publication = benchmark_publication.sort_values(
        ["evaluation_metric", "mean_superiority", "average_rank"],
        ascending=[True, False, True],
    )

    pairwise_columns = _available_columns(
        pairwise,
        [
            "evaluation_metric",
            "algorithm_a",
            "algorithm_b",
            "wins_a",
            "ties",
            "wins_b",
            "win_rate_a",
            "mean_probability_a_better",
            "probability_a_better_ci_low",
            "probability_a_better_ci_high",
            "mean_rank_difference_a_minus_b",
            "n_tasks",
        ],
    )
    pairwise_publication = pairwise[pairwise["performance_metric"] == primary].copy()[
        pairwise_columns
    ]
    pairwise_publication = pairwise_publication.sort_values(
        ["evaluation_metric", "algorithm_a", "algorithm_b"]
    )

    return benchmark_publication, pairwise_publication


def write_task_outputs(
    output: pathlib.Path,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    task_summary: pd.DataFrame,
    options: AnalysisOptions,
    comparison_design: str,
    warnings: Sequence[str],
) -> None:
    performance, comparisons, overview = _prepare_task_publication_frames(
        summary,
        pairwise,
        task_summary,
        options,
    )
    performance.to_csv(output / "task_performance.csv", index=False)
    _latex(
        performance,
        output / "task_performance.tex",
        "Task performance using observed IQM and BCa bootstrap confidence intervals.",
        "tab:task_performance",
    )

    comparisons.to_csv(output / "pairwise_statistics.csv", index=False)
    _latex(
        comparisons,
        output / "pairwise_statistics.tex",
        "Pairwise evidence from observed seed-level performance metrics.",
        "tab:pairwise_statistics",
    )

    overview.to_csv(output / "task_summary.csv", index=False)
    _latex(
        overview,
        output / "task_summary.tex",
        "Supporting task-level rank and superiority summary.",
        "tab:task_summary",
    )

    lines = [
        f"Comparison design: {comparison_design}.",
        f"Primary performance metric: raw trapezoidal {options.primary_performance_metric}.",
        (
            f"Algorithm IQM intervals: {options.bootstrap_confidence:.0%} BCa "
            f"bootstrap, {options.bootstrap_samples:,} resamples, "
            "resampling unit = seed."
        ),
        "Every evaluation metric is analysed independently.",
        (
            "Algorithms are compared only on the same evaluation metric and the "
            "same performance summary; different metrics are never compared against "
            "each other."
        ),
        (
            "Pairwise tests, probability of improvement, and Cliff's delta use "
            "observed seed metrics only."
        ),
        (
            "Task superiority is the mean probability of improvement against all "
            "opponents within that metric group."
        ),
        (
            "Rank is calculated from observed IQM within that metric group and is a "
            "supporting ordinal summary."
        ),
        (
            f"Holm-adjusted significance flags use alpha = "
            f"{options.significance_level:g}."
        ),
    ]
    lines.extend(f"Warning: {warning}" for warning in warnings)
    (output / "methodology.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def write_cross_task_outputs(
    output: pathlib.Path,
    benchmark: pd.DataFrame,
    pairwise: pd.DataFrame,
    friedman: pd.DataFrame,
    nemenyi: pd.DataFrame,
    options: AnalysisOptions,
) -> None:
    publication, pairwise_publication = _prepare_cross_task_publication_frames(
        benchmark,
        pairwise,
        options,
    )
    publication.to_csv(output / "benchmark_performance.csv", index=False)
    _latex(
        publication,
        output / "benchmark_performance.tex",
        (
            "Cross-task superiority, rank consistency, and top-k task performance, "
            "reported separately by evaluation metric."
        ),
        "tab:benchmark_performance",
    )

    # Publication pairwise output deliberately prioritises probability of
    # improvement. Rank-based Wilcoxon evidence remains available in the raw
    # cross_task_pairwise.csv written by cross_task_analysis.py.
    pairwise_publication.to_csv(
        output / "cross_task_pairwise_publication.csv", index=False
    )
    _latex(
        pairwise_publication,
        output / "cross_task_pairwise_publication.tex",
        (
            "Primary pairwise cross-task evidence based on probability of "
            "improvement, reported separately by evaluation metric."
        ),
        "tab:cross_task_pairwise",
    )

    friedman.to_csv(output / "friedman.csv", index=False)
    _latex(
        friedman,
        output / "friedman.tex",
        (
            "Supplementary Friedman omnibus tests over task-level ranks, conducted "
            "separately for every metric group."
        ),
        "tab:friedman",
    )
    if not nemenyi.empty:
        nemenyi.to_csv(output / "nemenyi.csv", index=False)
        _latex(
            nemenyi,
            output / "nemenyi.tex",
            (
                "Supplementary Nemenyi post-hoc comparisons following significant "
                "Friedman tests."
            ),
            "tab:nemenyi",
        )

    lines = [
        "Every evaluation metric and performance summary is analysed as an independent group.",
        "No comparisons are made between different evaluation metrics or between different performance summaries.",
        "No metric magnitudes are pooled or averaged across tasks.",
        "Every benchmark task receives equal weight in cross-task summaries.",
        (
            f"Primary cross-task probability-of-improvement intervals: "
            f"{options.bootstrap_confidence:.0%} stratified percentile bootstrap, "
            f"{options.bootstrap_samples:,} resamples, with seeds resampled within "
            "each fixed benchmark task."
        ),
        (
            "The fixed-task stratified bootstrap propagates run-to-run uncertainty "
            "without treating the deliberately selected benchmark tasks as a random "
            "sample from a larger task population."
        ),
        (
            "Pairwise mean probability of improvement is the primary evidence for a "
            "specific algorithm comparison."
        ),
        (
            "Mean superiority is a roster-level summary: the algorithm's mean "
            "probability of improvement against all opponents across the benchmark."
        ),
        (
            "Probability of improvement measures distributional overlap and "
            "consistency, not the numerical magnitude of an improvement."
        ),
        (
            "Average rank, rank dispersion, and top-k rates are descriptive "
            "cross-task consistency summaries."
        ),
        (
            "Friedman and Nemenyi analyses are supplementary rank-based consistency "
            "checks rather than the primary cross-task evidence."
        ),
        (
            "The raw cross_task_pairwise.csv retains exploratory rank-test output, "
            "but those p-values are excluded from publication-focused pairwise tables."
        ),
        (
            f"Holm correction uses alpha = {options.significance_level:g} within each "
            "evaluation-metric/performance-metric family where applicable."
        ),
        (
            "Nemenyi comparisons and critical-difference figures are produced only "
            "after a significant Friedman test for the same metric group."
        ),
    ]
    (output / "methodology.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------


def _display_name(value: object) -> str:
    return str(value).replace("_", " ").strip().title()


def _format_value(value: Any, column: str) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if "p_value" in column:
            return "<0.001" if 0.0 <= value < 0.001 else f"{value:.3f}"
        if (
            column.endswith("_rate")
            or "probability" in column
            or "superiority" in column
        ):
            return f"{value:.3f}"
        if "rank" in column or "delta" in column or "iqm" in column or "mean" in column:
            return f"{value:.3f}"
        return f"{value:.3g}"
    return str(value)


def _best_algorithm_sentences(benchmark: pd.DataFrame) -> list[str]:
    sentences: list[str] = []
    if benchmark.empty:
        return sentences
    for metric, group in benchmark.groupby("evaluation_metric", sort=False):
        ordered = group.sort_values(
            ["mean_superiority", "average_rank"], ascending=[False, True]
        )
        best = ordered.iloc[0]
        algorithm = str(best["algorithm"])
        metric_name = _display_name(metric)
        superiority = float(best["mean_superiority"])
        low = float(best["superiority_ci_low"])
        high = float(best["superiority_ci_high"])
        rank = float(best["average_rank"])
        top_count = int(best.get("top_1_count", 0))
        tasks = int(best.get("n_tasks", 0))
        sentences.append(
            f"For {metric_name}, {algorithm} had the highest mean superiority "
            f"({superiority:.3f}, {low:.3f}-{high:.3f}) and an average rank of "
            f"{rank:.2f}; it ranked first on {top_count} of {tasks} tasks."
        )
    return sentences


def _strong_pairwise_sentences(pairwise: pd.DataFrame) -> list[str]:
    sentences: list[str] = []
    if pairwise.empty:
        return sentences
    for metric, group in pairwise.groupby("evaluation_metric", sort=False):
        candidates = group.copy()
        candidates["distance"] = (candidates["mean_probability_a_better"] - 0.5).abs()
        row = candidates.sort_values("distance", ascending=False).iloc[0]
        probability = float(row["mean_probability_a_better"])
        low = float(row["probability_a_better_ci_low"])
        high = float(row["probability_a_better_ci_high"])
        a = str(row["algorithm_a"])
        b = str(row["algorithm_b"])
        if probability >= 0.5:
            winner, loser, shown, shown_low, shown_high = a, b, probability, low, high
        else:
            winner, loser = b, a
            shown, shown_low, shown_high = 1.0 - probability, 1.0 - high, 1.0 - low
        sentences.append(
            f"The strongest pairwise separation for {_display_name(metric)} was "
            f"{winner} over {loser}: mean probability of improvement {shown:.3f} "
            f"({shown_low:.3f}-{shown_high:.3f})."
        )
    return sentences


def _benchmark_summary_table(
    group: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare the compact, reviewer-facing benchmark summary table."""
    ordered = group.sort_values(
        ["mean_superiority", "average_rank"], ascending=[False, True]
    ).reset_index(drop=True)
    ordered.insert(0, "benchmark_rank", range(1, len(ordered) + 1))
    ordered["mean_superiority_ci"] = ordered.apply(
        lambda row: (
            f"{float(row['mean_superiority']):.3f} "
            f"[{float(row['superiority_ci_low']):.3f}, "
            f"{float(row['superiority_ci_high']):.3f}]"
        ),
        axis=1,
    )
    ordered["top_1"] = ordered.apply(
        lambda row: f"{int(row['top_1_count'])}/{int(row['n_tasks'])}", axis=1
    )

    columns = [
        "benchmark_rank",
        "algorithm",
        "mean_superiority_ci",
        "average_rank",
        "rank_std",
        "top_1",
    ]
    headings = [
        "Rank",
        "Algorithm",
        "Mean superiority [95% CI]",
        "Avg. rank",
        "Rank SD",
        "Top-1",
    ]

    n_algorithms = len(ordered)
    if n_algorithms > 2 and "top_2_count" in ordered.columns:
        ordered["top_2"] = ordered.apply(
            lambda row: f"{int(row['top_2_count'])}/{int(row['n_tasks'])}", axis=1
        )
        columns.append("top_2")
        headings.append("Top-2")
    if n_algorithms > 3 and "top_3_count" in ordered.columns:
        ordered["top_3"] = ordered.apply(
            lambda row: f"{int(row['top_3_count'])}/{int(row['n_tasks'])}", axis=1
        )
        columns.append("top_3")
        headings.append("Top-3")

    return ordered, columns, headings


def _reference_comparison(
    pairwise: pd.DataFrame,
    reference_name: str,
) -> pd.DataFrame:
    """Reorient cross-task pairwise rows so the reference comparison is always focal."""
    rows: list[dict[str, object]] = []
    for _, row in pairwise.iterrows():
        algorithm_a = str(row["algorithm_a"])
        algorithm_b = str(row["algorithm_b"])
        if reference_name not in {algorithm_a, algorithm_b}:
            continue

        reference_is_a = algorithm_a == reference_name
        opponent = algorithm_b if reference_is_a else algorithm_a
        probability, low, high = statistics.orient_probability_interval(
            float(row["mean_probability_a_better"]),
            float(row["probability_a_better_ci_low"]),
            float(row["probability_a_better_ci_high"]),
            reference_is_a=reference_is_a,
        )
        wins = int(row.get("wins_a" if reference_is_a else "wins_b", 0))
        losses = int(row.get("wins_b" if reference_is_a else "wins_a", 0))

        rows.append(
            {
                "evaluation_metric": row["evaluation_metric"],
                "reference_comparison": reference_name,
                "comparator": opponent,
                "probability_reference_better": probability,
                "probability_reference_better_ci_low": low,
                "probability_reference_better_ci_high": high,
                "tasks_won": wins,
                "tasks_tied": int(row.get("ties", 0)),
                "tasks_lost": losses,
                "n_tasks": int(row.get("n_tasks", wins + losses)),
                "ci_supports_advantage": low > 0.5,
                "ci_supports_disadvantage": high < 0.5,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["evaluation_metric", "probability_reference_better"],
        ascending=[True, False],
    ).reset_index(drop=True)


def _reference_comparison_sentences(
    benchmark: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    reference_name: str,
) -> list[str]:
    sentences: list[str] = []
    for metric, group in benchmark.groupby("evaluation_metric", sort=False):
        ordered = group.sort_values(
            ["mean_superiority", "average_rank"], ascending=[False, True]
        ).reset_index(drop=True)
        match = ordered[ordered["algorithm"] == reference_name]
        if match.empty:
            continue
        row = match.iloc[0]
        position = int(match.index[0]) + 1
        sentences.append(
            f"For {_display_name(metric)}, {reference_name} ranked {position} of "
            f"{len(ordered)} algorithms by mean superiority "
            f"({float(row['mean_superiority']):.3f}, "
            f"{float(row['superiority_ci_low']):.3f}-"
            f"{float(row['superiority_ci_high']):.3f}), with an average rank of "
            f"{float(row['average_rank']):.2f} and {int(row.get('top_1_count', 0))} "
            f"first-place finishes across {int(row.get('n_tasks', 0))} tasks."
        )

    for metric, group in comparison_frame.groupby("evaluation_metric", sort=False):
        supported_wins = int(group["ci_supports_advantage"].sum())
        supported_losses = int(group["ci_supports_disadvantage"].sum())
        sentences.append(
            f"For {_display_name(metric)}, {reference_name}'s probability-of-"
            f"improvement interval favoured it over {supported_wins} comparator(s) "
            f"and favoured a comparator over it in {supported_losses} comparison(s)."
        )
    return sentences


def _write_reference_comparison_outputs(
    benchmark_output: pathlib.Path,
    comparison_frame: pd.DataFrame,
    reference_name: str,
) -> None:
    if comparison_frame.empty:
        return
    csv_path = benchmark_output / "reference_comparison.csv"
    tex_path = benchmark_output / "reference_comparison.tex"
    comparison_frame.to_csv(csv_path, index=False)
    _latex(
        comparison_frame,
        tex_path,
        f"Cross-task comparison of {reference_name} against each baseline.",
        "tab:reference_comparison",
    )


def _build_pdf_report(
    output_path: pathlib.Path,
    task_results: Mapping[str, TaskAnalysisResult],
    benchmark_result: BenchmarkAnalysisResult | None,
    benchmark_output: pathlib.Path | None,
    options: AnalysisOptions,
    reference_comparison: str | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=23,
            leading=28,
            spaceAfter=14,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor("#1F2937"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubsectionTitle",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            spaceBefore=8,
            spaceAfter=5,
            textColor=colors.HexColor("#374151"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#4B5563"),
            spaceBefore=3,
            spaceAfter=8,
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="CARES RL Statistical Benchmark Report",
        author="CARES RL Statistical Tool",
    )
    page_width = A4[0] - doc.leftMargin - doc.rightMargin

    def paragraph(text: str, style: str = "BodyText") -> Paragraph:
        return Paragraph(text.replace("&", "&amp;"), styles[style])

    def frame_table(
        frame: pd.DataFrame,
        columns: Sequence[str],
        headings: Sequence[str] | None = None,
        max_rows: int | None = None,
        font_size: float = 7.0,
    ) -> Table:
        selected = frame[_available_columns(frame, columns)].copy()
        if max_rows is not None:
            selected = selected.head(max_rows)
        labels = (
            list(headings)
            if headings is not None
            else [_display_name(column) for column in selected.columns]
        )
        data: list[list[Any]] = [
            [Paragraph(str(label), styles["SmallBody"]) for label in labels]
        ]
        for _, row in selected.iterrows():
            data.append(
                [
                    Paragraph(_format_value(row[column], column), styles["SmallBody"])
                    for column in selected.columns
                ]
            )
        if len(data) == 1:
            data.append(
                [Paragraph("No results available.", styles["SmallBody"])]
                + [""] * (len(labels) - 1)
            )
        widths = [page_width / max(1, len(labels))] * max(1, len(labels))
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), font_size),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9CA3AF")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F9FAFB")],
                    ),
                ]
            )
        )
        return table

    benchmark = pd.DataFrame()
    pairwise = pd.DataFrame()
    friedman = pd.DataFrame()
    nemenyi = pd.DataFrame()
    if benchmark_result is not None:
        benchmark, pairwise = _prepare_cross_task_publication_frames(
            benchmark_result.benchmark_summary,
            benchmark_result.cross_task_pairwise,
            options,
        )
        friedman = benchmark_result.friedman_tests
        nemenyi = benchmark_result.nemenyi_posthoc

    reference_comparison_frame = pd.DataFrame()
    if reference_comparison is not None:
        if benchmark.empty:
            raise ValueError(
                "--reference-comparison requires a multi-task benchmark report."
            )

        available_comparisons = set(benchmark["algorithm"].astype(str))
        if reference_comparison not in available_comparisons:
            available = ", ".join(sorted(available_comparisons))
            raise ValueError(
                f"Reference comparison {reference_comparison!r} was not found. "
                f"Available comparisons: {available}."
            )

        reference_comparison_frame = _reference_comparison(
            pairwise,
            reference_comparison,
        )
        if benchmark_output is not None:
            _write_reference_comparison_outputs(
                benchmark_output,
                reference_comparison_frame,
                reference_comparison,
            )

    story: list[Any] = []
    story.extend(
        [
            Spacer(1, 25 * mm),
            Paragraph("CARES RL Statistical Benchmark Report", styles["ReportTitle"]),
            paragraph(
                f"Generated {dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
                "Caption",
            ),
            Spacer(1, 9 * mm),
            frame_table(
                pd.DataFrame(
                    [
                        {
                            "tasks": len(task_results),
                            "primary_metric": options.primary_performance_metric,
                            "confidence": options.bootstrap_confidence,
                            "bootstrap_samples": options.bootstrap_samples,
                            "significance_level": options.significance_level,
                            "reference_comparison": reference_comparison
                            or "Not specified",
                        }
                    ]
                ),
                [
                    "tasks",
                    "primary_metric",
                    "confidence",
                    "bootstrap_samples",
                    "significance_level",
                    "reference_comparison",
                ],
                [
                    "Tasks",
                    "Primary summary",
                    "CI level",
                    "Bootstrap replicates",
                    "Alpha",
                    "Reference comparison",
                ],
                font_size=8,
            ),
            Spacer(1, 10 * mm),
            paragraph(
                "This report is a guided summary of the generated statistical outputs. "
                "All raw CSV files remain unchanged in the task and benchmark output "
                "directories and should be retained for reproducibility.",
                "BodyText",
            ),
            PageBreak(),
        ]
    )

    story.append(Paragraph("1. Executive Summary", styles["SectionTitle"]))
    if benchmark.empty:
        story.append(
            paragraph(
                "A single task was analysed, so no cross-task benchmark summary was produced. "
                "The task-level results are presented below."
            )
        )
    else:
        for sentence in _best_algorithm_sentences(benchmark):
            story.append(paragraph(f"- {sentence}"))
        for sentence in _strong_pairwise_sentences(pairwise):
            story.append(paragraph(f"- {sentence}"))
        if reference_comparison is not None:
            for sentence in _reference_comparison_sentences(
                benchmark, reference_comparison_frame, reference_comparison
            ):
                story.append(paragraph(f"- {sentence}"))
        story.append(
            paragraph(
                "Primary cross-task conclusions should be based on pairwise probability "
                "of improvement and its confidence interval. Mean superiority summarises "
                "performance against the full algorithm roster. Rank statistics and "
                "Friedman/Nemenyi results are supporting consistency checks."
            )
        )

    if not benchmark.empty:
        story.extend(
            [PageBreak(), Paragraph("2. Benchmark Overview", styles["SectionTitle"])]
        )
        story.append(
            paragraph(
                "Mean superiority is the roster-level probability summary. Average rank, "
                "rank spread, and Top-1 counts describe consistency, but do not quantify "
                "the size of performance differences."
            )
        )
        for metric, group in benchmark.groupby("evaluation_metric", sort=False):
            story.append(Paragraph(_display_name(metric), styles["SubsectionTitle"]))
            table_group, table_columns, table_headings = _benchmark_summary_table(group)
            story.append(
                frame_table(
                    table_group,
                    table_columns,
                    table_headings,
                )
            )

        next_section = 3
        if reference_comparison is not None:
            story.extend(
                [
                    PageBreak(),
                    Paragraph(
                        f"3. Reference Comparison: {reference_comparison}",
                        styles["SectionTitle"],
                    ),
                ]
            )
            story.append(
                paragraph(
                    f"This section reports {reference_comparison} against every comparator. "
                    "Probability of improvement is oriented so values above 0.5 always "
                    f"favour {reference_comparison}. A confidence interval entirely above "
                    "0.5 provides interval-based evidence favouring the reference comparison; "
                    "an interval entirely below 0.5 favours the comparator."
                )
            )
            for sentence in _reference_comparison_sentences(
                benchmark, reference_comparison_frame, reference_comparison
            ):
                story.append(paragraph(sentence))
            for metric, group in reference_comparison_frame.groupby(
                "evaluation_metric", sort=False
            ):
                story.append(
                    Paragraph(_display_name(metric), styles["SubsectionTitle"])
                )
                display_group = group.copy()
                display_group["task_record"] = display_group.apply(
                    lambda row: (
                        f"{int(row['tasks_won'])}-{int(row['tasks_tied'])}-"
                        f"{int(row['tasks_lost'])}"
                    ),
                    axis=1,
                )
                story.append(
                    frame_table(
                        display_group,
                        [
                            "comparator",
                            "probability_reference_better",
                            "probability_reference_better_ci_low",
                            "probability_reference_better_ci_high",
                            "task_record",
                            "ci_supports_advantage",
                        ],
                        [
                            "Comparator",
                            f"P({reference_comparison} better)",
                            "CI low",
                            "CI high",
                            "Tasks W-T-L",
                            "CI favours reference",
                        ],
                    )
                )
            next_section = 4

        story.extend(
            [
                PageBreak(),
                Paragraph(
                    f"{next_section}. Pairwise Probability of Improvement",
                    styles["SectionTitle"],
                ),
            ]
        )
        story.append(
            paragraph(
                "This is the primary evidence for claims that one specific algorithm "
                "outperforms another across the fixed benchmark. A value above 0.5 favours "
                "Algorithm A; a value below 0.5 favours Algorithm B. The statistic reflects "
                "distributional overlap and consistency, not the numerical magnitude of the gap."
            )
        )
        for metric, group in pairwise.groupby("evaluation_metric", sort=False):
            story.append(Paragraph(_display_name(metric), styles["SubsectionTitle"]))
            story.append(
                frame_table(
                    group,
                    [
                        "algorithm_a",
                        "algorithm_b",
                        "mean_probability_a_better",
                        "probability_a_better_ci_low",
                        "probability_a_better_ci_high",
                        "wins_a",
                        "ties",
                        "wins_b",
                        "n_tasks",
                    ],
                    [
                        "Algorithm A",
                        "Algorithm B",
                        "P(A better)",
                        "CI low",
                        "CI high",
                        "A wins",
                        "Ties",
                        "B wins",
                        "Tasks",
                    ],
                )
            )
            if benchmark_output is not None:
                figure = (
                    benchmark_output
                    / "figures"
                    / f"{str(metric).replace(' ', '_')}__{options.primary_performance_metric}"
                    / "pairwise_dominance.png"
                )
                if figure.exists():
                    image = Image(str(figure))
                    image._restrictSize(page_width, 115 * mm)
                    story.extend(
                        [
                            Spacer(1, 3 * mm),
                            image,
                            paragraph(
                                "Pairwise probability-of-improvement matrix. Rows are the "
                                "candidate algorithms and columns are their opponents.",
                                "Caption",
                            ),
                        ]
                    )

    section_number = (
        (5 if reference_comparison is not None else 4) if not benchmark.empty else 2
    )
    story.extend(
        [
            PageBreak(),
            Paragraph(f"{section_number}. Per-Task Results", styles["SectionTitle"]),
        ]
    )
    story.append(
        paragraph(
            "Each task is reported independently. IQM and its BCa confidence interval "
            "are the primary task-level performance summary. Pairwise probability of "
            "improvement and Cliff's delta provide complementary distributional and "
            "effect-size evidence."
        )
    )

    for task_name, task_result in task_results.items():
        performance, task_pairwise, _ = _prepare_task_publication_frames(
            task_result.algorithm_summary,
            task_result.pairwise,
            task_result.task_summary,
            options,
        )
        story.append(Paragraph(task_name, styles["SubsectionTitle"]))
        if not performance.empty:
            for metric, group in performance.groupby("evaluation_metric", sort=False):
                story.append(paragraph(f"<b>{_display_name(metric)}</b>", "SmallBody"))
                story.append(
                    frame_table(
                        group.sort_values("iqm", ascending=False),
                        [
                            "algorithm",
                            "iqm",
                            "iqm_ci_low",
                            "iqm_ci_high",
                            "mean",
                            "std",
                            "n_seeds",
                        ],
                        [
                            "Algorithm",
                            "IQM",
                            "CI low",
                            "CI high",
                            "Mean",
                            "SD",
                            "Seeds",
                        ],
                    )
                )
        if not task_pairwise.empty:
            story.append(paragraph("Pairwise task evidence", "SmallBody"))
            story.append(
                frame_table(
                    task_pairwise,
                    [
                        "evaluation_metric",
                        "algorithm_a",
                        "algorithm_b",
                        "probability_b_better",
                        "cliffs_delta_b_vs_a",
                        "p_value_holm",
                    ],
                    [
                        "Metric",
                        "Algorithm A",
                        "Algorithm B",
                        "P(B better)",
                        "Cliff's delta",
                        "Holm p",
                    ],
                    max_rows=30,
                )
            )
        story.append(Spacer(1, 5 * mm))

    if not benchmark.empty:
        story.extend(
            [
                PageBreak(),
                Paragraph(
                    f"{6 if reference_comparison is not None else 5}. Supplementary Rank-Based Tests",
                    styles["SectionTitle"],
                ),
            ]
        )
        story.append(
            paragraph(
                "Friedman and Nemenyi analyses are supplementary consistency checks. "
                "They reduce task performance to ordinal ranks and therefore should not "
                "override the probability-of-improvement evidence presented earlier."
            )
        )
        if friedman.empty:
            story.append(paragraph("No Friedman results were available."))
        else:
            story.append(
                frame_table(
                    friedman,
                    _available_columns(
                        friedman,
                        [
                            "evaluation_metric",
                            "performance_metric",
                            "n_tasks",
                            "n_algorithms",
                            "test_statistic",
                            "p_value",
                            "significant",
                        ],
                    ),
                )
            )
        if not nemenyi.empty:
            story.append(
                Paragraph("Nemenyi post-hoc results", styles["SubsectionTitle"])
            )
            story.append(
                frame_table(
                    nemenyi,
                    _available_columns(
                        nemenyi,
                        [
                            "evaluation_metric",
                            "performance_metric",
                            "algorithm_a",
                            "algorithm_b",
                            "absolute_rank_difference",
                            "critical_difference",
                            "significant",
                        ],
                    ),
                    max_rows=50,
                )
            )

        story.extend(
            [
                PageBreak(),
                Paragraph(
                    f"{7 if reference_comparison is not None else 6}. Methodology and Interpretation",
                    styles["SectionTitle"],
                ),
            ]
        )
    else:
        story.extend(
            [
                PageBreak(),
                Paragraph("3. Methodology and Interpretation", styles["SectionTitle"]),
            ]
        )

    methodology_points = [
        "Evaluation metrics are analysed independently; values from different metrics are never compared or pooled.",
        f"The primary performance summary is {_display_name(options.primary_performance_metric)}.",
        f"Task-level IQM intervals use {options.bootstrap_confidence:.0%} BCa bootstrap confidence intervals over seeds with {options.bootstrap_samples:,} replicates.",
        "Cross-task probability intervals use a fixed-task stratified percentile bootstrap: seeds are resampled within each benchmark task and the probability statistic is recomputed.",
        "Pairwise probability of improvement is the primary cross-task evidence for a specific algorithm comparison.",
        "Mean superiority is the average probability of improvement against the complete algorithm roster and is an overall summary, not pair-specific evidence.",
        "Probability of improvement describes distributional overlap and consistency; it is not a measure of the numerical size of an improvement.",
        "Average rank, rank dispersion, Top-k rates, Friedman, and Nemenyi are supporting rank-based consistency summaries.",
        "Generated commentary is descriptive only and does not infer causal explanations for observed algorithm performance.",
        "All raw CSV outputs are retained alongside this report.",
    ]
    for point in methodology_points:
        story.append(paragraph(f"- {point}"))

    def footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(
            doc.leftMargin, 9 * mm, "CARES RL Statistical Benchmark Report"
        )
        canvas.drawRightString(A4[0] - doc.rightMargin, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def write_pdf_report(
    results_root: str | pathlib.Path,
    task_results: Mapping[str, TaskAnalysisResult],
    benchmark_result: BenchmarkAnalysisResult | None,
    benchmark_output: str | pathlib.Path | None,
    options: AnalysisOptions,
    reference_comparison: str | None = None,
) -> pathlib.Path:
    """Generate the final guided PDF after all statistical outputs are written."""
    root = pathlib.Path(results_root)
    benchmark = pathlib.Path(benchmark_output) if benchmark_output is not None else None
    destination = root / "statistical_report.pdf"
    _build_pdf_report(
        destination,
        task_results,
        benchmark_result,
        benchmark,
        options,
        reference_comparison,
    )
    return destination
