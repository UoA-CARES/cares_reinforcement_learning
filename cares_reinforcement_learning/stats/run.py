from __future__ import annotations

import argparse
import pathlib

from cares_reinforcement_learning.stats import reporting
from cares_reinforcement_learning.stats.cross_task_analysis import (
    run_cross_task_analysis,
)
from cares_reinforcement_learning.stats.discovery import discover_tasks
from cares_reinforcement_learning.stats.models import AnalysisOptions, MetricSpec
from cares_reinforcement_learning.stats.task_analysis import run_task_analysis


def _metric(value: str) -> MetricSpec:
    column, separator, direction = value.partition(":")

    if not column:
        raise argparse.ArgumentTypeError("Metric column must not be empty.")

    if not separator:
        direction = "higher"

    if direction not in {"higher", "lower"}:
        raise argparse.ArgumentTypeError("Metric direction must be higher or lower.")

    return MetricSpec(column, direction)  # type: ignore[arg-type]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cares-rl-stats",
        description="Analyse one CARES RL task or an entire folder of tasks.",
    )
    parser.add_argument(
        "tasks",
        type=pathlib.Path,
        help="Task directory, or directory containing multiple tasks.",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--metric",
        action="append",
        type=_metric,
        default=None,
        metavar="COLUMN[:higher|lower]",
        help=(
            "Evaluation metric to analyse. May be supplied repeatedly. "
            "Comparison conditions are compared only within the same metric."
        ),
    )
    parser.add_argument(
        "--comparison-parameter",
        action="append",
        default=[],
        metavar="CONFIG_PATH",
        help=(
            "Dotted configuration path used to distinguish ablation conditions "
            "of the same algorithm. May be supplied repeatedly when several "
            "parameters change. Example: "
            "alg_config.plasticity.replacement_rate"
        ),
    )
    parser.add_argument("--allow-unmatched-seeds", action="store_true")
    parser.add_argument("--early-window-fraction", type=float, default=0.25)
    parser.add_argument("--final-window-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument("--no-statistical-figures", action="store_true")
    parser.add_argument("--no-pdf-report", action="store_true")
    parser.add_argument(
        "--reference-comparison",
        default=None,
        help=(
            "Comparison condition to feature as the proposed method in the PDF "
            "report. The name must exactly match its discovered comparison label."
        ),
    )
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument(
        "--primary-performance-metric",
        default="auc",
        choices=["auc", "early_window_auc", "final_window_auc"],
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    options = AnalysisOptions(
        early_window_fraction=args.early_window_fraction,
        final_window_fraction=args.final_window_fraction,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_confidence=args.bootstrap_confidence,
        random_seed=args.random_seed,
        allow_unmatched_seeds=args.allow_unmatched_seeds,
        primary_performance_metric=args.primary_performance_metric,
        significance_level=args.significance_level,
        generate_statistical_figures=not args.no_statistical_figures,
        figure_dpi=args.figure_dpi,
    )
    metrics = args.metric or [MetricSpec("episode_reward", "higher")]
    tasks = discover_tasks(
        args.tasks,
        comparison_parameters=args.comparison_parameter,
    )

    task_outputs: dict[str, pathlib.Path] = {}
    for task_name, discovered_runs in tasks.items():
        task_output = args.output / "tasks" / task_name
        run_task_analysis(discovered_runs, task_output, metrics, options)
        task_outputs[task_name] = task_output

    benchmark_output: pathlib.Path | None = None
    if len(task_outputs) > 1:
        benchmark_output = args.output / "benchmark"
        run_cross_task_analysis(task_outputs, benchmark_output, options)

    if not args.no_pdf_report:
        reporting.write_pdf_report(
            results_root=args.output,
            task_outputs=task_outputs,
            benchmark_output=benchmark_output,
            options=options,
            reference_comparison=args.reference_comparison,
        )


if __name__ == "__main__":
    main()
