from __future__ import annotations

import argparse
import logging
import pathlib

import matplotlib.pyplot as plt

import cares_reinforcement_learning.reporting.loading as loading
import cares_reinforcement_learning.reporting.plotting.renderer as renderer
import cares_reinforcement_learning.reporting.analysis.cross_task_analysis as cross_task_analysis
import cares_reinforcement_learning.reporting.analysis.reporting_outputs as reporting_outputs
from cares_reinforcement_learning.reporting.analysis.models import (
    AnalysisOptions,
    BenchmarkAnalysisResult,
    MetricSpec,
    TaskAnalysisResult,
)
import cares_reinforcement_learning.reporting.analysis.task_analysis as task_analysis
from cares_reinforcement_learning.reporting.models import LoadedTask, PlotTask
from cares_reinforcement_learning.reporting.plotting.models import PanelSpec, SeriesSpec

LOGGER = logging.getLogger(__name__)


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
        description="Analyse one CARES RL task or a multi-task benchmark.",
    )
    parser.add_argument(
        "tasks",
        type=pathlib.Path,
        help="One task directory, or a root containing multiple task directories.",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--metric",
        action="append",
        type=_metric,
        default=None,
        metavar="COLUMN[:higher|lower]",
        help=(
            "Evaluation metric to analyse and plot. May be supplied repeatedly. "
            "Comparison conditions are compared only within the same metric."
        ),
    )
    parser.add_argument(
        "--step-column",
        default="total_steps",
        help="Evaluation x-axis column used for AUC and learning curves.",
    )
    parser.add_argument(
        "--comparison-parameter",
        action="append",
        default=[],
        metavar="CONFIG_PATH",
        help=(
            "Dotted configuration path used to distinguish conditions of the "
            "same algorithm. May be supplied repeatedly."
        ),
    )
    parser.add_argument("--allow-unmatched-seeds", action="store_true")
    parser.add_argument("--early-window-fraction", type=float, default=0.25)
    parser.add_argument("--final-window-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument("--no-learning-curves", action="store_true")
    parser.add_argument("--figure-columns", type=int, default=None)
    parser.add_argument("--figure-dpi", type=int, default=300)
    parser.add_argument(
        "--figure-format",
        action="append",
        default=None,
        dest="figure_formats",
        help="Learning-curve figure format. May be repeated. Default: png.",
    )
    parser.add_argument("--no-figure-std", action="store_true")
    parser.add_argument("--no-pdf-report", action="store_true")
    parser.add_argument(
        "--reference-comparison",
        default=None,
        help=(
            "Comparison condition to feature as the proposed method in the PDF "
            "report. The name must exactly match its discovered comparison label."
        ),
    )
    parser.add_argument(
        "--primary-performance-metric",
        default="auc",
        choices=["auc", "early_window_auc", "final_window_auc"],
    )
    return parser


def _validate_analysis_task_counts(tasks: dict[str, LoadedTask]) -> None:
    for task_name, task in tasks.items():
        if len(task.runs) < 2:
            raise ValueError(
                f"Task {task_name!r} contains only {len(task.runs)} valid comparison "
                "condition. At least two conditions are required for analysis."
            )


def _run_task_analyses(
    tasks: dict[str, LoadedTask],
    output_root: pathlib.Path,
    metrics: list[MetricSpec],
    options: AnalysisOptions,
) -> dict[str, TaskAnalysisResult]:
    LOGGER.info("Running task analyses for %d task(s)...", len(tasks))
    task_results: dict[str, TaskAnalysisResult] = {}
    for index, (task_name, task) in enumerate(tasks.items(), start=1):
        LOGGER.info("[%d/%d] Analysing task %s", index, len(tasks), task_name)
        task_output = output_root / "tasks" / task_name
        task_result, validation = task_analysis.run_task_analysis(
            task.runs,
            metrics,
            options,
        )
        task_result.write_csv(task_output)
        reporting_outputs.write_task_outputs(
            task_output,
            task_result.algorithm_summary,
            task_result.pairwise,
            task_result.task_summary,
            options,
            validation.comparison_design.value,
            validation.warnings,
        )
        task_results[task_name] = task_result

    LOGGER.info("Completed task analyses.")
    return task_results


def _run_cross_task_analysis(
    tasks: dict[str, LoadedTask],
    task_results: dict[str, TaskAnalysisResult],
    output_root: pathlib.Path,
    metrics: list[MetricSpec],
    args: argparse.Namespace,
    options: AnalysisOptions,
) -> tuple[pathlib.Path, BenchmarkAnalysisResult]:
    LOGGER.info("Running cross-task benchmark analysis...")
    benchmark_output = output_root / "benchmark"
    benchmark_result = cross_task_analysis.run_cross_task_analysis(
        task_results, options
    )
    benchmark_result.write_csv(benchmark_output)
    reporting_outputs.write_cross_task_outputs(
        benchmark_output,
        benchmark_result.benchmark_summary,
        benchmark_result.cross_task_pairwise,
        benchmark_result.friedman_tests,
        benchmark_result.nemenyi_posthoc,
        options,
    )

    if args.no_learning_curves:
        LOGGER.info("Skipping benchmark learning-curve figures (--no-learning-curves).")
        return benchmark_output, benchmark_result

    plot_tasks: tuple[PlotTask, ...] = tuple(
        task.to_plot_task() for task in tasks.values()
    )
    figure_output = benchmark_output / "figures"
    formats = args.figure_formats or ["png"]

    LOGGER.info(
        "Rendering benchmark learning-curve figures for %d metric(s)...", len(metrics)
    )
    for metric in metrics:
        LOGGER.info("Rendering figure series for metric %s", metric.column)
        plot = PanelSpec(
            source="eval",
            x=args.step_column,
            series=(SeriesSpec(metric.column),),
            x_label="Steps",
            y_label=metric.column,
        )
        metric_title = metric.column.replace("_", " ").title()
        figure = renderer.render_tasks(
            plot_tasks,
            plot=plot,
            title=f"{metric_title} ({metric.direction} is better)",
            columns=args.figure_columns,
            show_std=not args.no_figure_std,
        )
        renderer.save_figure(
            figure,
            figure_output,
            f"eval-{metric.column}",
            formats=formats,
            dpi=args.figure_dpi,
        )
        plt.close(figure)

    LOGGER.info("Completed cross-task benchmark analysis.")
    return benchmark_output, benchmark_result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args = _build_parser().parse_args()
    if args.figure_dpi < 72:
        raise ValueError("figure_dpi must be at least 72.")

    options = AnalysisOptions(
        early_window_fraction=args.early_window_fraction,
        final_window_fraction=args.final_window_fraction,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_confidence=args.bootstrap_confidence,
        random_seed=args.random_seed,
        allow_unmatched_seeds=args.allow_unmatched_seeds,
        primary_performance_metric=args.primary_performance_metric,
        significance_level=args.significance_level,
        evaluation_step_column=args.step_column,
    )

    metrics = args.metric or [MetricSpec("episode_reward", "higher")]
    LOGGER.info("Loading task data from %s", args.tasks)
    tasks = loading.load_tasks(
        args.tasks,
        comparison_parameters=args.comparison_parameter,
    )
    LOGGER.info("Loaded %d task(s).", len(tasks))

    LOGGER.info("Validating loaded tasks...")
    _validate_analysis_task_counts(tasks)
    LOGGER.info("Validation complete.")

    task_results = _run_task_analyses(
        tasks,
        args.output,
        metrics,
        options,
    )
    is_benchmark_run = len(task_results) > 1

    if is_benchmark_run:
        benchmark_output, benchmark_result = _run_cross_task_analysis(
            tasks,
            task_results,
            args.output,
            metrics,
            args,
            options,
        )
    else:
        LOGGER.info("Single-task run detected; skipping cross-task benchmark analysis.")
        benchmark_output = None
        benchmark_result = None

    if not args.no_pdf_report:
        LOGGER.info("Writing statistical PDF report...")
        reporting_outputs.write_pdf_report(
            results_root=args.output,
            task_results=task_results,
            benchmark_result=benchmark_result,
            benchmark_output=benchmark_output,
            options=options,
            reference_comparison=args.reference_comparison,
        )
        LOGGER.info("PDF report written to %s", args.output / "statistical_report.pdf")
    else:
        LOGGER.info("Skipping PDF report generation (--no-pdf-report).")

    LOGGER.info("Analysis pipeline complete.")


if __name__ == "__main__":
    main()
