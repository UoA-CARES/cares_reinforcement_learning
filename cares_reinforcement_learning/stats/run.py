from __future__ import annotations

import argparse
import logging
import pathlib

from cares_reinforcement_learning.stats import cross_task_analysis, task_analysis
from cares_reinforcement_learning.stats.data import MetricSpec
from cares_reinforcement_learning.stats.validation import ValidationMode

LOGGER = logging.getLogger(__name__)


def _parse_algorithm_runs(values: list[str]) -> dict[str, pathlib.Path]:
    runs: dict[str, pathlib.Path] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid run specification {value!r}. Expected ALGORITHM=PATH."
            )

        algorithm, path = value.split("=", maxsplit=1)
        algorithm = algorithm.strip()

        if not algorithm:
            raise ValueError(f"Missing algorithm name in {value!r}.")

        runs[algorithm] = pathlib.Path(path).expanduser().resolve()

    return runs


def _parse_task_stats(values: list[str]) -> dict[str, pathlib.Path]:
    task_stats: dict[str, pathlib.Path] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid task stats specification {value!r}. Expected TASK=PATH."
            )

        task_name, path = value.split("=", maxsplit=1)
        task_name = task_name.strip()

        if not task_name:
            raise ValueError(f"Missing task name in {value!r}.")

        task_stats[task_name] = pathlib.Path(path).expanduser().resolve()

    return task_stats


def _parse_task_runs(values: list[str]) -> dict[str, dict[str, pathlib.Path]]:
    task_runs: dict[str, dict[str, pathlib.Path]] = {}

    for value in values:
        if "=" not in value or ":" not in value:
            raise ValueError(
                f"Invalid task run specification {value!r}. "
                "Expected TASK:ALGORITHM=PATH."
            )

        task_and_algorithm, path = value.split("=", maxsplit=1)
        task_name, algorithm = task_and_algorithm.split(":", maxsplit=1)

        task_name = task_name.strip()
        algorithm = algorithm.strip()

        if not task_name or not algorithm:
            raise ValueError(
                f"Invalid task run specification {value!r}. "
                "Expected TASK:ALGORITHM=PATH."
            )

        task_runs.setdefault(task_name, {})[algorithm] = (
            pathlib.Path(path).expanduser().resolve()
        )

    return task_runs


def _parse_metric_specs(
    metrics: list[str], lower_is_better: list[str]
) -> list[MetricSpec]:
    lower_is_better_set = set(lower_is_better)

    return [
        MetricSpec(
            name=metric,
            direction="lower" if metric in lower_is_better_set else "higher",
        )
        for metric in metrics
    ]


def _add_common_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Evaluation metric columns to analyse, e.g. episode_reward win_rate.",
    )
    parser.add_argument(
        "--lower-is-better",
        nargs="*",
        default=[],
        help="Evaluation metrics where lower values are better.",
    )
    parser.add_argument(
        "--performance-metric",
        default="auc",
        choices=["auc", "early_auc_25", "final_window_mean"],
        help="Performance metric used for publication/cross-task summary.",
    )
    parser.add_argument(
        "--publication-evaluation-metric",
        default=None,
        help="Evaluation metric used for publication tables/commentary. Defaults to first metric.",
    )

    parser.add_argument(
        "--publication-performance-metric",
        default="auc",
        choices=["auc", "early_auc_25", "final_window_mean"],
        help="Performance metric used for publication tables/commentary.",
    )

    parser.add_argument(
        "--baseline-algorithm",
        default=None,
        help="Optional baseline algorithm for publication pairwise tables.",
    )
    parser.add_argument("--final-window-fraction", type=float, default=0.05)
    parser.add_argument("--final-window-min-points", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--allow-unmatched-seeds", action="store_true")
    parser.add_argument("--allow-different-seed-counts", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--allow-step-mismatch", action="store_true")
    parser.add_argument("--allow-missing-metrics", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CARES RL statistical analysis tools.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    task_parser = subparsers.add_parser(
        "task",
        help="Run statistical analysis for one task across multiple algorithms.",
    )
    task_parser.add_argument("--task-name", required=True)
    task_parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Algorithm runs as ALGORITHM=LOG_PATH.",
    )
    task_parser.add_argument("--output-dir", required=True)
    _add_common_analysis_args(task_parser)

    cross_parser = subparsers.add_parser(
        "cross",
        help=(
            "Run cross-task analysis. Can combine existing task stats folders and/or "
            "first process raw task algorithm runs."
        ),
    )
    cross_parser.add_argument("--output-dir", required=True)
    cross_parser.add_argument(
        "--task-stats",
        nargs="*",
        default=[],
        help="Existing task stats folders as TASK=STATS_OUTPUT_PATH.",
    )
    cross_parser.add_argument(
        "--task-runs",
        nargs="*",
        default=[],
        help="Raw task runs as TASK:ALGORITHM=LOG_PATH.",
    )
    _add_common_analysis_args(cross_parser)

    return parser


def _build_analysis_options(args: argparse.Namespace) -> task_analysis.AnalysisOptions:
    validation_mode = (
        ValidationMode.EXPLORATORY if args.exploratory else ValidationMode.STRICT
    )

    return task_analysis.AnalysisOptions(
        final_window_fraction=args.final_window_fraction,
        final_window_min_points=args.final_window_min_points,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_confidence=args.bootstrap_confidence,
        random_seed=args.random_seed,
        validation_mode=validation_mode,
        allow_unmatched_seeds=args.allow_unmatched_seeds,
        allow_different_seed_counts=args.allow_different_seed_counts,
        allow_config_mismatch=args.allow_config_mismatch,
        allow_step_mismatch=args.allow_step_mismatch,
        allow_missing_metrics=args.allow_missing_metrics,
    )


def _run_task_command(args: argparse.Namespace) -> None:
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    run_paths = _parse_algorithm_runs(args.runs)
    metric_specs = _parse_metric_specs(args.metrics, args.lower_is_better)
    options = _build_analysis_options(args)

    LOGGER.info(
        "Running task analysis for task %s with %d algorithm(s).",
        args.task_name,
        len(run_paths),
    )

    task_analysis.run_task_analysis(
        task_name=args.task_name,
        run_paths=run_paths,
        output_dir=output_dir,
        metric_specs=metric_specs,
        options=options,
        publication_evaluation_metric=args.publication_evaluation_metric,
        publication_performance_metric=args.publication_performance_metric,
        baseline_algorithm=args.baseline_algorithm,
    )


def _run_cross_command(args: argparse.Namespace) -> None:
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()

    metric_specs = _parse_metric_specs(
        args.metrics,
        args.lower_is_better,
    )

    publication_evaluation_metric = (
        args.publication_evaluation_metric or args.metrics[0]
    )

    options = _build_analysis_options(args)

    task_stats = _parse_task_stats(args.task_stats)
    task_runs = _parse_task_runs(args.task_runs)

    if not task_stats and not task_runs:
        raise ValueError("Provide --task-stats and/or --task-runs.")

    # Run raw task analyses first if provided.
    for task_name, run_paths in task_runs.items():
        LOGGER.info(
            "Processing task %s with %d algorithm(s).",
            task_name,
            len(run_paths),
        )

        task_output_dir = task_analysis.run_task_analysis(
            task_name=task_name,
            run_paths=run_paths,
            output_dir=output_dir,
            metric_specs=metric_specs,
            options=options,
            publication_evaluation_metric=publication_evaluation_metric,
            publication_performance_metric=args.publication_performance_metric,
            baseline_algorithm=args.baseline_algorithm,
        )

        task_stats[task_name] = task_output_dir

    benchmark_output_dir = output_dir / "cross_task"
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Running cross-task analysis over %d task(s).",
        len(task_stats),
    )

    cross_task_analysis.run_cross_task_analysis(
        task_outputs=task_stats,
        output_dir=benchmark_output_dir,
        evaluation_metric=publication_evaluation_metric,
        performance_metric=args.publication_performance_metric,
        options=options,
    )

    LOGGER.info("Cross-task analysis completed successfully.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "task":
        _run_task_command(args)
    elif args.command == "cross":
        _run_cross_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
