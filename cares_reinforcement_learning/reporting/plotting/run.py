from __future__ import annotations

import argparse
import dataclasses
import pathlib
from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

import matplotlib.pyplot as plt

import cares_reinforcement_learning.reporting.loading as loading
import cares_reinforcement_learning.reporting.plotting.models as plotting_models
import cares_reinforcement_learning.reporting.plotting.renderer as renderer
from cares_reinforcement_learning.reporting.models import LoadedTask, PlotTask
from cares_reinforcement_learning.reporting.plotting.models import (
    FigureSpec,
    PanelSpec,
    PlotSource,
    PlotStyleSpec,
    SeriesSpec,
    plot_style_spec,
)

LoadedTasks: TypeAlias = dict[str, LoadedTask]
PlotTasks: TypeAlias = dict[str, PlotTask]


def _parse_plot(value: str) -> PanelSpec:
    """Parse one plot specification.

    Required fields:
      source=train|eval
      y=<metric>[,<metric>]

    Optional fields:
      x=<metric>                 Default: total_steps
      y2=<metric>[,<metric>]     Secondary y-axis metrics
      title=<text>               Panel title
      x_label=<text>             Display label for the x-axis
      y_label=<text>             Display label for the primary y-axis
      y2_label=<text>            Display label for the secondary y-axis
      x_scale=linear|log
      y_scale=linear|log
      y2_scale=linear|log

    Examples:
      source=eval;y=episode_reward

      source=train;y=actor_loss;x_label=Environment Steps;y_label=Actor Loss

      source=train;y=critic_loss;y2=alpha;\
        x_label=Environment Steps;\
        y_label=Critic Loss;\
        y2_label=Temperature;\
        title=Training Diagnostics

    Repeating --plot creates multiple panels for one task. With combined
    --tasks, each --plot specification creates one task-grid figure.
    """
    fields: dict[str, str] = {}
    for item in value.split(";"):
        key, separator, raw = item.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(
                f"Invalid plot item {item!r}; expected key=value."
            )
        key = key.strip().lower()
        if key in fields:
            raise argparse.ArgumentTypeError(
                f"Plot key {key!r} was supplied more than once."
            )
        fields[key] = raw.strip()

    allowed = {
        "source",
        "x",
        "y",
        "y2",
        "title",
        "x_label",
        "y_label",
        "y2_label",
        "x_scale",
        "y_scale",
        "y2_scale",
    }
    unknown = sorted(set(fields).difference(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown plot keys: {unknown}")

    source = fields.get("source")
    if source not in {"train", "eval"}:
        raise argparse.ArgumentTypeError("Plot source must be train or eval.")
    source_literal = cast(PlotSource, source)

    primary = tuple(
        column.strip() for column in fields.get("y", "").split(",") if column.strip()
    )
    if not primary:
        raise argparse.ArgumentTypeError("Plot y must contain at least one column.")
    secondary = tuple(
        column.strip() for column in fields.get("y2", "").split(",") if column.strip()
    )

    try:
        return PanelSpec(
            source=source_literal,
            x=fields.get("x", "total_steps"),
            series=tuple(SeriesSpec(column) for column in primary)
            + tuple(SeriesSpec(column, axis="secondary") for column in secondary),
            title=fields.get("title") or None,
            x_label=fields.get("x_label") or None,
            y_label=fields.get("y_label") or None,
            secondary_y_label=fields.get("y2_label") or None,
            x_scale=fields.get("x_scale", "linear"),  # type: ignore[arg-type]
            y_scale=fields.get("y_scale", "linear"),  # type: ignore[arg-type]
            secondary_y_scale=fields.get("y2_scale", "linear"),  # type: ignore[arg-type]
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cares-rl-plot",
        description="Plot CARES RL training and evaluation logs.",
    )

    # ------------------------------------------------------------------
    # Input selection
    # ------------------------------------------------------------------

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--tasks",
        type=pathlib.Path,
        metavar="DIR",
        help="Root directory containing multiple task folders.",
    )
    inputs.add_argument(
        "--task",
        type=pathlib.Path,
        metavar="DIR",
        help="Single task directory containing algorithm result folders.",
    )
    inputs.add_argument(
        "-d",
        "--data",
        nargs="+",
        type=pathlib.Path,
        metavar="DIR",
        help="Explicit algorithm result folders to compare as one task.",
    )

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------

    general = parser.add_argument_group("General")

    general.add_argument(
        "--output",
        type=pathlib.Path,
        metavar="DIR",
        help="Output directory. Required unless --list-comparisons is used.",
    )
    general.add_argument(
        "--format",
        action="append",
        dest="formats",
        default=None,
        metavar="FORMAT",
        help="Output format (png, pdf, svg, ...). May be repeated. Default: png.",
    )
    general.add_argument(
        "--dpi",
        type=int,
        default=300,
        metavar="DPI",
        help="Output resolution in dots per inch. Default: 300.",
    )
    general.add_argument(
        "--title",
        default=None,
        metavar="TITLE",
        help="Override the figure title.",
    )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    comparison = parser.add_argument_group("Comparison")

    comparison.add_argument(
        "--comparison-parameter",
        action="append",
        default=[],
        metavar="CONFIG_PATH",
        help=(
            "Typed configuration path used to distinguish runs of the same "
            "algorithm. May be supplied multiple times."
        ),
    )
    comparison.add_argument(
        "--legend-label",
        action="append",
        default=[],
        metavar="LABEL",
        help=(
            "Override legend labels in discovered comparison order. "
            "May be supplied multiple times."
        ),
    )
    comparison.add_argument(
        "--list-comparisons",
        action="store_true",
        help="Print the discovered comparison order and exit.",
    )

    # ------------------------------------------------------------------
    # Plot specification
    # ------------------------------------------------------------------

    plotting = parser.add_argument_group("Plot specification")

    plotting.add_argument(
        "--plot",
        action="append",
        type=_parse_plot,
        default=None,
        metavar="SPEC",
        help=(
            "Plot specification using semicolon-separated key=value fields. "
            "Required: source=<train|eval>;y=<metric>[,<metric>]. "
            "Optional: x=<metric>;y2=<metric>[,<metric>];title=<text>;"
            "x_label=<text>;y_label=<text>;y2_label=<text>;"
            "x_scale=<linear|log>;y_scale=<linear|log>;"
            "y2_scale=<linear|log>. "
            "Repeat --plot to create multiple panels. "
            "Example: --plot "
            "'source=train;y=critic_loss;y2=alpha;"
            "x_label=Environment Steps;"
            "y_label=Critic Loss;"
            "y2_label=Temperature;"
            "title=Training Diagnostics'."
        ),
    )
    plotting.add_argument(
        "--separate",
        action="store_true",
        help=(
            "With --tasks, render one figure per task instead of one combined figure."
        ),
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    layout = parser.add_argument_group("Layout")

    layout.add_argument(
        "--rows",
        type=int,
        default=None,
        metavar="N",
        help="Number of subplot rows. Default: automatic.",
    )
    layout.add_argument(
        "--columns",
        type=int,
        default=None,
        metavar="N",
        help="Number of subplot columns. Default: automatic.",
    )
    layout.add_argument(
        "--figure-width",
        dest="figure_width",
        type=float,
        default=None,
        metavar="INCHES",
        help=(
            "Total figure width in inches. "
            "Defaults are chosen automatically for single and multi-panel figures."
        ),
    )
    layout.add_argument(
        "--row-height",
        dest="row_height",
        type=float,
        default=None,
        metavar="INCHES",
        help=(
            "Height allocated to each subplot row in inches. "
            "Total figure height is row_height x number of rows."
        ),
    )

    # ------------------------------------------------------------------
    # Training processing
    # ------------------------------------------------------------------

    processing = parser.add_argument_group("Training processing")

    processing.add_argument(
        "--train-window-size",
        type=int,
        default=20,
        metavar="N",
        help="Centered rolling mean window applied to training curves. Default: 20.",
    )
    processing.add_argument(
        "--train-bin-size",
        type=float,
        default=None,
        metavar="STEPS",
        help=(
            "Optional fixed training-step bin width used to align seeds before "
            "cross-seed aggregation."
        ),
    )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    display = parser.add_argument_group("Display")

    display.add_argument(
        "--plot-seeds",
        action="store_true",
        help="Plot individual seed curves in addition to the aggregate.",
    )
    display.add_argument(
        "--no-mean",
        action="store_true",
        help="Hide the aggregate mean curve.",
    )
    display.add_argument(
        "--no-std",
        action="store_true",
        help="Hide the standard deviation envelope.",
    )

    # ------------------------------------------------------------------
    # Font overrides
    # ------------------------------------------------------------------

    fonts = parser.add_argument_group("Font overrides")

    fonts.add_argument(
        "--label-fontsize",
        type=int,
        default=None,
        metavar="PT",
        help="Axis label font size in points.",
    )
    fonts.add_argument(
        "--title-fontsize",
        type=int,
        default=None,
        metavar="PT",
        help="Figure/panel title font size in points.",
    )
    fonts.add_argument(
        "--ticks-fontsize",
        type=int,
        default=None,
        metavar="PT",
        help="Axis tick label font size in points.",
    )
    fonts.add_argument(
        "--legend-fontsize",
        type=int,
        default=None,
        metavar="PT",
        help="Legend font size in points.",
    )

    return parser


def _load_tasks(args: argparse.Namespace) -> LoadedTasks:
    if args.tasks is not None:
        return loading.load_tasks(
            args.tasks,
            comparison_parameters=args.comparison_parameter,
        )

    if args.task is not None:
        task = loading.load_task(
            args.task,
            comparison_parameters=args.comparison_parameter,
        )
        return {task.name: task}

    task = loading.load_runs(
        args.data,
        comparison_parameters=args.comparison_parameter,
    )

    return {task.name: task}


def _comparison_order(
    loaded_tasks: Mapping[str, LoadedTask],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            run.comparison_name for task in loaded_tasks.values() for run in task.runs
        )
    )


def _print_comparison_order(comparison_names: Sequence[str]) -> None:
    print("Discovered comparisons (legend order):")
    for index, name in enumerate(comparison_names, start=1):
        print(f"  {index}. {name}")


def _input_description(args: argparse.Namespace) -> str:
    if args.tasks is not None:
        return f"multiple-task root: {args.tasks.expanduser().resolve()}"
    if args.task is not None:
        return f"single-task directory: {args.task.expanduser().resolve()}"
    directories = ", ".join(str(path.expanduser().resolve()) for path in args.data)
    return f"explicit run directories: {directories}"


def _print_discovery_summary(
    loaded_tasks: Mapping[str, LoadedTask],
    args: argparse.Namespace,
) -> None:
    print("\n=== CARES RL plot discovery ===")
    print(f"Input: {_input_description(args)}")
    if args.comparison_parameter:
        print("Comparison parameters:")
        for path in args.comparison_parameter:
            print(f"  - {path}")
    else:
        print("Comparison parameters: none")

    print(f"Tasks discovered: {len(loaded_tasks)}")
    for task_name, task in loaded_tasks.items():
        print(f"  Task {task_name!r}: {len(task.runs)} comparison(s)")
        for index, run in enumerate(task.runs, start=1):
            print(
                f"    {index}. {run.comparison_name} "
                f"(algorithm={run.algorithm}, root={run.root})"
            )
            if run.variant_parameters:
                parameters = ", ".join(
                    f"{name}={value!r}"
                    for name, value in run.variant_parameters.items()
                )
                print(f"       parameters: {parameters}")


def _resolve_legend_labels(
    loaded_tasks: Mapping[str, LoadedTask],
    labels: Sequence[str],
) -> dict[str, str]:
    comparison_names = _comparison_order(loaded_tasks)
    if not labels:
        return {name: name for name in comparison_names}

    if len(labels) != len(comparison_names):
        _print_comparison_order(comparison_names)
        raise ValueError(
            f"Received {len(labels)} legend labels for "
            f"{len(comparison_names)} discovered comparisons."
        )

    cleaned_labels = tuple(label.strip() for label in labels)
    if any(not label for label in cleaned_labels):
        raise ValueError("Legend labels must not be empty.")
    if len(set(cleaned_labels)) != len(cleaned_labels):
        raise ValueError("Legend labels must be unique.")

    return dict(zip(comparison_names, cleaned_labels, strict=True))


def _print_legend_labels(label_by_comparison: dict[str, str]) -> None:
    print("\nLegend labels:")
    for index, (comparison, label) in enumerate(
        label_by_comparison.items(),
        start=1,
    ):
        if comparison == label:
            print(f"  {index}. {label}")
        else:
            print(f"  {index}. {comparison} -> {label}")


def _apply_legend_labels(
    loaded_tasks: Mapping[str, LoadedTask],
    label_by_comparison: dict[str, str],
) -> LoadedTasks:
    return {
        task_name: dataclasses.replace(
            task,
            runs=tuple(
                dataclasses.replace(
                    run,
                    discovered=dataclasses.replace(
                        run.discovered,
                        comparison_name=label_by_comparison[run.comparison_name],
                    ),
                )
                for run in task.runs
            ),
        )
        for task_name, task in loaded_tasks.items()
    }


def _plot_description(plot: PanelSpec) -> str:
    primary = ", ".join(
        series.column for series in plot.series if series.axis == "primary"
    )
    secondary = ", ".join(
        series.column for series in plot.series if series.axis == "secondary"
    )
    description = (
        f"source={plot.source}, x={plot.x}, y={primary}, "
        f"x_label={plot.x_label or plot.x!r}, "
        f"y_label={plot.y_label or primary!r}"
    )
    if secondary:
        description += (
            f", y2={secondary}, " f"y2_label={plot.secondary_y_label or secondary!r}"
        )
    if plot.title:
        description += f", title={plot.title!r}"
    description += (
        f", scales=({plot.x_scale}, {plot.y_scale}"
        f"{', ' + plot.secondary_y_scale if secondary else ''})"
    )
    return description


def _print_loaded_task(task: LoadedTask) -> None:
    print(f"Loaded task {task.name!r}:")
    for run in task.runs:
        seed_ids = sorted(run.seeds)
        train_count = sum(seed.train_data is not None for seed in run.seeds.values())
        eval_count = sum(seed.eval_data is not None for seed in run.seeds.values())
        print(
            f"  - {run.comparison_name}: {len(seed_ids)} seed(s) "
            f"{seed_ids}; train logs={train_count}, eval logs={eval_count}"
        )


def _resolved_plot_style(
    args: argparse.Namespace,
    *,
    compact: bool,
) -> PlotStyleSpec:
    preset = plot_style_spec(compact=compact)
    return dataclasses.replace(
        preset,
        figure_width=(
            args.figure_width if args.figure_width is not None else preset.figure_width
        ),
        row_height=(
            args.row_height if args.row_height is not None else preset.row_height
        ),
        label_fontsize=(
            args.label_fontsize
            if args.label_fontsize is not None
            else preset.label_fontsize
        ),
        title_fontsize=(
            args.title_fontsize
            if args.title_fontsize is not None
            else preset.title_fontsize
        ),
        ticks_fontsize=(
            args.ticks_fontsize
            if args.ticks_fontsize is not None
            else preset.ticks_fontsize
        ),
        legend_fontsize=(
            args.legend_fontsize
            if args.legend_fontsize is not None
            else preset.legend_fontsize
        ),
    )


def _print_render_settings(
    args: argparse.Namespace,
    formats: Sequence[str],
) -> None:
    print("\n=== Plot settings ===")
    print(f"Output: {args.output.expanduser().resolve()}")
    print(f"Formats: {', '.join(formats)}; DPI: {args.dpi}")
    print(
        f"Figure: figure_width={args.figure_width or 'default'}, "
        f"row_height={args.row_height or 'default'}, "
        f"rows={args.rows or 'auto'}, columns={args.columns or 'auto'}"
    )
    print(
        f"Fonts: labels={args.label_fontsize or 'default'}, "
        f"titles={args.title_fontsize or 'default'}, "
        f"ticks={args.ticks_fontsize or 'default'}, "
        f"legend={args.legend_fontsize or 'default'}"
    )
    print(
        f"Training processing: window={args.train_window_size}, "
        f"bin_size={args.train_bin_size}"
    )
    print(
        f"Display: seeds={args.plot_seeds}, mean={not args.no_mean}, "
        f"std={not args.no_std}"
    )


def _plot_stem(plot: PanelSpec) -> str:
    series = "-".join(item.column for item in plot.series)
    return f"{plot.source}-{series}"


def _explicit_figure_spec(
    args: argparse.Namespace,
    task_name: str,
    plots: Sequence[PanelSpec],
    *,
    compact: bool,
) -> FigureSpec:
    style = _resolved_plot_style(args, compact=compact)
    return FigureSpec(
        panels=tuple(plots),
        title=args.title or task_name,
        rows=args.rows,
        columns=args.columns,
        dpi=args.dpi,
        style=style,
        train_window_size=args.train_window_size,
        train_bin_size=args.train_bin_size,
        show_seeds=args.plot_seeds,
        show_mean=not args.no_mean,
        show_std=not args.no_std,
        show_panel_titles=True,
    )


def _apply_figure_overrides(
    spec: FigureSpec,
    args: argparse.Namespace,
    *,
    title: str,
    compact: bool,
) -> FigureSpec:
    style = _resolved_plot_style(args, compact=compact)
    return dataclasses.replace(
        spec,
        title=args.title or title,
        rows=args.rows if args.rows is not None else spec.rows,
        columns=args.columns if args.columns is not None else spec.columns,
        dpi=args.dpi,
        style=style,
        train_window_size=args.train_window_size,
        train_bin_size=args.train_bin_size,
        show_seeds=args.plot_seeds,
        show_mean=not args.no_mean,
        show_std=not args.no_std,
    )


def _combined_task_panels(args: argparse.Namespace) -> tuple[PanelSpec, ...]:
    if args.plot is not None:
        return tuple(args.plot)
    return tuple(panel for _, panel in plotting_models.default_panels())


def _render_combined_tasks(
    plot_tasks: Mapping[str, PlotTask],
    args: argparse.Namespace,
    formats: Sequence[str],
) -> None:
    tasks = tuple(plot_tasks.values())

    for plot in _combined_task_panels(args):
        print(f"Rendering combined task figure: {_plot_description(plot)}")
        style = _resolved_plot_style(args, compact=True)
        spec = FigureSpec(
            panels=(plot,),
            title=args.title,
            rows=args.rows,
            columns=args.columns,
            dpi=args.dpi,
            style=style,
            train_window_size=args.train_window_size,
            train_bin_size=args.train_bin_size,
            show_seeds=args.plot_seeds,
            show_mean=not args.no_mean,
            show_std=not args.no_std,
            show_panel_titles=True,
        )
        figure = renderer.render_tasks(
            tasks,
            spec=spec,
        )
        paths = renderer.save_figure(
            figure,
            args.output,
            f"tasks-{_plot_stem(plot)}",
            formats=formats,
            dpi=args.dpi,
        )
        plt.close(figure)
        for path in paths:
            print(f"  Saved: {path.resolve()}")


def _render_task(
    task: PlotTask,
    args: argparse.Namespace,
    formats: Sequence[str],
) -> None:
    task_name = task.name

    figure_specs: tuple[tuple[str | None, FigureSpec], ...]
    if args.plot is None:
        figure_specs = tuple(
            (
                suffix,
                _apply_figure_overrides(
                    spec,
                    args,
                    title=task_name,
                    compact=False,
                ),
            )
            for suffix, spec in plotting_models.default_figure_specs(compact=False)
        )
    else:
        compact = len(args.plot) > 1
        figure_specs = (
            (
                None,
                _explicit_figure_spec(
                    args,
                    task_name,
                    args.plot,
                    compact=compact,
                ),
            ),
        )

    for suffix, spec in figure_specs:
        print(f"  Rendering figure: {spec.title or task_name}")
        for index, panel in enumerate(spec.panels, start=1):
            print(f"    Panel {index}: {_plot_description(panel)}")

        figure = renderer.render_task(
            task,
            spec,
        )
        output_name = task_name if suffix is None else f"{task_name}-{suffix}"
        paths = renderer.save_figure(
            figure,
            args.output,
            output_name,
            formats=formats,
            dpi=spec.dpi,
        )
        plt.close(figure)
        for path in paths:
            print(f"    Saved: {path.resolve()}")


def _render_separate_tasks(
    plot_tasks: Mapping[str, PlotTask],
    args: argparse.Namespace,
    formats: Sequence[str],
) -> None:
    for task in plot_tasks.values():
        _render_task(task, args, formats)


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.separate and args.tasks is None:
        raise ValueError("--separate may only be used with --tasks.")

    loaded_tasks = _load_tasks(args)
    _print_discovery_summary(loaded_tasks, args)
    comparison_names = _comparison_order(loaded_tasks)

    if args.list_comparisons:
        _print_comparison_order(comparison_names)
        return

    if args.output is None:
        raise ValueError("--output is required unless --list-comparisons is used.")

    label_by_comparison = _resolve_legend_labels(
        loaded_tasks,
        args.legend_label,
    )
    _print_legend_labels(label_by_comparison)
    loaded_tasks = _apply_legend_labels(loaded_tasks, label_by_comparison)

    print("\nLoaded data:")
    for task in loaded_tasks.values():
        _print_loaded_task(task)

    plot_tasks: PlotTasks = {
        task_name: task.to_plot_task() for task_name, task in loaded_tasks.items()
    }

    formats = tuple(args.formats or ["png"])
    _print_render_settings(args, formats)

    print("\n=== Rendering ===")
    if args.tasks is not None and not args.separate:
        _render_combined_tasks(plot_tasks, args, formats)
    else:
        _render_separate_tasks(plot_tasks, args, formats)
    print("\nPlotting complete.")


if __name__ == "__main__":
    main()
