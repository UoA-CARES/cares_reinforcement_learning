from __future__ import annotations

import dataclasses
import math
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

import cares_reinforcement_learning.reporting.analysis.metrics as analysis_metrics
from cares_reinforcement_learning.reporting.models import PlotRun, PlotTask
from cares_reinforcement_learning.reporting.plotting.models import (
    FigureSpec,
    PanelSpec,
    SeriesSpec,
)

_LINESTYLES = ("-", "--", "-.", ":")
SeedFrame = tuple[int, pd.DataFrame]
SeedFrames = tuple[SeedFrame, ...]


def _run_frames(
    run: PlotRun,
    source: str,
) -> Mapping[int, pd.DataFrame]:
    return run.train_frames if source == "train" else run.eval_frames


def _validate_panel(task: PlotTask, panel: PanelSpec) -> None:
    required_columns = tuple({panel.x, *(series.column for series in panel.series)})
    failures: list[str] = []
    available_source_count = 0

    for run in task.runs:
        frames = _run_frames(run, panel.source)
        if not frames:
            failures.append(f"{run.name}: missing {panel.source} data")
            continue

        for seed_number, frame in frames.items():
            try:
                analysis_metrics.ensure_required_columns(
                    frame,
                    required_columns,
                    context=f"{run.name} / seed {seed_number}",
                )
            except ValueError as error:
                failures.append(str(error))
                continue
            available_source_count += 1

    if available_source_count == 0:
        raise ValueError(
            f"Task {task.name!r} has no valid {panel.source} data for panel "
            f"{panel.title or panel.x!r}."
        )

    if failures:
        details = "\n  ".join(failures)
        raise ValueError(
            f"Panel {panel.title or panel.x!r} cannot be plotted:\n  {details}"
        )


def _prepare_seed_series(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    window_size: int,
    bin_size: float | None,
) -> pd.DataFrame:
    """Smooth and optionally bin one seed series for stable cross-seed aggregation."""
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    if bin_size is not None and bin_size <= 0:
        raise ValueError("train_bin_size must be positive when supplied.")

    seed_series = frame.loc[:, [x, y]].copy()
    seed_series = seed_series.loc[
        seed_series[x].notna() & seed_series[y].notna()
    ].sort_values(x)
    seed_series[y] = (
        seed_series[y].rolling(window=window_size, min_periods=1, center=True).mean()
    )

    if bin_size is None:
        return seed_series

    # Assign every training observation to a fixed-width step bucket, then
    # aggregate within each seed before aggregating across seeds.
    seed_series["__x_bin"] = (
        np.floor(np.asarray(seed_series[x], dtype=float) / bin_size) * bin_size
    )
    binned_series = seed_series.groupby("__x_bin", as_index=False, sort=True).agg(
        {y: "mean"}
    )
    binned_series = binned_series.rename(columns={"__x_bin": x})
    return binned_series


def _aggregate_run_series(
    run: PlotRun,
    panel: PanelSpec,
    series: SeriesSpec,
    *,
    train_window_size: int,
    train_bin_size: float | None,
) -> tuple[pd.DataFrame, SeedFrames]:
    """Build per-run summary stats and return prepared per-seed curves."""
    seed_frame_list: list[SeedFrame] = []
    frames = _run_frames(run, panel.source)
    for seed_number, frame in sorted(frames.items()):
        seed_series = _prepare_seed_series(
            frame,
            x=panel.x,
            y=series.column,
            window_size=train_window_size if panel.source == "train" else 1,
            bin_size=train_bin_size if panel.source == "train" else None,
        )
        seed_series["__seed"] = seed_number
        seed_frame_list.append((seed_number, seed_series))

    combined_series = pd.concat(
        [frame for _, frame in seed_frame_list],
        ignore_index=True,
    )
    summary = (
        combined_series.groupby(panel.x, as_index=False, sort=True)[series.column]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return summary, tuple(seed_frame_list)


def _humanise_metric_name(value: str) -> str:
    cleaned = re.sub(
        r"^(training|evaluation|train|eval)[\s:_-]+",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.replace("episode_reward", "reward")
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split()).title()


def _panel_title(panel: PanelSpec) -> str:
    if panel.title:
        return _humanise_metric_name(panel.title)
    metric_names = ", ".join(
        _humanise_metric_name(series.label or series.column) for series in panel.series
    )
    return metric_names


def _series_label(run: PlotRun, panel: PanelSpec, series: SeriesSpec) -> str:
    primary_count = sum(item.axis == "primary" for item in panel.series)
    secondary_count = sum(item.axis == "secondary" for item in panel.series)

    # Keep legends focused on run/comparison names. Only append metric names
    # when multiple series share one axis and need disambiguation.
    if primary_count <= 1 and secondary_count <= 1:
        return run.name

    axis_count = primary_count if series.axis == "primary" else secondary_count
    if axis_count <= 1:
        return run.name

    return f"{run.name} — {series.label or series.column}"


def _grid_shape(
    count: int,
    *,
    rows: int | None,
    columns: int | None,
    max_columns: int = 3,
) -> tuple[int, int]:
    """Resolve a subplot grid for a given number of plotted items."""
    if count < 1:
        raise ValueError("At least one item is required to create a subplot grid.")

    if rows is not None and rows < 1:
        raise ValueError("rows must be positive when supplied.")

    if columns is not None and columns < 1:
        raise ValueError("columns must be positive when supplied.")

    if rows is not None and columns is not None:
        if rows * columns < count:
            raise ValueError(
                "rows x columns is smaller than the number of items to plot."
            )
        return rows, columns

    if rows is not None:
        return rows, math.ceil(count / rows)

    if columns is not None:
        return math.ceil(count / columns), columns

    resolved_columns = min(max_columns, count)
    return math.ceil(count / resolved_columns), resolved_columns


def _color_cycle(size: int) -> list[str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if colors:
        return colors
    return [f"C{index}" for index in range(max(1, size))]


def _draw_series(
    target_axis: Axes,
    *,
    panel: PanelSpec,
    series: SeriesSpec,
    aggregate: pd.DataFrame,
    seed_frames: SeedFrames,
    color: str,
    linestyle: str,
    label: str,
    show_seeds: bool,
    show_mean: bool,
    show_std: bool,
    mean_linewidth: float,
    seed_linewidth: float,
    seed_alpha: float,
    std_alpha: float,
) -> None:
    if show_seeds:
        for _, seed_frame in seed_frames:
            target_axis.plot(
                seed_frame[panel.x],
                seed_frame[series.column],
                color=color,
                linestyle=linestyle,
                linewidth=seed_linewidth,
                alpha=seed_alpha,
            )

    if show_mean:
        target_axis.plot(
            aggregate[panel.x],
            aggregate["mean"],
            color=color,
            linestyle=linestyle,
            linewidth=mean_linewidth,
            label=label,
        )

    if show_std:
        valid = aggregate["std"].notna()
        if valid.any():
            target_axis.fill_between(
                aggregate.loc[valid, panel.x],
                aggregate.loc[valid, "mean"] - aggregate.loc[valid, "std"],
                aggregate.loc[valid, "mean"] + aggregate.loc[valid, "std"],
                color=color,
                alpha=std_alpha,
                linewidth=0,
            )


def _superscript_exponent(exponent: int) -> str:
    translation = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(exponent).translate(translation)


def _configure_compact_step_axis(axis: Axes, panel: PanelSpec) -> None:
    """Move large step scaling into the x-axis label instead of offset text."""
    if panel.x_scale != "linear":
        return

    x_min, x_max = axis.get_xlim()
    magnitude = max(abs(x_min), abs(x_max))
    if magnitude < 1_000:
        return

    exponent = int(math.floor(math.log10(magnitude) / 3) * 3)
    scale = 10.0**exponent
    base_label = panel.x_label or panel.x

    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / scale:g}"))
    axis.xaxis.offsetText.set_visible(False)
    axis.set_xlabel(
        f"{base_label} (×10{_superscript_exponent(exponent)})",
        fontsize=axis.xaxis.label.get_fontsize(),
    )


def _configure_panel_axes(
    axis: Axes,
    panel: PanelSpec,
    *,
    spec: FigureSpec,
    show_title: bool,
) -> None:
    if show_title:
        axis.set_title(_panel_title(panel), fontsize=spec.style.title_fontsize, pad=3)
    axis.set_xlabel(panel.x_label or panel.x, fontsize=spec.style.label_fontsize)
    axis.set_ylabel(
        panel.y_label or panel.series[0].column,
        fontsize=spec.style.label_fontsize,
    )
    axis.set_xscale(panel.x_scale)
    axis.set_yscale(panel.y_scale)
    axis.tick_params(axis="both", labelsize=spec.style.ticks_fontsize)
    axis.grid(True, alpha=spec.style.grid_alpha)
    if spec.style.compact_step_axis:
        _configure_compact_step_axis(axis, panel)
    if spec.style.axes_box_aspect is not None:
        axis.set_box_aspect(spec.style.axes_box_aspect)


def _combined_legend_entries(
    axis: Axes,
    secondary_axis: Axes | None,
    panel: PanelSpec,
    *,
    spec: FigureSpec,
) -> tuple[tuple[list[Artist], list[str]], tuple[list[Artist], list[str]]]:
    primary_handles, primary_labels = axis.get_legend_handles_labels()
    if secondary_axis is None:
        return (primary_handles, primary_labels), ([], [])

    secondary_axis.set_ylabel(
        panel.secondary_y_label
        or next(series.column for series in panel.series if series.axis == "secondary"),
        fontsize=spec.style.label_fontsize,
    )
    secondary_axis.set_yscale(panel.secondary_y_scale)
    secondary_axis.tick_params(axis="y", labelsize=spec.style.ticks_fontsize)
    secondary_handles, secondary_labels = secondary_axis.get_legend_handles_labels()
    return (primary_handles, primary_labels), (secondary_handles, secondary_labels)


def _remove_unused_axes(axes: np.ndarray, used_count: int) -> None:
    for unused_axis in list(axes.flat)[used_count:]:
        unused_axis.remove()


def _merge_legend_entries(
    target_handles: list[Artist],
    target_labels: list[str],
    handles: Sequence[Artist],
    labels: Sequence[str],
) -> None:
    for handle, label in zip(handles, labels, strict=True):
        if label not in target_labels:
            target_handles.append(handle)
            target_labels.append(label)


def _plot_panel(
    axis: Axes,
    task: PlotTask,
    spec: FigureSpec,
    panel: PanelSpec,
    *,
    show_legend: bool = True,
    color_by_comparison: Mapping[str, str] | None = None,
    show_panel_title: bool = True,
) -> tuple[tuple[list[Artist], list[str]], tuple[list[Artist], list[str]]]:
    """Render one panel including optional secondary axis and legend entries."""
    _validate_panel(task, panel)
    secondary_axis: Axes | None = None
    if any(series.axis == "secondary" for series in panel.series):
        secondary_axis = axis.twinx()

    cycle = _color_cycle(len(task.runs))

    if color_by_comparison is None:
        color_by_comparison = {
            run.name: cycle[index % len(cycle)] for index, run in enumerate(task.runs)
        }

    for run_index, run in enumerate(task.runs):
        color = color_by_comparison.get(run.name, cycle[run_index % len(cycle)])
        for series_index, series in enumerate(panel.series):
            target_axis = secondary_axis if series.axis == "secondary" else axis
            if target_axis is None:
                raise RuntimeError("Secondary axis was requested but not created.")

            aggregate, seed_frames = _aggregate_run_series(
                run,
                panel,
                series,
                train_window_size=spec.train_window_size,
                train_bin_size=spec.train_bin_size,
            )
            linestyle = series.linestyle or _LINESTYLES[series_index % len(_LINESTYLES)]
            label = _series_label(run, panel, series)
            _draw_series(
                target_axis,
                panel=panel,
                series=series,
                aggregate=aggregate,
                seed_frames=seed_frames,
                color=color,
                linestyle=linestyle,
                label=label,
                show_seeds=spec.show_seeds,
                show_mean=spec.show_mean,
                show_std=spec.show_std,
                mean_linewidth=spec.style.mean_linewidth,
                seed_linewidth=spec.style.seed_linewidth,
                seed_alpha=spec.style.seed_alpha,
                std_alpha=spec.style.std_alpha,
            )

    _configure_panel_axes(
        axis,
        panel,
        spec=spec,
        show_title=show_panel_title,
    )
    primary_entries, secondary_entries = _combined_legend_entries(
        axis,
        secondary_axis,
        panel,
        spec=spec,
    )
    primary_handles, primary_labels = primary_entries
    secondary_handles, secondary_labels = secondary_entries
    handles = primary_handles + secondary_handles
    labels = primary_labels + secondary_labels

    if handles and show_legend:
        axis.legend(
            handles,
            labels,
            loc="best",
            fontsize=spec.style.legend_fontsize,
        )
    return primary_entries, secondary_entries


def _axis_legend_title(
    panel: PanelSpec,
    axis_name: str,
) -> str | None:
    axis_series = tuple(series for series in panel.series if series.axis == axis_name)
    if len(axis_series) != 1:
        return None

    if axis_name == "primary" and panel.y_label:
        return panel.y_label

    if axis_name == "secondary" and panel.secondary_y_label:
        return panel.secondary_y_label

    series = axis_series[0]
    return _humanise_metric_name(series.label or series.column)


def _shared_axis_legend_title(
    panels: Sequence[PanelSpec],
    axis_name: str,
) -> str | None:
    titles = {
        title
        for panel in panels
        if (title := _axis_legend_title(panel, axis_name)) is not None
    }
    if len(titles) != 1:
        return None

    return next(iter(titles))


def _draw_shared_legends(
    figure: Figure,
    spec: FigureSpec,
    *,
    primary_handles: Sequence[Artist],
    primary_labels: Sequence[str],
    secondary_handles: Sequence[Artist],
    secondary_labels: Sequence[str],
    primary_title: str | None = None,
    secondary_title: str | None = None,
) -> int:
    def _draw_row(
        handles: Sequence[Artist],
        labels: Sequence[str],
        *,
        title: str | None,
        y: float,
    ) -> None:
        row_handles: list[Artist] = list(handles)
        row_labels = list(labels)

        if title:
            # Invisible handle for the inline row title only.
            title_handle = Line2D(
                [],
                [],
                linestyle="none",
                marker="",
                color="none",
            )
            row_handles.insert(0, title_handle)
            row_labels.insert(0, f"{title}:")

        legend = figure.legend(
            row_handles,
            row_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, y),
            ncol=max(1, len(row_labels)),
            frameon=False,
            fontsize=spec.style.legend_fontsize,
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=1.2,
        )

        # Collapse only the dummy title handle's allocated width.
        if title:
            legend.legend_handles[0].set_visible(False)

    row_count = 0
    has_primary = bool(primary_handles)
    has_secondary = bool(secondary_handles)
    base_y = spec.style.legend_y
    row_step = spec.style.legend_row_step

    if has_primary and has_secondary:
        _draw_row(
            primary_handles,
            primary_labels,
            title=primary_title,
            y=base_y + row_step,
        )
        _draw_row(
            secondary_handles,
            secondary_labels,
            title=secondary_title,
            y=base_y,
        )
        return 2

    if has_primary:
        _draw_row(
            primary_handles,
            primary_labels,
            title=primary_title,
            y=base_y,
        )
        row_count += 1

    if has_secondary:
        _draw_row(
            secondary_handles,
            secondary_labels,
            title=secondary_title,
            y=base_y,
        )
        row_count += 1

    return row_count


def _legend_bottom_margin(
    spec: FigureSpec,
    row_count: int,
) -> float:
    if row_count <= 0:
        return 0.0

    return min(
        0.45,
        spec.style.legend_bottom + spec.style.legend_margin_per_row * (row_count - 1),
    )


def _share_repeated_axis_labels(
    axes: np.ndarray,
    *,
    used_count: int,
    rows: int,
    columns: int,
) -> None:
    """Show repeated x labels only on the bottom row and y labels on the first column."""
    used_axes = list(axes.flat)[:used_count]
    if len(used_axes) < 2:
        return

    x_labels = {axis.get_xlabel() for axis in used_axes}
    y_labels = {axis.get_ylabel() for axis in used_axes}
    shared_x = len(x_labels) == 1
    shared_y = len(y_labels) == 1

    for index, axis in enumerate(used_axes):
        row, column = divmod(index, columns)
        if shared_x and row < rows - 1:
            axis.set_xlabel("")
        if shared_y and column > 0:
            axis.set_ylabel("")


def _render_subplot_grid(
    *,
    spec: FigureSpec,
    rows: int,
    columns: int,
    plot_items: Sequence[tuple[PlotTask, PanelSpec, bool]],
    grid_title: str | None,
    color_by_comparison: Mapping[str, str],
    use_shared_legend: bool,
    primary_legend_title: str | None,
    secondary_legend_title: str | None,
    sharex: bool = False,
    sharey: bool = False,
) -> Figure:
    """Render a configured collection of task/panel pairs into one subplot grid."""
    item_count = len(plot_items)
    if item_count < 1:
        raise ValueError("At least one plot item is required.")

    figure_width, figure_height = _figure_size(
        spec,
        rows=rows,
        columns=columns,
    )
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(figure_width, figure_height),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )

    primary_shared_handles: list[Artist] = []
    primary_shared_labels: list[str] = []
    secondary_shared_handles: list[Artist] = []
    secondary_shared_labels: list[str] = []

    for (task, panel, show_panel_title), axis in zip(
        plot_items,
        axes.flat,
        strict=False,
    ):
        primary_entries, secondary_entries = _plot_panel(
            axis,
            task,
            spec,
            panel,
            show_legend=not use_shared_legend,
            color_by_comparison=color_by_comparison,
            show_panel_title=show_panel_title,
        )

        if not use_shared_legend:
            continue

        primary_handles, primary_labels = primary_entries
        secondary_handles, secondary_labels = secondary_entries

        _merge_legend_entries(
            primary_shared_handles,
            primary_shared_labels,
            primary_handles,
            primary_labels,
        )
        _merge_legend_entries(
            secondary_shared_handles,
            secondary_shared_labels,
            secondary_handles,
            secondary_labels,
        )

    _remove_unused_axes(axes, item_count)

    _share_repeated_axis_labels(
        axes,
        used_count=item_count,
        rows=rows,
        columns=columns,
    )

    legend_rows = 0
    if use_shared_legend:
        legend_rows = _draw_shared_legends(
            figure,
            spec,
            primary_handles=primary_shared_handles,
            primary_labels=primary_shared_labels,
            secondary_handles=secondary_shared_handles,
            secondary_labels=secondary_shared_labels,
            primary_title=primary_legend_title,
            secondary_title=secondary_legend_title,
        )

    if grid_title:
        figure.suptitle(
            grid_title,
            fontsize=spec.style.title_fontsize + 2,
            y=spec.style.title_y,
        )

    bottom = _legend_bottom_margin(spec, legend_rows)
    top = spec.style.top_rect if grid_title else 1.0

    figure.tight_layout(
        rect=(0.0, bottom, 1.0, top),
        pad=spec.style.layout_pad,
        w_pad=spec.style.layout_w_pad,
        h_pad=spec.style.layout_h_pad,
    )

    return figure


def render_task(
    task: PlotTask,
    spec: FigureSpec,
) -> Figure:
    """Render one figure for a single task."""
    rows, columns = _grid_shape(
        len(spec.panels),
        rows=spec.rows,
        columns=spec.columns,
    )

    multiple_panels = len(spec.panels) > 1

    plot_items = tuple(
        (
            task,
            panel,
            multiple_panels and spec.show_panel_titles,
        )
        for panel in spec.panels
    )

    return _render_subplot_grid(
        spec=spec,
        rows=rows,
        columns=columns,
        plot_items=plot_items,
        grid_title=spec.title or task.name,
        color_by_comparison=_comparison_colors((task,)),
        use_shared_legend=multiple_panels,
        primary_legend_title=(
            _shared_axis_legend_title(spec.panels, "primary")
            if multiple_panels
            else None
        ),
        secondary_legend_title=(
            _shared_axis_legend_title(spec.panels, "secondary")
            if multiple_panels
            else None
        ),
    )


def _figure_size(
    spec: FigureSpec,
    *,
    rows: int,
    columns: int,
) -> tuple[float, float]:
    """Resolve total figure size from its grid geometry."""
    row_height = spec.style.row_height

    if row_height is None:
        panel_width = spec.style.figure_width / columns
        panel_aspect = spec.style.axes_box_aspect or 1.0
        row_height = panel_width * panel_aspect

    return (
        spec.style.figure_width,
        row_height * rows,
    )


def _comparison_colors(tasks: Sequence[PlotTask]) -> dict[str, str]:
    """Assign one stable color per comparison across all supplied tasks."""
    names = sorted({run.name for task in tasks for run in task.runs})
    color_cycle = _color_cycle(len(names))
    return {
        name: color_cycle[index % len(color_cycle)] for index, name in enumerate(names)
    }


def render_tasks(
    tasks: Sequence[PlotTask],
    *,
    spec: FigureSpec,
) -> Figure:
    """Render one plot across several tasks, with one subplot per task."""
    ordered_tasks = tuple(tasks)

    if len(spec.panels) != 1:
        raise ValueError("Combined task figures must contain exactly one panel.")

    rows, columns = _grid_shape(
        len(ordered_tasks),
        rows=spec.rows,
        columns=spec.columns,
    )

    panel = spec.panels[0]

    plot_items = tuple(
        (
            task,
            dataclasses.replace(panel, title=task.name),
            True,
        )
        for task in ordered_tasks
    )

    return _render_subplot_grid(
        spec=spec,
        rows=rows,
        columns=columns,
        plot_items=plot_items,
        grid_title=spec.title,
        color_by_comparison=_comparison_colors(ordered_tasks),
        use_shared_legend=True,
        primary_legend_title=_axis_legend_title(panel, "primary"),
        secondary_legend_title=_axis_legend_title(panel, "secondary"),
        sharex=False,
        sharey=False,
    )


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-._") or "plot"


def save_figure(
    figure: Figure,
    output_directory: pathlib.Path,
    name: str,
    *,
    formats: Iterable[str] = ("png",),
    dpi: int = 300,
) -> tuple[pathlib.Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(name)
    paths: list[pathlib.Path] = []
    for raw_format in formats:
        file_format = raw_format.lower().lstrip(".")
        if not file_format:
            raise ValueError("Output format must not be empty.")
        path = output_directory / f"{stem}.{file_format}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return tuple(paths)
