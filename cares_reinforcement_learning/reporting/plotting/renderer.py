from __future__ import annotations

import dataclasses
import math
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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


def _panel_title(panel: PanelSpec) -> str:
    if panel.title:
        return panel.title
    metric_names = ", ".join(series.label or series.column for series in panel.series)
    return f"{panel.source.title()}: {metric_names}"


def _series_label(run: PlotRun, panel: PanelSpec, series: SeriesSpec) -> str:
    if len(panel.series) == 1:
        return run.name
    return f"{run.name} — {series.label or series.column}"


def _grid_shape(spec: FigureSpec) -> tuple[int, int]:
    count = len(spec.panels)
    if spec.rows is not None and spec.columns is not None:
        return spec.rows, spec.columns
    if spec.rows is not None:
        return spec.rows, math.ceil(count / spec.rows)
    if spec.columns is not None:
        return math.ceil(count / spec.columns), spec.columns
    columns = min(2, count)
    return math.ceil(count / columns), columns


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
) -> None:
    if show_seeds:
        for _, seed_frame in seed_frames:
            target_axis.plot(
                seed_frame[panel.x],
                seed_frame[series.column],
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=0.18,
            )

    if show_mean:
        target_axis.plot(
            aggregate[panel.x],
            aggregate["mean"],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
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
                alpha=0.16,
                linewidth=0,
            )


def _configure_panel_axes(
    axis: Axes,
    panel: PanelSpec,
    *,
    ticks_fontsize: int,
    label_fontsize: int,
    title_fontsize: int,
) -> None:
    axis.set_title(_panel_title(panel), fontsize=title_fontsize)
    axis.set_xlabel(panel.x_label or panel.x, fontsize=label_fontsize)
    axis.set_ylabel(panel.y_label or panel.series[0].column, fontsize=label_fontsize)
    axis.set_xscale(panel.x_scale)
    axis.set_yscale(panel.y_scale)
    axis.tick_params(axis="both", labelsize=ticks_fontsize)
    axis.grid(True, alpha=0.25)


def _combined_legend_entries(
    axis: Axes,
    secondary_axis: Axes | None,
    panel: PanelSpec,
    *,
    ticks_fontsize: int,
    label_fontsize: int,
) -> tuple[list[Artist], list[str]]:
    handles, labels = axis.get_legend_handles_labels()
    if secondary_axis is None:
        return handles, labels

    secondary_axis.set_ylabel(
        panel.secondary_y_label
        or next(series.column for series in panel.series if series.axis == "secondary"),
        fontsize=label_fontsize,
    )
    secondary_axis.set_yscale(panel.secondary_y_scale)
    secondary_axis.tick_params(axis="y", labelsize=ticks_fontsize)
    handles_two, labels_two = secondary_axis.get_legend_handles_labels()
    return handles + handles_two, labels + labels_two


def _remove_unused_axes(axes: np.ndarray, used_count: int) -> None:
    for unused_axis in list(axes.flat)[used_count:]:
        unused_axis.remove()


def _unique_legend_entries(axes: np.ndarray) -> tuple[list[Artist], list[str]]:
    handles: list[Artist] = []
    labels: list[str] = []
    for axis in axes.flat:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    return handles, labels


def _plot_panel(
    axis: Axes,
    task: PlotTask,
    panel: PanelSpec,
    *,
    train_window_size: int,
    train_bin_size: float | None,
    show_seeds: bool,
    show_mean: bool,
    show_std: bool,
    ticks_fontsize: int,
    label_fontsize: int,
    title_fontsize: int,
    legend_fontsize: int,
    show_legend: bool = True,
    color_by_comparison: Mapping[str, str] | None = None,
) -> None:
    """Render one panel including optional secondary axis and legend entries."""
    _validate_panel(task, panel)
    secondary_axis: Axes | None = None
    if any(series.axis == "secondary" for series in panel.series):
        secondary_axis = axis.twinx()

    if color_by_comparison is None:
        cycle = _color_cycle(len(task.runs))
        color_by_comparison = {
            run.name: cycle[index % len(cycle)] for index, run in enumerate(task.runs)
        }

    for run_index, run in enumerate(task.runs):
        color = color_by_comparison.get(run.name)
        if color is None:
            cycle = _color_cycle(len(task.runs))
            color = cycle[run_index % len(cycle)]
        for series_index, series in enumerate(panel.series):
            target_axis = secondary_axis if series.axis == "secondary" else axis
            if target_axis is None:
                raise RuntimeError("Secondary axis was requested but not created.")

            aggregate, seed_frames = _aggregate_run_series(
                run,
                panel,
                series,
                train_window_size=train_window_size,
                train_bin_size=train_bin_size,
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
                show_seeds=show_seeds,
                show_mean=show_mean,
                show_std=show_std,
            )

    _configure_panel_axes(
        axis,
        panel,
        ticks_fontsize=ticks_fontsize,
        label_fontsize=label_fontsize,
        title_fontsize=title_fontsize,
    )
    handles, labels = _combined_legend_entries(
        axis,
        secondary_axis,
        panel,
        ticks_fontsize=ticks_fontsize,
        label_fontsize=label_fontsize,
    )

    if handles and show_legend:
        axis.legend(
            handles,
            labels,
            loc="best",
            fontsize=legend_fontsize,
        )


def render_task(
    task: PlotTask,
    spec: FigureSpec,
    *,
    train_window_size: int = 20,
    train_bin_size: float | None = None,
    show_seeds: bool = False,
    show_mean: bool = True,
    show_std: bool = True,
) -> Figure:
    """Render one figure for a single task."""
    rows, columns = _grid_shape(spec)
    colors = _comparison_colors((task,))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(spec.width, spec.height * rows),
        squeeze=False,
    )

    for panel, axis in zip(spec.panels, axes.flat):
        _plot_panel(
            axis,
            task,
            panel,
            train_window_size=train_window_size,
            train_bin_size=train_bin_size,
            show_seeds=show_seeds,
            show_mean=show_mean,
            show_std=show_std,
            ticks_fontsize=spec.ticks_fontsize,
            label_fontsize=spec.label_fontsize,
            title_fontsize=spec.title_fontsize,
            legend_fontsize=spec.legend_fontsize,
            show_legend=True,
            color_by_comparison=colors,
        )

    _remove_unused_axes(axes, len(spec.panels))

    figure.suptitle(spec.title or task.name, fontsize=spec.title_fontsize + 2)
    figure.tight_layout()
    return figure


def _tasks_grid_shape(task_count: int, columns: int | None) -> tuple[int, int]:
    if task_count < 1:
        raise ValueError("At least one task is required for a combined task figure.")
    if columns is not None:
        if columns < 1:
            raise ValueError("columns must be positive when supplied.")
        resolved_columns = min(columns, task_count)
    else:
        resolved_columns = min(4, math.ceil(math.sqrt(task_count)))
    return math.ceil(task_count / resolved_columns), resolved_columns


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
    plot: PanelSpec,
    title: str | None = None,
    columns: int | None = None,
    panel_width: float = 3.2,
    panel_height: float = 2.45,
    train_window_size: int = 20,
    train_bin_size: float | None = None,
    show_seeds: bool = False,
    show_mean: bool = True,
    show_std: bool = True,
    label_fontsize: int = 9,
    title_fontsize: int = 10,
    ticks_fontsize: int = 8,
    legend_fontsize: int = 9,
) -> Figure:
    """Render one plot across several tasks, with one subplot per task."""
    ordered_tasks = tuple(tasks)
    rows, resolved_columns = _tasks_grid_shape(len(ordered_tasks), columns)
    figure, axes = plt.subplots(
        rows,
        resolved_columns,
        figsize=(panel_width * resolved_columns, panel_height * rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    colors = _comparison_colors(ordered_tasks)

    for task, axis in zip(ordered_tasks, axes.flat, strict=False):
        task_plot = dataclasses.replace(plot, title=task.name)
        _plot_panel(
            axis,
            task,
            task_plot,
            train_window_size=train_window_size,
            train_bin_size=train_bin_size,
            show_seeds=show_seeds,
            show_mean=show_mean,
            show_std=show_std,
            ticks_fontsize=ticks_fontsize,
            label_fontsize=label_fontsize,
            title_fontsize=title_fontsize,
            legend_fontsize=legend_fontsize,
            show_legend=False,
            color_by_comparison=colors,
        )

    _remove_unused_axes(axes, len(ordered_tasks))

    handles, labels = _unique_legend_entries(axes)

    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(len(labels), max(1, resolved_columns)),
            frameon=False,
            fontsize=legend_fontsize,
        )

    figure.suptitle(
        title or _panel_title(plot),
        fontsize=title_fontsize + 1,
        y=0.995,
    )
    legend_rows = math.ceil(max(1, len(labels)) / max(1, resolved_columns))
    bottom = min(0.30, 0.09 + 0.035 * legend_rows)
    figure.tight_layout(
        rect=(0.0, bottom, 1.0, 0.96),
        pad=0.6,
        w_pad=0.7,
        h_pad=0.8,
    )
    return figure


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
