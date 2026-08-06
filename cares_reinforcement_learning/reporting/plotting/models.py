from __future__ import annotations

import dataclasses
from typing import Literal

PlotSource = Literal["train", "eval"]
PlotAxis = Literal["primary", "secondary"]
PlotScale = Literal["linear", "log"]


@dataclasses.dataclass(frozen=True)
class PlotStyleSpec:
    figure_width: float
    row_height: float | None
    label_fontsize: int
    title_fontsize: int
    ticks_fontsize: int
    legend_fontsize: int
    mean_linewidth: float
    seed_linewidth: float
    seed_alpha: float
    std_alpha: float
    grid_alpha: float
    layout_pad: float
    layout_w_pad: float
    layout_h_pad: float
    title_y: float
    top_rect: float
    legend_bottom: float
    legend_y: float
    legend_row_step: float
    legend_margin_per_row: float
    compact_step_axis: bool
    axes_box_aspect: float | None

    def __post_init__(self) -> None:
        for value, name in (
            (self.figure_width, "figure_width"),
            (self.label_fontsize, "label_fontsize"),
            (self.title_fontsize, "title_fontsize"),
            (self.ticks_fontsize, "ticks_fontsize"),
            (self.legend_fontsize, "legend_fontsize"),
            (self.mean_linewidth, "mean_linewidth"),
            (self.seed_linewidth, "seed_linewidth"),
            (self.legend_row_step, "legend_row_step"),
            (self.legend_margin_per_row, "legend_margin_per_row"),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        if self.row_height is not None and self.row_height <= 0:
            raise ValueError("row_height must be positive when supplied.")

        for value, name in (
            (self.seed_alpha, "seed_alpha"),
            (self.std_alpha, "std_alpha"),
            (self.grid_alpha, "grid_alpha"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1].")

        for value, name in (
            (self.layout_pad, "layout_pad"),
            (self.layout_w_pad, "layout_w_pad"),
            (self.layout_h_pad, "layout_h_pad"),
            (self.title_y, "title_y"),
            (self.top_rect, "top_rect"),
            (self.legend_bottom, "legend_bottom"),
            (self.legend_y, "legend_y"),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative.")

        if self.axes_box_aspect is not None and self.axes_box_aspect <= 0:
            raise ValueError("axes_box_aspect must be positive when supplied.")


def plot_style_spec(*, compact: bool) -> PlotStyleSpec:
    if compact:
        return PlotStyleSpec(
            figure_width=6.8,
            row_height=None,
            label_fontsize=7,
            title_fontsize=8,
            ticks_fontsize=6,
            legend_fontsize=6,
            mean_linewidth=1.2,
            seed_linewidth=0.55,
            seed_alpha=0.09,
            std_alpha=0.11,
            grid_alpha=0.16,
            layout_pad=0.15,
            layout_w_pad=0.22,
            layout_h_pad=0.28,
            title_y=0.985,
            top_rect=0.95,
            legend_bottom=0.065,
            legend_y=0.002,
            legend_row_step=0.04,
            legend_margin_per_row=0.055,
            compact_step_axis=True,
            axes_box_aspect=1.0,
        )
    return PlotStyleSpec(
        figure_width=7.0,
        row_height=2.2,
        label_fontsize=8,
        title_fontsize=10,
        ticks_fontsize=7,
        legend_fontsize=7,
        mean_linewidth=1.5,
        seed_linewidth=0.65,
        seed_alpha=0.10,
        std_alpha=0.12,
        grid_alpha=0.18,
        layout_pad=0.25,
        layout_w_pad=0.35,
        layout_h_pad=0.45,
        title_y=0.985,
        top_rect=0.94,
        legend_bottom=0.075,
        legend_y=0.005,
        legend_row_step=0.04,
        legend_margin_per_row=0.055,
        compact_step_axis=True,
        axes_box_aspect=None,
    )


@dataclasses.dataclass(frozen=True)
class SeriesSpec:
    column: str
    axis: PlotAxis = "primary"
    label: str | None = None
    linestyle: str | None = None

    def __post_init__(self) -> None:
        if not self.column.strip():
            raise ValueError("Series column must not be empty.")


@dataclasses.dataclass(frozen=True)
class PanelSpec:
    source: PlotSource
    series: tuple[SeriesSpec, ...]
    x: str = "total_steps"
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    secondary_y_label: str | None = None
    x_scale: PlotScale = "linear"
    y_scale: PlotScale = "linear"
    secondary_y_scale: PlotScale = "linear"

    def __post_init__(self) -> None:
        if not self.x.strip():
            raise ValueError("Panel x column must not be empty.")
        if not self.series:
            raise ValueError("Each panel must contain at least one series.")
        has_secondary = any(series.axis == "secondary" for series in self.series)
        has_primary = any(series.axis == "primary" for series in self.series)
        if has_secondary and not has_primary:
            raise ValueError(
                "A dual-axis panel must contain at least one primary-axis series."
            )


@dataclasses.dataclass(frozen=True)
class FigureSpec:
    panels: tuple[PanelSpec, ...]
    style: PlotStyleSpec
    title: str | None = None
    rows: int | None = None
    columns: int | None = None
    dpi: int = 300
    train_window_size: int = 20
    train_bin_size: float | None = None
    show_seeds: bool = False
    show_mean: bool = True
    show_std: bool = True
    show_panel_titles: bool = True

    def __post_init__(self) -> None:
        if not self.panels:
            raise ValueError("A figure must contain at least one panel.")
        for value, name in ((self.rows, "rows"), (self.columns, "columns")):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when supplied.")
        if self.rows is not None and self.columns is not None:
            if self.rows * self.columns < len(self.panels):
                raise ValueError(
                    "rows × columns is smaller than the number of configured panels."
                )
        if self.dpi < 72:
            raise ValueError("Figure DPI must be at least 72.")
        if self.train_window_size < 1:
            raise ValueError("train_window_size must be at least 1.")
        if self.train_bin_size is not None and self.train_bin_size <= 0:
            raise ValueError("train_bin_size must be positive when supplied.")


def default_panels() -> tuple[tuple[str, PanelSpec], ...]:
    """Return default panel definitions without binding figure-level style."""
    return (
        (
            "train",
            PanelSpec(
                source="train",
                x="total_steps",
                series=(SeriesSpec("episode_reward"),),
                x_label="Steps",
                y_label="Reward",
            ),
        ),
        (
            "eval",
            PanelSpec(
                source="eval",
                x="total_steps",
                series=(SeriesSpec("episode_reward"),),
                x_label="Steps",
                y_label="Reward",
            ),
        ),
    )


def default_figure_specs(
    *,
    compact: bool,
) -> tuple[tuple[str, FigureSpec], ...]:
    """Return the two independent default learning-curve figures."""
    preset = plot_style_spec(compact=compact)
    return tuple(
        (
            suffix,
            FigureSpec(
                panels=(panel,),
                rows=1,
                columns=1,
                style=preset,
                show_panel_titles=False,
            ),
        )
        for suffix, panel in default_panels()
    )
