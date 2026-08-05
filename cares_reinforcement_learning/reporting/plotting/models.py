from __future__ import annotations

import dataclasses
from typing import Literal

PlotSource = Literal["train", "eval"]
PlotAxis = Literal["primary", "secondary"]
PlotScale = Literal["linear", "log"]


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
    title: str | None = None
    rows: int | None = None
    columns: int | None = None
    width: float = 12.0
    height: float = 5.0
    label_fontsize: int = 13
    title_fontsize: int = 16
    ticks_fontsize: int = 10
    legend_fontsize: int = 9
    dpi: int = 300

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
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Figure width and height must be positive.")
        if self.dpi < 72:
            raise ValueError("Figure DPI must be at least 72.")


def default_figure_specs() -> tuple[tuple[str, FigureSpec], ...]:
    """Return the two independent default learning-curve figures."""
    return (
        (
            "train",
            FigureSpec(
                panels=(
                    PanelSpec(
                        source="train",
                        x="total_steps",
                        series=(SeriesSpec("episode_reward"),),
                        title="Training reward",
                        x_label="Steps",
                        y_label="Reward",
                    ),
                ),
                rows=1,
                columns=1,
            ),
        ),
        (
            "eval",
            FigureSpec(
                panels=(
                    PanelSpec(
                        source="eval",
                        x="total_steps",
                        series=(SeriesSpec("episode_reward"),),
                        title="Evaluation reward",
                        x_label="Steps",
                        y_label="Reward",
                    ),
                ),
                rows=1,
                columns=1,
            ),
        ),
    )
