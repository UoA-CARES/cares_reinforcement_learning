from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

METRIC_GROUP_COLUMNS = ["evaluation_metric", "performance_metric"]
PERFORMANCE_METRICS = ("auc", "early_window_auc", "final_window_auc")


def ensure_required_columns(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    context: str,
) -> pd.DataFrame:
    columns = tuple(required_columns)
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")
    return frame.loc[:, columns].copy()


def aggregate_evaluation_curve(
    frame: pd.DataFrame,
    metric_column: str,
    step_column: str = "total_steps",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    grouped = frame.groupby(step_column, sort=True, observed=True)[metric_column].mean()
    steps = grouped.index.to_numpy(dtype=np.float64)
    values = grouped.to_numpy(dtype=np.float64)
    if steps.size < 2:
        raise ValueError("At least two distinct evaluation steps are required for AUC.")
    return steps, values


def _slice_curve(
    steps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    start: float,
    end: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if not steps[0] <= start < end <= steps[-1]:
        raise ValueError(
            "Requested AUC window lies outside the observed evaluation interval."
        )
    inside = (steps > start) & (steps < end)
    window_steps = np.concatenate(([start], steps[inside], [end])).astype(np.float64)
    window_values = np.interp(window_steps, steps, values).astype(np.float64)
    return window_steps, window_values


def _trapezoidal_auc(
    steps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> float:
    return float(np.trapz(values, x=steps))


def _window_auc(
    steps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    start: float,
    end: float,
) -> float:
    window_steps, window_values = _slice_curve(steps, values, start, end)
    return _trapezoidal_auc(window_steps, window_values)


def compute_curve_metrics(
    steps: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    early_fraction: float,
    final_fraction: float,
) -> dict[str, float]:
    start = float(steps[0])
    end = float(steps[-1])
    width = end - start

    early_end = start + early_fraction * width
    final_start = end - final_fraction * width

    return {
        "auc": _trapezoidal_auc(steps, values),
        "early_window_auc": _window_auc(steps, values, start, early_end),
        "final_window_auc": _window_auc(steps, values, final_start, end),
    }
