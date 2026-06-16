from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def find_step_column(data: pd.DataFrame) -> str:
    candidates = [
        "total_steps",
        "step",
        "steps",
        "env_steps",
        "episode_step",
        "episode",
    ]
    for candidate in candidates:
        if candidate in data.columns and pd.api.types.is_numeric_dtype(data[candidate]):
            return candidate
    raise ValueError(
        "No valid x-axis column found. Expected one of: total_steps, step, steps, "
        "env_steps, episode_step, episode."
    )
