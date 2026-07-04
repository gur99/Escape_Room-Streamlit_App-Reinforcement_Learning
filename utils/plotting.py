"""Plotting helpers for Streamlit visualizations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.metrics import TrainingHistory


def build_metrics_dataframe(history: TrainingHistory) -> pd.DataFrame:
    """Convert collected metrics into a tabular format for plotting.

    The function returns an empty dataframe when no training has been performed
    yet, which is the expected behavior during Stage 1.
    """

    series_map = {
        "reward": history.rewards,
        "episode_length": history.episode_lengths,
        "success_rate": history.success_rates,
        "exploration_rate": history.exploration_rates,
        "loss": history.losses,
    }

    non_empty_series = {name: values for name, values in series_map.items() if values}
    if not non_empty_series:
        return pd.DataFrame()

    return pd.DataFrame(dict(non_empty_series))


def render_placeholder_metric(container: Any, label: str, values: list[float]) -> None:
    """Render a single compact metric summary in a Streamlit column."""

    if values:
        container.metric(label=label, value=f"{values[-1]:.3f}")
    else:
        container.metric(label=label, value="N/A")
