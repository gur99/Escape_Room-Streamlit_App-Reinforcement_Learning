"""Shared utilities for the RL Escape Room project."""

from utils.metrics import RunSummary, TrainingHistory
from utils.plotting import build_metrics_dataframe, render_placeholder_metric
from utils.replay import (
    EpisodeReplay,
    ReplayStep,
    SarsaEpisodeRecord,
    build_placeholder_replay,
    select_evenly_spaced_indices,
    select_most_recent_indices,
)
from utils.spaces import BoxSpace, CompositeSpace, DiscreteSpace

__all__ = [
    "BoxSpace",
    "CompositeSpace",
    "DiscreteSpace",
    "EpisodeReplay",
    "ReplayStep",
    "SarsaEpisodeRecord",
    "RunSummary",
    "TrainingHistory",
    "build_metrics_dataframe",
    "build_placeholder_replay",
    "render_placeholder_metric",
    "select_evenly_spaced_indices",
    "select_most_recent_indices",
]
