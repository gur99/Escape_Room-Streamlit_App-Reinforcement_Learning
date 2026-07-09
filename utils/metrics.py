"""Data containers for training metrics and comparison results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingHistory:
    """Stores common RL metrics collected during training."""

    rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    success_rates: list[float] = field(default_factory=list)
    exploration_rates: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)


@dataclass
class RunSummary:
    """Represents one saved training run for future comparisons."""

    run_name: str
    room_name: str
    algorithm_name: str
    notes: str = ""
