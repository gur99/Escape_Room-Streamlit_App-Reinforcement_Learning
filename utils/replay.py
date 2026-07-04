"""Replay data structures for visualizing trained episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReplayStep:
    """One step in an episode replay."""

    observation: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class EpisodeReplay:
    """Container for a full replayable episode."""

    room_name: str
    algorithm_name: str
    steps: list[ReplayStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the replay to a JSON-friendly dictionary."""

        return {
            "room_name": self.room_name,
            "algorithm_name": self.algorithm_name,
            "steps": [asdict(step) for step in self.steps],
        }


def build_placeholder_replay() -> EpisodeReplay:
    """Return an empty replay object used by the Stage 1 UI."""

    return EpisodeReplay(
        room_name="Stage 1 Skeleton",
        algorithm_name="Not Implemented Yet",
        steps=[],
    )
