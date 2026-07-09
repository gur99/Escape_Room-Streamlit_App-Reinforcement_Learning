"""Approximate RL skeleton used for future extensions and Room 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algorithms.base import RLAlgorithmBase


@dataclass
class ApproximateRLConfig:
    """Generic configuration for future approximate RL experiments."""

    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    episodes: int = 300
    max_steps: int = 500


class ApproximateRLAgent(RLAlgorithmBase):
    """Placeholder for future function-approximation experiments."""

    algorithm_name = "Approximate Reinforcement Learning"
    learning_type = "Function approximation"
    algorithm_summary = (
        "This module reserves space for future approximate RL variations beyond DQN."
    )

    def __init__(self, config: ApproximateRLConfig | None = None) -> None:
        self.config = config or ApproximateRLConfig()

    def train(self, environment: Any) -> dict[str, Any]:
        """Return a descriptive placeholder instead of training."""

        raise NotImplementedError(
            "Approximate RL training will be implemented in a later stage."
        )

    def predict(self, observation: Any) -> int:
        """Return a deterministic placeholder action."""

        return 0
