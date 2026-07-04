"""Q-Learning algorithm skeleton for Room 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algorithms.base import RLAlgorithmBase


@dataclass
class QLearningConfig:
    """Hyperparameters for Q-Learning."""

    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    episodes: int = 750
    max_steps: int = 250


class QLearningAgent(RLAlgorithmBase):
    """Placeholder for an off-policy Q-Learning implementation."""

    algorithm_name = "Q-Learning"
    learning_type = "Off-policy temporal-difference learning"
    algorithm_summary = (
        "This module will later host the tabular Q-Learning implementation for Room 3."
    )

    def __init__(self, config: QLearningConfig | None = None) -> None:
        self.config = config or QLearningConfig()

    def train(self, environment: Any) -> dict[str, Any]:
        """Return a descriptive placeholder instead of training."""

        raise NotImplementedError(
            "Q-Learning training will be implemented in a later stage."
        )

    def predict(self, observation: Any) -> int:
        """Return a deterministic placeholder action."""

        return 0
