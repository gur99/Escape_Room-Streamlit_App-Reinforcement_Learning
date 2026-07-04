"""SARSA algorithm skeleton for Room 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algorithms.base import RLAlgorithmBase


@dataclass
class SarsaConfig:
    """Hyperparameters for the SARSA algorithm."""

    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    episodes: int = 500
    max_steps: int = 200


class SarsaAgent(RLAlgorithmBase):
    """Placeholder for an on-policy SARSA implementation."""

    algorithm_name = "SARSA"
    learning_type = "On-policy temporal-difference learning"
    algorithm_summary = (
        "This module will later host the tabular SARSA implementation for Room 2."
    )

    def __init__(self, config: SarsaConfig | None = None) -> None:
        self.config = config or SarsaConfig()

    def train(self, environment: Any) -> dict[str, Any]:
        """Return a descriptive placeholder instead of training."""

        raise NotImplementedError("SARSA training will be implemented in a later stage.")

    def predict(self, observation: Any) -> int:
        """Return a deterministic placeholder action."""

        return 0
