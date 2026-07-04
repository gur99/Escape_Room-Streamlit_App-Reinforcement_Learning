"""Dynamic Programming algorithm skeleton for Room 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algorithms.base import RLAlgorithmBase


@dataclass
class DynamicProgrammingConfig:
    """Hyperparameters for Dynamic Programming methods."""

    gamma: float = 0.99
    theta: float = 0.0001
    max_iterations: int = 500


class DynamicProgrammingAgent(RLAlgorithmBase):
    """Placeholder for Value Iteration or Policy Iteration."""

    algorithm_name = "Dynamic Programming"
    learning_type = "Planning with a known model"
    algorithm_summary = (
        "This module will later host Value Iteration and Policy Iteration for Room 1."
    )

    def __init__(self, config: DynamicProgrammingConfig | None = None) -> None:
        self.config = config or DynamicProgrammingConfig()

    def train(self, environment: Any) -> dict[str, Any]:
        """Return a descriptive placeholder instead of running planning logic."""

        raise NotImplementedError(
            "Dynamic Programming training will be implemented in the next stage."
        )

    def predict(self, observation: Any) -> int:
        """Return a deterministic placeholder action."""

        return 0
