"""DQN algorithm skeleton for Room 4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algorithms.base import RLAlgorithmBase


@dataclass
class DQNConfig:
    """Hyperparameters for a future DQN implementation."""

    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    batch_size: int = 64
    replay_buffer_size: int = 10000
    target_update_frequency: int = 100
    update_frequency: int = 4
    episodes: int = 300
    max_steps: int = 500


class DQNAgent(RLAlgorithmBase):
    """Placeholder for a PyTorch-based DQN agent."""

    algorithm_name = "Deep Q-Network (DQN)"
    learning_type = "Value-based Deep Reinforcement Learning"
    algorithm_summary = (
        "This module will later host the PyTorch DQN implementation for Room 4."
    )

    def __init__(self, config: DQNConfig | None = None) -> None:
        self.config = config or DQNConfig()

    def train(self, environment: Any) -> dict[str, Any]:
        """Return a descriptive placeholder instead of training."""

        raise NotImplementedError("DQN training will be implemented in a later stage.")

    def predict(self, observation: Any) -> int:
        """Return a deterministic placeholder action."""

        return 0
