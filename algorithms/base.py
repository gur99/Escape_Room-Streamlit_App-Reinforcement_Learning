"""Common algorithm interfaces for the RL Escape Room project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any


class RLAlgorithmBase(ABC):
    """Abstract base class shared by all learning algorithms in the project."""

    algorithm_name: str = "Unnamed Algorithm"
    learning_type: str = "Unknown"
    algorithm_summary: str = "Algorithm summary is not available yet."

    config: Any

    def config_as_dict(self) -> dict[str, Any]:
        """Expose hyperparameters in a UI-friendly format."""

        return asdict(self.config)

    @abstractmethod
    def train(self, environment: Any) -> dict[str, Any]:
        """Train on the supplied environment."""

    @abstractmethod
    def predict(self, observation: Any) -> Any:
        """Choose an action for an observation."""

    def save(self, path: str) -> None:
        """Placeholder persistence hook for later stages."""

        raise NotImplementedError("Saving trained models will be added in a later stage.")

    def load(self, path: str) -> None:
        """Placeholder loading hook for later stages."""

        raise NotImplementedError("Loading trained models will be added in a later stage.")
