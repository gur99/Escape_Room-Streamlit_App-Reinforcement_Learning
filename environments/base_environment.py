"""Common Gymnasium-style environment interfaces for the project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any


class GymCompatibleEnvironment(ABC):
    """Abstract base class for all project environments.

    Each environment exposes the standard Gymnasium-style methods and metadata so
    algorithms can interact with environments through one shared interface.
    """

    room_title: str = "Unnamed Room"
    room_summary: str = "No room summary has been provided yet."
    state_space_description: str = "State description is not available yet."
    action_space_description: str = "Action description is not available yet."
    reward_description: str = "Reward description is not available yet."

    observation_space: Any
    action_space: Any
    config: Any

    @abstractmethod
    def reset(self) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and return `(observation, info)`."""

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance the environment and return Gymnasium-style step outputs."""

    @abstractmethod
    def render(self) -> dict[str, Any]:
        """Return a serializable environment snapshot for the UI."""

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the environment."""

    def config_as_dict(self) -> dict[str, Any]:
        """Expose the current configuration in a UI-friendly form."""

        return asdict(self.config)
