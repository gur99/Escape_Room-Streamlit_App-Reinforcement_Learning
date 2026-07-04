"""Shared grid-environment skeleton for Rooms 1-3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from environments.base_environment import GymCompatibleEnvironment
from utils.spaces import DiscreteSpace


@dataclass
class GridRoomConfig:
    """Common configuration values for grid-based rooms."""

    grid_width: int = 10
    grid_height: int = 10
    max_steps: int = 200
    slip_probability: float = 0.0
    step_penalty: float = -0.1
    goal_reward: float = 10.0
    trap_penalty: float = -5.0


class GridRoomBase(GymCompatibleEnvironment):
    """Base class for the grid rooms used in the early stages of the project.

    Stage 1 keeps the environment behavior intentionally lightweight. The class
    already follows the Gymnasium interface and exposes room metadata, while the
    detailed transition and reward logic will be implemented room by room later.
    """

    room_title = "Grid Room Base"
    room_summary = "Base skeleton for grid-based escape room environments."
    state_space_description = "Agent row and column on a finite grid."
    action_space_description = "Four discrete moves: up, right, down, left."
    reward_description = (
        "Rewards are room-specific and will be finalized in the next stages."
    )

    def __init__(self, config: GridRoomConfig | None = None) -> None:
        self.config = config or GridRoomConfig()
        self.observation_space = DiscreteSpace(
            n=self.config.grid_width * self.config.grid_height,
            description="Discrete grid position encoded over a 10x10 board.",
        )
        self.action_space = DiscreteSpace(
            n=4,
            description="0=up, 1=right, 2=down, 3=left.",
        )
        self.start_position = (0, 0)
        self.goal_position = (self.config.grid_height - 1, self.config.grid_width - 1)
        self.walls: set[tuple[int, int]] = set()
        self.traps: set[tuple[int, int]] = set()
        self.slippery_cells: set[tuple[int, int]] = set()
        self.small_rewards: dict[tuple[int, int], float] = {}
        self.agent_position = self.start_position
        self.steps_taken = 0

    def reset(self) -> tuple[tuple[int, int], dict[str, Any]]:
        """Reset the room to its initial state."""

        self.agent_position = self.start_position
        self.steps_taken = 0
        observation = self.agent_position
        info = {
            "room_title": self.room_title,
            "goal_position": self.goal_position,
        }
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[tuple[int, int], float, bool, bool, dict[str, Any]]:
        """Return a placeholder Gymnasium step result.

        The real transition model, slip handling, and reward shaping are delayed
        intentionally to later milestones so Stage 1 remains a clean skeleton.
        """

        self.steps_taken += 1
        observation = self.agent_position
        reward = 0.0
        terminated = False
        truncated = self.steps_taken >= self.config.max_steps
        info = {
            "selected_action": action,
            "implementation_status": "placeholder",
            "message": "Detailed room dynamics will be implemented in a later stage.",
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> dict[str, Any]:
        """Return a serializable snapshot used by the Streamlit UI."""

        return {
            "room_title": self.room_title,
            "grid_size": [self.config.grid_height, self.config.grid_width],
            "agent_position": list(self.agent_position),
            "start_position": list(self.start_position),
            "goal_position": list(self.goal_position),
            "walls": [list(cell) for cell in sorted(self.walls)],
            "traps": [list(cell) for cell in sorted(self.traps)],
            "slippery_cells": [list(cell) for cell in sorted(self.slippery_cells)],
            "small_rewards": {
                f"{row},{col}": reward
                for (row, col), reward in sorted(self.small_rewards.items())
            },
            "steps_taken": self.steps_taken,
        }

    def close(self) -> None:
        """Close the environment.

        There are no external resources to release in the skeleton stage.
        """

        return None
