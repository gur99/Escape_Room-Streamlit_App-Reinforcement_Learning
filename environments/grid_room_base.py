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
    ACTION_DELTAS: dict[int, tuple[int, int]] = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }
    ACTION_LABELS: dict[int, str] = {
        0: "U",
        1: "R",
        2: "D",
        3: "L",
    }

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

    def iter_states(self) -> list[tuple[int, int]]:
        """Return all traversable states in the grid."""

        states: list[tuple[int, int]] = []
        for row in range(self.config.grid_height):
            for col in range(self.config.grid_width):
                state = (row, col)
                if state not in self.walls:
                    states.append(state)
        return states

    def is_within_bounds(self, state: tuple[int, int]) -> bool:
        """Check whether a state lies inside the grid boundaries."""

        row, col = state
        return 0 <= row < self.config.grid_height and 0 <= col < self.config.grid_width

    def is_walkable(self, state: tuple[int, int]) -> bool:
        """Check whether the agent can occupy a state."""

        return self.is_within_bounds(state) and state not in self.walls

    def is_terminal_state(self, state: tuple[int, int]) -> bool:
        """Return whether the supplied state is terminal."""

        return state == self.goal_position

    def is_trap_state(self, state: tuple[int, int]) -> bool:
        """Return whether the supplied state is a trap."""

        return state in self.traps

    def is_slippery_state(self, state: tuple[int, int]) -> bool:
        """Return whether the supplied state is slippery."""

        return state in self.slippery_cells

    def get_intended_next_state(
        self,
        state: tuple[int, int],
        action: int,
    ) -> tuple[int, int]:
        """Return the cell that an action would target, without legality checks."""

        delta_row, delta_col = self.ACTION_DELTAS[action]
        return (state[0] + delta_row, state[1] + delta_col)

    def is_legal_action(self, state: tuple[int, int], action: int) -> bool:
        """Return whether an action stays inside the grid and avoids walls."""

        return self.is_walkable(self.get_intended_next_state(state, action))

    def get_legal_actions(self, state: tuple[int, int]) -> list[int]:
        """Return only actions that are legal from the given state.

        The available action set can change from cell to cell because boundary
        and wall cells block some directions at selection time.
        """

        return [
            action
            for action in range(self.action_space.n)
            if self.is_legal_action(state, action)
        ]

    def move_from_state(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        """Apply a movement using a legal action, or stay if the action is illegal."""

        next_state = self.get_intended_next_state(state, action)
        if not self.is_walkable(next_state):
            return state
        return next_state

    def get_action_label(self, action: int) -> str:
        """Return a short human-readable label for an action index."""

        return self.ACTION_LABELS[action]

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
