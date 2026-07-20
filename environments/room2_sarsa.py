"""Room 2: fixed educational GridWorld for SARSA.

Uses the same environmental constants and transition rules as Room 1
(Dynamic Programming). Only the maze layout differs: a harder 10x10 maze with
a central exit. Users cannot edit the environment; they change SARSA
hyperparameters only.
"""

from __future__ import annotations

from collections import deque
from random import Random

from environments.room1_dp import Room1Config, Room1DynamicProgrammingEnv

# Alias so existing imports of Room2Config still resolve to the shared constants.
Room2Config = Room1Config


class Room2SarsaEnv(Room1DynamicProgrammingEnv):
    """Fixed 10x10 maze for on-policy SARSA with a central Exit Door."""

    room_title = "Room 2 - SARSA"
    room_summary = (
        "A fixed unknown-model 10x10 maze for tabular SARSA. Start is top-left, "
        "the Exit Door sits in the center (+10). Walls, traps, and slippery cells "
        "form a harder multi-path maze. Change only the SARSA hyperparameters."
    )
    state_space_description = "Discrete 10x10 grid locations (walls blocked)."
    action_space_description = "Move up, right, down, or left."
    reward_description = (
        "Exit Door: +10. Trap: -0.5. Regular cells give 0. "
        "Only legal actions are selectable. Slippery cells use the same model as "
        "Room 1 (50% intended legal action, 50% random among other legal actions)."
    )

    def __init__(self, config: Room1Config | None = None) -> None:
        # Parent forces Room1Config and calls _setup_layout() once (this override).
        super().__init__(config)

    def _setup_layout(self) -> None:
        """Place a harder central-exit maze. Called once by the parent ``__init__``."""

        self.start_position = (0, 0)
        self.goal_position = (5, 5)
        # Dense corridors with multiple routes into the center. Slippery openings
        # sit on approach corridors; traps sit on tempting shortcuts.
        self.walls = {
            (0, 4),
            (0, 8),
            (1, 1),
            (1, 2),
            (1, 4),
            (1, 6),
            (1, 8),
            (2, 6),
            (2, 8),
            (3, 0),
            (3, 2),
            (3, 3),
            (3, 5),
            (3, 8),
            (4, 0),
            (4, 2),
            (4, 3),
            (4, 5),
            (4, 7),
            (5, 2),
            (5, 3),
            (5, 7),
            (6, 0),
            (6, 2),
            (6, 3),
            (6, 5),
            (6, 7),
            (6, 8),
            (7, 5),
            (7, 7),
            (8, 1),
            (8, 3),
            (8, 4),
            (8, 6),
            (8, 8),
            (9, 3),
            (9, 6),
            (9, 8),
        }
        self.traps = {
            (0, 6),
            (2, 1),
            (2, 5),
            (4, 1),
            (5, 0),
            (7, 2),
            (7, 6),
            (9, 0),
            (9, 5),
        }
        self.slippery_cells = {
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (5, 6),
        }
        self.random_generator = Random(11)
        self.agent_position = self.start_position

    def is_exit_reachable(self) -> bool:
        """Return True if a wall-avoiding path exists from start to exit."""

        if self.start_position in self.walls or self.goal_position in self.walls:
            return False

        queue: deque[tuple[int, int]] = deque([self.start_position])
        visited = {self.start_position}
        while queue:
            row, col = queue.popleft()
            if (row, col) == self.goal_position:
                return True
            for action in range(4):
                delta_row, delta_col = self.ACTION_DELTAS[action]
                neighbor = (row + delta_row, col + delta_col)
                if (
                    neighbor not in visited
                    and self.is_within_bounds(neighbor)
                    and neighbor not in self.walls
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False
