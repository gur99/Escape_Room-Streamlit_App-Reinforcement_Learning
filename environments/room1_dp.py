"""Room 1 environment skeleton for Dynamic Programming."""

from __future__ import annotations

from dataclasses import dataclass

from environments.grid_room_base import GridRoomBase, GridRoomConfig


@dataclass
class Room1Config(GridRoomConfig):
    """Default settings for Room 1."""

    slip_probability: float = 0.15
    step_penalty: float = -0.05
    goal_reward: float = 15.0
    trap_penalty: float = -4.0


class Room1DynamicProgrammingEnv(GridRoomBase):
    """A known-model room prepared for Value Iteration or Policy Iteration."""

    room_title = "Room 1 - Dynamic Programming"
    room_summary = (
        "Known-model 10x10 grid with a single goal, walls, traps, and slippery cells."
    )
    state_space_description = "Discrete 10x10 grid locations."
    action_space_description = "Move up, right, down, or left."
    reward_description = (
        "Small negative step penalty, strong positive goal reward, and trap penalty."
    )

    def __init__(self, config: Room1Config | None = None) -> None:
        super().__init__(config or Room1Config())
        self.start_position = (0, 0)
        self.goal_position = (9, 9)
        self.walls = {(1, 3), (2, 3), (3, 3), (5, 6), (6, 6)}
        self.traps = {(4, 1), (7, 7)}
        self.slippery_cells = {(2, 5), (2, 6), (3, 5)}
