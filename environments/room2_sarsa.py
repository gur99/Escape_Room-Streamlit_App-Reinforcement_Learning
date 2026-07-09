"""Room 2 environment skeleton for SARSA."""

from __future__ import annotations

from dataclasses import dataclass

from environments.grid_room_base import GridRoomBase, GridRoomConfig


@dataclass
class Room2Config(GridRoomConfig):
    """Default settings for Room 2."""

    slip_probability: float = 0.10
    step_penalty: float = -0.08
    goal_reward: float = 20.0
    trap_penalty: float = -6.0


class Room2SarsaEnv(GridRoomBase):
    """A model-free grid room intended for on-policy learning with SARSA."""

    room_title = "Room 2 - SARSA"
    room_summary = (
        "Unknown-model 10x10 grid with walls, traps, slippery cells, and small rewards."
    )
    state_space_description = "Discrete 10x10 grid locations."
    action_space_description = "Move up, right, down, or left."
    reward_description = (
        "Step penalty, positive goal reward, trap penalty, and small rewards along the path."
    )

    def __init__(self, config: Room2Config | None = None) -> None:
        super().__init__(config or Room2Config())
        self.start_position = (0, 0)
        self.goal_position = (9, 9)
        self.walls = {(1, 1), (1, 2), (2, 6), (3, 6), (5, 2), (6, 2)}
        self.traps = {(3, 8), (6, 7), (8, 4)}
        self.slippery_cells = {(4, 4), (4, 5), (5, 4)}
        self.small_rewards = {(0, 4): 0.3, (5, 8): 0.5, (8, 8): 0.8}
