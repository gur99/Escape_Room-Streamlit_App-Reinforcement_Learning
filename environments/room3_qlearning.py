"""Room 3 environment skeleton for Q-Learning."""

from __future__ import annotations

from dataclasses import dataclass

from environments.grid_room_base import GridRoomBase, GridRoomConfig


@dataclass
class Room3Config(GridRoomConfig):
    """Default settings for Room 3."""

    slip_probability: float = 0.12
    step_penalty: float = -0.10
    goal_reward: float = 25.0
    trap_penalty: float = -8.0


class Room3QLearningEnv(GridRoomBase):
    """A harder grid room intended for off-policy Q-Learning."""

    room_title = "Room 3 - Q-Learning"
    room_summary = (
        "Harder 10x10 room with denser obstacles, more traps, and a risky shortcut."
    )
    state_space_description = "Discrete 10x10 grid locations."
    action_space_description = "Move up, right, down, or left."
    reward_description = (
        "Step penalty, large terminal reward, trap penalty, and risk-reward shortcut design."
    )

    def __init__(self, config: Room3Config | None = None) -> None:
        super().__init__(config or Room3Config())
        self.start_position = (0, 0)
        self.goal_position = (9, 9)
        self.walls = {
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 1),
            (5, 2),
            (5, 3),
            (7, 5),
            (7, 6),
        }
        self.traps = {(2, 8), (3, 8), (6, 4), (8, 2), (8, 7)}
        self.slippery_cells = {(1, 7), (2, 7), (3, 7)}
        self.small_rewards = {(0, 8): 1.0, (6, 8): 1.2}
