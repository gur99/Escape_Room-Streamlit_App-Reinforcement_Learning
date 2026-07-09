"""Room 5 obstacle-avoidance environment skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from environments.room4_continuous import Room4Config, Room4ContinuousEnv
from utils.spaces import CompositeSpace, DiscreteSpace


@dataclass
class Room5Config(Room4Config):
    """Default settings for the optional obstacle-avoidance room."""

    observation_distance: float = 2.0
    obstacle_width: float = 0.5
    obstacle_count: int = 6
    collision_penalty: float = -20.0
    progress_reward: float = 0.2


class Room5ObstacleEnv(Room4ContinuousEnv):
    """Continuous room with randomized obstacles and observation-distance control."""

    room_title = "Room 5 - Obstacle Avoidance"
    room_summary = (
        "Optional continuous room that studies obstacle avoidance and generalization."
    )
    state_space_description = (
        "Continuous state (x, y, vx, vy) plus forward obstacle observations."
    )
    action_space_description = "Nine discrete motion-direction choices."
    reward_description = (
        "Goal reward, per-step penalty, collision penalty, and small progress reward."
    )

    def __init__(self, config: Room5Config | None = None) -> None:
        super().__init__(config or Room5Config())
        room5_config = self.config
        self.observation_space = CompositeSpace(
            components={
                "agent_state": {
                    "low": (0.0, 0.0, -1.0, -1.0),
                    "high": (
                        room5_config.room_width,
                        room5_config.room_height,
                        1.0,
                        1.0,
                    ),
                },
                "observation_distance": room5_config.observation_distance,
            },
            description=(
                "Agent state augmented with obstacle-awareness metadata for future work."
            ),
        )
        self.action_space = DiscreteSpace(
            n=9,
            description="Nine discrete movement-direction choices.",
        )
        self.obstacles: list[dict[str, float]] = []

    def reset(self) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
        """Reset the room and expose placeholder obstacle metadata."""

        observation, info = super().reset()
        self.obstacles = []
        info["obstacles"] = self.obstacles
        info["observation_distance"] = self.config.observation_distance
        return observation, info

    def render(self) -> dict[str, Any]:
        """Return a serializable snapshot including obstacle settings."""

        snapshot = super().render()
        snapshot["observation_distance"] = self.config.observation_distance
        snapshot["obstacle_width"] = self.config.obstacle_width
        snapshot["obstacle_count"] = self.config.obstacle_count
        snapshot["obstacles"] = self.obstacles
        return snapshot
