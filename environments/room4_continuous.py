"""Room 4 continuous environment skeleton for DQN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from environments.base_environment import GymCompatibleEnvironment
from utils.spaces import BoxSpace, DiscreteSpace


@dataclass
class Room4Config:
    """Default settings for the first continuous-control room."""

    room_width: float = 10.0
    room_height: float = 10.0
    time_delta: float = 0.02
    step_penalty: float = -0.02
    goal_reward: float = 30.0
    max_steps: int = 500


class Room4ContinuousEnv(GymCompatibleEnvironment):
    """Continuous movement room prepared for a future DQN implementation."""

    room_title = "Room 4 - Continuous Control"
    room_summary = (
        "Continuous 10x10 room where the agent moves with position and velocity state."
    )
    state_space_description = "Continuous state (x, y, vx, vy)."
    action_space_description = "Discrete direction choices that affect motion."
    reward_description = (
        "Small time penalty, efficiency incentive, and strong reward for reaching the exit."
    )

    def __init__(self, config: Room4Config | None = None) -> None:
        self.config = config or Room4Config()
        self.observation_space = BoxSpace(
            low=(0.0, 0.0, -1.0, -1.0),
            high=(
                self.config.room_width,
                self.config.room_height,
                1.0,
                1.0,
            ),
            shape=(4,),
            description="Continuous position and discrete-valued velocity components.",
        )
        self.action_space = DiscreteSpace(
            n=9,
            description="Nine velocity-direction choices derived from vx/vy combinations.",
        )
        self.start_position = (1.0, 1.0)
        self.goal_position = (9.0, 9.0)
        self.agent_state = (1.0, 1.0, 0.0, 0.0)
        self.steps_taken = 0

    def reset(self) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
        """Reset the continuous environment."""

        self.agent_state = (1.0, 1.0, 0.0, 0.0)
        self.steps_taken = 0
        return self.agent_state, {"goal_position": self.goal_position}

    def step(
        self,
        action: int,
    ) -> tuple[tuple[float, float, float, float], float, bool, bool, dict[str, Any]]:
        """Return a placeholder transition for the skeleton stage."""

        self.steps_taken += 1
        reward = 0.0
        terminated = False
        truncated = self.steps_taken >= self.config.max_steps
        info = {
            "selected_action": action,
            "implementation_status": "placeholder",
            "message": "Continuous transition dynamics will be added in a later stage.",
        }
        return self.agent_state, reward, terminated, truncated, info

    def render(self) -> dict[str, Any]:
        """Return a visualizable state snapshot."""

        x_pos, y_pos, velocity_x, velocity_y = self.agent_state
        return {
            "room_title": self.room_title,
            "dimensions": [self.config.room_width, self.config.room_height],
            "agent_state": {
                "x": x_pos,
                "y": y_pos,
                "vx": velocity_x,
                "vy": velocity_y,
            },
            "start_position": list(self.start_position),
            "goal_position": list(self.goal_position),
            "steps_taken": self.steps_taken,
        }

    def close(self) -> None:
        """Close the environment."""

        return None
