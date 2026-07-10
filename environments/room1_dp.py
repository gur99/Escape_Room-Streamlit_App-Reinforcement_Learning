"""Room 1: fixed educational GridWorld for Dynamic Programming.

The maze layout and environment parameters are intentionally constant so the
student focuses only on the DP algorithm hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from environments.grid_room_base import GridRoomBase, GridRoomConfig
from utils.replay import EpisodeReplay, ReplayStep


@dataclass
class Room1Config(GridRoomConfig):
    """Fixed internal settings for Room 1.

    These values are not exposed in the UI. Changing them requires editing code.
    """

    grid_width: int = 10
    grid_height: int = 10
    max_steps: int = 200
    # Slippery cells: 50% keep chosen action, 50% random among the other three.
    slip_probability: float = 0.5
    goal_reward: float = 10.0
    trap_penalty: float = -0.5


class Room1DynamicProgrammingEnv(GridRoomBase):
    """Fixed 10x10 maze with multiple paths, traps, and slippery tiles."""

    room_title = "Room 1 - Dynamic Programming"
    room_summary = (
        "A fixed known-model 10x10 maze. Start is top-left, the Exit Door is "
        "bottom-right (+10). Every path to the exit must cross at least one "
        "slippery cell (the openings in the wall band). Change only the DP "
        "hyperparameters to study how Value Iteration and Policy Iteration learn."
    )
    state_space_description = "Discrete 10x10 grid locations (walls blocked)."
    action_space_description = "Move up, right, down, or left."
    reward_description = (
        "Exit Door: +10. Trap: -0.5. Regular cells give 0. "
        "Only legal actions are selectable. A sealed wall band forces every "
        "path through at least one slippery opening (50% intended legal action, "
        "50% random among other legal actions)."
    )

    def __init__(self, config: Room1Config | None = None) -> None:
        # Always enforce the educational fixed config, ignoring user overrides.
        super().__init__(Room1Config())
        self.start_position = (0, 0)
        self.goal_position = (9, 9)
        # Multi-path maze with a horizontal wall band. The only passages through
        # that band are slippery double-openings (II), so every start→exit route
        # must visit at least one slippery cell.
        self.walls = {
            (1, 2),
            (1, 3),
            (1, 6),
            (1, 7),
            (3, 1),
            (3, 7),
            (3, 9),
            (4, 0),
            (4, 1),
            (4, 4),
            (4, 5),
            (4, 8),
            (4, 9),
            (5, 1),
            (5, 4),
            (6, 3),
            (6, 9),
            (7, 1),
            (7, 5),
            (7, 7),
            (8, 2),
            (8, 4),
            (8, 8),
        }
        self.traps = {(2, 2), (3, 5), (6, 5), (7, 9), (8, 6), (8, 9)}
        self.slippery_cells = {(4, 2), (4, 3), (4, 6), (4, 7)}
        self.random_generator = Random(7)
        self.agent_position = self.start_position

    def _build_uniform_slip_distribution(
        self,
        legal_actions: list[int],
        chosen_action: int,
    ) -> dict[int, float]:
        """Build slip model: P(chosen) plus uniform split over other legal actions."""

        other_actions = [
            candidate for candidate in legal_actions if candidate != chosen_action
        ]
        if not other_actions:
            return {chosen_action: 1.0}

        intended_probability = self.config.slip_probability
        distribution = {chosen_action: intended_probability}
        shared_probability = (1.0 - intended_probability) / len(other_actions)
        for other_action in other_actions:
            distribution[other_action] = shared_probability
        return distribution

    def _resolve_to_legal_action(self, state: tuple[int, int], action: int) -> int:
        """Map a requested action onto a legal action from this state."""

        legal_actions = self.get_legal_actions(state)
        if not legal_actions:
            raise ValueError(f"No legal actions are available from state {state}.")
        if action in legal_actions:
            return action
        return legal_actions[0]

    def _get_action_probabilities(self, state: tuple[int, int], action: int) -> dict[int, float]:
        """Return the execution distribution for one evaluated action.

        On slippery cells every legal action slips: the chosen action succeeds with
        ``slip_probability`` and otherwise one of the other legal actions is chosen
        uniformly. Regular cells execute the chosen action deterministically.
        """

        legal_actions = self.get_legal_actions(state)
        if not legal_actions:
            raise ValueError(f"No legal actions are available from state {state}.")

        chosen_action = self._resolve_to_legal_action(state, action)
        if state in self.slippery_cells:
            return self._build_uniform_slip_distribution(legal_actions, chosen_action)
        return {chosen_action: 1.0}

    def _get_reward_for_state(self, state: tuple[int, int]) -> float:
        """Return the immediate reward obtained after entering a state."""

        if state == self.goal_position:
            return self.config.goal_reward
        if state in self.traps:
            return self.config.trap_penalty
        return 0.0

    def get_transition_distribution(
        self,
        state: tuple[int, int],
        action: int,
    ) -> list[tuple[float, tuple[int, int], float, bool]]:
        """Return the full transition model for one state-action pair.

        Illegal actions are rejected at selection time and replaced by a legal
        action before the transition distribution is built. Each returned tuple
        follows `(probability, next_state, reward, terminated)`.
        """

        if self.is_terminal_state(state):
            return [(1.0, state, 0.0, True)]

        legal_action = self._resolve_to_legal_action(state, action)
        aggregated_outcomes: dict[tuple[tuple[int, int], float, bool], float] = {}
        for sampled_action, probability in self._get_action_probabilities(
            state,
            legal_action,
        ).items():
            # Sampled actions are already filtered to legal ones, so movement succeeds.
            next_state = self.move_from_state(state, sampled_action)
            reward = self._get_reward_for_state(next_state)
            terminated = self.is_terminal_state(next_state)
            outcome_key = (next_state, reward, terminated)
            aggregated_outcomes[outcome_key] = (
                aggregated_outcomes.get(outcome_key, 0.0) + probability
            )

        return [
            (probability, next_state, reward, terminated)
            for (next_state, reward, terminated), probability in aggregated_outcomes.items()
        ]

    def step(
        self,
        action: int,
    ) -> tuple[tuple[int, int], float, bool, bool, dict[str, Any]]:
        """Advance the environment using only legal actions from the current cell."""

        self.steps_taken += 1
        state_before_action = self.agent_position
        legal_action = self._resolve_to_legal_action(state_before_action, action)
        roll = self.random_generator.random()
        cumulative_probability = 0.0
        transitions = self.get_transition_distribution(state_before_action, legal_action)
        chosen_transition = transitions[-1]

        for transition in transitions:
            probability, next_state, reward, terminated = transition
            cumulative_probability += probability
            if roll <= cumulative_probability:
                chosen_transition = transition
                break

        _, next_state, reward, terminated = chosen_transition
        self.agent_position = next_state
        truncated = self.steps_taken >= self.config.max_steps and not terminated
        info = {
            "state_before_action": state_before_action,
            "requested_action": action,
            "selected_action": legal_action,
            "selected_action_label": self.get_action_label(legal_action),
            "legal_actions": self.get_legal_actions(state_before_action),
            "transition_distribution": transitions,
        }
        return next_state, reward, terminated, truncated, info

    def describe_slippery_cell(
        self,
        state: tuple[int, int],
        intended_action: int,
    ) -> dict[str, Any]:
        """Explain the slip distribution used from one slippery cell under a policy action."""

        arrow_by_action = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        name_by_action = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

        legal_action = self._resolve_to_legal_action(state, intended_action)
        legal_actions = self.get_legal_actions(state)
        uses_slip_distribution = len(legal_actions) > 1
        probabilities = self._get_action_probabilities(state, legal_action)
        ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
        outcomes = [
            {
                "action": action,
                "label": name_by_action[action],
                "short_label": self.get_action_label(action),
                "arrow": arrow_by_action[action],
                "probability": round(probability, 4),
            }
            for action, probability in ordered
        ]

        # Arrow and name are always derived from the same resolved action index.
        intended_label = name_by_action[legal_action]
        intended_short_label = self.get_action_label(legal_action)
        intended_arrow = arrow_by_action[legal_action]
        other_parts = [
            f"{item['arrow']} ({item['label']}) with probability {item['probability']:.2f}"
            for item in outcomes
            if item["action"] != legal_action
        ]
        if other_parts:
            slip_sentence = (
                f"On this slippery cell, choosing {intended_arrow} ({intended_label}) "
                f"moves {intended_arrow} with probability "
                f"{self.config.slip_probability:.2f}, and otherwise slips to "
                + ", ".join(other_parts)
                + ". Only legal directions are sampled."
            )
        else:
            slip_sentence = (
                f"If the policy chooses {intended_arrow} ({intended_label}) here, "
                f"that is the only legal action, so the agent always moves "
                f"{intended_arrow}."
            )

        compact_label = " ".join(f"{item['probability']:.2f}" for item in outcomes)
        return {
            "state": state,
            "intended_action": legal_action,
            "intended_label": intended_label,
            "intended_short_label": intended_short_label,
            "intended_arrow": intended_arrow,
            "outcomes": outcomes,
            "compact_label": compact_label,
            "explanation": slip_sentence,
            "uses_slip_distribution": uses_slip_distribution,
            "legal_actions": [
                {
                    "action": action,
                    "label": name_by_action[action],
                    "short_label": self.get_action_label(action),
                    "arrow": arrow_by_action[action],
                }
                for action in self.get_legal_actions(state)
            ],
        }

    def explain_slippery_cells(
        self,
        policy: dict[tuple[int, int], int],
    ) -> list[dict[str, Any]]:
        """Return slip explanations for every slippery cell under the given policy."""

        explanations: list[dict[str, Any]] = []
        for state in sorted(self.slippery_cells):
            intended_action = policy.get(state)
            if intended_action is None:
                legal_actions = self.get_legal_actions(state)
                if not legal_actions:
                    continue
                intended_action = legal_actions[0]
            explanations.append(self.describe_slippery_cell(state, intended_action))
        return explanations

    def build_unified_display_grid(
        self,
        policy: dict[tuple[int, int], int],
    ) -> list[list[dict[str, Any]]]:
        """Build one display grid with layout markers, policy arrows, and rewards.

        Slippery cells use the same centered policy arrow as regular actionable
        cells. Special rewards use the label format ``r = N``.
        """

        arrow_by_label = {"U": "↑", "R": "→", "D": "↓", "L": "←"}

        def format_reward_label(value: float) -> str:
            if float(value).is_integer():
                return f"r = {int(value)}"
            return f"r = {value:g}"

        grid: list[list[dict[str, Any]]] = []
        for row in range(self.config.grid_height):
            row_cells: list[dict[str, Any]] = []
            for col in range(self.config.grid_width):
                state = (row, col)
                if state == self.start_position:
                    marker = "S"
                    kind = "start"
                elif state == self.goal_position:
                    marker = "E"
                    kind = "exit"
                elif state in self.walls:
                    marker = "W"
                    kind = "wall"
                elif state in self.traps:
                    marker = "T"
                    kind = "trap"
                elif state in self.slippery_cells:
                    marker = ""
                    kind = "slippery"
                else:
                    marker = "."
                    kind = "empty"

                reward_label = ""
                if kind == "exit":
                    reward_label = format_reward_label(self.config.goal_reward)
                elif kind == "trap":
                    reward_label = format_reward_label(self.config.trap_penalty)

                cell: dict[str, Any] = {
                    "kind": kind,
                    "marker": marker,
                    "policy_arrow": "",
                    "policy_label": "",
                    "reward_label": reward_label,
                    "title": marker or kind,
                }

                if kind != "wall" and not self.is_terminal_state(state):
                    action = policy.get(state)
                    if action is None:
                        legal_actions = self.get_legal_actions(state)
                        action = legal_actions[0] if legal_actions else None
                    if action is not None:
                        label = self.get_action_label(action)
                        arrow = arrow_by_label[label]
                        cell["policy_arrow"] = arrow
                        cell["policy_label"] = label

                row_cells.append(cell)
            grid.append(row_cells)
        return grid

    def build_slippery_annotated_policy_grid(
        self,
        policy: dict[tuple[int, int], int],
    ) -> list[list[str]]:
        """Build a policy grid that annotates slippery cells with slip probabilities."""

        explanations = {
            tuple(item["state"]): item for item in self.explain_slippery_cells(policy)
        }
        grid = self.build_policy_grid(policy)
        for (row, col), explanation in explanations.items():
            grid[row][col] = f"I|{explanation['compact_label']}"
        return grid

    def get_transition_model_summary(self) -> dict[str, Any]:
        """Return a readable description of the fixed transition model."""

        return {
            "legal_action_filtering": (
                "From every cell only actions that stay inside the grid and "
                "avoid walls are available. Illegal actions are never chosen."
            ),
            "regular_cells": "The intended legal action is executed with probability 1.0.",
            "slippery_cells": {
                "intended_action_probability": self.config.slip_probability,
                "description": (
                    "From every slippery cell, any chosen legal action succeeds with "
                    f"probability {self.config.slip_probability:.0%} and otherwise the "
                    "agent uniformly picks one of the other legal actions from that "
                    "cell. Slip applies on every action while standing on a slippery "
                    "cell."
                ),
            },
            "terminal_state": "The Exit Door is the only terminal state.",
        }

    def get_reward_model_summary(self) -> dict[str, Any]:
        """Return the explicit fixed reward model."""

        return {
            "exit_door_reward": self.config.goal_reward,
            "trap_penalty": self.config.trap_penalty,
            "trap_behavior": (
                f"Traps give {self.config.trap_penalty:g} on entry but do not "
                "terminate the episode."
            ),
            "exit_door": (
                f"Bottom-right cell (9, 9) ends the episode with "
                f"+{self.config.goal_reward:g}."
            ),
        }

    def build_layout_grid(self) -> list[list[str]]:
        """Build an ASCII grid that describes the fixed room layout."""

        grid: list[list[str]] = []
        for row in range(self.config.grid_height):
            row_cells: list[str] = []
            for col in range(self.config.grid_width):
                state = (row, col)
                if state == self.start_position:
                    cell = "S"
                elif state == self.goal_position:
                    cell = "E"
                elif state in self.walls:
                    cell = "W"
                elif state in self.traps:
                    cell = "T"
                elif state in self.slippery_cells:
                    cell = "I"
                else:
                    cell = "."

                if state == self.agent_position and state not in {
                    self.start_position,
                    self.goal_position,
                }:
                    cell = "A" if cell == "." else f"{cell}/A"
                row_cells.append(cell)
            grid.append(row_cells)
        return grid

    def build_value_grid(
        self,
        value_function: dict[tuple[int, int], float],
        *,
        precision: int = 2,
    ) -> list[list[str]]:
        """Build a displayable value-function table."""

        grid: list[list[str]] = []
        for row in range(self.config.grid_height):
            row_values: list[str] = []
            for col in range(self.config.grid_width):
                state = (row, col)
                if state in self.walls:
                    row_values.append("W")
                else:
                    row_values.append(
                        f"{value_function.get(state, 0.0):.{precision}f}"
                    )
            grid.append(row_values)
        return grid

    def build_policy_grid(self, policy: dict[tuple[int, int], int]) -> list[list[str]]:
        """Build a displayable policy table."""

        grid: list[list[str]] = []
        for row in range(self.config.grid_height):
            row_actions: list[str] = []
            for col in range(self.config.grid_width):
                state = (row, col)
                if state == self.goal_position:
                    row_actions.append("E")
                elif state in self.walls:
                    row_actions.append("W")
                elif state in self.traps:
                    row_actions.append(self.get_action_label(policy.get(state, 0)))
                else:
                    row_actions.append(self.get_action_label(policy.get(state, 0)))
            grid.append(row_actions)
        return grid

    def build_replay_grid(self, replay: EpisodeReplay) -> list[list[str]]:
        """Overlay the replay visitation order on top of the room layout."""

        grid = self.build_layout_grid()
        for step_index, step in enumerate(replay.steps, start=1):
            state_after_action = tuple(step.info["state_after_action"])
            if state_after_action in {self.start_position, self.goal_position}:
                continue
            row, col = state_after_action
            if state_after_action not in self.walls:
                grid[row][col] = str(step_index)
        return grid

    def _choose_most_likely_transition(
        self,
        state: tuple[int, int],
        action: int,
    ) -> tuple[float, tuple[int, int], float, bool]:
        """Choose the highest-probability transition for deterministic replay."""

        transitions = self.get_transition_distribution(state, action)
        return max(
            transitions,
            key=lambda item: (item[0], item[3], item[2], -item[1][0], -item[1][1]),
        )

    def generate_policy_replay(
        self,
        policy: dict[tuple[int, int], int],
    ) -> EpisodeReplay:
        """Create a deterministic replay by following the learned policy."""

        current_state = self.start_position
        total_steps = 0
        replay_steps: list[ReplayStep] = []

        while (
            current_state != self.goal_position
            and total_steps < self.config.max_steps
            and current_state in policy
        ):
            legal_actions = self.get_legal_actions(current_state)
            if not legal_actions:
                break

            action = policy[current_state]
            if action not in legal_actions:
                action = legal_actions[0]

            probability, next_state, reward, terminated = self._choose_most_likely_transition(
                current_state,
                action,
            )
            total_steps += 1
            replay_steps.append(
                ReplayStep(
                    observation=next_state,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=False,
                    info={
                        "state_before_action": list(current_state),
                        "state_after_action": list(next_state),
                        "action_label": self.get_action_label(action),
                        "legal_actions": legal_actions,
                        "chosen_transition_probability": round(probability, 3),
                    },
                )
            )
            current_state = next_state
            if terminated:
                break

        if replay_steps and current_state != self.goal_position:
            replay_steps[-1].truncated = True

        return EpisodeReplay(
            room_name=self.room_title,
            algorithm_name="Dynamic Programming",
            steps=replay_steps,
        )

    def evaluate_policy_return(self, policy: dict[tuple[int, int], int]) -> float:
        """Return the total reward of one deterministic evaluation episode."""

        replay = self.generate_policy_replay(policy)
        return float(sum(step.reward for step in replay.steps))

    def render(self) -> dict[str, Any]:
        """Return the room state together with explicit model definitions."""

        snapshot = super().render()
        snapshot["transition_model"] = self.get_transition_model_summary()
        snapshot["reward_model"] = self.get_reward_model_summary()
        snapshot["layout_grid"] = self.build_layout_grid()
        snapshot["exit_door"] = list(self.goal_position)
        return snapshot
