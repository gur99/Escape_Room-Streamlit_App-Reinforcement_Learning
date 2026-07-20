"""Tabular SARSA algorithm for Room 2."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from random import Random
from time import perf_counter
from typing import Any

from algorithms.base import RLAlgorithmBase
from utils.replay import SarsaEpisodeRecord


@dataclass
class SarsaConfig:
    """Hyperparameters for the SARSA algorithm."""

    alpha: float = 0.1
    gamma: float = 0.98
    epsilon: float = 1.0
    epsilon_decay: float = 0.998
    episodes: int = 1000
    max_steps: int = 300


class SarsaAgent(RLAlgorithmBase):
    """On-policy tabular SARSA with epsilon-greedy exploration."""

    algorithm_name = "SARSA"
    learning_type = "On-policy temporal-difference learning"
    algorithm_summary = (
        "Tabular SARSA learns an action-value function from experience. "
        "Actions are chosen with epsilon-greedy exploration over legal moves; "
        "epsilon decays multiplicatively after each episode."
    )

    def __init__(self, config: SarsaConfig | None = None) -> None:
        self.config = config or SarsaConfig()
        self.q_table: dict[tuple[tuple[int, int], int], float] = {}
        self._random = Random(0)

    def train(self, environment: Any) -> dict[str, Any]:
        """Run tabular SARSA and record every episode for replay and metrics."""

        environment.config.max_steps = self.config.max_steps
        self.q_table = {}
        self._random = Random(0)
        epsilon = self.config.epsilon
        episodes: list[SarsaEpisodeRecord] = []
        visit_counts: dict[tuple[int, int], int] = defaultdict(int)
        cumulative_time_s = 0.0
        training_started = perf_counter()

        for episode_index in range(self.config.episodes):
            record = self._run_episode(
                environment,
                episode_index,
                epsilon,
                visit_counts,
                cumulative_time_s,
            )
            episodes.append(record)
            cumulative_time_s = record.cumulative_time_s
            epsilon *= self.config.epsilon_decay

        training_time_s = perf_counter() - training_started
        environment.reset()
        layout_grid = environment.build_layout_grid()
        value_function, greedy_policy, greedy_policy_arrows = self._derive_value_and_policy(
            environment
        )

        total_rewards = [episode.total_reward for episode in episodes]
        return {
            "algorithm": self.algorithm_name,
            "config": asdict(self.config),
            "episodes": episodes,
            "episode_count": len(episodes),
            "q_table": dict(self.q_table),
            "final_epsilon": epsilon,
            "visit_counts": dict(visit_counts),
            "layout_grid": layout_grid,
            "value_function": value_function,
            "greedy_policy": greedy_policy,
            "greedy_policy_arrows": greedy_policy_arrows,
            "value_grid": environment.build_value_grid(value_function, precision=2),
            "policy_grid": self._build_multi_arrow_policy_grid(
                environment,
                greedy_policy_arrows,
            ),
            "training_time_s": training_time_s,
            "metrics_summary": {
                "average_reward": (
                    sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
                ),
                "peak_reward": max(total_rewards) if total_rewards else 0.0,
                "episodes": len(episodes),
                "average_off_policy_steps": (
                    sum(episode.off_policy_steps for episode in episodes) / len(episodes)
                    if episodes
                    else 0.0
                ),
                "training_time_s": training_time_s,
            },
        }

    def predict(self, observation: Any) -> int:
        """Return the greedy legal action for an observation (deterministic ties)."""

        legal_actions = list(range(4))
        return self._greedy_action(observation, legal_actions)

    def _q(self, state: tuple[int, int], action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def _set_q(self, state: tuple[int, int], action: int, value: float) -> None:
        self.q_table[(state, action)] = value

    def _greedy_action(
        self,
        state: tuple[int, int],
        legal_actions: list[int],
    ) -> int:
        """Return the legal action with highest Q; lowest index wins ties."""

        if not legal_actions:
            raise ValueError(f"No legal actions available from state {state}.")
        best_action = legal_actions[0]
        best_value = self._q(state, best_action)
        for action in legal_actions[1:]:
            value = self._q(state, action)
            if value > best_value or (value == best_value and action < best_action):
                best_action = action
                best_value = value
        return best_action

    def _greedy_actions(
        self,
        state: tuple[int, int],
        legal_actions: list[int],
    ) -> list[int]:
        """Return all legal actions that share the maximum Q value."""

        if not legal_actions:
            return []
        best_value = max(self._q(state, action) for action in legal_actions)
        return [
            action
            for action in legal_actions
            if self._q(state, action) == best_value
        ]

    def _select_action(
        self,
        state: tuple[int, int],
        legal_actions: list[int],
        epsilon: float,
    ) -> tuple[int, bool, bool]:
        """Epsilon-greedy selection.

        Returns ``(action, is_off_policy, used_exploration)``.
        Off-policy means chosen action differs from the current greedy action,
        measured at selection time before any Q update. Exploration means the
        epsilon-random branch was taken.
        """

        greedy_action = self._greedy_action(state, legal_actions)
        used_exploration = self._random.random() < epsilon
        if used_exploration:
            chosen = legal_actions[self._random.randrange(len(legal_actions))]
        else:
            chosen = greedy_action
        is_off_policy = chosen != greedy_action
        return chosen, is_off_policy, used_exploration

    def _derive_value_and_policy(
        self,
        environment: Any,
    ) -> tuple[
        dict[tuple[int, int], float],
        dict[tuple[int, int], int],
        dict[tuple[int, int], list[int]],
    ]:
        """Compute V(s)=max_a Q(s,a) and greedy action sets for legal states."""

        value_function: dict[tuple[int, int], float] = {}
        greedy_policy: dict[tuple[int, int], int] = {}
        greedy_policy_arrows: dict[tuple[int, int], list[int]] = {}

        for state in environment.iter_states():
            if environment.is_terminal_state(state):
                value_function[state] = 0.0
                continue
            legal_actions = environment.get_legal_actions(state)
            if not legal_actions:
                value_function[state] = 0.0
                continue
            best_actions = self._greedy_actions(state, legal_actions)
            value_function[state] = self._q(state, best_actions[0])
            greedy_policy[state] = best_actions[0]
            greedy_policy_arrows[state] = best_actions
        return value_function, greedy_policy, greedy_policy_arrows

    def _build_multi_arrow_policy_grid(
        self,
        environment: Any,
        greedy_policy_arrows: dict[tuple[int, int], list[int]],
    ) -> list[list[str]]:
        """Build a policy grid; tied greedy actions become concatenated arrows."""

        arrow_by_action = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        grid: list[list[str]] = []
        for row in range(environment.config.grid_height):
            row_actions: list[str] = []
            for col in range(environment.config.grid_width):
                state = (row, col)
                if state == environment.goal_position:
                    row_actions.append("E")
                elif state in environment.walls:
                    row_actions.append("W")
                elif state in greedy_policy_arrows:
                    arrows = "".join(
                        arrow_by_action[action] for action in greedy_policy_arrows[state]
                    )
                    row_actions.append(arrows)
                else:
                    row_actions.append(".")
            grid.append(row_actions)
        return grid

    def _run_episode(
        self,
        environment: Any,
        episode_index: int,
        epsilon: float,
        visit_counts: dict[tuple[int, int], int],
        cumulative_time_s: float,
    ) -> SarsaEpisodeRecord:
        """Execute one SARSA episode and return its replay record."""

        episode_started = perf_counter()
        state, _ = environment.reset()
        visit_counts[state] = visit_counts.get(state, 0) + 1
        legal_actions = environment.get_legal_actions(state)
        action, is_off_policy, used_exploration = self._select_action(
            state,
            legal_actions,
            epsilon,
        )

        states: list[tuple[int, int]] = []
        actions: list[int] = []
        next_states: list[tuple[int, int]] = []
        rewards: list[float] = []
        was_exploration: list[bool] = []
        was_off_policy: list[bool] = []
        epsilon_per_step: list[float] = []
        off_policy_steps = 0
        total_reward = 0.0
        discounted_return = 0.0
        discount = 1.0
        termination_reason: str = "max_steps"

        for _ in range(self.config.max_steps):
            next_state, reward, terminated, truncated, _info = environment.step(action)
            visit_counts[next_state] = visit_counts.get(next_state, 0) + 1

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            was_exploration.append(used_exploration)
            was_off_policy.append(is_off_policy)
            epsilon_per_step.append(epsilon)
            if is_off_policy:
                off_policy_steps += 1
            total_reward += reward
            discounted_return += discount * reward
            discount *= self.config.gamma

            if terminated:
                td_target = reward
                self._set_q(
                    state,
                    action,
                    self._q(state, action)
                    + self.config.alpha * (td_target - self._q(state, action)),
                )
                termination_reason = "goal"
                break

            next_legal = environment.get_legal_actions(next_state)
            next_action, next_is_off_policy, next_used_exploration = self._select_action(
                next_state,
                next_legal,
                epsilon,
            )
            td_target = reward + self.config.gamma * self._q(next_state, next_action)
            self._set_q(
                state,
                action,
                self._q(state, action)
                + self.config.alpha * (td_target - self._q(state, action)),
            )

            state = next_state
            action = next_action
            is_off_policy = next_is_off_policy
            used_exploration = next_used_exploration

            if truncated:
                termination_reason = "max_steps"
                break
        else:
            termination_reason = "max_steps"

        time_ms = (perf_counter() - episode_started) * 1000.0
        cumulative_time_s = cumulative_time_s + (time_ms / 1000.0)

        return SarsaEpisodeRecord(
            episode_index=episode_index,
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            was_exploration=was_exploration,
            was_off_policy=was_off_policy,
            epsilon_per_step=epsilon_per_step,
            steps=len(actions),
            off_policy_steps=off_policy_steps,
            total_reward=total_reward,
            discounted_return=discounted_return,
            epsilon_start=epsilon,
            time_ms=time_ms,
            cumulative_time_s=cumulative_time_s,
            termination_reason=termination_reason,  # type: ignore[arg-type]
        )
