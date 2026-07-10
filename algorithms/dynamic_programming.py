"""Dynamic Programming for Room 1: Policy Iteration.

Every iteration is checkpointed so the student can inspect policy, values,
and policy performance over time. Environment parameters are fixed; only
algorithm hyperparameters are configurable from the UI.

The full learning loop always runs Policy Evaluation and Policy Improvement.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from time import time
from typing import Any

from algorithms.base import RLAlgorithmBase

# Internal fixed setting: save a snapshot after every planning iteration.
CHECKPOINT_FREQUENCY = 1
POLICY_EVALUATION_VALUE_PRECISION = 4


@dataclass
class DynamicProgrammingConfig:
    """User-facing hyperparameters for Dynamic Programming (Policy Iteration)."""

    gamma: float = 0.99
    theta: float = 0.0001
    max_iterations: int = 100
    # Fixed seed so the random initial policy is identical on every run.
    initial_policy_seed: int = 42


class DynamicProgrammingAgent(RLAlgorithmBase):
    """Dynamic Programming agent for the known-model first room."""

    algorithm_name = "Dynamic Programming"
    learning_type = "Planning with a known model"
    algorithm_summary = (
        "Policy Iteration on a fixed GridWorld model: alternating Policy "
        "Evaluation and Policy Improvement from a random legal initial policy. "
        "Change gamma, theta, and max iterations to study how planning behaves."
    )

    def __init__(self, config: DynamicProgrammingConfig | None = None) -> None:
        self.config = config or DynamicProgrammingConfig()
        self.learned_policy: dict[tuple[int, int], int] = {}
        self.learned_values: dict[tuple[int, int], float] = {}
        self.training_result: dict[str, Any] = {}
        self.iteration_snapshots: list[dict[str, Any]] = []

    def predict(self, observation: Any) -> int:
        """Return the learned policy action, or a safe default before training."""

        state = tuple(observation) if not isinstance(observation, tuple) else observation
        if state in self.learned_policy:
            return self.learned_policy[state]
        return 1

    def build_initial_policy(self, environment: Any) -> dict[tuple[int, int], int]:
        """Create a random initial policy over legal actions in every non-terminal cell."""

        rng = Random(self.config.initial_policy_seed)
        policy: dict[tuple[int, int], int] = {}
        for state in environment.iter_states():
            if environment.is_terminal_state(state):
                continue
            legal_actions = environment.get_legal_actions(state)
            if not legal_actions:
                continue
            policy[state] = rng.choice(legal_actions)
        return policy

    def train(
        self,
        environment: Any,
        initial_policy: dict[tuple[int, int], int] | None = None,
    ) -> dict[str, Any]:
        """Run Policy Iteration (evaluation + improvement) and store checkpoints."""

        if initial_policy is None:
            initial_policy = self.build_initial_policy(environment)
        self.iteration_snapshots = []

        values, policy, history = self._run_policy_iteration(
            environment,
            initial_policy,
        )

        replay = environment.generate_policy_replay(policy)
        total_replay_reward = sum(step.reward for step in replay.steps)
        self.learned_policy = policy
        self.learned_values = values
        self.training_result = {
            "planning_method": "policy_iteration",
            "value_function": values,
            "policy": policy,
            "initial_policy": initial_policy,
            "initial_policy_grid": environment.build_policy_grid(initial_policy),
            "value_grid": environment.build_value_grid(values),
            "policy_grid": environment.build_policy_grid(policy),
            "layout_grid": environment.build_layout_grid(),
            "replay_grid": environment.build_replay_grid(replay),
            "history": history,
            "iteration_snapshots": self.iteration_snapshots,
            "transition_model": environment.get_transition_model_summary(),
            "reward_model": environment.get_reward_model_summary(),
            "summary": {
                "iterations": history.get("policy_improvement_rounds", 0),
                "converged": history["converged"],
                "start_state_value": round(values.get(environment.start_position, 0.0), 4),
                "replay_steps": len(replay.steps),
                "replay_total_reward": round(total_replay_reward, 4),
                "goal_reached_in_replay": bool(
                    replay.steps and environment.is_terminal_state(replay.steps[-1].observation)
                ),
                "checkpoint_count": len(self.iteration_snapshots),
            },
            "replay": replay,
        }
        return self.training_result

    def _compute_policy_evaluation_backup(
        self,
        environment: Any,
        state: tuple[int, int],
        action: int,
        old_values: dict[tuple[int, int], float],
    ) -> float:
        """Bellman backup for Policy Evaluation under a fixed policy.

        Each sweep uses only ``V_old`` and applies the standard expectation
        equation:

        ``sum_{s'} P(s'|s,pi(s)) [ R(s,pi(s),s') + gamma * V_old(s') ]``

        Terminal successor values in ``V_old`` are treated as 0.
        """

        backup_value = 0.0
        for probability, next_state, reward, terminated in environment.get_transition_distribution(
            state,
            action,
        ):
            continuation_value = 0.0 if terminated else old_values.get(next_state, 0.0)
            backup_value += probability * (
                reward + self.config.gamma * continuation_value
            )
        return backup_value

    def _compute_action_value(
        self,
        environment: Any,
        state: tuple[int, int],
        action: int,
        values: dict[tuple[int, int], float],
    ) -> float:
        """Compute the Bellman expectation for one state-action pair.

        Implements:

        ``sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V(s') ]``

        Slippery cells use the full transition distribution from the environment
        model; deterministic cells are the special case where one outcome has
        probability 1.
        """

        action_value = 0.0
        for probability, next_state, reward, terminated in environment.get_transition_distribution(
            state,
            action,
        ):
            continuation_value = 0.0 if terminated else values[next_state]
            action_value += probability * (reward + self.config.gamma * continuation_value)
        return action_value

    def _resolve_policy_action(
        self,
        state: tuple[int, int],
        policy: dict[tuple[int, int], int],
        legal_actions: list[int],
    ) -> int:
        """Return the policy action for a state without mutating the policy."""

        action = policy.get(state, legal_actions[0])
        if action not in legal_actions:
            return legal_actions[0]
        return action

    def _compute_policy_evaluation_value(
        self,
        environment: Any,
        state: tuple[int, int],
        policy: dict[tuple[int, int], int],
        old_values: dict[tuple[int, int], float],
    ) -> float:
        """Compute ``V_new(s)`` from the fixed policy ``pi(s)`` and ``V_old``."""

        legal_actions = environment.get_legal_actions(state)
        if not legal_actions:
            return old_values.get(state, 0.0)

        action = self._resolve_policy_action(state, policy, legal_actions)
        return self._compute_policy_evaluation_backup(
            environment,
            state,
            action,
            old_values,
        )

    def _run_policy_evaluation_sweep(
        self,
        environment: Any,
        policy: dict[tuple[int, int], int],
        old_values: dict[tuple[int, int], float],
    ) -> tuple[dict[tuple[int, int], float], float]:
        """Apply one synchronous Policy Evaluation sweep.

        Every state is computed from ``V_old`` only. The previous table is
        replaced only after the full new table has been built.
        """

        frozen_old_values = deepcopy(old_values)
        new_values: dict[tuple[int, int], float] = {}
        eval_delta = 0.0

        for state in sorted(environment.iter_states()):
            old_value = frozen_old_values.get(state, 0.0)
            if environment.is_terminal_state(state):
                new_value = 0.0
            else:
                new_value = self._compute_policy_evaluation_value(
                    environment,
                    state,
                    policy,
                    frozen_old_values,
                )
                new_value = round(new_value, POLICY_EVALUATION_VALUE_PRECISION)
            new_values[state] = new_value
            eval_delta = max(eval_delta, abs(new_value - old_value))

        return new_values, eval_delta

    def _initialize_value_function(self, environment: Any) -> dict[tuple[int, int], float]:
        """Initialize every non-terminal state to 0 before the first evaluation."""

        values: dict[tuple[int, int], float] = {}
        for state in environment.iter_states():
            values[state] = 0.0
        return values

    def _policy_evaluation_converged(self, eval_delta: float) -> bool:
        """Return whether a sweep changed no state value (all deltas are 0)."""

        return eval_delta == 0.0

    def _run_policy_evaluation_phase(
        self,
        environment: Any,
        policy: dict[tuple[int, int], int],
        *,
        outer_round: int,
        global_step: int,
        policy_returns: list[float],
        deltas: list[float],
        start_state_values: list[float],
    ) -> tuple[dict[tuple[int, int], float], float, int, bool]:
        """Run Policy Evaluation until every state delta is 0 with full snapshots.

        Each Policy Evaluation phase starts from ``V(s)=0`` for every state.
        A sweep is the last one for the round when no state's value changes
        (maximum delta equals 0).
        """

        values = self._initialize_value_function(environment)
        last_eval_delta = 0.0
        evaluation_converged = False

        global_step += 1
        policy_returns.append(
            self._save_checkpoint(
                environment,
                iteration=global_step,
                outer_round=outer_round,
                eval_step=0,
                phase="policy_evaluation",
                values=values,
                policy=policy,
                delta=0.0,
                evaluation_converged=False,
                note=(
                    f"Policy Evaluation — step 0 (round {outer_round}). "
                    "Initial V(s) before the first synchronous Bellman sweep."
                ),
            )
        )

        eval_step = 0
        while eval_step < self.config.max_iterations:
            eval_step += 1
            old_values = values
            values, eval_delta = self._run_policy_evaluation_sweep(
                environment,
                policy,
                old_values,
            )
            last_eval_delta = eval_delta
            evaluation_converged = self._policy_evaluation_converged(eval_delta)

            global_step += 1
            policy_returns.append(
                self._save_checkpoint(
                    environment,
                    iteration=global_step,
                    outer_round=outer_round,
                    eval_step=eval_step,
                    phase="policy_evaluation",
                    values=values,
                    policy=policy,
                    delta=eval_delta,
                    evaluation_converged=evaluation_converged,
                    note=(
                        f"Policy Evaluation — sweep {eval_step} "
                        f"(round {outer_round}). "
                        f"Max delta={eval_delta:.6f}. "
                        + (
                            "All state deltas are 0; evaluation stops for this round."
                            if evaluation_converged
                            else "Values are still changing; another sweep is required."
                        )
                    ),
                )
            )
            deltas.append(eval_delta)
            start_state_values.append(values[environment.start_position])

            if evaluation_converged:
                break

        return values, last_eval_delta, global_step, evaluation_converged

    def _rounded_action_value(
        self,
        environment: Any,
        state: tuple[int, int],
        action: int,
        values: dict[tuple[int, int], float],
    ) -> float:
        """Return a policy-improvement action value rounded to four decimals."""

        return round(
            self._compute_action_value(environment, state, action, values),
            POLICY_EVALUATION_VALUE_PRECISION,
        )

    def _choose_improved_action_from_values(
        self,
        *,
        legal_actions: list[int],
        action_values_by_action: dict[int, float],
        current_action: int,
        rng: Random,
    ) -> tuple[int, bool, bool]:
        """Pick the improved action using the Policy Improvement tie rules.

        Returns ``(chosen_action, kept_current_all_zero, had_max_value_tie)``.
        """

        rounded_values = {
            action: round(action_values_by_action[action], POLICY_EVALUATION_VALUE_PRECISION)
            for action in legal_actions
        }

        if all(rounded_values[action] == 0.0 for action in legal_actions):
            return current_action, True, False

        best_value = max(rounded_values[action] for action in legal_actions)
        best_actions = [
            action
            for action in legal_actions
            if rounded_values[action] == best_value
        ]

        if len(best_actions) == 1:
            return best_actions[0], False, False

        return rng.choice(best_actions), False, True

    def _policy_improvement_rng_for_state(self, state: tuple[int, int]) -> Random:
        """Return a reproducible RNG for tie-breaking in one state.

        The seed depends only on the fixed project seed and the state position,
        so identical action-values produce the same tie-break in every round.
        """

        row, col = state
        return Random(self.config.initial_policy_seed + (row * 1000) + col)

    def _build_action_value_details(
        self,
        environment: Any,
        values: dict[tuple[int, int], float],
        improved_policy: dict[tuple[int, int], int],
        current_policy: dict[tuple[int, int], int],
        *,
        outer_round: int,
    ) -> list[dict[str, Any]]:
        """Build per-state action-value rows used after Policy Evaluation converges."""

        label_by_action = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
        short_by_action = {0: "U", 1: "R", 2: "D", 3: "L"}
        arrow_by_action = {0: "↑", 1: "→", 2: "↓", 3: "←"}
        rows: list[dict[str, Any]] = []

        for state in sorted(environment.iter_states()):
            row, col = state
            if state in environment.walls:
                continue

            if environment.is_terminal_state(state):
                rows.append(
                    {
                        "state": state,
                        "state_label": f"({row}, {col})",
                        "is_terminal": True,
                        "action_values": {},
                        "best_action_labels": [],
                        "chosen_action": None,
                        "chosen_label": "Terminal",
                        "chosen_arrow": "🚪",
                        "had_tie": False,
                        "kept_current_policy": False,
                    }
                )
                continue

            legal_actions = environment.get_legal_actions(state)
            if not legal_actions:
                continue

            action_values = {
                short_by_action[action]: self._rounded_action_value(
                    environment,
                    state,
                    action,
                    values,
                )
                for action in legal_actions
            }
            action_values_by_action = {
                action: self._rounded_action_value(environment, state, action, values)
                for action in legal_actions
            }
            best_value = max(action_values.values())
            best_short_labels = sorted(
                label
                for label, value in action_values.items()
                if value == best_value
            )
            current_action = self._resolve_policy_action(
                state,
                current_policy,
                legal_actions,
            )
            chosen_action, kept_current_policy, had_tie = (
                self._choose_improved_action_from_values(
                    legal_actions=legal_actions,
                    action_values_by_action=action_values_by_action,
                    current_action=current_action,
                    rng=self._policy_improvement_rng_for_state(state),
                )
            )
            rows.append(
                {
                    "state": state,
                    "state_label": f"({row}, {col})",
                    "is_terminal": False,
                    "action_values": action_values,
                    "best_action_labels": best_short_labels,
                    "chosen_action": chosen_action,
                    "chosen_label": label_by_action[chosen_action],
                    "chosen_arrow": arrow_by_action[chosen_action],
                    "had_tie": had_tie,
                    "kept_current_policy": kept_current_policy,
                }
            )
        return rows

    def _extract_greedy_policy(
        self,
        environment: Any,
        values: dict[tuple[int, int], float],
        current_policy: dict[tuple[int, int], int],
        *,
        outer_round: int,
    ) -> dict[tuple[int, int], int]:
        """Extract an improved policy with zero-hold and random tie-breaking."""

        improved_policy: dict[tuple[int, int], int] = {}
        for state in environment.iter_states():
            if environment.is_terminal_state(state):
                continue
            legal_actions = environment.get_legal_actions(state)
            if not legal_actions:
                continue

            action_values_by_action = {
                action: self._rounded_action_value(environment, state, action, values)
                for action in legal_actions
            }
            current_action = self._resolve_policy_action(
                state,
                current_policy,
                legal_actions,
            )
            chosen_action, _, _ = self._choose_improved_action_from_values(
                legal_actions=legal_actions,
                action_values_by_action=action_values_by_action,
                current_action=current_action,
                rng=self._policy_improvement_rng_for_state(state),
            )
            improved_policy[state] = chosen_action
        return improved_policy

    def _save_checkpoint(
        self,
        environment: Any,
        *,
        iteration: int,
        outer_round: int,
        eval_step: int | None,
        phase: str,
        values: dict[tuple[int, int], float],
        policy: dict[tuple[int, int], int],
        delta: float,
        note: str,
        evaluation_converged: bool = False,
        action_value_details: list[dict[str, Any]] | None = None,
    ) -> float:
        """Store one planning snapshot and return deterministic policy return."""

        replay = environment.generate_policy_replay(policy)
        policy_return = float(sum(step.reward for step in replay.steps))
        if iteration % CHECKPOINT_FREQUENCY == 0:
            self.iteration_snapshots.append(
                {
                    "iteration": iteration,
                    "outer_round": outer_round,
                    "eval_step": eval_step,
                    "phase": phase,
                    "delta": float(delta),
                    "evaluation_converged": evaluation_converged,
                    "theta": float(self.config.theta),
                    "policy_return": float(policy_return),
                    "start_state_value": float(values.get(environment.start_position, 0.0)),
                    "value_function": deepcopy(values),
                    "policy": deepcopy(policy),
                    "value_grid": environment.build_value_grid(values, precision=4),
                    "policy_grid": environment.build_policy_grid(policy),
                    "action_value_details": deepcopy(action_value_details or []),
                    "replay": replay if phase == "policy_improvement" else None,
                    "note": note,
                    "timestamp": time(),
                }
            )
        return policy_return

    def _policies_equal(
        self,
        left: dict[tuple[int, int], int],
        right: dict[tuple[int, int], int],
    ) -> bool:
        """Return whether two policies choose the same action in every state."""

        return left == right

    def _run_policy_iteration(
        self,
        environment: Any,
        initial_policy: dict[tuple[int, int], int],
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int], dict[str, Any]]:
        """Run Policy Iteration with explicit evaluation sweeps and improvement steps."""

        values = self._initialize_value_function(environment)
        policy = dict(initial_policy)
        deltas: list[float] = []
        start_state_values: list[float] = []
        policy_returns: list[float] = []
        converged = False
        global_step = 0

        policy_returns.append(
            self._save_checkpoint(
                environment,
                iteration=global_step,
                outer_round=0,
                eval_step=None,
                phase="initial",
                values=values,
                policy=policy,
                delta=0.0,
                note=(
                    "Initial random policy with V(s)=0 for all states before "
                    "Policy Evaluation begins."
                ),
            )
        )
        start_state_values.append(values[environment.start_position])
        deltas.append(0.0)

        for outer_round in range(1, self.config.max_iterations + 1):
            policy_before_improvement = dict(policy)

            values, last_eval_delta, global_step, _evaluation_converged = (
                self._run_policy_evaluation_phase(
                    environment,
                    policy,
                    outer_round=outer_round,
                    global_step=global_step,
                    policy_returns=policy_returns,
                    deltas=deltas,
                    start_state_values=start_state_values,
                )
            )

            # Policy Improvement: greedy policy from the converged value table.
            improved_policy = self._extract_greedy_policy(
                environment,
                values,
                policy_before_improvement,
                outer_round=outer_round,
            )
            action_value_details = self._build_action_value_details(
                environment,
                values,
                improved_policy,
                policy_before_improvement,
                outer_round=outer_round,
            )
            policy_stable = self._policies_equal(policy_before_improvement, improved_policy)
            policy = improved_policy

            global_step += 1
            policy_returns.append(
                self._save_checkpoint(
                    environment,
                    iteration=global_step,
                    outer_round=outer_round,
                    eval_step=None,
                    phase="policy_improvement",
                    values=values,
                    policy=policy,
                    delta=last_eval_delta,
                    evaluation_converged=True,
                    action_value_details=action_value_details,
                    note=(
                        f"Policy Improvement after round {outer_round} evaluation "
                        f"converged. Each state selects a maximal action-value; "
                        f"if every legal action is 0 the current action is kept, "
                        f"and remaining ties are broken randomly."
                    ),
                )
            )

            if policy_stable:
                converged = True
                break

        improvement_rounds = sum(
            1
            for snapshot in self.iteration_snapshots
            if snapshot.get("phase") == "policy_improvement"
        )
        history = {
            "delta": deltas,
            "start_state_value": start_state_values,
            "policy_return": policy_returns,
            "converged": converged,
            "method": "policy_iteration",
            "policy_improvement_rounds": improvement_rounds,
        }
        return values, policy, history
