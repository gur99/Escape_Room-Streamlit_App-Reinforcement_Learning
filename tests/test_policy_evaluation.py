"""Regression tests for Policy Evaluation backups."""

from __future__ import annotations

from algorithms.dynamic_programming import DynamicProgrammingAgent, DynamicProgrammingConfig
from environments.room1_dp import Room1DynamicProgrammingEnv


def _reference_policy_evaluation_sweep(
    environment: Room1DynamicProgrammingEnv,
    policy: dict[tuple[int, int], int],
    old_values: dict[tuple[int, int], float],
    *,
    gamma: float,
) -> tuple[dict[tuple[int, int], float], float]:
    """Independent synchronous Policy Evaluation sweep for cross-checking."""

    new_values: dict[tuple[int, int], float] = {}
    eval_delta = 0.0

    for state in sorted(environment.iter_states()):
        old_value = old_values.get(state, 0.0)
        if environment.is_terminal_state(state):
            new_value = 0.0
        else:
            legal_actions = environment.get_legal_actions(state)
            action = policy.get(state, legal_actions[0])
            if action not in legal_actions:
                action = legal_actions[0]
            new_value = 0.0
            for probability, next_state, reward, terminated in environment.get_transition_distribution(
                state,
                action,
            ):
                continuation = 0.0 if terminated else old_values.get(next_state, 0.0)
                new_value += probability * (reward + gamma * continuation)
        quantized_value = round(new_value, 4)
        new_values[state] = quantized_value
        eval_delta = max(eval_delta, abs(quantized_value - old_value))

    return new_values, eval_delta


def test_policy_evaluation_matches_reference_implementation() -> None:
    environment = Room1DynamicProgrammingEnv()
    config = DynamicProgrammingConfig(initial_policy_seed=7, gamma=0.5, theta=1e-8)
    agent = DynamicProgrammingAgent(config)
    policy = agent.build_initial_policy(environment)
    values = agent._initialize_value_function(environment)

    for _ in range(5):
        expected_values, expected_delta = _reference_policy_evaluation_sweep(
            environment,
            policy,
            values,
            gamma=config.gamma,
        )
        actual_values, actual_delta = agent._run_policy_evaluation_sweep(
            environment,
            policy,
            values,
        )
        assert abs(actual_delta - expected_delta) < 1e-12
        for state in environment.iter_states():
            assert abs(actual_values[state] - expected_values[state]) < 1e-12
        values = actual_values


def test_first_sweep_goal_propagation_example() -> None:
    environment = Room1DynamicProgrammingEnv()
    config = DynamicProgrammingConfig(initial_policy_seed=42, gamma=0.5)
    agent = DynamicProgrammingAgent(config)
    policy = agent.build_initial_policy(environment)
    values = agent._initialize_value_function(environment)

    values, _ = agent._run_policy_evaluation_sweep(environment, policy, values)

    assert values[(8, 9)] == 500.0
    assert values[(7, 9)] == 0.0


def test_second_sweep_uses_previous_successor_value() -> None:
    environment = Room1DynamicProgrammingEnv()
    config = DynamicProgrammingConfig(initial_policy_seed=42, gamma=0.5)
    agent = DynamicProgrammingAgent(config)
    policy = agent.build_initial_policy(environment)
    values = agent._initialize_value_function(environment)

    values, _ = agent._run_policy_evaluation_sweep(environment, policy, values)
    values, _ = agent._run_policy_evaluation_sweep(environment, policy, values)

    assert values[(7, 9)] == 250.0


def test_policy_evaluation_resets_to_zero_each_phase() -> None:
    environment = Room1DynamicProgrammingEnv()
    config = DynamicProgrammingConfig(initial_policy_seed=7, theta=1e-6)
    agent = DynamicProgrammingAgent(config)
    policy = agent.build_initial_policy(environment)

    carried_values = {state: 3.5 for state in environment.iter_states()}
    values, _, _, _ = agent._run_policy_evaluation_phase(
        environment,
        policy,
        outer_round=1,
        global_step=0,
        policy_returns=[],
        deltas=[],
        start_state_values=[],
    )

    assert carried_values != values
    step_zero = next(
        snapshot
        for snapshot in agent.iteration_snapshots
        if snapshot.get("outer_round") == 1 and snapshot.get("eval_step") == 0
    )
    assert all(value == 0.0 for state, value in step_zero["value_function"].items())


def test_policy_evaluation_stops_when_all_state_deltas_are_zero() -> None:
    environment = Room1DynamicProgrammingEnv()
    agent = DynamicProgrammingAgent(DynamicProgrammingConfig(initial_policy_seed=42))
    policy = agent.build_initial_policy(environment)

    agent._run_policy_evaluation_phase(
        environment,
        policy,
        outer_round=1,
        global_step=0,
        policy_returns=[],
        deltas=[],
        start_state_values=[],
    )

    sweeps = [
        snapshot
        for snapshot in agent.iteration_snapshots
        if snapshot.get("outer_round") == 1
        and snapshot.get("phase") == "policy_evaluation"
        and int(snapshot.get("eval_step", 0)) > 0
    ]
    assert sweeps
    assert sweeps[-1]["delta"] == 0.0
    assert sweeps[-1]["evaluation_converged"] is True
    assert all(snapshot["delta"] != 0.0 for snapshot in sweeps[:-1])


def test_policy_is_fixed_during_evaluation_snapshots() -> None:
    environment = Room1DynamicProgrammingEnv()
    agent = DynamicProgrammingAgent(DynamicProgrammingConfig(initial_policy_seed=11))
    agent.train(environment)

    for outer_round in (1, 2):
        evaluation_snapshots = [
            snapshot
            for snapshot in agent.iteration_snapshots
            if snapshot.get("outer_round") == outer_round
            and snapshot.get("phase") == "policy_evaluation"
        ]
        policies = [snapshot["policy"] for snapshot in evaluation_snapshots]
        assert policies and all(policy == policies[0] for policy in policies)
