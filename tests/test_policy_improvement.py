"""Regression tests for Policy Improvement tie-breaking."""

from __future__ import annotations

from random import Random

from algorithms.dynamic_programming import DynamicProgrammingAgent, DynamicProgrammingConfig
from environments.room1_dp import Room1DynamicProgrammingEnv


def test_keep_current_action_when_all_values_are_zero() -> None:
    agent = DynamicProgrammingAgent(DynamicProgrammingConfig(initial_policy_seed=42))
    rng = Random(1)

    chosen, kept, had_tie = agent._choose_improved_action_from_values(
        legal_actions=[0, 1, 2, 3],
        action_values_by_action={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        current_action=2,
        rng=rng,
    )

    assert chosen == 2
    assert kept is True
    assert had_tie is False


def test_random_tie_break_when_multiple_actions_share_maximum() -> None:
    agent = DynamicProgrammingAgent(DynamicProgrammingConfig(initial_policy_seed=42))
    rng = Random(7)

    chosen, kept, had_tie = agent._choose_improved_action_from_values(
        legal_actions=[0, 1, 2, 3],
        action_values_by_action={0: 0.0, 1: 0.0, 2: -0.5, 3: -1.0},
        current_action=3,
        rng=rng,
    )

    assert chosen in {0, 1}
    assert kept is False
    assert had_tie is True


def test_choose_single_best_action() -> None:
    agent = DynamicProgrammingAgent(DynamicProgrammingConfig(initial_policy_seed=42))
    rng = Random(1)

    chosen, kept, had_tie = agent._choose_improved_action_from_values(
        legal_actions=[0, 1, 2, 3],
        action_values_by_action={0: 1.4, 1: 0.9, 2: 0.7, 3: -1.0},
        current_action=3,
        rng=rng,
    )

    assert chosen == 0
    assert kept is False
    assert had_tie is False


def test_policy_improvement_is_reproducible_for_fixed_seed() -> None:
    environment = Room1DynamicProgrammingEnv()
    config = DynamicProgrammingConfig(initial_policy_seed=42)

    agent_a = DynamicProgrammingAgent(config)
    agent_b = DynamicProgrammingAgent(config)
    policy = agent_a.build_initial_policy(environment)

    values = {state: 0.0 for state in environment.iter_states()}
    improved_a = agent_a._extract_greedy_policy(
        environment,
        values,
        policy,
        outer_round=1,
    )
    improved_b = agent_b._extract_greedy_policy(
        environment,
        values,
        policy,
        outer_round=1,
    )

    assert improved_a == improved_b
