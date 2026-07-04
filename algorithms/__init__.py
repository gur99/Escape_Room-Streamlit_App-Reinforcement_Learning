"""Algorithm package for the RL Escape Room project."""

from algorithms.approximate_rl import ApproximateRLAgent, ApproximateRLConfig
from algorithms.base import RLAlgorithmBase
from algorithms.dqn import DQNAgent, DQNConfig
from algorithms.dynamic_programming import (
    DynamicProgrammingAgent,
    DynamicProgrammingConfig,
)
from algorithms.q_learning import QLearningAgent, QLearningConfig
from algorithms.sarsa import SarsaAgent, SarsaConfig

__all__ = [
    "ApproximateRLAgent",
    "ApproximateRLConfig",
    "DQNAgent",
    "DQNConfig",
    "DynamicProgrammingAgent",
    "DynamicProgrammingConfig",
    "QLearningAgent",
    "QLearningConfig",
    "RLAlgorithmBase",
    "SarsaAgent",
    "SarsaConfig",
]
