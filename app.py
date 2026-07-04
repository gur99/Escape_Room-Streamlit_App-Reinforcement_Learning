"""Streamlit entry point for the RL Escape Room project.

Stage 1 intentionally provides only the project skeleton:
- modular file structure
- Gymnasium-style environment interfaces
- algorithm placeholders
- Streamlit UI shell for future training controls
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import streamlit as st

from algorithms.approximate_rl import ApproximateRLAgent
from algorithms.dqn import DQNAgent
from algorithms.dynamic_programming import DynamicProgrammingAgent
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SarsaAgent
from environments.room1_dp import Room1DynamicProgrammingEnv
from environments.room2_sarsa import Room2SarsaEnv
from environments.room3_qlearning import Room3QLearningEnv
from environments.room4_continuous import Room4ContinuousEnv
from environments.room5_obstacles import Room5ObstacleEnv
from utils.metrics import TrainingHistory
from utils.plotting import build_metrics_dataframe, render_placeholder_metric
from utils.replay import build_placeholder_replay


ROOM_REGISTRY: dict[str, dict[str, Any]] = {
    "Room 1 - Dynamic Programming": {
        "environment_class": Room1DynamicProgrammingEnv,
        "algorithm_class": DynamicProgrammingAgent,
        "optional": False,
    },
    "Room 2 - SARSA": {
        "environment_class": Room2SarsaEnv,
        "algorithm_class": SarsaAgent,
        "optional": False,
    },
    "Room 3 - Q-Learning": {
        "environment_class": Room3QLearningEnv,
        "algorithm_class": QLearningAgent,
        "optional": False,
    },
    "Room 4 - DQN": {
        "environment_class": Room4ContinuousEnv,
        "algorithm_class": DQNAgent,
        "optional": False,
    },
    "Room 5 - Obstacle Avoidance": {
        "environment_class": Room5ObstacleEnv,
        "algorithm_class": ApproximateRLAgent,
        "optional": True,
    },
}


def _render_dataclass_controls(section_title: str, config_object: Any) -> dict[str, Any]:
    """Render Streamlit widgets for a dataclass and return the chosen values."""

    st.subheader(section_title)

    if not is_dataclass(config_object):
        st.warning("The selected configuration is not a dataclass.")
        return {}

    values: dict[str, Any] = {}
    for field in fields(config_object):
        current_value = getattr(config_object, field.name)
        label = field.name.replace("_", " ").title()

        if isinstance(current_value, bool):
            values[field.name] = st.checkbox(label, value=current_value)
        elif isinstance(current_value, int):
            values[field.name] = st.number_input(label, value=current_value, step=1)
        elif isinstance(current_value, float):
            values[field.name] = st.number_input(
                label,
                value=current_value,
                format="%.4f",
            )
        else:
            values[field.name] = st.text_input(label, value=str(current_value))

    return values


def _render_room_overview(environment: Any, algorithm: Any) -> None:
    """Render high-level educational information for the selected room."""

    st.markdown(f"### {environment.room_title}")
    st.write(environment.room_summary)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Environment")
        st.write(f"**State Space:** {environment.state_space_description}")
        st.write(f"**Action Space:** {environment.action_space_description}")
        st.write(f"**Reward Function:** {environment.reward_description}")
        st.write(f"**Observation Space:** {environment.observation_space.description}")
        st.write(f"**Action Space Object:** {environment.action_space.description}")

    with col2:
        st.markdown("#### Algorithm")
        st.write(f"**Name:** {algorithm.algorithm_name}")
        st.write(f"**Learning Type:** {algorithm.learning_type}")
        st.write(f"**Summary:** {algorithm.algorithm_summary}")

    st.markdown("#### Current Layout")
    st.json(environment.render())


def _render_metrics_tab() -> None:
    """Render placeholder metrics and charts for the skeleton stage."""

    history = TrainingHistory()
    metrics_frame = build_metrics_dataframe(history)

    cols = st.columns(5)
    render_placeholder_metric(cols[0], "Reward / Episode", history.rewards)
    render_placeholder_metric(cols[1], "Episode Length", history.episode_lengths)
    render_placeholder_metric(cols[2], "Success Rate", history.success_rates)
    render_placeholder_metric(cols[3], "Exploration Rate", history.exploration_rates)
    render_placeholder_metric(cols[4], "Loss", history.losses)

    if metrics_frame.empty:
        st.info(
            "Stage 1 includes the metrics pipeline and chart placeholders. "
            "Real training curves will appear in the next implementation stages."
        )
    else:
        st.line_chart(metrics_frame)


def _render_replay_tab() -> None:
    """Render a placeholder replay panel."""

    replay = build_placeholder_replay()
    st.info(
        "Replay is wired into the project structure, but full episode playback "
        "will be added after the first room and training loop are implemented."
    )
    st.json(replay.to_dict())


def main() -> None:
    """Launch the Stage 1 Streamlit application shell."""

    st.set_page_config(page_title="RL Escape Room", layout="wide")
    st.title("Reinforcement Learning Escape Room")
    st.caption(
        "Stage 1: a runnable project skeleton for a multi-room RL final project."
    )

    selected_room_label = st.sidebar.selectbox(
        "Choose a room",
        list(ROOM_REGISTRY.keys()),
    )
    room_definition = ROOM_REGISTRY[selected_room_label]

    environment_class = room_definition["environment_class"]
    algorithm_class = room_definition["algorithm_class"]

    environment = environment_class()
    algorithm = algorithm_class()

    st.sidebar.markdown("---")
    st.sidebar.write("Adjustments are visible in the UI now and will drive training in later stages.")
    _render_dataclass_controls("Environment Parameters", environment.config)
    _render_dataclass_controls("Algorithm Hyperparameters", algorithm.config)

    start_button = st.sidebar.button("Start Training")
    stop_button = st.sidebar.button("Stop Training")
    save_button = st.sidebar.button("Save Run")

    if start_button:
        st.warning(
            "Training is intentionally not implemented in Stage 1. "
            "The next stage should add Room 1 and its Dynamic Programming algorithm."
        )
    if stop_button:
        st.info("No active training session is running in the skeleton stage.")
    if save_button:
        st.info("Saving run results will be enabled once training outputs exist.")

    if room_definition["optional"]:
        st.sidebar.info("This room is optional and will be implemented after the core rooms.")

    overview_tab, metrics_tab, replay_tab = st.tabs(
        ["Overview", "Metrics", "Replay"]
    )

    with overview_tab:
        _render_room_overview(environment, algorithm)

    with metrics_tab:
        _render_metrics_tab()

    with replay_tab:
        _render_replay_tab()


if __name__ == "__main__":
    main()
