"""Streamlit entry point for the RL Escape Room project.

Room 1 (Dynamic Programming) is a fixed educational maze: users change only
algorithm hyperparameters. Other rooms remain structured placeholders.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import time
from typing import Any

import pandas as pd
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
from utils.plotting import (
    ITERATION_BROWSER_GRID_CELL_SIZE_REM,
    build_grid_html,
    build_grid_legend_html,
    build_max_values_table_html,
    build_metrics_dataframe,
    build_round_metrics_dataframe,
    compute_iteration_browser_grid_height_rem,
    build_slippery_explanation_html,
    build_unified_legend_html,
    build_unified_room_grid_html,
    build_value_evolution_table_html,
    group_planning_rounds,
    render_placeholder_metric,
    render_training_progress_chart,
)
from utils.replay import EpisodeReplay, build_placeholder_replay

ROOM1_REPLAY_ANIMATION_DELAY_SECONDS = 1.0


ROOM_REGISTRY: dict[str, dict[str, Any]] = {
    "Room 1 - Dynamic Programming": {
        "environment_class": Room1DynamicProgrammingEnv,
        "algorithm_class": DynamicProgrammingAgent,
        "optional": False,
        "fixed_environment": True,
    },
    "Room 2 - SARSA": {
        "environment_class": Room2SarsaEnv,
        "algorithm_class": SarsaAgent,
        "optional": False,
        "fixed_environment": False,
    },
    "Room 3 - Q-Learning": {
        "environment_class": Room3QLearningEnv,
        "algorithm_class": QLearningAgent,
        "optional": False,
        "fixed_environment": False,
    },
    "Room 4 - DQN": {
        "environment_class": Room4ContinuousEnv,
        "algorithm_class": DQNAgent,
        "optional": False,
        "fixed_environment": False,
    },
    "Room 5 - Obstacle Avoidance": {
        "environment_class": Room5ObstacleEnv,
        "algorithm_class": ApproximateRLAgent,
        "optional": True,
        "fixed_environment": False,
    },
}

ROOM1_LABEL = "Room 1 - Dynamic Programming"


def _render_dataclass_controls(
    section_title: str,
    config_object: Any,
    key_prefix: str,
    *,
    field_labels: dict[str, str] | None = None,
    include_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Render Streamlit widgets for selected dataclass fields."""

    st.subheader(section_title)

    if not is_dataclass(config_object):
        st.warning("The selected configuration is not a dataclass.")
        return {}

    labels = field_labels or {}
    values: dict[str, Any] = {}
    for field in fields(config_object):
        if include_fields is not None and field.name not in include_fields:
            continue
        # Room 1 no longer lets users pick algorithm type or a fixed initial policy.
        if field.name in {"planning_method", "initial_policy", "initial_policy_seed"}:
            continue

        current_value = getattr(config_object, field.name)
        label = labels.get(field.name, field.name.replace("_", " ").title())
        widget_key = f"{key_prefix}:{field.name}"
        # New default gamma=0.5: version the widget key so old session values do not stick.
        if field.name == "gamma":
            widget_key = f"{key_prefix}:{field.name}:default_0_5"

        if isinstance(current_value, bool):
            values[field.name] = st.checkbox(label, value=current_value, key=widget_key)
        elif isinstance(current_value, int):
            values[field.name] = st.number_input(
                label,
                value=current_value,
                step=1,
                key=widget_key,
            )
        elif isinstance(current_value, float):
            values[field.name] = st.number_input(
                label,
                value=current_value,
                format="%.4f",
                key=widget_key,
            )
        else:
            values[field.name] = st.text_input(
                label,
                value=str(current_value),
                key=widget_key,
            )

    return values


def _build_config_instance(config_object: Any, updates: dict[str, Any]) -> Any:
    """Create a new config instance from widget values."""

    base = {field.name: getattr(config_object, field.name) for field in fields(config_object)}
    base.update(updates)
    return type(config_object)(**base)


def _copy_grid(grid: list[list[str]]) -> list[list[str]]:
    """Create a shallow copy of a two-dimensional grid."""

    return [row[:] for row in grid]


def _is_special_layout_cell(cell_value: str) -> bool:
    """Return whether a layout cell already marks a special location."""

    return any(
        token in str(cell_value).split("/") for token in ("S", "E", "G", "W", "T", "I")
    )


def _build_replay_frames(
    base_grid: list[list[str]],
    replay: Any,
) -> list[list[list[str]]]:
    """Build one grid frame per replay step, including the initial state."""

    frames = [_copy_grid(base_grid)]
    visited_states: list[tuple[int, int]] = []

    for step in replay.steps:
        frame = _copy_grid(base_grid)

        for visit_index, visited_state in enumerate(visited_states, start=1):
            row_idx, col_idx = visited_state
            if frame[row_idx][col_idx] not in {"S", "E", "G", "W"}:
                frame[row_idx][col_idx] = str(visit_index)

        current_state = tuple(step.info["state_after_action"])
        row_idx, col_idx = current_state
        base_cell_value = frame[row_idx][col_idx]
        if base_cell_value not in {"S", "E", "G"}:
            frame[row_idx][col_idx] = (
                "A" if not _is_special_layout_cell(base_cell_value) else f"{base_cell_value}/A"
            )

        frames.append(frame)
        visited_states.append(current_state)

    return frames


def _build_replay_step_description(replay: Any, frame_index: int) -> str:
    """Return a short textual description of the currently displayed replay frame."""

    if frame_index == 0:
        return "Initial state before the replay starts."

    step = replay.steps[frame_index - 1]
    return (
        f"Step {frame_index}: {tuple(step.info['state_before_action'])} "
        f"--{step.info['action_label']}--> {tuple(step.info['state_after_action'])}, "
        f"reward {step.reward:.2f}, transition probability "
        f"{step.info['chosen_transition_probability']:.3f}"
    )


def _render_room_overview(environment: Any, algorithm: Any) -> None:
    """Render high-level educational information for the selected room."""

    environment_snapshot = environment.render()
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

    # Room 1 uses one large unified grid (markers + policy + slip odds) below.
    if hasattr(environment, "build_unified_display_grid"):
        return

    st.markdown("#### Current Layout")
    if "layout_grid" in environment_snapshot:
        st.markdown(build_grid_legend_html("layout"), unsafe_allow_html=True)
        st.markdown(
            build_grid_html(environment_snapshot["layout_grid"], mode="layout"),
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "A graphical layout is currently available only for Room 1. "
            "Other rooms still expose a raw environment snapshot."
        )


def _render_unified_room_grid(
    environment: Any,
    policy: dict[tuple[Any, Any], int],
    *,
    title: str,
    caption: str,
) -> None:
    """Show one large grid with markers, policy arrows, and slip probabilities."""

    unified_grid = environment.build_unified_display_grid(policy)
    explanations = environment.explain_slippery_cells(policy)

    st.markdown(f"#### {title}")
    st.caption(caption)
    st.markdown(build_unified_legend_html(), unsafe_allow_html=True)
    st.markdown(build_unified_room_grid_html(unified_grid), unsafe_allow_html=True)
    with st.expander("Written explanation of slippery cells", expanded=False):
        st.markdown(build_slippery_explanation_html(explanations), unsafe_allow_html=True)


def _render_placeholder_metrics_tab() -> None:
    """Render placeholder metrics and charts for rooms not yet implemented."""

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
        render_training_progress_chart(metrics_frame, y_label="Metric value")


def _render_placeholder_replay_tab() -> None:
    """Render a placeholder replay panel for later rooms."""

    replay = build_placeholder_replay()
    st.info(
        "Replay is wired into the project structure, but full episode playback "
        "will be added after the first room and training loop are implemented."
    )
    st.json(replay.to_dict())


def _render_room1_initial_policy(environment: Any, algorithm: Any) -> None:
    """Show one unified grid with layout, random initial policy, and slip probabilities."""

    policy_key = "room1_display_initial_policy"
    if policy_key not in st.session_state:
        st.session_state[policy_key] = algorithm.build_initial_policy(environment)

    initial_policy = st.session_state[policy_key]
    _render_unified_room_grid(
        environment,
        initial_policy,
        title="Room Grid (Random Initial Policy)",
        caption=(
            "Slippery cells (blue border): one large policy arrow in the center; "
            "transition probabilities appear as numbers at the top, bottom, left, "
            "and right edges. Regular cells show only the policy arrow. "
            "Rewards appear as r = N (Exit / traps). "
            "The initial policy is sampled from legal actions in each cell using a fixed seed "
            "(same policy on every run)."
        ),
    )


def _render_room1_training_result(environment: Any, training_result: dict[str, Any]) -> None:
    """Render Room 1 outputs after Dynamic Programming training."""

    summary = training_result["summary"]
    transition_model = training_result["transition_model"]
    reward_model = training_result["reward_model"]

    metric_columns = st.columns(5)
    metric_columns[0].metric("Method", "Policy Iteration")
    metric_columns[1].metric("Episodes", summary["iterations"])
    metric_columns[2].metric("Converged", "Yes" if summary["converged"] else "No")
    metric_columns[3].metric("Start State Value", f"{summary['start_state_value']:.4f}")
    metric_columns[4].metric("Policy Return", f"{summary['replay_total_reward']:.4f}")

    st.markdown("#### Transition Model (fixed)")
    slippery = transition_model["slippery_cells"]
    st.write(transition_model["legal_action_filtering"])
    st.write(transition_model["regular_cells"])
    st.write(slippery["description"])
    st.write(transition_model["terminal_state"])

    st.markdown("#### Reward Model (fixed)")
    st.write(f"Exit Door reward: {reward_model['exit_door_reward']}")
    st.write(f"Trap penalty: {reward_model['trap_penalty']}")
    st.write(reward_model["trap_behavior"])
    st.write(reward_model["exit_door"])

    _render_unified_room_grid(
        environment,
        training_result["policy"],
        title="Room Grid (Final Policy)",
        caption=(
            "Same grid after training: rewards as r = N, center policy arrow on every "
            "actionable cell, and edge transition probabilities on slippery cells only."
        ),
    )

    st.markdown("#### Final Value Function")
    st.markdown(build_grid_legend_html("value"), unsafe_allow_html=True)
    st.markdown(
        build_grid_html(
            training_result["value_grid"],
            mode="value",
            base_grid=training_result["layout_grid"],
        ),
        unsafe_allow_html=True,
    )


def _render_room1_metrics_tab(training_result: dict[str, Any]) -> None:
    """Render reward and delta convergence for a selected policy-improvement round."""

    snapshots = training_result.get("iteration_snapshots", [])
    if not snapshots:
        st.info("No convergence history is available yet.")
        return

    rounds = group_planning_rounds(snapshots)
    completed_rounds = sorted(
        round_number
        for round_number, round_data in rounds.items()
        if round_number >= 1 and round_data.get("improvement") is not None
    )
    if not completed_rounds:
        st.info("No completed policy-improvement rounds are available yet.")
        return

    history = training_result["history"]
    iteration_count = int(history.get("policy_improvement_rounds", len(completed_rounds)))
    iteration_count = min(iteration_count, len(completed_rounds))

    st.markdown("#### Metrics by Policy Improvement Round")
    st.caption(
        "Select a round to inspect Policy Evaluation sweeps within that episode. "
        "Total Reward is the deterministic policy return at each checkpoint; "
        "Delta is the maximum |V_new(s) - V_old(s)| after each sweep."
    )

    selected_iteration = st.slider(
        "Policy Improvement Round",
        min_value=1,
        max_value=iteration_count,
        value=iteration_count,
        key="room1_metrics_round_slider",
    )
    selected_round = completed_rounds[selected_iteration - 1]
    round_data = rounds[selected_round]
    evaluation_steps = round_data.get("evaluation_steps", [])
    metrics_frame = build_round_metrics_dataframe(evaluation_steps)

    if metrics_frame.empty:
        st.warning("No Policy Evaluation sweeps were recorded for this round.")
        return

    improvement = round_data.get("improvement")
    last_delta = float(metrics_frame["Delta"].iloc[-1])
    last_reward = float(
        improvement.get("policy_return", metrics_frame["Total Reward"].iloc[-1])
        if improvement
        else metrics_frame["Total Reward"].iloc[-1]
    )
    last_start_value = float(
        improvement.get("start_state_value", 0.0)
        if improvement
        else evaluation_steps[-1].get("start_state_value", 0.0)
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Round", selected_iteration)
    metric_columns[1].metric("Policy Return", f"{last_reward:.4f}")
    metric_columns[2].metric("Final Delta", f"{last_delta:.6f}")
    metric_columns[3].metric("Start State Value", f"{last_start_value:.4f}")

    reward_column, delta_column = st.columns(2)
    with reward_column:
        st.markdown("##### Policy Return")
        render_training_progress_chart(
            metrics_frame[["Total Reward"]],
            y_label="Total Reward",
            title=f"Policy Return — Round {selected_iteration}",
        )
    with delta_column:
        st.markdown("##### Delta Convergence")
        render_training_progress_chart(
            metrics_frame[["Delta"]],
            y_label="Delta",
            title=f"Delta Convergence — Round {selected_iteration}",
            y_tickformat=".6f",
        )

    with st.expander("Full training summary", expanded=False):
        if not history["delta"]:
            st.info("No global convergence history is available.")
            return

        performance_frame = pd.DataFrame(
            {
                "Iteration": list(range(len(history["policy_return"]))),
                "Total Reward": history["policy_return"],
            }
        ).set_index("Iteration")
        render_training_progress_chart(
            performance_frame,
            y_label="Total Reward",
            title="Policy Performance Across All Checkpoints",
            x_tick_interval=10,
        )

        delta_frame = pd.DataFrame(
            {
                "Iteration": list(range(len(history["delta"]))),
                "Delta": history["delta"],
            }
        ).set_index("Iteration")
        render_training_progress_chart(
            delta_frame,
            y_label="Delta",
            title="Delta Convergence Across All Checkpoints",
            y_tickformat=".6f",
            x_tick_interval=10,
        )


def _count_policy_evaluation_sweeps(evaluation_steps: list[dict[str, Any]]) -> int:
    """Return the number of computed Bellman sweeps shown after S0 in the PE table."""

    return sum(1 for snapshot in evaluation_steps if int(snapshot.get("eval_step", 0)) > 0)


def _limit_replay_steps(replay: EpisodeReplay, max_steps: int | None) -> EpisodeReplay:
    """Truncate a replay so it does not exceed the Policy Evaluation sweep count."""

    if max_steps is None or max_steps <= 0 or len(replay.steps) <= max_steps:
        return replay
    return EpisodeReplay(
        room_name=replay.room_name,
        algorithm_name=replay.algorithm_name,
        steps=replay.steps[:max_steps],
    )


def _list_actionable_states(layout_grid: list[list[str]]) -> list[tuple[int, int]]:
    """Return states shown in the Policy Evaluation value table.

    Walls and the goal/exit cell are excluded because they are not updated
    during evaluation sweeps.
    """

    excluded_tokens = {"W", "E", "G"}
    states: list[tuple[int, int]] = []
    for row_idx, row in enumerate(layout_grid):
        for col_idx, cell in enumerate(row):
            base_token = str(cell).split("/")[0]
            if base_token.startswith("I|"):
                base_token = "I"
            if base_token in excluded_tokens:
                continue
            states.append((row_idx, col_idx))
    return states


def _render_room1_iteration_browser(
    training_result: dict[str, Any],
) -> dict[str, Any] | None:
    """Browse each policy-improvement round and return the current selection."""

    snapshots = training_result.get("iteration_snapshots", [])
    if not snapshots:
        st.info("No iteration snapshots were saved.")
        return None

    rounds = group_planning_rounds(snapshots)
    completed_rounds = sorted(
        round_number
        for round_number, round_data in rounds.items()
        if round_number >= 1 and round_data.get("improvement") is not None
    )
    if not completed_rounds:
        st.info("No Policy Improvement rounds were recorded.")
        return None

    history = training_result.get("history", {})
    actual_round_count = int(history.get("policy_improvement_rounds", len(completed_rounds)))
    iteration_count = min(actual_round_count, len(completed_rounds))
    completed_rounds = completed_rounds[:iteration_count]

    st.markdown("#### Policy Iteration Rounds")
    st.caption(
        "Policy Evaluation keeps the policy fixed and updates only V(s) with synchronous "
        "Bellman sweeps until every state delta is 0. Policy Improvement then selects the best "
        "legal action in every state."
    )

    selected_iteration = st.slider(
        "Policy Improvement Iteration",
        min_value=1,
        max_value=iteration_count,
        value=iteration_count,
        key="room1_policy_round_slider",
    )
    selected_round = completed_rounds[selected_iteration - 1]
    round_data = rounds[selected_round]
    evaluation_steps = round_data.get("evaluation_steps", [])
    improvement = round_data.get("improvement")
    actionable_states = _list_actionable_states(training_result["layout_grid"])
    ordered_evaluation_steps = sorted(
        evaluation_steps,
        key=lambda snapshot: int(snapshot.get("eval_step", 0)),
    )
    bellman_sweeps = [
        snapshot
        for snapshot in ordered_evaluation_steps
        if int(snapshot.get("eval_step", 0)) > 0
    ]
    last_bellman_sweep = bellman_sweeps[-1] if bellman_sweeps else None

    if evaluation_steps:
        meta_columns = st.columns(4)
        meta_columns[0].metric("Iteration", selected_iteration)
        meta_columns[1].metric("Evaluation Sweeps", len(bellman_sweeps))
        if last_bellman_sweep is not None:
            meta_columns[2].metric(
                "Max Δ (last sweep)",
                f"{last_bellman_sweep['delta']:.6f}",
            )
            meta_columns[3].metric("Theta", f"{last_bellman_sweep.get('theta', 0):g}")
            if last_bellman_sweep.get("evaluation_converged"):
                st.success("Policy Evaluation converged before Policy Improvement.")
        else:
            meta_columns[2].metric("Max Δ (last sweep)", "0.000000")
            meta_columns[3].metric("Theta", f"{evaluation_steps[-1].get('theta', 0):g}")
    elif improvement:
        st.warning("This round has Policy Improvement data but no evaluation sweeps.")

    final_value_snapshot = last_bellman_sweep if last_bellman_sweep else improvement
    vs_table_expanded = st.session_state.get("room1_vs_table_expanded", False)
    layout_grid = training_result["layout_grid"]
    grid_row_count = len(layout_grid)
    iteration_panel_height_rem = compute_iteration_browser_grid_height_rem(grid_row_count)
    prior_value_function = None
    if evaluation_steps and int(ordered_evaluation_steps[0].get("eval_step", -1)) != 0:
        prior_value_function = {
            state: 0.0 for state in actionable_states
        }

    current_policy_snapshot = ordered_evaluation_steps[0] if ordered_evaluation_steps else improvement
    if current_policy_snapshot and current_policy_snapshot.get("policy_grid"):
        st.markdown("##### Current Policy (fixed during Policy Evaluation)")
        st.caption(
            "During Policy Evaluation the policy does not change. Each sweep updates "
            "only V(s) using the Bellman expectation equation under this policy."
        )
        st.markdown(build_grid_legend_html("policy"), unsafe_allow_html=True)
        st.markdown(
            build_grid_html(
                current_policy_snapshot["policy_grid"],
                mode="policy",
                base_grid=layout_grid,
                cell_size_rem=ITERATION_BROWSER_GRID_CELL_SIZE_REM,
            ),
            unsafe_allow_html=True,
        )

    evaluation_column, value_before_column = st.columns([0.68, 1.32], gap="medium")
    with evaluation_column:
        evaluation_header, evaluation_toggle = st.columns([4, 1])
        with evaluation_header:
            st.markdown("##### V(s) During Policy Evaluation")
        with evaluation_toggle:
            if vs_table_expanded:
                if st.button("Collapse", key="room1_vs_table_collapse", use_container_width=True):
                    st.session_state["room1_vs_table_expanded"] = False
                    st.rerun()
            elif st.button("Expand", key="room1_vs_table_expand", use_container_width=True):
                st.session_state["room1_vs_table_expanded"] = True
                st.rerun()
        st.caption(
            "S0 initializes every V(s) to 0. Each S_k uses only the previous table "
            "and the fixed policy: R + gamma * V_old(successor). "
            "Example: S1 gives (8,9)=500.0 and (7,9)=0 because V_old(8,9)=0. "
            "Per-state Δ is |last step − previous step|; sweeps stop when all Δ are 0."
        )
        st.markdown(
            build_value_evolution_table_html(
                evaluation_steps,
                states=actionable_states,
                expanded=vs_table_expanded,
                panel_height_rem=iteration_panel_height_rem,
                prior_value_function=prior_value_function,
            ),
            unsafe_allow_html=True,
        )

    with value_before_column:
        st.markdown("##### V(s) Before Policy Update")
        st.caption(
            "Converged V(s) after Policy Evaluation (all state deltas are 0), before "
            "Policy Improvement is applied. Arrows show the fixed policy used "
            "during evaluation."
        )
        if final_value_snapshot and final_value_snapshot.get("value_grid"):
            st.markdown(build_grid_legend_html("value"), unsafe_allow_html=True)
            st.markdown(
                build_grid_html(
                    final_value_snapshot["value_grid"],
                    mode="value",
                    base_grid=layout_grid,
                    cell_size_rem=ITERATION_BROWSER_GRID_CELL_SIZE_REM,
                    overlay_policy_grid=final_value_snapshot.get("policy_grid"),
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("No converged value grid is available for this iteration.")

    max_values_column, policy_grid_column = st.columns([0.68, 1.32], gap="medium")
    with max_values_column:
        st.markdown("##### Max Values")
        st.caption(
            "For each state, all legal action-values are shown. Blue cells are "
            "maximal values; * marks the action selected for the updated policy."
        )
        if improvement:
            st.markdown(
                build_max_values_table_html(
                    improvement.get("action_value_details", []),
                    panel_height_rem=iteration_panel_height_rem,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("Max action values are not available for this round.")

    with policy_grid_column:
        st.markdown("##### Updated Policy Grid")
        if improvement:
            st.caption(
                "After V(s) converges, every legal action is evaluated and the policy "
                "selects the action with the maximum expected return."
            )
            st.markdown(build_grid_legend_html("policy"), unsafe_allow_html=True)
            st.markdown(
                build_grid_html(
                    improvement["policy_grid"],
                    mode="policy",
                    base_grid=layout_grid,
                    cell_size_rem=ITERATION_BROWSER_GRID_CELL_SIZE_REM,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("Policy Improvement for this round is not available.")

    initial_round = rounds.get(0, {}).get("initial")
    if selected_round == 1 and initial_round:
        with st.expander("Initial random policy before iteration 1", expanded=False):
            st.markdown(build_grid_legend_html("policy"), unsafe_allow_html=True)
            st.markdown(
                build_grid_html(initial_round["policy_grid"], mode="policy"),
                unsafe_allow_html=True,
            )

    return {
        "selected_iteration": selected_iteration,
        "improvement": improvement,
        "layout_grid": layout_grid,
        "policy_evaluation_sweep_count": _count_policy_evaluation_sweeps(
            ordered_evaluation_steps
        ),
    }


def _render_room1_iteration_replay(
    training_result: dict[str, Any],
    iteration_selection: dict[str, Any] | None,
) -> None:
    """Render a step-by-step replay for the policy at the selected iteration."""

    layout_grid = training_result["layout_grid"]
    selected_iteration = (
        int(iteration_selection["selected_iteration"])
        if iteration_selection is not None
        else None
    )
    improvement = (
        iteration_selection.get("improvement")
        if iteration_selection is not None
        else None
    )
    if improvement and improvement.get("replay") is not None:
        replay = improvement["replay"]
    else:
        replay = training_result["replay"]

    policy_evaluation_sweep_count = (
        int(iteration_selection["policy_evaluation_sweep_count"])
        if iteration_selection is not None
        and iteration_selection.get("policy_evaluation_sweep_count") is not None
        else None
    )
    replay = _limit_replay_steps(replay, policy_evaluation_sweep_count)

    st.markdown("---")
    if selected_iteration is not None:
        st.markdown("#### Policy Episode Replay")
        replay_limit_note = (
            f" Replay is limited to {policy_evaluation_sweep_count} steps, matching "
            "the number of Policy Evaluation sweeps (S1…S_k) for this iteration."
            if policy_evaluation_sweep_count
            else ""
        )
        st.caption(
            f"Step-by-step rollout for the updated policy at Policy Improvement "
            f"Iteration {selected_iteration}.{replay_limit_note}"
        )
    else:
        st.markdown("#### Final Policy Episode Replay")

    if not replay.steps:
        st.info("The replay is empty because no solution path was generated.")
        return

    replay_steps = len(replay.steps)
    replay_total_reward = round(sum(step.reward for step in replay.steps), 4)
    goal_reached = bool(replay.steps and replay.steps[-1].terminated)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Replay Total Reward", f"{replay_total_reward:.4f}")
    metric_columns[1].metric("Reached Exit", "Yes" if goal_reached else "No")
    metric_columns[2].metric("Replay Steps", replay_steps)
    metric_columns[3].metric(
        "Animation Delay (seconds)",
        f"{ROOM1_REPLAY_ANIMATION_DELAY_SECONDS:g}",
    )

    replay_frames = _build_replay_frames(layout_grid, replay)
    replay_key = (
        f"replay:{training_result['planning_method']}:"
        f"{selected_iteration}:{replay_steps}"
    )

    st.markdown(build_grid_legend_html("replay"), unsafe_allow_html=True)
    replay_grid_placeholder = st.empty()
    replay_caption_placeholder = st.empty()

    replay_grid_placeholder.markdown(
        build_grid_html(
            replay_frames[0],
            mode="replay",
            base_grid=layout_grid,
        ),
        unsafe_allow_html=True,
    )
    replay_caption_placeholder.caption(_build_replay_step_description(replay, 0))

    if st.button("Play Step-by-Step Replay", key=f"{replay_key}:play"):
        for frame_index, frame in enumerate(replay_frames):
            replay_grid_placeholder.markdown(
                build_grid_html(
                    frame,
                    mode="replay",
                    base_grid=layout_grid,
                ),
                unsafe_allow_html=True,
            )
            replay_caption_placeholder.caption(
                _build_replay_step_description(replay, frame_index)
            )
            if frame_index < len(replay_frames) - 1:
                time.sleep(ROOM1_REPLAY_ANIMATION_DELAY_SECONDS)

    replay_rows: list[dict[str, Any]] = []
    for step_index, step in enumerate(replay.steps, start=1):
        replay_rows.append(
            {
                "step": step_index,
                "from_state": tuple(step.info["state_before_action"]),
                "action": step.info["action_label"],
                "to_state": tuple(step.info["state_after_action"]),
                "reward": step.reward,
                "transition_probability": step.info["chosen_transition_probability"],
                "terminated": step.terminated,
                "truncated": step.truncated,
            }
        )

    st.markdown("#### Replay Steps")
    st.dataframe(pd.DataFrame(replay_rows), use_container_width=True)


def _render_room1_replay_tab(training_result: dict[str, Any]) -> None:
    """Render episode replay and the iteration snapshot browser."""

    iteration_selection = _render_room1_iteration_browser(training_result)
    _render_room1_iteration_replay(training_result, iteration_selection)


def main() -> None:
    """Launch the Streamlit application."""

    st.set_page_config(page_title="RL Escape Room", layout="wide")
    st.title("Reinforcement Learning Escape Room")
    st.caption(
        "Room 1 is a fixed GridWorld for studying Dynamic Programming. "
        "Only algorithm hyperparameters are editable."
    )

    selected_room_label = st.sidebar.selectbox(
        "Choose a room",
        list(ROOM_REGISTRY.keys()),
    )
    room_definition = ROOM_REGISTRY[selected_room_label]

    environment_class = room_definition["environment_class"]
    algorithm_class = room_definition["algorithm_class"]
    preview_environment = environment_class()
    preview_algorithm = algorithm_class()

    with st.sidebar:
        if room_definition.get("fixed_environment"):
            st.info(
                "Environment is fixed for this room. "
                "You can only change the algorithm hyperparameters."
            )
            environment_values: dict[str, Any] = {}
        else:
            environment_values = _render_dataclass_controls(
                "Environment Parameters",
                preview_environment.config,
                key_prefix=f"{selected_room_label}:environment",
            )

        if selected_room_label == ROOM1_LABEL:
            algorithm_values = _render_dataclass_controls(
                "Algorithm Hyperparameters",
                preview_algorithm.config,
                key_prefix=f"{selected_room_label}:algorithm",
                include_fields={
                    "gamma",
                    "theta",
                    "max_iterations",
                },
                field_labels={
                    "gamma": "Discount Factor (γ)",
                    "theta": "Theta (convergence threshold)",
                    "max_iterations": "Max Iterations",
                },
            )
            st.sidebar.caption(
                "The agent always runs Policy Iteration (Evaluation + Improvement) "
                "from a random legal initial policy."
            )
        else:
            algorithm_values = _render_dataclass_controls(
                "Algorithm Hyperparameters",
                preview_algorithm.config,
                key_prefix=f"{selected_room_label}:algorithm",
            )

    if room_definition.get("fixed_environment"):
        environment = environment_class()
    else:
        environment = environment_class(
            _build_config_instance(preview_environment.config, environment_values)
        )
    algorithm = algorithm_class(
        _build_config_instance(preview_algorithm.config, algorithm_values)
    )
    training_result_key = f"training_result:{selected_room_label}"

    st.sidebar.markdown("---")
    st.sidebar.write("Adjust algorithm hyperparameters, then start training.")

    start_button = st.sidebar.button("Start Training")
    stop_button = st.sidebar.button("Stop Training")

    if start_button:
        if selected_room_label == ROOM1_LABEL:
            with st.spinner(
                "Running Policy Iteration (Evaluation + Improvement) on Room 1..."
            ):
                run_initial_policy = algorithm.build_initial_policy(environment)
                st.session_state["room1_display_initial_policy"] = run_initial_policy
                st.session_state[training_result_key] = algorithm.train(
                    environment,
                    initial_policy=run_initial_policy,
                )
            st.success(
                "Room 1 training finished. Inspect the value function, policy, "
                "graphs, and iteration snapshots below."
            )
        else:
            st.warning(
                "Only Room 1 is implemented at this stage. The other rooms remain placeholders."
            )
    if stop_button:
        st.info(
            "Training in Room 1 runs as a short planning step, so there is no active job to stop."
        )

    if room_definition["optional"]:
        st.sidebar.info("This room is optional and will be implemented after the core rooms.")

    overview_tab, metrics_tab, replay_tab = st.tabs(
        ["Overview", "Metrics", "Replay / Iterations"]
    )

    with overview_tab:
        _render_room_overview(environment, algorithm)
        if selected_room_label == ROOM1_LABEL:
            _render_room1_initial_policy(environment, algorithm)
            room1_result = st.session_state.get(training_result_key)
            if room1_result is None:
                st.info(
                    "Press 'Start Training' to run Value Iteration or Policy Iteration for Room 1."
                )
            else:
                _render_room1_training_result(environment, room1_result)

    with metrics_tab:
        if selected_room_label == ROOM1_LABEL:
            room1_result = st.session_state.get(training_result_key)
            if room1_result is None:
                st.info("Training metrics will appear here after Room 1 finishes planning.")
            else:
                _render_room1_metrics_tab(room1_result)
        else:
            _render_placeholder_metrics_tab()

    with replay_tab:
        if selected_room_label == ROOM1_LABEL:
            room1_result = st.session_state.get(training_result_key)
            if room1_result is None:
                st.info("Replay and iteration snapshots will appear after Room 1 training.")
            else:
                _render_room1_replay_tab(room1_result)
        else:
            _render_placeholder_replay_tab()


if __name__ == "__main__":
    main()
