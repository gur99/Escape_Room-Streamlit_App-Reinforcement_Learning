"""Plotting helpers for Streamlit visualizations."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd

from utils.metrics import TrainingHistory


def build_visit_counts_display_grid(
    layout_grid: list[list[str]],
    visit_counts: dict[tuple[int, int], int],
) -> list[list[str]]:
    """Build a numeric visit-count grid; walls remain ``W``."""

    display: list[list[str]] = []
    for row_idx, row in enumerate(layout_grid):
        display_row: list[str] = []
        for col_idx, cell in enumerate(row):
            if _base_layout_token(cell) == "W":
                display_row.append("W")
            else:
                display_row.append(str(int(visit_counts.get((row_idx, col_idx), 0))))
        display.append(display_row)
    return display


def build_metrics_dataframe(history: TrainingHistory) -> pd.DataFrame:
    """Convert collected metrics into a tabular format for plotting.

    The function returns an empty dataframe when no training has been performed
    yet, which is the expected behavior during Stage 1.
    """

    series_map = {
        "reward": history.rewards,
        "episode_length": history.episode_lengths,
        "success_rate": history.success_rates,
        "exploration_rate": history.exploration_rates,
        "loss": history.losses,
    }

    non_empty_series = {name: values for name, values in series_map.items() if values}
    if not non_empty_series:
        return pd.DataFrame()

    return pd.DataFrame(dict(non_empty_series))


def render_placeholder_metric(container: Any, label: str, values: list[float]) -> None:
    """Render a single compact metric summary in a Streamlit column."""

    if values:
        container.metric(label=label, value=f"{values[-1]:.3f}")
    else:
        container.metric(label=label, value="N/A")


# Shared fixed cell size used by every grid in the app.
GRID_CELL_SIZE_REM = 4.0
# Slightly smaller grids for the iteration browser so two fit side by side.
ITERATION_BROWSER_GRID_CELL_SIZE_REM = 3.15
ITERATION_BROWSER_GRID_SPACING_REM = 0.16

# Inset frame drawn inside the cell so rounded corners stay complete after deploy.
GRID_CELL_FRAME_WIDTH_PX = 2
GRID_CELL_HIGHLIGHT_FRAME_WIDTH_PX = 3
GRID_CELL_DEFAULT_FRAME_COLOR = "#cbd5e1"
GRID_CELL_FRAME_CSS = f"""
  border: none;
  outline: none;
  box-shadow: inset 0 0 0 var(--cell-frame-width, {GRID_CELL_FRAME_WIDTH_PX}px) var(--cell-frame-color, {GRID_CELL_DEFAULT_FRAME_COLOR});
  background-clip: padding-box;
  box-sizing: border-box;
"""


def _cell_frame_css_variable(
    border_color: str,
    frame_width_px: int | None = None,
) -> str:
    """Return inline CSS variables for the inset cell frame."""

    width = frame_width_px if frame_width_px is not None else GRID_CELL_FRAME_WIDTH_PX
    return f"--cell-frame-color:{border_color};--cell-frame-width:{width}px;"


def compute_iteration_browser_grid_height_rem(grid_rows: int) -> float:
    """Return the rendered grid height in rem, including the axis header row."""

    table_rows = grid_rows + 1
    return (
        table_rows * ITERATION_BROWSER_GRID_CELL_SIZE_REM
        + (table_rows - 1) * ITERATION_BROWSER_GRID_SPACING_REM
    )


def build_training_progress_figure(
    frame: pd.DataFrame,
    *,
    y_label: str,
    title: str | None = None,
    y_tickformat: str | None = None,
    x_tick_interval: int | None = None,
) -> Any:
    """Build an interactive Plotly line chart with a marker on every iteration."""

    import plotly.graph_objects as go

    if frame.empty:
        figure = go.Figure()
        figure.update_layout(
            title=title or "No data",
            height=280,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "No data",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14},
                }
            ],
        )
        return figure

    plot_frame = frame.copy()
    if plot_frame.index.name:
        x_values = [int(value) for value in plot_frame.index.to_numpy()]
        x_label = str(plot_frame.index.name)
    else:
        x_values = list(range(len(plot_frame)))
        x_label = "Iteration"

    figure = go.Figure()
    for column_name in plot_frame.columns:
        series_label = str(column_name)
        y_values = plot_frame[column_name].to_numpy()
        value_format = y_tickformat or ".4f"
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=series_label,
                marker={"size": 7},
                line={"width": 2},
                hovertemplate=(
                    f"{x_label}: %{{x}}<br>"
                    f"{series_label}: %{{y:{value_format}}}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=280,
        margin={"l": 48, "r": 16, "t": 44, "b": 48},
        hovermode="closest",
        dragmode="pan",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(
        tickmode="linear",
        dtick=x_tick_interval if x_tick_interval is not None else 1,
        tick0=(
            x_tick_interval
            if x_tick_interval is not None
            else (min(x_values) if x_values else 0)
        ),
    )
    yaxis_style: dict[str, Any] = {"gridcolor": "rgba(0,0,0,0.12)"}
    if y_tickformat:
        yaxis_style["tickformat"] = y_tickformat
        yaxis_style["exponentformat"] = "none"
        yaxis_style["showexponent"] = "none"
    figure.update_yaxes(**yaxis_style)
    return figure


def render_training_progress_chart(
    frame: pd.DataFrame,
    *,
    y_label: str,
    title: str | None = None,
    y_tickformat: str | None = None,
    x_tick_interval: int | None = None,
) -> None:
    """Render a metrics progress chart in Streamlit with hover on each point."""

    import streamlit as st

    figure = build_training_progress_figure(
        frame,
        y_label=y_label,
        title=title,
        y_tickformat=y_tickformat,
        x_tick_interval=x_tick_interval,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )


def group_planning_rounds(
    snapshots: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Group raw snapshots into policy-improvement rounds."""

    rounds: dict[int, dict[str, Any]] = {}
    for snapshot in snapshots:
        outer_round = int(snapshot.get("outer_round", 0))
        round_data = rounds.setdefault(
            outer_round,
            {
                "outer_round": outer_round,
                "initial": None,
                "evaluation_steps": [],
                "improvement": None,
            },
        )
        phase = snapshot.get("phase")
        if phase == "initial":
            round_data["initial"] = snapshot
        elif phase == "policy_evaluation":
            round_data["evaluation_steps"].append(snapshot)
        elif phase == "policy_improvement":
            round_data["improvement"] = snapshot
    return rounds


def build_round_metrics_dataframe(
    evaluation_steps: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build per-sweep reward and delta metrics for one policy-improvement round."""

    ordered_steps = sorted(
        evaluation_steps,
        key=lambda snapshot: int(snapshot.get("eval_step", 0)),
    )
    if not ordered_steps:
        return pd.DataFrame()

    sweep_labels = [f"S{int(snapshot.get('eval_step', 0))}" for snapshot in ordered_steps]
    return pd.DataFrame(
        {
            "Total Reward": [
                float(snapshot.get("policy_return", 0.0)) for snapshot in ordered_steps
            ],
            "Delta": [float(snapshot.get("delta", 0.0)) for snapshot in ordered_steps],
        },
        index=sweep_labels,
    )


def _collect_value_evolution_states(
    evaluation_steps: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Return sorted non-terminal states that appear in evaluation snapshots."""

    states: set[tuple[int, int]] = set()
    for snapshot in evaluation_steps:
        for state in snapshot.get("value_function", {}):
            states.add(tuple(state))
    return sorted(states)


def _read_state_value(
    value_function: dict[Any, float],
    state: tuple[int, int],
) -> float:
    """Read V(s) from a snapshot value table."""

    if state in value_function:
        return float(value_function[state])
    return float(value_function.get(tuple(state), 0.0))


def _normalize_evaluation_steps(
    evaluation_steps: list[dict[str, Any]],
    *,
    prior_value_function: dict[Any, float] | None = None,
) -> list[dict[str, Any]]:
    """Ensure step 0 is present and evaluation snapshots are ordered."""

    ordered_steps = sorted(
        evaluation_steps,
        key=lambda snapshot: int(snapshot.get("eval_step", 0)),
    )
    if ordered_steps and int(ordered_steps[0].get("eval_step", -1)) == 0:
        return ordered_steps

    if prior_value_function is None:
        return ordered_steps

    return [
        {
            "eval_step": 0,
            "value_function": prior_value_function,
            "delta": 0.0,
            "evaluation_converged": False,
        },
        *ordered_steps,
    ]


def _per_state_evaluation_delta(
    evaluation_steps: list[dict[str, Any]],
    state: tuple[int, int],
) -> float:
    """Return |V_last(s) - V_previous(s)| for the evaluation table."""

    if len(evaluation_steps) >= 2:
        last_values = evaluation_steps[-1].get("value_function", {})
        previous_values = evaluation_steps[-2].get("value_function", {})
        return abs(
            _read_state_value(last_values, state)
            - _read_state_value(previous_values, state)
        )

    return 0.0


def build_value_evolution_table_html(
    evaluation_steps: list[dict[str, Any]],
    *,
    states: list[tuple[int, int]] | None = None,
    expanded: bool = False,
    panel_height_rem: float | None = None,
    prior_value_function: dict[Any, float] | None = None,
    value_precision: int = 4,
) -> str:
    """Render V(s) across evaluation sweeps for one policy-improvement round."""

    if not evaluation_steps:
        return "<p>No Policy Evaluation sweeps were recorded for this round.</p>"

    normalized_steps = _normalize_evaluation_steps(
        evaluation_steps,
        prior_value_function=prior_value_function,
    )

    if states is None:
        states = _collect_value_evolution_states(normalized_steps)
    header_cells = ["<th>State</th>"]
    for snapshot in normalized_steps:
        step_number = snapshot.get("eval_step", "?")
        header_cells.append(f"<th>S{step_number}</th>")
    header_cells.append("<th>Δ</th>")

    body_rows: list[str] = []
    for state in states:
        row_idx, col_idx = state
        cells = [f"<td>({row_idx},{col_idx})</td>"]
        for snapshot in normalized_steps:
            value = _read_state_value(snapshot.get("value_function", {}), state)
            cells.append(f"<td>{value:.{value_precision}f}</td>")
        state_delta = _per_state_evaluation_delta(normalized_steps, state)
        cells.append(f"<td>{state_delta:.{value_precision}f}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    panel_class = (
        "planning-table-scroll-panel planning-table-scroll-panel-expanded planning-table-narrow-panel"
        if expanded
        else "planning-table-scroll-panel planning-table-scroll-panel-compact planning-table-narrow-panel"
    )
    table_font_size = "0.82rem" if expanded else "0.68rem"
    compact_height_rule = ""
    if not expanded and panel_height_rem is not None:
        compact_height_rule = (
            f"height: {panel_height_rem:.2f}rem; "
            f"max-height: {panel_height_rem:.2f}rem;"
        )

    return f"""
    <style>
      .planning-table-scroll-panel {{
        overflow: auto;
        margin: 0.25rem 0 0.5rem 0;
        border: 1px solid #e2e8f0;
        border-radius: 0.55rem;
        background: #ffffff;
        transition: max-height 0.35s ease, height 0.35s ease;
      }}
      .planning-table-narrow-panel {{
        width: fit-content;
        max-width: 100%;
      }}
      .planning-table-scroll-panel-compact {{
        {compact_height_rule}
      }}
      .planning-table-scroll-panel-expanded {{
        max-height: 2400px;
      }}
      .planning-table {{
        border-collapse: collapse;
        width: auto;
        max-width: 100%;
        font-size: {table_font_size};
      }}
      .planning-table th,
      .planning-table td {{
        border: 1px solid #cbd5e1;
        padding: 0.16rem 0.24rem;
        text-align: center;
        white-space: nowrap;
      }}
      .planning-table th {{
        background: #f8fafc;
        font-weight: 700;
      }}
      .planning-table td:first-child,
      .planning-table th:first-child {{
        text-align: left;
        font-weight: 600;
      }}
    </style>
    <div class="{panel_class}">
      <table class="planning-table">
        <thead><tr>{''.join(header_cells)}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


def build_max_values_table_html(
    action_value_details: list[dict[str, Any]],
    *,
    panel_height_rem: float | None = None,
) -> str:
    """Render all legal action-values per state and mark the improved policy choice."""

    if not action_value_details:
        return "<p>No action-value details are available for this round.</p>"

    direction_labels = ("Up", "Right", "Down", "Left")
    short_by_direction = {
        "Up": "U",
        "Right": "R",
        "Down": "D",
        "Left": "L",
    }

    rows: list[str] = []
    for item in action_value_details:
        state_label = html.escape(str(item.get("state_label", "")))
        if item.get("is_terminal"):
            rows.append(
                "<tr>"
                f"<td>{state_label}</td>"
                "<td colspan='4'>Terminal</td>"
                f"<td>{html.escape(str(item.get('chosen_arrow', '🚪')))}</td>"
                "</tr>"
            )
            continue

        action_values = item.get("action_values", {})
        best_labels = set(item.get("best_action_labels", []))
        direction_cells: list[str] = []
        for direction in direction_labels:
            short_label = short_by_direction[direction]
            if short_label not in action_values:
                direction_cells.append("<td>—</td>")
                continue
            value = action_values[short_label]
            is_best = short_label in best_labels
            chosen_short = None
            chosen_action = item.get("chosen_action")
            if chosen_action is not None:
                chosen_short = ("U", "R", "D", "L")[chosen_action]
            is_chosen = is_best and chosen_short == short_label
            marker = "*" if is_chosen else ""
            css_class = "planning-best-action" if is_best else ""
            direction_cells.append(
                f"<td class='{css_class}'><strong>{value:.4f}{marker}</strong></td>"
            )

        tie_note = (
            " (kept)"
            if item.get("kept_current_policy")
            else (" (tie)" if item.get("had_tie") else "")
        )
        policy_cell = (
            f"{html.escape(str(item.get('chosen_arrow', '')))} "
            f"({html.escape(str(item.get('chosen_label', '')))}){tie_note}"
        )
        rows.append(
            "<tr>"
            f"<td>{state_label}</td>"
            f"{''.join(direction_cells)}"
            f"<td>{policy_cell}</td>"
            "</tr>"
        )

    panel_height_rule = ""
    if panel_height_rem is not None:
        panel_height_rule = (
            f"height: {panel_height_rem:.2f}rem; "
            f"max-height: {panel_height_rem:.2f}rem;"
        )

    return f"""
    <style>
      .max-values-table-panel {{
        overflow: auto;
        {panel_height_rule}
        margin: 0.25rem 0 0.5rem 0;
        border: 1px solid #e2e8f0;
        border-radius: 0.55rem;
        background: #ffffff;
        width: fit-content;
        max-width: 100%;
      }}
      .planning-table {{
        border-collapse: collapse;
        width: auto;
        max-width: 100%;
        font-size: 0.68rem;
      }}
      .planning-table th,
      .planning-table td {{
        border: 1px solid #cbd5e1;
        padding: 0.16rem 0.24rem;
        text-align: center;
        white-space: nowrap;
      }}
      .planning-table th {{
        background: #f8fafc;
        font-weight: 700;
      }}
      .planning-table td:first-child,
      .planning-table th:first-child {{
        text-align: left;
        font-weight: 600;
      }}
      .planning-best-action {{
        background: #dbeafe;
        color: #1e3a8a;
        font-weight: 700;
      }}
    </style>
    <div class="max-values-table-panel">
      <table class="planning-table">
        <thead>
          <tr>
            <th>State</th>
            <th>U</th><th>R</th><th>D</th><th>L</th>
            <th>Policy</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def build_action_values_policy_table_html(
    action_value_details: list[dict[str, Any]],
) -> str:
    """Render legal action-values per state and mark the improved policy choice."""

    if not action_value_details:
        return "<p>No action-value details are available for this round.</p>"

    direction_labels = ("Up", "Right", "Down", "Left")
    short_by_direction = {
        "Up": "U",
        "Right": "R",
        "Down": "D",
        "Left": "L",
    }
    header = (
        "<tr>"
        "<th>State (row, col)</th>"
        "<th>Up</th><th>Right</th><th>Down</th><th>Left</th>"
        "<th>Chosen Policy</th>"
        "</tr>"
    )

    rows: list[str] = []
    for item in action_value_details:
        state_label = html.escape(str(item.get("state_label", "")))
        if item.get("is_terminal"):
            rows.append(
                "<tr>"
                f"<td>{state_label}</td>"
                "<td colspan='4'>Terminal</td>"
                f"<td>{html.escape(str(item.get('chosen_arrow', '🚪')))}</td>"
                "</tr>"
            )
            continue

        action_values = item.get("action_values", {})
        best_labels = set(item.get("best_action_labels", []))
        direction_cells: list[str] = []
        for direction in direction_labels:
            short_label = short_by_direction[direction]
            if short_label not in action_values:
                direction_cells.append("<td>—</td>")
                continue
            value = action_values[short_label]
            is_best = short_label in best_labels
            chosen_short = None
            chosen_action = item.get("chosen_action")
            if chosen_action is not None:
                chosen_short = ("U", "R", "D", "L")[chosen_action]
            is_chosen = is_best and chosen_short == short_label
            marker = "*" if is_chosen else ""
            css_class = "planning-best-action" if is_best else ""
            direction_cells.append(
                f"<td class='{css_class}'><strong>{value:.4f}{marker}</strong></td>"
            )

        tie_note = (
            " (kept)"
            if item.get("kept_current_policy")
            else (" (tie)" if item.get("had_tie") else "")
        )
        policy_cell = (
            f"{html.escape(str(item.get('chosen_arrow', '')))} "
            f"({html.escape(str(item.get('chosen_label', '')))}){tie_note}"
        )
        rows.append(
            "<tr>"
            f"<td>{state_label}</td>"
            f"{''.join(direction_cells)}"
            f"<td>{policy_cell}</td>"
            "</tr>"
        )

    return f"""
    <style>
      .planning-table-wrapper {{
        overflow-x: auto;
        margin: 0.25rem 0 0.75rem 0;
      }}
      .planning-table {{
        border-collapse: collapse;
        width: 100%;
        font-size: 0.82rem;
      }}
      .planning-table th,
      .planning-table td {{
        border: 1px solid #cbd5e1;
        padding: 0.28rem 0.45rem;
        text-align: center;
        white-space: nowrap;
      }}
      .planning-table th {{
        background: #f8fafc;
        font-weight: 700;
      }}
      .planning-table td:first-child,
      .planning-table th:first-child {{
        text-align: left;
        font-weight: 600;
      }}
      .planning-best-action {{
        background: #dbeafe;
        color: #1e3a8a;
        font-weight: 700;
      }}
    </style>
    <div class="planning-table-wrapper">
      <table class="planning-table">
        <thead>{header}</thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def _split_cell_tokens(cell_value: str) -> list[str]:
    """Split a cell marker such as `S/A` into its semantic tokens."""

    return [token for token in str(cell_value).split("/") if token]


def _base_layout_token(cell_value: str) -> str:
    """Return the most informative base token for a layout-style cell."""

    tokens = _split_cell_tokens(cell_value)
    for preferred_token in ("S", "E", "G", "W", "T", "I"):
        if preferred_token in tokens:
            return preferred_token
    for token in tokens:
        if token != "A":
            return token
    return "A" if "A" in tokens else str(cell_value)


def _display_text(cell_value: str, mode: str) -> str:
    """Convert raw cell values into user-facing labels."""

    arrow_map = {"U": "↑", "R": "→", "D": "↓", "L": "←"}
    text = str(cell_value)
    if text.startswith("I|"):
        return text[2:]
    if text == "I":
        return ""
    if mode == "policy":
        return arrow_map.get(text, text)
    if text == "E":
        return "🚪"
    return text


def _cell_badge_text(cell_value: str, mode: str, base_cell_value: str) -> str:
    """Return a small badge label that overlays a cell when helpful."""

    base_token = _base_layout_token(base_cell_value)
    if str(cell_value).startswith("I|"):
        return "SLIP"
    if mode == "policy" and base_token == "T":
        return "TRAP"
    if mode == "policy" and base_token == "I":
        return "SLIP"
    if mode == "value" and base_token in {"S", "E", "G", "T", "I"}:
        if base_token == "T":
            return "TRAP"
        if base_token == "I":
            return "SLIP"
        return "EXIT" if base_token in {"E", "G"} else base_token
    if mode == "replay" and "A" in _split_cell_tokens(cell_value):
        return "NOW"
    return ""


def _value_to_color(
    cell_value: str,
    base_cell_value: str,
    numeric_min: float,
    numeric_max: float,
) -> tuple[str, str, str, int]:
    """Choose a heatmap color for value-function cells.

    Returns ``(background, text_color, border_color, frame_width_px)``.
    Special layout cells keep a thicker colored inset frame and a semantic tint.
    """

    base_token = _base_layout_token(base_cell_value)
    frame_width = GRID_CELL_FRAME_WIDTH_PX
    if cell_value == "W":
        return "#334155", "#f8fafc", "#475569", GRID_CELL_HIGHLIGHT_FRAME_WIDTH_PX

    try:
        numeric_value = float(cell_value)
    except ValueError:
        return "#f8fafc", "#0f172a", "#cbd5e1", frame_width

    span = max(numeric_max - numeric_min, 1e-9)
    normalized = (numeric_value - numeric_min) / span
    red = int(245 - normalized * 90)
    green = int(245 - normalized * 25)
    blue = int(245 - normalized * 125)
    background = f"rgb({red}, {green}, {blue})"

    border_color = "#cbd5e1"
    if base_token == "S":
        border_color = "#16a34a"
        background = f"linear-gradient(135deg, #86efac 0%, #dcfce7 35%, {background} 100%)"
        frame_width = GRID_CELL_HIGHLIGHT_FRAME_WIDTH_PX
    elif base_token in {"E", "G"}:
        border_color = "#ca8a04"
        background = f"linear-gradient(135deg, #fde68a 0%, #fef3c7 35%, {background} 100%)"
        frame_width = GRID_CELL_HIGHLIGHT_FRAME_WIDTH_PX
    elif base_token == "T":
        border_color = "#dc2626"
        background = f"linear-gradient(135deg, #fecaca 0%, #fee2e2 35%, {background} 100%)"
        frame_width = GRID_CELL_HIGHLIGHT_FRAME_WIDTH_PX
    elif base_token == "I":
        border_color = "#2563eb"
        background = f"linear-gradient(135deg, #93c5fd 0%, #dbeafe 35%, {background} 100%)"
        frame_width = GRID_CELL_HIGHLIGHT_FRAME_WIDTH_PX

    text_color = "#0f172a" if normalized < 0.7 else "#ffffff"
    return background, text_color, border_color, frame_width


def _categorical_cell_style(
    cell_value: str,
    mode: str,
    base_cell_value: str,
) -> tuple[str, str, str]:
    """Choose colors for layout, policy, and replay grids."""

    base_token = _base_layout_token(base_cell_value)
    contains_agent = "A" in _split_cell_tokens(base_cell_value) or "A" in _split_cell_tokens(cell_value)
    is_slip_annotation = str(cell_value).startswith("I|")

    background = "#f8fafc"
    text_color = "#0f172a"
    border_color = "#cbd5e1"

    if base_token == "S":
        background = "#dcfce7"
        border_color = "#16a34a"
    elif base_token in {"E", "G"}:
        background = "#fef3c7"
        border_color = "#ca8a04"
    elif base_token == "W":
        background = "#334155"
        text_color = "#f8fafc"
        border_color = "#1e293b"
    elif base_token == "T":
        background = "#fee2e2"
        border_color = "#dc2626"
    elif base_token == "I" or is_slip_annotation:
        background = "#dbeafe"
        border_color = "#2563eb"

    if mode == "policy":
        if base_token == "T":
            background = "#fee2e2"
            border_color = "#dc2626"
            text_color = "#991b1b"
        elif base_token == "I" or is_slip_annotation:
            background = "#dbeafe"
            border_color = "#2563eb"
            text_color = "#1e3a8a"
        elif base_token == "S":
            background = "linear-gradient(135deg, #86efac 0%, #dcfce7 100%)"
        elif base_token in {"E", "G"}:
            background = "linear-gradient(135deg, #fde68a 0%, #fef3c7 100%)"
        elif cell_value in {"U", "R", "D", "L"}:
            background = "#eef2ff"
            border_color = "#6366f1"
            text_color = "#4338ca"

    if mode == "replay" and str(cell_value).isdigit():
        background = "#ede9fe"
        border_color = "#7c3aed"

    if contains_agent:
        border_color = "#f97316"

    return background, text_color, border_color


def _policy_arrow_symbol(policy_token: str) -> str:
    """Return a display arrow for a policy-grid token, if applicable."""

    return _display_text(str(policy_token), "policy")


def build_grid_html(
    grid: list[list[str]],
    mode: str = "layout",
    *,
    base_grid: list[list[str]] | None = None,
    cell_size_rem: float | None = None,
    overlay_policy_grid: list[list[str]] | None = None,
) -> str:
    """Render a grid as styled HTML for Streamlit.

    Parameters
    ----------
    grid:
        The visible grid contents.
    mode:
        One of `layout`, `policy`, `value`, or `replay`.
    base_grid:
        Optional reference layout used to preserve semantic cell meaning while a
        different grid, such as value or replay, is shown on top.
    cell_size_rem:
        Optional override for cell width/height. Defaults to ``GRID_CELL_SIZE_REM``.
    overlay_policy_grid:
        Optional policy grid shown as arrows on top of a value-function grid.
    """

    if not grid:
        return "<p>No grid data available.</p>"

    base_grid = base_grid or grid
    resolved_cell_size_rem = cell_size_rem if cell_size_rem is not None else GRID_CELL_SIZE_REM
    numeric_values: list[float] = []
    if mode == "value":
        for row in grid:
            for cell in row:
                try:
                    numeric_values.append(float(cell))
                except ValueError:
                    continue
    numeric_min = min(numeric_values) if numeric_values else 0.0
    numeric_max = max(numeric_values) if numeric_values else 1.0
    show_value_policy_overlay = mode == "value" and overlay_policy_grid is not None

    rows_html: list[str] = []
    header_cells = ["<th class='grid-corner'></th>"]
    for col_idx in range(len(grid[0])):
        header_cells.append(f"<th class='grid-axis'>{col_idx}</th>")
    rows_html.append(f"<tr>{''.join(header_cells)}</tr>")

    for row_idx, row in enumerate(grid):
        row_cells = [f"<th class='grid-axis'>{row_idx}</th>"]
        for col_idx, cell_value in enumerate(row):
            base_cell_value = base_grid[row_idx][col_idx]
            frame_width_px: int | None = None
            if mode == "value":
                background, text_color, border_color, frame_width_px = _value_to_color(
                    str(cell_value),
                    str(base_cell_value),
                    numeric_min,
                    numeric_max,
                )
            else:
                background, text_color, border_color = _categorical_cell_style(
                    str(cell_value),
                    mode,
                    str(base_cell_value),
                )

            display_text = html.escape(_display_text(str(cell_value), mode))
            badge_text = html.escape(
                _cell_badge_text(str(cell_value), mode, str(base_cell_value))
            )
            policy_arrow = ""
            if show_value_policy_overlay:
                policy_token = overlay_policy_grid[row_idx][col_idx]
                arrow_symbol = _policy_arrow_symbol(policy_token)
                if arrow_symbol and all(character in "↑→↓←" for character in arrow_symbol):
                    policy_arrow = (
                        "<span class='grid-cell-policy-arrow'>"
                        f"{html.escape(arrow_symbol)}"
                        "</span>"
                    )
            font_size = (
                f"{resolved_cell_size_rem * 0.2:.2f}rem"
                if str(cell_value).startswith("I|")
                else f"{resolved_cell_size_rem * 0.22:.2f}rem"
                if show_value_policy_overlay and policy_arrow
                else f"{resolved_cell_size_rem * 0.28:.2f}rem"
            )
            style = (
                "position:relative;"
                f"background:{background};color:{text_color};"
                f"{_cell_frame_css_variable(border_color, frame_width_px)}"
                f"font-size:{font_size};"
            )
            if policy_arrow:
                cell_content = (
                    "<span class='grid-cell-value-policy'>"
                    f"<span class='grid-cell-main'>{display_text}</span>"
                    f"{policy_arrow}"
                    "</span>"
                    + (
                        f"<span class='grid-cell-badge'>{badge_text}</span>"
                        if badge_text
                        else ""
                    )
                )
            else:
                cell_content = (
                    f"<span class='grid-cell-main'>{display_text}</span>"
                    + (
                        f"<span class='grid-cell-badge'>{badge_text}</span>"
                        if badge_text
                        else ""
                    )
                )
            row_cells.append(
                (
                    "<td class='grid-cell' "
                    f"style=\"{style}\">"
                    f"{cell_content}"
                    "</td>"
                )
            )
        rows_html.append(f"<tr>{''.join(row_cells)}</tr>")

    cell_size = f"{resolved_cell_size_rem}rem"
    is_compact_grid = resolved_cell_size_rem < GRID_CELL_SIZE_REM
    grid_spacing = "0.16rem" if is_compact_grid else "0.28rem"
    grid_margin = "0.25rem 0 0.35rem 0" if is_compact_grid else "0.5rem 0 1rem 0"
    wrapper_class = "escape-grid-wrapper"
    if show_value_policy_overlay:
        wrapper_class += " escape-grid-value-policy"
    return f"""
    <style>
      .escape-grid-wrapper {{
        overflow: visible;
        margin: {grid_margin};
      }}
      .escape-grid-table {{
        border-collapse: separate;
        border-spacing: {grid_spacing};
        margin: 0;
        table-layout: fixed;
      }}
      .escape-grid-table .grid-axis,
      .escape-grid-table .grid-corner {{
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
        text-align: center;
        min-width: 1.75rem;
      }}
      .escape-grid-table .grid-cell {{
        width: {cell_size};
        height: {cell_size};
        min-width: {cell_size};
        max-width: {cell_size};
        min-height: {cell_size};
        max-height: {cell_size};
        text-align: center;
        vertical-align: middle;
        border-radius: 0.8rem;
        font-weight: 700;
        {GRID_CELL_FRAME_CSS}
        white-space: normal;
        line-height: 1.1;
        padding: 0.2rem;
        overflow: visible;
        isolation: isolate;
      }}
      .escape-grid-table .grid-cell-main {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        line-height: 1;
      }}
      .escape-grid-table .grid-cell-badge {{
        position: absolute;
        top: 0.18rem;
        left: 0.28rem;
        font-size: 0.58rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: rgba(15, 23, 42, 0.82);
      }}
      .escape-grid-value-policy .grid-cell-value-policy {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.04rem;
        width: 100%;
        height: 100%;
        line-height: 1;
      }}
      .escape-grid-value-policy .grid-cell-policy-arrow {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: {resolved_cell_size_rem * 0.34:.2f}rem;
        font-weight: 800;
        color: #4338ca;
        line-height: 1;
      }}
    </style>
    <div class="{wrapper_class}">
      <table class="escape-grid-table">
        {''.join(rows_html)}
      </table>
    </div>
    """


def _slot_probability(directions: dict[str, Any], slot: str) -> float:
    """Read a directional transition probability from unified cell data."""

    value = directions.get(slot, 0.0)
    if isinstance(value, dict):
        return float(value.get("probability", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_mdp_zone_content(
    directions: dict[str, Any],
    slot: str,
    *,
    show_probability: bool,
) -> str:
    """Return probability markup for one fixed grid zone, or empty when hidden."""

    if not show_probability:
        return ""
    probability = _slot_probability(directions, slot)
    if probability <= 0.0:
        return ""
    return f"<span class='mdp-prob-value'>{probability:.2f}</span>"


def _build_fixed_mdp_grid(cell: dict[str, Any]) -> str:
    """Render the fixed 3x3 grid: four edge zones + one center policy zone."""

    directions = cell.get("directions", {})
    show_probability = bool(cell.get("show_probabilities", False))
    policy_arrow = str(cell.get("policy_arrow", ""))
    center_content = (
        f"<span class='mdp-policy-arrow'>{html.escape(policy_arrow)}</span>"
        if policy_arrow
        else ""
    )

    grid_class = (
        "mdp-fixed-grid mdp-fixed-grid-slippery"
        if show_probability
        else "mdp-fixed-grid"
    )

    return (
        f"<div class='{grid_class}'>"
        f"<div class='mdp-zone mdp-zone-up'>"
        f"{_build_mdp_zone_content(directions, 'up', show_probability=show_probability)}"
        "</div>"
        f"<div class='mdp-zone mdp-zone-left'>"
        f"{_build_mdp_zone_content(directions, 'left', show_probability=show_probability)}"
        "</div>"
        f"<div class='mdp-zone mdp-zone-center'>{center_content}</div>"
        f"<div class='mdp-zone mdp-zone-right'>"
        f"{_build_mdp_zone_content(directions, 'right', show_probability=show_probability)}"
        "</div>"
        f"<div class='mdp-zone mdp-zone-down'>"
        f"{_build_mdp_zone_content(directions, 'down', show_probability=show_probability)}"
        "</div>"
        "</div>"
    )


def _build_cell_annotations(cell: dict[str, Any]) -> str:
    """Place start markers and rewards in corners so the center stays clear."""

    kind = cell.get("kind", "empty")
    if kind in {"wall", "exit", "slippery"}:
        return ""

    marker = str(cell.get("marker", ""))
    reward_label = str(cell.get("reward_label", "") or "")
    center_label = ""
    if kind == "start":
        center_label = "S"
    elif kind != "trap" and marker not in {".", "", "I"}:
        center_label = marker

    parts: list[str] = []
    if center_label:
        parts.append(
            f"<span class='mdp-annotation mdp-annotation-marker'>"
            f"{html.escape(center_label)}</span>"
        )
    if reward_label:
        parts.append(
            f"<span class='mdp-annotation mdp-annotation-reward'>"
            f"{html.escape(reward_label)}</span>"
        )
    if not parts:
        return ""

    return f"<div class='mdp-annotations'>{''.join(parts)}</div>"


def _build_actionable_cell_body(cell: dict[str, Any]) -> str:
    """Actionable cell: fixed 3x3 zones for policy arrow and slip probabilities."""

    return (
        "<div class='compass compass-mdp'>"
        f"{_build_fixed_mdp_grid(cell)}"
        f"{_build_cell_annotations(cell)}"
        "</div>"
    )


def _build_compass_cell_body(cell: dict[str, Any]) -> str:
    """Build the compass-style inner content of one unified grid cell."""

    kind = cell.get("kind", "empty")
    reward_label = str(cell.get("reward_label", "") or "")

    if kind == "wall":
        return (
            "<div class='compass'>"
            "<div class='compass-center'><span class='compass-marker'>W</span></div>"
            "</div>"
        )

    if kind == "exit":
        return (
            "<div class='compass'>"
            "<div class='compass-center'>"
            "<span class='compass-marker'>🚪</span>"
            + (
                f"<span class='compass-reward'>{html.escape(reward_label)}</span>"
                if reward_label
                else ""
            )
            + "</div></div>"
        )

    if kind == "slippery":
        policy_arrow = str(cell.get("policy_arrow", ""))
        center_content = (
            f"<span class='mdp-policy-arrow'>{html.escape(policy_arrow)}</span>"
            if policy_arrow
            else ""
        )
        return (
            "<div class='compass'>"
            f"<div class='compass-center compass-center-policy'>{center_content}</div>"
            "</div>"
        )

    if cell.get("policy_arrow") or kind in {"start", "empty", "trap"}:
        return _build_actionable_cell_body(cell)

    return "<div class='compass'></div>"


def build_unified_room_grid_html(grid: list[list[dict[str, Any]]]) -> str:
    """Render one unified room grid with compass-style policy probabilities."""

    if not grid:
        return "<p>No grid data available.</p>"

    style_by_kind = {
        "start": ("#dcfce7", "#166534", "#16a34a"),
        "exit": ("#fef3c7", "#92400e", "#ca8a04"),
        "wall": ("#334155", "#f8fafc", "#1e293b"),
        "trap": ("#fee2e2", "#991b1b", "#dc2626"),
        "slippery": ("#dbeafe", "#1e3a8a", "#2563eb"),
        "empty": ("#f8fafc", "#0f172a", "#cbd5e1"),
    }

    rows_html: list[str] = []
    header_cells = ["<th class='unified-corner'></th>"]
    for col_idx in range(len(grid[0])):
        header_cells.append(f"<th class='unified-axis'>{col_idx}</th>")
    rows_html.append(f"<tr>{''.join(header_cells)}</tr>")

    for row_idx, row in enumerate(grid):
        row_cells = [f"<th class='unified-axis'>{row_idx}</th>"]
        for cell in row:
            kind = cell.get("kind", "empty")
            background, text_color, border_color = style_by_kind.get(
                kind,
                style_by_kind["empty"],
            )
            title = html.escape(str(cell.get("title", "")))
            body = _build_compass_cell_body(cell)
            row_cells.append(
                (
                    "<td class='unified-cell' "
                    f"title=\"{title}\" "
                    f"style=\"background:{background};color:{text_color};{_cell_frame_css_variable(border_color)}\">"
                    f"{body}"
                    "</td>"
                )
            )
        rows_html.append(f"<tr>{''.join(row_cells)}</tr>")

    cell_size = f"{GRID_CELL_SIZE_REM}rem"
    return f"""
    <style>
      .unified-grid-wrapper {{
        overflow: visible;
        margin: 0.5rem 0 1rem 0;
      }}
      .unified-grid-table {{
        border-collapse: separate;
        border-spacing: 0.35rem;
        margin: 0;
        table-layout: fixed;
      }}
      .unified-axis,
      .unified-corner {{
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 600;
        text-align: center;
        width: 1.55rem;
        min-width: 1.55rem;
      }}
      .unified-cell {{
        --cell-size: {cell_size};
        width: var(--cell-size);
        height: var(--cell-size);
        min-width: var(--cell-size);
        max-width: var(--cell-size);
        min-height: var(--cell-size);
        max-height: var(--cell-size);
        border-radius: 0.8rem;
        vertical-align: middle;
        text-align: center;
        padding: 0;
        {GRID_CELL_FRAME_CSS}
        overflow: visible;
        isolation: isolate;
      }}
      .compass {{
        position: relative;
        width: 100%;
        height: 100%;
        overflow: visible;
      }}
      .compass-mdp {{
        /* Slippery / actionable cells use the fixed 3x3 MDP grid below. */
      }}
      .mdp-fixed-grid {{
        --mdp-edge-band: 0.62rem;
        display: grid;
        width: 100%;
        height: 100%;
        grid-template-columns:
          var(--mdp-edge-band)
          minmax(0, 1fr)
          var(--mdp-edge-band);
        grid-template-rows:
          var(--mdp-edge-band)
          minmax(0, 1fr)
          var(--mdp-edge-band);
        grid-template-areas:
          ". up ."
          "left center right"
          ". down .";
        padding: 0;
        box-sizing: border-box;
      }}
      .mdp-fixed-grid-slippery {{
        grid-template-columns:
          max-content
          minmax(0, 1fr)
          max-content;
        grid-template-rows:
          max-content
          minmax(0, 1fr)
          max-content;
      }}
      .mdp-zone {{
        display: grid;
        min-width: 0;
        min-height: 0;
        overflow: visible;
        pointer-events: none;
      }}
      .mdp-zone-up {{
        grid-area: up;
        place-items: center;
      }}
      .mdp-zone-down {{
        grid-area: down;
        place-items: center;
      }}
      .mdp-zone-left {{
        grid-area: left;
        place-items: center start;
      }}
      .mdp-zone-right {{
        grid-area: right;
        place-items: center end;
      }}
      .mdp-zone-center {{
        grid-area: center;
        place-items: center;
      }}
      .mdp-fixed-grid-slippery .mdp-zone-up {{
        place-items: start center;
      }}
      .mdp-fixed-grid-slippery .mdp-zone-down {{
        place-items: end center;
      }}
      .mdp-fixed-grid-slippery .mdp-zone-left {{
        place-items: center start;
        justify-self: start;
      }}
      .mdp-fixed-grid-slippery .mdp-zone-right {{
        place-items: center end;
        justify-self: end;
      }}
      .mdp-prob-value {{
        font-size: 0.48rem;
        font-weight: 700;
        line-height: 1;
        font-variant-numeric: tabular-nums;
        color: #1e3a8a;
        background: rgba(255, 255, 255, 0.92);
        border-radius: 999px;
        padding: 0.03rem 0.16rem;
        white-space: nowrap;
        text-align: center;
        box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.14);
      }}
      .mdp-fixed-grid-slippery .mdp-zone-up .mdp-prob-value {{
        border-radius: 0 0 999px 999px;
        padding: 0 0.12rem 0.1rem 0.12rem;
      }}
      .mdp-fixed-grid-slippery .mdp-zone-down .mdp-prob-value {{
        border-radius: 999px 999px 0 0;
        padding: 0.1rem 0.12rem 0 0.12rem;
      }}
      .mdp-fixed-grid-slippery .mdp-zone-left .mdp-prob-value {{
        border-radius: 0 999px 999px 0;
        padding: 0.1rem 0.12rem 0.1rem 0;
      }}
      .mdp-fixed-grid-slippery .mdp-zone-right .mdp-prob-value {{
        border-radius: 999px 0 0 999px;
        padding: 0.1rem 0 0.1rem 0.12rem;
      }}
      .mdp-policy-arrow {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.35rem;
        height: 1.35rem;
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1;
        color: #1e3a8a;
        background: rgba(255, 255, 255, 0.92);
        border-radius: 50%;
        box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.18);
      }}
      .mdp-annotations {{
        position: absolute;
        inset: 0;
        z-index: 1;
        pointer-events: none;
      }}
      .mdp-annotation {{
        position: absolute;
        font-weight: 800;
        line-height: 1;
        pointer-events: none;
      }}
      .mdp-annotation-marker {{
        top: 0.14rem;
        left: 0.14rem;
        font-size: 0.72rem;
      }}
      .mdp-annotation-reward {{
        top: 0.14rem;
        right: 0.14rem;
        font-size: 0.52rem;
        background: rgba(255, 255, 255, 0.72);
        border-radius: 999px;
        padding: 0.04rem 0.22rem;
        box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.08);
        white-space: nowrap;
      }}
      .compass-center {{
        position: absolute;
        inset: 24% 18%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.05rem;
        z-index: 1;
        pointer-events: none;
      }}
      .compass-center-policy {{
        inset: 0;
      }}
      .compass-marker {{
        font-size: 0.9rem;
        font-weight: 800;
        line-height: 1;
      }}
      .compass-reward {{
        font-size: 0.58rem;
        font-weight: 800;
        line-height: 1.1;
        white-space: nowrap;
        background: rgba(255, 255, 255, 0.72);
        border-radius: 999px;
        padding: 0.05rem 0.24rem;
        box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.08);
      }}
    </style>
    <div class="unified-grid-wrapper">
      <table class="unified-grid-table">
        {''.join(rows_html)}
      </table>
    </div>
    """


def build_unified_legend_html() -> str:
    """Legend for the single unified Room 1 grid."""

    items = [
        ("S", "Start (center)", "#dcfce7", "#16a34a", "#166534"),
        ("🚪", "Exit Door", "#fef3c7", "#ca8a04", "#92400e"),
        ("W", "Wall / Obstacle", "#334155", "#1e293b", "#f8fafc"),
        ("T", "Trap", "#fee2e2", "#dc2626", "#991b1b"),
        ("SL", "Slippery cell (blue border)", "#dbeafe", "#2563eb", "#1e3a8a"),
        ("r = N", "Reward on that cell", "#fff7ed", "#ea580c", "#9a3412"),
        ("→", "Policy action (center arrow)", "#eef2ff", "#6366f1", "#4338ca"),
    ]
    legend_items: list[str] = []
    for symbol, label, background, border, text_color in items:
        legend_items.append(
            (
                "<span class='escape-legend-item'>"
                f"<span class='escape-legend-swatch' style=\"background:{background};{_cell_frame_css_variable(border)};color:{text_color};\">"
                f"{html.escape(symbol)}"
                "</span>"
                f"{html.escape(label)}"
                "</span>"
            )
        )

    return f"""
    <style>
      .escape-legend {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem 1rem;
        margin: 0.25rem 0 0.75rem 0;
      }}
      .escape-legend-item {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.95rem;
        color: #334155;
      }}
      .escape-legend-swatch {{
        min-width: 2.2rem;
        height: 1.55rem;
        border-radius: 0.4rem;
        {GRID_CELL_FRAME_CSS}
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0 0.3rem;
        font-size: 0.78rem;
        font-weight: 800;
        overflow: visible;
      }}
    </style>
    <div class="escape-legend">
      {''.join(legend_items)}
    </div>
    """


def build_grid_legend_html(mode: str) -> str:
    """Return an HTML legend that matches the rendered grid colors."""

    if mode == "value":
        items = [
            ("S", "Start state highlight", "#dcfce7", "#16a34a", "#0f172a"),
            ("E", "Exit Door highlight", "#fef3c7", "#ca8a04", "#0f172a"),
            ("T", "Trap cell border", "#fee2e2", "#dc2626", "#0f172a"),
            ("I", "Slippery cell border", "#dbeafe", "#2563eb", "#0f172a"),
            ("↑", "Fixed policy arrow", "#ffffff", "#4338ca", "#4338ca"),
            ("High", "Higher state value", "#9f1239", "#9f1239", "#ffffff"),
            ("Low", "Lower state value", "#f8fafc", "#cbd5e1", "#0f172a"),
        ]
    elif mode == "policy":
        items = [
            ("S", "Start", "#dcfce7", "#16a34a", "#0f172a"),
            ("E", "Exit Door", "#fef3c7", "#ca8a04", "#0f172a"),
            ("W", "Wall", "#334155", "#1e293b", "#f8fafc"),
            ("T", "Trap (-0.5)", "#fee2e2", "#dc2626", "#991b1b"),
            ("I", "Slippery cell", "#dbeafe", "#2563eb", "#1e3a8a"),
            ("↑→↓←", "Policy action", "#eef2ff", "#6366f1", "#4338ca"),
        ]
    elif mode == "replay":
        items = [
            ("S", "Start", "#dcfce7", "#16a34a", "#0f172a"),
            ("E", "Exit Door", "#fef3c7", "#ca8a04", "#0f172a"),
            ("W", "Wall", "#334155", "#1e293b", "#f8fafc"),
            ("T", "Trap (-0.5)", "#fee2e2", "#dc2626", "#0f172a"),
            ("I", "Slippery (50/50)", "#dbeafe", "#2563eb", "#0f172a"),
            ("1+", "Visited replay step #", "#ede9fe", "#7c3aed", "#5b21b6"),
            ("A", "Current replay position", "#ffffff", "#f97316", "#f97316"),
        ]
    else:
        items = [
            ("S", "Start (top-left)", "#dcfce7", "#16a34a", "#0f172a"),
            ("E", "Exit Door (+10)", "#fef3c7", "#ca8a04", "#0f172a"),
            ("W", "Wall / Obstacle", "#334155", "#1e293b", "#f8fafc"),
            ("T", "Trap (-0.5)", "#fee2e2", "#dc2626", "#0f172a"),
            ("I", "Slippery (50% action / 50% other)", "#dbeafe", "#2563eb", "#0f172a"),
            ("A", "Agent position", "#ffffff", "#f97316", "#f97316"),
        ]

    legend_items: list[str] = []
    for symbol, label, background, border, text_color in items:
        legend_items.append(
            (
                "<span class='escape-legend-item'>"
                f"<span class='escape-legend-swatch' style=\"background:{background};{_cell_frame_css_variable(border)};color:{text_color};\">"
                f"{html.escape(symbol)}"
                "</span>"
                f"{html.escape(label)}"
                "</span>"
            )
        )

    return f"""
    <style>
      .escape-legend {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem 1rem;
        margin: 0.25rem 0 0.75rem 0;
      }}
      .escape-legend-item {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.9rem;
        color: #334155;
      }}
      .escape-legend-swatch {{
        min-width: 1.55rem;
        height: 1.3rem;
        border-radius: 0.3rem;
        {GRID_CELL_FRAME_CSS}
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0 0.22rem;
        font-size: 0.72rem;
        font-weight: 800;
        overflow: visible;
      }}
    </style>
    <div class="escape-legend">
      {''.join(legend_items)}
    </div>
    """
