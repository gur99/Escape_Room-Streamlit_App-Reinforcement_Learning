"""Replay data structures for visualizing trained episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TerminationReason = Literal["goal", "max_steps"]

ACTION_LABELS: dict[int, str] = {0: "U", 1: "R", 2: "D", 3: "L"}


@dataclass
class ReplayStep:
    """One step in an episode replay."""

    observation: Any
    action: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class EpisodeReplay:
    """Container for a full replayable episode."""

    room_name: str
    algorithm_name: str
    steps: list[ReplayStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the replay to a JSON-friendly dictionary."""

        return {
            "room_name": self.room_name,
            "algorithm_name": self.algorithm_name,
            "steps": [asdict(step) for step in self.steps],
        }


@dataclass
class SarsaEpisodeRecord:
    """Full per-episode record for SARSA training replay.

    ``episode_index`` is 0-based internally. The UI displays ``episode_index + 1``.
    """

    episode_index: int
    states: list[tuple[int, int]]
    actions: list[int]
    next_states: list[tuple[int, int]]
    rewards: list[float]
    was_exploration: list[bool]
    was_off_policy: list[bool]
    epsilon_per_step: list[float]
    steps: int
    off_policy_steps: int
    total_reward: float
    discounted_return: float
    epsilon_start: float
    time_ms: float
    cumulative_time_s: float
    termination_reason: TerminationReason

    @property
    def display_episode_number(self) -> int:
        """1-based episode number for UI display."""

        return self.episode_index + 1

    @property
    def episode_length(self) -> int:
        """Alias for step count used by metrics charts."""

        return self.steps

    def to_episode_replay(
        self,
        *,
        room_name: str = "Room 2 - SARSA",
        algorithm_name: str = "SARSA",
    ) -> EpisodeReplay:
        """Convert this record into an ``EpisodeReplay`` for grid playback."""

        replay_steps: list[ReplayStep] = []
        last_index = len(self.actions) - 1
        for step_index, action in enumerate(self.actions):
            is_last = step_index == last_index
            terminated = is_last and self.termination_reason == "goal"
            truncated = is_last and self.termination_reason == "max_steps"
            replay_steps.append(
                ReplayStep(
                    observation=self.next_states[step_index],
                    action=action,
                    reward=self.rewards[step_index],
                    terminated=terminated,
                    truncated=truncated,
                    info={
                        "state_before_action": list(self.states[step_index]),
                        "state_after_action": list(self.next_states[step_index]),
                        "action_label": ACTION_LABELS.get(action, str(action)),
                        "epsilon": self.epsilon_per_step[step_index],
                        "was_exploration": self.was_exploration[step_index],
                        "was_off_policy": self.was_off_policy[step_index],
                        "chosen_transition_probability": 1.0,
                    },
                )
            )
        return EpisodeReplay(
            room_name=room_name,
            algorithm_name=algorithm_name,
            steps=replay_steps,
        )


def select_evenly_spaced_indices(n_total: int, n_sample: int) -> list[int]:
    """Return evenly spaced 0-based indices spanning the full training range.

    Includes the first and last episode when ``n_sample >= 2``. Duplicates from
    rounding are removed while preserving order.
    """

    if n_total <= 0 or n_sample <= 0:
        return []
    if n_sample >= n_total:
        return list(range(n_total))
    if n_sample == 1:
        return [0]

    selected: list[int] = []
    seen: set[int] = set()
    for sample_slot in range(n_sample):
        index = int(round(sample_slot * (n_total - 1) / (n_sample - 1)))
        if index not in seen:
            seen.add(index)
            selected.append(index)
    return selected


def select_most_recent_indices(n_total: int, n_recent: int) -> list[int]:
    """Return the last ``n_recent`` 0-based indices in chronological order.

    If ``n_recent >= n_total``, returns every index.
    """

    if n_total <= 0 or n_recent <= 0:
        return []
    if n_recent >= n_total:
        return list(range(n_total))
    start_index = n_total - n_recent
    return list(range(start_index, n_total))


def build_placeholder_replay() -> EpisodeReplay:
    """Return an empty replay object used by the Stage 1 UI."""

    return EpisodeReplay(
        room_name="Stage 1 Skeleton",
        algorithm_name="Not Implemented Yet",
        steps=[],
    )
