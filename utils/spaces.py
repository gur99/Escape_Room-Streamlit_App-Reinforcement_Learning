"""Lightweight Gymnasium-style space objects used by the project skeleton.

The real project may later replace these classes with Gymnasium spaces directly.
For Stage 1 they provide a clear, dependency-free contract that mirrors the
concept of observation and action spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiscreteSpace:
    """A minimal discrete space description compatible with the project needs."""

    n: int
    description: str

    def sample(self) -> int:
        """Return a deterministic placeholder sample for the skeleton stage."""

        return 0


@dataclass(frozen=True)
class BoxSpace:
    """A minimal continuous box space description."""

    low: tuple[float, ...]
    high: tuple[float, ...]
    shape: tuple[int, ...]
    description: str

    def sample(self) -> tuple[float, ...]:
        """Return the midpoint as a simple placeholder sample."""

        return tuple((low + high) / 2 for low, high in zip(self.low, self.high))


@dataclass(frozen=True)
class CompositeSpace:
    """A simple named container for richer observations such as Room 5."""

    components: dict[str, Any]
    description: str

    def sample(self) -> dict[str, Any]:
        """Return raw component definitions for inspection in Stage 1."""

        return dict(self.components)
