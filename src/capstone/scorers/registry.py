"""Scorer registry and factory utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from capstone.scorers.base import Scorer
from capstone.scorers.baselines import FifoScorer, RandomScorer
from capstone.scorers.edge import S2EdgeScorer
from capstone.scorers.variance import S1VarianceScorer

ScorerFactory = Callable[..., Scorer]

_CANONICAL_SCORERS: dict[str, ScorerFactory] = {
    "s1": S1VarianceScorer,
    "s2": S2EdgeScorer,
    "s2_mean": lambda **kwargs: S2EdgeScorer(mode="mean_magnitude", **kwargs),
    "random": RandomScorer,
    "fifo": FifoScorer,
}

_SCORER_ALIASES: dict[str, str] = {
    "s1_variance": "s1",
    "variance": "s1",
    "s2_edge": "s2",
    "edge": "s2",
    "sobel": "s2",
    "s2_magnitude": "s2_mean",
}


def register_scorer(name: str, factory: ScorerFactory) -> None:
    """Register a custom scorer factory under a canonical name."""
    clean_name = name.strip().lower()
    if not clean_name:
        raise ValueError("Scorer name cannot be empty")
    _CANONICAL_SCORERS[clean_name] = factory


def build_scorer(name: str, **kwargs: Any) -> Scorer:
    """Instantiate a scorer by name or alias with optional arguments."""
    clean_name = name.strip().lower()
    canonical_name = _SCORER_ALIASES.get(clean_name, clean_name)

    if canonical_name not in _CANONICAL_SCORERS:
        available = sorted(_CANONICAL_SCORERS.keys())
        raise KeyError(
            f"Unknown scorer: {name!r}. Available canonical scorers: {available}"
        )
    return _CANONICAL_SCORERS[canonical_name](**kwargs)


def list_scorers() -> list[str]:
    """Return a sorted list of canonical registered scorer names only."""
    return sorted(_CANONICAL_SCORERS.keys())
