"""Scorer registry and factory utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from capstone.scorers.base import Scorer
from capstone.scorers.baselines import FifoScorer, RandomScorer
from capstone.scorers.edge import S2EdgeScorer
from capstone.scorers.variance import S1VarianceScorer

ScorerFactory = Callable[..., Scorer]

_SCORER_REGISTRY: dict[str, ScorerFactory] = {
    "s1": S1VarianceScorer,
    "s1_variance": S1VarianceScorer,
    "variance": S1VarianceScorer,
    "s2": S2EdgeScorer,
    "s2_edge": S2EdgeScorer,
    "edge": S2EdgeScorer,
    "random": RandomScorer,
    "fifo": FifoScorer,
}


def register_scorer(name: str, factory: ScorerFactory) -> None:
    """Register a custom scorer factory under a given name."""
    clean_name = name.strip().lower()
    if not clean_name:
        raise ValueError("Scorer name cannot be empty")
    _SCORER_REGISTRY[clean_name] = factory


def build_scorer(name: str, **kwargs: Any) -> Scorer:
    """Instantiate a scorer by registered name with optional arguments."""
    clean_name = name.strip().lower()
    if clean_name not in _SCORER_REGISTRY:
        available = sorted(_SCORER_REGISTRY.keys())
        raise KeyError(f"Unknown scorer: {name!r}. Available scorers: {available}")
    return _SCORER_REGISTRY[clean_name](**kwargs)


def list_scorers() -> list[str]:
    """Return a sorted list of registered scorer names."""
    return sorted(_SCORER_REGISTRY.keys())
