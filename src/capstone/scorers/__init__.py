"""Tile scoring algorithms and benchmarking tools."""

from typing import TYPE_CHECKING, Any

from capstone.scorers.base import Scorer, timed_score
from capstone.scorers.baselines import FifoScorer, RandomScorer
from capstone.scorers.edge import S2EdgeScorer
from capstone.scorers.registry import build_scorer, list_scorers, register_scorer
from capstone.scorers.variance import S1VarianceScorer

if TYPE_CHECKING:
    from capstone.scorers.timing import benchmark_scorer

__all__ = [
    "FifoScorer",
    "RandomScorer",
    "S1VarianceScorer",
    "S2EdgeScorer",
    "Scorer",
    "benchmark_scorer",
    "build_scorer",
    "list_scorers",
    "register_scorer",
    "timed_score",
]


def __getattr__(name: str) -> Any:
    """Lazy-load timing benchmark utility to prevent runpy circular warnings."""
    if name == "benchmark_scorer":
        from capstone.scorers.timing import benchmark_scorer

        return benchmark_scorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
