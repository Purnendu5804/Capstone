"""Base protocol and timing utility for tile scorers."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from capstone.contracts.types import ScoreResult, Tile


@runtime_checkable
class Scorer(Protocol):
    """Protocol for tile scoring algorithms."""

    name: str

    def score(self, tile: Tile) -> float:
        """Compute an onboard priority score for a tile (higher means more valuable)."""
        ...


def timed_score(scorer: Scorer, tile: Tile) -> ScoreResult:
    """Score a tile and measure execution time using time.perf_counter_ns.

    Includes any preprocessing inside the scorer in the measured cost.
    """
    start_ns = time.perf_counter_ns()
    score_value = scorer.score(tile)
    end_ns = time.perf_counter_ns()

    cost_seconds = (end_ns - start_ns) / 1_000_000_000.0

    return ScoreResult(
        tile_id=tile.tile_id,
        scorer_name=scorer.name,
        score=float(score_value),
        cost_seconds=cost_seconds,
    )
