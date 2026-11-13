"""Baseline scorers: random assignment and FIFO capture order."""

from __future__ import annotations

import numpy as np

from capstone.contracts.types import Tile


class RandomScorer:
    """Assigns pseudo-random scores using an explicit seed."""

    def __init__(self, seed: int = 42, name: str = "random") -> None:
        self.seed = seed
        self.name = name
        self._rng = np.random.default_rng(seed)

    def score(self, tile: Tile) -> float:
        """Draw a uniform random float in [0.0, 1.0) as the tile score."""
        return float(self._rng.uniform(0.0, 1.0))

    def reset(self) -> None:
        """Reset the internal generator to the initial seed state."""
        self._rng = np.random.default_rng(self.seed)


class FifoScorer:
    """FIFO baseline prioritizing tiles by capture timestamp (earlier scores higher)."""

    def __init__(self, name: str = "fifo") -> None:
        self.name = name

    def score(self, tile: Tile) -> float:
        """Return inverted capture timestamp so earlier tiles receive higher scores."""
        return -float(tile.timestamp)
