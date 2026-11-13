"""Baseline scorers: random assignment and FIFO capture order."""

from __future__ import annotations

import hashlib

from capstone.contracts.types import Tile


class RandomScorer:
    """Assigns pseudo-random scores as a pure function of (seed, tile_id)."""

    def __init__(self, seed: int = 42, name: str = "random") -> None:
        self.seed = seed
        self.name = name

    def score(self, tile: Tile) -> float:
        """Map (seed, tile_id) stably via blake2b to a float in [0.0, 1.0)."""
        key = f"{self.seed}:{tile.tile_id}".encode()
        digest = hashlib.blake2b(key, digest_size=8).digest()
        int_val = int.from_bytes(digest, byteorder="big")
        # Map 64-bit unsigned integer to [0.0, 1.0)
        return float(int_val / (2**64))


class FifoScorer:
    """FIFO baseline prioritizing tiles by capture timestamp (earlier scores higher)."""

    def __init__(self, name: str = "fifo") -> None:
        self.name = name

    def score(self, tile: Tile) -> float:
        """Return inverted capture timestamp so earlier tiles receive higher scores."""
        return -float(tile.timestamp)
