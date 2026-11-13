"""Contracts and interfaces for the satellite tile prioritization pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, eq=False)
class Tile:
    """Represents an onboard satellite image tile with metadata and pixel data."""

    tile_id: str
    strip_id: str
    row: int
    col: int
    pixels: np.ndarray
    band_names: tuple[str, ...]
    timestamp: float

    def __post_init__(self) -> None:
        """Validate tile fields and pixel array properties."""
        if not isinstance(self.tile_id, str) or not self.tile_id.strip():
            raise ValueError(f"tile_id must be a non-empty string, got {self.tile_id!r}")

        if not isinstance(self.strip_id, str) or not self.strip_id.strip():
            raise ValueError(f"strip_id must be a non-empty string, got {self.strip_id!r}")

        if not isinstance(self.row, int) or self.row < 0:
            raise ValueError(f"row must be a non-negative integer, got {self.row!r}")

        if not isinstance(self.col, int) or self.col < 0:
            raise ValueError(f"col must be a non-negative integer, got {self.col!r}")

        if not isinstance(self.pixels, np.ndarray):
            raise TypeError(f"pixels must be a numpy.ndarray, got {type(self.pixels).__name__}")

        if self.pixels.ndim != 3:
            raise ValueError(
                f"pixels must have 3 dimensions (H, W, C), got shape {self.pixels.shape}"
            )

        if self.pixels.dtype != np.uint16:
            raise ValueError(
                f"pixels dtype must be uint16, got {self.pixels.dtype}"
            )

        if not isinstance(self.band_names, tuple):
            raise TypeError(
                f"band_names must be a tuple[str, ...], got {type(self.band_names).__name__}"
            )

        if len(self.band_names) != self.pixels.shape[2]:
            raise ValueError(
                f"band_names count ({len(self.band_names)}) does not match "
                f"channel count ({self.pixels.shape[2]})"
            )

        for band in self.band_names:
            if not isinstance(band, str) or not band.strip():
                raise ValueError(f"All band_names must be non-empty strings, got {band!r}")

        if not isinstance(self.timestamp, (int, float)) or self.timestamp < 0:
            raise ValueError(
                f"timestamp must be a non-negative float (seconds), got {self.timestamp!r}"
            )

    @property
    def size_bytes(self) -> int:
        """Return raw memory size in bytes of the pixel buffer."""
        return int(self.pixels.nbytes)

    def __eq__(self, other: object) -> bool:
        """Element-wise equality safe for numpy pixel arrays."""
        if not isinstance(other, Tile):
            return False
        return (
            self.tile_id == other.tile_id
            and self.strip_id == other.strip_id
            and self.row == other.row
            and self.col == other.col
            and self.timestamp == other.timestamp
            and self.band_names == other.band_names
            and np.array_equal(self.pixels, other.pixels)
        )


@dataclass(frozen=True)
class ScoreResult:
    """Represents the output score and execution latency of a scorer for a tile."""

    tile_id: str
    scorer_name: str
    score: float
    cost_seconds: float

    def __post_init__(self) -> None:
        """Validate score result attributes."""
        if not isinstance(self.tile_id, str) or not self.tile_id.strip():
            raise ValueError(f"tile_id must be a non-empty string, got {self.tile_id!r}")

        if not isinstance(self.scorer_name, str) or not self.scorer_name.strip():
            raise ValueError(f"scorer_name must be a non-empty string, got {self.scorer_name!r}")

        if not isinstance(self.score, (int, float)):
            raise TypeError(f"score must be a numeric float, got {type(self.score).__name__}")

        if not isinstance(self.cost_seconds, (int, float)) or self.cost_seconds < 0.0:
            raise ValueError(
                f"cost_seconds must be a non-negative float, got {self.cost_seconds!r}"
            )


@runtime_checkable
class TileSource(Protocol):
    """Protocol for tile providers yielding tiles in capture order."""

    def tiles(self) -> Iterator[Tile]:
        """Yield Tile objects in capture order."""
        ...
