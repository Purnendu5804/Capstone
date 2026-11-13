"""S2 edge scorer: measures edge density or mean Sobel gradient magnitude."""

from __future__ import annotations

import warnings

import numpy as np
from skimage.filters import sobel

from capstone.contracts.types import Tile


class S2EdgeScorer:
    """Computes edge density or mean Sobel gradient magnitude on a normalised band."""

    def __init__(
        self,
        band: str = "B08",
        scale: float = 10000.0,
        mode: str = "density",
        threshold: float = 0.02,
        strict: bool = True,
        name: str = "s2",
    ) -> None:
        if scale <= 0.0:
            raise ValueError(f"scale must be strictly positive, got {scale}")
        if threshold < 0.0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if mode not in ("density", "mean_magnitude"):
            raise ValueError(
                f"mode must be 'density' or 'mean_magnitude', got {mode!r}"
            )

        self.band = band
        self.scale = float(scale)
        self.mode = mode
        self.threshold = float(threshold)
        self.strict = strict
        self.name = name
        self._warned_missing_band = False

    def _get_band_channel(self, tile: Tile) -> int:
        """Locate target band channel index, respecting strict flag if band is missing."""
        if self.band in tile.band_names:
            return tile.band_names.index(self.band)

        if self.strict:
            raise ValueError(
                f"Band {self.band!r} not found in tile.band_names: {tile.band_names}. "
                f"Set strict=False to fall back to channel 0."
            )

        if not self._warned_missing_band:
            warnings.warn(
                f"Band {self.band!r} not found in tile.band_names {tile.band_names}; "
                f"falling back to channel 0.",
                UserWarning,
                stacklevel=3,
            )
            self._warned_missing_band = True

        return 0

    def score(self, tile: Tile) -> float:
        """Compute edge density in [0, 1] or mean Sobel gradient magnitude."""
        ch_idx = self._get_band_channel(tile)
        band_data = tile.pixels[:, :, ch_idx].astype(np.float64) / self.scale
        edge_map = sobel(band_data)

        if self.mode == "density":
            # True edge density: fraction of pixels exceeding gradient threshold
            return float(np.mean(edge_map > self.threshold, dtype=np.float64))

        return float(np.mean(edge_map, dtype=np.float64))
