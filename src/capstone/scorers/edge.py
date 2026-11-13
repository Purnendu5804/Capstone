"""S2 edge scorer: measures mean Sobel edge gradient magnitude on a normalized band."""

from __future__ import annotations

import numpy as np
from skimage.filters import sobel

from capstone.contracts.types import Tile


class S2EdgeScorer:
    """Computes mean Sobel edge gradient magnitude normalised by a reflectance scale."""

    def __init__(
        self,
        band: str = "B08",
        scale: float = 10000.0,
        name: str = "s2",
    ) -> None:
        if scale <= 0.0:
            raise ValueError(f"scale must be strictly positive, got {scale}")
        self.band = band
        self.scale = float(scale)
        self.name = name

    def _get_band_channel(self, tile: Tile) -> int:
        """Locate target band channel index, defaulting to the first channel (0)."""
        if self.band in tile.band_names:
            return tile.band_names.index(self.band)
        return 0

    def score(self, tile: Tile) -> float:
        """Compute mean Sobel edge gradient magnitude in float64."""
        ch_idx = self._get_band_channel(tile)
        band_data = tile.pixels[:, :, ch_idx].astype(np.float64) / self.scale
        edge_map = sobel(band_data)
        return float(np.mean(edge_map, dtype=np.float64))
