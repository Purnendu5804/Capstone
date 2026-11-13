"""S1 variance scorer: measures pixel variance on a selected spectral band."""

from __future__ import annotations

import numpy as np

from capstone.contracts.types import Tile


class S1VarianceScorer:
    """Computes pixel variance of a configurable band in float64 precision."""

    def __init__(self, band: str = "B08", name: str = "s1") -> None:
        self.band = band
        self.name = name

    def _get_band_channel(self, tile: Tile) -> int:
        """Locate the target band channel index, defaulting to the first channel (0)."""
        if self.band in tile.band_names:
            return tile.band_names.index(self.band)
        return 0

    def score(self, tile: Tile) -> float:
        """Compute pixel variance of the selected band in float64."""
        ch_idx = self._get_band_channel(tile)
        band_data = tile.pixels[:, :, ch_idx].astype(np.float64)
        return float(np.var(band_data, dtype=np.float64))
