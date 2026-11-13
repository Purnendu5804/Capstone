"""S1 variance scorer: measures pixel variance on a selected spectral band."""

from __future__ import annotations

import warnings

import numpy as np

from capstone.contracts.types import Tile


class S1VarianceScorer:
    """Computes pixel variance of a configurable band in float64 precision."""

    def __init__(
        self,
        band: str = "B08",
        strict: bool = True,
        name: str = "s1",
    ) -> None:
        self.band = band
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
        """Compute pixel variance of the selected band in float64."""
        ch_idx = self._get_band_channel(tile)
        band_data = tile.pixels[:, :, ch_idx].astype(np.float64)
        return float(np.var(band_data, dtype=np.float64))
