"""Synthetic tile generator producing diverse test tiles."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from capstone.contracts.types import Tile


class SyntheticTileSource:
    """Generates synthetic satellite tiles with deterministic patterns and hidden ground truth."""

    def __init__(
        self,
        num_tiles: int = 12,
        height: int = 64,
        width: int = 64,
        band_names: tuple[str, ...] = ("B02", "B03", "B04", "B08"),
        seed: int = 42,
        strip_id: str = "strip-001",
    ) -> None:
        if num_tiles <= 0:
            raise ValueError(f"num_tiles must be positive, got {num_tiles}")
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be positive, got ({height}, {width})")
        if not band_names:
            raise ValueError("band_names tuple must not be empty")

        self.num_tiles = num_tiles
        self.height = height
        self.width = width
        self.band_names = band_names
        self.seed = seed
        self.strip_id = strip_id

        self._tiles: list[Tile] = []
        self._ground_truth: dict[str, float] = {}
        self._generate_tiles()

    def _generate_tiles(self) -> None:
        """Create deterministic tiles mixing flat, noisy, and high-edge regions."""
        rng = np.random.default_rng(self.seed)
        num_channels = len(self.band_names)

        # Distribute tiles across a simple spatial grid
        grid_cols = max(1, int(np.ceil(np.sqrt(self.num_tiles))))

        for idx in range(self.num_tiles):
            tile_id = f"{self.strip_id}_tile_{idx:04d}"
            row = idx // grid_cols
            col = idx % grid_cols
            timestamp = float(idx * 2.0)

            pattern_type = idx % 3
            pixels, gt_val = self._create_pattern(
                pattern_type=pattern_type,
                rng=rng,
                num_channels=num_channels,
            )

            tile = Tile(
                tile_id=tile_id,
                strip_id=self.strip_id,
                row=row,
                col=col,
                pixels=pixels,
                band_names=self.band_names,
                timestamp=timestamp,
            )
            self._tiles.append(tile)
            self._ground_truth[tile_id] = float(gt_val)

    def _create_pattern(
        self,
        pattern_type: int,
        rng: np.random.Generator,
        num_channels: int,
    ) -> tuple[np.ndarray, float]:
        """Generate pixel array and corresponding ground truth priority value."""
        # Baseline Sentinel-2 reflectance scale (0 - 10000 range in uint16)
        if pattern_type == 0:
            # Flat region: very low variance, low interest
            base_val = rng.uniform(800.0, 1500.0)
            noise = rng.normal(0.0, 3.0, size=(self.height, self.width, num_channels))
            arr = base_val + noise
            gt = float(rng.uniform(0.05, 0.20))
        elif pattern_type == 1:
            # Noisy region: diffuse variance without distinct edges
            base_val = rng.uniform(2000.0, 3500.0)
            noise = rng.normal(0.0, 450.0, size=(self.height, self.width, num_channels))
            arr = base_val + noise
            gt = float(rng.uniform(0.35, 0.60))
        else:
            # Strong edges / structural patterns: high contrast boundary or checkerboard
            arr = np.full((self.height, self.width, num_channels), 1000.0, dtype=np.float64)
            split = self.width // 2
            arr[:, split:, :] = 8000.0
            # Add moderate texture
            texture = rng.normal(0.0, 50.0, size=(self.height, self.width, num_channels))
            arr += texture
            gt = float(rng.uniform(0.80, 0.99))

        clipped = np.clip(arr, 0.0, 65535.0).astype(np.uint16)
        return clipped, gt

    def tiles(self) -> Iterator[Tile]:
        """Yield tiles in capture order."""
        yield from self._tiles

    def ground_truth_value(self, tile_id: str) -> float:
        """Return the hidden ground truth value for evaluation metrics."""
        if tile_id not in self._ground_truth:
            raise KeyError(f"Tile ID {tile_id!r} not found in ground truth store")
        return self._ground_truth[tile_id]

    def get_ground_truth(self, tile_id: str) -> float:
        """Alias for ground_truth_value."""
        return self.ground_truth_value(tile_id)
