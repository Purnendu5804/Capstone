"""Synthetic tile generator producing diverse test tiles.

WARNING:
    SyntheticTileSource is intended strictly for plumbing and integration tests,
    NOT for evaluating or benchmarking scorer quality. The patterns and ground truth
    values are artificial heuristics designed to verify pipeline flow, contracts,
    data types, and error handling. Real evaluation must use actual Earth observation data.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from capstone.contracts.types import Tile


class SyntheticTileSource:
    """Generates synthetic satellite tiles with deterministic patterns and hidden ground truth.

    WARNING:
        This source is for plumbing tests, not for evaluating scorer quality.
    """

    PATTERN_FLAT_LAND = 0
    PATTERN_TEXTURED_LAND = 1
    PATTERN_STRONG_EDGES = 2
    PATTERN_SENSOR_NOISE = 3
    PATTERN_BRIGHT_CLOUD = 4

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
        """Create deterministic tiles using seeded random choice across 5 patterns."""
        rng = np.random.default_rng(self.seed)
        num_channels = len(self.band_names)

        grid_cols = max(1, int(np.ceil(np.sqrt(self.num_tiles))))
        chosen_patterns = rng.choice(5, size=self.num_tiles)

        for idx, pattern_type in enumerate(chosen_patterns):
            tile_id = f"{self.strip_id}_tile_{idx:04d}"
            row = idx // grid_cols
            col = idx % grid_cols
            timestamp = float(idx * 2.0)

            pixels, gt_val = self._create_pattern(
                pattern_type=int(pattern_type),
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
        """Generate pixel array and pattern ground truth independent of variance or Sobel."""
        shape = (self.height, self.width, num_channels)

        if pattern_type == self.PATTERN_FLAT_LAND:
            # Flat clear land: low variance, moderate utility
            base = rng.uniform(1400.0, 1800.0)
            noise = rng.normal(0.0, 15.0, size=shape)
            arr = base + noise
            gt = 0.60

        elif pattern_type == self.PATTERN_TEXTURED_LAND:
            # Textured land: natural variations across spatial axes, high utility
            y_coords = np.linspace(0, 4 * np.pi, self.height)[:, None, None]
            x_coords = np.linspace(0, 4 * np.pi, self.width)[None, :, None]
            gradient = (np.sin(y_coords) + np.cos(x_coords)) * 500.0
            noise = rng.normal(2000.0, 100.0, size=shape)
            arr = noise + gradient
            gt = 0.85

        elif pattern_type == self.PATTERN_STRONG_EDGES:
            # Strong edges: high contrast checkerboard grid with multiple steps, high utility
            check_size = max(4, min(self.height, self.width) // 4)
            grid_y, grid_x = np.ogrid[:self.height, :self.width]
            checker = ((grid_y // check_size) + (grid_x // check_size)) % 2
            base = np.where(checker[:, :, None] == 1, 8000.0, 1000.0)
            noise = rng.normal(0.0, 40.0, size=shape)
            arr = base + noise
            gt = 0.95

        elif pattern_type == self.PATTERN_SENSOR_NOISE:
            # Pure sensor noise: high variance, but LOW utility
            noise = rng.normal(3000.0, 2500.0, size=shape)
            arr = noise
            gt = 0.05

        elif pattern_type == self.PATTERN_BRIGHT_CLOUD:
            # Uniform bright cloud: very bright, low variance, LOW utility
            base = rng.uniform(22000.0, 26000.0)
            noise = rng.normal(0.0, 20.0, size=shape)
            arr = base + noise
            gt = 0.10

        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")

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

    def value(self, tile_id: str) -> float:
        """Return ground truth value (conforms to LabeledTileSource)."""
        return self.ground_truth_value(tile_id)
