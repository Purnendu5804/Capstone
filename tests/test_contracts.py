"""Tests for contracts, dataclasses, and validation logic."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from capstone.contracts.types import ScoreResult, Tile, TileSource
from capstone.sources.synthetic import SyntheticTileSource


def make_valid_tile(**overrides) -> Tile:
    """Helper to create a valid Tile instance with optional overrides."""
    kwargs = {
        "tile_id": "tile_001",
        "strip_id": "strip_A",
        "row": 0,
        "col": 0,
        "pixels": np.zeros((32, 32, 4), dtype=np.uint16),
        "band_names": ("B02", "B03", "B04", "B08"),
        "timestamp": 10.0,
    }
    kwargs.update(overrides)
    return Tile(**kwargs)


class TestTileContract:
    """Test validation and properties of the Tile dataclass."""

    def test_valid_tile(self) -> None:
        tile = make_valid_tile()
        assert tile.tile_id == "tile_001"
        assert tile.strip_id == "strip_A"
        assert tile.row == 0
        assert tile.col == 0
        assert tile.pixels.shape == (32, 32, 4)
        assert tile.pixels.dtype == np.uint16
        assert tile.band_names == ("B02", "B03", "B04", "B08")
        assert tile.timestamp == 10.0
        assert tile.size_bytes == 32 * 32 * 4 * 2

    def test_frozen_immutability(self) -> None:
        tile = make_valid_tile()
        with pytest.raises(FrozenInstanceError):
            tile.tile_id = "new_id"  # type: ignore[misc]

    def test_wrong_dtype(self) -> None:
        # float32 instead of uint16
        with pytest.raises(ValueError, match="dtype must be uint16"):
            make_valid_tile(pixels=np.zeros((32, 32, 4), dtype=np.float32))

        # uint8 instead of uint16
        with pytest.raises(ValueError, match="dtype must be uint16"):
            make_valid_tile(pixels=np.zeros((32, 32, 4), dtype=np.uint8))

    def test_wrong_shape_dimensions(self) -> None:
        # 2D shape (H, W)
        with pytest.raises(ValueError, match="pixels must have 3 dimensions"):
            make_valid_tile(pixels=np.zeros((32, 32), dtype=np.uint16))

        # 4D shape (N, H, W, C)
        with pytest.raises(ValueError, match="pixels must have 3 dimensions"):
            make_valid_tile(pixels=np.zeros((1, 32, 32, 4), dtype=np.uint16))

    def test_pixels_not_ndarray(self) -> None:
        with pytest.raises(TypeError, match="pixels must be a numpy.ndarray"):
            make_valid_tile(pixels=[[1, 2], [3, 4]])

    def test_band_names_mismatch(self) -> None:
        # 3 band names for 4 channels
        with pytest.raises(ValueError, match="band_names count .* does not match"):
            make_valid_tile(band_names=("B02", "B03", "B04"))

    def test_band_names_not_tuple(self) -> None:
        with pytest.raises(TypeError, match="band_names must be a tuple"):
            make_valid_tile(band_names=["B02", "B03", "B04", "B08"])

    def test_invalid_tile_id(self) -> None:
        with pytest.raises(ValueError, match="tile_id must be a non-empty string"):
            make_valid_tile(tile_id="")

        with pytest.raises(ValueError, match="tile_id must be a non-empty string"):
            make_valid_tile(tile_id=123)  # type: ignore[arg-type]

    def test_invalid_strip_id(self) -> None:
        with pytest.raises(ValueError, match="strip_id must be a non-empty string"):
            make_valid_tile(strip_id="   ")

    def test_negative_row_or_col(self) -> None:
        with pytest.raises(ValueError, match="row must be a non-negative integer"):
            make_valid_tile(row=-1)

        with pytest.raises(ValueError, match="col must be a non-negative integer"):
            make_valid_tile(col=-5)

    def test_negative_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timestamp must be a non-negative float"):
            make_valid_tile(timestamp=-0.1)

    def test_tile_equality(self) -> None:
        t1 = make_valid_tile()
        t2 = make_valid_tile()
        assert t1 == t2

        t3 = make_valid_tile(pixels=np.ones((32, 32, 4), dtype=np.uint16))
        assert t1 != t3


class TestScoreResultContract:
    """Test validation of ScoreResult dataclass."""

    def test_valid_score_result(self) -> None:
        sr = ScoreResult(
            tile_id="tile_01",
            scorer_name="variance_scorer",
            score=0.85,
            cost_seconds=0.0012,
        )
        assert sr.tile_id == "tile_01"
        assert sr.scorer_name == "variance_scorer"
        assert sr.score == 0.85
        assert sr.cost_seconds == 0.0012

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="cost_seconds must be a non-negative float"):
            ScoreResult(
                tile_id="tile_01",
                scorer_name="scorer",
                score=0.5,
                cost_seconds=-0.01,
            )

    def test_empty_scorer_name_raises(self) -> None:
        with pytest.raises(ValueError, match="scorer_name must be a non-empty string"):
            ScoreResult(
                tile_id="tile_01",
                scorer_name="",
                score=0.5,
                cost_seconds=0.01,
            )


class TestTileSourceProtocol:
    """Verify TileSource protocol compliance."""

    def test_synthetic_source_conforms_to_protocol(self) -> None:
        source = SyntheticTileSource(num_tiles=2, seed=42)
        assert isinstance(source, TileSource)
