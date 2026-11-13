"""Tests for SyntheticTileSource and FolderTileSource."""

from pathlib import Path

import numpy as np
import pytest

from capstone.contracts.types import TileSource
from capstone.sources.folder import FolderTileSource, save_tiles_to_folder
from capstone.sources.synthetic import SyntheticTileSource


class TestSyntheticTileSource:
    """Tests for SyntheticTileSource generation and properties."""

    def test_protocol_conformance(self) -> None:
        source = SyntheticTileSource(num_tiles=3, seed=1)
        assert isinstance(source, TileSource)

    def test_determinism_with_fixed_seed(self) -> None:
        source1 = SyntheticTileSource(num_tiles=6, height=32, width=32, seed=42)
        source2 = SyntheticTileSource(num_tiles=6, height=32, width=32, seed=42)

        tiles1 = list(source1.tiles())
        tiles2 = list(source2.tiles())

        assert len(tiles1) == len(tiles2) == 6
        for t1, t2 in zip(tiles1, tiles2, strict=True):
            assert t1.tile_id == t2.tile_id
            assert t1.strip_id == t2.strip_id
            assert t1.row == t2.row
            assert t1.col == t2.col
            assert t1.timestamp == t2.timestamp
            assert np.array_equal(t1.pixels, t2.pixels)
            assert source1.ground_truth_value(t1.tile_id) == source2.ground_truth_value(t2.tile_id)

    def test_different_seeds_produce_different_data(self) -> None:
        source1 = SyntheticTileSource(num_tiles=3, height=32, width=32, seed=42)
        source2 = SyntheticTileSource(num_tiles=3, height=32, width=32, seed=999)

        tiles1 = list(source1.tiles())
        tiles2 = list(source2.tiles())

        # At least some pixels must differ between different seeds
        assert not all(np.array_equal(t1.pixels, t2.pixels) for t1, t2 in zip(tiles1, tiles2, strict=True))

    def test_ground_truth_isolation(self) -> None:
        source = SyntheticTileSource(num_tiles=3, seed=10)
        tiles = list(source.tiles())
        tile = tiles[0]

        # Verify ground truth is NOT stored inside the Tile dataclass
        assert not hasattr(tile, "ground_truth_value")
        assert not hasattr(tile, "ground_truth")

        # But accessible via source method
        gt = source.ground_truth_value(tile.tile_id)
        assert isinstance(gt, float)
        assert 0.0 <= gt <= 1.0

        with pytest.raises(KeyError):
            source.ground_truth_value("non_existent_tile")

    def test_diverse_patterns(self) -> None:
        source = SyntheticTileSource(num_tiles=3, height=32, width=32, seed=42)
        tiles = list(source.tiles())

        flat_tile = tiles[0]
        noisy_tile = tiles[1]
        edge_tile = tiles[2]

        # Flat tile should have much lower standard deviation than noisy or edge tile
        assert np.std(flat_tile.pixels) < np.std(noisy_tile.pixels)
        assert np.std(flat_tile.pixels) < np.std(edge_tile.pixels)


class TestFolderTileSource:
    """Tests for FolderTileSource and round-trip serialization."""

    def test_protocol_conformance(self, tmp_path: Path) -> None:
        source = SyntheticTileSource(num_tiles=2, seed=7)
        save_tiles_to_folder(source.tiles(), tmp_path)
        folder_source = FolderTileSource(tmp_path)
        assert isinstance(folder_source, TileSource)

    def test_round_trip(self, tmp_path: Path) -> None:
        """Verify saving synthetic tiles to folder and loading with FolderTileSource is lossless."""
        original_source = SyntheticTileSource(num_tiles=6, height=32, width=32, seed=123)
        original_tiles = list(original_source.tiles())

        save_tiles_to_folder(original_tiles, tmp_path)

        assert (tmp_path / "manifest.csv").is_file()
        for t in original_tiles:
            assert (tmp_path / f"{t.tile_id}.npy").is_file()

        folder_source = FolderTileSource(tmp_path)
        loaded_tiles = list(folder_source.tiles())

        assert len(loaded_tiles) == len(original_tiles)
        for orig, loaded in zip(original_tiles, loaded_tiles, strict=True):
            assert loaded.tile_id == orig.tile_id
            assert loaded.strip_id == orig.strip_id
            assert loaded.row == orig.row
            assert loaded.col == orig.col
            assert loaded.timestamp == orig.timestamp
            assert loaded.band_names == orig.band_names
            assert loaded.pixels.shape == orig.pixels.shape
            assert loaded.pixels.dtype == orig.pixels.dtype
            assert np.array_equal(loaded.pixels, orig.pixels)

    def test_capture_order_by_timestamp(self, tmp_path: Path) -> None:
        """Verify FolderTileSource yields tiles sorted by capture timestamp."""
        source = SyntheticTileSource(num_tiles=4, seed=5)
        tiles = list(source.tiles())
        # Reverse the order when saving
        save_tiles_to_folder(reversed(tiles), tmp_path)

        folder_source = FolderTileSource(tmp_path)
        loaded = list(folder_source.tiles())

        timestamps = [t.timestamp for t in loaded]
        assert timestamps == sorted(timestamps)

    def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Tile directory not found"):
            FolderTileSource(tmp_path / "non_existent_subfolder")

    def test_missing_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            FolderTileSource(tmp_path)

    def test_invalid_manifest_columns(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.csv"
        manifest.write_text("tile_id,row,col\n1,0,0\n", encoding="utf-8")
        folder_source = FolderTileSource(tmp_path)
        with pytest.raises(ValueError, match="Manifest must contain columns"):
            list(folder_source.tiles())

    def test_missing_pixel_file(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.csv"
        manifest.write_text(
            "tile_id,strip_id,row,col,timestamp\ntile_001,strip_01,0,0,1.0\n",
            encoding="utf-8",
        )
        folder_source = FolderTileSource(tmp_path)
        with pytest.raises(FileNotFoundError, match="Tile pixel file missing"):
            list(folder_source.tiles())
