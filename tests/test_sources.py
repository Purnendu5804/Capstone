"""Tests for SyntheticTileSource and FolderTileSource."""

from pathlib import Path

import numpy as np
import pytest

from capstone.contracts.types import LabeledTileSource, TileSource
from capstone.sources.folder import FolderTileSource, save_tiles_to_folder
from capstone.sources.synthetic import SyntheticTileSource


class TestSyntheticTileSource:
    """Tests for SyntheticTileSource generation and properties."""

    def test_protocol_conformance(self) -> None:
        source = SyntheticTileSource(num_tiles=3, seed=1)
        assert isinstance(source, TileSource)
        assert isinstance(source, LabeledTileSource)

    def test_docstring_warning(self) -> None:
        assert SyntheticTileSource.__doc__ is not None
        assert "for plumbing tests, not for evaluating scorer quality" in SyntheticTileSource.__doc__

    def test_get_ground_truth_removed(self) -> None:
        source = SyntheticTileSource(num_tiles=2, seed=1)
        assert not hasattr(source, "get_ground_truth")

    def test_determinism_with_fixed_seed(self) -> None:
        source1 = SyntheticTileSource(num_tiles=10, height=32, width=32, seed=42)
        source2 = SyntheticTileSource(num_tiles=10, height=32, width=32, seed=42)

        tiles1 = list(source1.tiles())
        tiles2 = list(source2.tiles())

        assert len(tiles1) == len(tiles2) == 10
        for t1, t2 in zip(tiles1, tiles2, strict=True):
            assert t1.tile_id == t2.tile_id
            assert t1.strip_id == t2.strip_id
            assert t1.row == t2.row
            assert t1.col == t2.col
            assert t1.timestamp == t2.timestamp
            assert np.array_equal(t1.pixels, t2.pixels)
            assert source1.ground_truth_value(t1.tile_id) == source2.ground_truth_value(t2.tile_id)
            assert source1.value(t1.tile_id) == source2.value(t2.tile_id)

    def test_different_seeds_produce_different_data(self) -> None:
        source1 = SyntheticTileSource(num_tiles=5, height=32, width=32, seed=42)
        source2 = SyntheticTileSource(num_tiles=5, height=32, width=32, seed=999)

        tiles1 = list(source1.tiles())
        tiles2 = list(source2.tiles())

        assert not all(
            np.array_equal(t1.pixels, t2.pixels) for t1, t2 in zip(tiles1, tiles2, strict=True)
        )

    def test_ground_truth_isolation(self) -> None:
        source = SyntheticTileSource(num_tiles=3, seed=10)
        tiles = list(source.tiles())
        tile = tiles[0]

        # Verify ground truth is NOT stored inside the Tile dataclass
        assert not hasattr(tile, "ground_truth_value")
        assert not hasattr(tile, "ground_truth")
        assert not hasattr(tile, "value")

        # But accessible via source methods
        gt = source.ground_truth_value(tile.tile_id)
        assert isinstance(gt, float)
        assert 0.0 <= gt <= 1.0
        assert source.value(tile.tile_id) == gt

        with pytest.raises(KeyError):
            source.ground_truth_value("non_existent_tile")

    def test_five_patterns_and_decoupled_ground_truth(self) -> None:
        source = SyntheticTileSource(num_tiles=5, height=32, width=32, seed=100)
        rng = np.random.default_rng(100)

        patterns = [
            source.PATTERN_FLAT_LAND,
            source.PATTERN_TEXTURED_LAND,
            source.PATTERN_STRONG_EDGES,
            source.PATTERN_SENSOR_NOISE,
            source.PATTERN_BRIGHT_CLOUD,
        ]

        generated = {}
        for p in patterns:
            pix, gt = source._create_pattern(p, rng=rng, num_channels=4)
            generated[p] = (pix, gt)

        # Flat land: low variance, moderate value
        flat_pix, flat_gt = generated[source.PATTERN_FLAT_LAND]
        assert flat_gt == 0.60
        assert np.std(flat_pix) < 50.0

        # Textured land: moderate variance, high value
        _, text_gt = generated[source.PATTERN_TEXTURED_LAND]
        assert text_gt == 0.85

        # Strong edges: high contrast, high value
        edge_pix, edge_gt = generated[source.PATTERN_STRONG_EDGES]
        assert edge_gt == 0.95
        assert np.std(edge_pix) > 1000.0

        # Sensor noise: very high variance, but LOW value
        noise_pix, noise_gt = generated[source.PATTERN_SENSOR_NOISE]
        assert noise_gt == 0.05
        assert np.std(noise_pix) > 1000.0

        # Bright cloud: very bright, low variance, LOW value
        cloud_pix, cloud_gt = generated[source.PATTERN_BRIGHT_CLOUD]
        assert cloud_gt == 0.10
        assert np.mean(cloud_pix) > 20000.0
        assert np.std(cloud_pix) < 50.0

        # Verify ground truth is decoupled from variance
        # Sensor noise has much higher variance than flat land, but lower value
        assert np.std(noise_pix) > np.std(flat_pix)
        assert noise_gt < flat_gt


class TestFolderTileSource:
    """Tests for FolderTileSource and round-trip serialization."""

    def test_protocol_conformance(self, tmp_path: Path) -> None:
        source = SyntheticTileSource(num_tiles=2, seed=7)
        save_tiles_to_folder(source.tiles(), tmp_path)
        folder_source = FolderTileSource(tmp_path)
        assert isinstance(folder_source, TileSource)

    def test_round_trip_with_labels(self, tmp_path: Path) -> None:
        """Verify saving synthetic tiles with labels and loading via FolderTileSource."""
        original_source = SyntheticTileSource(num_tiles=6, height=32, width=32, seed=123)
        original_tiles = list(original_source.tiles())
        labels = {t.tile_id: original_source.value(t.tile_id) for t in original_tiles}

        save_tiles_to_folder(original_tiles, tmp_path, labels=labels)

        assert (tmp_path / "manifest.csv").is_file()
        assert (tmp_path / "labels.csv").is_file()
        for t in original_tiles:
            assert (tmp_path / f"{t.tile_id}.npy").is_file()

        folder_source = FolderTileSource(tmp_path)
        assert isinstance(folder_source, LabeledTileSource)

        loaded_tiles = list(folder_source.tiles())
        assert len(loaded_tiles) == len(original_tiles)
        for orig, loaded in zip(original_tiles, loaded_tiles, strict=True):
            assert loaded.tile_id == orig.tile_id
            assert loaded.strip_id == orig.strip_id
            assert loaded.row == orig.row
            assert loaded.col == orig.col
            assert loaded.timestamp == orig.timestamp
            assert loaded.band_names == orig.band_names
            assert np.array_equal(loaded.pixels, orig.pixels)
            assert folder_source.value(loaded.tile_id) == labels[orig.tile_id]

    def test_duplicate_tile_id_in_manifest_raises(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.csv"
        manifest.write_text(
            "tile_id,strip_id,row,col,timestamp\n"
            "dup_01,strip_A,0,0,1.0\n"
            "dup_01,strip_A,0,1,2.0\n",
            encoding="utf-8",
        )
        folder_source = FolderTileSource(tmp_path)
        with pytest.raises(ValueError, match="Duplicate tile_id found in manifest: 'dup_01'"):
            list(folder_source.tiles())

    def test_missing_labels_error(self, tmp_path: Path) -> None:
        source = SyntheticTileSource(num_tiles=2, seed=1)
        save_tiles_to_folder(source.tiles(), tmp_path)  # No labels.csv
        folder_source = FolderTileSource(tmp_path)

        with pytest.raises(ValueError, match="No labels loaded"):
            folder_source.value("any_id")

    def test_missing_label_for_specific_tile_raises(self, tmp_path: Path) -> None:
        source = SyntheticTileSource(num_tiles=2, seed=1)
        tiles = list(source.tiles())
        # Provide label only for first tile
        partial_labels = {tiles[0].tile_id: 0.5}
        save_tiles_to_folder(tiles, tmp_path, labels=partial_labels)

        folder_source = FolderTileSource(tmp_path)
        assert folder_source.value(tiles[0].tile_id) == 0.5

        with pytest.raises(KeyError, match=f"No label found for tile_id: '{tiles[1].tile_id}'"):
            folder_source.value(tiles[1].tile_id)

    def test_capture_order_by_timestamp(self, tmp_path: Path) -> None:
        """Verify FolderTileSource yields tiles sorted by capture timestamp."""
        source = SyntheticTileSource(num_tiles=4, seed=5)
        tiles = list(source.tiles())
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
