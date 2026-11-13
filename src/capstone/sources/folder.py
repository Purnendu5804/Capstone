"""Folder-based tile ingest for pre-sliced tiles."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from capstone.contracts.types import Tile


def save_tiles_to_folder(
    tiles: Iterable[Tile],
    folder: Path | str,
    manifest_name: str = "manifest.csv",
) -> Path:
    """Save an iterable of Tile objects to disk as .npy files and a manifest.csv."""
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    manifest_file = folder_path / manifest_name
    fieldnames = ["tile_id", "strip_id", "row", "col", "timestamp", "band_names"]

    tiles_list = list(tiles)

    with manifest_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tile in tiles_list:
            tile_path = folder_path / f"{tile.tile_id}.npy"
            np.save(tile_path, tile.pixels)

            writer.writerow(
                {
                    "tile_id": tile.tile_id,
                    "strip_id": tile.strip_id,
                    "row": tile.row,
                    "col": tile.col,
                    "timestamp": tile.timestamp,
                    "band_names": ";".join(tile.band_names),
                }
            )

    return folder_path


class FolderTileSource:
    """Reads pre-sliced tiles from a folder containing .npy files and a manifest.csv."""

    def __init__(
        self,
        folder: Path | str,
        manifest_name: str = "manifest.csv",
        default_band_names: tuple[str, ...] | None = None,
    ) -> None:
        self.folder = Path(folder)
        self.manifest_path = self.folder / manifest_name
        self.default_band_names = default_band_names

        if not self.folder.is_dir():
            raise FileNotFoundError(f"Tile directory not found: {self.folder}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

    def tiles(self) -> Iterator[Tile]:
        """Load and yield Tile objects sorted by capture order (timestamp)."""
        records: list[dict[str, str]] = []

        with self.manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = {"tile_id", "strip_id", "row", "col", "timestamp"}
            if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Manifest must contain columns {required_cols}, found: {reader.fieldnames}"
                )
            records = list(reader)

        # Sort by timestamp to guarantee capture order
        records.sort(key=lambda r: float(r["timestamp"]))

        for record in records:
            tile_id = record["tile_id"]
            npy_path = self.folder / f"{tile_id}.npy"
            if not npy_path.is_file():
                raise FileNotFoundError(f"Tile pixel file missing: {npy_path}")

            pixels = np.load(npy_path)

            if "band_names" in record and record["band_names"].strip():
                band_names = tuple(record["band_names"].strip().split(";"))
            elif self.default_band_names is not None:
                band_names = self.default_band_names
            else:
                num_channels = pixels.shape[2] if pixels.ndim == 3 else 0
                if num_channels == 4:
                    band_names = ("B02", "B03", "B04", "B08")
                else:
                    band_names = tuple(f"B{i+1:02d}" for i in range(num_channels))

            yield Tile(
                tile_id=tile_id,
                strip_id=record["strip_id"],
                row=int(record["row"]),
                col=int(record["col"]),
                pixels=pixels,
                band_names=band_names,
                timestamp=float(record["timestamp"]),
            )
