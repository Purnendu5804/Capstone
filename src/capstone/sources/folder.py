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
    labels: dict[str, float] | None = None,
    labels_name: str = "labels.csv",
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

    if labels is not None:
        labels_file = folder_path / labels_name
        with labels_file.open("w", newline="", encoding="utf-8") as f:
            lbl_writer = csv.DictWriter(f, fieldnames=["tile_id", "value"])
            lbl_writer.writeheader()
            for tid, val in labels.items():
                lbl_writer.writerow({"tile_id": tid, "value": val})

    return folder_path


class FolderTileSource:
    """Reads pre-sliced tiles from a folder containing .npy files and a manifest.csv."""

    def __init__(
        self,
        folder: Path | str,
        manifest_name: str = "manifest.csv",
        labels_name: str | Path | None = "labels.csv",
        default_band_names: tuple[str, ...] | None = None,
    ) -> None:
        self.folder = Path(folder)
        self.manifest_path = self.folder / manifest_name
        self.default_band_names = default_band_names

        if not self.folder.is_dir():
            raise FileNotFoundError(f"Tile directory not found: {self.folder}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self._labels: dict[str, float] | None = None
        if labels_name is not None:
            if isinstance(labels_name, Path) and labels_name.is_absolute():
                labels_file = labels_name
            else:
                labels_file = self.folder / labels_name

            if labels_file.is_file():
                self._load_labels(labels_file)
            elif labels_name != "labels.csv":
                raise FileNotFoundError(f"Specified labels file not found: {labels_file}")

    def _load_labels(self, labels_file: Path) -> None:
        """Load ground truth labels from CSV file."""
        self._labels = {}
        with labels_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"tile_id", "value"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Labels file must contain columns {required}, found: {reader.fieldnames}"
                )
            for row in reader:
                self._labels[row["tile_id"].strip()] = float(row["value"])

    def value(self, tile_id: str) -> float:
        """Return the ground truth value for tile_id (implements LabeledTileSource)."""
        if self._labels is None:
            raise ValueError(
                f"No labels loaded for source at {self.folder}. Provide a valid labels.csv."
            )
        if tile_id not in self._labels:
            raise KeyError(f"No label found for tile_id: {tile_id!r}")
        return self._labels[tile_id]

    def tiles(self) -> Iterator[Tile]:
        """Load and yield Tile objects sorted by capture order (timestamp)."""
        with self.manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = {"tile_id", "strip_id", "row", "col", "timestamp"}
            if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Manifest must contain columns {required_cols}, found: {reader.fieldnames}"
                )
            records = list(reader)

        # Validate duplicate tile_ids in manifest
        seen_tile_ids: set[str] = set()
        for record in records:
            tile_id = record["tile_id"]
            if tile_id in seen_tile_ids:
                raise ValueError(f"Duplicate tile_id found in manifest: {tile_id!r}")
            seen_tile_ids.add(tile_id)

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
