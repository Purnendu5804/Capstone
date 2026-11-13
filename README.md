# Satellite Tile Prioritization Pipeline

An emulated onboard satellite pipeline that scores and prioritizes Sentinel-2 image tiles under compute, memory, and downlink constraints.
The system ingests raw image tiles from an upstream slicer and routes them through lightweight scoring algorithms.
High-priority tiles are queued in a bounded priority queue while lower-value tiles are dropped or deferred.
Prioritized tiles are then scheduled and transmitted via a downlink simulator to a ground receiver station.
Scorers include pixel variance (S1), Sobel edge density (S2), and a lightweight preview model (S3).
Each scorer must operate at a fraction of the cost of heavy downstream models, with execution latency tracked per tile.
The project emphasizes reproducible experiments, deterministic execution with seeds, and modular contracts.
All modules are tested via pytest and linted with ruff under Python 3.11+.

## Usage

```bash
# Install dependencies into a virtual environment (.venv)
make install

# Run the test suite
make test

# Run code style and lint checks
make lint
```

## Slicer Interface Specification (Tile Format)

The pipeline interacts with the upstream slicer via the `Tile` contract or pre-sliced on-disk folders.

### 1. In-Memory `Tile` Contract (`src/capstone/contracts/types.py`)

A frozen, validated dataclass representing an individual tile captured by the satellite:

| Field | Type | Description |
|---|---|---|
| `tile_id` | `str` | Unique tile identifier (e.g. `strip-001_tile_0000`). |
| `strip_id` | `str` | Identifier of the parent image strip / swath. |
| `row` | `int` | Non-negative row index in the strip grid. |
| `col` | `int` | Non-negative column index in the strip grid. |
| `pixels` | `np.ndarray` | 3D array of shape `(H, W, C)` with `uint16` dtype (read-only, writeable=False). |
| `band_names` | `tuple[str, ...]` | Spectral band names matching channel count `C` (e.g. `("B02", "B03", "B04", "B08")`). |
| `timestamp` | `float` | Capture time in seconds; dictates ingestion order. |
| `size_bytes` | `int` (property) | Size of `pixels` in memory in bytes (`pixels.nbytes`). |

Tiles are immutable and hashable based on `tile_id`. Modifying `pixels` in place is disallowed (`writeable = False`), and passing a writeable array creates a defensive read-only copy.

### 2. On-Disk Folder Format (`FolderTileSource`)

To stand in for the real slicer's output, a folder must contain:
1. `<tile_id>.npy` files: Binary NumPy arrays saved via `numpy.save` with shape `(H, W, C)` and dtype `uint16`.
2. `manifest.csv`: A CSV index containing the required columns (duplicate `tile_id` entries are rejected):
   ```csv
   tile_id,strip_id,row,col,timestamp,band_names
   strip-001_tile_0000,strip-001,0,0,0.0,B02;B03;B04;B08
   strip-001_tile_0001,strip-001,0,1,2.0,B02;B03;B04;B08
   ```
   - `band_names` is optional; if omitted, defaults to standard Sentinel-2 4-band channels or auto-generated band identifiers.
   - Ingestion order is strictly sorted by `timestamp` in ascending order.
3. `labels.csv` (optional): Ground truth evaluation values:
   ```csv
   tile_id,value
   strip-001_tile_0000,0.85
   strip-001_tile_0001,0.12
   ```
   When present, `FolderTileSource` implements `LabeledTileSource` and exposes `value(tile_id: str) -> float`. If labels are requested but missing or not loaded, a clear `KeyError` or `ValueError` is raised.

---

## Project Status

### Step 0: Scaffolding (Completed)
- Initial repository layout, packaging, `pyproject.toml`, and Makefile.

### Step 1 & 1.5: Contracts and Tile Sources (Completed)
- **Contracts (`src/capstone/contracts/`)**:
  - `Tile`: Frozen dataclass with read-only pixel arrays (`flags.writeable = False`), `__hash__` on `tile_id`, and strict validation.
  - `ScoreResult`: Frozen dataclass capturing raw, scorer-specific scores and execution latencies.
  - `TileSource`: Protocol defining `tiles() -> Iterator[Tile]`.
  - `LabeledTileSource`: Protocol defining `TileSource` with `value(tile_id: str) -> float`.
- **Sources (`src/capstone/sources/`)**:
  - `SyntheticTileSource`: Seeded deterministic generator with random choice across 5 distinct synthetic patterns (flat clear land, textured land, strong edges, pure sensor noise, uniform bright cloud) with decoupled ground truth values. Docstring warning specifies it is for plumbing tests only.
  - `FolderTileSource`: Ingestion of `.npy` tiles with `manifest.csv` (duplicate detection) and optional `labels.csv`.
  - `save_tiles_to_folder`: Utility for writing tile files, manifests, and optional labels.
- **Dependencies**:
  - `pyproject.toml` isolates `pytest` and `ruff` under `[project.optional-dependencies] dev`, installed via `make install`.

### Not Done / Next Steps (Step 2+)
- Scorers implementation (S1 pixel variance, S2 Sobel edge density, S3 mock preview model).
- Bounded priority queue and onboard pipeline orchestrator.
- Downlink simulator with pass windows and bandwidth budgets.
- Metrics evaluation and experiment harness.
