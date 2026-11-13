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

# Benchmark scorers on 256x256 synthetic tiles and output results to results/
python -m capstone.scorers.timing --scorers s1,s2,s2_mean,random,fifo
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

## Scorers and Registry

Scorers compute an onboard priority score (`float`) for each tile. Higher scores denote higher priority for downlink.

### Built-in Scorers

1. **S1 Variance (`S1VarianceScorer`)**:
   - Computes pixel variance of a single spectral band in `float64` (default `"B08"`).
   - `strict: bool = True`: raises `ValueError` if band is missing; when `strict=False`, warns once per instance and falls back to channel 0.
2. **S2 Edge Density (`S2EdgeScorer`)**:
   - Computes true edge density (`mode="density"`, returning fraction of pixels exceeding gradient `threshold` in `[0, 1]`) or mean Sobel gradient magnitude (`mode="mean_magnitude"`).
   - Normalised by reflectance constant (default `10000.0`). Default `threshold=0.02`.
   - *Note: The 0.02 default threshold is a placeholder to be calibrated on real tiles.*
   - `strict: bool = True`: raises `ValueError` if band is missing; when `strict=False`, warns once per instance and falls back to channel 0.
3. **Random Baseline (`RandomScorer`)**:
   - Pure function of `(seed, tile_id)` mapped stably via `hashlib.blake2b` to $[0.0, 1.0)$, completely independent of call order.
4. **FIFO Baseline (`FifoScorer`)**:
   - Assigns priority based on capture order (`-tile.timestamp`), ensuring earlier tiles receive higher scores.

### Scorer Protocol & Timing Harness

All scorers conform to the `Scorer` Protocol:
```python
class Scorer(Protocol):
    name: str
    def score(self, tile: Tile) -> float: ...
```

The helper `timed_score(scorer, tile) -> ScoreResult` measures execution cost in nanoseconds using `time.perf_counter_ns` (including preprocessing) and returns a validated `ScoreResult`.

Latency and peak memory usage are measured in two separate passes: latency without `tracemalloc` distortion using `timed_score`, and peak memory with `tracemalloc`. `benchmark_scorer` returns `samples_s` of shape `(repeats, num_tiles)`, along with `median`, `p95`, `mean`, and `peak_memory_bytes`.

### Registry & Plugging in External Models

Scorers are constructed dynamically using `build_scorer(name, **kwargs)`. Canonical names are `"s1"`, `"s2"` (edge density), `"s2_mean"` (mean Sobel magnitude), `"random"`, and `"fifo"`.

Teammates can register external lightweight or preview models without modifying pipeline logic:

```python
from capstone.contracts.types import Tile
from capstone.scorers import register_scorer

class TeammatePreviewScorer:
    name = "s3_preview"

    def score(self, tile: Tile) -> float:
        # Preprocessing & inference logic
        return 0.85

# Register with the pipeline
register_scorer("s3", TeammatePreviewScorer)
```

### Benchmarking CLI

Benchmark scorer throughput and memory overhead using synthetic tiles:
```bash
python -m capstone.scorers.timing --scorers s1,s2,s2_mean,random,fifo --num-tiles 10 --height 256 --width 256 --repeats 20
```
Outputs execution latency metrics and tracemalloc peak memory to `results/scorer_benchmarks.csv`.

---

## Project Status

### Step 0: Scaffolding (Completed)
- Initial repository layout, packaging, `pyproject.toml`, and Makefile.

### Step 1 & 1.5: Contracts and Tile Sources (Completed)
- `Tile` contract with strict validation, read-only pixel enforcement, and hashing.
- `ScoreResult` contract with documentation on scorer-relative scores.
- `SyntheticTileSource` with 5 decoupled test patterns and `FolderTileSource` with `manifest.csv` / `labels.csv`.

### Step 2 & 2.5: Scorers, Baselines & Timing Harness (Completed)
- `Scorer` Protocol and `timed_score` latency measurement harness.
- `S1VarianceScorer`: Configurable band variance in float64 with `strict` flag.
- `S2EdgeScorer`: True edge density (`mode="density"`, `threshold=0.02`) and mean gradient magnitude (`mode="mean_magnitude"`) with `strict` flag.
- `RandomScorer`: Pure function of `(seed, tile_id)` using blake2b stable hashing independent of call order.
- `FifoScorer`: Capture-order baseline.
- Scorer registry with canonical names (`s1`, `s2`, `s2_mean`, `random`, `fifo`) and alias mapping.
- Two-pass `benchmark_scorer` harness and CLI (`python -m capstone.scorers.timing --scorers ...`).

### Not Done / Next Steps (Step 3+)
- Bounded priority queue and onboard pipeline orchestrator.
- Downlink simulator with pass windows and bandwidth budgets.
- Experiment orchestration and metric logging against ground truth.
