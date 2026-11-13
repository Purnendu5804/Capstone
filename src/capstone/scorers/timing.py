"""Timing harness and benchmarking utilities for tile scorers."""

from __future__ import annotations

import argparse
import csv
import tracemalloc
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from capstone.contracts.types import Tile
from capstone.scorers.base import Scorer, timed_score
from capstone.scorers.registry import build_scorer
from capstone.sources.synthetic import SyntheticTileSource


def benchmark_scorer(
    scorer: Scorer,
    tiles: Iterable[Tile],
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, Any]:
    """Benchmark scorer latency and peak memory usage in two separate passes."""
    tiles_list = list(tiles)
    if not tiles_list:
        raise ValueError("Cannot benchmark on empty tiles collection")

    # Warmup phase
    warmup_done = 0
    while warmup_done < warmup:
        for t in tiles_list:
            scorer.score(t)
            warmup_done += 1
            if warmup_done >= warmup:
                break

    # Pass 1: Latency measurement (WITHOUT tracemalloc, using timed_score)
    samples_s = np.zeros((repeats, len(tiles_list)), dtype=np.float64)
    for r in range(repeats):
        for idx, tile in enumerate(tiles_list):
            res = timed_score(scorer, tile)
            samples_s[r, idx] = res.cost_seconds

    # Pass 2: Peak memory measurement (WITH tracemalloc)
    tracemalloc.start()
    tracemalloc.reset_peak()
    for tile in tiles_list:
        scorer.score(tile)
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    median_sec = float(np.median(samples_s))
    p95_sec = float(np.percentile(samples_s, 95))
    mean_sec = float(np.mean(samples_s))

    return {
        "scorer_name": scorer.name,
        "num_tiles": len(tiles_list),
        "warmup": warmup,
        "repeats": repeats,
        "median": median_sec,
        "p95": p95_sec,
        "mean": mean_sec,
        "samples_s": samples_s,
        "peak_memory_bytes": int(peak_mem_bytes),
    }


def run_benchmark_cli() -> None:
    """CLI runner to benchmark scorers on synthetic tiles and record results."""
    parser = argparse.ArgumentParser(
        description="Benchmark Capstone tile scorers on synthetic tiles."
    )
    parser.add_argument(
        "--scorers",
        type=str,
        default="s1,s2,random,fifo",
        help="Comma-separated list of scorer names (default: s1,s2,random,fifo)",
    )
    parser.add_argument("--num-tiles", type=int, default=10, help="Number of tiles to test")
    parser.add_argument("--height", type=int, default=256, help="Tile height in pixels")
    parser.add_argument("--width", type=int, default=256, help="Tile width in pixels")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=20, help="Repeat iterations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic tiles")
    parser.add_argument(
        "--output",
        type=str,
        default="results/scorer_benchmarks.csv",
        help="Path to output CSV file",
    )
    args = parser.parse_args()

    scorer_names = [s.strip() for s in args.scorers.split(",") if s.strip()]

    print(f"Generating {args.num_tiles} synthetic tiles ({args.height}x{args.width})...")
    source = SyntheticTileSource(
        num_tiles=args.num_tiles,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    tiles = list(source.tiles())

    results: list[dict[str, Any]] = []

    print(f"Benchmarking scorers: {', '.join(scorer_names)}")
    print(f"{'Scorer':<10} | {'Mean (ms)':<10} | {'Median (ms)':<12} | {'P95 (ms)':<10} | {'Peak Mem (KB)':<14}")
    print("-" * 65)

    for name in scorer_names:
        scorer = build_scorer(name)
        bench = benchmark_scorer(
            scorer=scorer,
            tiles=tiles,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        mean_ms = bench["mean"] * 1000.0
        median_ms = bench["median"] * 1000.0
        p95_ms = bench["p95"] * 1000.0
        peak_kb = bench["peak_memory_bytes"] / 1024.0

        print(
            f"{name:<10} | {mean_ms:<10.3f} | {median_ms:<12.3f} | {p95_ms:<10.3f} | {peak_kb:<14.2f}"
        )

        results.append(
            {
                "scorer": name,
                "num_tiles": args.num_tiles,
                "height": args.height,
                "width": args.width,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "mean_ms": f"{mean_ms:.4f}",
                "median_ms": f"{median_ms:.4f}",
                "p95_ms": f"{p95_ms:.4f}",
                "peak_memory_kb": f"{peak_kb:.2f}",
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scorer",
        "num_tiles",
        "height",
        "width",
        "warmup",
        "repeats",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "peak_memory_kb",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBenchmark results saved to: {out_path}")


if __name__ == "__main__":
    run_benchmark_cli()
