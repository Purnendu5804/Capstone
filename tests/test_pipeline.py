"""Tests for BoundedPriorityQueue, DownlinkSimulator, and run_pipeline."""

import numpy as np

from capstone.contracts.types import Tile
from capstone.downlink.simulator import DownlinkSimulator, Pass
from capstone.pipeline.queue import BoundedPriorityQueue
from capstone.pipeline.runner import run_pipeline
from capstone.scorers.registry import build_scorer
from capstone.sources.synthetic import SyntheticTileSource


def make_tile(tile_id: str, height: int = 16, width: int = 16, timestamp: float = 1.0) -> Tile:
    """Helper to create a valid tile of known size (16x16x4 uint16 = 2048 bytes)."""
    pixels = np.zeros((height, width, 4), dtype=np.uint16)
    return Tile(
        tile_id=tile_id,
        strip_id="strip_01",
        row=0,
        col=0,
        pixels=pixels,
        band_names=("B02", "B03", "B04", "B08"),
        timestamp=timestamp,
    )


class TestBoundedPriorityQueue:
    """Tests for BoundedPriorityQueue."""

    def test_capacity_never_exceeded(self) -> None:
        tile_size = 16 * 16 * 4 * 2  # 2048 bytes
        max_bytes = tile_size * 3  # Capacity for 3 tiles
        queue = BoundedPriorityQueue(max_bytes=max_bytes)

        # Push 10 tiles with varying scores
        rng = np.random.default_rng(42)
        for i in range(10):
            tile = make_tile(f"t_{i}", timestamp=float(i))
            score = float(rng.uniform(0.1, 10.0))
            queue.push(tile, score)
            assert queue.current_bytes <= max_bytes
            assert len(queue) <= 3

    def test_eviction_keeps_highest_scored_tiles(self) -> None:
        tile_size = 2048
        queue = BoundedPriorityQueue(max_bytes=tile_size * 2)

        t1 = make_tile("t1", timestamp=1.0)
        t2 = make_tile("t2", timestamp=2.0)
        t3 = make_tile("t3", timestamp=3.0)

        # Push low-scoring tiles
        queue.push(t1, score=1.0)
        queue.push(t2, score=2.0)
        assert len(queue) == 2

        # Push high-scoring tile: t1 (score 1.0) must be evicted
        evicted = queue.push(t3, score=5.0)
        assert evicted == ["t1"]
        assert queue.evicted == 1
        assert len(queue) == 2

        # The remaining tiles should be t3 (5.0) and t2 (2.0)
        best = queue.pop_best()
        assert best is not None
        assert best.tile_id == "t3"
        second = queue.pop_best()
        assert second is not None
        assert second.tile_id == "t2"
        assert len(queue) == 0

    def test_rejection_when_score_lower_than_everything(self) -> None:
        tile_size = 2048
        queue = BoundedPriorityQueue(max_bytes=tile_size * 2)

        queue.push(make_tile("t1"), score=10.0)
        queue.push(make_tile("t2"), score=20.0)

        # Incoming tile has lower score than everything in the full queue
        t_low = make_tile("t_low")
        evicted = queue.push(t_low, score=5.0)

        assert evicted == ["t_low"]
        assert queue.rejected == 1
        assert queue.evicted == 0
        assert len(queue) == 2

    def test_tile_exceeding_total_capacity_rejected(self) -> None:
        queue = BoundedPriorityQueue(max_bytes=1000)
        large_tile = make_tile("large", height=16, width=16)  # 2048 bytes
        res = queue.push(large_tile, score=100.0)
        assert res == ["large"]
        assert queue.rejected == 1
        assert len(queue) == 0

    def test_deterministic_tie_break_timestamp(self) -> None:
        tile_size = 2048
        queue = BoundedPriorityQueue(max_bytes=tile_size * 2)

        # Same score, different timestamps
        t_early = make_tile("t_early", timestamp=5.0)
        t_late = make_tile("t_late", timestamp=10.0)

        queue.push(t_late, score=3.0)
        queue.push(t_early, score=3.0)

        # Earlier timestamp should be popped first
        best = queue.pop_best()
        assert best is not None
        assert best.tile_id == "t_early"

        second = queue.pop_best()
        assert second is not None
        assert second.tile_id == "t_late"

    def test_deterministic_tie_break_tile_id(self) -> None:
        tile_size = 2048
        queue = BoundedPriorityQueue(max_bytes=tile_size * 2)

        # Same score, same timestamp, different tile_ids
        t_alpha = make_tile("alpha", timestamp=5.0)
        t_beta = make_tile("beta", timestamp=5.0)

        queue.push(t_beta, score=4.0)
        queue.push(t_alpha, score=4.0)

        # Lexicographically smaller tile_id should be popped first
        best = queue.pop_best()
        assert best is not None
        assert best.tile_id == "alpha"


class TestDownlinkSimulator:
    """Tests for DownlinkSimulator."""

    def test_pass_never_sends_more_than_capacity(self) -> None:
        tile_size = 2048
        # Pass capacity: 5.0s * 1000 bytes/s = 5000 bytes (fits at most 2 tiles = 4096 bytes)
        passes = [Pass(start_s=10.0, duration_s=5.0, bandwidth_bytes_per_s=1000.0)]
        sim = DownlinkSimulator(passes)

        queue = BoundedPriorityQueue(max_bytes=tile_size * 10)
        for i in range(5):
            queue.push(make_tile(f"t_{i}"), score=float(i))

        assert len(queue) == 5

        # Transmit at active pass time
        delivered = sim.transmit(queue, now_s=12.0)
        assert len(delivered) == 2  # Exactly 2 tiles fit (4096 <= 5000)

        bytes_sent = len(delivered) * tile_size
        assert bytes_sent <= passes[0].capacity_bytes
        assert len(queue) == 3  # Remaining 3 tiles stay in queue

    def test_partial_tiles_never_sent(self) -> None:
        tile_size = 2048
        # Pass capacity: 3000 bytes (fits 1 tile of 2048, remaining 952 bytes cannot fit second tile)
        passes = [Pass(start_s=0.0, duration_s=3.0, bandwidth_bytes_per_s=1000.0)]
        sim = DownlinkSimulator(passes)

        queue = BoundedPriorityQueue(max_bytes=tile_size * 4)
        queue.push(make_tile("t1"), score=2.0)
        queue.push(make_tile("t2"), score=1.0)

        delivered = sim.transmit(queue, now_s=1.0)
        assert len(delivered) == 1
        assert delivered[0].tile_id == "t1"
        assert len(queue) == 1
        assert queue.peek_best() is not None
        assert queue.peek_best().tile_id == "t2"

    def test_inactive_pass_transmits_nothing(self) -> None:
        passes = [Pass(start_s=10.0, duration_s=5.0, bandwidth_bytes_per_s=1000.0)]
        sim = DownlinkSimulator(passes)

        queue = BoundedPriorityQueue(max_bytes=10000)
        queue.push(make_tile("t1"), score=1.0)

        # Before pass starts
        assert sim.transmit(queue, now_s=5.0) == []
        # After pass ends
        assert sim.transmit(queue, now_s=20.0) == []
        assert len(queue) == 1


class TestRunPipeline:
    """Tests for run_pipeline."""

    def test_end_to_end_deterministic(self) -> None:
        source1 = SyntheticTileSource(num_tiles=8, height=32, width=32, seed=42)
        source2 = SyntheticTileSource(num_tiles=8, height=32, width=32, seed=42)

        scorer1 = build_scorer("s1")
        scorer2 = build_scorer("s1")

        tile_size = 32 * 32 * 4 * 2  # 8192 bytes
        queue_cap = tile_size * 3  # Hold 3 tiles

        passes = [
            Pass(start_s=4.0, duration_s=2.0, bandwidth_bytes_per_s=8192.0),
            Pass(start_s=20.0, duration_s=5.0, bandwidth_bytes_per_s=8192.0),
        ]

        res1 = run_pipeline(source1, scorer1, queue_cap, passes)
        res2 = run_pipeline(source2, scorer2, queue_cap, passes)

        assert res1.delivered_tiles == res2.delivered_tiles
        assert res1.total_bytes_sent == res2.total_bytes_sent
        assert res1.evicted_tile_ids == res2.evicted_tile_ids
        assert res1.rejected_tile_ids == res2.rejected_tile_ids
        assert res1.pushed_count == res2.pushed_count
        assert res1.evicted_count == res2.evicted_count
        assert res1.rejected_count == res2.rejected_count
        assert len(res1.score_results) == 8

        # Delays must be non-negative
        for dt in res1.delivered_tiles:
            assert dt.delay_s_since_capture >= 0.0

    def test_baselines_run_through_runner(self) -> None:
        source = SyntheticTileSource(num_tiles=6, height=32, width=32, seed=7)
        queue_cap = 32 * 32 * 4 * 2 * 2
        passes = [Pass(start_s=5.0, duration_s=10.0, bandwidth_bytes_per_s=5000.0)]

        # Random baseline
        res_random = run_pipeline(source, build_scorer("random", seed=42), queue_cap, passes)
        assert len(res_random.score_results) == 6

        # FIFO baseline
        source_fifo = SyntheticTileSource(num_tiles=6, height=32, width=32, seed=7)
        res_fifo = run_pipeline(source_fifo, build_scorer("fifo"), queue_cap, passes)
        assert len(res_fifo.score_results) == 6
