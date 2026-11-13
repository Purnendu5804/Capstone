"""End-to-end pipeline runner orchestrating tile ingest, scoring, queueing, and downlink."""

from __future__ import annotations

from dataclasses import dataclass

from capstone.contracts.types import ScoreResult, TileSource
from capstone.downlink.simulator import DeliveredTile, DownlinkSimulator, Pass
from capstone.pipeline.queue import BoundedPriorityQueue
from capstone.scorers.base import Scorer, timed_score


@dataclass(frozen=True)
class RunResult:
    """Telemetry and outcome data from a complete pipeline execution run."""

    delivered_tiles: list[DeliveredTile]
    evicted_tile_ids: list[str]
    rejected_tile_ids: list[str]
    score_results: list[ScoreResult]
    total_bytes_sent: int
    pushed_count: int
    evicted_count: int
    rejected_count: int


def run_pipeline(
    source: TileSource,
    scorer: Scorer,
    queue_capacity_bytes: int,
    passes: list[Pass],
) -> RunResult:
    """Stream tiles in capture order, score them, prioritize in bounded queue, and downlink during passes."""
    queue = BoundedPriorityQueue(max_bytes=queue_capacity_bytes)
    simulator = DownlinkSimulator(passes)

    delivered_tiles: list[DeliveredTile] = []
    evicted_tile_ids: list[str] = []
    rejected_tile_ids: list[str] = []
    score_results: list[ScoreResult] = []

    tile_sizes: dict[str, int] = {}
    last_time = 0.0

    # Stream tiles in capture order
    for tile in source.tiles():
        tile_sizes[tile.tile_id] = tile.size_bytes
        now_s = tile.timestamp

        # Trigger any pass that started between last_time and now_s
        for p in simulator.passes:
            if last_time < p.start_s <= now_s:
                deliv = simulator.transmit(queue, p.start_s)
                delivered_tiles.extend(deliv)

        # Score tile with timed_score
        score_res = timed_score(scorer, tile)
        score_results.append(score_res)

        # Push to bounded queue
        push_result = queue.push(tile, score_res.score)
        if push_result == [tile.tile_id]:
            rejected_tile_ids.append(tile.tile_id)
        elif push_result:
            evicted_tile_ids.extend(push_result)

        # Transmit if a pass is active at now_s
        if simulator.is_pass_active(now_s):
            deliv = simulator.transmit(queue, now_s)
            delivered_tiles.extend(deliv)

        last_time = max(last_time, now_s)

    # Process passes occurring after or extending beyond the last tile capture
    for p in simulator.passes:
        if p.start_s >= last_time:
            deliv = simulator.transmit(queue, p.start_s)
            delivered_tiles.extend(deliv)
        elif p.start_s <= last_time <= p.end_s:
            deliv = simulator.transmit(queue, p.end_s)
            delivered_tiles.extend(deliv)

    total_bytes_sent = sum(tile_sizes[dt.tile_id] for dt in delivered_tiles)

    return RunResult(
        delivered_tiles=delivered_tiles,
        evicted_tile_ids=evicted_tile_ids,
        rejected_tile_ids=rejected_tile_ids,
        score_results=score_results,
        total_bytes_sent=total_bytes_sent,
        pushed_count=queue.pushed,
        evicted_count=queue.evicted,
        rejected_count=queue.rejected,
    )
