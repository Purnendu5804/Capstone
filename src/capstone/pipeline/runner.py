"""End-to-end pipeline runner orchestrating tile ingest, scoring, queueing, and downlink."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from capstone.contracts.types import ScoreResult, Tile, TileSource
from capstone.downlink.simulator import DeliveredTile, DownlinkSimulator, Pass
from capstone.pipeline.queue import BoundedPriorityQueue
from capstone.scorers.base import Scorer, timed_score


@dataclass(frozen=True)
class RunResult:
    """Telemetry and outcome data from a complete pipeline execution run."""

    delivered_tiles: list[DeliveredTile]
    evicted_tile_ids: list[str]
    rejected_tile_ids: list[str]
    remaining_tile_ids: list[str]
    dropped_unscored_tile_ids: list[str]
    score_results: list[ScoreResult]
    total_bytes_sent: int
    pushed_count: int
    evicted_count: int
    rejected_count: int


def _score_with_override(
    scorer: Scorer,
    tile: Tile,
    cost_override: Callable[[Tile], float] | None,
) -> ScoreResult:
    """Score a tile via timed_score and apply cost_override if provided."""
    res = timed_score(scorer, tile)
    if cost_override is None:
        return res

    override_cost = float(cost_override(tile))
    if override_cost < 0.0:
        raise ValueError(f"cost_override returned negative cost: {override_cost}")

    return ScoreResult(
        tile_id=res.tile_id,
        scorer_name=res.scorer_name,
        score=res.score,
        cost_seconds=override_cost,
    )


def run_pipeline(
    source: TileSource,
    scorer: Scorer,
    queue_capacity_bytes: int,
    passes: list[Pass],
    compute_speed_factor: float = 1.0,
    input_buffer_bytes: int | None = None,
    cost_override: Callable[[Tile], float] | None = None,
) -> RunResult:
    """Stream tiles in capture order, simulate single-core scoring, queue, and downlink.

    Args:
        source: Tile provider yielding tiles in capture order.
        scorer: Scorer implementation conforming to Scorer protocol.
        queue_capacity_bytes: Max memory capacity in bytes for the priority queue.
        passes: List of non-overlapping downlink contact windows.
        compute_speed_factor: Simulated onboard seconds per measured scoring second.
        input_buffer_bytes: Max bytes for the FIFO input buffer while the scorer is busy;
            None means unlimited buffering.
        cost_override: Optional deterministic replacement for measured scoring cost (seconds).
    """
    if compute_speed_factor < 0.0:
        raise ValueError(
            f"compute_speed_factor must be non-negative, got {compute_speed_factor}"
        )
    if input_buffer_bytes is not None and input_buffer_bytes < 0:
        raise ValueError(
            f"input_buffer_bytes must be non-negative or None, got {input_buffer_bytes}"
        )

    queue = BoundedPriorityQueue(max_bytes=queue_capacity_bytes)
    simulator = DownlinkSimulator(passes)

    delivered_tiles: list[DeliveredTile] = []
    evicted_tile_ids: list[str] = []
    rejected_tile_ids: list[str] = []
    dropped_unscored_tile_ids: list[str] = []
    score_results: list[ScoreResult] = []
    tile_sizes: dict[str, int] = {}

    tile_iter = iter(source.tiles())
    next_tile: Tile | None = next(tile_iter, None)

    pass_idx = 0
    num_passes = len(simulator.passes)

    input_buffer: deque[Tile] = deque()
    input_buffer_used = 0
    # (tile, score_res, finish_time_s) when the single simulated core is busy
    scoring_in_flight: tuple[Tile, ScoreResult, float] | None = None

    def start_scoring(tile: Tile, now_s: float) -> tuple[Tile, ScoreResult, float]:
        res = _score_with_override(scorer, tile, cost_override)
        score_results.append(res)
        finish_s = now_s + res.cost_seconds * compute_speed_factor
        return (tile, res, finish_s)

    while (
        next_tile is not None
        or scoring_in_flight is not None
        or pass_idx < num_passes
    ):
        # Determine earliest pending event time
        t_finish = scoring_in_flight[2] if scoring_in_flight is not None else float("inf")
        t_arrival = next_tile.timestamp if next_tile is not None else float("inf")
        t_pass = simulator.passes[pass_idx].start_s if pass_idx < num_passes else float("inf")

        earliest = min(t_finish, t_arrival, t_pass)

        # Tie-break at equal timestamps: scoring_finish -> arrival -> pass_start
        if t_finish == earliest and scoring_in_flight is not None:
            finished_tile, finished_res, now_s = scoring_in_flight
            scoring_in_flight = None

            push_result = queue.push(finished_tile, finished_res.score)
            if push_result == [finished_tile.tile_id]:
                rejected_tile_ids.append(finished_tile.tile_id)
            elif push_result:
                evicted_tile_ids.extend(push_result)

            if simulator.is_pass_active(now_s):
                delivered_tiles.extend(simulator.transmit(queue, now_s))

            if input_buffer:
                buffered_tile = input_buffer.popleft()
                input_buffer_used -= buffered_tile.size_bytes
                scoring_in_flight = start_scoring(buffered_tile, now_s)

        elif t_arrival == earliest and next_tile is not None:
            arriving = next_tile
            next_tile = next(tile_iter, None)
            tile_sizes[arriving.tile_id] = arriving.size_bytes
            now_s = arriving.timestamp

            if scoring_in_flight is None:
                scoring_in_flight = start_scoring(arriving, now_s)
            elif (
                input_buffer_bytes is None
                or input_buffer_used + arriving.size_bytes <= input_buffer_bytes
            ):
                input_buffer.append(arriving)
                input_buffer_used += arriving.size_bytes
            else:
                dropped_unscored_tile_ids.append(arriving.tile_id)

        else:
            p = simulator.passes[pass_idx]
            pass_idx += 1
            delivered_tiles.extend(simulator.transmit(queue, p.start_s))

    total_bytes_sent = sum(tile_sizes[dt.tile_id] for dt in delivered_tiles)

    return RunResult(
        delivered_tiles=delivered_tiles,
        evicted_tile_ids=evicted_tile_ids,
        rejected_tile_ids=rejected_tile_ids,
        remaining_tile_ids=queue.tile_ids(),
        dropped_unscored_tile_ids=dropped_unscored_tile_ids,
        score_results=score_results,
        total_bytes_sent=total_bytes_sent,
        pushed_count=queue.pushed,
        evicted_count=queue.evicted,
        rejected_count=queue.rejected,
    )
