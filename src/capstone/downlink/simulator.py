"""Downlink pass simulation and tile transmission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capstone.pipeline.queue import BoundedPriorityQueue


@dataclass(frozen=True)
class Pass:
    """Represents a ground station contact window."""

    start_s: float
    duration_s: float
    bandwidth_bytes_per_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0.0:
            raise ValueError(f"start_s must be non-negative, got {self.start_s}")
        if self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be strictly positive, got {self.duration_s}")
        if self.bandwidth_bytes_per_s <= 0.0:
            raise ValueError(
                f"bandwidth_bytes_per_s must be strictly positive, got {self.bandwidth_bytes_per_s}"
            )

    @property
    def end_s(self) -> float:
        """Pass end timestamp in seconds."""
        return self.start_s + self.duration_s

    @property
    def capacity_bytes(self) -> float:
        """Total byte capacity of the pass."""
        return self.duration_s * self.bandwidth_bytes_per_s


@dataclass(frozen=True)
class DeliveredTile:
    """Represents a tile successfully downlinked to ground."""

    tile_id: str
    score: float
    sent_at_s: float
    delay_s_since_capture: float

    def __post_init__(self) -> None:
        if not self.tile_id.strip():
            raise ValueError("tile_id cannot be empty")
        if self.delay_s_since_capture < 0.0:
            raise ValueError(
                f"delay_s_since_capture cannot be negative, got {self.delay_s_since_capture}"
            )


class DownlinkSimulator:
    """Simulates ground passes and transmits prioritized tiles within pass bandwidth."""

    def __init__(self, passes: list[Pass]) -> None:
        self.passes = sorted(passes, key=lambda p: p.start_s)
        for i in range(1, len(self.passes)):
            prev_p = self.passes[i - 1]
            curr_p = self.passes[i]
            if curr_p.start_s < prev_p.end_s:
                raise ValueError(
                    f"Passes must not overlap: {prev_p} (ends at {prev_p.end_s}) "
                    f"overlaps with {curr_p} (starts at {curr_p.start_s})"
                )

        self._remaining_capacity = [float(p.capacity_bytes) for p in self.passes]
        self._bytes_sent = [0 for _ in self.passes]

    def is_pass_active(self, now_s: float) -> bool:
        """Check if any ground station contact pass is currently active at now_s."""
        return any(p.start_s <= now_s <= p.end_s for p in self.passes)

    def transmit(
        self,
        queue: BoundedPriorityQueue,
        now_s: float,
    ) -> list[DeliveredTile]:
        """Transmit the highest-score tiles that fully fit into remaining active pass capacity.

        Head-of-line blocking: tiles are evaluated strictly in priority order; the best
        tile that does not fit in the remaining pass capacity stops the pass immediately
        (lower-scoring tiles behind it are not transmitted in this pass, and stay queued).
        """
        delivered: list[DeliveredTile] = []

        for idx, p in enumerate(self.passes):
            if not (p.start_s <= now_s <= p.end_s):
                continue

            rem_cap = self._remaining_capacity[idx]
            while rem_cap > 0:
                best = queue.peek_best()
                if best is None:
                    break

                # Partial tiles are never sent
                if best.size_bytes <= rem_cap:
                    popped = queue.pop_best()
                    if popped is None:
                        break
                    rem_cap -= popped.size_bytes
                    self._remaining_capacity[idx] = rem_cap
                    self._bytes_sent[idx] += popped.size_bytes

                    sent_at_s = (
                        max(p.start_s, now_s)
                        + self._bytes_sent[idx] / p.bandwidth_bytes_per_s
                    )
                    delay = max(0.0, sent_at_s - popped.tile.timestamp)
                    delivered.append(
                        DeliveredTile(
                            tile_id=popped.tile.tile_id,
                            score=popped.score,
                            sent_at_s=sent_at_s,
                            delay_s_since_capture=delay,
                        )
                    )
                else:
                    # Head-of-line blocking: best tile does not fit, stop this pass
                    break

        return delivered
