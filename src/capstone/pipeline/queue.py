"""Bounded priority queue with byte-capacity constraints and deterministic tie-breaking."""

from __future__ import annotations

import heapq
from typing import NamedTuple

from capstone.contracts.types import Tile

# Choice: heapq with lazy deletion (from Python standard library, avoids external dependencies).


class QueueEntry(NamedTuple):
    """An entry stored in the priority queue containing a tile and its priority score."""

    tile: Tile
    score: float

    @property
    def tile_id(self) -> str:
        """Tile identifier."""
        return self.tile.tile_id

    @property
    def size_bytes(self) -> int:
        """Memory size in bytes."""
        return self.tile.size_bytes

    @property
    def timestamp(self) -> float:
        """Capture epoch timestamp."""
        return self.tile.timestamp


class PriorityKey:
    """Key where smaller value indicates LOWER priority (used for eviction min-heap)."""

    __slots__ = ("score", "tile_id", "timestamp")

    def __init__(self, score: float, timestamp: float, tile_id: str) -> None:
        self.score = score
        self.timestamp = timestamp
        self.tile_id = tile_id

    def __lt__(self, other: PriorityKey) -> bool:
        if self.score != other.score:
            return self.score < other.score
        if self.timestamp != other.timestamp:
            return self.timestamp > other.timestamp
        return self.tile_id > other.tile_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PriorityKey):
            return False
        return (
            self.score == other.score
            and self.timestamp == other.timestamp
            and self.tile_id == other.tile_id
        )


class ReversePriorityKey:
    """Key where smaller value indicates HIGHER priority (used for pop_best min-heap)."""

    __slots__ = ("score", "tile_id", "timestamp")

    def __init__(self, score: float, timestamp: float, tile_id: str) -> None:
        self.score = score
        self.timestamp = timestamp
        self.tile_id = tile_id

    def __lt__(self, other: ReversePriorityKey) -> bool:
        if self.score != other.score:
            return self.score > other.score
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.tile_id < other.tile_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReversePriorityKey):
            return False
        return (
            self.score == other.score
            and self.timestamp == other.timestamp
            and self.tile_id == other.tile_id
        )


class BoundedPriorityQueue:
    """Memory-bounded priority queue prioritizing high-scoring tiles.

    Tie-breaking policy:
    1. Higher score first
    2. Earlier capture timestamp first
    3. Lexicographically smaller tile_id first
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be strictly positive, got {max_bytes}")

        self._max_bytes = max_bytes
        self._current_bytes = 0

        self.pushed: int = 0
        self.evicted: int = 0
        self.rejected: int = 0

        self._entry_counter: int = 0
        self._entries: dict[int, QueueEntry] = {}
        self._min_heap: list[tuple[PriorityKey, int]] = []
        self._max_heap: list[tuple[ReversePriorityKey, int]] = []

    @property
    def max_bytes(self) -> int:
        """Maximum queue memory capacity in bytes."""
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        """Current total memory occupied by queued tiles in bytes."""
        return self._current_bytes

    def __len__(self) -> int:
        """Number of active tiles in the queue."""
        return len(self._entries)

    def _insert(self, tile: Tile, score: float) -> None:
        """Insert a new entry and push onto internal heaps."""
        self._entry_counter += 1
        entry_id = self._entry_counter
        entry = QueueEntry(tile=tile, score=score)

        self._entries[entry_id] = entry
        self._current_bytes += entry.size_bytes

        min_k = PriorityKey(score, tile.timestamp, tile.tile_id)
        max_k = ReversePriorityKey(score, tile.timestamp, tile.tile_id)

        heapq.heappush(self._min_heap, (min_k, entry_id))
        heapq.heappush(self._max_heap, (max_k, entry_id))

    def _cleanup_min_heap(self) -> None:
        """Discard lazily deleted entries from the top of the min-heap."""
        while self._min_heap and self._min_heap[0][1] not in self._entries:
            heapq.heappop(self._min_heap)

    def _cleanup_max_heap(self) -> None:
        """Discard lazily deleted entries from the top of the max-heap."""
        while self._max_heap and self._max_heap[0][1] not in self._entries:
            heapq.heappop(self._max_heap)

    def peek_best(self) -> QueueEntry | None:
        """Return the highest priority entry without removing it, or None if empty."""
        self._cleanup_max_heap()
        if not self._max_heap:
            return None
        return self._entries[self._max_heap[0][1]]

    def pop_best(self) -> QueueEntry | None:
        """Remove and return the highest priority entry, or None if empty."""
        self._cleanup_max_heap()
        if not self._max_heap:
            return None

        _, entry_id = heapq.heappop(self._max_heap)
        entry = self._entries.pop(entry_id)
        self._current_bytes -= entry.size_bytes
        return entry

    def _peek_lowest(self) -> tuple[int, QueueEntry] | None:
        """Return the entry_id and QueueEntry with lowest priority."""
        self._cleanup_min_heap()
        if not self._min_heap:
            return None
        entry_id = self._min_heap[0][1]
        return entry_id, self._entries[entry_id]

    def _pop_lowest(self) -> QueueEntry | None:
        """Remove and return the lowest priority entry."""
        self._cleanup_min_heap()
        if not self._min_heap:
            return None

        _, entry_id = heapq.heappop(self._min_heap)
        entry = self._entries.pop(entry_id)
        self._current_bytes -= entry.size_bytes
        return entry

    def push(self, tile: Tile, score: float) -> list[str]:
        """Push a tile with its priority score into the queue.

        Returns:
            list[str]: Evicted tile IDs if space was made, or [tile.tile_id] if rejected.
        """
        # If tile alone exceeds capacity, reject immediately
        if tile.size_bytes > self._max_bytes:
            self.rejected += 1
            return [tile.tile_id]

        # If it fits within current capacity, insert directly
        if self._current_bytes + tile.size_bytes <= self._max_bytes:
            self._insert(tile, score)
            self.pushed += 1
            return []

        # Candidate needs space; check if candidate can evict lower-priority items
        needed_bytes = (self._current_bytes + tile.size_bytes) - self._max_bytes
        cand_key = PriorityKey(score, tile.timestamp, tile.tile_id)

        evicted_entries: list[QueueEntry] = []
        bytes_freed = 0

        while bytes_freed < needed_bytes:
            lowest = self._peek_lowest()
            if lowest is None:
                break

            _, low_entry = lowest
            low_key = PriorityKey(
                low_entry.score,
                low_entry.tile.timestamp,
                low_entry.tile.tile_id,
            )

            # If the lowest entry in the queue has equal or higher priority than candidate,
            # we cannot evict it.
            if not (low_key < cand_key):
                break

            popped = self._pop_lowest()
            if popped is not None:
                evicted_entries.append(popped)
                bytes_freed += popped.size_bytes

        if bytes_freed >= needed_bytes:
            # Successfully freed enough space: candidate is accepted
            self.evicted += len(evicted_entries)
            self._insert(tile, score)
            self.pushed += 1
            return [e.tile_id for e in evicted_entries]

        # Could not free enough space: restore all popped items and reject candidate
        for e in evicted_entries:
            self._insert(e.tile, e.score)

        self.rejected += 1
        return [tile.tile_id]
