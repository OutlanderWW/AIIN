"""Per-ID history of positions used to draw and analyse trajectories."""

from __future__ import annotations

import math
from collections import deque

Point = tuple[float, float]


class TrajectoryStore:
    """Keep the last ``max_length`` positions of every tracked person.

    Tracks not seen for more than ``max_missing_frames`` frames are dropped so
    the display does not fill with stale trails. Total travelled distance is
    kept for the whole lifetime of a track, not only for the visible tail.
    """

    def __init__(self, max_length: int = 50, max_missing_frames: int = 30) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames cannot be negative")
        self.max_length = max_length
        self.max_missing_frames = max_missing_frames
        self._paths: dict[int, deque[Point]] = {}
        self._last_seen: dict[int, int] = {}
        self._distance: dict[int, float] = {}
        self._seen_ids: set[int] = set()

    def update(self, points: dict[int, Point], frame_index: int) -> None:
        """Append this frame's ``track_id -> point`` positions."""
        for track_id, point in points.items():
            path = self._paths.get(track_id)
            if path is None:
                path = self._paths[track_id] = deque(maxlen=self.max_length)
            if path:
                last = path[-1]
                self._distance[track_id] = self._distance.get(track_id, 0.0) + math.hypot(
                    point[0] - last[0], point[1] - last[1]
                )
            path.append((float(point[0]), float(point[1])))
            self._last_seen[track_id] = frame_index
            self._seen_ids.add(track_id)

        stale = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if frame_index - last_seen > self.max_missing_frames
        ]
        for track_id in stale:
            del self._paths[track_id]
            del self._last_seen[track_id]

    def path(self, track_id: int) -> list[Point]:
        """Return the stored tail of one track (oldest point first)."""
        return list(self._paths.get(track_id, ()))

    def paths(self) -> dict[int, list[Point]]:
        """Return all currently kept trajectories."""
        return {track_id: list(path) for track_id, path in self._paths.items()}

    def distance(self, track_id: int) -> float:
        """Total distance in pixels travelled by ``track_id`` so far."""
        return self._distance.get(track_id, 0.0)

    @property
    def unique_ids_seen(self) -> int:
        """Number of distinct track IDs observed since creation."""
        return len(self._seen_ids)

    def reset(self) -> None:
        self._paths.clear()
        self._last_seen.clear()
        self._distance.clear()
        self._seen_ids.clear()
