"""Polygonal zones of interest: occupancy and enter/exit events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ZoneEvent:
    """A track entered or left a zone. ``kind`` is ``"enter"`` or ``"exit"``."""

    zone: str
    track_id: int
    kind: str
    frame_index: int


class Zone:
    """A named polygon (at least 3 vertices) in image pixel coordinates."""

    def __init__(self, name: str, polygon: Sequence[Point]) -> None:
        contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(contour) < 3:
            raise ValueError("a zone polygon needs at least 3 points")
        self.name = name
        self._contour = contour

    @classmethod
    def rectangle(cls, name: str, top_left: Point, bottom_right: Point) -> "Zone":
        (x1, y1), (x2, y2) = top_left, bottom_right
        return cls(name, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    @property
    def polygon(self) -> np.ndarray:
        """Integer vertices, ready for ``cv2.polylines``."""
        return np.round(self._contour).astype(np.int32)

    def contains(self, point: Point) -> bool:
        """True if the point is inside the polygon or on its border."""
        return cv2.pointPolygonTest(self._contour, (float(point[0]), float(point[1])), False) >= 0


class ZoneMonitor:
    """Track which IDs are inside each zone.

    A track that disappears (occlusion, missed detection) is kept as an
    occupant for ``max_missing_frames`` frames before an ``exit`` is emitted;
    this prevents enter/exit flicker when the tracker briefly loses someone.
    """

    def __init__(self, zones: Iterable[Zone], *, max_missing_frames: int = 15) -> None:
        self.zones = list(zones)
        names = [zone.name for zone in self.zones]
        if len(set(names)) != len(names):
            raise ValueError("zone names must be unique")
        self.max_missing_frames = max_missing_frames
        self._occupants: dict[str, dict[int, int]] = {name: {} for name in names}
        self.total_entries: dict[str, int] = {name: 0 for name in names}

    def update(self, points: dict[int, Point], frame_index: int) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        for zone in self.zones:
            occupants = self._occupants[zone.name]
            for track_id, point in points.items():
                if zone.contains(point):
                    if track_id not in occupants:
                        self.total_entries[zone.name] += 1
                        events.append(ZoneEvent(zone.name, track_id, "enter", frame_index))
                    occupants[track_id] = frame_index
                elif track_id in occupants:
                    del occupants[track_id]
                    events.append(ZoneEvent(zone.name, track_id, "exit", frame_index))
            for track_id, last_seen in list(occupants.items()):
                if frame_index - last_seen > self.max_missing_frames:
                    del occupants[track_id]
                    events.append(ZoneEvent(zone.name, track_id, "exit", frame_index))
        return events

    def counts(self) -> dict[str, int]:
        """Current number of people per zone."""
        return {name: len(occupants) for name, occupants in self._occupants.items()}

    def occupants(self, zone_name: str) -> set[int]:
        return set(self._occupants[zone_name])

    def reset(self) -> None:
        for name in self._occupants:
            self._occupants[name].clear()
            self.total_entries[name] = 0
