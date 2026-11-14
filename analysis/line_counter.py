"""Counting tracked people crossing a virtual line segment."""

from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class LineCrossing:
    """A track crossed the line. ``direction`` is ``"in"`` or ``"out"``."""

    line: str
    track_id: int
    direction: str
    frame_index: int


def _side(a: Point, b: Point, p: Point) -> float:
    """Cross product sign: which side of the directed line a->b point p lies on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    d1 = _side(q1, q2, p1)
    d2 = _side(q1, q2, p2)
    d3 = _side(p1, p2, q1)
    d4 = _side(p1, p2, q2)
    return d1 * d2 <= 0 and d3 * d4 <= 0


class LineCounter:
    """Count people whose anchor point moves across the segment ``start``-``end``.

    Direction convention (image coordinates, y grows downward): ``"in"`` is a
    move from the negative to the positive side of the directed line
    start->end. For a horizontal line drawn left->right, ``"in"`` means moving
    *down* the image. Swap the endpoints to flip the meaning.

    Event history per ID: with ``count_once_per_direction`` (default) each
    track ID is counted at most once per direction, so a person jittering on
    the line is not counted repeatedly. Points lying exactly on the line keep
    the previous side, which also suppresses noise.
    """

    def __init__(
        self,
        start: Point,
        end: Point,
        *,
        name: str = "line",
        count_once_per_direction: bool = True,
    ) -> None:
        if start == end:
            raise ValueError("line start and end must differ")
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self.name = name
        self.count_once_per_direction = count_once_per_direction
        self.in_count = 0
        self.out_count = 0
        self._last: dict[int, tuple[float, Point]] = {}
        self._counted: dict[int, set[str]] = {}

    @property
    def total(self) -> int:
        return self.in_count + self.out_count

    def update(self, points: dict[int, Point], frame_index: int) -> list[LineCrossing]:
        """Process this frame's ``track_id -> point`` map; return new crossings."""
        events: list[LineCrossing] = []
        for track_id, point in points.items():
            side = _side(self.start, self.end, point)
            if side == 0:
                continue
            previous = self._last.get(track_id)
            self._last[track_id] = (side, point)
            if previous is None:
                continue
            prev_side, prev_point = previous
            if (prev_side > 0) == (side > 0):
                continue
            # Crossed the infinite line; check the movement actually passed
            # through the segment and not beside it.
            if not _segments_intersect(prev_point, point, self.start, self.end):
                continue
            direction = "in" if side > 0 else "out"
            counted = self._counted.setdefault(track_id, set())
            if self.count_once_per_direction and direction in counted:
                continue
            counted.add(direction)
            if direction == "in":
                self.in_count += 1
            else:
                self.out_count += 1
            events.append(LineCrossing(self.name, track_id, direction, frame_index))
        return events

    def reset(self) -> None:
        self.in_count = 0
        self.out_count = 0
        self._last.clear()
        self._counted.clear()
