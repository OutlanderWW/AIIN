"""Drawing helpers. All functions draw *in place* and return the same frame."""

from __future__ import annotations

import colorsys
from typing import Iterable, Sequence

import cv2
import numpy as np

from analysis.line_counter import LineCounter
from analysis.zones import ZoneMonitor
from tracking.person_tracker import Point, Track

Color = tuple[int, int, int]
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for_id(track_id: int) -> Color:
    """Stable, well-separated BGR color for a track ID (golden-ratio hues)."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _label(frame: np.ndarray, text: str, origin: tuple[int, int], color: Color) -> None:
    (tw, th), baseline = cv2.getTextSize(text, _FONT, 0.5, 1)
    x, y = origin
    y = max(th + baseline + 2, y)
    cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - baseline), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_tracks(frame: np.ndarray, tracks: Iterable[Track]) -> np.ndarray:
    """Bounding box, ID label and anchor point for every track."""
    for track in tracks:
        color = color_for_id(track.track_id)
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _label(frame, f"ID {track.track_id} {track.confidence:.2f}", (x1, y1), color)
        ax, ay = track.anchor
        cv2.circle(frame, (int(ax), int(ay)), 4, color, -1)
    return frame


def draw_trajectories(
    frame: np.ndarray, paths: dict[int, Sequence[Point]], *, thickness: int = 2
) -> np.ndarray:
    """Polyline per track; older segments are thinner."""
    for track_id, path in paths.items():
        if len(path) < 2:
            continue
        color = color_for_id(track_id)
        points = np.round(np.asarray(path)).astype(np.int32)
        n = len(points)
        for i in range(1, n):
            width = max(1, int(round(thickness * (0.4 + 0.6 * i / n))))
            cv2.line(frame, tuple(points[i - 1]), tuple(points[i]), color, width, cv2.LINE_AA)
    return frame


def draw_line_counter(frame: np.ndarray, counter: LineCounter) -> np.ndarray:
    start = tuple(int(round(v)) for v in counter.start)
    end = tuple(int(round(v)) for v in counter.end)
    cv2.line(frame, start, end, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, start, 6, (0, 255, 255), -1)
    text = f"{counter.name}: in {counter.in_count} / out {counter.out_count}"
    _label(frame, text, (start[0], start[1] - 8), (0, 255, 255))
    return frame


def draw_zones(frame: np.ndarray, monitor: ZoneMonitor, *, alpha: float = 0.2) -> np.ndarray:
    """Semi-transparent zone fill with the current occupancy."""
    if not monitor.zones:
        return frame
    counts = monitor.counts()
    overlay = frame.copy()
    for zone in monitor.zones:
        color = (255, 120, 0) if counts[zone.name] == 0 else (0, 80, 255)
        cv2.fillPoly(overlay, [zone.polygon], color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
    for zone in monitor.zones:
        color = (255, 120, 0) if counts[zone.name] == 0 else (0, 80, 255)
        cv2.polylines(frame, [zone.polygon], True, color, 2, cv2.LINE_AA)
        x, y = zone.polygon.min(axis=0)
        _label(frame, f"{zone.name}: {counts[zone.name]}", (int(x), int(y) + 18), color)
    return frame


def draw_stats_panel(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Dark translucent panel in the top-left corner with one line per entry."""
    if not lines:
        return frame
    line_height = 22
    width = max(cv2.getTextSize(line, _FONT, 0.55, 1)[0][0] for line in lines) + 20
    height = line_height * len(lines) + 12
    roi = frame[0:height, 0:width]
    roi[:] = (roi * 0.4).astype(frame.dtype)
    for i, line in enumerate(lines):
        y = 22 + i * line_height
        cv2.putText(frame, line, (10, y), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
