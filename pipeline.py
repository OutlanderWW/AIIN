"""Per-frame processing: tracking -> analysis -> annotated frame + statistics.

The pipeline does not read video or write files; ``main.py`` (CLI) and a
future Streamlit dashboard can both drive it frame by frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from analysis import LineCounter, LineCrossing, MotionHeatmap, TrajectoryStore, Zone, ZoneEvent, ZoneMonitor
from tracking import Track, tracks_to_points
from visualization import (
    draw_line_counter,
    draw_stats_panel,
    draw_tracks,
    draw_trajectories,
    draw_zones,
)


class Tracker(Protocol):
    def update(self, frame: np.ndarray) -> list[Track]: ...


@dataclass(slots=True)
class FrameStats:
    frame_index: int
    timestamp_s: float
    people_in_frame: int
    unique_people: int
    line_in: int
    line_out: int
    fps: float
    zone_counts: dict[str, int] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "frame": self.frame_index,
            "time_s": round(self.timestamp_s, 3),
            "people_in_frame": self.people_in_frame,
            "unique_people": self.unique_people,
            "line_in": self.line_in,
            "line_out": self.line_out,
            "fps": round(self.fps, 2),
        }
        for name, count in self.zone_counts.items():
            row[f"zone_{name}"] = count
        return row


Event = LineCrossing | ZoneEvent


class AnalysisPipeline:
    """Glue between the tracker and the analysis/visualization modules."""

    def __init__(
        self,
        tracker: Tracker,
        frame_size: tuple[int, int],
        *,
        line: Sequence[tuple[float, float]] | None = None,
        zones: Sequence[Zone] = (),
        trail_length: int = 50,
        show_heatmap: bool = False,
        heatmap_alpha: float = 0.45,
        heatmap_decay: float = 1.0,
        record_tracks: bool = True,
    ) -> None:
        self.tracker = tracker
        self.frame_size = frame_size
        self.trajectories = TrajectoryStore(max_length=trail_length)
        self.heatmap = MotionHeatmap(frame_size, decay=heatmap_decay)
        self.line_counter = LineCounter(line[0], line[1]) if line else None
        self.zone_monitor = ZoneMonitor(zones)
        self.show_heatmap = show_heatmap
        self.heatmap_alpha = heatmap_alpha
        self.record_tracks = record_tracks
        # Full history: (frame, track_id, x1, y1, x2, y2, confidence).
        self.track_rows: list[tuple[int, int, float, float, float, float, float]] = []
        self._fps = 0.0
        self._last_time: float | None = None

    def process(
        self, frame: np.ndarray, frame_index: int, timestamp_s: float
    ) -> tuple[np.ndarray, FrameStats, list[Event]]:
        tracks = self.tracker.update(frame)
        points = tracks_to_points(tracks)

        if self.record_tracks:
            self.track_rows.extend(
                (frame_index, t.track_id, *t.bbox, t.confidence) for t in tracks
            )
        self.trajectories.update(points, frame_index)
        self.heatmap.decay_step()
        self.heatmap.add_tracks(tracks)
        events: list[Event] = []
        if self.line_counter is not None:
            events.extend(self.line_counter.update(points, frame_index))
        events.extend(self.zone_monitor.update(points, frame_index))

        self._update_fps()
        stats = FrameStats(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            people_in_frame=len(tracks),
            unique_people=self.trajectories.unique_ids_seen,
            line_in=self.line_counter.in_count if self.line_counter else 0,
            line_out=self.line_counter.out_count if self.line_counter else 0,
            fps=self._fps,
            zone_counts=self.zone_monitor.counts(),
        )
        return self.render(frame, tracks, stats), stats, events

    def render(self, frame: np.ndarray, tracks: list[Track], stats: FrameStats) -> np.ndarray:
        if self.show_heatmap:
            output = self.heatmap.render(frame, alpha=self.heatmap_alpha)
        else:
            output = frame.copy()
        draw_zones(output, self.zone_monitor)
        if self.line_counter is not None:
            draw_line_counter(output, self.line_counter)
        draw_trajectories(output, self.trajectories.paths())
        draw_tracks(output, tracks)

        lines = [
            f"FPS: {stats.fps:.1f}",
            f"People in frame: {stats.people_in_frame}",
            f"Unique IDs: {stats.unique_people}",
        ]
        if self.line_counter is not None:
            lines.append(f"Line in/out: {stats.line_in}/{stats.line_out}")
        lines.extend(f"Zone {name}: {count}" for name, count in stats.zone_counts.items())
        draw_stats_panel(output, lines)
        return output

    def tracks_dataframe(self, frame_offset: int = 0):
        """Recorded tracks as a DataFrame (``frame`` shifted by ``frame_offset``)."""
        import pandas as pd

        df = pd.DataFrame(
            self.track_rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]
        )
        df["frame"] += frame_offset
        return df

    def _update_fps(self) -> None:
        now = time.perf_counter()
        if self._last_time is not None:
            elapsed = now - self._last_time
            if elapsed > 0:
                instant = 1.0 / elapsed
                self._fps = instant if self._fps == 0 else 0.9 * self._fps + 0.1 * instant
        self._last_time = now
