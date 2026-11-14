"""Run the whole system on one video source and save the results.

Used by the CLI (``main.py``) and the demo notebook. Output folder
``data/results/<video>_<tracker>_<timestamp>/``:

* ``stats.csv``   – per-frame statistics (people in frame, line counts, zones, FPS)
* ``events.csv``  – line crossings and zone enter/exit events
* ``tracks.csv``  – full track history (frame, id, box, confidence)
* ``heatmap.png``, ``heatmap_overlay.png``, ``summary.json``
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from analysis import Zone
from pipeline import AnalysisPipeline, FrameStats
from tracking import PersonTracker

Point = tuple[float, float]


# --------------------------------------------------------------------------- parsing
def parse_points(text: str, frame_size: tuple[int, int]) -> list[Point]:
    """Parse ``"x1,y1,x2,y2,..."`` (``;`` also allowed) into pixel points.

    Values are fractions of the frame width/height when all of them are <= 1.
    """
    try:
        values = [float(v) for v in re.split(r"[;,\s]+", text.strip()) if v]
    except ValueError as exc:
        raise ValueError(f"invalid coordinates: {text!r}") from exc
    if len(values) < 4 or len(values) % 2:
        raise ValueError(f"expected an even number (>= 4) of coordinates: {text!r}")
    if all(0.0 <= v <= 1.0 for v in values):
        width, height = frame_size
        values = [v * (width if i % 2 == 0 else height) for i, v in enumerate(values)]
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def parse_line(text: str, frame_size: tuple[int, int]) -> list[Point]:
    points = parse_points(text, frame_size)
    if len(points) != 2:
        raise ValueError("a line needs exactly two points: x1,y1,x2,y2")
    return points


def parse_zone(text: str, frame_size: tuple[int, int], index: int) -> Zone:
    """``[name:]x1,y1,x2,y2`` is a rectangle; 3+ points form a polygon."""
    name, _, coords = text.rpartition(":")
    name = name or f"zone{index + 1}"
    points = parse_points(coords, frame_size)
    if len(points) == 2:
        return Zone.rectangle(name, points[0], points[1])
    return Zone(name, points)


# --------------------------------------------------------------------------- running
@dataclass
class RunConfig:
    source: str | int | Path  # video path, camera index or stream URL
    tracker: str = "bytetrack"  # "bytetrack", "botsort" or a tracker YAML path
    model: str = "yolo26n.pt"
    confidence: float = 0.1
    image_size: int = 640
    device: str | None = None
    line: str | None = None
    zones: list[str] = field(default_factory=list)
    trail_length: int = 50
    heatmap_overlay: bool = False
    output_video: Path | None = None
    results_dir: Path = Path("data/results")
    max_frames: int | None = None


@dataclass
class RunResult:
    run_dir: Path
    summary: dict
    stats: pd.DataFrame
    events: pd.DataFrame
    tracks: pd.DataFrame
    fps: float
    frame_size: tuple[int, int]
    first_frame: np.ndarray
    heatmap_image: np.ndarray


def run(
    config: RunConfig,
    *,
    on_frame: Callable[[np.ndarray, FrameStats], bool | None] | None = None,
    on_start: Callable[[AnalysisPipeline], None] | None = None,
    progress: bool = False,
) -> RunResult:
    """Process the whole source.

    ``on_start`` receives the pipeline after the first frame; ``on_frame``
    gets every annotated frame and may return ``False`` to stop early.
    """
    source = int(config.source) if str(config.source).isdecimal() else config.source
    capture = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video source: {config.source}")
    is_file = isinstance(source, (str, Path)) and Path(source).exists()
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else None
    if config.max_frames:
        total = min(total or config.max_frames, config.max_frames)

    tracker = PersonTracker(
        config.model,
        tracker=config.tracker,
        confidence=config.confidence,
        image_size=config.image_size,
        device=config.device,
    )
    name = Path(source).stem if is_file else f"camera{source}"
    run_dir = Path(config.results_dir) / f"{name}_{Path(config.tracker).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline: AnalysisPipeline | None = None
    writer: cv2.VideoWriter | None = None
    stats_rows: list[dict] = []
    event_rows: list[list] = []
    first_frame = last_frame = None
    frame_number = 0
    started = time.perf_counter()
    try:
        while config.max_frames is None or frame_number < config.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if pipeline is None:
                size = (frame.shape[1], frame.shape[0])
                pipeline = AnalysisPipeline(
                    tracker,
                    size,
                    line=parse_line(config.line, size) if config.line else None,
                    zones=[parse_zone(z, size, i) for i, z in enumerate(config.zones)],
                    trail_length=config.trail_length,
                    show_heatmap=config.heatmap_overlay,
                )
                first_frame = frame.copy()
                if on_start is not None:
                    on_start(pipeline)
                if config.output_video:
                    Path(config.output_video).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(config.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
                    )

            timestamp = (frame_number - 1) / fps if is_file else time.perf_counter() - started
            annotated, stats, events = pipeline.process(frame, frame_number, timestamp)
            last_frame = frame
            stats_rows.append(stats.as_row())
            for event in events:
                is_line = hasattr(event, "direction")
                event_rows.append([
                    event.frame_index,
                    round(timestamp, 3),
                    "line" if is_line else "zone",
                    event.line if is_line else event.zone,
                    event.track_id,
                    event.direction if is_line else event.kind,
                ])
            if writer is not None:
                writer.write(annotated)
            if progress and (frame_number % 50 == 0 or frame_number == total):
                print(f"\r{name}: {frame_number}/{total or '?'} klatek", end="", flush=True)
            if on_frame is not None and on_frame(annotated, stats) is False:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if progress:
        print()
    if pipeline is None:
        raise RuntimeError(f"no frames could be read from {config.source}")

    elapsed = time.perf_counter() - started
    stats_df = pd.DataFrame(stats_rows)
    events_df = pd.DataFrame(event_rows, columns=["frame", "time_s", "type", "name", "track_id", "kind"])
    tracks_df = pipeline.tracks_dataframe()
    stats_df.to_csv(run_dir / "stats.csv", index=False)
    events_df.to_csv(run_dir / "events.csv", index=False)
    tracks_df.to_csv(run_dir / "tracks.csv", index=False)
    pipeline.heatmap.save(run_dir / "heatmap.png")
    heatmap_image = pipeline.heatmap.render(last_frame)
    cv2.imwrite(str(run_dir / "heatmap_overlay.png"), heatmap_image)

    summary = {
        "source": str(config.source),
        "tracker": config.tracker,
        "model": config.model,
        "frames": frame_number,
        "video_fps": fps,
        "resolution": f"{pipeline.frame_size[0]}x{pipeline.frame_size[1]}",
        "processing_time_s": round(elapsed, 2),
        "average_fps": round(frame_number / elapsed, 2) if elapsed > 0 else None,
        "unique_people": pipeline.trajectories.unique_ids_seen,
        "max_people_in_frame": int(stats_df["people_in_frame"].max()),
        "mean_people_in_frame": round(float(stats_df["people_in_frame"].mean()), 2),
        "line_in": pipeline.line_counter.in_count if pipeline.line_counter else None,
        "line_out": pipeline.line_counter.out_count if pipeline.line_counter else None,
        "zone_total_entries": pipeline.zone_monitor.total_entries,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return RunResult(
        run_dir=run_dir,
        summary=summary,
        stats=stats_df,
        events=events_df,
        tracks=tracks_df,
        fps=fps,
        frame_size=pipeline.frame_size,
        first_frame=first_frame,
        heatmap_image=heatmap_image,
    )
