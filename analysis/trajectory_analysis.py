"""Offline analysis of complete trajectories (pandas).

Input is a tracks table with one row per (frame, track) and columns
``frame, track_id, x1, y1, x2, y2`` (``tracks.csv`` saved by the runner).
Positions are bottom-centre "foot points".

Distances are in image pixels; without camera calibration they are not
metres and depend on perspective, so speed is also given in "body heights per
second", which partially compensates for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPASS = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]  # image y axis points down


def add_anchor_points(tracks: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
    """Add foot point ``ax, ay`` and its smoothed version ``sx, sy``.

    A centred rolling mean removes box jitter, which would otherwise inflate
    path length and speed.
    """
    df = tracks.sort_values(["track_id", "frame"]).copy()
    df["ax"] = (df["x1"] + df["x2"]) / 2.0
    df["ay"] = df["y2"]
    grouped = df.groupby("track_id")
    smooth = lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean()  # noqa: E731
    df["sx"] = grouped["ax"].transform(smooth)
    df["sy"] = grouped["ay"].transform(smooth)
    return df.reset_index(drop=True)


def summarize_tracks(tracks: pd.DataFrame, fps: float, *, min_frames: int = 15) -> pd.DataFrame:
    """One row per trajectory: duration, path length, straightness, speed, direction.

    Tracks shorter than ``min_frames`` (fragments after an ID switch, false
    detections) are skipped.
    """
    df = add_anchor_points(tracks)
    g = df.groupby("track_id")
    df["step"] = np.hypot(g["sx"].diff(), g["sy"].diff()).fillna(0.0)
    df["height"] = df["y2"] - df["y1"]
    g = df.groupby("track_id")
    summary = pd.DataFrame({
        "points": g.size(),
        "first_frame": g["frame"].min(),
        "last_frame": g["frame"].max(),
        "path_length_px": g["step"].sum(),
        "dx": g["sx"].last() - g["sx"].first(),
        "dy": g["sy"].last() - g["sy"].first(),
        "box_height_px": g["height"].median(),
    })
    summary = summary[summary["points"] >= min_frames].copy()
    summary["duration_s"] = (summary["last_frame"] - summary["first_frame"] + 1) / fps
    summary["displacement_px"] = np.hypot(summary["dx"], summary["dy"])
    path = summary["path_length_px"].where(summary["path_length_px"] > 0)
    summary["straightness"] = (summary["displacement_px"] / path).clip(upper=1.0)
    summary["speed_px_s"] = summary["path_length_px"] / summary["duration_s"]
    summary["speed_heights_s"] = summary["speed_px_s"] / summary["box_height_px"]
    angle = np.degrees(np.arctan2(summary["dy"], summary["dx"])) % 360  # 0 = right, 90 = down
    summary["direction"] = np.array(COMPASS)[((angle + 22.5) // 45).astype(int) % 8]
    summary.loc[summary["displacement_px"] < 0.5 * summary["box_height_px"], "direction"] = "stationary"
    return summary.drop(columns=["dx", "dy"]).reset_index()


def occupancy_heatmap(tracks: pd.DataFrame, frame_size: tuple[int, int], cell: int = 16) -> np.ndarray:
    """Grid heatmap: how many person-frames were spent in each ``cell`` x ``cell`` square."""
    df = add_anchor_points(tracks)
    width, height = frame_size
    rows, cols = int(np.ceil(height / cell)), int(np.ceil(width / cell))
    hist, _, _ = np.histogram2d(df["ay"], df["ax"], bins=(rows, cols),
                                range=[[0, rows * cell], [0, cols * cell]])
    return hist
