"""Matplotlib figures for the demo notebook.

Every function returns a ``matplotlib.figure.Figure``; callers decide whether
to show or save it.
"""

from __future__ import annotations

from typing import Mapping

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from analysis.trajectory_analysis import COMPASS, add_anchor_points
from visualization.drawing import color_for_id

# Fixed categorical order: series 1, 2, 3 ... always get the same colours.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def _image_axes(background: np.ndarray | None, frame_size: tuple[int, int], width_in: float = 10):
    width, height = frame_size
    fig, ax = plt.subplots(figsize=(width_in, width_in * height / width))
    if background is not None:
        gray = cv2.cvtColor(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        ax.imshow(gray, alpha=0.55, extent=(0, width, height, 0))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    return fig, ax


def show_image(image_bgr: np.ndarray, title: str = "", width_in: float = 10) -> Figure:
    height, width = image_bgr.shape[:2]
    fig, ax = plt.subplots(figsize=(width_in, width_in * height / width))
    ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_trajectories(
    tracks: pd.DataFrame,
    frame_size: tuple[int, int],
    background: np.ndarray | None = None,
    *,
    min_points: int = 15,
    title: str = "Trajektorie",
    width_in: float = 10,
) -> Figure:
    """Every trajectory (same colour as its ID in the video); dot = end point."""
    df = add_anchor_points(tracks)
    fig, ax = _image_axes(background, frame_size, width_in)
    shown = 0
    for track_id, t in df.groupby("track_id"):
        if len(t) < min_points:
            continue
        b, g, r = color_for_id(int(track_id))
        color = (r / 255, g / 255, b / 255)
        ax.plot(t["sx"], t["sy"], color=color, linewidth=1.6)
        ax.plot(t["sx"].iloc[-1], t["sy"].iloc[-1], "o", color=color, markersize=4,
                markeredgecolor="white", markeredgewidth=0.8)
        shown += 1
    ax.set_title(f"{title} ({shown} trajektorii)")
    fig.tight_layout()
    return fig


def plot_heatmap(
    heatmap: np.ndarray,
    frame_size: tuple[int, int],
    background: np.ndarray | None = None,
    *,
    cell: int = 16,
    title: str = "Mapa cieplna",
    width_in: float = 10,
) -> Figure:
    """Grid heatmap (from ``occupancy_heatmap`` with the same ``cell``) over the frame."""
    fig, ax = _image_axes(background, frame_size, width_in)
    data = cv2.GaussianBlur(heatmap.astype(np.float32), (0, 0), 1.0)
    masked = np.ma.masked_less_equal(data, data.max() * 0.03 if data.max() > 0 else 0)
    rows, cols = heatmap.shape
    image = ax.imshow(masked, cmap="inferno", alpha=0.75, interpolation="bilinear",
                      extent=(0, cols * cell, rows * cell, 0))
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("osobo-klatki", color=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_people_over_time(series: Mapping[str, pd.DataFrame], *, title: str = "Liczba osób w kadrze") -> Figure:
    """One line per run: ``time_s`` vs ``people_in_frame`` (from ``stats.csv``)."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for i, (label, stats) in enumerate(series.items()):
        ax.plot(stats["time_s"], stats["people_in_frame"], color=SERIES_COLORS[i % len(SERIES_COLORS)], label=label)
    ax.set_xlabel("czas [s]")
    ax.set_ylabel("osoby")
    ax.set_ylim(bottom=0)
    if len(series) > 1:
        ax.legend(loc="upper left", ncols=len(series))
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_direction_rose(summary: pd.DataFrame, *, title: str = "Kierunki ruchu", ax=None) -> Figure:
    """Polar bars: number of trajectories per compass direction (N = up in the image)."""
    counts = summary["direction"].value_counts().reindex(COMPASS, fill_value=0)
    angles = np.radians([0, 315, 270, 225, 180, 135, 90, 45])  # E, SE, S, ... with y pointing down
    if ax is None:
        fig = plt.figure(figsize=(4.4, 4.4))
        ax = fig.add_subplot(projection="polar")
    ax.bar(angles, counts.to_numpy(), width=np.radians(38), color=SERIES_COLORS[0],
           edgecolor=SURFACE, linewidth=2)
    ax.set_thetalim(0, 2 * np.pi)
    ax.set_xticks(angles)
    ax.set_xticklabels(COMPASS, color=TEXT_SECONDARY)
    ax.set_yticklabels([])
    ax.grid(color=GRID)
    stationary = int((summary["direction"] == "stationary").sum())
    ax.set_title(f"{title}\n(stojące: {stationary})", pad=14)
    return ax.figure


def plot_speed_histogram(summary: pd.DataFrame, *, title: str = "Prędkość", ax=None) -> Figure:
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(summary["speed_heights_s"].dropna(), bins=15, color=SERIES_COLORS[0],
            edgecolor=SURFACE, linewidth=2)
    ax.set_xlabel("prędkość [wysokości sylwetki / s]")
    ax.set_ylabel("trajektorie")
    ax.set_title(title)
    return ax.figure
