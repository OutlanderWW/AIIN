"""Rendering of tracking and analysis results onto video frames."""

from .drawing import (
    color_for_id,
    draw_line_counter,
    draw_stats_panel,
    draw_tracks,
    draw_trajectories,
    draw_zones,
)

__all__ = [
    "color_for_id",
    "draw_line_counter",
    "draw_stats_panel",
    "draw_tracks",
    "draw_trajectories",
    "draw_zones",
]
