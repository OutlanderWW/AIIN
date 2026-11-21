"""Accumulation and rendering of pedestrian movement heatmaps."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from detection.yolo_detector import Detection


class MotionHeatmap:
    """Accumulate person positions and render them as a colored overlay.

    Positions are stored in source-image pixel coordinates. For detections,
    the bottom-center of each bounding box is used as an estimate of the
    person's point of contact with the ground.
    """

    def __init__(
        self,
        frame_size: tuple[int, int],
        *,
        blur_kernel: int | None = None,
        radius: int | None = None,
        decay: float = 1.0,
        min_visible: float = 0.04,
    ) -> None:
        """Create a heatmap for ``(width, height)`` frames.

        ``blur_kernel`` and ``radius`` default to values proportional to the
        frame size (so 640p and 1080p videos look alike); the kernel is
        rounded up to an odd number as required by OpenCV.

        ``decay`` < 1 makes old activity fade (``decay_step`` multiplies the
        map once per frame), e.g. 0.995 ~ "last few seconds". ``min_visible``
        hides values below this fraction of the maximum so that the video is
        not covered by a blue halo.
        """
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame_size dimensions must be greater than 0")
        scale = min(width, height)
        if blur_kernel is None:
            blur_kernel = max(15, int(scale * 0.06))
        if radius is None:
            radius = max(3, int(scale * 0.008))
        if blur_kernel <= 0:
            raise ValueError("blur_kernel must be greater than 0")
        if radius < 0:
            raise ValueError("radius cannot be negative")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.min_visible = min_visible

        self.width = width
        self.height = height
        self.blur_kernel = blur_kernel if blur_kernel % 2 else blur_kernel + 1
        self.radius = radius
        self._accumulator = np.zeros((height, width), dtype=np.float32)
        diameter = 2 * radius + 1
        self._stamp = np.zeros((diameter, diameter), dtype=np.float32)
        cv2.circle(self._stamp, (radius, radius), radius, 1.0, thickness=-1)
        # ``render`` recomputes the colour map only every ``refresh_every`` calls
        # (the heatmap changes slowly; 1 = every frame, used e.g. by the live monitor).
        self.refresh_every = 1
        self._cache: tuple[np.ndarray, np.ndarray] | None = None
        self._renders_since_refresh = 0

    @property
    def accumulator(self) -> np.ndarray:
        """Return a copy of the unblurred accumulated activity map."""
        return self._accumulator.copy()

    def add_point(self, x: float, y: float, *, weight: float = 1.0) -> None:
        """Add one person's position to the heatmap."""
        if weight <= 0:
            raise ValueError("weight must be greater than 0")
        px, py = int(round(x)), int(round(y))
        if 0 <= px < self.width and 0 <= py < self.height:
            left, top = px - self.radius, py - self.radius
            x1, y1 = max(0, left), max(0, top)
            x2 = min(self.width, px + self.radius + 1)
            y2 = min(self.height, py + self.radius + 1)
            stamp = self._stamp[y1 - top:y2 - top, x1 - left:x2 - left]
            self._accumulator[y1:y2, x1:x2] += stamp * weight

    def add_detections(self, detections: list[Detection]) -> None:
        """Accumulate bottom-center positions from detector output."""
        for detection in detections:
            x1, _, x2, y2 = detection.bbox
            self.add_point((x1 + x2) / 2.0, y2)

    def add_tracks(self, tracks: object) -> None:
        """Accumulate bottom-center points from objects exposing ``bbox``.

        This convenience method accepts any iterable of track-like objects
        whose ``bbox`` is ``(x1, y1, x2, y2)``; track IDs are not required.
        """
        try:
            iterator = iter(tracks)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("tracks must be an iterable of objects with bbox") from exc
        for track in iterator:
            try:
                x1, _, x2, y2 = track.bbox
            except (AttributeError, TypeError, ValueError) as exc:
                raise TypeError("each track must expose bbox=(x1, y1, x2, y2)") from exc
            self.add_point((x1 + x2) / 2.0, y2)

    def decay_step(self) -> None:
        """Fade accumulated activity by ``decay`` (call once per frame)."""
        if self.decay < 1.0:
            self._accumulator *= self.decay

    def colorized(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(BGR colour map, visibility mask)`` of the blurred heatmap.

        The large blur is computed on a 4x smaller copy and scaled back up: the
        result looks the same (blurring removes fine detail anyway) but it is
        many times faster, which matters when the heatmap is shown live.
        """
        factor = 4 if min(self.width, self.height) >= 160 else 1
        if factor > 1:
            small = cv2.resize(self._accumulator, (self.width // factor, self.height // factor),
                               interpolation=cv2.INTER_AREA)
            kernel = max(3, (self.blur_kernel // factor) | 1)
            small = cv2.GaussianBlur(small, (kernel, kernel), sigmaX=0)
            blurred = cv2.resize(small, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        else:
            blurred = cv2.GaussianBlur(self._accumulator, (self.blur_kernel, self.blur_kernel), sigmaX=0)
        peak = float(blurred.max())
        if peak <= 0:
            return np.zeros((self.height, self.width, 3), np.uint8), np.zeros((self.height, self.width), bool)
        normalized = np.clip(blurred / peak * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return colored, blurred > peak * self.min_visible

    def render(self, frame: np.ndarray, *, alpha: float = 0.55) -> np.ndarray:
        """Return a BGR frame with the normalized heatmap blended on top.

        If no positions have been accumulated, a copy of ``frame`` is returned.
        """
        self._validate_frame(frame)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not np.any(self._accumulator):
            return frame.copy()

        if self._cache is None or self._renders_since_refresh >= self.refresh_every - 1:
            self._cache, self._renders_since_refresh = self.colorized(), 0
        else:
            self._renders_since_refresh += 1
        colored, mask = self._cache
        # Hide the near-zero background so untouched areas retain the video.
        output = frame.copy()
        if np.any(mask):
            blended = cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)
            cv2.copyTo(blended, mask.astype(np.uint8), output)  # much faster than output[mask] = ...
        return output

    def save(self, path: str | Path) -> None:
        """Save the current heatmap visualization as a PNG or other image."""
        colored, _ = self.colorized()
        if not cv2.imwrite(str(path), colored):
            raise OSError(f"could not save heatmap to {path}")

    def reset(self) -> None:
        """Clear all accumulated activity."""
        self._accumulator.fill(0)
        self._cache = None

    def _validate_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array (an OpenCV image)")
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"frame must have shape ({self.height}, {self.width}, 3), "
                f"got {frame.shape}"
            )
