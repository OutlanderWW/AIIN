"""Multi-person tracking (ByteTrack / BoT-SORT) on top of Ultralytics YOLO.

Ultralytics ships ByteTrack and BoT-SORT implementations. ``model.track`` runs
YOLO detection and then the selected tracker (Kalman filter prediction + IoU
matching) on every frame. ``persist=True`` keeps tracker state between calls,
so frames must be fed in order from a single video source.

Example::

    tracker = PersonTracker(tracker="bytetrack")
    for frame in frames:
        for track in tracker.update(frame):
            print(track.track_id, track.anchor)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeAlias

from detection.yolo_detector import Box, ImageFrame, YoloPersonDetector

Point: TypeAlias = tuple[float, float]

TRACKER_CONFIGS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


@dataclass(frozen=True, slots=True)
class Track:
    """One tracked person in image pixel coordinates (x1, y1, x2, y2)."""

    track_id: int
    bbox: Box
    confidence: float
    class_id: int = 0

    @property
    def anchor(self) -> Point:
        """Bottom-center of the box: approximate point of contact with the floor."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


def tracks_to_points(tracks: Iterable[Track]) -> dict[int, Point]:
    """Map ``track_id -> anchor point``; the input format of analysis modules."""
    return {track.track_id: track.anchor for track in tracks}


class PersonTracker:
    """Detect and track people frame by frame.

    Args:
        model_path: Ultralytics model name or local weights path.
        tracker: ``"bytetrack"``, ``"botsort"`` or a path to a tracker YAML.
        confidence: Detector threshold applied *before* tracking. It is lower
            than the detector default on purpose: ByteTrack's second
            association stage uses low-confidence boxes (``track_low_thresh``
            = 0.1 in ``bytetrack.yaml``), so filtering them out earlier would
            disable the main idea of the algorithm.
        image_size: Inference image size in pixels.
        device: Optional Ultralytics device selector, e.g. ``"cpu"`` or ``"0"``.
        person_class_id: Class index for ``person`` in the selected model.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo26n.pt",
        *,
        tracker: str = "bytetrack",
        confidence: float = 0.1,
        image_size: int = 640,
        device: str | None = None,
        person_class_id: int = 0,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be greater than 0")
        if person_class_id < 0:
            raise ValueError("person_class_id cannot be negative")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required. Install project dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        self.model_path = str(model_path)
        self.tracker_config = TRACKER_CONFIGS.get(tracker, tracker)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.person_class_id = person_class_id
        self._model = YOLO(self.model_path)

    def update(self, frame: ImageFrame) -> list[Track]:
        """Process the next frame and return the currently tracked people.

        Only tracks matched in this frame are returned; a person that is
        briefly occluded disappears from the list but keeps its ID when the
        tracker re-associates it (within the tracker's ``track_buffer``).
        """
        YoloPersonDetector._validate_frame(frame)
        kwargs: dict[str, object] = {
            "source": frame,
            "persist": True,
            "tracker": self.tracker_config,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "classes": [self.person_class_id],
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self._model.track(**kwargs)
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        as_numpy = YoloPersonDetector._as_numpy
        xyxy = as_numpy(boxes.xyxy)
        ids = as_numpy(boxes.id).astype(int)
        confidences = as_numpy(boxes.conf)
        class_ids = as_numpy(boxes.cls).astype(int)
        return [
            Track(
                track_id=int(track_id),
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(score),
                class_id=int(class_id),
            )
            for box, track_id, score, class_id in zip(xyxy, ids, confidences, class_ids)
            if int(class_id) == self.person_class_id
        ]

    def reset(self) -> None:
        """Forget all tracks, e.g. before processing a different video."""
        predictor = getattr(self._model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
