"""Person detection for OpenCV frames using an Ultralytics YOLO model.

The detector deliberately returns plain Python values so tracking and analysis
modules do not need to depend on Ultralytics result objects.

Example::

    import cv2
    from detection import YoloPersonDetector

    detector = YoloPersonDetector()
    capture = cv2.VideoCapture("video.mp4")
    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame)
            tracker_input = detector.to_tracker_input(detections)
    finally:
        capture.release()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

ImageFrame: TypeAlias = np.ndarray
Box: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected person in image pixel coordinates (x1, y1, x2, y2)."""

    bbox: Box
    confidence: float
    class_id: int
    label: str = "person"

    def to_tracker_row(self) -> tuple[float, float, float, float, float, int]:
        """Return ``x1, y1, x2, y2, confidence, class_id`` for a tracker."""
        return (*self.bbox, self.confidence, self.class_id)


class YoloPersonDetector:
    """Run YOLO inference on individual BGR image frames.

    Args:
        model_path: Ultralytics model name or local model/checkpoint path.
        confidence: Minimum confidence threshold in the inclusive range [0, 1].
        image_size: Inference image size in pixels.
        device: Optional Ultralytics device selector, e.g. ``"cpu"`` or ``"0"``.
        person_class_id: Class index for ``person`` in the selected model.
    """

    def __init__(
        self,
        model_path: str | Path = "yolo26n.pt",
        *,
        confidence: float = 0.25,
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
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.person_class_id = person_class_id
        self._model = YOLO(self.model_path)

    def detect(self, frame: ImageFrame) -> list[Detection]:
        """Detect people in one OpenCV BGR frame.

        Returns an empty list when no person is detected. Bounding boxes use
        the source frame's pixel coordinates and are not rounded.
        """
        self._validate_frame(frame)
        kwargs: dict[str, object] = {
            "source": frame,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "classes": [self.person_class_id],
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self._model.predict(**kwargs)
        if not results:
            return []

        return self._convert_result(results[0])

    def detect_batch(self, frames: list[ImageFrame]) -> list[list[Detection]]:
        """Detect people in a batch of frames, preserving input order."""
        if not frames:
            return []
        for frame in frames:
            self._validate_frame(frame)

        kwargs: dict[str, object] = {
            "source": frames,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "classes": [self.person_class_id],
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        results = self._model.predict(**kwargs)
        return [self._convert_result(result) for result in results]

    @staticmethod
    def to_tracker_input(
        detections: list[Detection],
    ) -> np.ndarray:
        """Convert detections to an ``(N, 6)`` float array.

        The columns are ``x1, y1, x2, y2, confidence, class_id``. Empty
        detections produce an array with shape ``(0, 6)``.
        """
        if not detections:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(
            [detection.to_tracker_row() for detection in detections],
            dtype=np.float32,
        )

    def _convert_result(self, result: object) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = self._as_numpy(boxes.xyxy)
        confidences = self._as_numpy(boxes.conf)
        class_ids = self._as_numpy(boxes.cls).astype(int)
        return [
            Detection(
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(score),
                class_id=int(class_id),
            )
            for box, score, class_id in zip(xyxy, confidences, class_ids)
            if int(class_id) == self.person_class_id
        ]

    @staticmethod
    def _as_numpy(value: object) -> np.ndarray:
        """Convert torch tensors or array-like values to NumPy."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _validate_frame(frame: ImageFrame) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array (an OpenCV image)")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (height, width, 3) in BGR format")
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("frame cannot be empty")
