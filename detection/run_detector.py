"""Run the person detector on a video file or camera source."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from detection import Detection, YoloPersonDetector


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw person bounding boxes and confidence labels on a copy of a frame."""
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 40), 2)
        caption = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            output,
            caption,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    return output


def _parse_source(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect people in video or camera input")
    parser.add_argument("source", help="Video path, camera index (e.g. 0), or stream URL")
    parser.add_argument("--model", default="yolo26n.pt", help="Model name or weights path")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", help="Inference device, for example cpu or 0")
    parser.add_argument("--output", type=Path, help="Optional output video path")
    parser.add_argument("--show", action="store_true", help="Display the annotated video")
    args = parser.parse_args()

    if not args.output and not args.show:
        parser.error("specify --output, --show, or both")

    detector = YoloPersonDetector(
        args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
    )
    capture = cv2.VideoCapture(_parse_source(args.source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video source: {args.source}")

    writer: cv2.VideoWriter | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated = draw_detections(frame, detector.detect(frame))
            if args.output and writer is None:
                height, width = annotated.shape[:2]
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps) or fps <= 0:
                    fps = 25.0
                args.output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {args.output}")
            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("Person detector (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
