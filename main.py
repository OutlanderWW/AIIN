"""Command-line entry point: detect, track and analyse people in a video.

Examples::

    python main.py data/videos/people-walking.mp4 --show
    python main.py 0 --show --heatmap
    python main.py video.mp4 --line 0.1,0.6,0.9,0.6 --zone door:0.6,0.3,0.9,0.9 --output out.mp4

Coordinates for ``--line`` and ``--zone`` are pixels, or fractions of the
frame width/height when every value is <= 1.
Keys in the preview window: ``q`` quit, ``h`` toggle heatmap overlay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from runner import RunConfig, run
from tracking import TRACKER_CONFIGS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect, track and analyse people in video")
    parser.add_argument("source", help="Video path, camera index (e.g. 0) or stream URL")
    parser.add_argument("--model", default="yolo26n.pt", help="Model name or weights path")
    parser.add_argument(
        "--tracker", default="bytetrack", help=f"{' / '.join(TRACKER_CONFIGS)} or a tracker YAML"
    )
    parser.add_argument("--confidence", type=float, default=0.1, help="Detector threshold")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", help="Inference device, e.g. cpu or 0")
    parser.add_argument("--line", help="Counting line x1,y1,x2,y2")
    parser.add_argument(
        "--zone", action="append", default=[], help="[name:]x1,y1,x2,y2 or polygon; repeatable"
    )
    parser.add_argument("--trail-length", type=int, default=50)
    parser.add_argument("--heatmap", action="store_true", help="Overlay the heatmap on video")
    parser.add_argument("--show", action="store_true", help="Display the annotated video")
    parser.add_argument("--output", type=Path, help="Optional annotated output video path")
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = RunConfig(
        source=args.source,
        tracker=args.tracker,
        model=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
        line=args.line,
        zones=args.zone,
        trail_length=args.trail_length,
        heatmap_overlay=args.heatmap,
        output_video=args.output,
        results_dir=args.results_dir,
        max_frames=args.max_frames,
    )

    on_frame = on_start = None
    if args.show:
        state: dict = {}

        def on_start(pipeline) -> None:  # noqa: ANN001
            state["pipeline"] = pipeline

        def on_frame(annotated, stats) -> bool:  # noqa: ANN001
            cv2.imshow("Person tracking (q: quit, h: heatmap)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("h"):
                state["pipeline"].show_heatmap = not state["pipeline"].show_heatmap
            return key != ord("q")

    try:
        result = run(config, on_frame=on_frame, on_start=on_start, progress=not args.show)
    finally:
        if args.show:
            cv2.destroyAllWindows()

    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
    print(f"Results saved to {result.run_dir}")


if __name__ == "__main__":
    main()
