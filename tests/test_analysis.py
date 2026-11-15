import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from analysis import MotionHeatmap
from runner import RunConfig, run
from tracking import Track


class MotionHeatmapTests(unittest.TestCase):
    def test_repeated_observations_are_more_intense(self):
        heatmap = MotionHeatmap((32, 32), radius=2, blur_kernel=3)
        for _ in range(20):
            heatmap.add_point(6, 6)
        heatmap.add_point(24, 24)

        accumulator = heatmap.accumulator
        self.assertEqual(accumulator[6, 6], 20)
        self.assertEqual(accumulator[24, 24], 1)
        _, visible = heatmap.colorized()
        self.assertTrue(visible[6, 6])

    def test_weight_decay_and_clipping_at_frame_boundary(self):
        heatmap = MotionHeatmap((8, 8), radius=2, decay=0.5)
        heatmap.add_point(0, 0, weight=2)
        heatmap.decay_step()
        heatmap.add_point(1, 0, weight=3)

        accumulator = heatmap.accumulator
        self.assertEqual(accumulator[0, 0], 4)
        self.assertEqual(accumulator[0, 3], 3)
        self.assertEqual(accumulator[-1, -1], 0)


class RunnerTests(unittest.TestCase):
    def test_tracks_line_zones_and_exports(self):
        def track_at(y):
            return [Track(1, (26, y - 14, 38, y), 0.9)]

        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.mp4"
            output = root / "output.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
            try:
                self.assertTrue(writer.isOpened())
                for _ in range(5):
                    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            finally:
                writer.release()

            config = RunConfig(
                source=source, line="0,0.5,1,0.5", zones=["door:22,33,48,55"],
                heatmap_overlay=True, output_video=output, results_dir=root / "results",
            )
            with patch("runner.PersonTracker") as tracker:
                tracker.return_value.update.side_effect = [track_at(y) for y in (20, 44, 20, 44, 20)]
                result = run(config, on_frame=lambda _, stats: stats.frame_index < 4)

            self.assertEqual(result.stats["zone_door"].tolist(), [0, 1, 0, 1])
            self.assertEqual(result.events["kind"].tolist(), ["in", "enter", "out", "exit", "enter"])
            self.assertEqual(result.tracks["track_id"].tolist(), [1] * 4)
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(sorted(p.name for p in result.run_dir.iterdir()), [
                "events.csv", "heatmap.png", "heatmap_overlay.png", "stats.csv", "summary.json", "tracks.csv",
            ])
            summary = json.loads((result.run_dir / "summary.json").read_text())
            self.assertEqual((summary["frames"], summary["line_in"], summary["line_out"]), (4, 1, 1))

    def test_rtsp_connection_failure_releases_capture_and_hides_credentials(self):
        source = "rtsp://user:password@camera.invalid/stream"
        with patch("runner.cv2.VideoCapture") as capture_class:
            capture = capture_class.return_value
            capture.isOpened.return_value = False
            with self.assertRaisesRegex(RuntimeError, "^could not open video source: RTSP$"):
                run(RunConfig(source=source))

            capture.open.assert_called_once_with(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ])
            capture.release.assert_called_once()
