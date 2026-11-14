"""Public test videos (static cameras with pedestrians) and their download.

    python videos.py          # downloads all test videos to data/videos/
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

VIDEOS_DIR = Path("data/videos")

TEST_VIDEOS = {
    "people-walking.mp4": {
        "url": "https://media.roboflow.com/supervision/video-examples/people-walking.mp4",
        "description": "hala, widok z góry, ~30 osób, 1920x1080, 25 FPS, 14 s (Roboflow Supervision)",
        "line": "0,0.5,1,0.5",
        "zones": ["srodek:0.35,0.3,0.65,0.8"],
        "image_size": 640,
    },
    "market-square.mp4": {
        "url": "https://media.roboflow.com/supervision/video-examples/market-square.mp4",
        "description": "plac z fontanną, widok z góry, tłum, 2160x3840 (pion), 60 FPS, 8 s (Roboflow Supervision)",
        "line": "0,0.6,1,0.6",
        "zones": ["fontanna:0.3,0.25,0.75,0.45"],
        # People are tiny in this 4K top view: at the default 640 px input
        # YOLO finds ~3 of ~60 people, at 1280 px most of them.
        "image_size": 1280,
    },
    "mot17-09.mp4": {
        "url": "https://motchallenge.net/sequenceVideos/MOT17-09-SDP-raw.mp4",
        "description": "deptak przy wejściu do sklepów, widok z boku, 960x540, 30 FPS, 18 s (MOTChallenge MOT17-09)",
        "line": "0.55,0.35,0.55,1",
        "zones": ["wejscie:0.05,0.2,0.35,0.8"],
        "image_size": 640,
    },
}


def download_test_videos(directory: str | Path = VIDEOS_DIR) -> dict[str, Path]:
    """Download missing test videos; return ``name -> local path``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, info in TEST_VIDEOS.items():
        path = directory / name
        if not path.exists():
            print(f"Pobieranie {name} ...")
            urllib.request.urlretrieve(info["url"], path)
        paths[name] = path
    return paths


if __name__ == "__main__":
    for name, path in download_test_videos().items():
        print(path, "-", TEST_VIDEOS[name]["description"])
