"""Public test videos (static cameras with pedestrians) and their download.

    python videos.py          # downloads all test videos to data/videos/

Each entry also defines how the video is used as a "camera": label, counting
line, zone and ``zone_limit`` - the number of people in the zone that raises
an alarm in ``monitor.py``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Iterable

VIDEOS_DIR = Path("data/videos")

TEST_VIDEOS = {
    "people-walking.mp4": {
        "url": "https://media.roboflow.com/supervision/video-examples/people-walking.mp4",
        "description": "hala, widok z góry, ~30 osób, 1920x1080, 25 FPS, 14 s (Roboflow Supervision)",
        "label": "HALA",
        "line": "0,0.5,1,0.5",
        "zones": ["srodek:0.35,0.3,0.65,0.8"],
        "zone_limit": 7,
        "image_size": 640,
    },
    "market-square.mp4": {
        "url": "https://media.roboflow.com/supervision/video-examples/market-square.mp4",
        "description": "plac z fontanną, widok z góry, tłum, 2160x3840 (pion), 60 FPS, 8 s (Roboflow Supervision)",
        "label": "PLAC",
        "line": "0,0.6,1,0.6",
        "zones": ["fontanna:0.3,0.25,0.75,0.45"],
        "zone_limit": 5,
        # People are tiny in this 4K top view: at the default 640 px input
        # YOLO finds ~3 of ~60 people, at 1280 px most of them.
        "image_size": 1280,
    },
    "mot17-09.mp4": {
        "url": "https://motchallenge.net/sequenceVideos/MOT17-09-SDP-raw.mp4",
        "description": "deptak przy wejściu do sklepów, widok z boku, 960x540, 30 FPS, 18 s (MOTChallenge MOT17-09)",
        "label": "DEPTAK",
        "line": "0.55,0.35,0.55,1",
        # Floor inside the Vodafone shop - the whole shop front from the left edge
        # of the frame to the pillar at x = 0.56, up to the door threshold. Zones work
        # on the people's foot points: passers-by on the pavement stand below the
        # threshold line; above y = 0.53 stands a balloon display that YOLO sometimes
        # takes for a person.
        "zones": ["sklep:0,0.53;0.56,0.53;0.56,0.665;0.13,0.68;0,0.705"],
        "zone_limit": 4,
        "image_size": 640,
    },
    "mot17-04.mp4": {
        "url": "https://motchallenge.net/sequenceVideos/MOT17-04-SDP-raw.mp4",
        "description": "ulica handlowa nocą, widok z wysoka, tłum, 960x540, 30 FPS, 35 s (MOTChallenge MOT17-04)",
        "label": "ULICA",
        "line": "0,0.55,1,0.55",
        "zones": ["chodnik:0,0.3,0.32,1"],
        "zone_limit": 8,
        # Many small, distant people: 1280 px input finds ~85% of them vs ~36% at 640 px.
        "image_size": 1280,
    },
    "mot20-01.mp4": {
        "url": "https://motchallenge.net/sequenceVideos/MOT20-01-raw.mp4",
        "description": "hala dworca, widok z góry, tłum, 960x540, 25 FPS, 17 s (MOTChallenge MOT20-01)",
        "label": "DWORZEC",
        "line": "0.2,0.75,0.95,0.75",
        "zones": ["hol:0.35,0.45,0.7,0.85"],
        "zone_limit": 10,
        # Crowd far from the camera: 1280 px finds ~73% of people vs ~39% at 640 px.
        "image_size": 1280,
    },
}

# The demo notebook uses only these three (different perspectives, short runtime).
NOTEBOOK_VIDEOS = ["people-walking.mp4", "market-square.mp4", "mot17-09.mp4"]


def download_test_videos(
    names: Iterable[str] | None = None, directory: str | Path = VIDEOS_DIR
) -> dict[str, Path]:
    """Download missing test videos (all, or only ``names``); return ``name -> local path``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in names or TEST_VIDEOS:
        path = directory / name
        if not path.exists():
            print(f"Pobieranie {name} ...")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(TEST_VIDEOS[name]["url"], tmp)
            tmp.replace(path)  # a broken download never looks like a finished file
        paths[name] = path
    return paths


if __name__ == "__main__":
    for name, path in download_test_videos().items():
        print(path, "-", TEST_VIDEOS[name]["description"])
