"""Security-room style live view: several "cameras" analysed in real time.

    python monitor.py                              # the test videos as cameras
    python monitor.py --video a.mp4 --video 0      # own videos and/or a webcam

Keys: 1-9 or click on the map - choose camera, Tab - next camera,
H - heatmap, D - YOLO input size (accuracy vs speed), Q / Esc - quit.

How it works
------------
* Every camera has a reader thread that decodes its video at the video's own
  pace (the clock keeps running like a live camera) and keeps only the newest
  frame.
* An analysis thread runs YOLO + ByteTrack on the newest frame of the camera on
  screen. If it is slower than the video, frames in between are skipped, so
  the picture never lags behind "reality" - like real CCTV analytics.
* A background watcher checks the other cameras every few seconds with plain
  YOLO detection (no tracking) and raises an alarm when too many people are in
  a camera's zone - the camera then blinks red on the map.
* The interface is drawn by the main thread in parallel (30 FPS). Boxes, IDs,
  trajectories, zones and the line are drawn at screen resolution with proper
  fonts (PIL), on top of the clean camera frame.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analysis import Zone
from detection import YoloPersonDetector
from pipeline import AnalysisPipeline, FrameStats
from runner import parse_line, parse_zone
from tracking import PersonTracker, Track
from visualization.drawing import color_for_id

# --------------------------------------------------------------------------- layout
WINDOW = "MONITORING"
CANVAS_W, CANVAS_H = 1280, 720
PAD = 16
HEADER_H = 52
VIEW_X, VIEW_Y = PAD, HEADER_H
VIEW_W, VIEW_H = 928, 522                     # main camera view (16:9)
SIDE_X = VIEW_X + VIEW_W + PAD                # right column
SIDE_W = CANVAS_W - SIDE_X - PAD
LOG_Y = VIEW_Y + VIEW_H + 12                  # event log under the view
MAP_CARD = (SIDE_X, VIEW_Y, SIDE_W, 262)
STATS_CARD = (SIDE_X, VIEW_Y + 262 + 12, SIDE_W, CANVAS_H - PAD - (VIEW_Y + 274))
MAP_AREA = (MAP_CARD[0] + 12, MAP_CARD[1] + 36, MAP_CARD[2] - 24, MAP_CARD[3] - 48)
UI_FPS = 30
CHROME_FPS = 10                               # panels/map/log are redrawn this often
MAX_ANALYSIS_SIDE = 1280                      # frames are downscaled before analysis
IMAGE_SIZES = (640, 960, 1280)                # choices for the D key
ROLL_END, STATIC_END, SETTLE_END = 0.15, 0.42, 0.62   # "signal lost" transition
ALARM_CLEAR_S = 1.5                           # alarm ends after this long below the limit
WATCH_INTERVAL_S = 1.0                        # pause between background checks
WATCH_MAX_IMAGE_SIZE = 960                    # background checks need only an approximate count
HISTORY_S = 60                                # people-count sparkline length


def bgr(hex_color: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> BGR tuple (the canvas is a BGR image for both OpenCV and PIL)."""
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


BG = bgr("#0b0e13")
CARD = bgr("#141a22")
CARD_EDGE = bgr("#232c38")
TILE = bgr("#1b222c")
TEXT = bgr("#e6edf3")
MUTED = bgr("#8b97a5")
DIM = bgr("#4a5563")
ACCENT = bgr("#35d0ba")
OK = bgr("#3dd68c")
WARN = bgr("#f5b83d")
ALARM = bgr("#ff4d4f")
LINE_COLOR = bgr("#ffd24d")
ZONE_COLOR = bgr("#f59e3d")
BLACK = (0, 0, 0)

# Same physical keys in the Russian layout (Unicode code point and cp1251 byte).
KEY_ALIASES = {
    0x432: "d", 0x412: "d", 0xE2: "d", 0xC2: "d",   # в В
    0x440: "h", 0x420: "h", 0xF0: "h", 0xD0: "h",   # р Р
    0x439: "q", 0x419: "q", 0xE9: "q", 0xC9: "q",   # й Й
}
TOAST_S = 2.0                                 # how long a short notice stays on the view

# Floor plan of our fictional site: room rectangles (x, y, w, h) as fractions of
# the map area, in reading order (camera 1 = first room, camera 2 = second ...).
SITE_ROOMS = [
    (0.00, 0.00, 0.44, 0.42),
    (0.48, 0.00, 0.52, 0.20),
    (0.48, 0.24, 0.52, 0.20),
    (0.00, 0.48, 0.48, 0.52),
    (0.52, 0.52, 0.48, 0.48),
]
SITE_LINKS = [(0, 1), (1, 2), (0, 3), (2, 4), (3, 4)]  # corridors between rooms


# --------------------------------------------------------------------------- fonts
def _first_file(candidates: list[Path]) -> str | None:
    return next((str(p) for p in candidates if p.is_file()), None)


_WIN_FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
try:
    import matplotlib

    _MPL_FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
except ImportError:  # pragma: no cover
    _MPL_FONTS = Path(".")
_FONT_FILES = {
    "ui": _first_file([_WIN_FONTS / "bahnschrift.ttf", _WIN_FONTS / "segoeui.ttf", _MPL_FONTS / "DejaVuSans.ttf"]),
    "bold": _first_file([_WIN_FONTS / "bahnschrift.ttf", _WIN_FONTS / "segoeuib.ttf", _MPL_FONTS / "DejaVuSans-Bold.ttf"]),
    "mono": _first_file([_WIN_FONTS / "consola.ttf", _MPL_FONTS / "DejaVuSansMono.ttf"]),
}


@lru_cache(maxsize=None)
def font(size: int, kind: str = "ui") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _FONT_FILES.get(kind)
    if path is None:
        return ImageFont.load_default()
    f = ImageFont.truetype(path, size)
    if kind == "bold" and "bahnschrift" in path.lower():
        try:
            f.set_variation_by_name("SemiBold")
        except Exception:  # noqa: BLE001 - variable fonts are optional
            pass
    return f


def text_width(text: str, f) -> int:
    return int(f.getlength(text))


# --------------------------------------------------------------------------- camera
@dataclass
class CameraConfig:
    source: str | int
    label: str
    line: str | None = None
    zones: tuple[str, ...] = ()
    image_size: int = 640
    zone_limit: int | None = None             # people in the first zone that raise an alarm


@dataclass
class Snapshot:
    """What the analysis thread hands over to the interface."""

    camera: int
    image: np.ndarray                         # clean frame (or heatmap overlay), analysis resolution
    tracks: list[Track]
    paths: dict[int, list[tuple[float, float]]]
    created: float = field(default_factory=time.perf_counter)


class CameraFeed:
    """A looping video (or a webcam) that behaves like a live camera."""

    def __init__(self, config: CameraConfig, clock_origin: float, tracker_name: str, model: str) -> None:
        self.config = config
        self.clock_origin = clock_origin
        self.tracker_name = tracker_name
        self.model = model
        self.is_live = isinstance(config.source, int)
        self.capture = cv2.VideoCapture(config.source if self.is_live else str(config.source))
        if not self.capture.isOpened():
            raise RuntimeError(f"nie można otworzyć źródła: {config.source}")
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if np.isfinite(fps) and fps > 0 else 25.0
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else 0
        self._probe: cv2.VideoCapture | None = None  # second reader for background checks
        self.pipeline: AnalysisPipeline | None = None
        self.last_stats: FrameStats | None = None
        # Shared by the analysis thread and the background watcher.
        self.people: int | None = None
        self.zone_count: int | None = None
        self.alarm = False
        self.alarm_clear_since: float | None = None   # alarm hysteresis
        self.heatmap_visible = False   # the heatmap restarts from zero whenever it becomes visible
        self.zone_objects: list[Zone] | None = None
        self.line_points: list[tuple[float, float]] | None = None
        self.history: deque[tuple[float, int]] = deque()  # (time, people) for the sparkline
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def zone_name(self) -> str | None:
        if not self.config.zones:
            return None
        return self.config.zones[0].rpartition(":")[0] or "zone1"

    def live_position(self) -> int:
        """Frame index a live camera would show right now."""
        return int((time.perf_counter() - self.clock_origin) * self.fps) % max(self.frame_count, 1)

    @staticmethod
    def _downscale(frame: np.ndarray) -> np.ndarray:
        scale = MAX_ANALYSIS_SIDE / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return frame

    def record_people(self, count: int) -> None:
        now = time.perf_counter()
        self.people = count
        if not self.history or now - self.history[-1][0] >= 0.5:
            self.history.append((now, count))
        while self.history and now - self.history[0][0] > HISTORY_S:
            self.history.popleft()

    # -- reader thread -------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        if not self.is_live and self.frame_count > 0:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.live_position())
        if self.pipeline is not None:
            self.pipeline.reset_fps()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.heatmap_visible = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._running

    def _read_loop(self) -> None:
        period = 1.0 / self.fps
        next_time: float | None = None  # pace starts after the first (slow, post-seek) frame
        while self._running:
            ok, frame = self.capture.read()
            if not ok:
                if self.is_live:
                    time.sleep(0.01)
                else:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the recording
                continue
            frame = self._downscale(frame)
            with self._lock:
                self._frame = frame
                self._frame_id += 1
            if not self.is_live:
                next_time = (next_time or time.perf_counter()) + period
                delay = next_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.5:  # decoding cannot keep up - do not accumulate lag
                    next_time = time.perf_counter()

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._frame_id, self._frame

    def probe(self) -> np.ndarray | None:
        """Current frame of a camera that is not on screen (background check)."""
        if self.is_live:
            return None
        if self._probe is None:
            self._probe = cv2.VideoCapture(str(self.config.source))
        self._probe.set(cv2.CAP_PROP_POS_FRAMES, self.live_position())
        ok, frame = self._probe.read()
        return self._downscale(frame) if ok else None

    def geometry_for(self, frame: np.ndarray) -> list[Zone]:
        """Zones (and the counting line) in pixels of the analysed frame."""
        if self.zone_objects is None:
            size = (frame.shape[1], frame.shape[0])
            self.zone_objects = [parse_zone(z, size, i) for i, z in enumerate(self.config.zones)]
            self.line_points = parse_line(self.config.line, size) if self.config.line else None
        return self.zone_objects

    def set_image_size(self, size: int) -> None:
        self.config.image_size = size
        if self.pipeline is not None:
            self.pipeline.tracker.image_size = size

    # -- analysis ------------------------------------------------------------
    def analyse(self, frame: np.ndarray, frame_id: int, heatmap: bool) -> tuple[Snapshot, list]:
        if self.pipeline is None:
            size = (frame.shape[1], frame.shape[0])
            zones = self.geometry_for(frame)
            tracker = PersonTracker(self.model, tracker=self.tracker_name, image_size=self.config.image_size)
            self.pipeline = AnalysisPipeline(
                tracker, size, line=self.line_points, zones=zones, trail_length=40,
                record_tracks=False, show_stats_panel=False, draw_overlays=False,
            )
        self.pipeline.heatmap.refresh_every = 3  # colour map every 3rd frame, blending every frame
        if heatmap and not self.heatmap_visible:
            # Turned on (or camera switched to): build the heatmap live from now on
            # instead of showing everything accumulated since the camera started.
            self.pipeline.heatmap.reset()
        self.heatmap_visible = heatmap
        self.pipeline.show_heatmap = heatmap
        image, stats, events = self.pipeline.process(frame, frame_id, frame_id / self.fps)
        self.last_stats = stats
        self.record_people(stats.people_in_frame)
        if self.zone_objects:
            # People visible right now whose foot point is in the zone. (The zone
            # monitor's own count also keeps people lost for a few frames, which
            # double-counts a person whose track got a new ID.)
            zone = self.zone_objects[0]
            self.zone_count = sum(zone.contains(t.anchor) for t in self.pipeline.last_tracks)
        snapshot = Snapshot(-1, image, self.pipeline.last_tracks, self.pipeline.trajectories.paths())
        return snapshot, events

    def release(self) -> None:
        self.stop()
        self.capture.release()
        if self._probe is not None:
            self._probe.release()


# --------------------------------------------------------------------------- image effects
def static_noise(width: int, height: int) -> np.ndarray:
    """A frame of analogue-TV static with a few bright horizontal bands."""
    noise = np.random.randint(0, 170, (height // 2, width // 2), dtype=np.uint8)
    noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_NEAREST)
    for _ in range(3):
        y, band = np.random.randint(0, height - 20), np.random.randint(4, 20)
        noise[y:y + band] = np.clip(noise[y:y + band].astype(np.int16) + 70, 0, 255).astype(np.uint8)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)


def tear(image: np.ndarray, bands: int = 6, max_shift: int = 60) -> np.ndarray:
    """Shift random horizontal bands sideways (a torn analogue signal)."""
    out = image.copy()
    height = image.shape[0]
    for _ in range(bands):
        y, h = np.random.randint(0, height - 8), np.random.randint(4, 40)
        out[y:y + h] = np.roll(out[y:y + h], np.random.randint(-max_shift, max_shift), axis=1)
    return out


@lru_cache(maxsize=4)
def _blank(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), BG, np.uint8)


def letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    """Fit ``image`` into ``width`` x ``height``; return (view, scale, offset_x, offset_y)."""
    scale = min(width / image.shape[1], height / image.shape[0])
    w, h = int(image.shape[1] * scale), int(image.shape[0] * scale)
    view = _blank(width, height).copy()
    ox, oy = (width - w) // 2, (height - h) // 2
    view[oy:oy + h, ox:ox + w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return view, scale, ox, oy


# --------------------------------------------------------------------------- drawing helpers
class Painter:
    """Collects text/pill drawing done with PIL in one pass over the canvas."""

    def __init__(self, canvas: np.ndarray) -> None:
        self.image = Image.fromarray(canvas)
        self.draw = ImageDraw.Draw(self.image)

    def text(self, xy, text, size=14, color=TEXT, kind="ui", anchor="la") -> None:
        self.draw.text(xy, text, font=font(size, kind), fill=color, anchor=anchor)

    def pill(self, x, y, text, size=12, fg=TEXT, bg=CARD, kind="bold", pad=(7, 3), radius=6) -> int:
        """Rounded label with its top-left corner at (x, y); returns its width."""
        f = font(size, kind)
        w = text_width(text, f) + 2 * pad[0]
        h = size + 2 * pad[1] + 2
        self.draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=bg)
        self.draw.text((x + pad[0], y + pad[1] + size * 0.5 + 1), text, font=f, fill=fg, anchor="lm")
        return w

    def rounded(self, box, fill=None, outline=None, width=1, radius=10) -> None:
        self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def result(self) -> np.ndarray:
        return np.asarray(self.image)


def corner_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color, thickness: int = 2) -> None:
    """Bounding box drawn as four corner brackets (a HUD look)."""
    length = max(5, int(min(x2 - x1, y2 - y1) * 0.22))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)


def text_on(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white text, whichever reads better on ``color`` (BGR)."""
    b, g, r = color
    return BLACK if 0.299 * r + 0.587 * g + 0.114 * b > 140 else TEXT


# --------------------------------------------------------------------------- app
class MonitorApp:
    def __init__(self, configs: list[CameraConfig], *, tracker: str = "bytetrack", model: str = "yolo26n.pt") -> None:
        origin = time.perf_counter()
        self.model = model
        self.cameras = [CameraFeed(c, origin, tracker, model) for c in configs]
        self.active = 0
        self.show_heatmap = False
        self.events: deque[tuple[str, int, str, str]] = deque(maxlen=4)  # (time, camera, kind, text)
        self.rooms = self._layout_rooms()
        self._snapshot: Snapshot | None = None
        self._previous_view: np.ndarray | None = None
        self._switch_at = time.perf_counter()
        self._lock = threading.Lock()
        self._background = self._build_background()
        self._chrome: np.ndarray | None = None
        self._toast: tuple[str, float] | None = None   # (text, visible until)
        self._chrome_at = 0.0
        self.cameras[0].start()
        self._running = True
        self._threads = [
            threading.Thread(target=self._analysis_loop, daemon=True),
            threading.Thread(target=self._watch_loop, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    # -- layout --------------------------------------------------------------
    def _layout_rooms(self) -> list[tuple[int, int, int, int]]:
        """Pixel rectangles of the rooms on the site map, one per camera, in order."""
        ax, ay, aw, ah = MAP_AREA
        if len(self.cameras) <= len(SITE_ROOMS):
            fractions = SITE_ROOMS[: len(self.cameras)]
        else:  # more cameras than rooms: a plain grid, also in reading order
            cols, rows = 2, math.ceil(len(self.cameras) / 2)
            fractions = [((i % cols) / cols, (i // cols) / rows, 1 / cols - 0.04, 1 / rows - 0.05)
                         for i in range(len(self.cameras))]
        return [(ax + int(fx * aw), ay + int(fy * ah), int(fw * aw), int(fh * ah)) for fx, fy, fw, fh in fractions]

    def _build_background(self) -> np.ndarray:
        """Static parts of the interface, drawn once."""
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
        p = Painter(canvas)
        # Header.
        p.draw.ellipse((PAD, 17, PAD + 14, 31), outline=ACCENT, width=3)
        p.text((PAD + 24, 24), "MONITORING", 20, TEXT, "bold", anchor="lm")
        p.text((PAD + 152, 25), "system wykrywania i śledzenia osób", 14, MUTED, anchor="lm")
        # Cards.
        for (x, y, w, h), title in ((MAP_CARD, "PLAN OBIEKTU"), (STATS_CARD, None)):
            p.rounded((x, y, x + w, y + h), fill=CARD, outline=CARD_EDGE, radius=12)
            if title:
                p.text((x + 14, y + 20), title, 12, MUTED, "bold", anchor="lm")
        p.rounded((VIEW_X, LOG_Y, VIEW_X + VIEW_W, CANVAS_H - PAD), fill=CARD, outline=CARD_EDGE, radius=12)
        p.text((VIEW_X + 14, LOG_Y + 18), "ZDARZENIA", 12, MUTED, "bold", anchor="lm")
        p.text((VIEW_X + VIEW_W - 14, LOG_Y + 18),
               "1-9 / klik  kamera     Tab  następna     H  mapa cieplna     D  dokładność     Q  wyjście",
               12, DIM, anchor="rm")
        canvas = p.result().copy()
        # Corridors on the site map.
        if len(self.cameras) <= len(SITE_ROOMS):
            centers = [(r[0] + r[2] // 2, r[1] + r[3] // 2) for r in self.rooms]
            for a, b in SITE_LINKS:
                if a < len(centers) and b < len(centers):
                    cv2.line(canvas, centers[a], centers[b], CARD_EDGE, 8)
        return canvas

    # -- control -------------------------------------------------------------
    def switch_to(self, index: int) -> None:
        if index == self.active or not 0 <= index < len(self.cameras):
            return
        self._previous_view = self._camera_view(time.perf_counter())[0]
        self.cameras[self.active].stop()
        self.active = index
        self.cameras[index].start()
        self._switch_at = time.perf_counter()

    def on_mouse(self, event, x, y, flags, param) -> None:  # noqa: ANN001
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (rx, ry, rw, rh) in enumerate(self.rooms):
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    self.switch_to(i)

    def handle_key(self, key: int) -> bool:
        """Return False to quit. ``key`` is a code from ``cv2.waitKeyEx`` (-1 = none)."""
        if key in KEY_ALIASES:
            key = ord(KEY_ALIASES[key])
        if key in (ord("q"), ord("Q"), 27):
            return False
        if ord("1") <= key <= ord("9"):
            self.switch_to(key - ord("1"))
        elif key == 9:  # Tab
            self.switch_to((self.active + 1) % len(self.cameras))
        elif key in (ord("h"), ord("H")):
            self.show_heatmap = not self.show_heatmap
        elif key in (ord("d"), ord("D")):
            cam = self.cameras[self.active]
            sizes = list(IMAGE_SIZES)
            current = cam.config.image_size
            size = sizes[(sizes.index(current) + 1) % len(sizes)] if current in sizes else sizes[0]
            cam.set_image_size(size)
            self._log(self.active, "info", f"wejście YOLO: {size} px")
            hint = {640: "szybko, gorzej widać małe postacie", 960: "kompromis",
                    1280: "dokładnie, wolniej"}.get(size, "")
            self._toast = (f"WEJŚCIE YOLO: {size} px  ·  {hint}", time.perf_counter() + TOAST_S)
        return True

    # -- events and alarms ---------------------------------------------------
    def _log(self, index: int, kind: str, text: str) -> None:
        with self._lock:
            self.events.appendleft((datetime.now().strftime("%H:%M:%S"), index, kind, text))

    def _update_alarm(self, index: int) -> None:
        cam = self.cameras[index]
        limit = cam.config.zone_limit
        over = limit is not None and cam.zone_count is not None and cam.zone_count >= limit
        if over:
            cam.alarm_clear_since = None
            if not cam.alarm:
                self._log(index, "alarm", f"ALARM  strefa „{cam.zone_name}”: {cam.zone_count} os. (limit {limit})")
            cam.alarm = True
        elif cam.alarm:
            # Switch off only after the count stays below the limit for a while,
            # so a single missed detection does not make the alarm blink.
            now = time.perf_counter()
            if cam.alarm_clear_since is None:
                cam.alarm_clear_since = now
            elif now - cam.alarm_clear_since >= ALARM_CLEAR_S:
                cam.alarm, cam.alarm_clear_since = False, None

    # -- analysis thread (camera on screen) ----------------------------------
    def _analysis_loop(self) -> None:
        last: tuple[int, int] = (-1, -1)  # (camera, frame id) analysed last
        while self._running:
            index = self.active
            camera = self.cameras[index]
            frame_id, frame = camera.latest()
            if frame is None or (index, frame_id) == last:
                time.sleep(0.002)
                continue
            last = (index, frame_id)
            snapshot, events = camera.analyse(frame, frame_id, self.show_heatmap)
            snapshot.camera = index
            with self._lock:
                self._snapshot = snapshot
            for e in events:
                if hasattr(e, "direction"):
                    self._log(index, "line", f"#{e.track_id}  przekroczenie linii ({e.direction})")
                else:
                    verb = "wejście do" if e.kind == "enter" else "wyjście ze"
                    self._log(index, "zone", f"#{e.track_id}  {verb} strefy „{e.zone}”")
            self._update_alarm(index)

    # -- background watcher (cameras not on screen) --------------------------
    def _watch_loop(self) -> None:
        detector: YoloPersonDetector | None = None
        index = 0
        while self._running:
            time.sleep(WATCH_INTERVAL_S)
            if len(self.cameras) < 2:
                continue
            index = (index + 1) % len(self.cameras)
            cam = self.cameras[index]
            if index == self.active or cam.running:
                continue
            frame = cam.probe()
            if frame is None:
                continue
            if detector is None:
                detector = YoloPersonDetector(self.model, confidence=0.25)
            detector.image_size = min(cam.config.image_size, WATCH_MAX_IMAGE_SIZE)
            detections = detector.detect(frame)
            zones = cam.geometry_for(frame)
            if index == self.active:  # user switched meanwhile - the tracker owns it now
                continue
            feet = [((d.bbox[0] + d.bbox[2]) / 2, d.bbox[3]) for d in detections]
            cam.record_people(len(detections))
            cam.zone_count = sum(zones[0].contains(pt) for pt in feet) if zones else None
            self._update_alarm(index)

    # -- the camera view -----------------------------------------------------
    def _camera_view(self, now: float) -> tuple[np.ndarray | None, list]:
        """Camera picture with HUD shapes (OpenCV) + text labels to draw later (PIL)."""
        with self._lock:
            snap = self._snapshot if self._snapshot is not None and self._snapshot.camera == self.active else None
        if snap is None:
            return None, []
        cam = self.cameras[self.active]
        view, s, ox, oy = letterbox(snap.image, VIEW_W, VIEW_H)
        to_view = lambda x, y: (int(x * s + ox), int(y * s + oy))  # noqa: E731
        labels: list = []

        # Zones: translucent fill + outline.
        for i, zone in enumerate(cam.zone_objects or []):
            pts = np.array([to_view(x, y) for x, y in zone.polygon], np.int32)
            alarm = cam.alarm and i == 0
            color = ALARM if alarm else ZONE_COLOR
            x, y, w, h = cv2.boundingRect(pts)
            x0, y0 = max(x, 0), max(y, 0)
            roi = view[y0:y + h, x0:x + w]
            tinted = roi.copy()
            cv2.fillPoly(tinted, [pts - (x0, y0)], color)
            cv2.addWeighted(tinted, 0.15, roi, 0.85, 0, dst=roi)
            cv2.polylines(view, [pts], True, color, 2, cv2.LINE_AA)
            count = cam.zone_count if i == 0 and cam.zone_count is not None else ""
            limit = f" / {cam.config.zone_limit}" if i == 0 and cam.config.zone_limit else ""
            labels.append(("pill", max(x, 0) + 4, max(y, 0) + 4, f"{zone.name.upper()}  {count}{limit}", color))

        # Counting line with its counters.
        if cam.line_points and cam.pipeline is not None and cam.pipeline.line_counter is not None:
            a, b = to_view(*cam.line_points[0]), to_view(*cam.line_points[1])
            cv2.line(view, a, b, LINE_COLOR, 2, cv2.LINE_AA)
            lc = cam.pipeline.line_counter
            lx = min(max(a[0] + 6, 4), VIEW_W - 150)
            ly = min(max(a[1] - 26, 4), VIEW_H - 30)
            labels.append(("pill", lx, ly, f"IN {lc.in_count}  ·  OUT {lc.out_count}", LINE_COLOR))

        # Trajectories and boxes - thin lines so the people stay visible.
        for track_id, path in snap.paths.items():
            if len(path) < 2:
                continue
            pts = np.array([to_view(x, y) for x, y in path], np.int32)
            cv2.polylines(view, [pts], False, color_for_id(track_id), 1, cv2.LINE_AA)
        counted_zone = cam.zone_objects[0] if cam.zone_objects else None
        zone_color = ALARM if cam.alarm else ZONE_COLOR
        for t in snap.tracks:
            x1, y1 = to_view(t.bbox[0], t.bbox[1])
            x2, y2 = to_view(t.bbox[2], t.bbox[3])
            color = color_for_id(t.track_id)
            corner_box(view, x1, y1, x2, y2, color, 1)
            foot = ((x1 + x2) // 2, y2)
            if counted_zone is not None and counted_zone.contains(t.anchor):
                # This person is counted in the zone: the foot point is inside it.
                cv2.circle(view, foot, 6, zone_color, 2, cv2.LINE_AA)
            cv2.circle(view, foot, 2, color, -1, cv2.LINE_AA)
            tag = str(t.track_id)
            (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)
            ty = max(y1 - 12, 0)
            cv2.rectangle(view, (x1, ty), (x1 + tw + 5, ty + 11), color, -1)
            cv2.putText(view, tag, (x1 + 2, ty + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.33, text_on(color), 1, cv2.LINE_AA)
        return view, labels

    def _transition(self, now: float) -> tuple[np.ndarray, str | None] | None:
        """(picture, caption) of the "signal lost" animation, or None when it is over."""
        t = now - self._switch_at
        if t < ROLL_END and self._previous_view is not None:
            shift = int(VIEW_H * 0.7 * (t / ROLL_END) ** 2)
            return tear(np.roll(self._previous_view, -shift, axis=0), bands=8, max_shift=120), None
        if t < STATIC_END:
            return static_noise(VIEW_W, VIEW_H), "BRAK SYGNAŁU" if int(t * 10) % 2 == 0 else None
        if t >= SETTLE_END and self._has_view():
            return None
        view, _ = self._camera_view(now)
        if view is None:  # model still loading or no frame yet
            return static_noise(VIEW_W, VIEW_H), "ŁĄCZENIE…"
        if t < SETTLE_END:
            u = (t - STATIC_END) / (SETTLE_END - STATIC_END)
            rolled = np.roll(view, int(VIEW_H * 0.35 * (1 - u) ** 2), axis=0)
            return cv2.addWeighted(tear(rolled, bands=4, max_shift=int(50 * (1 - u)) + 1), u,
                                   static_noise(VIEW_W, VIEW_H), 1 - u, 0), None
        return None

    def _has_view(self) -> bool:
        with self._lock:
            return self._snapshot is not None and self._snapshot.camera == self.active

    # -- whole interface -----------------------------------------------------
    def step(self) -> np.ndarray:
        return self.render()

    def render(self) -> np.ndarray:
        now = time.perf_counter()
        if self._chrome is None or now - self._chrome_at >= 1 / CHROME_FPS:
            self._chrome, self._chrome_at = self._render_chrome(now), now
        canvas = self._chrome.copy()

        transition = self._transition(now)
        labels: list = []
        caption = None
        if transition is not None:
            view, caption = transition
        else:
            view, labels = self._camera_view(now)
            if view is None:
                view, caption = static_noise(VIEW_W, VIEW_H), "ŁĄCZENIE…"
        cam = self.cameras[self.active]
        if cam.alarm and transition is None and int(now * 2) % 2 == 0:
            cv2.rectangle(view, (2, 2), (VIEW_W - 3, VIEW_H - 3), ALARM, 4)
        p = Painter(view)
        self._draw_view_text(p, labels, caption, now)
        canvas[VIEW_Y:VIEW_Y + VIEW_H, VIEW_X:VIEW_X + VIEW_W] = p.result()
        return canvas

    def _render_chrome(self, now: float) -> np.ndarray:
        """Everything except the camera view: header, site map, statistics, event log."""
        canvas = self._background.copy()
        self._draw_map_shapes(canvas, now)
        self._draw_sparkline(canvas)
        p = Painter(canvas)
        self._draw_header(p, now)
        self._draw_map_text(p, now)
        self._draw_stats(p)
        self._draw_log(p)
        return p.result()

    def _draw_view_text(self, p: Painter, labels: list, caption: str | None, now: float) -> None:
        cam = self.cameras[self.active]
        for kind, x, y, text, color in labels:
            p.pill(x, y, text, 12, fg=text_on(color), bg=color)
        # Camera name, REC, time.
        x = 12
        x += p.pill(x, 12, f"CAM {self.active + 1:02d}", 15, fg=BLACK, bg=ACCENT, pad=(9, 4)) + 6
        p.pill(x, 12, cam.config.label, 15, fg=TEXT, bg=CARD, pad=(9, 4))
        rec_x = VIEW_W - 78
        p.rounded((rec_x, 12, rec_x + 66, 38), fill=CARD, radius=6)
        if int(now * 2) % 2 == 0:
            p.draw.ellipse((rec_x + 10, 19, rec_x + 22, 31), fill=ALARM)
        p.text((rec_x + 30, 25), "REC", 14, TEXT, "bold", anchor="lm")
        stamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        p.pill(12, VIEW_H - 36, stamp, 13, fg=TEXT, bg=CARD, kind="mono", pad=(8, 4))
        if self.show_heatmap:
            p.pill(VIEW_W - 130, VIEW_H - 36, "MAPA CIEPLNA", 12, fg=BLACK, bg=WARN, pad=(8, 4))
        if self._toast is not None and now < self._toast[1]:
            text = self._toast[0]
            w = text_width(text, font(18, "bold")) + 32
            p.pill((VIEW_W - w) // 2, VIEW_H - 80, text, 18, fg=BLACK, bg=ACCENT, pad=(16, 7), radius=8)
        if cam.alarm and caption is None:
            text = f"ALARM  ·  {cam.zone_count} os. w strefie „{cam.zone_name}”  (limit {cam.config.zone_limit})"
            w = text_width(text, font(15, "bold")) + 24
            p.pill((VIEW_W - w) // 2, 12, text, 15, fg=TEXT, bg=ALARM, pad=(12, 4))
        if caption:
            p.text((VIEW_W // 2, VIEW_H // 2), caption, 44, TEXT, "bold", anchor="mm")

    def _draw_header(self, p: Painter, now: float) -> None:
        alarms = sum(c.alarm for c in self.cameras)
        x = CANVAS_W - PAD
        clock = datetime.now().strftime("%d.%m.%Y   %H:%M:%S")
        p.text((x, 25), clock, 16, TEXT, "mono", anchor="rm")
        x -= text_width(clock, font(16, "mono")) + 16
        chip = f"{alarms} ALARM" if alarms else "BRAK ALARMÓW"
        bg = ALARM if alarms and int(now * 2) % 2 == 0 else (bgr("#5c1d22") if alarms else bgr("#173a2c"))
        w = text_width(chip, font(12, "bold")) + 14
        p.pill(x - w, 14, chip, 12, fg=TEXT if alarms else OK, bg=bg)
        x -= w + 8
        chip = f"{len(self.cameras)} KAMER"
        w = text_width(chip, font(12, "bold")) + 14
        p.pill(x - w, 14, chip, 12, fg=MUTED, bg=CARD)

    def _draw_map_shapes(self, canvas: np.ndarray, now: float) -> None:
        """Room rectangles (OpenCV); their text is drawn later with PIL."""
        blink = int(now * 3) % 2 == 0
        for i, (cam, (rx, ry, rw, rh)) in enumerate(zip(self.cameras, self.rooms)):
            active = i == self.active
            fill = bgr("#173a33") if active else TILE
            if cam.alarm:
                fill = bgr("#5c1d22") if blink else bgr("#3a171b")
            edge = ALARM if cam.alarm else ACCENT if active else CARD_EDGE
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), fill, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), edge, 2 if active or cam.alarm else 1)

    def _draw_map_text(self, p: Painter, now: float) -> None:
        for i, (cam, (rx, ry, rw, rh)) in enumerate(zip(self.cameras, self.rooms)):
            active = i == self.active
            badge_bg = ALARM if cam.alarm else ACCENT if active else CARD_EDGE
            p.pill(rx + 6, ry + 6, f"{i + 1:02d}", 11, fg=BLACK if (active or cam.alarm) else MUTED, bg=badge_bg,
                   pad=(5, 2), radius=4)
            p.text((rx + 38, ry + 15), cam.config.label, 12, TEXT if active else MUTED, "bold", anchor="lm")
            if cam.people is not None:
                color = ALARM if cam.alarm else TEXT if active else MUTED
                p.text((rx + rw - 8, ry + rh - 6), f"{cam.people}", 18 if rh > 60 else 14, color, "bold", anchor="rd")
                if rh > 60:
                    p.text((rx + 8, ry + rh - 8), "osób", 11, DIM, anchor="ld")

    def _stats_tiles(self) -> list[tuple[str, str, str, tuple[int, int, int]]]:
        cam = self.cameras[self.active]
        stats = cam.last_stats
        if stats is None:
            return []
        fps_color = OK if stats.fps >= 0.85 * cam.fps or stats.fps >= 25 else WARN if stats.fps >= 12 else ALARM
        tiles = [("OSOBY W KADRZE", str(stats.people_in_frame), "", TEXT),
                 ("FPS ANALIZY", f"{stats.fps:.0f}" if stats.fps > 0 else "–", f"/ {cam.fps:.0f}",
                  fps_color if stats.fps > 0 else MUTED)]
        if cam.pipeline is not None and cam.pipeline.line_counter is not None:
            tiles += [("LINIA  IN", str(stats.line_in), "", LINE_COLOR), ("LINIA  OUT", str(stats.line_out), "", LINE_COLOR)]
        return tiles

    def _draw_stats(self, p: Painter) -> None:
        x0, y0, w0, h0 = STATS_CARD
        cam = self.cameras[self.active]
        p.text((x0 + 14, y0 + 20), f"ANALIZA  ·  CAM {self.active + 1:02d}", 12, MUTED, "bold", anchor="lm")
        p.text((x0 + w0 - 14, y0 + 20), f"YOLO {cam.config.image_size} px", 12, DIM, anchor="rm")
        tiles = self._stats_tiles()
        if not tiles:
            p.text((x0 + 14, y0 + 56), "łączenie…", 14, MUTED)
            return
        tw, th = (w0 - 36) // 2, 58
        for k, (label, value, suffix, color) in enumerate(tiles):
            tx = x0 + 12 + (k % 2) * (tw + 12)
            ty = y0 + 36 + (k // 2) * (th + 10)
            p.rounded((tx, ty, tx + tw, ty + th), fill=TILE, radius=8)
            p.text((tx + 10, ty + 14), label, 10, MUTED, "bold", anchor="lm")
            p.text((tx + 10, ty + 42), value, 26, color, "bold", anchor="ls")
            if suffix:
                vx = tx + 14 + text_width(value, font(26, "bold"))
                p.text((vx, ty + 42), suffix, 13, MUTED, anchor="ls")
        # Zone occupancy bar.
        y = y0 + 36 + 2 * (th + 10) + 4
        if cam.zone_name is not None:
            count = cam.zone_count or 0
            limit = cam.config.zone_limit
            p.text((x0 + 14, y + 6), f"STREFA „{cam.zone_name.upper()}”", 10, MUTED, "bold", anchor="lm")
            p.text((x0 + w0 - 14, y + 6), f"{count} / {limit}" if limit else str(count), 13,
                   ALARM if cam.alarm else TEXT, "bold", anchor="rm")
            bar = (x0 + 14, y + 16, x0 + w0 - 14, y + 24)
            p.rounded(bar, fill=TILE, radius=4)
            if limit:
                frac = min(count / limit, 1.0)
                color = ALARM if frac >= 1 else WARN if frac >= 0.7 else OK
                if frac > 0:
                    p.rounded((bar[0], bar[1], bar[0] + int((bar[2] - bar[0]) * frac), bar[3]), fill=color, radius=4)
        p.text((x0 + 14, self._spark_box()[1] - 8), "OSOBY  ·  OSTATNIA MINUTA", 10, MUTED, "bold", anchor="lm")

    def _spark_box(self) -> tuple[int, int, int, int]:
        x0, y0, w0, h0 = STATS_CARD
        top = y0 + 36 + 2 * (58 + 10) + 4 + 24 + 34  # below the tiles and the zone bar
        return x0 + 14, top, x0 + w0 - 14, y0 + h0 - 14

    def _draw_sparkline(self, canvas: np.ndarray) -> None:
        cam = self.cameras[self.active]
        x1, y1, x2, y2 = self._spark_box()
        cv2.line(canvas, (x1, y2), (x2, y2), CARD_EDGE, 1)
        if len(cam.history) < 2:
            return
        now = time.perf_counter()
        values = [v for _, v in cam.history]
        top = max(max(values), 1)
        pts = np.array([(x2 - (now - t) / HISTORY_S * (x2 - x1), y2 - v / top * (y2 - y1)) for t, v in cam.history],
                       np.int32)
        area = np.vstack([pts, [[pts[-1][0], y2], [pts[0][0], y2]]])
        overlay = canvas[y1:y2 + 1, x1:x2 + 1].copy()
        cv2.fillPoly(canvas, [area], bgr("#1c3f3a"))
        cv2.addWeighted(canvas[y1:y2 + 1, x1:x2 + 1], 0.6, overlay, 0.4, 0, dst=canvas[y1:y2 + 1, x1:x2 + 1])
        cv2.polylines(canvas, [pts], False, ACCENT, 2, cv2.LINE_AA)

    def _draw_log(self, p: Painter) -> None:
        with self._lock:
            events = list(self.events)
        x0, y = VIEW_X + 14, LOG_Y + 40
        if not events:
            p.text((x0, y + 8), "brak zdarzeń", 13, DIM)
            return
        colors = {"alarm": ALARM, "line": LINE_COLOR, "zone": ZONE_COLOR, "info": ACCENT}
        for i, (stamp, index, kind, text) in enumerate(events):
            yy = y + i * 24
            p.text((x0, yy + 9), stamp, 13, MUTED, "mono", anchor="lm")
            p.pill(x0 + 76, yy, f"CAM {index + 1:02d}", 11, fg=BLACK, bg=colors.get(kind, MUTED), pad=(6, 2), radius=4)
            p.text((x0 + 146, yy + 9), text, 14, ALARM if kind == "alarm" else TEXT if i == 0 else MUTED,
                   "bold" if kind == "alarm" else "ui", anchor="lm")

    # -- main loop -----------------------------------------------------------
    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, CANVAS_W, CANVAS_H)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        try:
            while True:
                started = time.perf_counter()
                cv2.imshow(WINDOW, self.step())
                wait_ms = max(1, int((1 / UI_FPS - (time.perf_counter() - started)) * 1000))
                if not self.handle_key(cv2.waitKeyEx(wait_ms)):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.close()
            cv2.destroyAllWindows()

    def close(self) -> None:
        self._running = False
        for thread in self._threads:
            thread.join(timeout=3)
        for camera in self.cameras:
            camera.release()


# --------------------------------------------------------------------------- CLI
def default_cameras() -> list[CameraConfig]:
    from videos import TEST_VIDEOS, download_test_videos

    paths = download_test_videos()
    return [
        CameraConfig(str(paths[name]), info["label"], info["line"], tuple(info["zones"]),
                     info["image_size"], info.get("zone_limit"))
        for name, info in TEST_VIDEOS.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live multi-camera monitoring view")
    parser.add_argument("--video", action="append", default=[],
                        help="video path or camera index; repeatable (default: the test videos)")
    parser.add_argument("--image-size", type=int, help="YOLO input size for --video sources (default 960)")
    parser.add_argument("--tracker", default="bytetrack", help="bytetrack or botsort")
    parser.add_argument("--model", default="yolo26n.pt")
    args = parser.parse_args()

    if args.video:
        configs = [
            CameraConfig(int(v) if v.isdecimal() else v,
                         "KAMERA" if v.isdecimal() else Path(v).stem.upper()[:12],
                         image_size=args.image_size or 960)
            for v in args.video
        ]
    else:
        configs = default_cameras()
    MonitorApp(configs, tracker=args.tracker, model=args.model).run()


if __name__ == "__main__":
    main()
