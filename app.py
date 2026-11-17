from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from time import monotonic
from urllib.parse import urlsplit

import cv2
import numpy as np
import streamlit as st

from pipeline import FrameStats
from runner import RunConfig, RunResult, run
from tracking import TRACKER_CONFIGS
from videos import TEST_VIDEOS, VIDEOS_DIR


class AnalysisJob:
    def __init__(self, config: RunConfig, upload: bytes | None = None) -> None:
        self.result: Future[RunResult] = Future()
        self.stop = Event()
        self._lock = Lock()
        self._last_refresh = monotonic()
        self._latest: tuple[np.ndarray, FrameStats] | None = None
        Thread(target=self._run, args=(config, upload), daemon=True).start()

    def snapshot(self) -> tuple[np.ndarray, FrameStats] | None:
        with self._lock:
            self._last_refresh = monotonic()
            return self._latest

    def _on_frame(self, frame: np.ndarray, stats: FrameStats) -> bool:
        with self._lock:
            self._latest = frame, stats
            return not self.stop.is_set() and monotonic() - self._last_refresh < 30

    def _run(self, config: RunConfig, upload: bytes | None) -> None:
        try:
            if upload is None:
                result = run(config, on_frame=self._on_frame)
            else:
                with TemporaryDirectory(prefix="aiin-upload-") as folder:
                    source = Path(folder) / Path(config.source).name
                    source.write_bytes(upload)
                    result = run(replace(config, source=source), on_frame=self._on_frame)
        except Exception as exc:
            self.result.set_exception(exc)
        else:
            self.result.set_result(result)


def start_job(active: bool) -> AnalysisJob | None:
    source_type = st.selectbox("Źródło obrazu", [
        "Plik wideo", "Nagranie testowe", "Kamera komputera", "Kamera IP (RTSP)",
    ])
    source = None
    upload = None
    preset = None
    config_key = source_type
    if source_type == "Plik wideo":
        upload = st.file_uploader("Wybierz nagranie", type=["mp4", "mov", "avi", "mkv"])
        if upload is not None:
            source = Path(upload.name).name
    elif source_type == "Nagranie testowe":
        sample = st.selectbox("Nagranie", list(TEST_VIDEOS))
        config_key = sample
        source = VIDEOS_DIR / sample
        preset = TEST_VIDEOS[sample]
        if not source.is_file():
            st.info("Pobierz nagrania poleceniem `python videos.py`.")
            source = None
    elif source_type == "Kamera komputera":
        source = st.number_input("Indeks kamery", min_value=0, value=0, step=1)
        st.caption("Kamera musi być podłączona do komputera uruchamiającego Streamlit.")
    else:
        source = st.text_input("Adres RTSP", type="password", placeholder="rtsp://adres/strumien").strip()

    tracker = st.selectbox("Algorytm śledzenia", list(TRACKER_CONFIGS))
    image_sizes = [640, 1280, 1920]
    default_size = preset["image_size"] if preset else 640
    image_size = st.selectbox(
        "Rozmiar wejścia YOLO", image_sizes, index=image_sizes.index(default_size),
        key=f"image_size_{config_key}",
    )
    with st.expander("Linia i strefy"):
        st.caption("Współrzędne podaj w pikselach lub jako ułamki od 0 do 1.")
        line = st.text_input(
            "Linia (x1,y1,x2,y2)", value=preset["line"] if preset else "", key=f"line_{config_key}",
        )
        zones = st.text_area(
            "Strefy (po jednej w wierszu)", value="\n".join(preset["zones"]) if preset else "",
            placeholder="wejscie:0.1,0.2,0.4,0.7", key=f"zones_{config_key}",
        )
    heatmap = st.checkbox("Nakładaj mapę cieplną podczas analizy")
    max_frames = st.number_input("Limit klatek (0 = bez limitu)", min_value=0, value=0, step=1)
    if not st.button("Uruchom analizę", type="primary", disabled=active or source is None or source == ""):
        return None
    if source_type == "Kamera IP (RTSP)":
        try:
            url = urlsplit(source)
            valid_rtsp = url.scheme in {"rtsp", "rtsps"} and bool(url.hostname)
        except ValueError:
            valid_rtsp = False
        if not valid_rtsp:
            st.error("Podaj adres `rtsp://` lub `rtsps://` z nazwą hosta.")
            return None

    return AnalysisJob(RunConfig(
        source=source,
        tracker=tracker,
        image_size=image_size,
        line=line.strip() or None,
        zones=[zone.strip() for zone in zones.splitlines() if zone.strip()],
        heatmap_overlay=heatmap,
        max_frames=max_frames or None,
    ), upload.getvalue() if upload is not None else None)


def show_image(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / max(width, height))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    st.image(frame, channels="BGR", width="stretch")


def show_results(result: RunResult) -> None:
    st.subheader("Liczba osób w czasie")
    st.line_chart(result.stats, x="time_s", y="people_in_frame", x_label="Czas [s]", y_label="Osoby")
    average_fps = result.summary["average_fps"]
    fps_label = f"{average_fps:.1f} FPS" if average_fps is not None else "—"
    st.caption(f"Klatki {result.summary['frames']} · Średnia szybkość przetwarzania {fps_label}")
    if not result.events.empty:
        with st.expander("Przekroczenia linii i zdarzenia w strefach"):
            st.dataframe(result.events, hide_index=True)
    for column, filename in zip(st.columns(3), ("stats.csv", "events.csv", "tracks.csv")):
        column.download_button(f"Pobierz {filename}", (result.run_dir / filename).read_bytes(), filename,
                               mime="text/csv", on_click="ignore")


def main() -> None:
    st.set_page_config(page_title="Analiza ruchu osób", layout="wide")
    st.title("Analiza ruchu osób")
    job: AnalysisJob | None = st.session_state.get("analysis_job")
    active = job is not None and not job.result.done()
    with st.sidebar:
        st.header("Konfiguracja")
        new_job = start_job(active)
    if new_job is not None:
        st.session_state.analysis_job = job = new_job
        active = True
    if job is None:
        st.info("Wybierz źródło obrazu i uruchom analizę.")
        return

    @st.fragment(run_every=0.25 if active else None)
    def dashboard() -> None:
        if active and job.result.done():
            st.rerun()
        latest = job.snapshot()
        result = None
        if active:
            stop_requested = st.button("Zatrzymaj analizę", disabled=job.stop.is_set())
            if stop_requested:
                job.stop.set()
            if job.stop.is_set():
                st.info("Zatrzymywanie analizy…")
            elif latest is None:
                st.info("Otwieranie źródła i ładowanie modelu…")
        else:
            try:
                result = job.result.result()
            except Exception as exc:
                st.error(f"Analiza zakończona błędem. {exc}")
                return
        if latest is not None:
            frame, stats = latest
            people, incoming, outgoing, fps = st.columns(4)
            people.metric("Osoby w kadrze", stats.people_in_frame)
            incoming.metric("Przejścia IN", stats.line_in)
            outgoing.metric("Przejścia OUT", stats.line_out)
            fps.metric("FPS przetwarzania", f"{stats.fps:.1f}")
            if stats.zone_counts:
                st.caption("Osoby w strefach · " + " · ".join(
                    f"{name} {count}" for name, count in stats.zone_counts.items()
                ))
            if result is None:
                show_image(frame)
            else:
                preview, heatmap = st.columns(2)
                with preview:
                    st.subheader("Ostatnia klatka")
                    show_image(frame)
                with heatmap:
                    st.subheader("Mapa cieplna")
                    show_image(result.heatmap_image)
        if result is not None:
            show_results(result)

    dashboard()


if __name__ == "__main__":
    main()
