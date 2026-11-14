# Wykrywanie, śledzenie i analiza ruchu osób

YOLO (Ultralytics) + ByteTrack / BoT-SORT + analiza ruchu: trajektorie, mapa cieplna,
wirtualna linia i strefy.

## Szybki start

```
python -m pip install -r requirements.txt
python videos.py          # pobiera 3 nagrania testowe do data/videos/
```

Następnie otwórz **`demo.ipynb`** i uruchom wszystkie komórki – notebook opisuje działanie systemu,
przetwarza nagrania testowe i pokazuje wyniki (śledzenie, trajektorie, mapy cieplne, kierunki, prędkości,
porównanie ByteTrack vs BoT-SORT). Przy pierwszym uruchomieniu Ultralytics pobierze model `yolo26n.pt` (~5 MB).

## Nagrania testowe

| Plik | Scena | Źródło |
|---|---|---|
| `people-walking.mp4` | hala, widok z góry, ~30 osób, 1920×1080 | Roboflow Supervision |
| `market-square.mp4` | plac, widok z góry, tłum, 4K pionowo | Roboflow Supervision |
| `mot17-09.mp4` | deptak przy sklepach, widok z boku, 960×540 | MOTChallenge (MOT17-09) |

Adresy, linie i strefy dla każdego nagrania: `videos.py`.

## Wiersz poleceń

```
python main.py data/videos/people-walking.mp4 --show --heatmap --line 0,0.5,1,0.5 --zone srodek:0.35,0.3,0.65,0.8
python main.py data/videos/market-square.mp4 --image-size 1280 --output wynik.mp4
python main.py 0 --show                     # kamera internetowa
python main.py nagranie.mp4 --tracker botsort --show
```

- `--line x1,y1,x2,y2` – linia zliczająca; `in` = przejście z ujemnej na dodatnią stronę
  linii start→koniec (dla linii poziomej rysowanej od lewej: ruch w dół obrazu).
- `--zone [nazwa:]x1,y1,x2,y2` – prostokąt; 3+ punkty (`x,y;x,y;x,y`) – wielokąt. Można podać wiele razy.
- Współrzędne w pikselach lub jako ułamki szerokości/wysokości (gdy wszystkie wartości ≤ 1).
- `--image-size` – rozmiar wejścia YOLO; dla małych postaci (np. widok z wysoka w 4K) zwiększ do 1280.
- W oknie podglądu: `q` – wyjście, `h` – włącz/wyłącz mapę cieplną.

Każde uruchomienie tworzy folder w `data/results/` z plikami `stats.csv` (statystyki każdej klatki),
`events.csv` (przejścia przez linię, wejścia/wyjścia ze stref), `tracks.csv` (pełna historia śladów),
`heatmap.png`, `heatmap_overlay.png`, `summary.json`.


## Struktura

| Plik | Rola |
|---|---|
| `detection/yolo_detector.py` | detekcja osób (YOLO), osobny CLI `run_detector.py` |
| `tracking/person_tracker.py` | ByteTrack / BoT-SORT przez `model.track(persist=True)` |
| `analysis/trajectories.py` | historia pozycji na żywo |
| `analysis/heatmap.py` | mapa cieplna na żywo |
| `analysis/line_counter.py` | zliczanie przejść przez linię |
| `analysis/zones.py` | liczba osób w strefach, wejścia/wyjścia |
| `analysis/trajectory_analysis.py` | analiza po przetworzeniu: statystyki trajektorii, mapa cieplna z `tracks.csv` |
| `visualization/drawing.py`, `plots.py` | rysowanie na klatkach, wykresy do notebooka |
| `pipeline.py` | przetwarzanie jednej klatki |
| `runner.py` | przetwarzanie całego nagrania i zapis wyników |
| `main.py`, `demo.ipynb`, `videos.py` | CLI, notebook demonstracyjny, nagrania testowe |

Pozycja osoby = środek dolnej krawędzi ramki (przybliżony punkt kontaktu z podłożem).
