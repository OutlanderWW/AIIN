# Wykrywanie, śledzenie i analiza ruchu osób

YOLO (Ultralytics) + ByteTrack / BoT-SORT + analiza ruchu: trajektorie, mapa cieplna,
wirtualna linia i strefy.

## Szybki start

```bash
python -m pip install -r requirements.txt
python videos.py          # pobiera 3 nagrania testowe do data/videos/
python -m streamlit run app.py --server.address 127.0.0.1
```

Otwórz `http://127.0.0.1:8501` i wybierz źródło obrazu. Możesz wgrać własny plik,
użyć nagrania testowego z gotową linią i strefą, wybrać lokalną kamerę lub podać adres RTSP.
Kamera lokalna musi być podłączona do komputera uruchamiającego Streamlit.
Podczas analizy widać obraz z adnotacjami, bieżące liczniki i FPS. Po zakończeniu
lub ręcznym zatrzymaniu dostępne są mapa cieplna, wykres liczby osób w czasie i pliki CSV.
Przy pierwszym uruchomieniu Ultralytics pobierze model `yolo26n.pt` (~5 MB).

Notebook **`demo.ipynb`** zawiera analizę trzech nagrań, wykresy kierunków i prędkości
oraz porównanie ByteTrack z BoT-SORT.

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

Każda zakończona analiza tworzy folder w `data/results/` z plikami `stats.csv` (statystyki każdej klatki),
`events.csv` (przejścia przez linię, wejścia/wyjścia ze stref), `tracks.csv` (pełna historia śladów),
`heatmap.png`, `heatmap_overlay.png`, `summary.json`.

## Weryfikacja

```bash
python -m unittest discover -s tests -t . -v
```

Testy działają bez pobierania modelu. Sprawdzają akumulację mapy cieplnej,
zliczanie linii i stref na znanej trajektorii, eksport wyników oraz obsługę błędnego adresu RTSP.

1. W Streamlit wybierz **Nagranie testowe → people-walking.mp4**, ustaw limit 120 klatek i włącz nakładkę mapy. Podczas analizy sprawdź ramki z ID, kolorowe ślady, linię i strefę. Po zakończeniu sprawdź wykres, mapę i trzy pliki CSV.
2. Zatrzymaj drugie uruchomienie w trakcie. Wyniki powinny pojawić się po przetworzeniu ostatniej klatki. Do kontroli zliczania użyj krótkiego nagrania, na którym jedna osoba przechodzi w dół poziomej linii narysowanej od lewej do prawej, wraca i wchodzi oraz wychodzi z wyznaczonej strefy. Porównaj zdarzenia z `events.csv` i zajętość strefy z kolumną `zone_*` w `stats.csv`. Linia liczy każde ID raz na kierunek, nawet jeśli wraca ono tą samą drogą ponownie.
3. Wybierz **Kamera komputera** (zwykle indeks `0`) i wykonaj ten sam ruch przed kamerą. Dla kamery IP powtórz próbę z jej adresem RTSP. Sprawdź też działanie przycisku **Zatrzymaj analizę**.
4. Porównaj `average_fps` z `video_fps` w `summary.json`. Do porównania CPU i GPU użyj tego samego nagrania, trackera i rozmiaru wejścia. Na CPU ustaw `--device cpu`, na GPU NVIDIA `--device 0`, a na zgodnym Macu `--device mps`. Własnoręczne oznaczenie osób i ich ID w próbce klatek pozwoli ocenić pominięte detekcje, fałszywe detekcje i zmiany ID po zasłonięciu.

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
| `app.py` | interfejs Streamlit |
| `main.py`, `demo.ipynb`, `videos.py` | CLI, notebook demonstracyjny, nagrania testowe |

Pozycja osoby = środek dolnej krawędzi ramki (przybliżony punkt kontaktu z podłożem).
