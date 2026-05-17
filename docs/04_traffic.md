# 04 — Traffic Analysis

## Overview

The traffic module provides time-aware congestion estimation, travel time multipliers, spatial heatmaps, and isochrone (reachable area) generation. It combines crowd-sourced observations stored in the database with hard-coded Vietnamese traffic patterns as a prior. The routing engine uses this module to weight edges and score candidate routes.

---

## Architecture / Data Flow

```
DB: traffic_observations          Default Vietnamese patterns
(hour, weekday, congestion)       _DEFAULT_HOURLY[0..23]
         │                                │
         ▼                                ▼
  TrafficAnalyzer.refresh()    ← every 5 min or on force
         │
         ├── _hourly_cache[(hour, weekday)] → avg congestion
         └── CongestionGrid → spatial heatmap
                  │
                  ▼
         congestion_at(hour, weekday, lat, lon)
         = 0.6 × temporal + 0.4 × spatial  (if spatial data exists)
                  │
                  ▼
         speed_multiplier(c) = max(0.2, 1.0 - 0.8 × c)   [Greenshields]
         travel_time_factor(c) = 1 / speed_multiplier(c)
                  │
         ┌────────┴────────┐
         ▼                 ▼
  full_day_curve()   best_departure_window()
  (24-item list)     (optimal hour ±2h)
         │
         ▼
  render_traffic_timeline()  → HTML bar chart
  IsochroneGenerator.generate() → convex hull polygons
```

---

## Key Classes and Functions

### `core/traffic_analyzer.py`

#### `TrafficAnalyzer`

```python
class TrafficAnalyzer:
    async def refresh(force: bool = False) -> None
    def congestion_at(hour, weekday, lat, lon) -> float   # [0.0, 1.0]
    def speed_multiplier(congestion: float) -> float       # [0.2, 1.0]
    def travel_time_factor(congestion: float) -> float     # [1.0, 5.0]
    def full_day_curve(weekday: int = 1) -> list[dict]
    def best_departure_window(target_hour, tolerance_hours, weekday) -> dict
    def heatmap_data() -> list[dict]
    @staticmethod _status_label(c: float) -> str
```

**Default hourly patterns** (`_DEFAULT_HOURLY`): Based on typical HCMC traffic rhythms.

| Hour | Congestion | Status |
|---|---|---|
| 0–4 | 0.02–0.05 | thông thoáng |
| 6–7 | 0.50–0.85 | hơi đông → tắc nghẽn |
| 7–8 | 0.85 | tắc nghẽn |
| 11–12 | 0.55–0.60 | hơi đông |
| 17–18 | 0.90 | tắc nghẽn |
| 18–19 | 0.85 | tắc nghẽn |
| 22–23 | 0.15–0.30 | thông thoáng |

**Weekday multipliers** (`_WEEKDAY_MULT`):
- Mon–Thu: 1.0
- Fri: 1.1 (slightly heavier)
- Sat: 0.8
- Sun: 0.6

**`congestion_at`**: Blends DB data with default pattern:
- If DB has data for `(hour, weekday)`: `base = db_val`
- Otherwise: `base = DEFAULT_HOURLY[hour] × WEEKDAY_MULT[weekday]`
- If spatial data exists for `(lat, lon)`: `result = 0.6 × base + 0.4 × spatial`

**`speed_multiplier`** (Greenshields model approximation):
```
speed_mult = max(0.2, 1.0 - 0.8 × congestion)
```
- `congestion = 0.0` → `speed_mult = 1.0` (free flow)
- `congestion = 1.0` → `speed_mult = 0.2` (gridlock, 20% of free-flow speed)

**`_status_label`**:
| Congestion | Label |
|---|---|
| < 0.30 | thông thoáng |
| 0.30–0.55 | bình thường |
| 0.55–0.75 | hơi đông |
| 0.75–0.90 | tắc nghẽn |
| ≥ 0.90 | kẹt xe nặng |

**`best_departure_window`**: Scans ±`tolerance_hours` in 30-minute increments, finds the slot with minimum congestion. Returns:
```json
{
  "recommended_hour": 9,
  "congestion": 0.35,
  "status": "bình thường",
  "save_minutes": 12
}
```

**`refresh`**: Reloads from DB every 5 minutes (or immediately if `force=True`). Loads:
1. Hourly averages grouped by `(hour, weekday)` into `_hourly_cache`.
2. Last 7 days of spatial observations into `CongestionGrid`.

---

#### `CongestionGrid`

```python
class CongestionGrid:
    cell_size_deg: float = 0.002   # ~220 m per cell at equator
    def add(lat, lon, congestion) -> None
    def get(lat, lon) -> float     # average congestion for cell, or 0.0
    def to_heatmap_data() -> list[dict]  # [{lat, lon, intensity}]
```

Grid cells are indexed by `(round(lat / 0.002), round(lon / 0.002))`. Each cell accumulates a list of congestion values; `get()` returns their mean. `to_heatmap_data()` returns all non-empty cells for the Folium HeatMap plugin.

---

#### `IsochroneGenerator`

```python
class IsochroneGenerator:
    def __init__(osm_graph: OSMGraph, analyzer: TrafficAnalyzer)
    def generate(lat, lon, minutes: list[int], depart_time) -> dict[int, list[tuple]]
```

**Algorithm**:
1. Finds the nearest OSM node to `(lat, lon)`.
2. Computes `congestion_at(hour, weekday, lat, lon)` and `travel_time_factor`.
3. Sets `_iso_w = base_travel_time × factor` for every edge.
4. Runs `nx.single_source_dijkstra_path_length(G, center, cutoff=max_sec, weight="_iso_w")`.
5. Collects all reachable node coordinates.
6. Computes convex hull via Graham scan (`_convex_hull`).

Returns `{5: [(lat,lon),...], 10: [...], 15: [...]}` — one polygon per minute value.

---

### `web/routes/traffic.py`

| Endpoint | Method | Description |
|---|---|---|
| `/api/traffic/timeline` | GET | 24-hour congestion curve + HTML chart |
| `/api/traffic/best-time` | GET | Best departure window for a given hour |
| `/api/traffic` | POST | Submit a traffic observation |
| `/api/environment` | POST | Submit crowd/weather observation |
| `/api/traffic/heatmap` | GET | Spatial congestion grid data |
| `/api/isochrone` | GET | Reachable area polygons (HTML Folium map) |

---

## Configuration (Environment Variables)

Traffic patterns are configured via `settings.peak_hours` (list of `(start_h, end_h, factor)` tuples):

```python
# Default peak hours
peak_hours = [
    (6, 8, 1.8),    # morning rush
    (11, 13, 1.3),  # lunch
    (17, 19, 2.0),  # evening rush
    (21, 23, 1.1),  # late night
]
```

Other relevant settings:

| Variable | Default | Description |
|---|---|---|
| `ROUTE_CONGESTION_WEIGHT` | `0.2` | Congestion weight in route scoring |
| `ROUTE_CROWD_WEIGHT` | `0.1` | Crowd level weight in route scoring |
| `ROUTE_WEATHER_WEIGHT` | `0.05` | Weather severity weight in route scoring |

---

## How to Test

### Get 24-hour traffic curve

```bash
curl "http://192.168.1.217:8000/api/traffic/timeline?weekday=1"
```

### Get best departure time for 17:00

```bash
curl "http://192.168.1.217:8000/api/traffic/best-time?hour=17&weekday=1"
```

### Submit a traffic observation

```bash
curl -X POST http://192.168.1.217:8000/api/traffic \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 10.9085,
    "lon": 106.760,
    "hour": 17,
    "weekday": 1,
    "congestion": 0.85,
    "speed_kmh": 8.0
  }'
```

### Get spatial heatmap data

```bash
curl http://192.168.1.217:8000/api/traffic/heatmap
```

### Generate isochrone map

```
http://192.168.1.217:8000/api/isochrone?lat=10.9085&lon=106.760&minutes=5,10,15
```
Opens in browser as a Folium map with green/yellow/red polygons.

### Submit environment observation

```bash
curl -X POST http://192.168.1.217:8000/api/environment \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 10.9085, "lon": 106.760,
    "hour": 12, "weekday": 3,
    "crowd_level": 0.7,
    "weather_severity": 0.2,
    "notes": "Trưa đông người, nắng nhẹ"
  }'
```

---

## Healthy Output Examples

**Traffic timeline:**
```json
{
  "curve": [
    {"hour": 0, "congestion": 0.05, "label": "00:00", "status": "thông thoáng"},
    {"hour": 7, "congestion": 0.85, "label": "07:00", "status": "tắc nghẽn"},
    {"hour": 17, "congestion": 0.90, "label": "17:00", "status": "tắc nghẽn"}
  ],
  "html": "<div style=...>...</div>"
}
```

**Best departure time:**
```json
{
  "recommended_hour": 9,
  "congestion": 0.35,
  "status": "bình thường",
  "save_minutes": 14
}
```

**Heatmap data:**
```json
{
  "data": [
    {"lat": 10.908, "lon": 106.760, "intensity": 0.85},
    {"lat": 10.910, "lon": 106.762, "intensity": 0.45}
  ]
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `503 "Router not ready"` on isochrone | OSM graph not loaded | Wait for startup; check `GET /api/status` for `osm_graph_cached: true` |
| Empty `curve` | `refresh()` failed | Check DB connectivity; verify `traffic_observations` table exists |
| Isochrone returns tiny polygon | High congestion → very small reachable area | Expected behaviour; try `depart_hour=3` for off-peak |
| `congestion` always 0 | No observations in DB | Submit observations via `POST /api/traffic` |
| Heatmap shows wrong area | Grid cells use raw lat/lon, not projected | Expected; cells are ~220 m at equator, ~190 m at 10°N |

---

## Performance Notes

- **`refresh()`**: Queries DB for all `(hour, weekday)` pairs and last 7 days of spatial data. ~5–20 ms for typical DB sizes.
- **`congestion_at()`**: Pure in-memory lookup after refresh. ~0.01 ms.
- **`full_day_curve()`**: 24 calls to `congestion_at()`. ~0.5 ms.
- **`IsochroneGenerator.generate()`**: Dijkstra on full OSM graph. ~200 ms–2 s depending on graph size and cutoff time. Runs in the async event loop — consider offloading to a thread pool for large graphs.
- **`CongestionGrid`**: Memory usage is proportional to the number of unique grid cells. For a 10 km × 10 km area at 0.002° resolution: ~2500 cells × ~50 bytes = ~125 KB.
- Traffic data is refreshed every 5 minutes automatically. Force refresh after each new observation via `force=True`.
