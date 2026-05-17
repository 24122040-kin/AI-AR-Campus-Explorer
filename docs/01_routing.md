# 01 — Routing Engine

## Overview

The routing module is the navigation backbone of LocalNavBot. It computes optimal turn-by-turn routes between two GPS coordinates, taking into account real-time traffic congestion, crowd levels, weather severity, road type preferences, and user-defined custom edges (alleys, shortcuts). It supports two backends:

- **Primary**: Valhalla (self-hosted Docker container) — full-featured, supports costing options, alternates, and Vietnamese directions.
- **Fallback**: `osmnx` + NetworkX A* — pure-Python, works fully offline, uses a custom multi-profile scoring system.

The module also provides geometry utilities for map-matching, a maneuver plan builder for the AR renderer, and rich HTML/Folium output for the web UI.

---

## Architecture / Data Flow

```
User Request (POST /api/route)
        │
        ▼
  NavRouter.find_route()
        │
        ├─── Valhalla healthy? ──YES──► ValhallaClient.route()
        │                                    │
        │                                    ▼
        │                             Parse Valhalla JSON
        │                             → Route dataclass
        │
        └─── NO ──────────────────► SmartOSMNXRouter.route()
                                          │
                                    ┌─────┴──────┐
                                    │ 3 profiles │
                                    │ balanced   │
                                    │ local_frd  │
                                    │ fast_main  │
                                    └─────┬──────┘
                                          │ A* per profile
                                          ▼
                                   TrafficHeuristic.edge_weight()
                                          │
                                   Score & rank candidates
                                          │
                                          ▼
                                    Best Route + alternates
                                          │
                                          ▼
                              _attach_images_to_steps()
                              build_maneuver_plan()
                              build_ar_path()
                              render_route_html()
                              render_route_map()
```

---

## Key Classes and Functions

### `routing/router.py`

#### `RouteStep` (dataclass)
Represents one navigation step.

| Field | Type | Description |
|---|---|---|
| `instruction` | `str` | Vietnamese turn instruction |
| `distance_m` | `float` | Distance of this step in metres |
| `duration_s` | `float` | Estimated travel time in seconds |
| `lat`, `lon` | `float` | Anchor coordinate of the step |
| `bearing` | `float` | Compass bearing at this step (0–360°) |
| `street_name` | `str` | Road name |
| `maneuver` | `str` | `"turn_left"` \| `"turn_right"` \| `"straight"` \| `"arrive"` \| … |
| `image_paths` | `list[str]` | Paths to illustrative photos near this step |

#### `Route` (dataclass)
Complete route result.

| Field | Type | Description |
|---|---|---|
| `steps` | `list[RouteStep]` | Ordered turn-by-turn steps |
| `total_distance_m` | `float` | Total route length in metres |
| `total_duration_s` | `float` | Total estimated travel time in seconds |
| `geometry` | `list[tuple[float,float]]` | Full polyline as `[(lat, lon), …]` |
| `origin` / `destination` | `tuple[float,float]` | Start and end coordinates |
| `depart_time` | `datetime` | Departure timestamp used for congestion scoring |
| `via_pois` | `list[dict]` | POIs along the route |
| `analysis` | `dict` | Scoring breakdown (congestion, crowd, profile, score) |

#### `RouteProfile` (frozen dataclass)
Defines a routing personality.

| Field | Default | Effect |
|---|---|---|
| `local_bias` | `1.0` | `< 1.0` → prefer local roads |
| `highway_bias` | `1.0` | `> 1.0` → avoid highways |
| `turn_bias` | `1.0` | `> 1.0` → penalise turns more |

Three built-in profiles: `balanced`, `local_friendly`, `fast_main`.

---

#### `TrafficHeuristic`

```python
class TrafficHeuristic:
    async def warm_cache(weekday: int | None) -> None
    def congestion_factor(depart_time: datetime) -> float
    def edge_weight(base_time_s, depart_time, road_type, is_custom_local,
                    lat, lon, local_bias, highway_bias) -> float
```

`congestion_factor` returns a multiplier `[1.0, 2.0]` — the maximum of:
- DB-observed average congestion for the hour (scaled from `[0,1]` → `[1,2]`)
- Static peak-hour schedule from `settings.peak_hours`

`edge_weight` combines:
```
weight = base_time × (TIME_W + CONGESTION_W × congestion_factor)
       × (1 + CROWD_W × env_penalty)
       × (1 + WEATHER_W × env_penalty)
       × local_road_bonus  (if custom local edge)
       × highway_penalty   (if motorway/trunk/primary)
       + base_time × DISTANCE_W × 0.1
```

---

#### `OSMGraph`

```python
class OSMGraph:
    def load() -> nx.MultiDiGraph
    async def patch_custom_edges() -> None
    def add_travel_times(speed_kph: float = 30.0) -> None
```

- Downloads OSM graph for `settings.osm_area` via `osmnx.graph_from_place()` and caches as GraphML.
- `patch_custom_edges()` reads all rows from `custom_edges` DB table and injects them as NetworkX edges. Edges farther than `settings.custom_edge_snap_max_m` (default 80 m) from the nearest OSM node are skipped.
- `add_travel_times()` estimates `travel_time` for edges missing it, using `length / (speed_kph / 3.6)`.

---

#### `ValhallaClient`

```python
class ValhallaClient:
    async def is_healthy() -> bool
    async def route(origin_lat, origin_lon, dest_lat, dest_lon,
                    depart_time, costing, extra_costing,
                    via_waypoints, alternates) -> dict | None
```

- `is_healthy()` does a `GET /status` with 3-second timeout.
- `route()` posts to `POST /route` with `date_time.type=1` (depart at) and `directions_options.language="vi-VI"`.
- If `alternates > 0` and the request fails, retries without alternates.
- Returns raw Valhalla JSON or `None` on error.

---

#### `OSMNXRouter`

```python
class OSMNXRouter:
    def _weighted_graph(depart_time) -> nx.MultiDiGraph
    async def route(origin_lat, origin_lon, dest_lat, dest_lon,
                    depart_time) -> Route | None
```

Runs `nx.astar_path()` with `weight="_weight"` (set by `TrafficHeuristic.edge_weight`). Heuristic function uses haversine distance divided by estimated speed. Maneuver detection uses bearing difference between consecutive edges:
- `diff < 45° or > 315°` → `straight`
- `45°–135°` → `turn_right`
- `135°–225°` → `u_turn`
- `225°–315°` → `turn_left`

---

#### `SmartOSMNXRouter` (extends `OSMNXRouter`)

```python
class SmartOSMNXRouter:
    async def route(origin_lat, origin_lon, dest_lat, dest_lon,
                    depart_time, avoid_discs, alternates_count) -> Route | None
```

Runs A* for each of the 3 `RouteProfile` instances simultaneously. Each candidate route is scored:

```
route_score = TIME_W × duration_min
            + CONGESTION_W × (avg_congestion × 25)
            + CROWD_W × (avg_crowd × 20)
            + WEATHER_W × (avg_weather × 20)
            + COMPLEXITY_W × (turn_ratio × 50)
            - LANDMARK_W × (landmark_density × 5)
            - LOCALITY_W × (custom_edge_ratio × 15)
            + HIGHWAY_PENALTY × highway_ratio
```

The lowest-scoring route wins. Alternate routes are deduplicated by geometry fingerprint (distance + midpoint + endpoint).

`avoid_discs` is a list of `(lat, lon, radius_m)` — edges whose midpoint falls inside a disc get their weight multiplied by `settings.route_avoid_disc_penalty` (default 5×).

---

#### `_vn_instruction(maneuver, street, dist) -> str`

Maps maneuver codes to Vietnamese turn instructions:

| Maneuver | Output |
|---|---|
| `turn_left` | `"Rẽ trái vào {street}, đi {dist}"` |
| `turn_right` | `"Rẽ phải vào {street}, đi {dist}"` |
| `straight` | `"Đi thẳng trên {street}, đi {dist}"` |
| `arrive` | `"Đã đến nơi — {street}"` |
| `u_turn` | `"Quay đầu xe"` |
| `roundabout_enter` | `"Vào vòng xuyến, ra lối {street}"` |

---

### `routing/maneuver_plan.py`

#### `build_maneuver_plan(route: Route) -> list[dict]`

Converts a `Route` into a list of bearing-annotated maneuver dicts for the AR renderer.

**Input**: `Route` object  
**Output**: list of dicts, one per step:

```json
{
  "maneuver_id": 2,
  "instruction": "Rẽ phải vào Đường Lê Lợi, đi 150 m",
  "maneuver": "turn_right",
  "anchor_lat": 10.9085,
  "anchor_lon": 106.760,
  "distance_m": 150.0,
  "duration_s": 18.0,
  "bearing_before": 45.0,
  "bearing_after": 135.0,
  "street_name": "Đường Lê Lợi",
  "instruction_priority": "high"
}
```

`instruction_priority` is `"high"` for any maneuver that is not `straight` or `depart`.

---

### `routing/geo_utils.py`

#### `haversine_m(lat1, lon1, lat2, lon2) -> float`
Great-circle distance in metres using the Haversine formula. Accurate to within ~0.5% for distances under 1000 km.

#### `distance_point_to_segment_m(lat, lon, lat1, lon1, lat2, lon2) -> float`
Shortest perpendicular distance from point P to line segment A–B, using local equirectangular projection centred at the segment midpoint. Safe for segments under ~50 km.

#### `distance_point_to_polyline_m(lat, lon, polyline) -> float`
Minimum distance from P to any segment of a polyline. Iterates all segments and returns the minimum.

#### `snap_point_to_polyline(lat, lon, polyline) -> tuple[float, float, float, int]`
Projects a GPS point onto the closest point on the polyline.  
Returns `(snap_lat, snap_lon, residual_distance_m, segment_index)`.  
Used for map-matching and step-advance logic in `NavSession.update_gps()`.

#### `distance_for_navigation(raw_lat, raw_lon, polyline, use_snap) -> tuple[float, tuple | None]`
Convenience wrapper: returns `(cross_track_m, (snap_lat, snap_lon))`. Used by the session manager to decide when to advance to the next step.

---

### `routing/route_renderer.py`

#### `render_route_html(route, analyzer, show_images, compact) -> str`
Returns a self-contained HTML fragment (no external dependencies) with:
- Dark header showing distance, duration, departure time, congestion status badge
- Traffic advisory if congestion > 0.5 (recommends better departure time)
- "Why this route" panel with profile, congestion, landmark density, turn count
- Turn-by-turn steps with emoji icons and inline base64 photos
- Footer with totals

#### `render_route_map(route, analyzer) -> str`
Returns Folium map HTML with:
- Animated `AntPath` polyline in indigo
- Green start / red end markers
- Circle markers at each step with popup showing instruction + photo
- Traffic `HeatMap` overlay from `analyzer.heatmap_data()`
- Layer control

#### `render_traffic_timeline(analyzer, weekday) -> str`
Returns a 24-bar HTML chart. Each bar height is proportional to congestion (0–100%). The current hour bar has a blue outline. Color coding: green → yellow → orange → red.

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `VALHALLA_URL` | `http://localhost:8002` | Valhalla routing engine URL |
| `VALHALLA_TIMEOUT` | `10` | HTTP timeout in seconds |
| `USE_OSMNX_FALLBACK` | `true` | Enable osmnx A* when Valhalla is down |
| `OSM_AUTO_DOWNLOAD` | `false` | Download OSM graph if cache missing |
| `OSM_AREA` | `"Dĩ An, Bình Dương, Vietnam"` | Area for OSM graph download |
| `OSM_NETWORK_TYPE` | `drive` | `drive` \| `walk` \| `bike` \| `all` |
| `CUSTOM_EDGE_SNAP_MAX_M` | `80.0` | Max snap distance for custom edges |
| `ROUTE_AVOID_DISC_PENALTY` | `5.0` | Weight multiplier for avoided zones |
| `ROUTE_ALTERNATES_MAX` | `2` | Max alternate routes to return |
| `ROUTE_CANDIDATE_PROFILES` | `3` | Number of A* profiles to run |
| `LOCAL_ROAD_BONUS` | `0.85` | Weight multiplier for local roads |
| `HIGHWAY_PENALTY` | `1.2` | Weight multiplier for highways |
| `ROUTE_TIME_WEIGHT` | `0.5` | Scoring weight for travel time |
| `ROUTE_CONGESTION_WEIGHT` | `0.2` | Scoring weight for congestion |
| `ROUTE_COMPLEXITY_WEIGHT` | `0.08` | Scoring weight for turn complexity |
| `ROUTE_LANDMARK_WEIGHT` | `0.07` | Scoring bonus for landmark density |

---

## How to Test

### Find a route (POST)

```bash
curl -s -X POST http://192.168.1.217:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "chợ Dĩ An",
    "origin_lat": 10.9085,
    "origin_lon": 106.760,
    "depart_hour": 8,
    "alternates": 1
  }' | python -m json.tool
```

### View route as map (GET)

```
http://192.168.1.217:8000/api/route/map?from_q=nhà+tôi&to_q=chợ+Dĩ+An
```

### Check Valhalla health

```bash
curl http://192.168.1.217:8002/status
```

### Test with avoid zone

```bash
curl -X POST http://192.168.1.217:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "trường học",
    "origin_lat": 10.9085,
    "origin_lon": 106.760,
    "avoid_discs": [{"lat": 10.909, "lon": 106.761, "radius_m": 200}]
  }'
```

---

## Healthy Output Example

```json
{
  "ok": true,
  "distance_km": 2.34,
  "duration_min": 8.5,
  "steps": [
    {
      "instruction": "Xuất phát từ Đường Lê Lợi, đi 120 m",
      "distance_m": 120.0,
      "duration_s": 14.4,
      "lat": 10.9085,
      "lon": 106.760,
      "maneuver": "depart"
    },
    {
      "instruction": "Rẽ phải vào Đường Nguyễn Trãi, đi 800 m",
      "distance_m": 800.0,
      "duration_s": 96.0,
      "maneuver": "turn_right"
    }
  ],
  "geometry": [[10.9085, 106.760], [10.9090, 106.761], "..."],
  "ar_path": {"point_count": 18, "points": [...]},
  "html_card": "<div style=...>...</div>",
  "analysis": {
    "strategy": "multi_profile_offline_rerank",
    "selected_profile": "local_friendly",
    "route_score": 12.4,
    "avg_congestion": 0.45,
    "landmark_density": 0.3
  }
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `503 "Router not ready"` | `NavRouter.init()` not complete | Wait for startup; check logs for OSM download errors |
| `404 "No route found"` | Origin/destination outside OSM graph, or graph disconnected | Verify coordinates are within `OSM_AREA`; try `osm_network_type=all` |
| `"No cached OSM graph found"` | `OSM_AUTO_DOWNLOAD=false` and no `.graphml` file | Set `OSM_AUTO_DOWNLOAD=true` or run `python main.py index` |
| Empty `geometry` array | Valhalla returned a route with no shape | Check Valhalla container logs; fall back to osmnx |
| `"snap too far"` for custom edge | Custom edge endpoints far from OSM nodes | Increase `CUSTOM_EDGE_SNAP_MAX_M` or move edge endpoints closer to roads |
| Valhalla `503` | Container not running | `docker-compose up valhalla` |

---

## Performance Notes

- **OSM graph load**: ~2–5 s from GraphML cache; ~30–120 s for first download.
- **A* per profile**: ~50–300 ms for a 5 km urban route on CPU.
- **Valhalla**: ~20–80 ms per route request (Docker on same host).
- **`patch_custom_edges()`**: ~1 ms per edge; called once at startup and after each `POST /api/edge`.
- **`render_route_html()`** with images: ~100–500 ms if PIL resizing is needed; ~5 ms without images.
- The `SmartOSMNXRouter` runs 3 A* searches sequentially. For very large graphs (>500k nodes), consider reducing `ROUTE_CANDIDATE_PROFILES` to 1 or 2.
