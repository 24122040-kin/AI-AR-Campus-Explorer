# 08 — Indoor Routing

## Overview

The indoor routing module provides multi-floor navigation inside buildings using GeoJSON floor plans. It builds an in-memory directed graph from uploaded floor plans, then runs A* with a floor-penalty heuristic to find the shortest-time path between any two nodes, including transitions via stairs and elevators.

---

## Architecture / Data Flow

```
GeoJSON Upload (POST /api/indoor/map)
        │
        ├── Parse FeatureCollection
        │       ├── Point features → IndoorNode
        │       └── LineString features → IndoorEdge
        │
        ├── db.upsert_floor_map()     — store GeoJSON in SQLite
        ├── db.upsert_floor_nodes()   — denormalise nodes for spatial queries
        └── IndoorBuildingRegistry.load_geojson()  — in-memory graph

Indoor Route Request (POST /api/indoor/route)
        │
        ├── Load building from registry (or DB if not cached)
        │
        ├── Resolve origin
        │       ├── node_id (exact)
        │       ├── lat/lon + floor → nearest_node()
        │       └── name + floor → find_node_by_name()
        │
        ├── Resolve destination (same options)
        │
        └── IndoorRouter.route()
                │
                └── A* with floor-penalty heuristic
                        ├── corridor: dist / 1.2 m/s
                        ├── door: 3 s + dist / 1.2 m/s
                        ├── stairs: 20 s × |floor_delta|
                        └── elevator: 30 s + 5 s × |floor_delta|
```

---

## GeoJSON Schema

Floor plans are uploaded as GeoJSON `FeatureCollection` objects. Each feature is either a **node** (Point) or an **edge** (LineString).

### Node Feature (Point)
```json
{
  "type": "Feature",
  "id": "node_101",
  "geometry": {
    "type": "Point",
    "coordinates": [106.760, 10.9085]
  },
  "properties": {
    "node_type": "room",
    "name": "Phòng 101",
    "floor": 1,
    "accessible": true
  }
}
```

**`node_type` values**: `room` | `corridor` | `stairs` | `elevator` | `entrance` | `exit`

### Edge Feature (LineString)
```json
{
  "type": "Feature",
  "id": "edge_101_102",
  "geometry": {
    "type": "LineString",
    "coordinates": [[106.760, 10.9085], [106.761, 10.9086]]
  },
  "properties": {
    "edge_type": "corridor",
    "from_node": "node_101",
    "to_node": "node_102",
    "from_floor": 1,
    "to_floor": 1,
    "distance_m": 12.5,
    "bidirectional": true,
    "accessible": true
  }
}
```

**`edge_type` values**: `corridor` | `stairs` | `elevator` | `door`

**Multi-floor edges**: Set `from_floor` ≠ `to_floor` for stairs/elevator edges that connect floors.

---

## Key Classes and Functions

### `core/indoor_router.py`

#### Cost Constants

| Constant | Value | Description |
|---|---|---|
| `WALK_SPEED_MS` | `1.2 m/s` | Indoor walking speed |
| `STAIR_TIME_PER_FLOOR` | `20 s` | Time per floor via stairs |
| `ELEVATOR_WAIT_S` | `30 s` | Average elevator wait time |
| `ELEVATOR_TIME_PER_FLOOR` | `5 s` | Travel time per floor in elevator |
| `DOOR_PENALTY_S` | `3 s` | Time to open/pass a door |

#### `IndoorNode` (dataclass)
```python
@dataclass
class IndoorNode:
    node_id: str
    building_id: str
    floor: int
    name: str
    node_type: str      # room | corridor | stairs | elevator | entrance | exit
    lat: float
    lon: float
    accessible: bool = True
    properties: dict = {}
```

#### `IndoorEdge` (dataclass)
```python
@dataclass
class IndoorEdge:
    edge_id: str
    from_node: str
    to_node: str
    from_floor: int
    to_floor: int
    edge_type: str      # corridor | stairs | elevator | door
    distance_m: float
    bidirectional: bool = True
    accessible: bool = True
    cost_s: float       # auto-computed in __post_init__
```

**Cost computation** (`_edge_cost`):
```python
if edge_type == "stairs":
    cost = STAIR_TIME_PER_FLOOR × max(1, |to_floor - from_floor|)
elif edge_type == "elevator":
    cost = ELEVATOR_WAIT_S + ELEVATOR_TIME_PER_FLOOR × max(1, |to_floor - from_floor|)
elif edge_type == "door":
    cost = DOOR_PENALTY_S + distance_m / WALK_SPEED_MS
else:  # corridor
    cost = distance_m / WALK_SPEED_MS
```

---

#### `IndoorGraph`

```python
class IndoorGraph:
    def add_node(node: IndoorNode) -> None
    def add_edge(edge: IndoorEdge) -> None
    def load_geojson(geojson: dict) -> None
    def nodes_on_floor(floor: int) -> list[IndoorNode]
    def nearest_node(lat, lon, floor, node_types) -> IndoorNode | None
    def find_node_by_name(name, floor) -> IndoorNode | None
```

**`load_geojson`**: Parses a `FeatureCollection`. Point features → `add_node()`. LineString features → `add_edge()`. Edge distance is taken from `properties.distance_m` or computed from polyline length if missing.

**`nearest_node`**: Finds the closest node by Euclidean distance in lat/lon degrees. Optionally filtered by floor and/or node_type list.

**`find_node_by_name`**: Case-insensitive substring match on `node.name`. Optionally filtered by floor.

**Bidirectional edges**: When `bidirectional=True`, a reverse edge is automatically added with the same cost (stairs and elevators are symmetric).

---

#### `IndoorRouter`

```python
class IndoorRouter:
    def route(origin_node_id, dest_node_id, *,
              prefer_accessible, prefer_elevator) -> IndoorRoute | None
    def _heuristic(node_id, dest) -> float
    def _reconstruct(came_from, dest_id) -> IndoorRoute
```

**A* algorithm**:
- Priority queue: `(f_cost, g_cost, node_id)`.
- `f_cost = g_cost + heuristic`.
- Stale entries are skipped by checking `g > g_cost[current]`.

**Heuristic** (admissible):
```
h = Euclidean_distance_m / WALK_SPEED_MS + |floor_delta| × STAIR_TIME_PER_FLOOR
```
This never overestimates the true cost, ensuring A* finds the optimal path.

**`prefer_accessible`**: Skips edges with `accessible=False`.

**`prefer_elevator`**: Multiplies stair edge costs by 3× to strongly prefer elevators.

---

#### `IndoorRoute` (dataclass)

```python
@dataclass
class IndoorRoute:
    steps: list[IndoorRouteStep]
    total_distance_m: float
    total_duration_s: float
    floors_visited: list[int]
    building_id: str
    origin_node: str
    destination_node: str

    def as_dict() -> dict   # includes html_card
```

**`as_dict()`** includes `html_card` — a dark-themed HTML card with floor badges and step list.

#### `IndoorRouteStep` (dataclass)
```python
@dataclass
class IndoorRouteStep:
    instruction: str      # Vietnamese
    from_node_id: str
    to_node_id: str
    from_floor: int
    to_floor: int
    edge_type: str
    distance_m: float
    duration_s: float
    lat: float
    lon: float
```

**Vietnamese instructions** (`_vn_indoor_instruction`):
| Edge type | Direction | Instruction |
|---|---|---|
| stairs | up | `"Đi lên cầu thang {name} → Tầng {floor}"` |
| stairs | down | `"Đi xuống cầu thang {name} → Tầng {floor}"` |
| elevator | up | `"Đi thang máy lên Tầng {floor}"` |
| elevator | down | `"Đi thang máy xuống Tầng {floor}"` |
| door | — | `"Qua cửa vào {to_node.name}"` |
| corridor | — | `"Đi theo hành lang — {dist} m đến {to_node.name}"` |

---

#### `IndoorBuildingRegistry` (singleton: `indoor_registry`)

```python
class IndoorBuildingRegistry:
    def load_geojson(building_id, geojson) -> IndoorGraph
    def get(building_id) -> IndoorGraph | None
    def list_buildings() -> list[str]
    def get_router(building_id) -> IndoorRouter | None
```

All buildings are pre-loaded from DB at app startup. New uploads immediately update the registry.

---

### `web/routes/indoor.py`

| Endpoint | Method | Description |
|---|---|---|
| `/api/indoor/map` | POST | Upload/replace a floor plan GeoJSON |
| `/api/indoor/buildings` | GET | List all buildings |
| `/api/indoor/map/{bid}` | GET | List floors for a building |
| `/api/indoor/map/{bid}/{floor}` | GET | Get raw GeoJSON for one floor |
| `/api/indoor/map/{bid}/{floor}` | DELETE | Remove a floor plan |
| `/api/indoor/route` | POST | Find indoor route |
| `/api/indoor/nodes` | GET | Nearby indoor nodes (GPS snap) |

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `INDOOR_GPS_ACCURACY_THRESHOLD_M` | `15.0` | Switch to indoor mode when GPS accuracy exceeds this |

---

## How to Test

### List buildings

```bash
curl http://192.168.1.217:8000/api/indoor/buildings
```

### Upload a floor plan

```bash
curl -X POST http://192.168.1.217:8000/api/indoor/map \
  -F "file=@floor1.geojson" \
  -F "building_id=main_building" \
  -F "floor=1" \
  -F "name=Tầng 1 — Tòa nhà chính"
```

### Find indoor route by node name

```bash
curl -X POST http://192.168.1.217:8000/api/indoor/route \
  -H "Content-Type: application/json" \
  -d '{
    "building_id": "main_building",
    "origin_lat": 10.9085,
    "origin_lon": 106.760,
    "origin_floor": 1,
    "dest_name": "Phòng 302",
    "prefer_elevator": false
  }'
```

### Find indoor route by node ID

```bash
curl -X POST http://192.168.1.217:8000/api/indoor/route \
  -H "Content-Type: application/json" \
  -d '{
    "building_id": "main_building",
    "origin_node": "node_entrance_1",
    "dest_node": "node_room_302"
  }'
```

### Get nearby indoor nodes

```bash
curl "http://192.168.1.217:8000/api/indoor/nodes?lat=10.9085&lon=106.760&floor=1"
```

### Delete a floor plan

```bash
curl -X DELETE http://192.168.1.217:8000/api/indoor/map/main_building/1
```

---

## Healthy Output Examples

**Route response:**
```json
{
  "ok": true,
  "building_id": "main_building",
  "origin_node": "node_entrance_1",
  "destination_node": "node_room_302",
  "total_distance_m": 85.5,
  "total_duration_s": 112.3,
  "total_duration_min": 1.87,
  "floors_visited": [1, 2, 3],
  "summary": "Tầng 1 → Cầu thang A → Tầng 3 → Phòng 302",
  "steps": [
    {
      "instruction": "Đi theo hành lang — 20 m đến Cầu thang A",
      "from_floor": 1, "to_floor": 1,
      "edge_type": "corridor",
      "distance_m": 20.0, "duration_s": 16.7
    },
    {
      "instruction": "Đi lên cầu thang Cầu thang A → Tầng 3",
      "from_floor": 1, "to_floor": 3,
      "edge_type": "stairs",
      "distance_m": 0.0, "duration_s": 40.0
    },
    {
      "instruction": "Đi theo hành lang — 65 m đến Phòng 302",
      "from_floor": 3, "to_floor": 3,
      "edge_type": "corridor",
      "distance_m": 65.5, "duration_s": 54.6
    }
  ],
  "html_card": "<div style='background:#1e293b;...'>...</div>"
}
```

**Buildings list:**
```json
{
  "ok": true,
  "buildings": [
    {
      "building_id": "main_building",
      "floor_count": 5,
      "min_floor": 1,
      "max_floor": 5,
      "lat": 10.9085,
      "lon": 106.760
    }
  ]
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `404 "No floor maps loaded"` | Building not uploaded or not in registry | Upload GeoJSON via `POST /api/indoor/map` |
| `404 "Cannot find indoor node named '...'"` | Node name not in graph | Check node names in GeoJSON; use `GET /api/indoor/nodes` to browse |
| `404 "No indoor route found"` | Graph is disconnected | Ensure all floors are connected via stairs/elevator edges |
| `400 "GeoJSON must be a FeatureCollection"` | Wrong GeoJSON type | Wrap features in `{"type": "FeatureCollection", "features": [...]}` |
| `400 "Provide origin_node or origin_lat/lon"` | Missing origin specification | Add `origin_node` or `origin_lat`/`origin_lon` + `origin_floor` |
| Route ignores elevator | `prefer_elevator=false` (default) | Set `prefer_elevator: true` in request |
| Stair cost too high | `STAIR_TIME_PER_FLOOR` default 20 s | Adjust constant in `core/indoor_router.py` |

---

## Performance Notes

- **Graph load from DB**: ~5–50 ms per floor (JSON parsing + node/edge insertion).
- **A* routing**: ~1–10 ms for typical buildings (< 500 nodes, < 5 floors).
- **Memory**: ~1 KB per node + edge. A 5-floor building with 200 nodes uses ~200 KB.
- **Pre-loading at startup**: All buildings are loaded from DB into `indoor_registry` during the startup event. New uploads update the registry immediately.
- The `nearest_node()` function is O(N) — for buildings with thousands of nodes, consider a spatial index (e.g., k-d tree) for production use.
- GeoJSON files up to 10 MB are accepted. For very large buildings, split into separate uploads per floor.
