# 07 — Data Management & Database

## Overview

The data module handles all persistent storage, image ingestion, and computer vision analysis. The database is SQLite with WAL mode for concurrent async access. Images are processed through a quality/deduplication pipeline before being stored and indexed into VPR. The experimental endpoints expose YOLO and EasyOCR for on-demand scene analysis.

---

## Architecture / Data Flow

```
Image Upload (POST /api/upload/image)
        │
        ├── validate_upload() — extension + MIME check
        ├── read_gps_exif()   — extract lat/lon from EXIF
        ├── db.add_location() — create location if needed
        ├── db.add_image()    — store image record
        └── VPREngine.index_image() — add to FAISS index

Batch Import (POST /api/upload/batch)
        │
        └── BatchImageImporter.import_folder()
                ├── read_gps_exif()     — skip if no GPS
                ├── assess_quality()    — skip if score < min_quality
                ├── phash()             — skip if duplicate (hamming ≤ 8)
                ├── _cluster()          — group images within 15 m
                ├── db.add_location()   — one per cluster
                └── db.add_image()      — up to 4 best per cluster

Scene Analysis (POST /api/experimental/scene)
        │
        ├── LandmarkDetector.detect()  — YOLOv8n
        └── OCRReader.detect()         — EasyOCR en+vi

Database Schema (SQLite WAL)
        ├── locations          — GPS-tagged places
        ├── images             — photos linked to locations
        ├── pois               — custom points of interest
        ├── custom_edges       — shortcuts/alleys for routing
        ├── traffic_observations
        ├── environmental_observations
        ├── nav_sessions
        ├── floor_maps         — GeoJSON floor plans
        └── floor_nodes        — denormalised node table
```

---

## Key Classes and Functions

### `core/database.py`

#### Schema Overview

| Table | Purpose | Key Columns |
|---|---|---|
| `locations` | GPS-tagged places | `id, name, lat, lon, importance(1-5), category, tags(JSON)` |
| `images` | Photos per location | `id, location_id, filepath, caption, bearing, faiss_index_id` |
| `pois` | Custom POIs | `id, name, type, lat, lon, address, is_active` |
| `custom_edges` | Routing shortcuts | `id, from_lat/lon, to_lat/lon, distance_m, travel_time_s, road_type, is_bidirectional` |
| `traffic_observations` | Crowdsourced congestion | `lat, lon, hour(0-23), weekday(0-6), speed_kmh, congestion(0-1)` |
| `environmental_observations` | Crowd/weather data | `lat, lon, hour, weekday, crowd_level, weather_severity` |
| `nav_sessions` | Route history | `origin/dest lat/lon, route_json, total_distance_m` |
| `floor_maps` | Indoor GeoJSON | `building_id, floor, geojson(TEXT), lat_center, lon_center` |
| `floor_nodes` | Indoor nodes | `building_id, floor, node_id, name, node_type, lat, lon, accessible` |

**Indexes**: `idx_locations_lat_lon`, `idx_pois_lat_lon`, `idx_floor_nodes_lat_lon`, `idx_traffic_hour_weekday`, `idx_floor_maps_building_floor` (UNIQUE).

**WAL mode**: Enabled via `PRAGMA journal_mode=WAL` — allows concurrent reads while writing.

#### `Database` class

```python
class Database:
    async def init() -> None                    # create tables + indexes
    async def execute(sql, params) -> int       # returns lastrowid
    async def fetchall(sql, params) -> list[dict]
    async def fetchone(sql, params) -> dict | None

    # Locations
    async def add_location(name, lat, lon, description, category, importance, tags, osm_node_id) -> int
    async def get_location(loc_id) -> dict | None
    async def nearby_locations(lat, lon, radius_deg=0.01) -> list[dict]
    async def search_locations(query) -> list[dict]

    # Images
    async def add_image(location_id, filename, filepath, caption, bearing, faiss_index_id) -> int
    async def get_images_for_location(location_id) -> list[dict]
    async def update_faiss_id(image_id, faiss_id) -> None

    # POIs
    async def add_poi(name, poi_type, lat, lon, address, notes, location_id) -> int
    async def search_pois(query) -> list[dict]
    async def nearby_pois(lat, lon, radius_deg=0.01) -> list[dict]

    # Custom edges
    async def add_custom_edge(from_lat, from_lon, to_lat, to_lon, name, road_type, bidirectional) -> int
    async def get_all_custom_edges() -> list[dict]

    # Traffic
    async def add_traffic_obs(lat, lon, hour, weekday, speed_kmh, congestion) -> int
    async def avg_congestion(hour, weekday) -> float

    # Indoor
    async def upsert_floor_map(building_id, floor, name, geojson, lat_center, lon_center) -> int
    async def get_floor_map(building_id, floor) -> dict | None
    async def list_floor_maps(building_id) -> list[dict]
    async def list_buildings() -> list[dict]
    async def upsert_floor_nodes(building_id, floor, nodes) -> int
    async def nearby_floor_nodes(lat, lon, radius_deg, floor) -> list[dict]
```

**`nearby_locations`**: Bounding-box query using `lat BETWEEN lat-r AND lat+r AND lon BETWEEN lon-r AND lon+r`. Sorts by `dy² + dx²` (approximate Euclidean in metres). Returns up to 50 results.

**`add_custom_edge`**: Auto-computes `distance_m` from haversine and `travel_time_s` from distance / speed (20 km/h for alleys, 30 km/h for shortcuts).

---

### `web/routes/data.py`

#### `POST /api/upload/image`

```
Content-Type: multipart/form-data
Fields:
  file          (required) — image file (.jpg/.png/.webp)
  location_id   (optional) — link to existing location
  location_name (optional) — create new location with this name
  lat, lon      (optional) — GPS override (uses EXIF if not provided)
  caption       (optional) — text description
  category      (optional) — "general" | "cafe" | "landmark" | ...
  importance    (optional) — 1–5
  auto_caption  (optional) — bool, use LLM to generate caption
```

Returns:
```json
{
  "ok": true,
  "image_id": 42,
  "location_id": 15,
  "faiss_id": 41,
  "lat": 10.9085,
  "lon": 106.760,
  "caption": "Cổng chính chợ Dĩ An"
}
```

Raises `400 "GPS required"` if neither EXIF nor form fields provide coordinates.

#### `POST /api/upload/batch`

Starts a background job to import all images from a folder. The folder must be inside `data/` for safety.

```bash
curl -X POST http://192.168.1.217:8000/api/upload/batch \
  -F "folder=/path/to/local_nav_bot/data/images/my_photos" \
  -F "auto_caption=false" \
  -F "min_quality=0.3"
```

#### `POST /api/location`
```json
{"name": "Chợ Dĩ An", "lat": 10.9085, "lon": 106.760, "category": "market", "importance": 4}
```

#### `POST /api/poi`
```json
{"name": "Quán cà phê Hoa", "poi_type": "cafe", "lat": 10.9090, "lon": 106.761}
```

#### `POST /api/edge`
```json
{
  "from_lat": 10.9085, "from_lon": 106.760,
  "to_lat": 10.9090, "to_lon": 106.761,
  "name": "Hẻm 12", "road_type": "alley", "bidirectional": true
}
```
After adding, immediately calls `OSMGraph.patch_custom_edges()` to inject into the live routing graph.

---

### `web/routes/experimental.py`

#### `POST /api/experimental/landmarks`

Runs YOLOv8n on an uploaded image. Returns detections with labels, confidence, and bounding boxes.

```bash
curl -X POST http://192.168.1.217:8000/api/experimental/landmarks \
  -F "file=@photo.jpg" \
  -F "conf=0.3"
```

Response:
```json
{
  "ok": true,
  "available": true,
  "model": "data/yolo/yolov8n.pt",
  "detections": [
    {"label": "car", "confidence": 0.87, "bbox": [100.0, 200.0, 300.0, 400.0]},
    {"label": "person", "confidence": 0.72, "bbox": [50.0, 100.0, 150.0, 350.0]}
  ],
  "preview_url": "/api/detections/photo_detect.jpg"
}
```

#### `POST /api/experimental/ocr`

Runs EasyOCR (en+vi) on an uploaded image.

```bash
curl -X POST http://192.168.1.217:8000/api/experimental/ocr \
  -F "file=@photo.jpg"
```

Response:
```json
{
  "ok": true,
  "available": true,
  "backend": "easyocr",
  "languages": ["en", "vi"],
  "blocks": [
    {"text": "Chợ Dĩ An", "confidence": 0.94, "bbox": [[10,20],[200,20],[200,50],[10,50]]}
  ],
  "preview_url": "/api/detections/photo_ocr.jpg"
}
```

#### `POST /api/experimental/scene`

Runs both YOLO and OCR and returns a combined summary.

---

### `core/landmark_detector.py`

#### `LandmarkDetector`

```python
class LandmarkDetector:
    def __init__(model_name, output_dir)
    def detect(image_path, conf, save_preview) -> LandmarkDetectionResult
    @property available: bool
    @property model_name: str
```

- Loads `yolov8n.pt` from `data/yolo/` (local cache) or downloads from Ultralytics.
- `available = False` if `ultralytics` is not installed — all methods return empty results gracefully.
- `save_preview=True` saves an annotated JPEG to `data/detections/`.

---

### `core/ocr_reader.py`

#### `OCRReader`

```python
class OCRReader:
    def __init__(languages, output_dir)
    def detect(image_path, min_conf, save_preview) -> OCRResult
    @property available: bool
    @property backend: str
    @property init_error: str | None
```

- Initialises `easyocr.Reader(["en", "vi"], gpu=True/False)` on construction.
- Models are cached in `data/ocr_models/` (CRAFT + recognition models).
- Converts image to grayscale before OCR for better accuracy.
- `available = False` if `easyocr` is not installed.
- `save_preview=True` draws bounding boxes and text on the image.

---

### `core/image_manager.py`

#### `read_gps_exif(path) -> tuple[float, float, float | None] | None`
Returns `(lat, lon, bearing)` from EXIF GPS tags, or `None` if not present.

#### `assess_quality(img) -> float`
Quality score `[0.0, 1.0]`:
- Laplacian variance → sharpness (weight 0.55)
- Mean brightness deviation from 0.45 → exposure (weight 0.45)
- Returns 0.1 for images < 0.1 MP

#### `phash(img, size=16) -> str`
Mean-hash: resize to 16×16 grayscale, compare each pixel to mean. Returns 64-char hex string.

#### `hamming_distance(h1, h2) -> int`
Bit-level difference between two hex hashes. ≤ 8 bits → considered duplicate.

#### `BatchImageImporter`

```python
class BatchImageImporter:
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    MIN_QUALITY = 0.25
    DEDUP_THRESHOLD = 8      # hamming distance
    CLUSTER_RADIUS_M = 15    # metres

    async def import_folder(folder: Path) -> dict
```

**Pipeline**:
1. Scan folder recursively for image files.
2. For each image: read GPS EXIF → skip if missing.
3. Assess quality → skip if `score < min_quality`.
4. Compute perceptual hash → skip if duplicate (hamming ≤ 8).
5. Cluster images within 15 m → one location per cluster.
6. Check if location already exists in DB (within 0.00015° ≈ 16 m).
7. Save up to 4 best-quality images per cluster.
8. Optionally auto-caption via LLM.

Returns summary:
```json
{
  "total_files": 150,
  "processed": 120,
  "skipped_no_gps": 20,
  "skipped_quality": 8,
  "skipped_dup": 2,
  "locations_created": 35,
  "images_added": 98
}
```

---

### `web/uploads.py`

```python
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

def validate_upload(file: UploadFile) -> str          # returns suffix or raises 400
def build_upload_path(suffix: str) -> Path            # timestamped path in images_dir
def ensure_safe_batch_folder(folder: str) -> Path     # must be inside data/
```

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `data/navbot.db` | SQLite database path |
| `IMAGES_DIR` | `data/images` | Image storage directory |
| `DETECTIONS_DIR` | `data/detections` | YOLO/OCR preview output |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model file |
| `YOLO_CONFIDENCE` | `0.25` | Default detection confidence threshold |
| `OCR_BACKEND` | `easyocr` | OCR backend (only easyocr supported) |
| `OCR_LANGUAGES` | `en,vi` | Comma-separated language codes |
| `OCR_CONFIDENCE` | `0.35` | Minimum OCR confidence threshold |
| `OCR_MODELS_DIR` | `data/ocr_models` | EasyOCR model cache directory |

---

## How to Test

### Add a location

```bash
curl -X POST http://192.168.1.217:8000/api/location \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Location", "lat": 10.9085, "lon": 106.760}'
```

### Find nearby locations

```bash
curl "http://192.168.1.217:8000/api/nearby?lat=10.9085&lon=106.760"
```

### Search locations

```bash
curl "http://192.168.1.217:8000/api/search?q=chợ"
```

### Detect landmarks in image

```bash
curl -X POST http://192.168.1.217:8000/api/experimental/landmarks \
  -F "file=@photo.jpg"
```

### Run full scene analysis

```bash
curl -X POST http://192.168.1.217:8000/api/experimental/scene \
  -F "file=@photo.jpg"
```

### Upload image with GPS

```bash
curl -X POST http://192.168.1.217:8000/api/upload/image \
  -F "file=@photo.jpg" \
  -F "lat=10.9085" \
  -F "lon=106.760" \
  -F "caption=Cổng chính chợ Dĩ An"
```

---

## Healthy Output Examples

**Location added:**
```json
{"ok": true, "id": 42}
```

**Nearby locations:**
```json
{
  "locations": [
    {"id": 42, "name": "Chợ Dĩ An", "lat": 10.9085, "lon": 106.760, "importance": 4}
  ],
  "pois": []
}
```

**YOLO detections:**
```json
{
  "ok": true,
  "available": true,
  "detections": [{"label": "car", "confidence": 0.87, "bbox": [100, 200, 300, 400]}]
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `400 "GPS required"` | No EXIF GPS and no lat/lon form fields | Add GPS to image EXIF or provide `lat`/`lon` form fields |
| `400 "Unsupported image extension"` | File is not .jpg/.png/.webp | Convert image format |
| `YOLO available: false` | `ultralytics` not installed | `pip install ultralytics` |
| `OCR available: false` | `easyocr` not installed | `pip install easyocr` |
| `400 "Batch import folder must be inside data/"` | Security check failed | Use a path under `data/` |
| `faiss_id: -1` | VPR vocabulary not built | Run `POST /api/vpr/rebuild` after uploading images |
| Duplicate images imported | `phash` threshold too high | Lower `DEDUP_THRESHOLD` in `BatchImageImporter` |

---

## Performance Notes

- **Single image upload**: ~10–50 ms (DB + VPR index if vocabulary exists).
- **Batch import**: ~0.5–2 s per image (quality assessment + dedup + DB). For 500 images: ~5–15 min.
- **YOLO inference**: ~30–100 ms on GPU, ~200–800 ms on CPU.
- **EasyOCR**: ~50–200 ms on GPU, ~500 ms–2 s on CPU. First call is slower due to model loading.
- **`nearby_locations`**: ~1–5 ms for typical DB sizes (< 10,000 locations).
- **`search_locations`**: Full-table LIKE scan — ~5–50 ms. Add FTS5 virtual table for large datasets.
- SQLite WAL mode allows concurrent reads during writes. For high-throughput scenarios, consider PostgreSQL with PostGIS.
