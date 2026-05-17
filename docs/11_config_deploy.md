# 11 — Configuration & Deployment

## Overview

LocalNavBot is configured entirely through environment variables (loaded from `.env` via `pydantic-settings`). It can be run directly with Conda/pip or containerised with Docker Compose. The startup sequence initialises all subsystems in dependency order.

---

## Configuration Reference (`config/settings.py`)

All settings are defined in the `Settings` class (Pydantic `BaseSettings`). They can be overridden by environment variables (case-insensitive) or a `.env` file in the project root.

### App & Server

| Variable | Type | Default | Description |
|---|---|---|---|
| `APP_NAME` | str | `LocalNavBot` | Application name |
| `DEBUG` | bool | `false` | Enable debug mode |
| `HOST` | str | `0.0.0.0` | Bind address |
| `PORT` | int | `8000` | Listen port |
| `CORS_ORIGINS` | str | `*` | Comma-separated allowed origins. `*` = allow all |

### GPU / Compute

| Variable | Type | Default | Description |
|---|---|---|---|
| `DEVICE` | str | `cuda` | `cuda` \| `cpu` \| `mps` |
| `GPU_ID` | int | `0` | CUDA device index |
| `TORCH_DTYPE` | str | `float16` | `float16` \| `float32` \| `bfloat16` |

### Paths

| Variable | Type | Default | Description |
|---|---|---|---|
| `DATA_DIR` | Path | `data/` | Root data directory |
| `IMAGES_DIR` | Path | `data/images/` | Uploaded image storage |
| `DETECTIONS_DIR` | Path | `data/detections/` | YOLO/OCR preview output |
| `REALTIME_FRAMES_DIR` | Path | `data/realtime_frames/` | Camera frame storage |
| `YOLO_CONFIG_DIR` | Path | `data/yolo/` | YOLO model cache |
| `OCR_MODELS_DIR` | Path | `data/ocr_models/` | EasyOCR model cache |
| `DB_PATH` | Path | `data/navbot.db` | SQLite database |
| `FAISS_INDEX_PATH` | Path | `data/vpr_index.faiss` | FAISS vector index |
| `FAISS_META_PATH` | Path | `data/vpr_meta.json` | FAISS metadata |
| `OSM_CACHE_DIR` | Path | `data/osm_cache/` | OSM GraphML cache |

### VPR / AnyLoc

| Variable | Type | Default | Description |
|---|---|---|---|
| `VPR_MODEL` | str | `dinov2_vitg14` | DINOv2 variant |
| `VPR_BACKEND` | str | `auto` | `auto` \| `dinov2` \| `orb` |
| `VPR_LAYER` | int | `31` | Transformer block for feature extraction |
| `VPR_FACET` | str | `value` | Feature facet |
| `VPR_NUM_CLUSTERS` | int | `32` | VLAD vocabulary size |
| `VPR_TOP_K` | int | `5` | Number of VPR matches to return |

### Routing

| Variable | Type | Default | Description |
|---|---|---|---|
| `VALHALLA_URL` | str | `http://localhost:8002` | Valhalla engine URL |
| `VALHALLA_TIMEOUT` | int | `10` | Valhalla HTTP timeout (s) |
| `USE_OSMNX_FALLBACK` | bool | `true` | Use osmnx A* when Valhalla is down |
| `OSM_AUTO_DOWNLOAD` | bool | `false` | Auto-download OSM graph if missing |
| `OSM_AREA` | str | `Dĩ An, Bình Dương, Vietnam` | Area for OSM graph |
| `OSM_NETWORK_TYPE` | str | `drive` | `drive` \| `walk` \| `bike` \| `all` |
| `ALLOW_REMOTE_GEOCODING` | bool | `false` | Allow Nominatim geocoding |
| `CUSTOM_EDGE_SNAP_MAX_M` | float | `80.0` | Max snap distance for custom edges |
| `ROUTE_ALTERNATES_MAX` | int | `2` | Max alternate routes |
| `ROUTE_CANDIDATE_PROFILES` | int | `3` | Number of A* profiles |
| `LOCAL_ROAD_BONUS` | float | `0.85` | Weight multiplier for local roads |
| `HIGHWAY_PENALTY` | float | `1.2` | Weight multiplier for highways |

### Traffic / Heuristic Weights

| Variable | Type | Default | Description |
|---|---|---|---|
| `ROUTE_DISTANCE_WEIGHT` | float | `0.15` | Distance component weight |
| `ROUTE_TIME_WEIGHT` | float | `0.5` | Travel time weight |
| `ROUTE_CONGESTION_WEIGHT` | float | `0.2` | Congestion weight |
| `ROUTE_CROWD_WEIGHT` | float | `0.1` | Crowd level weight |
| `ROUTE_WEATHER_WEIGHT` | float | `0.05` | Weather severity weight |
| `ROUTE_COMPLEXITY_WEIGHT` | float | `0.08` | Turn complexity weight |
| `ROUTE_LANDMARK_WEIGHT` | float | `0.07` | Landmark density bonus |
| `ROUTE_LOCALITY_WEIGHT` | float | `0.06` | Local road bonus |
| `ROUTE_TURN_PENALTY_TURN` | float | `12.0` | Seconds added per turn |
| `ROUTE_TURN_PENALTY_UTURN` | float | `30.0` | Seconds added per U-turn |

### LLM / Bot

| Variable | Type | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | str | `anthropic` | `anthropic` \| `openai` \| `ollama` |
| `LLM_MODEL` | str | `claude-sonnet-4-20250514` | Model name |
| `LLM_API_KEY` | str | `""` | API key |
| `LLM_BASE_URL` | str | `""` | Base URL (for Ollama: `http://localhost:11434/v1`) |
| `LLM_MAX_TOKENS` | int | `1024` | Max tokens per response |
| `LLM_TEMPERATURE` | float | `0.2` | Response temperature |
| `LLM_TIMEOUT_SECONDS` | int | `45` | LLM request timeout |
| `CHAT_MAX_CHARS` | int | `4000` | Max message length |

### Realtime / Sensors

| Variable | Type | Default | Description |
|---|---|---|---|
| `REALTIME_ENABLED` | bool | `true` | Enable realtime endpoints |
| `REALTIME_FRAME_INTERVAL_MS` | int | `400` | Target frame interval |
| `REALTIME_YOLO_FPS` | float | `3.0` | Max YOLO inference rate |
| `REALTIME_OCR_INTERVAL_MS` | int | `1500` | Min OCR interval |
| `REALTIME_VPR_INTERVAL_MS` | int | `2500` | Min VPR interval |
| `REALTIME_FRAME_MAX_MB` | int | `8` | Max frame upload size |
| `SENSOR_FUSION_MODE` | str | `complementary` | `raw` \| `complementary` \| `kalman_lite` |
| `FUSION_HEADING_ALPHA` | float | `0.82` | Heading blend factor |
| `FUSION_POSITION_ALPHA` | float | `0.7` | GPS position blend factor |

### VIO

| Variable | Type | Default | Description |
|---|---|---|---|
| `VIO_VPR_MIN_SCORE` | float | `0.72` | Min VPR score for re-localization |
| `VIO_DRIFT_TRIGGER_M` | float | `2.0` | Drift threshold for VPR trigger |
| `VIO_FLOW_PX_PER_M` | float | `554.0` | Optical flow calibration |

### Speech / Whisper

| Variable | Type | Default | Description |
|---|---|---|---|
| `WHISPER_MODEL` | str | `base` | `tiny` \| `base` \| `small` \| `medium` \| `large` |
| `WHISPER_DEVICE` | str | `""` | Override device for Whisper |

### Map Display

| Variable | Type | Default | Description |
|---|---|---|---|
| `MAP_DEFAULT_LAT` | float | `10.9085` | Default map centre latitude |
| `MAP_DEFAULT_LON` | float | `106.7600` | Default map centre longitude |
| `MAP_DEFAULT_ZOOM` | int | `15` | Default map zoom level |

### Indoor

| Variable | Type | Default | Description |
|---|---|---|---|
| `INDOOR_GPS_ACCURACY_THRESHOLD_M` | float | `15.0` | Switch to indoor mode threshold |

---

## `.env.example` (Annotated)

```dotenv
# Device
DEVICE=cuda              # cuda | cpu | mps
GPU_ID=0
TORCH_DTYPE=float16

# LLM — choose one provider:
# Anthropic (cloud):
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
LLM_API_KEY=your_anthropic_api_key_here

# OpenAI:
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o
# LLM_API_KEY=your_openai_api_key_here

# Ollama (local, fully offline):
# LLM_PROVIDER=ollama
# LLM_MODEL=llama3.2-vision:11b
# LLM_API_KEY=ollama
# LLM_BASE_URL=http://localhost:11434/v1

# Routing
VALHALLA_URL=http://localhost:8002
USE_OSMNX_FALLBACK=true

# Map area — change to your local area
OSM_AREA=Dĩ An, Bình Dương, Vietnam
MAP_DEFAULT_LAT=10.9085
MAP_DEFAULT_LON=106.7600

# VPR
VPR_MODEL=dinov2_vitg14
VPR_LAYER=31
VPR_FACET=value
VPR_NUM_CLUSTERS=32

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false
CORS_ORIGINS=*

# Geocoding
ALLOW_REMOTE_GEOCODING=false
```

---

## How to Run

### Conda environment (recommended)

```bash
# Create environment
conda env create -f environment.yml
conda activate localnavbot

# Copy and edit config
cp .env.example .env
# Edit .env: set LLM_API_KEY, DEVICE, OSM_AREA

# Start server
python main.py serve

# Or with options:
python main.py serve --host 0.0.0.0 --port 8000 --reload
```

### CLI commands

```bash
# Index images from a folder
python main.py index data/images/my_photos/

# Force rebuild VPR vocabulary
python main.py index data/images/ --rebuild

# Add a location interactively
python main.py add-location

# Run a demo query
python main.py demo "Đường nào đi từ Dĩ An đến chợ Bình Dương ít tắc nhất?"

# Check system status
python main.py status
```

### Docker Compose

```bash
# Build and start all services (Valhalla + NavBot)
docker-compose up -d

# View logs
docker-compose logs -f navbot

# Stop
docker-compose down
```

The `docker-compose.yml` starts:
1. **Valhalla** on port 8002 — downloads Vietnam OSM data on first run (~500 MB).
2. **NavBot** on port 8000 — depends on Valhalla being healthy.

GPU passthrough requires `nvidia-container-toolkit` installed on the host.

---

## Startup Sequence (`web/app.py` startup event)

```
@app.on_event("startup")
async def startup():
    1. settings.setup_dirs()          — create data/ subdirectories
    2. await db.init()                — create SQLite tables + indexes
    3. _router = NavRouter()          — create router instance
       await _router.init()           — load OSM graph, patch custom edges,
                                        warm traffic cache, check Valhalla
    4. _vpr = await _build_vpr_async() — load DINOv2 + VLAD vocab + FAISS index
                                         (in thread pool, non-blocking)
    5. _bot = NavBot(_router, _vpr)   — create bot instance
    6. _realtime_manager = RealtimeSessionManager(_router, _vpr)
    7. set_runtime_state(...)         — register all singletons in web/state.py
    8. session_manager.set_nav_router(_router)
    9. await traffic_analyzer.refresh(force=True)
   10. await environmental_analyzer.refresh(force=True)
   11. await session_manager.start()  — start session cleanup loop
   12. Pre-load all indoor floor maps from DB into indoor_registry
   13. logger.info("LocalNavBot v2 ready")
```

**Typical startup time**:
- DB init: ~0.1 s
- OSM graph load from cache: ~2–5 s
- VPR load (DINOv2 ViT-G/14 on GPU): ~5–15 s
- VPR load (ORB fallback): ~0.5 s
- Total: ~10–25 s on GPU, ~5–10 s on CPU with ORB

---

## Runtime State (`web/state.py`)

Module-level singletons set during startup:

```python
def set_runtime_state(*, router, vpr, bot, realtime_manager) -> None
def get_router() -> NavRouter | None
def get_vpr() -> VPREngine | None
def get_bot() -> NavBot | None
def get_realtime_manager() -> RealtimeSessionManager | None
```

All route handlers call `get_router()` etc. to access the live instances. Returns `None` if startup is not complete → handlers return `503`.

---

## Background Jobs (`web/jobs.py`)

In-memory job tracker for long-running operations (batch import, VPR rebuild).

```python
class JobStore:
    def create(job_type, message) -> JobRecord
    def update(job_id, *, status, message, result, error) -> JobRecord | None
    def get(job_id) -> JobRecord | None
    def list() -> list[dict]
```

**`JobRecord` fields**: `job_id`, `job_type`, `status` (`queued` → `running` → `completed` / `failed`), `message`, `created_at`, `updated_at`, `result`, `error`.

Jobs are stored in memory only — they are lost on server restart. Poll status via `GET /api/jobs/{job_id}`.

---

## Upload Validation (`web/uploads.py`)

```python
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

def validate_upload(file: UploadFile) -> str          # returns suffix or raises 400
def build_upload_path(suffix: str) -> Path            # timestamped path in images_dir
def ensure_safe_batch_folder(folder: str) -> Path     # must be inside data/
```

`ensure_safe_batch_folder` checks that the target path is inside `data/`, `data/images/`, or `data/navbot.db`'s parent. This prevents path traversal attacks.

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥2.1.0 | DINOv2 feature extraction, GPU compute |
| `torchvision` | ≥0.16.0 | Image transforms for VPR |
| `faiss-gpu` / `faiss-cpu` | ≥1.7.4 | Vector similarity search |
| `scikit-learn` | ≥1.4.2 | MiniBatchKMeans for VLAD vocabulary |
| `osmnx` | ≥1.9.0 | OSM graph download and manipulation |
| `networkx` | ≥3.2 | A* pathfinding on OSM graph |
| `folium` | ≥0.16.0 | Interactive map rendering |
| `aiosqlite` | ≥0.20.0 | Async SQLite access |
| `fastapi` | ≥0.111.0 | HTTP API framework |
| `uvicorn` | ≥0.29.0 | ASGI server |
| `websockets` | ≥12.0 | WebSocket support |
| `anthropic` | ≥0.28.0 | Anthropic Claude API client |
| `openai` | ≥1.25.0 | OpenAI / Ollama API client |
| `ultralytics` | ≥8.2.0 | YOLOv8 object detection (optional) |
| `easyocr` | ≥1.7.1 | OCR for Vietnamese text (optional) |
| `openai-whisper` | — | Speech transcription (optional, not in requirements.txt) |
| `Pillow` | ≥10.3.0 | Image processing |
| `opencv-contrib-python` | ≥4.13.0 | ORB features, image processing |
| `pydantic-settings` | ≥2.2.1 | Environment variable configuration |
| `loguru` | ≥0.7.2 | Structured logging |
| `typer` | ≥0.12.3 | CLI framework |
| `rich` | ≥13.7.1 | Terminal output formatting |

---

## Health Check Endpoints

### `GET /api/status`

```bash
curl http://192.168.1.217:8000/api/status | python -m json.tool
```

```json
{
  "status": "ok",
  "valhalla": true,
  "osm_graph_cached": true,
  "vpr_ready": true,
  "vpr_index_size": 150,
  "vpr_backend": "dinov2",
  "locations": 85,
  "pois": 12,
  "images": 150,
  "sessions": {"total": 2, "by_state": {"navigating": 1, "idle": 1}},
  "device": "cuda",
  "model": "dinov2_vitg14",
  "cors_origins": ["*"]
}
```

### `GET /api/ai/readiness`

Returns LLM and VPR readiness status.

---

## Common Deployment Issues

| Issue | Cause | Fix |
|---|---|---|
| `RuntimeError: No cached OSM graph` | `OSM_AUTO_DOWNLOAD=false` and no `.graphml` | Set `OSM_AUTO_DOWNLOAD=true` or run `python main.py index` |
| VPR loads ORB instead of DINOv2 | No GPU or PyTorch not installed | Install CUDA PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `503` on all endpoints | Startup not complete | Wait 10–30 s; check logs for errors |
| Valhalla `503` | Container not running | `docker-compose up valhalla` |
| `LLM_API_KEY` error | Missing or invalid key | Set `LLM_API_KEY` in `.env`; or use Ollama |
| Port 8000 in use | Another process | Change `PORT=8001` in `.env` |
| CORS error from phone | `CORS_ORIGINS` too restrictive | Set `CORS_ORIGINS=*` |
| GPU OOM during VPR | ViT-G/14 too large | Use `VPR_MODEL=dinov2_vitb14` or `VPR_BACKEND=orb` |
| Whisper not found | Package not installed | `pip install openai-whisper` |
| EasyOCR download fails | No internet | Pre-download models to `data/ocr_models/` |

---

## Performance Tuning

| Scenario | Recommendation |
|---|---|
| CPU-only server | `DEVICE=cpu`, `VPR_BACKEND=orb`, `WHISPER_MODEL=tiny` |
| Low VRAM (< 4 GB) | `VPR_MODEL=dinov2_vitb14`, `TORCH_DTYPE=float32` |
| High-traffic | Increase `uvicorn --workers 2` (note: VPR index is not shared between workers) |
| Slow routing | Reduce `ROUTE_CANDIDATE_PROFILES=1`, enable Valhalla |
| Large image collection | Use `faiss-gpu` for faster search |
| Offline deployment | `OSM_AUTO_DOWNLOAD=false`, pre-build graph, `LLM_PROVIDER=ollama` |
