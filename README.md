# LocalNavBot v2 — ĐHKHTN Campus 2

Indoor navigation assistant for **HCMUS Campus 2, Linh Trung, Thu Duc**.

Combines GPS routing, Visual Place Recognition (VPS), AR navigation, floor detection, and an LLM chat bot — all running locally on your machine, accessible from any phone on the same network.

---

## Features

| Module | What it does |
|--------|-------------|
| **Route finding** | OSMnx A* with traffic heuristics; Valhalla optional |
| **AR Navigation** | Camera overlay with 3D arrows + voice guidance |
| **VPS (Visual Place Recognition)** | DINOv2 + VLAD + FAISS — identify location from photo |
| **Floor detection** | Barometer + GPS altitude + accelerometer + manual calibration |
| **LLM chat bot** | Ollama / Anthropic / OpenAI — ask directions in Vietnamese |
| **Local map editor** | Leaflet map — add locations, draw paths, walk-track routes |
| **Traffic analysis** | Crowd-sourced congestion heatmap + best departure time |
| **Indoor routing** | Multi-floor A* with stairs/elevator cost model |
| **Campus scope** | All search/geocoding scoped to HCMUS CS2 polygon |

---

## Quick Start

### 1. Environment

```powershell
# Enable fast solver (once)
conda install -n base -c conda-forge conda-libmamba-solver -y
conda config --set solver libmamba
conda config --set channel_priority strict

# Create env
conda env create -f environment.yml
conda activate localnavbot
```

### 2. Install CUDA-enabled PyTorch (required for DINOv2 VPR)

The `environment.yml` installs CPU torch by default on some systems. Force CUDA:

```powershell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

Verify:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1+cu121 True
```

### 3. Configure `.env`

```powershell
Copy-Item .env.example .env
```

Minimum required settings:

```ini
# GPU
DEVICE=cuda
TORCH_DTYPE=float16

# LLM — pick one:
LLM_PROVIDER=ollama
LLM_MODEL=llava:7b
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1

# Map area (must match existing OSM cache or will re-download)
OSM_AREA=Binh Duong, Vietnam
MAP_DEFAULT_LAT=10.8700
MAP_DEFAULT_LON=106.8030

# VPR — vitb14 fits in 6 GB VRAM
VPR_MODEL=dinov2_vitb14
VPR_BACKEND=dinov2

# Routing
OSM_AUTO_DOWNLOAD=true
ALLOW_REMOTE_GEOCODING=true
```

### 4. Start Ollama (if using local LLM)

```powershell
# In a separate terminal — keep running
ollama serve
```

Recommended models (pick one based on VRAM):

| Model | VRAM | Capability |
|-------|------|-----------|
| `llava:7b` | ~5 GB | Chat + image recognition ✅ |
| `qwen2.5:7b-instruct` | ~5 GB | Chat only, faster |
| `qwen2.5:3b-instruct` | ~2 GB | Lightweight, chat only |

```powershell
ollama pull llava:7b
```

### 5. Start the server

```powershell
conda activate localnavbot
python main.py serve
```

Expected startup log (clean):
```
INFO  Loading cached OSM graph from data/osm_cache/osm_94c5d114def0.graphml
INFO  Loading dinov2_vitb14 on cuda…
INFO  DINOv2 feature dim = 768
INFO  LocalNavBot v2 ready
INFO  Uvicorn running on http://0.0.0.0:8000
```

---

## Accessing on Phone

### Option A — LAN (Android Chrome, basic features)

```powershell
ipconfig   # find "IPv4 Address" under Wi-Fi, e.g. 192.168.1.46
```

Open: `http://192.168.1.46:8000`

GPS, camera, microphone are **blocked on iOS Safari over HTTP**. Use Option B.

### Option B — ngrok HTTPS (full features, recommended)

```powershell
# Install once
winget install ngrok.ngrok
ngrok update   # must be >= 3.20.0

# Register free account at https://dashboard.ngrok.com/signup
# Then save your token:
ngrok authtoken YOUR_TOKEN

# Every session (in a second terminal while server runs):
ngrok http 8000
```

Get current URL anytime:
```powershell
(Invoke-RestMethod http://localhost:4040/api/tunnels).tunnels[0].public_url
```

| Feature | LAN HTTP | ngrok HTTPS |
|---------|----------|-------------|
| Chat / routing | ✅ | ✅ |
| GPS | ❌ iOS | ✅ |
| Camera / VPS | ❌ iOS | ✅ |
| Microphone / voice | ❌ iOS | ✅ |
| AR DeviceMotion | ❌ iOS | ✅ |
| Floor detection | ❌ iOS | ✅ |

> Free ngrok URL changes on every restart. Paid plan gives a fixed subdomain.

---

## Using the App

### Add locations (required for VPS and named routing)

**Via CLI (fastest):**
```powershell
python main.py add-location
# Interactive wizard: name, lat, lon, floor, category
```

**Via web UI:**
Tab ➕ → "Thêm địa điểm + ảnh" → fill name, floor, upload 1–5 photos → Add

**Via API:**
```powershell
curl -X POST http://localhost:8000/api/location `
  -H "Content-Type: application/json" `
  -d '{"name":"Toa B","lat":10.870,"lon":106.803,"floor":2,"category":"classroom"}'
```

### Build VPS index (after uploading photos)

Tab 📊 → **Rebuild VPR** — or:
```powershell
curl -X POST http://localhost:8000/api/vpr/rebuild
```

Check status: `GET /api/status` → `vpr_ready: true`, `vpr_index_size > 0`

### Find a route

Tab 🧭 → type destination → **Tìm đường**

Or chat: `Tìm đường đến thư viện`

### Identify your location with camera (VPS)

Tab 🧭 → click 📷 next to "Từ" → choose camera or gallery → system matches photo against database → fills location name + updates floor.

### AR Navigation

1. Find a route first
2. Click **"Bật AR Cam"** (appears after route is found)
3. Allow camera + DeviceMotion
4. Walk — 3D arrows + voice guidance follow the route

AR modes:
- **WebXR** (Android Chrome with ARCore): full 3D overlay
- **Compass 2D** (iOS / most Android): compass with route dots

### Floor detection

The floor HUD (`🏢 Tầng ?`) is always visible. Sources used in priority order:

| Source | Availability | Accuracy |
|--------|-------------|---------|
| Manual calibration | Always | Exact |
| VPS match | When photo taken | Exact |
| Barometer | High-end Android | ±0.5 floor |
| GPS altitude | Most phones | ±1–2 floors |
| Step detector | All phones | Relative |

Tap the floor HUD to manually set your current floor.

### Local map editor

Tab ➕ → **"Mở Local Map Editor"**

- **View mode**: see all locations colour-coded by floor
- **Add path mode**: click point A → click point B → name it → save
- **Walk tracking**: tap 🚶 → walk the actual path → stop → app simplifies and saves the real curve

---

## CLI Commands

```powershell
python main.py serve                    # start server
python main.py status                   # system health check
python main.py add-location             # interactive location wizard
python main.py index data\images\       # batch index GPS-tagged photos
python main.py demo "Đi đến thư viện"  # test bot query
```

---

## API Reference

### Core
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System health, VPR index size, OSM cache |
| `/api/ai/readiness` | GET | LLM + VPR + routing readiness + risk score |
| `/api/campus/boundary` | GET | Campus polygon + bbox for map rendering |

### Navigation
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/route` | POST | Find route (origin+dest text or lat/lon) |
| `/api/gps` | POST | Update GPS fix for a session |
| `/api/route/map` | GET | Folium map HTML of a route |
| `/api/isochrone` | GET | Reachable area in N minutes |

### Chat
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Single-turn chat |
| `/api/chat/image` | POST | Chat with image (VPS + LLM) |
| `/api/chat/stream` | POST | SSE streaming response |
| `/ws/chat` | WS | Bidirectional streaming chat |

### Data
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/location` | POST | Add location (name, lat, lon, floor, category) |
| `/api/location/images` | POST | Upload 1–5 images for a location |
| `/api/edge` | POST | Add path (road_type, slope, covered, surface) |
| `/api/upload/image` | POST | Quick upload with EXIF GPS |
| `/api/search` | GET | Semantic search, top 10, campus-scoped |
| `/api/nearby` | GET | Nearby locations/POIs |
| `/api/localmap` | GET | Interactive Leaflet map editor |
| `/api/locations/all` | GET | All locations with floor info |
| `/api/edges/all` | GET | All custom paths with geometry |

### VPR
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vpr/query` | POST | Match image against index |
| `/api/vpr/rebuild` | POST | Rebuild FAISS index (background job) |

### Realtime
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/realtime/frame` | POST | Ingest camera frame (YOLO + OCR + VPR) |
| `/api/realtime/sensors` | POST | Compass, accel, pressure |
| `/api/realtime/floor` | POST | Floor update from barometer + accel |
| `/api/realtime/floor/calibrate` | POST | Manual floor calibration |
| `/api/realtime/vio/imu` | POST | High-rate IMU for dead-reckoning |
| `/ws/realtime/{sid}` | WS | Bidirectional realtime state stream |

### Traffic
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/traffic/timeline` | GET | 24h congestion curve |
| `/api/traffic/best-time` | GET | Best departure window |
| `/api/traffic` | POST | Report congestion observation |
| `/api/traffic/heatmap` | GET | Spatial congestion grid |

### Indoor
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/indoor/map` | POST | Upload GeoJSON floor plan |
| `/api/indoor/route` | POST | Multi-floor A* routing |
| `/api/indoor/buildings` | GET | List buildings with floor plans |

---

## Project Layout

```
local_nav_bot/
├── main.py                  # entry point
├── .env                     # your config (not committed)
├── .env.example             # template
├── environment.yml          # conda env
├── requirements.txt         # pip deps
├── config/
│   └── settings.py          # all settings via pydantic-settings
├── core/
│   ├── database.py          # SQLite async (locations, edges, images, traffic)
│   ├── vpr_engine.py        # DINOv2 + VLAD + FAISS
│   ├── floor_detector.py    # barometer + step + elevator detection
│   ├── vio_fusion.py        # EKF dead-reckoning (IMU + optical flow)
│   ├── sensor_fusion.py     # GPS + IMU + VIO fusion
│   ├── traffic_analyzer.py  # congestion heuristics + isochrone
│   ├── realtime_manager.py  # frame ingestion pipeline
│   ├── scene_fusion.py      # YOLO + OCR + VPR fusion
│   ├── alert_engine.py      # proactive alerts (turn, floor, drift)
│   ├── indoor_router.py     # multi-floor A*
│   ├── campus_scope.py      # campus polygon filter
│   ├── geo_ar.py            # WGS84 → ENU coordinate transform
│   └── route_projection.py  # route geometry → AR path
├── routing/
│   ├── router.py            # NavRouter (Valhalla + osmnx fallback)
│   ├── maneuver_plan.py     # bearing-annotated maneuver list for AR
│   ├── geo_utils.py         # haversine, snap-to-polyline, map-matching
│   └── route_renderer.py    # HTML card + Folium map + traffic chart
├── bot/
│   ├── nav_bot.py           # LLM client + intent parsing + route bot
│   ├── session_manager.py   # GPS state machine + live rerouting
│   └── realtime_navigator.py # instruction builder from scene state
├── web/
│   ├── app.py               # FastAPI app + startup
│   ├── ui.html              # single-page mobile web app
│   ├── ar_renderer.js       # Three.js WebXR + compass 2D AR
│   ├── vio_client.js        # JS-side VIO (IMU loop + optical flow)
│   ├── routes/              # API routers (chat, data, navigation, ...)
│   └── static/js/           # frontend modules
│       ├── globals.js       # shared state, fetchWithTimeout, toast
│       ├── gps.js           # watchPosition + heading
│       ├── ar.js            # AR toggle + hazard warnings
│       ├── vps.js           # VPS photo picker + location identification
│       ├── floor.js         # floor detection + GPS altitude + calibration
│       ├── vio.js           # VIO client integration
│       ├── route.js         # route finding + semantic search
│       ├── data.js          # add location/edge/images
│       ├── chat.js          # sendMsg + stop bot
│       ├── speech.js        # STT (Web Speech + Whisper) + TTS
│       ├── traffic.js       # traffic chart + reporting
│       └── layout.js        # accordion modules + help system
└── docs/                    # per-module technical documentation
    ├── 01_routing.md
    ├── 02_vpr.md
    ├── 03_floor_vio.md
    ├── 04_traffic.md
    ├── 05_realtime.md
    ├── 06_bot_chat.md
    ├── 07_data_db.md
    ├── 08_indoor.md
    ├── 09_speech_ar.md
    ├── 10_frontend_mobile.md
    └── 11_config_deploy.md
```

---

## Troubleshooting

### `OSM graph not ready` on startup

```
WARNING  OSM graph not ready yet: No cached OSM graph found.
```

Cause: `OSM_AREA` in `.env` changed — the MD5 hash no longer matches the cached `.graphml` file.

Fix: either set `OSM_AUTO_DOWNLOAD=true` (re-downloads) or rename the cache file:
```powershell
# Check what area the cache was built for:
python -c "
import hashlib
areas = ['Binh Duong, Vietnam']
for a in areas:
    print(hashlib.md5(a.encode()).hexdigest()[:12], a)
"
# Then check data/osm_cache/ for matching filename
```

### `Torch not compiled with CUDA enabled`

```
WARNING  DINOv2 unavailable, falling back to ORB: Torch not compiled with CUDA enabled
```

Cause: CPU-only torch installed. Fix:
```powershell
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### `model requires more system memory` (Ollama OOM)

`llama3.2-vision:11b` needs 10+ GB RAM. Switch to a smaller model:
```ini
# .env
LLM_MODEL=llava:7b          # 5 GB VRAM — recommended
# or
LLM_MODEL=qwen2.5:3b-instruct  # 2 GB VRAM — chat only
```

### Port 8000 already in use

```powershell
# Find and kill the process holding port 8000
$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -First 1
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force }
```

### Custom edges "snap too far"

```
WARNING  Skip custom edge id=1: snap too far (396 m > 80 m)
```

Edges added while testing with approximate coordinates won't snap to the OSM graph. Add edges with accurate GPS coordinates (from the Local Map Editor or Google Maps).

### VPR always returns 0 matches

`vpr_ready: false` means the FAISS index is empty. Steps:
1. Upload photos via Tab ➕ → "Thêm địa điểm + ảnh"
2. Tab 📊 → **Rebuild VPR**
3. Check `GET /api/status` → `vpr_index_size > 0`

### GPS / Camera not working on phone

Requires HTTPS. Use ngrok (see above). On iOS, also requires a user gesture before DeviceMotion fires.

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | `cuda` / `cpu` / `mps` |
| `TORCH_DTYPE` | `float16` | `float16` / `float32` |
| `LLM_PROVIDER` | `ollama` | `ollama` / `anthropic` / `openai` |
| `LLM_MODEL` | `llava:7b` | Model name |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama base URL |
| `OSM_AREA` | `Binh Duong, Vietnam` | Area for OSM graph download |
| `OSM_AUTO_DOWNLOAD` | `true` | Download OSM if cache missing |
| `ALLOW_REMOTE_GEOCODING` | `true` | Use Nominatim for unknown place names |
| `VPR_MODEL` | `dinov2_vitb14` | `vitb14` (350 MB) / `vitl14` / `vitg14` (2.5 GB) |
| `VPR_BACKEND` | `dinov2` | `dinov2` / `orb` (CPU fallback) |
| `MAP_DEFAULT_LAT` | `10.8700` | Campus 2 centre latitude |
| `MAP_DEFAULT_LON` | `106.8030` | Campus 2 centre longitude |
| `CAMPUS_BOUNDARY_ENABLED` | `true` | Scope search to campus polygon |
| `VALHALLA_URL` | `http://localhost:8002` | Optional Valhalla routing engine |
| `CORS_ORIGINS` | `*` | Allowed origins (use `*` for LAN access) |

Full reference: `config/settings.py` and `docs/11_config_deploy.md`.

---

## Recent Updates

### ✅ Phase 4: Enhanced AR Navigation (May 2026)

**3D Stair Arrows with Improved Visibility**

- **Larger arrows**: 2.2x-2.5x scale for better visibility on mobile
- **Correct direction**: Arrows now point in the exact direction of travel using `atan2`
- **Persistent display**: 150px screen buffer prevents arrows from disappearing at edges
- **3D stair arrows**: Special animated arrows for floor transitions with:
  - Bounce animation (8px vertical movement)
  - Floor labels ("Tầng 2", "Tầng 3")
  - Distance indicators
  - Orange gradient to distinguish from normal navigation
- **Auto-detection**: Automatically shows stair arrows within 30m of floor transitions

See `AR_STAIRS_PHASE4_COMPLETE.md` for full technical details.

### ✅ Indoor Routing Fix (May 2026)

- Fixed 0 km bug when routing between indoor locations
- Indoor graph now builds from database on startup
- Smart detection when both origin and destination are in database
- Correct multi-floor routing with stairs/elevators

See `ROUTING_FIX_SUMMARY.md` for details.

### ✅ Duplicate Road Detection (May 2026)

- System now detects duplicate edges before creating new ones
- User can choose to replace old edge or create new one
- Tolerance: ~1m for duplicate detection

See `DUPLICATE_ROAD_DETECTION.md` for details.

---

## Documentation

For detailed technical documentation, see:
- **Project Overview**: `PROJECT_OVERVIEW.md` - Complete system architecture and features
- **Module Docs**: `docs/` folder - Per-module technical documentation
- **Phase Reports**: `AR_STAIRS_PHASE*.md` - Implementation phase details
- **Deployment**: `PHASE3_DEPLOYMENT_GUIDE.md` - Production deployment guide
