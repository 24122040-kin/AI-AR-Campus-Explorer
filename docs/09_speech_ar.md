# 09 — Speech Transcription & Augmented Reality

## Overview

This module covers two distinct but complementary features:

1. **Speech transcription** — converts voice input (WebM/OGG/WAV audio from the browser) to text using OpenAI Whisper, with Vietnamese language optimisation.
2. **AR rendering** — transforms route geometry from WGS84 coordinates to a local ENU (East-North-Up) frame for Three.js WebXR rendering, with a compass-based 2D fallback for devices without WebXR support.

---

## Architecture / Data Flow

```
Speech Pipeline:
Browser (MediaRecorder) → audio/webm blob
        │
        ▼
POST /api/speech/transcribe
        │
        ├── Validate MIME type + size (max 10 MB)
        ├── Write to temp file
        ├── _load_whisper() — lazy load, cached in process
        │       └── whisper.load_model(settings.whisper_model, device=...)
        ├── loop.run_in_executor(None, model.transcribe(...))
        │       └── fp16=True (GPU), initial_prompt="Điều hướng..."
        ├── Unlink temp file
        └── Return {text, language, session_id}

AR Pipeline:
Route.geometry [(lat,lon), ...]
        │
        ▼
build_ar_path(route, ref_lat, ref_lon, ref_alt, min_spacing_m=8)
        │
        ├── route_to_local_frame(geometry, ref_lat, ref_lon)
        │       └── wgs84_to_enu(lat, lon, 0, ref_lat, ref_lon)
        │               └── wgs84_to_ecef() → ecef_to_enu()
        │
        ├── Downsample: keep points ≥ 8 m apart (ENU distance)
        └── Return {reference, points, point_count, source_geometry_points}

Client-side AR (ar_renderer.js):
        ├── WebXR mode: Three.js scene, XRSession, camera pose from device
        │       └── Route arrows rendered as 3D objects in world space
        └── Compass fallback: 2D canvas overlay, arrows rotated by heading
```

---

## Key Classes and Functions

### `web/routes/speech.py`

#### `POST /api/speech/transcribe`

```
Content-Type: multipart/form-data
Fields:
  file        (required) — audio blob (webm/ogg/wav/mp4/mp3/m4a/aac)
  language    (optional) — ISO 639-1 code: "vi", "en", or empty for auto-detect
  session_id  (optional) — for logging
```

**Whisper model selection** (via `settings.whisper_model`):

| Model | Size | Speed | Vietnamese accuracy |
|---|---|---|---|
| `tiny` | 39 MB | Fastest | Low |
| `base` | 74 MB | Fast | Moderate (default) |
| `small` | 244 MB | Medium | Good |
| `medium` | 769 MB | Slow | Best |
| `large` | 1.5 GB | Slowest | Highest |

**Lazy loading**: The Whisper model is loaded on the first request and cached in the module-level `_whisper_model` variable. Subsequent requests reuse the cached model. An `asyncio.Lock` prevents concurrent loading.

**Thread pool execution**: `model.transcribe()` is CPU/GPU-bound and blocks the event loop. It is run in `loop.run_in_executor(None, ...)` to avoid blocking other requests.

**Vietnamese optimisation**:
- `initial_prompt="Điều hướng, tìm đường, địa điểm."` — primes the model with navigation vocabulary.
- `condition_on_previous_text=False` — prevents hallucination from previous segments.
- `fp16=True` on GPU for ~2× speed.

**Allowed audio types**:
```python
_ALLOWED_AUDIO = {
    "audio/webm", "audio/ogg", "audio/wav", "audio/wave",
    "audio/mp4", "audio/mpeg", "audio/mp3", "audio/x-m4a",
    "audio/aac", "video/webm",           # Chrome records as video/webm
    "application/octet-stream",          # fallback when browser omits content-type
}
```

**Temp file cleanup**: The temp file is always deleted in a `finally` block, even if transcription fails.

---

### `core/geo_ar.py`

#### `wgs84_to_ecef(lat, lon, alt) -> tuple[float, float, float]`

Converts WGS84 geodetic coordinates to Earth-Centred Earth-Fixed (ECEF) Cartesian coordinates.

```
N = a / sqrt(1 - e² × sin²(lat))
X = (N + alt) × cos(lat) × cos(lon)
Y = (N + alt) × cos(lat) × sin(lon)
Z = (N × (1 - e²) + alt) × sin(lat)
```

Where `a = 6378137.0 m` (WGS84 semi-major axis) and `e² = 6.69437999014e-3`.

#### `ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt) -> tuple[float, float, float]`

Converts ECEF coordinates to local ENU frame centred at `(ref_lat, ref_lon, ref_alt)`.

```
dx, dy, dz = ECEF_point - ECEF_reference
East  = -sin(lon_ref) × dx + cos(lon_ref) × dy
North = -sin(lat_ref)×cos(lon_ref)×dx - sin(lat_ref)×sin(lon_ref)×dy + cos(lat_ref)×dz
Up    =  cos(lat_ref)×cos(lon_ref)×dx + cos(lat_ref)×sin(lon_ref)×dy + sin(lat_ref)×dz
```

#### `wgs84_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt) -> tuple[float, float, float]`

Convenience wrapper: `ecef_to_enu(*wgs84_to_ecef(lat, lon, alt), ref_lat, ref_lon, ref_alt)`.

#### `route_to_local_frame(geometry, ref_lat, ref_lon, ref_alt) -> list[dict]`

Converts a route polyline to ENU local frame. Returns:
```json
[
  {"index": 0, "lat": 10.9085, "lon": 106.760, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0},
  {"index": 1, "lat": 10.9086, "lon": 106.761, "east_m": 88.5, "north_m": 11.1, "up_m": 0.0}
]
```

---

### `core/route_projection.py`

#### `build_ar_path(route, ref_lat, ref_lon, ref_alt, min_spacing_m=8.0) -> dict`

Converts route geometry to a downsampled ENU point list for the AR renderer.

**Downsampling algorithm**:
1. Convert all geometry points to ENU via `route_to_local_frame()`.
2. Start with the first point.
3. For each subsequent point: add it only if ENU distance from the last kept point ≥ `min_spacing_m`.
4. Always include the last point.

This reduces a 500-point polyline to ~30–60 AR waypoints, keeping the path smooth without overwhelming the renderer.

**Output**:
```json
{
  "reference": {"lat": 10.9085, "lon": 106.760, "alt": 0.0},
  "points": [
    {"index": 0, "lat": 10.9085, "lon": 106.760, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0},
    {"index": 5, "lat": 10.9086, "lon": 106.761, "east_m": 88.5, "north_m": 11.1, "up_m": 0.0}
  ],
  "point_count": 18,
  "source_geometry_points": 245
}
```

---

### `web/ar_renderer.js`

The Three.js WebXR AR renderer. Key functions:

#### `setUserPose(lat, lon, headingDeg)`
Updates the user's position in the ENU frame. Converts lat/lon to ENU offset from the route reference point.

#### `setArPath(arPathData)`
Loads the AR path from `build_ar_path()` output. Creates 3D arrow objects at each waypoint in the Three.js scene.

#### `setPois(poisArray)`
Places POI markers (floating labels) at their ENU positions.

#### `setNextInstruction(instructionDict)`
Updates the instruction overlay (text + urgency colour).

#### `_initAR()`
Initialises the AR session:
1. Checks `navigator.xr.isSessionSupported('immersive-ar')`.
2. If supported: starts WebXR session, renders route arrows in world space.
3. If not supported: falls back to compass 2D mode — canvas overlay with arrows rotated by device heading.

**Mode badge**: Shows `"🥽 WebXR"` or `"🧭 Compass"` in the UI.

---

### `web/vio_client.js`

JavaScript-side VIO client. Key components:

#### `VIOClient`
- **IMU loop**: Listens to `DeviceMotionEvent` at ~50 Hz. Sends `POST /api/realtime/vio/imu` with `ax, ay, az, gyro_z, compass_deg, dt_s`.
- **Optical flow**: Captures consecutive camera frames via canvas, computes mean feature displacement using pixel difference. Sends `POST /api/realtime/vio/flow` at ~5 Hz.
- **`_vioOnGpsFix(lat, lon, heading)`**: Called when GPS updates. Sends `POST /api/realtime/vio/relocalize` to reset drift.
- **`_vioTriggerVprFrame()`**: When `needs_relocalization=true` is received from the server, captures a frame and sends it to `POST /api/realtime/frame` for VPR re-localization.

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny` \| `base` \| `small` \| `medium` \| `large` |
| `WHISPER_DEVICE` | `""` | Override device for Whisper: `cpu` \| `cuda`. Empty = inherit from `DEVICE` |
| `DEVICE` | `cuda` | Main compute device |
| `AR_ENABLED` | `true` | Enable AR endpoints and UI |

---

## How to Test

### Transcribe audio

```bash
# Record a short audio clip first, then:
curl -X POST http://192.168.1.217:8000/api/speech/transcribe \
  -F "file=@voice.webm" \
  -F "language=vi" \
  -F "session_id=test"
```

### Check AR path in route response

```bash
curl -X POST http://192.168.1.217:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{"destination": "chợ", "origin_lat": 10.9085, "origin_lon": 106.760}' \
  | python -c "import sys,json; d=json.load(sys.stdin); print('AR points:', d['ar_path']['point_count'])"
```

### Test AR in browser

1. Open `http://192.168.1.217:8000` on Android Chrome (same WiFi).
2. Find a route first (Navigation tab).
3. Click the AR button.
4. Allow DeviceMotion permission when prompted.
5. Check the badge in the top-right corner of the AR view.

### Test speech in browser

1. Open `http://192.168.1.217:8000` on phone.
2. Click the microphone button in the chat panel.
3. Speak a navigation command in Vietnamese.
4. The transcript should appear in the chat input.

---

## Healthy Output Examples

**Transcription response:**
```json
{
  "ok": true,
  "text": "tìm đường đến chợ Dĩ An",
  "language": "vi",
  "session_id": "test"
}
```

**AR path in route response:**
```json
{
  "ar_path": {
    "reference": {"lat": 10.9085, "lon": 106.760, "alt": 0.0},
    "points": [
      {"index": 0, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0},
      {"index": 8, "east_m": 88.5, "north_m": 11.1, "up_m": 0.0}
    ],
    "point_count": 18,
    "source_geometry_points": 245
  }
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `503 "openai-whisper not installed"` | Whisper package missing | `pip install openai-whisper` |
| `503 "Whisper load failed"` | GPU OOM or CUDA error | Set `WHISPER_DEVICE=cpu` in `.env` |
| `400 "Audio too large"` | File > 10 MB | Reduce recording duration or bitrate |
| `400 "Unsupported audio type"` | Browser sends unexpected MIME | The `application/octet-stream` fallback handles most cases |
| `ar_path.point_count = 0` | Route has no geometry | Check that route returned non-empty `geometry` array |
| AR canvas black | WebXR not supported on device | Expected on iOS or non-AR Android; compass mode activates automatically |
| AR arrows misaligned | Compass calibration error | Calibrate device compass; move phone in figure-8 pattern |
| Transcription in wrong language | Auto-detect failed | Set `language=vi` explicitly in the request |
| Whisper hallucinating | Short or silent audio | Ensure audio is at least 1 second; check microphone permissions |

---

## Performance Notes

- **Whisper `base` on GPU**: ~0.5–1 s for a 3-second voice command.
- **Whisper `base` on CPU**: ~3–8 s for a 3-second voice command.
- **Whisper `medium` on GPU**: ~2–4 s — significantly better Vietnamese accuracy.
- **Model loading**: First request takes 2–10 s to load the model into memory. Subsequent requests are fast.
- **`build_ar_path()`**: ~1–5 ms for a 500-point route.
- **`wgs84_to_enu()`**: ~0.01 ms per point — pure Python math, no external dependencies.
- **AR rendering**: Three.js WebXR runs at 60 fps on modern Android devices. The compass fallback runs at 30 fps.
- For production, consider pre-loading the Whisper model at startup (call `_load_whisper()` in the startup event) to avoid the first-request delay.
