# 05 — Realtime Navigation Pipeline

## Overview

The realtime module processes a continuous stream of camera frames, GPS fixes, and sensor data to maintain a live navigation state per session. It orchestrates scene understanding (YOLO, OCR, VPR), navigation event detection (step advance, off-route, arrival), proactive alerts, and VIO dead-reckoning. State changes are pushed to connected clients over WebSocket.

---

## Architecture / Data Flow

```
Client (mobile browser)
    │
    ├── POST /api/realtime/frame  (image + GPS, ~400 ms interval)
    │         │
    │         ▼
    │   RealtimeSessionManager.ingest_frame()
    │         │
    │         ├── session_manager.process_gps_update()
    │         │       └── NavSession.update_gps()
    │         │           ├── snap_point_to_polyline()
    │         │           ├── step_advance (d < 25 m)
    │         │           ├── arrived (d_dest < 30 m)
    │         │           └── off_route (d > 50 m × 3 times)
    │         │
    │         ├── FusionPoseEstimator.update_gps()
    │         │
    │         ├── SceneFusionService.build_scene_state()
    │         │       ├── YOLO (throttled at realtime_yolo_fps)
    │         │       ├── OCR  (throttled at realtime_ocr_interval_ms)
    │         │       └── VPR  (throttled at realtime_vpr_interval_ms)
    │         │
    │         ├── RealtimeNavigator.build_instruction()
    │         │
    │         ├── VIO auto-relocalize (if drift > 2 m)
    │         │
    │         └── AlertEngine.evaluate()
    │
    ├── POST /api/realtime/sensors  (compass, accel, pressure)
    │         └── FloorDetector + FusionPoseEstimator.update_imu()
    │
    ├── POST /api/realtime/vio/imu  (high-rate, 10–100 Hz)
    │
    └── WS /ws/realtime/{session_id}
              │
              ├── Push realtime_state on revision change
              ├── Push pending alerts immediately
              └── Handle: ping, sensors, floor, vio_imu, vio_flow
```

---

## Key Classes and Functions

### `core/realtime_manager.py`

#### `RealtimeSession` (dataclass)

```python
@dataclass
class RealtimeSession:
    session_id: str
    latest_frame_path: str | None
    latest_frame_meta: dict
    latest_gps: dict
    latest_sensors: dict
    latest_scene_state: dict
    latest_instruction: dict
    latest_nav_event: dict
    latest_floor: dict
    latest_vio_pose: dict
    revision: int                    # increments on every touch()
    fusion_state: SceneFusionState   # throttle timestamps for YOLO/OCR/VPR
    pose_estimator: FusionPoseEstimator
    floor_detector: FloorDetector
    pending_alerts: list[dict]       # drained by WebSocket loop
```

`touch()` increments `revision` and updates `updated_at`. The WebSocket loop pushes state to the client whenever `revision` changes.

`pop_alerts()` drains and returns `pending_alerts` — called by the WebSocket loop to push alerts independently of state revision.

---

#### `RealtimeSessionManager`

```python
class RealtimeSessionManager:
    def get_or_create(session_id) -> RealtimeSession
    async def update_sensors(session_id, payload) -> dict
    async def update_floor(session_id, pressure_hpa, accel) -> dict
    async def calibrate_floor(session_id, floor) -> dict
    async def vio_update_imu(session_id, payload) -> dict
    async def vio_update_flow(session_id, flow_x_px, flow_y_px, dt_s) -> dict
    async def vio_relocalize(session_id, lat, lon, heading_deg, accuracy_m, source) -> dict
    async def vio_get_pose(session_id) -> dict
    async def vio_try_vpr_relocalize(session_id, frame_path, vpr_engine) -> dict | None
    async def ingest_frame(session_id, frame_path, *, lat, lon, ...) -> dict
    def build_frame_path(suffix) -> Path
```

**`ingest_frame` pipeline** (called on every camera frame):
1. Save frame path and metadata to session.
2. If GPS provided: call `session_manager.process_gps_update()` → get nav event (step_advance / arrived / off_route / rerouted).
3. Update `FusionPoseEstimator` with GPS fix.
4. Call `SceneFusionService.build_scene_state()` → get scene state with YOLO/OCR/VPR results.
5. Call `RealtimeNavigator.build_instruction()` → get instruction dict.
6. Auto-trigger VPR re-localization if VIO drift > threshold.
7. Call `AlertEngine.evaluate()` → get new alerts.
8. Append alerts to `session.pending_alerts`.
9. `session.touch()` → increment revision.

**`build_frame_path`**: Generates a unique path under `data/realtime_frames/` with timestamp and UUID suffix.

---

### `core/scene_fusion.py`

#### `SceneFusionState` (dataclass)
Tracks last-run timestamps and cached results for each vision module:
- `last_ocr_at`, `last_yolo_at`, `last_vpr_at`
- `last_ocr_blocks`, `last_landmarks`, `last_vpr_hint`

#### `SceneFusionService`

```python
class SceneFusionService:
    async def build_scene_state(frame_path, *, gps, nav_event,
                                nav_session, fused_pose, fusion_state) -> dict
    @staticmethod _should_run(now, last_run, interval_ms) -> bool
    @staticmethod _build_route_progress(nav_event, nav_session) -> dict
```

**Throttling logic** (`_should_run`):
- YOLO: runs if `(now - last_yolo_at) × 1000 ≥ 1000 / realtime_yolo_fps` (default 3 fps → every 333 ms)
- OCR: runs if elapsed ≥ `realtime_ocr_interval_ms` (default 1500 ms)
- VPR: runs if elapsed ≥ `realtime_vpr_interval_ms` (default 2500 ms)

If a module is not due to run, the previous cached result is returned. This prevents GPU overload while keeping results fresh.

**`_build_route_progress`**: Extracts from the nav session:
- `state`: `"idle"` \| `"navigating"` \| `"rerouting"` \| `"arrived"`
- `current_step_idx`
- `off_route`: bool
- `distance_to_route_m`
- `next_maneuver`: instruction text of the next step
- `distance_to_next_turn_m`
- `map_match`: `{lat, lon, residual_m, segment_index}`

**Scene state output structure**:
```json
{
  "timestamp": "2025-01-15T08:30:00.123456",
  "gps": {"lat": 10.9085, "lon": 106.760, "accuracy_m": 5.0},
  "fused_pose": {"lat": 10.9085, "lon": 106.760, "confidence": 0.85},
  "route_progress": {
    "state": "navigating",
    "current_step_idx": 2,
    "off_route": false,
    "next_maneuver": "Rẽ phải vào Đường Lê Lợi",
    "distance_to_next_turn_m": 45.0
  },
  "visual": {
    "landmarks": [{"label": "car", "confidence": 0.87, "bbox": [100, 200, 300, 400]}],
    "ocr_blocks": [{"text": "Chợ Dĩ An", "confidence": 0.92}],
    "vpr_hint": {"location_name": "Ngã tư Bình Dương", "score": 0.81},
    "confidence": 0.85
  }
}
```

---

### `core/alert_engine.py`

#### Alert Types and Thresholds

| Alert Type | Trigger Condition | Urgency | Cooldown |
|---|---|---|---|
| `turn_soon` | `distance_to_next_turn_m ≤ 80` | normal | 15 s |
| `turn_now` | `distance_to_next_turn_m ≤ 20` | high | 8 s |
| `off_route` | `route_progress.off_route == true` | high | 30 s |
| `rerouting` | `nav_event.type == "rerouted"` (first time) | normal | 15 s |
| `arrived` | `nav_event.type == "arrived"` (first time) | normal | 60 s |
| `stairs_detected` | YOLO detects "stair"/"cầu thang" | normal | 20 s |
| `elevator_detected` | YOLO detects "elevator"/"thang máy" | low | 20 s |
| `floor_change` | `floor != prev_floor` AND `confidence ≥ 0.5` | normal | 5 s |
| `vio_drift` | `vio_pose.drift_m > 2.0` | low | 20 s |
| `low_battery` | `sensors.battery_level ≤ 20%` | low | 60 s |

#### `AlertEngine`

```python
class AlertEngine:
    def evaluate(session_state: dict) -> list[Alert]
    def reset() -> None
    def _can_fire(alert_type, suppress_s, now) -> bool
```

`evaluate()` checks all 8 alert conditions in order and returns only alerts whose cooldown has expired. Cooldown is tracked per alert type in `_last_fired`.

#### `Alert` (dataclass)
```python
@dataclass
class Alert:
    type: AlertType
    message: str          # Vietnamese, ready for TTS
    urgency: str          # "high" | "normal" | "low"
    distance_m: float | None
    suppress_s: float     # cooldown in seconds
    ts: datetime
```

#### `AlertEngineRegistry` (singleton: `alert_registry`)
One `AlertEngine` per session ID.

---

### `web/routes/realtime.py`

| Endpoint | Method | Description |
|---|---|---|
| `/api/realtime/frame` | POST | Ingest camera frame + GPS |
| `/api/realtime/sensors` | POST | Update compass, accel, pressure |
| `/api/realtime/floor` | POST | Dedicated floor update |
| `/api/realtime/floor/calibrate` | POST | Manual floor calibration |
| `/api/realtime/vio/imu` | POST | High-rate IMU update |
| `/api/realtime/vio/flow` | POST | Optical flow correction |
| `/api/realtime/vio/relocalize` | POST | Absolute position reset |
| `/api/realtime/vio/pose/{sid}` | GET | Get current VIO pose |
| `/api/realtime/state/{sid}` | GET | Get full session state |
| `/api/realtime/sessions` | GET | List all active sessions |
| `/ws/realtime/{session_id}` | WS | Bidirectional realtime channel |

---

## WebSocket Protocol

### Connect
```
ws://192.168.1.217:8000/ws/realtime/test_session
```

### Server → Client messages

| Type | When | Payload |
|---|---|---|
| `realtime_state` | On every revision change | Full session state dict |
| `alert` | Immediately when alert fires | `{"type": "alert", "alert": {...}}` |
| `sensor_update` | After `sensors` message | Updated fused pose + floor |
| `floor_update` | After `floor` message | Floor estimate |
| `vio_pose` | After `vio_imu` or `vio_flow` | VIO pose dict |
| `pong` | After `ping` | `{"type": "pong"}` |

### Client → Server messages

```json
// Ping
{"type": "ping"}

// Sensor update
{
  "type": "sensors",
  "session_id": "test_session",
  "compass_heading": 90.0,
  "accel_norm": 9.81,
  "accel_x": 0.1, "accel_y": -9.75, "accel_z": 0.05,
  "pressure_hpa": 1013.25
}

// Floor update
{
  "type": "floor",
  "pressure_hpa": 1012.85,
  "accel_x": 0.1, "accel_y": -9.75, "accel_z": 0.05
}

// VIO IMU (high-rate)
{
  "type": "vio_imu",
  "ax": 0.1, "ay": -9.75, "az": 0.05,
  "gyro_z": 0.02,
  "compass_deg": 90.0,
  "dt_s": 0.05
}

// VIO optical flow
{
  "type": "vio_flow",
  "flow_x_px": 3.5,
  "flow_y_px": -1.2,
  "dt_s": 0.1
}
```

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `REALTIME_ENABLED` | `true` | Enable/disable realtime endpoints |
| `REALTIME_FRAME_INTERVAL_MS` | `400` | Target frame interval (client-side) |
| `REALTIME_YOLO_FPS` | `3.0` | Max YOLO inference rate |
| `REALTIME_OCR_INTERVAL_MS` | `1500` | Min interval between OCR runs |
| `REALTIME_VPR_INTERVAL_MS` | `2500` | Min interval between VPR runs |
| `REALTIME_FRAME_MAX_MB` | `8` | Max frame upload size in MB |

---

## How to Test

### WebSocket test (browser console)

```javascript
const ws = new WebSocket('ws://192.168.1.217:8000/ws/realtime/test_session');
ws.onmessage = e => console.log(JSON.parse(e.data));
ws.onopen = () => ws.send(JSON.stringify({type: 'ping'}));
// Expected: {"type": "pong"}
```

### Send sensor data via WebSocket

```javascript
ws.send(JSON.stringify({
  type: 'sensors',
  session_id: 'test_session',
  compass_heading: 45,
  accel_norm: 9.8
}));
```

### POST a frame with GPS

```bash
curl -X POST http://192.168.1.217:8000/api/realtime/frame \
  -F "file=@photo.jpg" \
  -F "session_id=test" \
  -F "lat=10.9085" \
  -F "lon=106.760" \
  -F "accuracy_m=5.0"
```

### Check session state

```bash
curl http://192.168.1.217:8000/api/realtime/state/test
```

---

## Healthy Output Examples

**Frame ingestion response:**
```json
{
  "ok": true,
  "session_id": "test",
  "scene_state": {
    "route_progress": {"state": "navigating", "current_step_idx": 2},
    "visual": {"landmarks": [{"label": "car", "confidence": 0.87}]}
  },
  "instruction": {
    "instruction": "Rẽ phải sau khoảng 45 m.",
    "urgency": "normal"
  },
  "nav_event": {"type": "none", "d_route_m": 3.2},
  "alerts": [],
  "revision": 12
}
```

**Alert pushed via WebSocket:**
```json
{
  "type": "alert",
  "alert": {
    "type": "turn_soon",
    "message": "Rẽ phải sau 45 mét.",
    "urgency": "normal",
    "distance_m": 45.0,
    "suppress_s": 15.0,
    "ts": "2025-01-15T08:30:00.123456"
  }
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `503 "Realtime manager not ready"` | App startup incomplete | Wait for startup; check logs |
| `revision` stuck at 0 | No frames or sensors being sent | Verify client is sending data; check network |
| Alerts not firing | Cooldown not expired | Wait for `suppress_s` seconds; call `AlertEngine.reset()` |
| YOLO/OCR not running | Throttle interval not elapsed | Expected; results are cached between runs |
| `400 "Frame too large"` | Image exceeds `REALTIME_FRAME_MAX_MB` | Reduce camera resolution or JPEG quality |
| WebSocket closes immediately | `REALTIME_ENABLED=false` | Set `REALTIME_ENABLED=true` in `.env` |

---

## Performance Notes

- **Frame ingestion**: ~50–500 ms depending on which vision modules run.
  - GPS + nav event only: ~5 ms
  - + YOLO: +30–100 ms (GPU) / +200–800 ms (CPU)
  - + OCR: +50–200 ms (GPU) / +500 ms–2 s (CPU)
  - + VPR: +80–500 ms (GPU)
- **WebSocket loop**: Polls every 350 ms for state changes. Alerts are pushed immediately.
- **Session cleanup**: Sessions inactive for 2 hours are automatically deleted.
- **Frame storage**: Frames are saved to `data/realtime_frames/`. Clean up periodically — each frame is ~50–500 KB.
- The throttling system ensures GPU is not overwhelmed even at 400 ms frame intervals. Adjust `REALTIME_YOLO_FPS` and `REALTIME_OCR_INTERVAL_MS` based on available GPU memory.
