# 03 — Floor Detection & Visual-Inertial Odometry (VIO)

## Overview

This module provides indoor positioning when GPS is unavailable or inaccurate. It has three layers:

1. **FloorDetector** — determines which floor the user is on using barometric pressure and accelerometer step/stair/elevator pattern recognition.
2. **VIOFusion** — a 4-state Extended Kalman Filter (EKF) that dead-reckons position from IMU (accelerometer + gyroscope), optical flow, and compass, with VPR/GPS re-localization to correct drift.
3. **FusionPoseEstimator** — top-level fuser that blends GPS, IMU heading, and VIO into a single `FusedPose` output.

---

## Architecture / Data Flow

```
DeviceMotion (JS)          Barometer (JS)
     │                          │
     ▼                          ▼
update_accel(ax,ay,az)    update_pressure(hpa)
     │                          │
     ├── _detect_step()         ├── ISA model → delta_floors
     ├── _detect_elevator()     └── calibrate_floor()
     └── _classify_stair_dir()
              │
              ▼
         FloorDetector._recompute()
         W_BARO=0.65 × baro_conf + W_STEP=0.35 × step_conf
              │
              ▼
         FusedPose.floor / floor_confidence / floor_method

─────────────────────────────────────────────────────────────

IMU (10–100 Hz)          Optical Flow (~5 Hz)      GPS/VPR fix
     │                          │                       │
     ▼                          ▼                       ▼
_EKF4DOF.predict()    _EKF4DOF.update_optical_flow()  relocalize()
  gyro_z, accel_fwd        flow_x_m, flow_y_m        lat/lon → ENU
     │                          │                       │
     └──────────────────────────┴───────────────────────┘
                                │
                          VIOPose (px, py, heading, speed, drift_m)
                                │
                                ▼
                     FusionPoseEstimator.update_vio()
                                │
                                ▼
                           FusedPose (lat, lon, heading, floor, vio_*)
```

---

## Key Classes and Functions

### `core/floor_detector.py`

#### Constants

| Constant | Value | Meaning |
|---|---|---|
| `FLOOR_HEIGHT_M` | `3.2` | Assumed floor height in metres |
| `HPA_PER_METRE` | `0.1198` | ISA sea-level pressure gradient |
| `HPA_PER_FLOOR` | `0.383` | Pressure change per floor (~3.2 m × 0.1198) |
| `STAIR_Z_THRESHOLD` | `1.5 m/s²` | Minimum Z-axis excess above gravity to count as step |
| `STAIR_CADENCE_MIN_HZ` | `0.8` | Minimum step frequency for stair detection |
| `STAIR_CADENCE_MAX_HZ` | `2.5` | Maximum step frequency for stair detection |
| `STAIR_WINDOW_S` | `2.0` | Accelerometer history window for cadence analysis |
| `ELEV_Z_OFFSET_MIN` | `0.3 m/s²` | Minimum sustained Z offset for elevator detection |
| `ELEV_SUSTAINED_S` | `0.8` | Minimum duration of Z offset to confirm elevator |
| `W_BARO` | `0.65` | Barometer confidence weight in fusion |
| `W_STEP` | `0.35` | Step detector confidence weight in fusion |

#### `FloorDetector`

```python
class FloorDetector:
    def update_pressure(hpa: float) -> None
    def calibrate_floor(floor: int) -> None
    def update_accel(ax: float, ay: float, az: float) -> None
    def get_floor() -> dict   # {"floor": int, "confidence": float, "method": str}
    def reset() -> None
```

**Barometer branch**:
- Baseline pressure is set from the median of the first 3 readings.
- `delta_hpa = baseline - current` (positive = higher floor).
- `raw_floor = 1 + (delta_hpa / HPA_PER_METRE) / FLOOR_HEIGHT_M`
- Confidence = `max(0.3, 1.0 - frac × 2.0)` where `frac` is the fractional part of `raw_floor`.

**Step/stair branch**:
- `_detect_step()`: counts peaks where `norm - 9.81 > 1.5 m/s²`, debounced at 0.25 s.
- `_classify_stair_direction()`: computes cadence from peak count / window duration. If cadence is in `[0.8, 2.5]` Hz, checks mean `az`: `az < -0.5` → climbing (+1), `az > 0.5` → descending (-1).
- `_detect_elevator()`: detects sustained `|norm - 9.81| >= 0.3 m/s²` for ≥ 0.8 s.

**Fusion**:
- If both baro and step agree on floor: `confidence = min(1.0, W_BARO × baro_conf + W_STEP × step_conf + 0.15)`, method = `"barometer+step"`.
- If they disagree: barometer wins with reduced confidence (`× 0.7`), method = `"barometer"`.
- If only one source: use that source alone.

**`calibrate_floor(floor)`**: Adjusts `_baseline_hpa` so that the current pressure maps to the given floor number. This is the manual override for when the user tells the app "I'm on floor 2."

---

### `core/vio_fusion.py`

#### Coordinate Conventions

- **ENU frame**: East = +X, North = +Y, Up = +Z. Origin is the session's first GPS fix.
- **Heading convention**: Math convention — 0 = East, increases counter-clockwise (CCW).
- **Compass → Math**: `math_rad = (90 - compass_deg) × π/180`
- **Device frame**: Phone held upright, screen facing user. +Y = toward top of phone (forward when walking). Gravity ≈ -9.81 on Y axis when stationary.

#### `_EKF4DOF`

State vector: `x = [px, py, heading_rad, speed_m_s]`

```python
class _EKF4DOF:
    def initialize(px, py, heading_rad, speed) -> None
    def predict(dt, gyro_z, accel_fwd) -> None
    def update_position(px_meas, py_meas, r_m2) -> None
    def update_heading(heading_rad, r_rad2) -> None
    def update_optical_flow(flow_x_m, flow_y_m, dt) -> None
```

**Predict step** (motion model):
```
h_new = h + gyro_z × dt
px_new = px + v × cos(h_mid) × dt
py_new = py + v × sin(h_mid) × dt
v_new  = clamp(v + accel_fwd × dt, 0, 3.0)
```
Jacobian `F` is computed analytically. Process noise `Q` scales with `dt`.

**Process noise** (tuned for indoor pedestrian):
- `Q_POS = 0.04 m²/s`
- `Q_HEAD = 0.02 rad²/s`
- `Q_SPEED = 0.5 (m/s)²/s`

**Measurement noise**:
- `R_FLOW = 4.0 px²` → converted to metres via `_FLOW_PX_PER_M = 554.0`
- `R_GPS = 9.0 m²` (3 m std dev)
- `R_COMPASS = 0.04 rad²` (~11.5° std dev)

**Optical flow update**: Converts pixel displacement to world-frame velocity, then updates the speed state only.

**Heading update**: Uses circular interpolation to handle angle wrap-around in the innovation term.

---

#### `VIOFusion`

```python
class VIOFusion:
    def reset(lat, lon, heading_deg) -> None
    def update_imu(ax, ay, az, gyro_z_rad_s, compass_deg, dt_s) -> VIOPose
    def update_optical_flow(flow_x_px, flow_y_px, dt_s) -> VIOPose
    def relocalize(lat, lon, heading_deg, accuracy_m) -> VIOPose
    def get_pose() -> VIOPose
    @property drift_m: float
    @property needs_relocalization: bool  # drift_m > VPR_DRIFT_TRIGGER_M (2.0 m)
```

**`update_imu`**:
1. Extracts forward acceleration: `ay_ng = ay + 9.81` (removes gravity from Y axis).
2. Applies compass complementary filter: `α = 0.15` (slow trust of compass).
3. Calls `_EKF4DOF.predict(dt, gyro_z, accel_fwd)`.
4. Calls `_EKF4DOF.update_heading(compass_math)` if compass available.

**`relocalize`**:
1. Converts `lat/lon` to ENU metres relative to session origin.
2. Calls `_EKF4DOF.update_position(px_abs, py_abs, accuracy_m²)`.
3. Resets drift counter.

**`VIOPose.to_latlon()`**: Converts ENU offset back to absolute lat/lon using:
```
lat = origin_lat + py / 111320
lon = origin_lon + px / (111320 × cos(origin_lat))
```

#### `VIORegistry` (singleton: `vio_registry`)
Holds one `VIOFusion` per session ID. `get_or_create(session_id)` creates on first access.

---

### `core/sensor_fusion.py`

#### `FusedPose` (dataclass)

```python
@dataclass
class FusedPose:
    lat, lon: float | None
    accuracy_m: float | None
    heading_deg: float | None
    speed_kmh: float
    confidence: float          # 0–1
    source: str                # "gps" | "gps+fusion" | "gps+imu" | "vio"
    floor: int
    floor_confidence: float
    floor_method: str
    vio_px, vio_py: float | None   # ENU metres
    vio_drift_m: float
    vio_source: str
```

#### `FusionPoseEstimator`

```python
class FusionPoseEstimator:
    def update_gps(lat, lon, *, accuracy_m, speed_kmh, bearing) -> FusedPose
    def update_imu(*, compass_heading, gyro_heading, accel_norm,
                   floor, floor_confidence, floor_method) -> FusedPose
    def update_vio(vio_pose: VIOPose) -> FusedPose
```

**GPS complementary filter** (`update_gps`):
```
fused_lat = α × new_lat + (1-α) × old_lat
fused_lon = α × new_lon + (1-α) × old_lon
```
where `α = settings.fusion_position_alpha` (default 0.7).

**Heading blend** (`_blend_heading`): Circular interpolation using unit vectors to avoid wrap-around issues. `α = settings.fusion_heading_alpha` (default 0.82).

**VIO merge** (`update_vio`): When GPS is absent and VIO has an origin, synthesises lat/lon from ENU offset. Accuracy is estimated from `sqrt(cov_px + cov_py)`.

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `SENSOR_FUSION_MODE` | `complementary` | `raw` \| `complementary` \| `kalman_lite` |
| `FUSION_HEADING_ALPHA` | `0.82` | Heading blend factor (higher = trust new heading more) |
| `FUSION_POSITION_ALPHA` | `0.7` | GPS position blend factor |
| `VIO_VPR_MIN_SCORE` | `0.72` | Minimum VPR score to accept re-localization |
| `VIO_DRIFT_TRIGGER_M` | `2.0` | Drift threshold to trigger VPR re-localization |
| `VIO_FLOW_PX_PER_M` | `554.0` | Optical flow calibration (pixels per metre at 1 m) |

---

## How to Test

### Floor detection (barometer)

```bash
curl -X POST http://192.168.1.217:8000/api/realtime/floor \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "pressure_hpa": 1013.25}'
```

### Floor calibration

```bash
curl -X POST http://192.168.1.217:8000/api/realtime/floor/calibrate \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "floor": 2}'
```

### VIO IMU update (high-rate)

```bash
curl -X POST http://192.168.1.217:8000/api/realtime/vio/imu \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "ax": 0.1, "ay": -9.75, "az": 0.05,
    "gyro_z": 0.02,
    "compass_deg": 90.0,
    "dt_s": 0.05
  }'
```

### Get current VIO pose

```bash
curl http://192.168.1.217:8000/api/realtime/vio/pose/test
```

### VIO re-localization from GPS

```bash
curl -X POST http://192.168.1.217:8000/api/realtime/vio/relocalize \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "lat": 10.9085, "lon": 106.760,
    "heading_deg": 90.0,
    "accuracy_m": 3.0,
    "source": "gps"
  }'
```

---

## Healthy Output Examples

**Floor detection:**
```json
{
  "ok": true,
  "session_id": "test",
  "floor": {
    "floor": 2,
    "confidence": 0.82,
    "method": "barometer+step"
  },
  "revision": 5
}
```

**VIO pose:**
```json
{
  "ok": true,
  "vio_pose": {
    "px": 12.345,
    "py": -3.210,
    "heading_deg": 87.5,
    "speed_ms": 1.2,
    "cov_px": 0.12,
    "cov_py": 0.15,
    "cov_heading_deg": 8.3,
    "origin_lat": 10.9085,
    "origin_lon": 106.760,
    "source": "imu",
    "drift_m": 0.45,
    "updated_at": "2025-01-15T08:30:00.123456"
  }
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `method: "none"`, `confidence: 0` | No barometer data received | Check if device has barometer; send `pressure_hpa` in sensor payload |
| Floor stuck at 1 | Baseline not calibrated | Call `POST /api/realtime/floor/calibrate` with current floor |
| `drift_m` growing unbounded | No GPS/VPR fixes arriving | Ensure GPS is active; check VPR index is built |
| `needs_relocalization: true` always | VPR score below `VIO_VPR_MIN_SCORE` | Lower threshold or add more reference images |
| Heading oscillating | Compass interference (metal, magnets) | Increase `FUSION_HEADING_ALPHA` to trust gyro more |
| `503 "Realtime manager not ready"` | App startup not complete | Wait for startup; check logs |

---

## Performance Notes

- **IMU update rate**: Designed for 10–100 Hz. The EKF predict step takes ~0.1 ms on CPU.
- **Optical flow update**: ~0.5 ms per call.
- **Floor detection**: `get_floor()` is synchronous and takes ~0.1 ms.
- **VPR re-localization**: Triggered automatically when `drift_m > 2.0 m` and a frame is available. Takes 80–500 ms depending on VPR backend.
- **Memory**: One `VIOFusion` per session uses ~5 KB (4×4 covariance matrix + state).
- The EKF covariance matrix is symmetrised after each update to prevent numerical drift accumulation.
