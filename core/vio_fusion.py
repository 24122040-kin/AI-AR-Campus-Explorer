"""
core/vio_fusion.py — Visual-Inertial Odometry (VIO Lite) for indoor positioning

Architecture
============
                  ┌──────────────┐
  DeviceMotion ──►│  IMU EKF     │──► predicted pose (x, y, heading, v)
                  └──────┬───────┘
                         │ predict step (10–100 Hz)
                  ┌──────▼───────┐
  Optical Flow ──►│  EKF update  │──► corrected pose
  (JS → server)   └──────┬───────┘
                         │ correct step (~5 Hz)
                  ┌──────▼───────┐
  VPR match ─────►│  Relocalize  │──► absolute reset when drift > threshold
                  └──────────────┘

EKF State vector  x = [px, py, heading_rad, speed_m_s]  (4-DOF)
  px, py   : position in metres relative to session origin (ENU)
  heading  : radians, 0 = East, π/2 = North  (standard math convention)
  speed    : scalar forward speed m/s

Coordinate convention
  ENU: East = +X, North = +Y  (matches geo_ar.py)
  heading 0 = East, increases CCW (standard math)
  Device heading from compass: 0 = North, increases CW → convert on input

Public API
==========
  VIOFusion(session_id)
  .update_imu(ax, ay, az, gyro_z_rad_s, compass_deg, dt_s)  → VIOPose
  .update_optical_flow(flow_x_px, flow_y_px, dt_s)          → VIOPose
  .relocalize(lat, lon, heading_deg, accuracy_m)             → VIOPose
  .get_pose()                                                → VIOPose
  .reset(lat, lon, heading_deg)
  .drift_m                                                   → float
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from config.settings import settings

# ── Constants ─────────────────────────────────────────────────────────────────

# EKF process noise (tuned for indoor pedestrian motion)
_Q_POS   = 0.04    # m²  position process noise per second
_Q_HEAD  = 0.02    # rad² heading process noise per second
_Q_SPEED = 0.5     # (m/s)² speed process noise per second

# EKF measurement noise
_R_FLOW_PX   = 4.0    # px²  optical flow measurement noise
_R_GPS_M     = 9.0    # m²   GPS/VPR position measurement noise
_R_COMPASS   = 0.04   # rad² compass heading measurement noise

# Optical flow → metres conversion (calibrated for ~60° FOV at 1m height)
# Pixels per metre at 1 m distance, 640px wide, 60° FOV:
#   px_per_m = 640 / (2 * tan(30°)) ≈ 554
_FLOW_PX_PER_M = 554.0

# Drift threshold for VPR re-localization trigger
VPR_DRIFT_TRIGGER_M: float = 2.0

# Gravity constant
_G = 9.81

# Maximum believable speed indoors (m/s)
_MAX_SPEED_MS = 3.0

# Complementary filter alpha for heading (IMU vs compass)
_COMPASS_ALPHA = 0.15   # low-pass: trust compass slowly


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class VIOPose:
    """Current VIO position estimate in ENU metres relative to session origin."""
    px: float = 0.0          # East metres
    py: float = 0.0          # North metres
    heading_rad: float = 0.0 # 0=East, π/2=North (math convention)
    speed_ms: float = 0.0    # forward speed m/s
    # Uncertainty
    cov_px: float = 1.0      # variance of px (m²)
    cov_py: float = 1.0      # variance of py (m²)
    cov_heading: float = 0.1 # variance of heading (rad²)
    # Absolute reference (set after GPS/VPR fix)
    origin_lat: float | None = None
    origin_lon: float | None = None
    # Metadata
    source: str = "none"     # "imu" | "flow" | "vpr" | "gps"
    drift_m: float = 0.0     # accumulated drift since last absolute fix
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> dict:
        return {
            "px": round(self.px, 3),
            "py": round(self.py, 3),
            "heading_deg": round(math.degrees(self.heading_rad) % 360, 2),
            "speed_ms": round(self.speed_ms, 3),
            "cov_px": round(self.cov_px, 4),
            "cov_py": round(self.cov_py, 4),
            "cov_heading_deg": round(math.degrees(math.sqrt(max(0, self.cov_heading))), 2),
            "origin_lat": self.origin_lat,
            "origin_lon": self.origin_lon,
            "source": self.source,
            "drift_m": round(self.drift_m, 3),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_latlon(self) -> tuple[float, float] | None:
        """Convert ENU offset back to absolute lat/lon (requires origin)."""
        if self.origin_lat is None or self.origin_lon is None:
            return None
        lat_m = 111320.0
        lon_m = 111320.0 * math.cos(math.radians(self.origin_lat))
        lat = self.origin_lat + self.py / lat_m
        lon = self.origin_lon + self.px / lon_m
        return lat, lon


# ── EKF core ──────────────────────────────────────────────────────────────────

class _EKF4DOF:
    """
    Minimal 4-state EKF for pedestrian dead-reckoning.

    State:  x = [px, py, heading, speed]
    Motion: constant-heading, constant-speed model with IMU corrections.

    All angles in radians (math convention: 0=East, CCW positive).
    """

    def __init__(self) -> None:
        # State vector
        self._x = np.zeros(4, dtype=np.float64)   # [px, py, heading, speed]
        # Covariance matrix (4×4)
        self._P = np.diag([1.0, 1.0, 0.1, 0.25]).astype(np.float64)
        # Process noise matrix (built per-step from dt)
        self._initialized = False

    def initialize(self, px: float, py: float, heading_rad: float, speed: float = 0.0) -> None:
        self._x[:] = [px, py, heading_rad, speed]
        self._P = np.diag([0.25, 0.25, 0.04, 0.01]).astype(np.float64)
        self._initialized = True

    # ── Predict step ──────────────────────────────────────────────────────────

    def predict(self, dt: float, gyro_z: float = 0.0, accel_fwd: float = 0.0) -> None:
        """
        Propagate state forward by dt seconds.
        gyro_z   : yaw rate rad/s (positive = CCW = left turn)
        accel_fwd: forward acceleration m/s² (along heading direction)
        """
        if not self._initialized or dt <= 0:
            return

        px, py, h, v = self._x

        # Clamp speed
        v = max(0.0, min(v + accel_fwd * dt, _MAX_SPEED_MS))

        # Heading update from gyro
        h_new = h + gyro_z * dt

        # Position update (midpoint integration)
        h_mid = h + gyro_z * dt * 0.5
        px_new = px + v * math.cos(h_mid) * dt
        py_new = py + v * math.sin(h_mid) * dt

        self._x[:] = [px_new, py_new, h_new, v]

        # Jacobian F = ∂f/∂x
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = -v * math.sin(h_mid) * dt   # ∂px/∂h
        F[0, 3] =  math.cos(h_mid) * dt        # ∂px/∂v
        F[1, 2] =  v * math.cos(h_mid) * dt    # ∂py/∂h
        F[1, 3] =  math.sin(h_mid) * dt        # ∂py/∂v

        # Process noise Q (scaled by dt)
        Q = np.diag([
            _Q_POS * dt,
            _Q_POS * dt,
            _Q_HEAD * dt,
            _Q_SPEED * dt,
        ]).astype(np.float64)

        self._P = F @ self._P @ F.T + Q

    # ── Update steps ──────────────────────────────────────────────────────────

    def update_position(self, px_meas: float, py_meas: float, r_m2: float = _R_GPS_M) -> None:
        """Correct position from GPS or VPR fix (metres)."""
        H = np.zeros((2, 4), dtype=np.float64)
        H[0, 0] = 1.0   # px
        H[1, 1] = 1.0   # py
        R = np.eye(2, dtype=np.float64) * r_m2
        z = np.array([px_meas, py_meas], dtype=np.float64)
        self._ekf_update(H, R, z)

    def update_heading(self, heading_rad: float, r_rad2: float = _R_COMPASS) -> None:
        """Correct heading from compass (radians, math convention)."""
        H = np.zeros((1, 4), dtype=np.float64)
        H[0, 2] = 1.0
        R = np.array([[r_rad2]], dtype=np.float64)
        # Wrap innovation to [-π, π]
        innov = _wrap_angle(heading_rad - self._x[2])
        z_pred = np.array([self._x[2]], dtype=np.float64)
        z_meas = np.array([self._x[2] + innov], dtype=np.float64)
        self._ekf_update(H, R, z_meas, z_pred_override=z_pred)

    def update_optical_flow(self, flow_x_m: float, flow_y_m: float, dt: float) -> None:
        """
        Correct velocity from optical flow.
        flow_x_m, flow_y_m: apparent motion in camera frame (metres/frame).
        We treat this as a velocity measurement in the body frame.
        """
        if dt <= 0:
            return
        # Convert body-frame flow to world-frame velocity
        h = self._x[2]
        vx_world = (flow_x_m * math.cos(h) - flow_y_m * math.sin(h)) / dt
        vy_world = (flow_x_m * math.sin(h) + flow_y_m * math.cos(h)) / dt
        v_fwd = math.sqrt(vx_world**2 + vy_world**2)

        # Update speed only
        H = np.zeros((1, 4), dtype=np.float64)
        H[0, 3] = 1.0
        R = np.array([[(_R_FLOW_PX / _FLOW_PX_PER_M)**2 / max(dt, 0.01)]], dtype=np.float64)
        z = np.array([v_fwd], dtype=np.float64)
        self._ekf_update(H, R, z)

    def _ekf_update(
        self,
        H: np.ndarray,
        R: np.ndarray,
        z: np.ndarray,
        z_pred_override: np.ndarray | None = None,
    ) -> None:
        """Generic EKF measurement update."""
        z_pred = H @ self._x if z_pred_override is None else z_pred_override
        innov = z - z_pred
        S = H @ self._P @ H.T + R
        try:
            K = self._P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self._x = self._x + K @ innov
        self._P = (np.eye(4) - K @ H) @ self._P
        # Symmetrise to prevent numerical drift
        self._P = 0.5 * (self._P + self._P.T)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> tuple[float, float, float, float]:
        return tuple(self._x)  # type: ignore[return-value]

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    @property
    def initialized(self) -> bool:
        return self._initialized


# ── VIO Fusion orchestrator ───────────────────────────────────────────────────

class VIOFusion:
    """
    Per-session VIO orchestrator.

    Combines:
      1. IMU (accelerometer + gyroscope) — high-rate predict step
      2. Optical flow (from JS client) — velocity correction
      3. Compass — heading correction
      4. VPR / GPS — absolute position reset

    Thread-safety: single-threaded async; no locking needed.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._ekf = _EKF4DOF()
        self._pose = VIOPose()

        # Drift tracking
        self._last_abs_px: float = 0.0
        self._last_abs_py: float = 0.0
        self._drift_m: float = 0.0

        # IMU state
        self._last_imu_ts: float | None = None
        self._gyro_z_rad: float = 0.0       # latest gyro yaw rate
        self._accel_fwd: float = 0.0        # latest forward accel

        # Compass complementary filter
        self._heading_comp: float | None = None  # radians, math convention

        # Optical flow state
        self._last_flow_ts: float | None = None

        # VPR re-localization state
        self._vpr_pending: bool = False
        self._last_vpr_ts: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, lat: float, lon: float, heading_deg: float) -> None:
        """
        Initialize or hard-reset the VIO to a known absolute position.
        heading_deg: compass heading (0=North, CW positive).
        """
        h_math = _compass_to_math(heading_deg)
        self._ekf.initialize(0.0, 0.0, h_math, 0.0)
        self._pose = VIOPose(
            px=0.0, py=0.0,
            heading_rad=h_math,
            origin_lat=lat, origin_lon=lon,
            source="gps",
            drift_m=0.0,
        )
        self._last_abs_px = 0.0
        self._last_abs_py = 0.0
        self._drift_m = 0.0
        self._heading_comp = h_math

    def update_imu(
        self,
        ax: float, ay: float, az: float,
        gyro_z_rad_s: float,
        compass_deg: float | None,
        dt_s: float,
    ) -> VIOPose:
        """
        High-rate IMU update (10–100 Hz).

        ax, ay, az    : accelerometer m/s² (device frame, Z up)
        gyro_z_rad_s  : yaw rate rad/s (positive = CCW)
        compass_deg   : absolute compass heading (0=North, CW) or None
        dt_s          : time since last call in seconds
        """
        if not self._ekf.initialized:
            if compass_deg is not None:
                self.reset(
                    self._pose.origin_lat or 0.0,
                    self._pose.origin_lon or 0.0,
                    compass_deg,
                )
            return self._pose

        # Clamp dt to avoid huge jumps after pauses
        dt = min(max(dt_s, 0.001), 0.5)

        # Forward acceleration in body frame.
        # Device frame (phone held upright, screen facing user):
        #   +X = right, +Y = toward top of phone (forward when walking)
        #   gravity ≈ -Y when phone is upright
        # accelerationIncludingGravity.y ≈ -9.81 when stationary.
        # Gravity-free forward accel = ay - (-g) = ay + g
        h = self._ekf.state[2]
        ay_ng = ay + _G   # remove gravity from Y axis (gravity = -9.81 on Y)
        # accel_fwd is the scalar forward acceleration along the device's +Y axis.
        # The EKF predict step projects this onto world frame via cos(h)/sin(h).
        accel_fwd = ay_ng
        # Suppress noise floor
        if abs(accel_fwd) < 0.3:
            accel_fwd = 0.0

        # Compass complementary filter (circular mean, wrap-safe)
        if compass_deg is not None:
            h_compass = _compass_to_math(compass_deg)
            if self._heading_comp is None:
                self._heading_comp = h_compass
            else:
                # Circular interpolation: blend via unit vectors to avoid wrap issues
                old_x = math.cos(self._heading_comp)
                old_y = math.sin(self._heading_comp)
                new_x = math.cos(h_compass)
                new_y = math.sin(h_compass)
                blend_x = (1.0 - _COMPASS_ALPHA) * old_x + _COMPASS_ALPHA * new_x
                blend_y = (1.0 - _COMPASS_ALPHA) * old_y + _COMPASS_ALPHA * new_y
                self._heading_comp = math.atan2(blend_y, blend_x)
            self._ekf.update_heading(self._heading_comp, _R_COMPASS)

        # EKF predict
        self._ekf.predict(dt, gyro_z=gyro_z_rad_s, accel_fwd=accel_fwd)

        self._sync_pose("imu")
        return self._pose

    def update_optical_flow(
        self,
        flow_x_px: float,
        flow_y_px: float,
        dt_s: float,
    ) -> VIOPose:
        """
        Optical flow correction (~5 Hz from JS client).

        flow_x_px, flow_y_px : mean feature displacement in pixels
                                (positive X = rightward, positive Y = downward)
        dt_s                 : time since last frame in seconds
        """
        if not self._ekf.initialized:
            return self._pose

        dt = min(max(dt_s, 0.01), 1.0)

        # Convert pixels → metres (camera model: pinhole, ~60° FOV)
        flow_x_m = flow_x_px / _FLOW_PX_PER_M
        flow_y_m = -flow_y_px / _FLOW_PX_PER_M  # invert Y (image Y down, world Y up)

        self._ekf.update_optical_flow(flow_x_m, flow_y_m, dt)
        self._sync_pose("flow")
        return self._pose

    def relocalize(
        self,
        lat: float,
        lon: float,
        heading_deg: float | None,
        accuracy_m: float = 3.0,
    ) -> VIOPose:
        """
        Absolute position correction from GPS or VPR match.
        Converts lat/lon to ENU metres relative to session origin,
        then applies EKF position update.
        """
        if self._pose.origin_lat is None:
            # First fix — initialize
            self.reset(lat, lon, heading_deg or 0.0)
            return self._pose

        # Convert to ENU relative to origin
        lat_m = 111320.0
        lon_m = 111320.0 * math.cos(math.radians(self._pose.origin_lat))
        px_abs = (lon - self._pose.origin_lon) * lon_m
        py_abs = (lat - self._pose.origin_lat) * lat_m

        # EKF position update with accuracy-scaled noise
        r_m2 = max(accuracy_m ** 2, 0.25)
        self._ekf.update_position(px_abs, py_abs, r_m2)

        if heading_deg is not None:
            h_math = _compass_to_math(heading_deg)
            self._ekf.update_heading(h_math, _R_COMPASS * 2)
            self._heading_comp = h_math

        # Reset drift counter
        px, py, _, _ = self._ekf.state
        self._last_abs_px = px
        self._last_abs_py = py
        self._drift_m = 0.0

        self._sync_pose("vpr" if accuracy_m < 5.0 else "gps")
        return self._pose

    def get_pose(self) -> VIOPose:
        return self._pose

    @property
    def drift_m(self) -> float:
        return self._drift_m

    @property
    def needs_relocalization(self) -> bool:
        return self._drift_m > VPR_DRIFT_TRIGGER_M

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sync_pose(self, source: str) -> None:
        """Copy EKF state into the public VIOPose and update drift."""
        px, py, h, v = self._ekf.state
        P = self._ekf.covariance

        # Drift = distance from last absolute fix
        self._drift_m = math.sqrt(
            (px - self._last_abs_px) ** 2 + (py - self._last_abs_py) ** 2
        )

        self._pose = VIOPose(
            px=px, py=py,
            heading_rad=h,
            speed_ms=max(0.0, v),
            cov_px=float(P[0, 0]),
            cov_py=float(P[1, 1]),
            cov_heading=float(P[2, 2]),
            origin_lat=self._pose.origin_lat,
            origin_lon=self._pose.origin_lon,
            source=source,
            drift_m=self._drift_m,
            updated_at=datetime.utcnow(),
        )


# ── Per-session registry ──────────────────────────────────────────────────────

class VIORegistry:
    """Holds one VIOFusion instance per session."""

    def __init__(self) -> None:
        self._sessions: dict[str, VIOFusion] = {}

    def get_or_create(self, session_id: str) -> VIOFusion:
        if session_id not in self._sessions:
            self._sessions[session_id] = VIOFusion(session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> VIOFusion | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# Singleton
vio_registry = VIORegistry()


# ── Utility ───────────────────────────────────────────────────────────────────

def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _compass_to_math(compass_deg: float) -> float:
    """
    Convert compass heading (0=North, CW) to math convention (0=East, CCW).
    math_rad = (90 - compass_deg) * π/180
    """
    return math.radians(90.0 - compass_deg)
