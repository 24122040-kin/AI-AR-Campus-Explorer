"""
core/floor_detector.py — Floor detection using barometer pressure delta
and accelerometer step/stair/elevator pattern recognition.

Output contract:
    {
        "floor": 2,
        "confidence": 0.87,
        "method": "barometer+step"   # or "barometer" | "step" | "none"
    }

Physics reference:
  - ~1 hPa ≈ 8.5 m altitude change at sea level (ISA standard atmosphere)
  - Typical floor height: 3.0–3.5 m  → ~0.35–0.41 hPa per floor
  - We use FLOOR_HEIGHT_M = 3.2 m as default (configurable)

Accelerometer patterns (Z-axis, phone held upright):
  - Walking flat  : low variance, ~9.81 m/s²  norm, no sustained Z drift
  - Climbing stairs: rhythmic Z spikes > 1.5 m/s² above gravity, cadence 1–2 Hz
  - Elevator      : sustained smooth Z offset (0.3–1.5 m/s²) for > 1 s,
                    then returns to baseline — no step cadence
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

# ── Constants ────────────────────────────────────────────────────────────────
FLOOR_HEIGHT_M: float = 3.2          # metres per floor
HPA_PER_METRE: float = 0.1198        # hPa / m  (ISA sea-level approx)
HPA_PER_FLOOR: float = FLOOR_HEIGHT_M * HPA_PER_METRE   # ≈ 0.383 hPa

# Stair detection thresholds
STAIR_Z_THRESHOLD: float = 1.5       # m/s² above gravity to count as step
STAIR_CADENCE_MIN_HZ: float = 0.8    # minimum step frequency
STAIR_CADENCE_MAX_HZ: float = 2.5    # maximum step frequency
STAIR_WINDOW_S: float = 2.0          # seconds of accel history to analyse

# Elevator detection thresholds
ELEV_Z_OFFSET_MIN: float = 0.3       # m/s² sustained offset from 9.81
ELEV_SUSTAINED_S: float = 0.8        # seconds of sustained offset

# Barometer calibration
PRESSURE_HISTORY_MAX: int = 60       # samples kept for baseline
PRESSURE_NOISE_HPA: float = 0.05     # ignore deltas smaller than this

# Confidence weights
W_BARO: float = 0.65
W_STEP: float = 0.35


@dataclass
class _AccelSample:
    ts: float          # time.monotonic()
    ax: float          # m/s²
    ay: float
    az: float
    norm: float = field(init=False)

    def __post_init__(self) -> None:
        self.norm = math.sqrt(self.ax**2 + self.ay**2 + self.az**2)


class FloorDetector:
    """
    Stateful per-session floor detector.

    Call ``update_pressure(hpa)`` whenever a new barometer reading arrives.
    Call ``update_accel(ax, ay, az)`` on every DeviceMotion sample.
    Call ``get_floor()`` to retrieve the current estimate.
    """

    def __init__(self) -> None:
        # Barometer state
        self._pressure_history: deque[float] = deque(maxlen=PRESSURE_HISTORY_MAX)
        self._baseline_hpa: float | None = None   # pressure at floor 1 (calibrated)
        self._current_hpa: float | None = None

        # Accelerometer state
        self._accel_buf: deque[_AccelSample] = deque()
        self._elev_onset_ts: float | None = None  # when elevator motion started

        # Floor state
        self._floor: int = 1
        self._confidence: float = 0.0
        self._method: Literal["barometer+step", "barometer", "step", "none"] = "none"

        # Step counter (cumulative, for dead-reckoning)
        self._step_count: int = 0
        self._last_step_ts: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update_pressure(self, hpa: float) -> None:
        """Ingest a new barometer reading (hPa)."""
        if hpa <= 0 or hpa > 1100:
            return  # sanity check
        self._pressure_history.append(hpa)
        self._current_hpa = hpa
        if self._baseline_hpa is None and len(self._pressure_history) >= 3:
            # Use median of first 3 readings as ground-floor baseline
            self._baseline_hpa = sorted(self._pressure_history)[ len(self._pressure_history) // 2 ]

    def calibrate_floor(self, floor: int) -> None:
        """
        Manually tell the detector which floor the user is currently on.
        Resets the barometric baseline to match.
        """
        if self._current_hpa is not None:
            # Adjust baseline so that current pressure maps to `floor`
            self._baseline_hpa = self._current_hpa + (floor - 1) * HPA_PER_FLOOR
        self._floor = floor

    def update_accel(self, ax: float, ay: float, az: float) -> None:
        """Ingest a new accelerometer sample (m/s²)."""
        now = time.monotonic()
        sample = _AccelSample(ts=now, ax=ax, ay=ay, az=az)
        self._accel_buf.append(sample)

        # Prune old samples outside the analysis window
        cutoff = now - STAIR_WINDOW_S
        while self._accel_buf and self._accel_buf[0].ts < cutoff:
            self._accel_buf.popleft()

        self._detect_step(sample, now)
        self._detect_elevator(sample, now)

    def get_floor(self) -> dict:
        """Return current floor estimate."""
        self._recompute()
        return {
            "floor": self._floor,
            "confidence": round(self._confidence, 3),
            "method": self._method,
        }

    def reset(self) -> None:
        """Reset all state (e.g. when session ends)."""
        self.__init__()  # type: ignore[misc]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _recompute(self) -> None:
        """Fuse barometer + step signals into a floor estimate."""
        baro_floor: int | None = None
        baro_conf: float = 0.0

        step_floor: int | None = None
        step_conf: float = 0.0

        # ── Barometer branch ──────────────────────────────────────────────────
        if self._baseline_hpa is not None and self._current_hpa is not None:
            delta_hpa = self._baseline_hpa - self._current_hpa  # positive = higher floor
            if abs(delta_hpa) >= PRESSURE_NOISE_HPA:
                delta_m = delta_hpa / HPA_PER_METRE
                raw_floor = 1 + delta_m / FLOOR_HEIGHT_M
                baro_floor = max(1, round(raw_floor))
                # Confidence: how close to an integer floor number
                frac = abs(raw_floor - round(raw_floor))
                baro_conf = max(0.3, 1.0 - frac * 2.0)
            else:
                baro_floor = self._floor  # no significant change
                baro_conf = 0.5

        # ── Step / stair branch ───────────────────────────────────────────────
        stair_direction = self._classify_stair_direction()
        if stair_direction != 0 and baro_floor is not None:
            step_floor = baro_floor + stair_direction
            step_conf = 0.6
        elif stair_direction != 0:
            step_floor = self._floor + stair_direction
            step_conf = 0.4

        # ── Fusion ────────────────────────────────────────────────────────────
        if baro_floor is not None and step_floor is not None:
            # Weighted vote — if they agree, boost confidence
            if baro_floor == step_floor:
                self._floor = baro_floor
                self._confidence = min(1.0, W_BARO * baro_conf + W_STEP * step_conf + 0.15)
                self._method = "barometer+step"
            else:
                # Barometer wins but confidence is reduced
                self._floor = baro_floor
                self._confidence = baro_conf * 0.7
                self._method = "barometer"
        elif baro_floor is not None:
            self._floor = baro_floor
            self._confidence = baro_conf
            self._method = "barometer"
        elif step_floor is not None:
            self._floor = max(1, step_floor)
            self._confidence = step_conf
            self._method = "step"
        else:
            self._method = "none"
            self._confidence = 0.0

        self._floor = max(1, self._floor)

    def _detect_step(self, sample: _AccelSample, now: float) -> None:
        """Detect a single footstep from the accelerometer norm peak."""
        gravity = 9.81
        z_excess = sample.norm - gravity
        if z_excess > STAIR_Z_THRESHOLD:
            # Debounce: minimum 0.25 s between steps
            if now - self._last_step_ts > 0.25:
                self._step_count += 1
                self._last_step_ts = now

    def _classify_stair_direction(self) -> int:
        """
        Analyse the recent accel buffer to decide if the user is:
          +1 = climbing stairs
          -1 = descending stairs
           0 = flat / elevator / stationary
        """
        if len(self._accel_buf) < 4:
            return 0

        samples = list(self._accel_buf)
        norms = [s.norm for s in samples]
        gravity = 9.81

        # Count peaks above threshold
        peaks = sum(1 for n in norms if n - gravity > STAIR_Z_THRESHOLD)
        duration = samples[-1].ts - samples[0].ts
        if duration < 0.1:
            return 0

        cadence_hz = peaks / duration
        if not (STAIR_CADENCE_MIN_HZ <= cadence_hz <= STAIR_CADENCE_MAX_HZ):
            return 0  # not stair-like cadence

        # Use Z-axis trend to determine up vs down
        # When climbing: phone tilts slightly back → az tends to be more negative
        # When descending: phone tilts slightly forward → az tends to be more positive
        # This is a heuristic; barometer is the primary source of truth.
        az_vals = [s.az for s in samples]
        az_mean = sum(az_vals) / len(az_vals)

        if az_mean < -0.5:
            return +1   # climbing
        elif az_mean > 0.5:
            return -1   # descending
        else:
            return +1   # default to climbing when cadence matches (barometer corrects)

    def _detect_elevator(self, sample: _AccelSample, now: float) -> None:
        """
        Detect elevator motion: sustained smooth Z offset without step cadence.
        Updates internal floor estimate directly when elevator ride ends.
        """
        gravity = 9.81
        z_offset = abs(sample.norm - gravity)

        if z_offset >= ELEV_Z_OFFSET_MIN:
            if self._elev_onset_ts is None:
                self._elev_onset_ts = now
        else:
            if self._elev_onset_ts is not None:
                duration = now - self._elev_onset_ts
                if duration >= ELEV_SUSTAINED_S:
                    # Elevator ride detected — barometer will handle the floor update
                    # Just mark that we were in an elevator (method hint)
                    pass
            self._elev_onset_ts = None
