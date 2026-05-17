/**
 * floor.js — Floor Detection (Barometer + Step Counter) — Bước 1
 * Depends on: globals.js
 */
'use strict';

// ── Constants ─────────────────────────────────────────────────────────────────
const FLOOR_HEIGHT_M  = 3.2;
const HPA_PER_METRE   = 0.1198;
const HPA_PER_FLOOR   = FLOOR_HEIGHT_M * HPA_PER_METRE;  // ≈ 0.383
const STAIR_Z_THRESH  = 1.5;   // m/s² above gravity
const STAIR_WIN_S     = 2.0;
const STAIR_CAD_MIN   = 0.8;   // Hz
const STAIR_CAD_MAX   = 2.5;   // Hz
const ELEV_OFFSET_MIN = 0.3;   // m/s²
const ELEV_SUSTAIN_S  = 0.8;
const GRAVITY         = 9.81;

// ── State ─────────────────────────────────────────────────────────────────────
const floorState = {
  pressureHpa:    null,
  baselineHpa:    null,
  pressureHistory: [],
  accelBuf:       [],   // [{ts, ax, ay, az, norm}]
  stepCount:      0,
  lastStepTs:     0,
  elevOnsetTs:    null,
  floor:          1,
  confidence:     0,
  method:         'none',
  lastSentTs:     0,
  SEND_INTERVAL_MS: 800,
};

// ── Pressure → floor ──────────────────────────────────────────────────────────
function _floorFromPressure() {
  if (floorState.baselineHpa === null || floorState.pressureHpa === null) return null;
  const delta = floorState.baselineHpa - floorState.pressureHpa;
  if (Math.abs(delta) < 0.05) return { floor: floorState.floor, conf: 0.5 };
  const deltaM = delta / HPA_PER_METRE;
  const raw    = 1 + deltaM / FLOOR_HEIGHT_M;
  const floor  = Math.max(1, Math.round(raw));
  const frac   = Math.abs(raw - Math.round(raw));
  const conf   = Math.max(0.3, 1.0 - frac * 2.0);
  return { floor, conf };
}

// ── Stair classifier ──────────────────────────────────────────────────────────
function _classifyStairs() {
  const now    = performance.now() / 1000;
  const cutoff = now - STAIR_WIN_S;
  while (floorState.accelBuf.length && floorState.accelBuf[0].ts < cutoff) {
    floorState.accelBuf.shift();
  }
  const buf = floorState.accelBuf;
  if (buf.length < 4) return 0;
  const peaks    = buf.filter(s => s.norm - GRAVITY > STAIR_Z_THRESH).length;
  const duration = buf[buf.length - 1].ts - buf[0].ts;
  if (duration < 0.1) return 0;
  const cadence  = peaks / duration;
  if (cadence < STAIR_CAD_MIN || cadence > STAIR_CAD_MAX) return 0;
  const azMean   = buf.reduce((s, x) => s + x.az, 0) / buf.length;
  return azMean < -0.5 ? 1 : azMean > 0.5 ? -1 : 1;
}

// ── Fusion ────────────────────────────────────────────────────────────────────
function _fuseFloor() {
  const baro     = _floorFromPressure();
  const stairDir = _classifyStairs();
  let floor = floorState.floor, conf = 0, method = 'none';

  if (baro !== null && stairDir !== 0) {
    if (baro.floor === floorState.floor + stairDir) {
      // Barometer and step-counter agree → high confidence fusion
      floor = baro.floor; conf = Math.min(1, 0.65 * baro.conf + 0.35 * 0.6 + 0.15);
      method = 'barometer+step';
    } else {
      // Disagreement → trust barometer but reduce confidence
      floor = baro.floor; conf = baro.conf * 0.7; method = 'barometer';
    }
  } else if (baro !== null) {
    floor = baro.floor; conf = baro.conf; method = 'barometer';
  } else if (stairDir !== 0) {
    floor = Math.max(1, floorState.floor + stairDir); conf = 0.4; method = 'step';
  }

  floorState.floor      = Math.max(1, floor);
  floorState.confidence = conf;
  floorState.method     = method;
}

// ── HUD update ────────────────────────────────────────────────────────────────
function _updateFloorHUD() {
  const hud = el('floor-hud');
  if (!hud) return;
  hud.style.display = '';  // always show — user can tap to calibrate
  const conf  = floorState.confidence;
  const color = conf >= 0.7 ? 'var(--green)'
              : conf >= 0.4 ? 'var(--amber)'
              : floorState.method === 'none' ? 'var(--text3)' : 'var(--text3)';
  hud.style.color = color;
  if (floorState.method === 'none') {
    hud.textContent = '🏢 Tầng ?';
    hud.title = 'Bấm để nhập tầng thủ công';
  } else {
    hud.textContent = `🏢 Tầng ${floorState.floor}`;
    hud.title = `${(conf * 100).toFixed(0)}% · ${floorState.method} · Bấm để hiệu chỉnh`;
  }
}

// ── API send (throttled) ──────────────────────────────────────────────────────
async function _sendFloorUpdate() {
  const now = Date.now();
  if (now - floorState.lastSentTs < floorState.SEND_INTERVAL_MS) return;
  floorState.lastSentTs = now;

  const body = { session_id: sid };
  if (floorState.pressureHpa !== null) body.pressure_hpa = floorState.pressureHpa;
  const buf = floorState.accelBuf;
  if (buf.length) {
    const last = buf[buf.length - 1];
    body.accel_x = last.ax; body.accel_y = last.ay; body.accel_z = last.az;
  }
  try {
    const r = await fetchWithTimeout(API + '/api/realtime/floor', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }, 5000);
    if (r.ok) {
      const d = await r.json();
      if (d.floor) {
        floorState.floor      = d.floor.floor;
        floorState.confidence = d.floor.confidence;
        floorState.method     = d.floor.method;
        _updateFloorHUD();
      }
    }
  } catch (e) { /* network error — use local estimate */ }
}

// ── Barometer init ────────────────────────────────────────────────────────────
function _initBarometer() {
  const _onReading = pressure => {
    floorState.pressureHpa = pressure;
    if (floorState.baselineHpa === null) {
      floorState.pressureHistory.push(pressure);
      if (floorState.pressureHistory.length >= 3) {
        const sorted = [...floorState.pressureHistory].sort((a, b) => a - b);
        floorState.baselineHpa = sorted[Math.floor(sorted.length / 2)];
      }
    }
    _fuseFloor(); _updateFloorHUD(); _sendFloorUpdate();
  };

  if (typeof DevicePressureEvent !== 'undefined') {
    try {
      const sensor = new DevicePressureEvent();
      sensor.addEventListener('reading', () => _onReading(sensor.pressure));
      sensor.start(); return;
    } catch (e) { /* not available */ }
  }
  if (typeof Barometer !== 'undefined') {
    try {
      const baro = new Barometer({ frequency: 1 });
      baro.addEventListener('reading', () => _onReading(baro.pressure));
      baro.start();
    } catch (e) { /* not available */ }
  }
}

// ── Step detector ─────────────────────────────────────────────────────────────
function _initStepDetector() {
  if (typeof DeviceMotionEvent === 'undefined') return;
  window.addEventListener('devicemotion', evt => {
    const a = evt.accelerationIncludingGravity;
    if (!a) return;
    const ax = a.x || 0, ay = a.y || 0, az = a.z || 0;
    const norm = Math.sqrt(ax * ax + ay * ay + az * az);
    const now  = performance.now() / 1000;

    floorState.accelBuf.push({ ts: now, ax, ay, az, norm });
    const cutoff = now - 3.0;
    while (floorState.accelBuf.length && floorState.accelBuf[0].ts < cutoff) {
      floorState.accelBuf.shift();
    }
    if (norm - GRAVITY > STAIR_Z_THRESH && now - floorState.lastStepTs > 0.25) {
      floorState.stepCount++; floorState.lastStepTs = now;
    }
    const zOffset = Math.abs(norm - GRAVITY);
    floorState.elevOnsetTs = zOffset >= ELEV_OFFSET_MIN
      ? (floorState.elevOnsetTs ?? now)
      : null;

    _fuseFloor(); _updateFloorHUD(); _sendFloorUpdate();
  }, { passive: true });
}

// ── Public init ───────────────────────────────────────────────────────────────
function initFloorDetection() {
  _initBarometer();
  _initStepDetector();
  _initGpsAltitude();
}

// ── GPS altitude → floor ──────────────────────────────────────────────────────
// GeolocationCoordinates.altitude is available on most phones (GPS chip).
// Accuracy is ±3–15m, enough to distinguish floors (3.2m each) with calibration.
let _gpsAltBaseline = null;   // altitude at floor 1 (set on first fix or calibration)

function _initGpsAltitude() {
  // Hook into the existing GPS watcher — gps.js calls _vioOnGpsFix which we extend
  // We listen for position updates via a separate watcher here
  if (!navigator.geolocation) return;
  navigator.geolocation.watchPosition(pos => {
    const alt = pos.coords.altitude;
    if (alt === null || alt === undefined) return;  // not available on this device
    const acc = pos.coords.altitudeAccuracy || 99;
    if (acc > 20) return;  // too inaccurate to use

    // Set baseline on first good reading (assume user is on floor 1)
    if (_gpsAltBaseline === null) {
      _gpsAltBaseline = alt;
      return;
    }

    const deltaM = alt - _gpsAltBaseline;
    const rawFloor = 1 + deltaM / FLOOR_HEIGHT_M;
    const floor = Math.max(1, Math.round(rawFloor));
    const frac  = Math.abs(rawFloor - Math.round(rawFloor));
    const conf  = Math.max(0.2, 0.8 - frac * 1.5) * Math.min(1, 10 / acc);

    // Only update if GPS altitude gives higher confidence than current method
    if (conf > floorState.confidence || floorState.method === 'none') {
      floorState.floor      = floor;
      floorState.confidence = conf;
      floorState.method     = 'gps_altitude';
      _updateFloorHUD();
    }
  }, () => {}, { enableHighAccuracy: true, maximumAge: 5000 });
}

// ── Manual floor calibration ──────────────────────────────────────────────────
function openFloorModal() {
  const modal = el('floor-modal');
  if (modal) {
    el('floor-cal-input').value = floorState.floor;
    modal.style.display = 'flex';
  }
}

function closeFloorModal() {
  const modal = el('floor-modal');
  if (modal) modal.style.display = 'none';
}

async function calibrateFloor() {
  const floor = parseInt(el('floor-cal-input').value) || 1;

  // Update local state immediately
  floorState.floor      = floor;
  floorState.confidence = 1.0;
  floorState.method     = 'manual';
  _updateFloorHUD();

  // Reset GPS altitude baseline to current altitude at this floor
  // (next GPS reading will recalibrate relative to this floor)
  _gpsAltBaseline = null;  // will be reset on next GPS altitude reading

  // Reset barometer baseline
  if (floorState.pressureHpa !== null) {
    // Adjust baseline so current pressure maps to this floor
    floorState.baselineHpa = floorState.pressureHpa + (floor - 1) * HPA_PER_FLOOR;
  }

  // Send to server
  try {
    await fetchWithTimeout(API + '/api/realtime/floor/calibrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sid, floor }),
    }, 5000);
  } catch (e) { /* offline — local calibration still applied */ }

  closeFloorModal();
  toast(`🏢 Đã hiệu chỉnh: Tầng ${floor}`, 'ok');
}
