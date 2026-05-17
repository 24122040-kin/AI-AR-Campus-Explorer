/**
 * vio_client.js — VIO Lite client for LocalNavBot
 *
 * Implements:
 *   1. Optical flow via canvas 2D (Lucas-Kanade sparse tracker, JS-native)
 *   2. IMU integration (DeviceMotionEvent → gyro + accel)
 *   3. Complementary filter heading (DeviceOrientation)
 *   4. Throttled POST to /api/realtime/vio/imu and /api/realtime/vio/flow
 *   5. VPR re-localization trigger when server signals drift > threshold
 *
 * Design goals:
 *   - Zero external dependencies (pure JS + Canvas 2D API)
 *   - Works on Android Chrome and iOS Safari 16+
 *   - Graceful degradation: IMU-only if camera unavailable
 *   - < 5 ms per optical flow frame on mid-range phone
 *
 * Public API:
 *   VIOClient.init(sessionId, videoElement)  → Promise<void>
 *   VIOClient.start()
 *   VIOClient.stop()
 *   VIOClient.onPose(callback)   // callback(vio_pose_dict)
 *   VIOClient.isRunning()        → bool
 *   VIOClient.getLatestPose()    → dict | null
 */

'use strict';

const VIOClient = (() => {

  // ── Configuration ──────────────────────────────────────────────────────────
  const CFG = {
    // Optical flow
    FLOW_GRID_COLS:    8,      // feature grid columns
    FLOW_GRID_ROWS:    6,      // feature grid rows
    FLOW_WIN_HALF:     4,      // Lucas-Kanade window half-size (pixels)
    FLOW_MAX_ITER:     8,      // max LK iterations
    FLOW_EPS:          0.03,   // LK convergence threshold
    FLOW_MAX_DISP_PX:  40,     // reject features with displacement > this
    FLOW_MIN_FEATURES: 6,      // minimum tracked features to trust flow
    FLOW_INTERVAL_MS:  200,    // optical flow computation interval

    // IMU
    IMU_INTERVAL_MS:   50,     // IMU send interval (20 Hz)
    IMU_GYRO_THRESH:   0.02,   // rad/s — ignore below this (noise floor)
    IMU_ACCEL_THRESH:  0.3,    // m/s² — ignore below this

    // API
    API_BASE:          '',
    VIO_IMU_PATH:      '/api/realtime/vio/imu',
    VIO_FLOW_PATH:     '/api/realtime/vio/flow',
    VIO_RELOC_PATH:    '/api/realtime/vio/relocalize',

    // Drift
    DRIFT_WARN_M:      1.5,
    DRIFT_RELOC_M:     2.0,
  };

  // ── State ──────────────────────────────────────────────────────────────────
  let _sid          = null;
  let _video        = null;
  let _running      = false;
  let _poseCallback = null;
  let _latestPose   = null;

  // Canvas for optical flow
  let _flowCanvas   = null;
  let _flowCtx      = null;
  let _prevFrame    = null;   // ImageData of previous frame
  let _prevFeatures = [];     // [{x, y}] grid points in previous frame
  let _flowTimer    = null;
  let _lastFlowTs   = 0;

  // IMU state
  let _imuTimer     = null;
  let _lastImuTs    = 0;
  let _ax = 0, _ay = 0, _az = 0;
  let _gyroZ        = 0;      // rad/s yaw rate
  let _compassDeg   = null;   // degrees, 0=North

  // ── Optical Flow (Lucas-Kanade sparse) ────────────────────────────────────

  /**
   * Sample a bilinear-interpolated pixel value from a flat RGBA array.
   * Returns the luminance (0–255).
   */
  function _sampleLuma(data, w, x, y) {
    const xi = x | 0, yi = y | 0;
    if (xi < 0 || yi < 0 || xi >= w - 1 || yi >= data.length / (4 * w) - 1) return 0;
    const fx = x - xi, fy = y - yi;
    const i00 = (yi * w + xi) * 4;
    const i10 = i00 + 4;
    const i01 = i00 + w * 4;
    const i11 = i01 + 4;
    // Luminance: 0.299R + 0.587G + 0.114B
    const l = (r, g, b) => 0.299 * r + 0.587 * g + 0.114 * b;
    const v00 = l(data[i00], data[i00+1], data[i00+2]);
    const v10 = l(data[i10], data[i10+1], data[i10+2]);
    const v01 = l(data[i01], data[i01+1], data[i01+2]);
    const v11 = l(data[i11], data[i11+1], data[i11+2]);
    return (1-fx)*(1-fy)*v00 + fx*(1-fy)*v10 + (1-fx)*fy*v01 + fx*fy*v11;
  }

  /**
   * Compute image gradient at (x, y) using central differences.
   * Returns {gx, gy}.
   */
  function _gradient(data, w, x, y) {
    return {
      gx: (_sampleLuma(data, w, x+1, y) - _sampleLuma(data, w, x-1, y)) * 0.5,
      gy: (_sampleLuma(data, w, x, y+1) - _sampleLuma(data, w, x, y-1)) * 0.5,
    };
  }

  /**
   * Lucas-Kanade optical flow for a single feature point.
   * Returns {dx, dy, valid} — displacement in pixels.
   */
  function _lkTrack(prevData, currData, w, h, px, py) {
    const win = CFG.FLOW_WIN_HALF;
    let vx = 0, vy = 0;

    for (let iter = 0; iter < CFG.FLOW_MAX_ITER; iter++) {
      let Ixx = 0, Ixy = 0, Iyy = 0, Ixt = 0, Iyt = 0;

      for (let dy = -win; dy <= win; dy++) {
        for (let dx = -win; dx <= win; dx++) {
          const x = px + dx, y = py + dy;
          const xw = px + dx + vx, yw = py + dy + vy;
          if (x < 1 || y < 1 || x >= w-1 || y >= h-1) continue;
          if (xw < 1 || yw < 1 || xw >= w-1 || yw >= h-1) continue;

          const { gx, gy } = _gradient(prevData, w, x, y);
          const It = _sampleLuma(currData, w, xw, yw) - _sampleLuma(prevData, w, x, y);

          Ixx += gx * gx;
          Ixy += gx * gy;
          Iyy += gy * gy;
          Ixt += gx * It;
          Iyt += gy * It;
        }
      }

      // Solve 2×2 system: [Ixx Ixy; Ixy Iyy] * [dvx; dvy] = -[Ixt; Iyt]
      const det = Ixx * Iyy - Ixy * Ixy;
      if (Math.abs(det) < 1e-6) return { dx: 0, dy: 0, valid: false };

      const dvx = -(Iyy * Ixt - Ixy * Iyt) / det;
      const dvy = -(Ixx * Iyt - Ixy * Ixt) / det;
      vx += dvx;
      vy += dvy;

      if (Math.abs(dvx) < CFG.FLOW_EPS && Math.abs(dvy) < CFG.FLOW_EPS) break;
    }

    const mag = Math.sqrt(vx*vx + vy*vy);
    return {
      dx: vx, dy: vy,
      valid: mag < CFG.FLOW_MAX_DISP_PX && !isNaN(vx) && !isNaN(vy),
    };
  }

  /**
   * Build a uniform grid of feature points, avoiding image borders.
   */
  function _buildGrid(w, h) {
    const pts = [];
    const cols = CFG.FLOW_GRID_COLS, rows = CFG.FLOW_GRID_ROWS;
    const margin = CFG.FLOW_WIN_HALF + 2;
    const stepX = (w - 2*margin) / (cols - 1);
    const stepY = (h - 2*margin) / (rows - 1);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        pts.push({ x: margin + c * stepX, y: margin + r * stepY });
      }
    }
    return pts;
  }

  /**
   * Compute mean optical flow across all valid tracked features.
   * Returns {flowX, flowY, count} in pixels.
   */
  function _computeFlow(prevData, currData, w, h, features) {
    let sumX = 0, sumY = 0, count = 0;
    for (const pt of features) {
      const { dx, dy, valid } = _lkTrack(prevData, currData, w, h, pt.x, pt.y);
      if (valid) { sumX += dx; sumY += dy; count++; }
    }
    if (count < CFG.FLOW_MIN_FEATURES) return { flowX: 0, flowY: 0, count };
    return { flowX: sumX / count, flowY: sumY / count, count };
  }

  // ── Flow loop ──────────────────────────────────────────────────────────────

  function _captureFrame() {
    if (!_video || !_flowCanvas || !_flowCtx) return null;
    const vw = _video.videoWidth || 320;
    const vh = _video.videoHeight || 240;
    // Downscale to 160×120 for speed
    const W = 160, H = Math.round(vh / vw * 160) || 120;
    _flowCanvas.width = W;
    _flowCanvas.height = H;
    _flowCtx.drawImage(_video, 0, 0, W, H);
    return _flowCtx.getImageData(0, 0, W, H);
  }

  // Recursive setTimeout instead of setInterval to avoid stacking async calls
  function _scheduleFlow() {
    if (!_running) return;
    _flowTimer = setTimeout(async () => {
      try { await _flowTick(); } catch(e) { /* ignore */ }
      _scheduleFlow();
    }, CFG.FLOW_INTERVAL_MS);
  }

  async function _flowTick() {
    const now = performance.now();
    const dt_s = (now - _lastFlowTs) / 1000;
    _lastFlowTs = now;

    const currFrame = _captureFrame();
    if (currFrame && _prevFrame && _prevFeatures.length > 0) {
      const W = currFrame.width, H = currFrame.height;
      const { flowX, flowY, count } = _computeFlow(
        _prevFrame.data, currFrame.data, W, H, _prevFeatures
      );

      if (count >= CFG.FLOW_MIN_FEATURES) {
        await _sendFlow(flowX, flowY, dt_s);
      }

      // Rebuild grid every ~10 ticks to handle scene changes
      if (Math.random() < 0.1) {
        _prevFeatures = _buildGrid(W, H);
      }
    }

    if (currFrame) {
      _prevFrame = currFrame;
      if (_prevFeatures.length === 0) {
        _prevFeatures = _buildGrid(currFrame.width, currFrame.height);
      }
    }
  }

  // ── IMU ────────────────────────────────────────────────────────────────────

  function _onDeviceMotion(evt) {
    const a = evt.accelerationIncludingGravity;
    if (!a) return;
    _ax = a.x || 0;
    _ay = a.y || 0;
    _az = a.z || 0;

    const r = evt.rotationRate;
    if (r) {
      // rotationRate.alpha = yaw rate (deg/s) — convert to rad/s
      // Note: on iOS, rotationRate.alpha is the yaw rate around Z axis
      _gyroZ = (r.alpha || 0) * Math.PI / 180;
    }
  }

  function _onDeviceOrientation(evt) {
    if (evt.webkitCompassHeading !== undefined) {
      _compassDeg = evt.webkitCompassHeading;
    } else if (evt.absolute && evt.alpha !== null) {
      _compassDeg = (360 - evt.alpha) % 360;
    }
  }

  // Recursive setTimeout for IMU to avoid stacking async fetch calls
  function _scheduleImu() {
    if (!_running) return;
    _imuTimer = setTimeout(async () => {
      try { await _imuTick(); } catch(e) { /* ignore */ }
      _scheduleImu();
    }, CFG.IMU_INTERVAL_MS);
  }

  async function _imuTick() {
    const now = performance.now();
    const dt_s = Math.min((now - _lastImuTs) / 1000, 0.5);
    _lastImuTs = now;

    // Suppress noise
    const gyroZ = Math.abs(_gyroZ) > CFG.IMU_GYRO_THRESH ? _gyroZ : 0;

    await _sendImu(_ax, _ay, _az, gyroZ, _compassDeg, dt_s);
  }

  // ── API calls ──────────────────────────────────────────────────────────────

  async function _sendImu(ax, ay, az, gyroZ, compassDeg, dt_s) {
    try {
      const body = {
        session_id: _sid,
        ax, ay, az,
        gyro_z: gyroZ,
        dt_s,
      };
      if (compassDeg !== null) body.compass_deg = compassDeg;

      const r = await fetch(CFG.API_BASE + CFG.VIO_IMU_PATH, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!r.ok) return;
      const d = await r.json();
      _handlePoseResponse(d);
    } catch (e) { /* network error — continue */ }
  }

  async function _sendFlow(flowX, flowY, dt_s) {
    try {
      const r = await fetch(CFG.API_BASE + CFG.VIO_FLOW_PATH, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: _sid,
          flow_x_px: flowX,
          flow_y_px: flowY,
          dt_s,
        }),
      });
      if (!r.ok) return;
      const d = await r.json();
      _handlePoseResponse(d);
    } catch (e) { /* network error */ }
  }

  function _handlePoseResponse(d) {
    if (!d || !d.vio_pose) return;
    _latestPose = d.vio_pose;
    if (_poseCallback) _poseCallback(d.vio_pose);

    // Trigger VPR re-localization if server requests it
    if (d.needs_relocalization || d.vpr_requested) {
      window.dispatchEvent(new CustomEvent('vio-needs-relocalization', {
        detail: { drift_m: d.vio_pose.drift_m, session_id: _sid }
      }));
    }

    // Dispatch pose update event for AR renderer
    window.dispatchEvent(new CustomEvent('vio-pose-update', {
      detail: d.vio_pose
    }));
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  async function init(sessionId, videoElement) {
    _sid = sessionId;
    _video = videoElement || null;

    // Create offscreen canvas for optical flow
    _flowCanvas = document.createElement('canvas');
    _flowCtx = _flowCanvas.getContext('2d', { willReadFrequently: true });

    // IMU listeners
    window.addEventListener('devicemotion', _onDeviceMotion, { passive: true });
    window.addEventListener('deviceorientation', _onDeviceOrientation, { passive: true });
    window.addEventListener('deviceorientationabsolute', _onDeviceOrientation, { passive: true });
  }

  function start() {
    if (_running) return;
    _running = true;
    _lastImuTs = performance.now();
    _lastFlowTs = performance.now();

    // IMU loop (recursive setTimeout — avoids stacking async fetches)
    _scheduleImu();

    // Optical flow loop — check video is actually streaming
    // Re-check _video at start time in case camera was opened after init()
    if (!_video) {
      // Try to find the camera preview element
      const v = document.getElementById('cam-preview');
      if (v && v.srcObject) _video = v;
    }
    if (_video && _video.srcObject) {
      _scheduleFlow();
    }
  }

  function stop() {
    _running = false;
    if (_imuTimer) { clearTimeout(_imuTimer); _imuTimer = null; }
    if (_flowTimer) { clearTimeout(_flowTimer); _flowTimer = null; }
    _prevFrame = null;
    _prevFeatures = [];
    window.removeEventListener('devicemotion', _onDeviceMotion);
    window.removeEventListener('deviceorientation', _onDeviceOrientation);
    window.removeEventListener('deviceorientationabsolute', _onDeviceOrientation);
  }

  function onPose(callback) { _poseCallback = callback; }
  function isRunning() { return _running; }
  function getLatestPose() { return _latestPose; }

  /**
   * Manually trigger a GPS/VPR re-localization.
   * Call this when you have a fresh GPS fix or VPR match.
   */
  async function relocalize(lat, lon, headingDeg, accuracyM, source = 'gps') {
    try {
      const r = await fetch(CFG.API_BASE + CFG.VIO_RELOC_PATH, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: _sid,
          lat, lon,
          heading_deg: headingDeg,
          accuracy_m: accuracyM || 5.0,
          source,
        }),
      });
      if (!r.ok) return null;
      const d = await r.json();
      if (d.vio_pose) {
        _latestPose = d.vio_pose;
        if (_poseCallback) _poseCallback(d.vio_pose);
      }
      return d;
    } catch (e) { return null; }
  }

  return { init, start, stop, onPose, isRunning, getLatestPose, relocalize };

})();

window.VIOClient = VIOClient;
