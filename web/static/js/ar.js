/**
 * ar.js — AR Navigation (Passthrough mặc định)
 * Depends on: globals.js, route.js (_lastArPath, _lastRouteSteps)
 * External: Three.js, /ar_renderer.js (WebXR / Compass — không tự bật)
 */
'use strict';

let _arReady = false;
let _arOn = false;
let _passthroughOn = false;

let _arHazardLastTs = 0;
let _arHazardStepKey = '';

// ── Init WebXR / Compass renderer (chỉ khi gọi tay _initAR + ARRenderer.start) ─
async function _initAR() {
  if (_arReady) return true;
  if (typeof THREE === 'undefined') { toast('Three.js chưa tải xong, thử lại', 'warn'); return false; }
  if (typeof ARRenderer === 'undefined') { toast('AR module chưa tải', 'warn'); return false; }
  const ok = await ARRenderer.init('ar-canvas');
  if (!ok) { toast('Không khởi tạo được AR canvas', 'warn'); return false; }

  window.addEventListener('ar-mode-change', e => {
    const mode = e.detail.mode;
    const badge = el('ar-mode-badge');
    if (badge) {
      badge.textContent = mode === 'webxr' ? '🥽 WebXR' : mode === 'compass' ? '🧭 Compass 2D' : mode === 'passthrough' ? '📷 Passthrough' : '';
      badge.className = mode !== 'none' ? 'visible' : '';
    }
    if (mode === 'webxr') toast('🥽 WebXR AR đang chạy', 'ok');
    if (mode === 'compass') toast('🧭 Compass 2D — hướng theo la bàn', 'ok');
  });
  _arReady = true;
  return true;
}

// ── Feed route data into AR renderer (WebXR / Compass) ────────────────────────
function _arFeedRoute(arPath, steps) {
  if (!window.ARRenderer) return;
  ARRenderer.setArPath(arPath);
  if (steps && steps.length) ARRenderer.setNextInstruction(steps[0].instruction, steps[0].distance_m);
  const pois = steps.slice(0, 8).map(s => ({
    name: s.instruction.length > 30 ? s.instruction.slice(0, 28) + '…' : s.instruction,
    lat: s.lat,
    lon: s.lon,
    distance_m: s.distance_m,
    floor: 0,
    type: s.maneuver || 'waypoint',
  }));
  ARRenderer.setPois(pois);
}

// ── Haversine (GPS → bước kế tiếp) ───────────────────────────────────────────
function _arHaversineM(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const toR = x => x * Math.PI / 180;
  const dLat = toR(lat2 - lat1);
  const dLon = toR(lon2 - lon1);
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(toR(lat1)) * Math.cos(toR(lat2)) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/** Gọi từ vòng GPS: cảnh báo hazard khi gần bước kế (< 30m), có throttle */
function _arMaybeCheckHazardsFromGps() {
  if (!_passthroughOn && !_arOn) return;
  const steps = (typeof _lastRouteSteps !== 'undefined' && _lastRouteSteps) ? _lastRouteSteps : [];
  if (!steps.length || curLat == null || curLon == null) return;
  const s0 = steps[0];
  if (!s0 || s0.lat == null || s0.lon == null) return;
  const d = _arHaversineM(curLat, curLon, s0.lat, s0.lon);
  if (d >= 30) return;
  const key = (s0.instruction || '') + '|' + (s0.maneuver || '');
  const now = Date.now();
  if (now - _arHazardLastTs < 12000 && _arHazardStepKey === key) return;
  _arHazardLastTs = now;
  _arHazardStepKey = key;
  _arCheckStepHazards(s0);
}

// ── Toggle AR: chỉ Passthrough (camera + overlay), không bật ARRenderer ───────
async function toggleAR() {
  if (_arOn) { stopAR(); return; }

  if (curLat == null || curLon == null || isNaN(curLat) || isNaN(curLon)) {
    return toast('Cần GPS để bật AR Navigation', 'warn');
  }
  if (!_lastArPath || !_lastArPath.points || !_lastArPath.points.length) {
    return toast('Hãy tìm đường trước khi bật AR', 'warn');
  }
  if (!window.AREnhanced) return toast('AR Enhanced chưa tải', 'warn');

  AREnhanced.setRoute(_lastArPath, _lastRouteSteps);
  AREnhanced.setUserPose(curLat, curLon, _userHeading, floorState.floor);
  await AREnhanced.startPassthrough();
  _passthroughOn = true;
  _arOn = true;

  el('ar-overlay').classList.add('active');
  el('ar-close-btn').classList.add('visible');
  el('ar-instruction').classList.add('visible');
  const instr0 = _lastRouteSteps && _lastRouteSteps[0];
  const ie = el('ar-instruction');
  if (ie && instr0 && instr0.instruction) {
    const dm = instr0.distance_m > 0
      ? ' · ' + (instr0.distance_m >= 1000 ? (instr0.distance_m / 1000).toFixed(1) + ' km' : Math.round(instr0.distance_m) + ' m')
      : '';
    ie.textContent = instr0.instruction + dm;
  }
  const badge = el('ar-mode-badge');
  if (badge) {
    badge.textContent = '📷 Passthrough';
    badge.classList.add('visible');
  }

  const btn = el('btn-ar-nav');
  if (btn) { btn.textContent = '⏹ Tắt AR'; btn.style.background = 'var(--red)'; }

  if (typeof startVIO === 'function') startVIO();
  
  // Enable VPR auto-relocalization
  if (window.VPRRelocalization) {
    VPRRelocalization.enable();
    console.log('[AR] VPR auto-relocalization enabled');
  }
}

function stopAR() {
  if (_passthroughOn && window.AREnhanced) {
    AREnhanced.stopPassthrough();
    _passthroughOn = false;
  }
  if (window.ARRenderer && typeof ARRenderer.isActive === 'function' && ARRenderer.isActive()) {
    ARRenderer.stop();
  }
  _arOn = false;

  el('ar-overlay').classList.remove('active');
  el('ar-close-btn').classList.remove('visible');
  el('ar-instruction').classList.remove('visible');
  el('ar-mode-badge').classList.remove('visible');
  const overlay = el('ar-compass-overlay');
  if (overlay) overlay.style.display = 'none';

  const btn = el('btn-ar-nav');
  if (btn) { btn.textContent = '📷 Bật AR Navigation'; btn.style.background = ''; }

  if (typeof stopVIO === 'function') stopVIO();
  
  // Disable VPR auto-relocalization
  if (window.VPRRelocalization) {
    VPRRelocalization.disable();
    console.log('[AR] VPR auto-relocalization disabled');
  }
}

function closeAllAR() {
  stopAR();
}

// ── Passthrough: gộp vào nút chính (nút phụ ẩn — giữ API) ───────────────────
function togglePassthroughAR() {
  toggleAR();
}

// ── Slope / hazard warning ────────────────────────────────────────────────────
let _arWarnTimer = null;
function _arShowWarning(msg) {
  const w = el('ar-warning');
  if (!w) return;
  w.textContent = msg;
  w.classList.add('show');
  clearTimeout(_arWarnTimer);
  _arWarnTimer = setTimeout(() => w.classList.remove('show'), 4000);
}

function _arCheckStepHazards(step) {
  if (!step) return;
  const maneuver = step.maneuver || '';
  const slope = step.slope_deg || 0;
  const surface = step.surface || '';
  const covered = step.is_covered;

  if (maneuver === 'stairs') _arShowWarning('⚠️ Cầu thang phía trước');
  else if (maneuver === 'elevator') _arShowWarning('🛗 Thang máy phía trước');
  else if (maneuver === 'ramp' || Math.abs(slope) > 10)
    _arShowWarning(`⚠️ Đường dốc ${slope > 0 ? 'lên' : 'xuống'} ${Math.abs(slope).toFixed(0)}°`);
  else if (surface === 'grass') _arShowWarning('⚠️ Đường cỏ / đất — cẩn thận trơn trượt');
}

// ── Update from VIO state (indoor positioning) ────────────────────────────────
function _arUpdateFromVIO(vioState) {
  if (!vioState || !window.AREnhanced) return;
  
  // Convert VIO ENU position back to lat/lon if origin is set
  let lat = curLat;
  let lon = curLon;
  
  if (vioState.origin_lat && vioState.origin_lon) {
    // Convert ENU metres to lat/lon offset
    const lat_m = 111320.0;
    const lon_m = 111320.0 * Math.cos(vioState.origin_lat * Math.PI / 180);
    lat = vioState.origin_lat + vioState.py / lat_m;
    lon = vioState.origin_lon + vioState.px / lon_m;
  }
  
  const heading = vioState.heading_deg || _userHeading;
  
  // Update AR with VIO position
  AREnhanced.setUserPose(lat, lon, heading, floorState.floor);
  
  // Show VIO drift warning if > 2m
  if (vioState.drift_m > 2.0) {
    _arShowWarning(`⚠️ VIO drift: ${vioState.drift_m.toFixed(1)}m - đang định vị lại...`);
  }
  
  // Update badge to show VIO mode
  const badge = el('ar-mode-badge');
  if (badge && vioState.source === 'vpr') {
    badge.textContent = '🎯 VIO (VPR)';
  } else if (badge && vioState.source === 'flow') {
    badge.textContent = '📹 VIO (Flow)';
  } else if (badge && vioState.source === 'imu') {
    badge.textContent = '📱 VIO (IMU)';
  }
}

// ── Handle VPR relocalization success ──────────────────────────────────────────
function _arHandleVPRRelocalization(event) {
  const { location_name, lat, lon, score, vio_pose } = event.detail;
  
  console.log('[AR] VPR relocalization:', location_name, 'score:', score);
  
  // Update AR with new position
  if (vio_pose) {
    _arUpdateFromVIO(vio_pose);
  } else if (lat && lon) {
    // Fallback: update with GPS position
    AREnhanced.setUserPose(lat, lon, _userHeading, floorState.floor);
  }
  
  // Show success notification
  _arShowWarning(`✅ VIO relocalized: ${location_name}`);
  
  // Update badge
  const badge = el('ar-mode-badge');
  if (badge) {
    badge.textContent = '🎯 VIO (VPR)';
  }
}

// Listen for VPR relocalization events
window.addEventListener('vpr-relocalized', _arHandleVPRRelocalization);

// ── Handle floor transitions (stairs/elevator) ────────────────────────────────
let _floorTransitionTimer = null;
function _arHandleFloorTransition(arPath) {
  if (!arPath || !arPath.has_transition) return;
  
  const overlay = el('ar-floor-transition');
  if (!overlay) return;
  
  const type = arPath.transition_type;
  const target = arPath.target_floor;
  const current = arPath.current_floor || floorState.floor;
  const direction = arPath.transition_direction;
  
  // Build transition card HTML
  let icon = '🚶';
  let action = 'Đi bộ';
  if (type === 'stairs') {
    icon = direction === 'up' ? '⬆️' : '⬇️';
    action = direction === 'up' ? 'Lên cầu thang' : 'Xuống cầu thang';
  } else if (type === 'elevator') {
    icon = '🛗';
    action = direction === 'up' ? 'Lên thang máy' : 'Xuống thang máy';
  } else if (type === 'ramp') {
    icon = direction === 'up' ? '↗️' : '↘️';
    action = direction === 'up' ? 'Lên dốc' : 'Xuống dốc';
  }
  
  overlay.innerHTML = `
    <div class="floor-transition-card">
      <div class="floor-transition-icon">${icon}</div>
      <div class="floor-transition-content">
        <div class="floor-transition-action">${action}</div>
        <div class="floor-transition-target">Tầng ${current} → Tầng ${target}</div>
      </div>
      <div class="floor-transition-arrow">${direction === 'up' ? '↑' : '↓'}</div>
    </div>
  `;
  
  overlay.classList.add('show');
  
  // Auto-hide after 5 seconds
  clearTimeout(_floorTransitionTimer);
  _floorTransitionTimer = setTimeout(() => {
    overlay.classList.remove('show');
  }, 5000);
  
  // Speak instruction if available
  if (typeof SpeechModule !== 'undefined') {
    SpeechModule.speak(`${action} đến tầng ${target}`, 'high');
  }
}

// ── Update from realtime WebSocket state ──────────────────────────────────────
function _arUpdateFromRealtimeState(state) {
  if (!window.AREnhanced) return;

  // Update route if available
  if (_lastArPath) {
    AREnhanced.setRoute(_lastArPath, _lastRouteSteps);
    
    // Check for floor transitions
    _arHandleFloorTransition(_lastArPath);
  }
  
  // Update VIO position if available
  if (state.vio_state) {
    _arUpdateFromVIO(state.vio_state);
  } else if (curLat) {
    // Fallback to GPS
    AREnhanced.setUserPose(curLat, curLon, _userHeading, floorState.floor);
  }

  const vis = state.latest_scene_state;
  if (vis) {
    window.dispatchEvent(new CustomEvent('realtime-scene-update', { detail: vis }));
  }

  const arActive = _arOn || _passthroughOn;
  if (!arActive) return;

  const instr = state.latest_instruction;
  if (instr && instr.instruction) {
    if (_arOn && window.ARRenderer && typeof ARRenderer.isActive === 'function' && ARRenderer.isActive()) {
      ARRenderer.setNextInstruction(instr.instruction, 0);
    }
    const instrEl = el('ar-instruction');
    if (instrEl) instrEl.textContent = instr.instruction;
    if (typeof SpeechModule !== 'undefined' && instr.urgency === 'high') {
      SpeechModule.speak(instr.instruction, 'high');
    }
  }
  const floor = state.latest_floor;
  if (floor && curLat && window.ARRenderer && typeof ARRenderer.isActive === 'function' && ARRenderer.isActive()) {
    ARRenderer.setUserPose(curLat, curLon, _userHeading, floor.floor || 1);
  }
}
