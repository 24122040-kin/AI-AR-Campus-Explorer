/**
 * vio.js — VIO (Visual-Inertial Odometry) client integration — Bước 4
 * Depends on: globals.js, ar.js, gps.js, floor.js
 * External: /vio_client.js
 */
'use strict';

let _vioRunning = false;

// ── Init ──────────────────────────────────────────────────────────────────────
async function initVIO() {
  if (typeof VIOClient === 'undefined') return;

  // Use camera stream if already open, otherwise IMU-only
  const videoEl = el('cam-preview');
  await VIOClient.init(sid, videoEl && videoEl.srcObject ? videoEl : null);

  // Pose callback → update HUD + AR renderer
  VIOClient.onPose(pose => {
    _updateVioHud(pose);
    if (_arOn && window.ARRenderer && pose.origin_lat) {
      const lat = pose.origin_lat + pose.py / 111320;
      const lon = pose.origin_lon + pose.px / (111320 * Math.cos(pose.origin_lat * Math.PI / 180));
      ARRenderer.setUserPose(lat, lon, pose.heading_deg, floorState.floor);
    }
  });

  // Server requests re-localization when drift > threshold
  window.addEventListener('vio-needs-relocalization', async e => {
    const drift = e.detail.drift_m || 0;
    toast(`📡 VIO drift ${drift.toFixed(1)}m — đang re-localize...`, 'warn');
    if (curLat && curLon) {
      await VIOClient.relocalize(curLat, curLon, _userHeading, _gpsAccuracyM, 'gps');
    } else if (el('cam-preview')?.srcObject) {
      _vioTriggerVprFrame();
    }
  });
}

// ── HUD ───────────────────────────────────────────────────────────────────────
function _updateVioHud(pose) {
  const hud = el('vio-hud');
  if (!hud) return;
  hud.style.display = '';
  const drift   = pose.drift_m || 0;
  const src     = pose.source || 'imu';
  const srcIcon = src === 'vpr' ? '🎯' : src === 'flow' ? '👁' : src === 'gps' ? '📍' : '📡';
  if (drift > 1.5) {
    hud.style.color = 'var(--amber)';
    hud.textContent = `${srcIcon} ${drift.toFixed(1)}m drift`;
  } else {
    hud.style.color = 'var(--green)';
    hud.textContent = `${srcIcon} VIO ✓`;
  }
  hud.title = `VIO: px=${pose.px?.toFixed(1)}m py=${pose.py?.toFixed(1)}m drift=${drift.toFixed(2)}m source=${src}`;
}

// ── VPR frame trigger ─────────────────────────────────────────────────────────
async function _vioTriggerVprFrame() {
  const v = el('cam-preview');
  if (!v || !v.srcObject) return;
  const c = document.createElement('canvas');
  c.width = v.videoWidth || 320; c.height = v.videoHeight || 240;
  c.getContext('2d').drawImage(v, 0, 0);
  c.toBlob(async blob => {
    if (!blob) return;
    const fd = new FormData();
    fd.append('file', new File([blob], 'vio_frame.jpg', { type: 'image/jpeg' }));
    fd.append('session_id', sid);
    if (curLat) { fd.append('lat', curLat); fd.append('lon', curLon); }
    try {
      const r = await fetch(API + '/api/realtime/frame', { method: 'POST', body: fd });
      const d = await r.json();
      if (d.vpr_relocalized) toast('🎯 VPR re-localized!', 'ok');
    } catch (e) { /* ignore */ }
  }, 'image/jpeg', 0.75);
}

// ── Start / Stop ──────────────────────────────────────────────────────────────
function startVIO() {
  if (_vioRunning || typeof VIOClient === 'undefined') return;
  VIOClient.start();
  _vioRunning = true;
  if (curLat && curLon) VIOClient.relocalize(curLat, curLon, _userHeading, _gpsAccuracyM, 'gps');
}

function stopVIO() {
  if (!_vioRunning || typeof VIOClient === 'undefined') return;
  VIOClient.stop();
  _vioRunning = false;
  const hud = el('vio-hud');
  if (hud) hud.style.display = 'none';
}

// ── GPS hook ──────────────────────────────────────────────────────────────────
function _vioOnGpsFix(lat, lon, accuracy) {
  if (_vioRunning && window.VIOClient) {
    VIOClient.relocalize(lat, lon, _userHeading, accuracy, 'gps');
  }
}
