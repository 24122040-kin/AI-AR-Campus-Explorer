/**
 * gps.js — GPS watcher + device orientation (heading)
 * Depends on: globals.js, ar.js, vio.js
 */
'use strict';

let _userHeading = 0;

// Absolute heading preferred (Android), fallback to relative (iOS)
window.addEventListener('deviceorientationabsolute', e => {
  if (e.alpha !== null) _userHeading = (360 - e.alpha) % 360;
}, { passive: true });

window.addEventListener('deviceorientation', e => {
  if (e.webkitCompassHeading !== undefined) _userHeading = e.webkitCompassHeading;
  else if (e.alpha !== null) _userHeading = (360 - e.alpha) % 360;
}, { passive: true });

let _gpsWatchId = null;

function getGPS() {
  if (!navigator.geolocation) return toast('GPS không hỗ trợ', 'warn');
  if (location.protocol !== 'https:' &&
      location.hostname !== 'localhost' &&
      location.hostname !== '127.0.0.1') {
    toast('GPS cần HTTPS trên thiết bị thực', 'warn');
  }
  // Avoid stacking multiple watchers
  if (_gpsWatchId !== null) return;
  _gpsWatchId = navigator.geolocation.watchPosition(pos => {
    const lat = pos.coords.latitude;
    const lon = pos.coords.longitude;
    if (lat == null || lon == null || isNaN(lat) || isNaN(lon)) return;
    curLat = lat;
    curLon = lon;
    _gpsAccuracyM = pos.coords.accuracy || 5.0;
    el('gps-txt').textContent = `${curLat.toFixed(5)}, ${curLon.toFixed(5)}`;
    el('gps-dot').classList.add('live');
    el('d-lat').value = curLat.toFixed(6);
    el('d-lon').value = curLon.toFixed(6);
    // Update AR renderer pose
    if (_arOn && window.ARRenderer) {
      ARRenderer.setUserPose(curLat, curLon, _userHeading, floorState.floor);
    }
    if (window.AREnhanced) {
      AREnhanced.setUserPose(curLat, curLon, _userHeading, floorState.floor);
    }
    if (typeof _arMaybeCheckHazardsFromGps === 'function') _arMaybeCheckHazardsFromGps();
    // Feed VIO re-localization
    _vioOnGpsFix(curLat, curLon, _gpsAccuracyM);
  }, e => {
    _gpsWatchId = null; // allow retry on next call
    const msg = e.code === 1 ? 'Bị từ chối quyền GPS'
              : e.code === 2 ? 'Không xác định được vị trí'
              : 'GPS timeout';
    toast('GPS: ' + msg, 'warn');
  }, { enableHighAccuracy: true, timeout: 15000, maximumAge: 5000 });
}
