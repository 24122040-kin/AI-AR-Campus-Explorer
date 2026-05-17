/**
 * vpr_reloc.js — VPR Auto-Relocalization for VIO Drift Correction
 * 
 * Listens for VIO drift events and automatically triggers VPR to relocalize.
 * Depends on: globals.js, VIOClient
 */
'use strict';

const VPRRelocalization = (() => {

  // ── Configuration ──────────────────────────────────────────────────────────
  const CFG = {
    DRIFT_THRESHOLD_M: 2.0,        // Trigger VPR when drift > this
    MIN_INTERVAL_MS: 10000,        // Min time between VPR attempts (10s)
    VPR_TIMEOUT_MS: 8000,          // VPR query timeout
    MAX_RETRIES: 3,                // Max VPR retries before giving up
  };

  // ── State ──────────────────────────────────────────────────────────────────
  let _enabled = false;
  let _lastVprTs = 0;
  let _retryCount = 0;
  let _isRelocating = false;

  // ── VPR Query ──────────────────────────────────────────────────────────────

  /**
   * Capture current camera frame and query VPR for location match.
   * Returns VPR match result or null if failed.
   */
  async function _queryVPR() {
    try {
      // Get camera preview element
      const video = el('cam-preview');
      if (!video || !video.srcObject) {
        console.warn('[VPR] Camera not available for VPR query');
        return null;
      }

      // Check video is actually playing
      if (video.paused || video.readyState < 2) {
        console.warn('[VPR] Camera not ready (paused or loading)');
        return null;
      }

      // Capture frame to canvas
      const canvas = document.createElement('canvas');
      const vw = video.videoWidth || video.clientWidth || 640;
      const vh = video.videoHeight || video.clientHeight || 480;
      
      if (vw === 0 || vh === 0) {
        console.warn('[VPR] Invalid video dimensions');
        return null;
      }

      canvas.width = vw;
      canvas.height = vh;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, vw, vh);

      // Convert to blob
      const blob = await new Promise(resolve => {
        canvas.toBlob(resolve, 'image/jpeg', 0.85);
      });

      if (!blob) {
        console.warn('[VPR] Failed to create image blob');
        return null;
      }

      // Query VPR via experimental/vpr endpoint
      const fd = new FormData();
      fd.append('file', new File([blob], `vpr_${Date.now()}.jpg`, { type: 'image/jpeg' }));
      
      // Add approximate GPS for proximity re-ranking
      if (curLat && curLon) {
        fd.append('lat', curLat);
        fd.append('lon', curLon);
      }

      console.log('[VPR] Querying VPR engine...');
      
      const r = await fetchWithTimeout(
        API + '/api/experimental/vpr',
        { method: 'POST', body: fd },
        CFG.VPR_TIMEOUT_MS
      );

      if (!r.ok) {
        console.warn('[VPR] VPR query failed:', r.status);
        return null;
      }

      const data = await r.json();
      
      if (!data.ok || !data.matches || data.matches.length === 0) {
        console.warn('[VPR] No VPR matches found');
        return null;
      }

      const best = data.matches[0];
      console.log('[VPR] Best match:', best.location_name, 'score:', best.score);

      return best;

    } catch (e) {
      console.error('[VPR] VPR query error:', e);
      return null;
    }
  }

  /**
   * Relocalize VIO with VPR match.
   */
  async function _relocalize(vprMatch) {
    try {
      if (!window.VIOClient) {
        console.warn('[VPR] VIOClient not available');
        return false;
      }

      console.log('[VPR] Relocalizing VIO with VPR match:', vprMatch.location_name);

      // Call VIOClient.relocalize()
      const result = await VIOClient.relocalize(
        vprMatch.lat,
        vprMatch.lon,
        null,  // heading_deg (let VIO keep current heading)
        Math.max(1.5, vprMatch.distance_m * 0.15),  // accuracy_m
        'vpr'
      );

      if (!result || !result.ok) {
        console.warn('[VPR] VIO relocalization failed');
        return false;
      }

      console.log('[VPR] VIO relocalized successfully, drift reset');

      // Show success notification
      toast(`📍 VIO relocalized: ${vprMatch.location_name}`, 'ok');

      // Dispatch event for AR to update
      window.dispatchEvent(new CustomEvent('vpr-relocalized', {
        detail: {
          location_name: vprMatch.location_name,
          lat: vprMatch.lat,
          lon: vprMatch.lon,
          score: vprMatch.score,
          vio_pose: result.vio_pose,
        }
      }));

      return true;

    } catch (e) {
      console.error('[VPR] Relocalization error:', e);
      return false;
    }
  }

  // ── Event Handlers ─────────────────────────────────────────────────────────

  /**
   * Handle VIO drift event from VIOClient.
   */
  async function _onVIODrift(event) {
    if (!_enabled || _isRelocating) return;

    const { drift_m, session_id } = event.detail;

    // Check drift threshold
    if (drift_m < CFG.DRIFT_THRESHOLD_M) return;

    // Throttle VPR attempts
    const now = Date.now();
    if (now - _lastVprTs < CFG.MIN_INTERVAL_MS) {
      console.log('[VPR] Throttled (too soon since last attempt)');
      return;
    }

    // Check retry limit
    if (_retryCount >= CFG.MAX_RETRIES) {
      console.warn('[VPR] Max retries reached, giving up');
      toast('⚠️ VIO drift cao, không thể relocalize tự động', 'warn');
      return;
    }

    _isRelocating = true;
    _lastVprTs = now;
    _retryCount++;

    console.log(`[VPR] VIO drift ${drift_m.toFixed(2)}m > ${CFG.DRIFT_THRESHOLD_M}m, triggering VPR (attempt ${_retryCount}/${CFG.MAX_RETRIES})`);

    // Show loading indicator
    toast('🔍 Đang relocalize VIO bằng VPR...', 'info');

    try {
      // Query VPR
      const vprMatch = await _queryVPR();

      if (!vprMatch) {
        console.warn('[VPR] VPR query failed or no match');
        toast('⚠️ VPR không tìm thấy vị trí phù hợp', 'warn');
        return;
      }

      // Relocalize VIO
      const success = await _relocalize(vprMatch);

      if (success) {
        // Reset retry counter on success
        _retryCount = 0;
      }

    } finally {
      _isRelocating = false;
    }
  }

  /**
   * Handle VPR relocalization success event.
   */
  function _onVPRRelocalized(event) {
    const { location_name, score } = event.detail;
    console.log('[VPR] Relocalization confirmed:', location_name, 'score:', score);
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Enable VPR auto-relocalization.
   */
  function enable() {
    if (_enabled) return;
    _enabled = true;
    _retryCount = 0;

    // Listen for VIO drift events
    window.addEventListener('vio-needs-relocalization', _onVIODrift);
    window.addEventListener('vpr-relocalized', _onVPRRelocalized);

    console.log('[VPR] Auto-relocalization enabled');
  }

  /**
   * Disable VPR auto-relocalization.
   */
  function disable() {
    if (!_enabled) return;
    _enabled = false;

    window.removeEventListener('vio-needs-relocalization', _onVIODrift);
    window.removeEventListener('vpr-relocalized', _onVPRRelocalized);

    console.log('[VPR] Auto-relocalization disabled');
  }

  /**
   * Manually trigger VPR relocalization (for testing).
   */
  async function trigger() {
    if (_isRelocating) {
      console.warn('[VPR] Already relocating');
      return false;
    }

    console.log('[VPR] Manual VPR relocalization triggered');
    toast('🔍 Đang relocalize VIO bằng VPR...', 'info');

    _isRelocating = true;
    try {
      const vprMatch = await _queryVPR();
      if (!vprMatch) {
        toast('⚠️ VPR không tìm thấy vị trí phù hợp', 'warn');
        return false;
      }

      const success = await _relocalize(vprMatch);
      return success;

    } finally {
      _isRelocating = false;
    }
  }

  /**
   * Check if VPR relocalization is enabled.
   */
  function isEnabled() {
    return _enabled;
  }

  /**
   * Get current state.
   */
  function getState() {
    return {
      enabled: _enabled,
      isRelocating: _isRelocating,
      retryCount: _retryCount,
      lastVprTs: _lastVprTs,
    };
  }

  return {
    enable,
    disable,
    trigger,
    isEnabled,
    getState,
  };

})();

// Export to global scope
window.VPRRelocalization = VPRRelocalization;
