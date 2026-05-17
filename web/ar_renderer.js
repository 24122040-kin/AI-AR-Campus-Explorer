/**
 * ar_renderer.js — WebXR AR Renderer for LocalNavBot
 *
 * Renders 3D navigation arrows and POI billboard labels anchored to the
 * real world using WebXR (ARCore on Android Chrome / ARKit via WebXR on
 * iOS Safari 16+). Falls back to a 2D compass overlay when WebXR is
 * unavailable.
 *
 * Dependencies (loaded via CDN in ui.html):
 *   - Three.js r165  (3D rendering)
 *   - No other framework dependencies
 *
 * Public API:
 *   ARRenderer.init(canvasId)          → Promise<bool>  (true = WebXR, false = 2D fallback)
 *   ARRenderer.setArPath(arPathData)   → void
 *   ARRenderer.setPois(poiList)        → void
 *   ARRenderer.setUserPose(lat, lon, headingDeg, floor) → void
 *   ARRenderer.setNextInstruction(text, distanceM) → void
 *   ARRenderer.start()                 → void
 *   ARRenderer.stop()                  → void
 *   ARRenderer.isActive()              → bool
 *
 * ar_path data format (from /api/route → ar_path):
 *   {
 *     reference: { lat, lon, alt },
 *     points: [{ east_m, north_m, up_m, lat, lon, index }, ...]
 *   }
 *
 * POI format:
 *   [{ name, distance_m, lat, lon, floor, type }, ...]
 */

'use strict';

const ARRenderer = (() => {

  // ── Constants ──────────────────────────────────────────────────────────────
  const ARROW_COLOR        = 0x6366f1;   // indigo-500
  const ARROW_HOVER_COLOR  = 0x22c55e;   // green-500 (next waypoint)
  const LABEL_BG_COLOR     = 'rgba(15,23,42,0.82)';
  const LABEL_TEXT_COLOR   = '#e2e8f0';
  const LABEL_ACCENT_COLOR = '#6366f1';
  const ARROW_HEIGHT_M     = 1.6;        // metres above ground
  const ARROW_SCALE        = 0.35;       // Three.js units
  const LABEL_SCALE        = 0.6;
  const MAX_VISIBLE_ARROWS = 8;          // render at most N arrows ahead
  const MAX_VISIBLE_POIS   = 6;
  const POI_MAX_DIST_M     = 80;
  const COMPASS_UPDATE_MS  = 100;

  // ── State ──────────────────────────────────────────────────────────────────
  let _canvas       = null;
  let _renderer     = null;
  let _scene        = null;
  let _camera       = null;
  let _xrSession    = null;
  let _xrRefSpace   = null;
  let _animFrameId  = null;
  let _mode         = 'none';            // 'webxr' | 'compass' | 'none'
  let _active       = false;

  // Data
  let _arPath       = null;             // { reference, points[] }
  let _pois         = [];
  let _userLat      = null;
  let _userLon      = null;
  let _userHeading  = 0;                // degrees, 0=North
  let _userFloor    = 1;
  let _nextInstruction = '';
  let _nextDistM    = 0;

  // Three.js objects
  let _arrowGroup   = null;
  let _labelGroup   = null;
  let _arrowMeshes  = [];
  let _labelSprites = [];

  // 2D compass overlay
  let _compassEl    = null;
  let _compassCtx   = null;
  let _compassTimer = null;

  // ── Geometry helpers ───────────────────────────────────────────────────────

  /** ENU (east, north) → Three.js XZ plane (X=east, Z=-north, Y=up) */
  function enuToThree(east_m, north_m, up_m = 0) {
    return new THREE.Vector3(east_m, up_m + ARROW_HEIGHT_M, -north_m);
  }

  /** Haversine distance in metres */
  function haversineM(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2
            + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180)
            * Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  /** Bearing from (lat1,lon1) to (lat2,lon2) in degrees [0,360) */
  function bearingDeg(lat1, lon1, lat2, lon2) {
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const y = Math.sin(dLon) * Math.cos(lat2 * Math.PI / 180);
    const x = Math.cos(lat1 * Math.PI / 180) * Math.sin(lat2 * Math.PI / 180)
            - Math.sin(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.cos(dLon);
    return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
  }

  // ── Arrow geometry ─────────────────────────────────────────────────────────

  function _makeArrowMesh(color) {
    const group = new THREE.Group();

    // Shaft — thin cylinder
    const shaftGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.55, 8);
    const shaftMat = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.4,
      roughness: 0.4,
      metalness: 0.3,
    });
    const shaft = new THREE.Mesh(shaftGeo, shaftMat);
    shaft.position.y = 0.275;
    group.add(shaft);

    // Head — cone
    const headGeo = new THREE.ConeGeometry(0.14, 0.35, 8);
    const headMat = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.6,
      roughness: 0.3,
      metalness: 0.4,
    });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 0.725;
    group.add(head);

    // Glow ring at base
    const ringGeo = new THREE.TorusGeometry(0.18, 0.025, 8, 24);
    const ringMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    group.add(ring);

    group.scale.setScalar(ARROW_SCALE);
    return group;
  }

  // ── Billboard label sprite ─────────────────────────────────────────────────

  function _makeLabelSprite(name, distM, floor) {
    const W = 320, H = 80;
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');

    // Background pill
    ctx.fillStyle = LABEL_BG_COLOR;
    _roundRect(ctx, 0, 0, W, H, 14);
    ctx.fill();

    // Accent left bar
    ctx.fillStyle = LABEL_ACCENT_COLOR;
    _roundRect(ctx, 0, 0, 6, H, [14, 0, 0, 14]);
    ctx.fill();

    // Name
    ctx.fillStyle = LABEL_TEXT_COLOR;
    ctx.font = 'bold 22px -apple-system, sans-serif';
    ctx.fillText(_truncate(name, 22), 18, 30);

    // Distance + floor
    ctx.fillStyle = '#94a3b8';
    ctx.font = '17px -apple-system, sans-serif';
    const distStr = distM >= 1000
      ? `${(distM / 1000).toFixed(1)} km`
      : `${Math.round(distM)} m`;
    const floorStr = floor > 0 ? ` · Tầng ${floor}` : '';
    ctx.fillText(distStr + floorStr, 18, 58);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(LABEL_SCALE * 2.0, LABEL_SCALE * 0.5, 1);
    return sprite;
  }

  function _roundRect(ctx, x, y, w, h, r) {
    if (typeof r === 'number') r = [r, r, r, r];
    ctx.beginPath();
    ctx.moveTo(x + r[0], y);
    ctx.lineTo(x + w - r[1], y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r[1]);
    ctx.lineTo(x + w, y + h - r[2]);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r[2], y + h);
    ctx.lineTo(x + r[3], y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r[3]);
    ctx.lineTo(x, y + r[0]);
    ctx.quadraticCurveTo(x, y, x + r[0], y);
    ctx.closePath();
  }

  function _truncate(str, maxLen) {
    return str.length > maxLen ? str.slice(0, maxLen - 1) + '…' : str;
  }

  // ── Three.js scene setup ───────────────────────────────────────────────────

  function _initThree(canvas) {
    _renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    });
    _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    _renderer.setSize(canvas.clientWidth || window.innerWidth,
                      canvas.clientHeight || window.innerHeight);
    _renderer.xr.enabled = true;
    _renderer.setClearColor(0x000000, 0);

    _scene = new THREE.Scene();

    _camera = new THREE.PerspectiveCamera(
      70,
      (canvas.clientWidth || window.innerWidth) / (canvas.clientHeight || window.innerHeight),
      0.01,
      200,
    );
    _scene.add(_camera);

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.7);
    _scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(0, 5, 3);
    _scene.add(dir);

    // Groups
    _arrowGroup = new THREE.Group();
    _labelGroup = new THREE.Group();
    _scene.add(_arrowGroup);
    _scene.add(_labelGroup);

    window.addEventListener('resize', _onResize);
  }

  function _onResize() {
    if (!_renderer || !_canvas) return;
    const w = _canvas.clientWidth || window.innerWidth;
    const h = _canvas.clientHeight || window.innerHeight;
    _renderer.setSize(w, h);
    if (_camera) {
      _camera.aspect = w / h;
      _camera.updateProjectionMatrix();
    }
  }

  // ── Scene update ───────────────────────────────────────────────────────────

  function _rebuildScene() {
    _clearGroup(_arrowGroup);
    _clearGroup(_labelGroup);
    _arrowMeshes = [];
    _labelSprites = [];

    if (!_arPath || !_arPath.points || !_arPath.points.length) return;
    if (_userLat === null) return;

    const ref = _arPath.reference;

    // Find the closest path point to the user → start rendering from there
    let startIdx = 0;
    let minDist = Infinity;
    for (let i = 0; i < _arPath.points.length; i++) {
      const d = haversineM(_userLat, _userLon, _arPath.points[i].lat, _arPath.points[i].lon);
      if (d < minDist) { minDist = d; startIdx = i; }
    }

    // Render up to MAX_VISIBLE_ARROWS arrows ahead
    const slice = _arPath.points.slice(startIdx, startIdx + MAX_VISIBLE_ARROWS + 1);

    for (let i = 0; i < slice.length - 1; i++) {
      const from = slice[i];
      const to   = slice[i + 1];

      const posFrom = enuToThree(from.east_m, from.north_m, from.up_m);
      const posTo   = enuToThree(to.east_m,   to.north_m,   to.up_m);

      const color = i === 0 ? ARROW_HOVER_COLOR : ARROW_COLOR;
      const arrow = _makeArrowMesh(color);
      arrow.position.copy(posFrom);

      // Rotate arrow to point toward next waypoint
      const dir = new THREE.Vector3().subVectors(posTo, posFrom).normalize();
      if (dir.length() > 0.001) {
        const up = new THREE.Vector3(0, 1, 0);
        const axis = new THREE.Vector3().crossVectors(up, dir).normalize();
        const angle = Math.acos(Math.max(-1, Math.min(1, up.dot(dir))));
        if (axis.length() > 0.001) {
          arrow.setRotationFromAxisAngle(axis, angle);
        }
      }

      // Pulse animation tag
      arrow.userData.pulsePhase = i * 0.4;
      _arrowGroup.add(arrow);
      _arrowMeshes.push(arrow);
    }

    // POI labels
    const visiblePois = _pois
      .filter(p => {
        if (!p.lat || !p.lon) return false;
        const d = haversineM(_userLat, _userLon, p.lat, p.lon);
        return d <= POI_MAX_DIST_M;
      })
      .sort((a, b) => haversineM(_userLat, _userLon, a.lat, a.lon)
                    - haversineM(_userLat, _userLon, b.lat, b.lon))
      .slice(0, MAX_VISIBLE_POIS);

    for (const poi of visiblePois) {
      const distM = haversineM(_userLat, _userLon, poi.lat, poi.lon);
      // Approximate ENU for POI relative to ar_path reference
      const dLat = (poi.lat - ref.lat) * 111000;
      const dLon = (poi.lon - ref.lon) * 111000 * Math.cos(ref.lat * Math.PI / 180);
      const pos = enuToThree(dLon, dLat, 0);
      pos.y = ARROW_HEIGHT_M + 0.8;  // labels float higher

      const sprite = _makeLabelSprite(poi.name, distM, poi.floor || 0);
      sprite.position.copy(pos);
      _labelGroup.add(sprite);
      _labelSprites.push(sprite);
    }
  }

  function _clearGroup(group) {
    while (group.children.length) {
      const child = group.children[0];
      group.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (child.material.map) child.material.map.dispose();
        child.material.dispose();
      }
    }
  }

  // ── WebXR render loop ──────────────────────────────────────────────────────

  function _xrRenderLoop(time, frame) {
    if (!frame) return;
    const pose = frame.getViewerPose(_xrRefSpace);
    if (!pose) return;

    // Pulse arrows
    const t = time * 0.001;
    for (const arrow of _arrowMeshes) {
      const phase = arrow.userData.pulsePhase || 0;
      const s = ARROW_SCALE * (1 + 0.08 * Math.sin(t * 2.5 + phase));
      arrow.scale.setScalar(s);
    }

    // Billboard labels always face camera
    for (const sprite of _labelSprites) {
      sprite.quaternion.copy(_camera.quaternion);
    }

    _renderer.render(_scene, _camera);
  }

  // ── Compass (2D fallback) render loop ─────────────────────────────────────

  function _initCompass() {
    _compassEl = document.getElementById('ar-compass-canvas');
    if (!_compassEl) return;
    _compassCtx = _compassEl.getContext('2d');
    _compassTimer = setInterval(_drawCompass, COMPASS_UPDATE_MS);
  }

  function _drawCompass() {
    if (!_compassCtx || !_compassEl) return;
    const W = _compassEl.width;
    const H = _compassEl.height;
    const ctx = _compassCtx;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    const R = Math.min(cx, cy) - 8;

    // Outer ring
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(99,102,241,0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Cardinal labels
    const cardinals = [['N', 0], ['E', 90], ['S', 180], ['W', 270]];
    ctx.font = 'bold 11px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const [label, deg] of cardinals) {
      const rad = (deg - _userHeading) * Math.PI / 180;
      const x = cx + (R - 14) * Math.sin(rad);
      const y = cy - (R - 14) * Math.cos(rad);
      ctx.fillStyle = label === 'N' ? '#ef4444' : '#94a3b8';
      ctx.fillText(label, x, y);
    }

    // Route arrows projected onto compass
    if (_arPath && _arPath.points && _userLat !== null) {
      const ref = _arPath.reference;
      let startIdx = 0, minDist = Infinity;
      for (let i = 0; i < _arPath.points.length; i++) {
        const d = haversineM(_userLat, _userLon, _arPath.points[i].lat, _arPath.points[i].lon);
        if (d < minDist) { minDist = d; startIdx = i; }
      }
      const slice = _arPath.points.slice(startIdx, startIdx + 5);
      for (let i = 0; i < slice.length - 1; i++) {
        const pt = slice[i + 1];
        const bearing = bearingDeg(_userLat, _userLon, pt.lat, pt.lon);
        const relBearing = (bearing - _userHeading + 360) % 360;
        const rad = relBearing * Math.PI / 180;
        const dist = haversineM(_userLat, _userLon, pt.lat, pt.lon);
        // Scale: 0m=center, 50m=edge
        const r = Math.min(R - 20, (dist / 50) * (R - 20));
        const x = cx + r * Math.sin(rad);
        const y = cy - r * Math.cos(rad);
        const alpha = i === 0 ? 1.0 : 0.5;
        ctx.beginPath();
        ctx.arc(x, y, i === 0 ? 7 : 4, 0, Math.PI * 2);
        ctx.fillStyle = i === 0
          ? `rgba(34,197,94,${alpha})`
          : `rgba(99,102,241,${alpha})`;
        ctx.fill();
      }
    }

    // POI dots
    if (_userLat !== null) {
      for (const poi of _pois.slice(0, MAX_VISIBLE_POIS)) {
        if (!poi.lat || !poi.lon) continue;
        const dist = haversineM(_userLat, _userLon, poi.lat, poi.lon);
        if (dist > POI_MAX_DIST_M) continue;
        const bearing = bearingDeg(_userLat, _userLon, poi.lat, poi.lon);
        const relBearing = (bearing - _userHeading + 360) % 360;
        const rad = relBearing * Math.PI / 180;
        const r = Math.min(R - 20, (dist / POI_MAX_DIST_M) * (R - 20));
        const x = cx + r * Math.sin(rad);
        const y = cy - r * Math.cos(rad);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(245,158,11,0.85)';
        ctx.fill();
      }
    }

    // Heading arrow (always points up = forward)
    ctx.save();
    ctx.translate(cx, cy);
    ctx.beginPath();
    ctx.moveTo(0, -R + 4);
    ctx.lineTo(6, -R + 18);
    ctx.lineTo(-6, -R + 18);
    ctx.closePath();
    ctx.fillStyle = '#6366f1';
    ctx.fill();
    ctx.restore();

    // Next instruction overlay
    if (_nextInstruction) {
      ctx.fillStyle = 'rgba(15,23,42,0.75)';
      _roundRectCtx(ctx, 4, H - 36, W - 8, 30, 8);
      ctx.fill();
      ctx.fillStyle = '#e2e8f0';
      ctx.font = '11px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const distStr = _nextDistM > 0
        ? (_nextDistM >= 1000 ? `${(_nextDistM/1000).toFixed(1)}km` : `${Math.round(_nextDistM)}m`)
        : '';
      ctx.fillText(_truncate(_nextInstruction, 28) + (distStr ? ' · ' + distStr : ''), cx, H - 21);
    }
  }

  function _roundRectCtx(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ── WebXR session management ───────────────────────────────────────────────

  async function _startWebXR() {
    if (!navigator.xr) return false;
    const supported = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
    if (!supported) return false;

    try {
      _xrSession = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['dom-overlay', 'hit-test', 'anchors'],
        domOverlay: { root: document.getElementById('ar-overlay') || document.body },
      });

      _renderer.xr.setReferenceSpaceType('local-floor');
      await _renderer.xr.setSession(_xrSession);
      _xrRefSpace = await _xrSession.requestReferenceSpace('local-floor');

      _xrSession.addEventListener('end', () => {
        _mode = 'none';
        _active = false;
        _xrSession = null;
        _notifyModeChange('none');
      });

      _renderer.setAnimationLoop(_xrRenderLoop);
      _mode = 'webxr';
      return true;
    } catch (e) {
      console.warn('[ARRenderer] WebXR session failed:', e);
      return false;
    }
  }

  function _startCompassFallback() {
    _mode = 'compass';
    _initCompass();
    // Show compass overlay
    const overlay = document.getElementById('ar-compass-overlay');
    if (overlay) overlay.style.display = 'flex';
    // Start a simple rAF loop for Three.js preview (no XR)
    function loop() {
      if (!_active || _mode !== 'compass') return;
      _animFrameId = requestAnimationFrame(loop);
      const t = performance.now() * 0.001;
      for (const arrow of _arrowMeshes) {
        const phase = arrow.userData.pulsePhase || 0;
        const s = ARROW_SCALE * (1 + 0.08 * Math.sin(t * 2.5 + phase));
        arrow.scale.setScalar(s);
      }
      for (const sprite of _labelSprites) {
        if (_camera) sprite.quaternion.copy(_camera.quaternion);
      }
      if (_renderer && _scene && _camera) _renderer.render(_scene, _camera);
    }
    loop();
  }

  function _notifyModeChange(mode) {
    window.dispatchEvent(new CustomEvent('ar-mode-change', { detail: { mode } }));
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  async function init(canvasId = 'ar-canvas') {
    _canvas = document.getElementById(canvasId);
    if (!_canvas) {
      console.error('[ARRenderer] Canvas not found:', canvasId);
      return false;
    }

    // Check Three.js availability
    if (typeof THREE === 'undefined') {
      console.error('[ARRenderer] Three.js not loaded');
      return false;
    }

    _initThree(_canvas);
    return true;
  }

  async function start() {
    if (_active) return;
    _active = true;

    const xrOk = await _startWebXR();
    if (!xrOk) {
      _startCompassFallback();
      _notifyModeChange('compass');
    } else {
      _notifyModeChange('webxr');
    }

    _rebuildScene();
  }

  function stop() {
    _active = false;
    if (_xrSession) {
      _xrSession.end().catch(() => {});
      _xrSession = null;
    }
    if (_animFrameId) {
      cancelAnimationFrame(_animFrameId);
      _animFrameId = null;
    }
    if (_compassTimer) {
      clearInterval(_compassTimer);
      _compassTimer = null;
    }
    if (_renderer) {
      _renderer.setAnimationLoop(null);
    }
    const overlay = document.getElementById('ar-compass-overlay');
    if (overlay) overlay.style.display = 'none';
    _mode = 'none';
  }

  function setArPath(arPathData) {
    _arPath = arPathData;
    if (_active) _rebuildScene();
  }

  function setPois(poiList) {
    _pois = Array.isArray(poiList) ? poiList : [];
    if (_active) _rebuildScene();
  }

  function setUserPose(lat, lon, headingDeg, floor = 1) {
    const changed = lat !== _userLat || lon !== _userLon;
    _userLat = lat;
    _userLon = lon;
    _userHeading = headingDeg || 0;
    _userFloor = floor;
    if (_active && changed) _rebuildScene();
  }

  function setNextInstruction(text, distanceM) {
    _nextInstruction = text || '';
    _nextDistM = distanceM || 0;
    // Update the DOM instruction banner if present
    const banner = document.getElementById('ar-instruction');
    if (banner) {
      banner.textContent = _nextInstruction
        + (_nextDistM > 0 ? ' · ' + (_nextDistM >= 1000
            ? `${(_nextDistM/1000).toFixed(1)} km`
            : `${Math.round(_nextDistM)} m`) : '');
    }
  }

  function isActive() { return _active; }
  function getMode()  { return _mode; }

  return { init, start, stop, setArPath, setPois, setUserPose, setNextInstruction, isActive, getMode };

})();

// Make globally available
window.ARRenderer = ARRenderer;
