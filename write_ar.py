"""Write ar_enhanced.js — run once then delete"""
import pathlib

JS = """\
'use strict';

const AREnhanced = (() => {

  // Shared state
  let _userLat = null, _userLon = null, _userHeading = 0, _userFloor = 1;
  let _arPath = null, _steps = [];

  // ─────────────────────────────────────────────────────────────────────────
  // FEATURE 2: Camera Passthrough AR
  // Real camera video as background + route waypoints projected onto screen
  // using DeviceOrientation. Works on ALL phones — no WebXR needed.
  // ─────────────────────────────────────────────────────────────────────────

  const PT = {
    canvas: null, ctx: null, video: null, stream: null,
    animId: null, running: false,
    fovH: 60,
  };

  function _ptOnOrientation(e) {
    if (e.webkitCompassHeading !== undefined) _userHeading = e.webkitCompassHeading;
    else if (e.alpha !== null) _userHeading = (360 - e.alpha) % 360;
  }

  function _enuToScreen(east_m, north_m, up_m, W, H) {
    const headRad = _userHeading * Math.PI / 180;
    const camX =  east_m  * Math.cos(headRad) + north_m * Math.sin(headRad);
    const camZ = -east_m  * Math.sin(headRad) + north_m * Math.cos(headRad);
    const camY = (up_m || 0) - 1.6;
    if (camZ <= 0.5) return null;
    const f = (W / 2) / Math.tan(PT.fovH * Math.PI / 360);
    const sx = W / 2 + f * camX / camZ;
    const sy = H / 2 - f * camY / camZ;
    if (sx < -60 || sx > W + 60 || sy < -60 || sy > H + 60) return null;
    return { x: sx, y: sy, depth: camZ };
  }

  function _rrFill(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); }
    else {
      ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y);
      ctx.quadraticCurveTo(x+w,y,x+w,y+r);
      ctx.lineTo(x+w,y+h-r); ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
      ctx.lineTo(x+r,y+h); ctx.quadraticCurveTo(x,y+h,x,y+h-r);
      ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y);
      ctx.closePath();
    }
    ctx.fill();
  }
  function _rrStroke(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); }
    else {
      ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y);
      ctx.quadraticCurveTo(x+w,y,x+w,y+r);
      ctx.lineTo(x+w,y+h-r); ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
      ctx.lineTo(x+r,y+h); ctx.quadraticCurveTo(x,y+h,x,y+h-r);
      ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y);
      ctx.closePath();
    }
    ctx.stroke();
  }

  function _drawPassthrough() {
    if (!PT.running) return;
    const ctx = PT.ctx;
    const W = PT.canvas.width, H = PT.canvas.height;

    if (PT.video && PT.video.readyState >= 2) {
      ctx.drawImage(PT.video, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, W, H);
    }

    if (_arPath && _arPath.points && _userLat !== null) {
      const ref = _arPath.reference;
      const latM = 111320;
      const lonM = 111320 * Math.cos(ref.lat * Math.PI / 180);
      const userE = (_userLon - ref.lon) * lonM;
      const userN = (_userLat - ref.lat) * latM;

      let startIdx = 0, minD = Infinity;
      for (let i = 0; i < _arPath.points.length; i++) {
        const p = _arPath.points[i];
        const d = Math.hypot(p.east_m - userE, p.north_m - userN);
        if (d < minD) { minD = d; startIdx = i; }
      }

      const slice = _arPath.points.slice(startIdx, startIdx + 10);
      const proj = [];
      for (const pt of slice) {
        const s = _enuToScreen(pt.east_m - userE, pt.north_m - userN, pt.up_m, W, H);
        if (s) proj.push({ s, pt });
      }

      if (proj.length >= 2) {
        ctx.beginPath();
        ctx.moveTo(proj[0].s.x, proj[0].s.y);
        for (let i = 1; i < proj.length; i++) ctx.lineTo(proj[i].s.x, proj[i].s.y);
        ctx.strokeStyle = 'rgba(99,102,241,0.65)';
        ctx.lineWidth = 4;
        ctx.setLineDash([14, 8]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      for (let i = 0; i < proj.length; i++) {
        const { s, pt } = proj[i];
        const isNext = i === 0;
        const dist = Math.hypot(pt.east_m - userE, pt.north_m - userN);
        ctx.beginPath();
        ctx.arc(s.x, s.y, isNext ? 16 : 9, 0, Math.PI * 2);
        ctx.fillStyle = isNext ? 'rgba(34,197,94,0.92)' : 'rgba(99,102,241,0.78)';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        if (isNext) {
          ctx.fillStyle = '#fff';
          ctx.font = 'bold 15px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('\\u2191', s.x, s.y);
        }
        if (dist < 200) {
          const lbl = dist < 1000 ? Math.round(dist) + 'm' : (dist/1000).toFixed(1) + 'km';
          ctx.font = '11px sans-serif';
          const lw = ctx.measureText(lbl).width + 12;
          ctx.fillStyle = 'rgba(15,23,42,0.82)';
          _rrFill(ctx, s.x - lw/2, s.y - 30, lw, 20, 5);
          ctx.fillStyle = '#e2e8f0';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(lbl, s.x, s.y - 20);
        }
      }
    }

    if (_steps.length > 0) {
      const step = _steps[0];
      const bh = 54, bw = Math.min(W - 32, 400);
      const bx = (W - bw) / 2, by = H - bh - 24;
      ctx.fillStyle = 'rgba(15,23,42,0.9)';
      _rrFill(ctx, bx, by, bw, bh, 14);
      ctx.strokeStyle = 'rgba(99,102,241,0.55)';
      ctx.lineWidth = 1.5;
      _rrStroke(ctx, bx, by, bw, bh, 14);
      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 14px -apple-system,sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const instr = (step.instruction || '');
      ctx.fillText(instr.length > 46 ? instr.slice(0,44) + '\\u2026' : instr, W/2, by + 20);
      if (step.distance_m > 0) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px sans-serif';
        const d = step.distance_m;
        ctx.fillText(d < 1000 ? Math.round(d) + ' m' : (d/1000).toFixed(1) + ' km', W/2, by + 38);
      }
    }

    PT.animId = requestAnimationFrame(_drawPassthrough);
  }

  async function startPassthrough() {
    if (PT.running) return;
    PT.canvas = document.getElementById('ar-passthrough-canvas');
    if (!PT.canvas) {
      PT.canvas = document.createElement('canvas');
      PT.canvas.id = 'ar-passthrough-canvas';
      PT.canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;z-index:795;';
      document.body.appendChild(PT.canvas);
    }
    PT.canvas.width  = window.innerWidth;
    PT.canvas.height = window.innerHeight;
    PT.ctx = PT.canvas.getContext('2d');
    PT.canvas.style.display = 'block';

    try {
      PT.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
        audio: false,
      });
      PT.video = document.createElement('video');
      PT.video.srcObject = PT.stream;
      PT.video.setAttribute('playsinline', '');
      PT.video.muted = true;
      await PT.video.play().catch(() => {});
    } catch (e) {
      console.warn('[AREnhanced] Camera unavailable:', e.message);
    }

    window.addEventListener('deviceorientation', _ptOnOrientation, { passive: true });
    window.addEventListener('deviceorientationabsolute', _ptOnOrientation, { passive: true });
    PT.running = true;
    _drawPassthrough();

    const closeBtn = document.getElementById('ar-close-btn');
    if (closeBtn) closeBtn.classList.add('visible');
    window.dispatchEvent(new CustomEvent('ar-mode-change', { detail: { mode: 'passthrough' } }));
  }

  function stopPassthrough() {
    PT.running = false;
    if (PT.animId) { cancelAnimationFrame(PT.animId); PT.animId = null; }
    if (PT.stream) { PT.stream.getTracks().forEach(t => t.stop()); PT.stream = null; }
    if (PT.canvas) PT.canvas.style.display = 'none';
    window.removeEventListener('deviceorientation', _ptOnOrientation);
    window.removeEventListener('deviceorientationabsolute', _ptOnOrientation);
    window.dispatchEvent(new CustomEvent('ar-mode-change', { detail: { mode: 'none' } }));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // FEATURE 1: AR Semantic Labeling
  // YOLO detections, OCR text, VPS hints as floating labels on camera feed.
  // ─────────────────────────────────────────────────────────────────────────

  const LBL = {
    canvas: null, ctx: null,
    labels: [], animId: null, running: false,
  };

  const LABEL_ICONS = {
    person:'\\u{1F464}', car:'\\u{1F697}', bicycle:'\\u{1F6B2}',
    stairs:'\\u{1FA9C}', door:'\\u{1F6AA}', sign:'\\u{1FAA7}',
    tree:'\\u{1F333}', vps:'\\u{1F4CD}', ocr:'\\u{1F4DD}', default:'\\u{1F4CD}',
  };
  const LABEL_COLORS = {
    person:'#f59e0b', car:'#ef4444', bicycle:'#22c55e',
    stairs:'#f97316', door:'#14b8a6', sign:'#8b5cf6',
    vps:'#6366f1', ocr:'#94a3b8', default:'#e2e8f0',
  };

  function _drawLabels() {
    if (!LBL.running) return;
    const ctx = LBL.ctx;
    const W = LBL.canvas.width, H = LBL.canvas.height;
    ctx.clearRect(0, 0, W, H);
    const now = Date.now();
    LBL.labels = LBL.labels.filter(l => l.ttl > now);

    for (const lbl of LBL.labels) {
      const alpha = Math.min(1, (lbl.ttl - now) / 500);
      ctx.globalAlpha = alpha;
      ctx.font = 'bold 13px -apple-system,sans-serif';
      const tw = ctx.measureText(lbl.text).width;
      const pw = tw + 30, ph = 26;
      const px = Math.max(4, Math.min(W - pw - 4, lbl.x - pw/2));
      const py = Math.max(4, Math.min(H - ph - 4, lbl.y - ph - 10));
      ctx.fillStyle = 'rgba(15,23,42,0.88)';
      _rrFill(ctx, px, py, pw, ph, 8);
      ctx.strokeStyle = lbl.color;
      ctx.lineWidth = 1.5;
      _rrStroke(ctx, px, py, pw, ph, 8);
      ctx.fillStyle = lbl.color;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      ctx.fillText(lbl.icon + ' ' + lbl.text, px + 8, py + ph/2);
      ctx.beginPath();
      ctx.arc(lbl.x, lbl.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = lbl.color;
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(px + pw/2, py + ph);
      ctx.lineTo(lbl.x, lbl.y);
      ctx.strokeStyle = lbl.color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = alpha * 0.4;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    LBL.animId = requestAnimationFrame(_drawLabels);
  }

  function _ensureLabelCanvas() {
    if (LBL.running) return;
    LBL.canvas = document.getElementById('ar-label-canvas');
    if (!LBL.canvas) {
      LBL.canvas = document.createElement('canvas');
      LBL.canvas.id = 'ar-label-canvas';
      LBL.canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;z-index:815;pointer-events:none;';
      document.body.appendChild(LBL.canvas);
    }
    LBL.canvas.width  = window.innerWidth;
    LBL.canvas.height = window.innerHeight;
    LBL.ctx = LBL.canvas.getContext('2d');
    LBL.canvas.style.display = 'block';
    LBL.running = true;
    _drawLabels();
  }

  function updateLabels(detections, ocrBlocks, vpsHint, imageW, imageH) {
    _ensureLabelCanvas();
    const W = LBL.canvas.width, H = LBL.canvas.height;
    const sx = imageW ? W / imageW : 1;
    const sy = imageH ? H / imageH : 1;
    const TTL = 3500, now = Date.now();

    for (const det of (detections || [])) {
      if (det.confidence < 0.35) continue;
      const [x1, y1, x2, y2] = det.bbox;
      LBL.labels.push({
        text: det.label + ' ' + Math.round(det.confidence * 100) + '%',
        x: ((x1+x2)/2)*sx, y: ((y1+y2)/2)*sy,
        color: LABEL_COLORS[det.label] || LABEL_COLORS.default,
        icon: LABEL_ICONS[det.label] || LABEL_ICONS.default,
        ttl: now + TTL,
      });
    }

    for (const blk of (ocrBlocks || [])) {
      if (blk.confidence < 0.5 || !blk.text.trim()) continue;
      const pts = blk.bbox || [];
      if (!pts.length) continue;
      const cx = pts.reduce((s,p) => s+p[0], 0) / pts.length * sx;
      const cy = pts.reduce((s,p) => s+p[1], 0) / pts.length * sy;
      LBL.labels.push({
        text: blk.text.slice(0, 24),
        x: cx, y: cy,
        color: LABEL_COLORS.ocr, icon: LABEL_ICONS.ocr,
        ttl: now + TTL,
      });
    }

    if (vpsHint && vpsHint.score >= 0.6) {
      LBL.labels.push({
        text: vpsHint.location_name + ' (' + Math.round(vpsHint.score*100) + '%)',
        x: W/2, y: 80,
        color: LABEL_COLORS.vps, icon: LABEL_ICONS.vps,
        ttl: now + 5000,
      });
    }

    if (LBL.labels.length > 14) LBL.labels = LBL.labels.slice(-14);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // FEATURE 5: AR Indoor Map Overlay
  // Semi-transparent floor plan in corner when GPS accuracy > 15m (indoors).
  // Rotates with compass. Shows user position + destination.
  // ─────────────────────────────────────────────────────────────────────────

  const IND = {
    canvas: null, ctx: null,
    geojson: null, userPos: null, destPos: null,
    animId: null, running: false,
    scale: 9,
  };

  function _drawIndoorMap() {
    if (!IND.running) return;
    const ctx = IND.ctx;
    const W = IND.canvas.width, H = IND.canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (!IND.geojson || !IND.userPos) {
      IND.animId = requestAnimationFrame(_drawIndoorMap);
      return;
    }

    const cx = W/2, cy = H/2;
    const headRad = _userHeading * Math.PI / 180;
    const latM = 111320;
    const lonM = 111320 * Math.cos(IND.userPos.lat * Math.PI / 180);

    function toScreen(lat, lon) {
      const dy = (lat - IND.userPos.lat) * latM;
      const dx = (lon - IND.userPos.lon) * lonM;
      const rx =  dx * Math.cos(-headRad) - dy * Math.sin(-headRad);
      const ry =  dx * Math.sin(-headRad) + dy * Math.cos(-headRad);
      return { x: cx + rx * IND.scale, y: cy - ry * IND.scale };
    }

    ctx.globalAlpha = 0.75;
    const features = (IND.geojson.features || []);
    for (const feat of features) {
      const geom = feat.geometry || {};
      const props = feat.properties || {};
      if ((props.floor || 1) !== _userFloor) continue;
      const et = props.edge_type || props.node_type || 'corridor';
      const colors = {
        corridor:'rgba(99,102,241,0.55)', stairs:'rgba(245,158,11,0.75)',
        elevator:'rgba(20,184,166,0.75)', room:'rgba(148,163,184,0.35)',
        entrance:'rgba(34,197,94,0.65)', default:'rgba(99,102,241,0.45)',
      };
      ctx.strokeStyle = colors[et] || colors.default;
      ctx.fillStyle   = colors[et] || colors.default;
      ctx.lineWidth = et === 'corridor' ? 3 : 2;

      if (geom.type === 'LineString') {
        const coords = geom.coordinates || [];
        if (coords.length < 2) continue;
        ctx.beginPath();
        const s0 = toScreen(coords[0][1], coords[0][0]);
        ctx.moveTo(s0.x, s0.y);
        for (let i = 1; i < coords.length; i++) {
          const s = toScreen(coords[i][1], coords[i][0]);
          ctx.lineTo(s.x, s.y);
        }
        ctx.stroke();
      } else if (geom.type === 'Point') {
        const s = toScreen(geom.coordinates[1], geom.coordinates[0]);
        ctx.beginPath();
        ctx.arc(s.x, s.y, 4, 0, Math.PI * 2);
        ctx.fill();
        if (props.name) {
          ctx.globalAlpha = 0.85;
          ctx.fillStyle = '#e2e8f0';
          ctx.font = '8px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText(props.name.slice(0, 10), s.x, s.y - 5);
        }
      }
    }

    if (IND.destPos && IND.destPos.floor === _userFloor) {
      const ds = toScreen(IND.destPos.lat, IND.destPos.lon);
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(ds.x, ds.y, 9, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239,68,68,0.9)';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    const pulse = 0.7 + 0.3 * Math.sin(Date.now() * 0.004);
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, 11 * pulse, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(34,197,94,0.22)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#22c55e';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = 'rgba(15,23,42,0.82)';
    _rrFill(ctx, 4, 4, 72, 22, 6);
    ctx.fillStyle = '#14b8a6';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('\\uD83C\\uDFE2 T\\u1EA7ng ' + _userFloor, 10, 15);

    IND.animId = requestAnimationFrame(_drawIndoorMap);
  }

  function _ensureIndoorCanvas() {
    if (IND.running) return;
    IND.canvas = document.getElementById('ar-indoor-canvas');
    if (!IND.canvas) {
      IND.canvas = document.createElement('canvas');
      IND.canvas.id = 'ar-indoor-canvas';
      IND.canvas.style.cssText = [
        'position:fixed', 'bottom:90px', 'right:14px',
        'width:220px', 'height:220px', 'z-index:812',
        'border-radius:12px',
        'border:1.5px solid rgba(99,102,241,0.4)',
        'background:rgba(15,23,42,0.72)',
        'box-shadow:0 4px 20px rgba(0,0,0,0.5)',
        'pointer-events:none',
      ].join(';');
      document.body.appendChild(IND.canvas);
    }
    IND.canvas.width = 220; IND.canvas.height = 220;
    IND.ctx = IND.canvas.getContext('2d');
    IND.canvas.style.display = 'block';
    IND.running = true;
    _drawIndoorMap();
  }

  function updateIndoorMap(geojson, userPos, destPos) {
    IND.geojson = geojson;
    IND.userPos = userPos;
    IND.destPos = destPos || null;
    if (geojson && userPos) { _ensureIndoorCanvas(); }
    else if (IND.running) {
      IND.running = false;
      if (IND.animId) { cancelAnimationFrame(IND.animId); IND.animId = null; }
      if (IND.canvas) IND.canvas.style.display = 'none';
    }
  }

  async function _autoLoadFloorMap() {
    if (IND.running || _userLat === null) return;
    try {
      const r = await fetch('/api/indoor/nodes?lat=' + _userLat + '&lon=' + _userLon + '&radius=0.001');
      const d = await r.json();
      if (!d.nodes || !d.nodes.length) return;
      const bid = d.nodes[0].building_id;
      const r2 = await fetch('/api/indoor/map/' + bid + '/' + _userFloor);
      const d2 = await r2.json();
      if (d2.geojson) {
        updateIndoorMap(d2.geojson, { lat: _userLat, lon: _userLon, floor: _userFloor }, null);
      }
    } catch (e) {}
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Public API
  // ─────────────────────────────────────────────────────────────────────────

  function init() {
    // Auto indoor map when GPS degrades
    setInterval(() => {
      const acc = typeof _gpsAccuracyM !== 'undefined' ? _gpsAccuracyM : 5;
      if (acc > 15) _autoLoadFloorMap();
      else if (acc <= 10 && IND.running) updateIndoorMap(null, null, null);
    }, 5000);

    // Listen for realtime scene updates from websocket.js
    window.addEventListener('realtime-scene-update', e => {
      const vis = (e.detail || {}).visual || {};
      updateLabels(
        vis.landmarks || [], vis.ocr_blocks || [], vis.vpr_hint || null,
        e.detail.image_width, e.detail.image_height,
      );
    });
  }

  function setRoute(arPath, steps) {
    _arPath = arPath;
    _steps  = steps || [];
  }

  function setUserPose(lat, lon, headingDeg, floor) {
    _userLat = lat; _userLon = lon;
    _userHeading = headingDeg || 0;
    _userFloor = floor || 1;
    if (IND.running && IND.userPos) {
      IND.userPos.lat = lat; IND.userPos.lon = lon; IND.userPos.floor = _userFloor;
    }
  }

  return { init, startPassthrough, stopPassthrough, updateLabels, updateIndoorMap, setRoute, setUserPose };

})();

window.AREnhanced = AREnhanced;
"""

pathlib.Path('web/static/js/ar_enhanced.js').write_text(JS, encoding='utf-8')
size = pathlib.Path('web/static/js/ar_enhanced.js').stat().st_size
print(f'ar_enhanced.js written: {size} bytes ({size//1024} KB)')
