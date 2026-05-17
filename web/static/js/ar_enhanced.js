'use strict';

const AREnhanced = (() => {

  // Shared state
  let _userLat = null, _userLon = null, _userHeading = 0, _userFloor = 1;
  let _lastGpsHeading = 0, _prevLat = null, _prevLon = null;
  let _userPitch = 0, _userRoll = 0;
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
    lastResize: 0,
    lastRelocalizeTs: 0,
    lastMotionTs: 0,
    lastHeading: null,
    arrivalBannerUntil: 0,
    arrivalAnnounced: false,
    headingLp: null,
    _boundResize: null,
    cameraSpinner: false,
  };

  function _lerpHeadingDeg(current, target, t) {
    const d = ((target - current + 540) % 360) - 180;
    return (current + d * t + 360) % 360;
  }

  function _ptOnOrientation(e) {
    let raw = 0;
    if (e.webkitCompassHeading !== undefined && e.webkitCompassHeading !== null) {
      raw = e.webkitCompassHeading;
    } else if (e.alpha !== null && e.alpha !== undefined) {
      raw = (360 - e.alpha) % 360;
    }
    if (PT.headingLp === null || !Number.isFinite(PT.headingLp)) PT.headingLp = raw;
    else PT.headingLp = _lerpHeadingDeg(PT.headingLp, raw, 0.2);
    _userHeading = PT.headingLp;
    _userPitch = Number.isFinite(e.beta) ? e.beta : _userPitch;
    _userRoll = Number.isFinite(e.gamma) ? e.gamma : _userRoll;
    if (PT.lastHeading === null || Math.abs(_angleDelta(_userHeading, PT.lastHeading)) > 3) {
      PT.lastMotionTs = Date.now();
      PT.lastHeading = _userHeading;
    }
  }

  function _angleDelta(a, b) {
    return ((a - b + 540) % 360) - 180;
  }

  async function _requestOrientationPermission() {
    const motionReq = typeof DeviceMotionEvent !== 'undefined' &&
      typeof DeviceMotionEvent.requestPermission === 'function';
    const orientReq = typeof DeviceOrientationEvent !== 'undefined' &&
      typeof DeviceOrientationEvent.requestPermission === 'function';
    try {
      if (motionReq) await DeviceMotionEvent.requestPermission();
      if (orientReq) await DeviceOrientationEvent.requestPermission();
    } catch (e) {
      console.warn('[AREnhanced] Orientation permission denied:', e.message);
    }
  }

  function _enuCameraProj(east_m, north_m, up_m, W, H) {
    const effectiveHeading = (_userHeading !== 0) ? _userHeading
      : (_lastGpsHeading !== undefined ? _lastGpsHeading : 0);
    const headRad = effectiveHeading * Math.PI / 180;
    const camX = east_m * Math.cos(headRad) + north_m * Math.sin(headRad);
    const camZ = -east_m * Math.sin(headRad) + north_m * Math.cos(headRad);
    const pitchRad = Math.max(-60, Math.min(60, _userPitch || 0)) * Math.PI / 180;
    const camY = ((up_m || 0) - 1.6) - Math.tan(pitchRad) * Math.max(0.3, camZ) * 0.35;
    const worldBrg = Math.atan2(east_m, north_m) * 180 / Math.PI;
    const relDeg = ((worldBrg - effectiveHeading + 540) % 360) - 180;
    let x = W / 2, y = H / 2, onScreen = false;
    if (camZ > 0.5) {
      const f = (W / 2) / Math.tan(PT.fovH * Math.PI / 360);
      x = W / 2 + f * camX / camZ;
      y = H / 2 - f * camY / camZ;
      const sab = _safeAreaBottomPx();
      const topPad = 40;
      const botPad = 76 + sab;
      const side = 10;
      onScreen = x >= side && x <= W - side && y >= topPad && y <= H - botPad;
    }
    return { onScreen, x, y, depth: camZ, relDeg, camX, camZ, camY };
  }

  function _enuToScreen(east_m, north_m, up_m, W, H) {
    const p = _enuCameraProj(east_m, north_m, up_m, W, H);
    if (!p.onScreen || p.depth <= 0.5) return null;
    return { x: p.x, y: p.y, depth: p.depth };
  }

  /** Chiếu điểm trước mặt camera (không cần nằm trong khung chặt) — dùng vẽ ribbon + chuỗi chevron */
  function _enuToScreenLoose(east_m, north_m, up_m, W, H) {
    const p = _enuCameraProj(east_m, north_m, up_m, W, H);
    if (p.depth <= 0.5) return null;
    return { x: p.x, y: p.y, depth: p.depth };
  }

  function _pathLengthFromIndexM(pts, i0) {
    if (!pts || i0 >= pts.length - 1) return 0;
    let s = 0;
    for (let i = i0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      s += Math.hypot(b.east_m - a.east_m, b.north_m - a.north_m);
    }
    return s;
  }

  function _fullRouteLengthM(pts) {
    if (!pts || pts.length < 2) return 1;
    return Math.max(1, _pathLengthFromIndexM(pts, 0));
  }

  function _safeAreaBottomPx() {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue('--sab').trim();
      const n = parseFloat(v);
      return Number.isFinite(n) ? n : 0;
    } catch (e) { return 0; }
  }

  /** Giao điểm tia (cx,cy)+t*u, t>0, |u|≈1 với hình chữ nhật lề margin */
  function _rayExitOnScreen(cx, cy, ux, uy, W, H, margin) {
    const xmin = margin, xmax = W - margin, ymin = margin, ymax = H - margin;
    let tMin = Infinity;
    if (ux > 1e-6) tMin = Math.min(tMin, (xmax - cx) / ux);
    if (ux < -1e-6) tMin = Math.min(tMin, (xmin - cx) / ux);
    if (uy > 1e-6) tMin = Math.min(tMin, (ymax - cy) / uy);
    if (uy < -1e-6) tMin = Math.min(tMin, (ymin - cy) / uy);
    if (!Number.isFinite(tMin) || tMin <= 0) tMin = Math.min(W, H) * 0.35;
    return { x: cx + ux * tMin, y: cy + uy * tMin };
  }

  /** Mũi tên ở rìa khi waypoint ngoài khung */
  function _drawEdgeNavIndicator(ctx, relDeg, W, H, distM) {
    const cx = W / 2;
    const cy = Math.min(H - 96, H * 0.62);
    const screenAngle = (relDeg - 90) * Math.PI / 180;
    const ux = Math.cos(screenAngle);
    const uy = Math.sin(screenAngle);
    const margin = 32;
    const exy = _rayExitOnScreen(cx, cy, ux, uy, W, H, margin);
    const ex = exy.x, ey = exy.y;
    _drawNavChevron(ctx, ex, ey, screenAngle, 'edge');
    if (distM != null && distM >= 0 && distM < 5000) {
      const lbl = distM < 1000 ? Math.round(distM) + ' m' : (distM / 1000).toFixed(1) + ' km';
      ctx.font = '600 12px -apple-system,sans-serif';
      const lw = ctx.measureText(lbl).width + 16;
      ctx.fillStyle = 'rgba(15,23,42,0.9)';
      _rrFill(ctx, ex - lw / 2, ey - 42, lw, 24, 8);
      ctx.fillStyle = '#ecfdf5';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(lbl, ex, ey - 30);
    }
  }

  function _drawNavChevron(ctx, x, y, angleRad, kind) {
    // Enhanced sizing: larger arrows for better visibility
    const scale = kind === 'main' ? 2.2 : kind === 'edge' ? 1.8 : kind === 'stair' ? 2.5 : 1.2;
    const L = 24 * scale, Ww = 15 * scale;
    
    // Pulsing animation for main and stair arrows
    const pulse = (kind === 'main' || kind === 'stair') 
      ? 1 + 0.12 * Math.sin(Date.now() * 0.006) 
      : 1;
    
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angleRad);
    ctx.scale(pulse, pulse);
    
    ctx.beginPath();
    ctx.moveTo(L, 0);
    ctx.lineTo(-L * 0.38, -Ww);
    ctx.lineTo(-L * 0.08, 0);
    ctx.lineTo(-L * 0.38, Ww);
    ctx.closePath();
    
    // Different colors for different arrow types
    let g;
    if (kind === 'stair') {
      // Orange gradient for stairs
      g = ctx.createLinearGradient(-L * 0.4, 0, L, 0);
      g.addColorStop(0, '#fed7aa');
      g.addColorStop(0.45, '#fb923c');
      g.addColorStop(1, '#ea580c');
    } else {
      // Green gradient for normal navigation
      g = ctx.createLinearGradient(-L * 0.4, 0, L, 0);
      g.addColorStop(0, '#99f6e4');
      g.addColorStop(0.45, '#34d399');
      g.addColorStop(1, '#16a34a');
    }
    
    ctx.fillStyle = g;
    ctx.shadowColor = kind === 'stair' 
      ? 'rgba(251,146,60,0.8)' 
      : (kind === 'trail' ? 'rgba(52,211,153,0.5)' : 'rgba(16,185,129,0.7)');
    ctx.shadowBlur = kind === 'trail' ? 10 : (kind === 'stair' ? 28 : 22);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(255,255,255,0.95)';
    ctx.lineWidth = kind === 'trail' ? 2.0 : (kind === 'stair' ? 3.0 : 2.8);
    ctx.stroke();
    ctx.restore();
  }

  function _drawRouteRibbon(ctx, screenPts) {
    if (!screenPts || screenPts.length < 2) return;
    ctx.save();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(screenPts[0].x, screenPts[0].y);
    for (let i = 1; i < screenPts.length; i++) ctx.lineTo(screenPts[i].x, screenPts[i].y);
    ctx.strokeStyle = 'rgba(15,23,42,0.52)';
    ctx.lineWidth = 16;
    ctx.stroke();
    ctx.strokeStyle = 'rgba(67,56,202,0.38)';
    ctx.lineWidth = 11;
    ctx.stroke();
    const ax = screenPts[0].x, ay = screenPts[0].y;
    const bx = screenPts[screenPts.length - 1].x, by = screenPts[screenPts.length - 1].y;
    const lg = ctx.createLinearGradient(ax, ay, bx, by);
    lg.addColorStop(0, 'rgba(129,140,248,0.88)');
    lg.addColorStop(0.55, 'rgba(56,189,248,0.78)');
    lg.addColorStop(1, 'rgba(52,211,153,0.92)');
    ctx.strokeStyle = lg;
    ctx.lineWidth = 6;
    ctx.setLineDash([12, 10]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  function _drawChevronsAlongScreenPath(ctx, screenPts, W, H) {
    if (!screenPts || screenPts.length < 2) return;
    
    // Increased spacing for better visibility, reduced density
    const spacing = 75;
    
    for (let i = 0; i < screenPts.length - 1; i++) {
      const a = screenPts[i], b = screenPts[i + 1];
      const dx = b.x - a.x, dy = b.y - a.y;
      const segLen = Math.hypot(dx, dy);
      if (segLen < 20) continue;
      
      const ux = dx / segLen, uy = dy / segLen;
      const ang = Math.atan2(uy, ux);
      
      // Start closer to beginning for better visibility
      let t = spacing * 0.3;
      while (t < segLen - 15) {
        const x = a.x + ux * t, y = a.y + uy * t;
        
        // More generous screen bounds - don't hide arrows too early
        if (x > -150 && x < W + 150 && y > -150 && y < H + 150) {
          // Check if this is near a stair waypoint
          const isNearStair = screenPts.some(pt => {
            const dist = Math.hypot(pt.x - x, pt.y - y);
            return dist < 80 && pt.pt && (pt.pt.maneuver === 'stairs' || pt.pt.maneuver === 'elevator');
          });
          
          _drawNavChevron(ctx, x, y, ang, isNearStair ? 'stair' : 'trail');
        }
        t += spacing;
      }
    }
  }

  function _drawTopVignette(ctx, W) {
    const vg = ctx.createLinearGradient(0, 0, 0, 130);
    vg.addColorStop(0, 'rgba(15,23,42,0.42)');
    vg.addColorStop(1, 'rgba(15,23,42,0)');
    ctx.fillStyle = vg;
    ctx.fillRect(0, 0, W, 130);
  }

  function _drawDirectionArrow(ctx, tx, ty, W, H) {
    const cx = W / 2;
    const cy = Math.min(H - 96, H * 0.62);
    const angle = Math.atan2(ty - cy, tx - cx);
    _drawNavChevron(ctx, cx, cy, angle, 'main');
  }

  /**
   * Draw 3D stair arrow with floor label
   * Enhanced for Phase 4: larger, animated, with text
   */
  function _draw3DStairArrow(ctx, x, y, direction, targetFloor, distance, W, H) {
    // Bounce animation
    const bounce = Math.sin(Date.now() * 0.004) * 8;
    const arrowY = y + bounce;
    
    // Direction angle: up = -90°, down = 90°
    const angleRad = direction === 'up' ? -Math.PI / 2 : Math.PI / 2;
    
    // Draw large stair arrow
    _drawNavChevron(ctx, x, arrowY, angleRad, 'stair');
    
    // Draw floor label above/below arrow
    const labelY = direction === 'up' ? arrowY - 65 : arrowY + 65;
    
    // Label background
    ctx.font = 'bold 16px -apple-system,sans-serif';
    const labelText = `Tầng ${targetFloor}`;
    const labelWidth = ctx.measureText(labelText).width + 24;
    const labelHeight = 32;
    
    ctx.fillStyle = 'rgba(15,23,42,0.92)';
    _rrFill(ctx, x - labelWidth / 2, labelY - labelHeight / 2, labelWidth, labelHeight, 10);
    
    // Label border (orange for stairs)
    ctx.strokeStyle = 'rgba(251,146,60,0.9)';
    ctx.lineWidth = 2;
    _rrStroke(ctx, x - labelWidth / 2, labelY - labelHeight / 2, labelWidth, labelHeight, 10);
    
    // Label text
    ctx.fillStyle = '#fed7aa';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, x, labelY);
    
    // Distance label if close
    if (distance < 50) {
      const distText = Math.round(distance) + 'm';
      ctx.font = '600 12px -apple-system,sans-serif';
      const distWidth = ctx.measureText(distText).width + 16;
      
      ctx.fillStyle = 'rgba(15,23,42,0.85)';
      _rrFill(ctx, x - distWidth / 2, arrowY - 45, distWidth, 22, 6);
      
      ctx.fillStyle = '#fbbf24';
      ctx.fillText(distText, x, arrowY - 34);
    }
    
    // Direction icon
    const icon = direction === 'up' ? '⬆️' : '⬇️';
    ctx.font = '28px sans-serif';
    ctx.fillText(icon, x, arrowY);
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
    PT._routeHud = null;
    const ctx = PT.ctx;
    const W = PT.canvas.width, H = PT.canvas.height;
    if (W !== window.innerWidth || H !== window.innerHeight) {
      PT.canvas.width = window.innerWidth;
      PT.canvas.height = window.innerHeight;
    }

    if (PT.video && PT.video.readyState >= 2) {
      ctx.drawImage(PT.video, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, W, H);
      if (PT.cameraSpinner) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '600 17px -apple-system,sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Đang mở camera…', W / 2, H / 2 - 8);
        ctx.font = '13px sans-serif';
        ctx.fillStyle = '#64748b';
        ctx.fillText('Cho phép quyền camera khi được hỏi', W / 2, H / 2 + 18);
      }
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

      const sliceEnd = Math.min(startIdx + 40, _arPath.points.length);
      const slice = _arPath.points.slice(startIdx, sliceEnd);
      const proj = [];
      const projLoose = [];
      for (const pt of slice) {
        const de = pt.east_m - userE, dn = pt.north_m - userN, uu = pt.up_m;
        const s = _enuToScreen(de, dn, uu, W, H);
        if (s) proj.push({ s, pt });
        const sl = _enuToScreenLoose(de, dn, uu, W, H);
        if (sl) projLoose.push({ x: sl.x, y: sl.y, depth: sl.depth, pt });
      }

      _drawTopVignette(ctx, W);
      const ribbonPts = projLoose.map(o => ({ x: o.x, y: o.y }));
      _drawRouteRibbon(ctx, ribbonPts);
      _drawChevronsAlongScreenPath(ctx, ribbonPts, W, H);

      const nextPt = _arPath.points[startIdx];
      const dNext = Math.hypot(nextPt.east_m - userE, nextPt.north_m - userN);
      const full = _enuCameraProj(nextPt.east_m - userE, nextPt.north_m - userN, nextPt.up_m, W, H);
      const onScreenNext = full.onScreen && full.depth > 0.5;

      // Draw 3D stair arrows for upcoming floor transitions
      for (let i = startIdx; i < Math.min(startIdx + 8, _arPath.points.length); i++) {
        const pt = _arPath.points[i];
        const dist = Math.hypot(pt.east_m - userE, pt.north_m - userN);
        
        // Only show stair arrows for nearby transitions (< 30m)
        if (dist > 30) continue;
        
        // Check if this is a floor transition
        if (pt.maneuver === 'stairs' || pt.maneuver === 'elevator') {
          const stairProj = _enuCameraProj(pt.east_m - userE, pt.north_m - userN, pt.up_m, W, H);
          
          // Draw stair arrow if in view (more generous bounds)
          if (stairProj.depth > 0.5 && stairProj.x > -100 && stairProj.x < W + 100 && 
              stairProj.y > -100 && stairProj.y < H + 100) {
            
            // Determine direction (up/down) from floor change
            const currentFloor = pt.floor || _userFloor;
            const targetFloor = pt.target_floor || currentFloor;
            const direction = targetFloor > currentFloor ? 'up' : 'down';
            
            _draw3DStairArrow(ctx, stairProj.x, stairProj.y, direction, targetFloor, dist, W, H);
          }
        }
      }

      for (let i = 0; i < proj.length; i++) {
        const { s, pt } = proj[i];
        if (pt === nextPt) continue;
        
        // Skip drawing regular waypoint if it's a stair (already drawn as 3D arrow)
        if (pt.maneuver === 'stairs' || pt.maneuver === 'elevator') continue;
        
        const dist = Math.hypot(pt.east_m - userE, pt.north_m - userN);
        ctx.beginPath();
        ctx.arc(s.x, s.y, 6, 0, Math.PI * 2);
        const g2 = ctx.createRadialGradient(s.x - 2, s.y - 2, 0, s.x, s.y, 10);
        g2.addColorStop(0, '#e9d5ff');
        g2.addColorStop(1, 'rgba(99,102,241,0.88)');
        ctx.fillStyle = g2;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.92)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        if (dist < 200) {
          const lbl = dist < 1000 ? Math.round(dist) + 'm' : (dist / 1000).toFixed(1) + 'km';
          ctx.font = '600 10px -apple-system,sans-serif';
          const lw = ctx.measureText(lbl).width + 10;
          ctx.fillStyle = 'rgba(15,23,42,0.85)';
          _rrFill(ctx, s.x - lw / 2, s.y - 26, lw, 18, 5);
          ctx.fillStyle = '#f1f5f9';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(lbl, s.x, s.y - 17);
        }
      }

      if (onScreenNext) {
        const pulse = 1 + 0.14 * Math.sin(Date.now() * 0.0055);
        const r0 = 18 * pulse;
        ctx.beginPath();
        ctx.arc(full.x, full.y, r0 + 10, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(52,211,153,0.4)';
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(full.x, full.y, r0, 0, Math.PI * 2);
        const g3 = ctx.createRadialGradient(full.x - 4, full.y - 4, 2, full.x, full.y, r0);
        g3.addColorStop(0, '#bbf7d0');
        g3.addColorStop(0.5, '#22c55e');
        g3.addColorStop(1, '#166534');
        ctx.fillStyle = g3;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
        ctx.lineWidth = 2.5;
        ctx.stroke();
        _drawDirectionArrow(ctx, full.x, full.y, W, H);
        if (dNext < 220) {
          const lbl = dNext < 1000 ? Math.round(dNext) + ' m' : (dNext / 1000).toFixed(1) + ' km';
          ctx.font = '600 12px -apple-system,sans-serif';
          const lw = ctx.measureText(lbl).width + 16;
          ctx.fillStyle = 'rgba(15,23,42,0.92)';
          _rrFill(ctx, full.x - lw / 2, full.y - 38, lw, 24, 8);
          ctx.fillStyle = '#ecfdf5';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(lbl, full.x, full.y - 26);
        }
      } else {
        _drawEdgeNavIndicator(ctx, full.relDeg, W, H, dNext);
      }

      const lastPt = _arPath.points[_arPath.points.length - 1];
      const dEnd = Math.hypot(lastPt.east_m - userE, lastPt.north_m - userN);
      PT._routeHud = {
        dEnd,
        totalM: _fullRouteLengthM(_arPath.points),
        remainPolyM: _pathLengthFromIndexM(_arPath.points, startIdx),
      };
      if (dEnd < 12) {
        if (!PT.arrivalAnnounced) {
          PT.arrivalAnnounced = true;
          PT.arrivalBannerUntil = Date.now() + 12000;
          if (typeof toast === 'function') toast('Đã đến nơi', 'ok');
          if (typeof SpeechModule !== 'undefined') SpeechModule.speak('Bạn đã đến nơi', 'normal');
        }
      } else if (dEnd > 22) {
        PT.arrivalAnnounced = false;
      }
    }

    if (_steps.length > 0) {
      const step = _steps[0];
      const roadName = step.street_name || step.road_name || '';
      if (roadName) {
        ctx.fillStyle = 'rgba(15,23,42,0.72)';
        _rrFill(ctx, 18, 86, Math.min(W - 36, 260), 34, 8);
        ctx.fillStyle = '#93c5fd';
        ctx.font = 'bold 13px -apple-system,sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText('Đường: ' + String(roadName).slice(0, 28), 30, 103);
      }
      const hud = PT._routeHud;
      if (hud && hud.totalM > 1) {
        const frac = Math.max(0, Math.min(1, 1 - hud.dEnd / hud.totalM));
        const pgW = Math.min(W - 48, 340);
        const pgX = 24;
        const pgY = roadName ? 124 : 88;
        ctx.font = '600 11px -apple-system,sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        const distLab = hud.dEnd < 1000 ? Math.round(hud.dEnd) + ' m' : (hud.dEnd / 1000).toFixed(1) + ' km';
        ctx.fillText('Tới đích: ' + distLab + '  ·  ' + Math.round(frac * 100) + '% tuyến', pgX, pgY);
        ctx.fillStyle = 'rgba(15,23,42,0.55)';
        _rrFill(ctx, pgX, pgY + 4, pgW, 10, 5);
        const wFill = Math.max(6, (pgW - 4) * frac);
        const pg = ctx.createLinearGradient(pgX, 0, pgX + pgW, 0);
        pg.addColorStop(0, '#818cf8');
        pg.addColorStop(1, '#34d399');
        ctx.fillStyle = pg;
        _rrFill(ctx, pgX + 2, pgY + 6, wFill, 6, 3);
      }
      if (_steps.length > 1) {
        const s2 = _steps[1];
        const t2 = (s2.instruction || 'Bước tiếp').slice(0, 40);
        const ty2 = (hud && hud.totalM > 1) ? (roadName ? 152 : 116) : (roadName ? 126 : 88);
        ctx.fillStyle = 'rgba(15,23,42,0.72)';
        _rrFill(ctx, 16, ty2, Math.min(W - 32, 320), 30, 8);
        ctx.fillStyle = '#c7d2fe';
        ctx.font = '600 11px -apple-system,sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText('⏭ Tiếp: ' + t2, 26, ty2 + 15);
      }
      const bh = 54, bw = Math.min(W - 32, 400);
      const sab = _safeAreaBottomPx();
      const bx = (W - bw) / 2, by = H - bh - 24 - sab;
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
      ctx.fillText(instr.length > 46 ? instr.slice(0,44) + '\u2026' : instr, W/2, by + 20);
      if (step.distance_m > 0) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px sans-serif';
        const d = step.distance_m;
        ctx.fillText(d < 1000 ? Math.round(d) + ' m' : (d/1000).toFixed(1) + ' km', W/2, by + 38);
      }
    }

    if (PT.arrivalBannerUntil > Date.now()) {
      const sab2 = _safeAreaBottomPx();
      const bhBar = 54;
      const bwA = Math.min(W - 24, 360);
      const bxA = (W - bwA) / 2;
      const byA = H - bhBar - 24 - sab2 - 58;
      ctx.fillStyle = 'rgba(22,163,74,0.92)';
      _rrFill(ctx, bxA, byA, bwA, 46, 12);
      ctx.strokeStyle = 'rgba(255,255,255,0.45)';
      ctx.lineWidth = 1.5;
      _rrStroke(ctx, bxA, byA, bwA, 46, 12);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 17px -apple-system,sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Đã đến nơi', bxA + bwA / 2, byA + 23);
    }

    _maybeRelocalizeFromStillFrame();

    PT.animId = requestAnimationFrame(_drawPassthrough);
  }

  function _videoFrameBlob() {
    return new Promise(resolve => {
      if (!PT.video || PT.video.readyState < 2) return resolve(null);
      const w = PT.video.videoWidth || 640;
      const h = PT.video.videoHeight || 480;
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      c.getContext('2d').drawImage(PT.video, 0, 0, w, h);
      c.toBlob(resolve, 'image/jpeg', 0.78);
    });
  }

  async function _maybeRelocalizeFromStillFrame() {
    if (!PT.running || !PT.video) return;
    const now = Date.now();
    if (now - PT.lastRelocalizeTs < 9000) return;
    if (now - PT.lastMotionTs < 5000) return;
    PT.lastRelocalizeTs = now;

    const blob = await _videoFrameBlob();
    if (!blob) return;
    const fd = new FormData();
    fd.append('file', new File([blob], 'ar_relocalize.jpg', { type: 'image/jpeg' }));
    if (_userLat !== null) { fd.append('lat', _userLat); fd.append('lon', _userLon); }
    try {
      const r = await fetchWithTimeout('/api/vpr/query', { method: 'POST', body: fd }, 12000);
      const d = await r.json();
      const best = d.matches && d.matches[0];
      if (!d.ok || !d.vpr_ready || !best || best.score < 0.62) return;

      updateLabels([], [], {
        location_name: best.location_name,
        score: best.score,
        summary: best.description || best.caption || '',
      }, PT.video.videoWidth, PT.video.videoHeight);

      if (typeof toast === 'function') {
        toast('VPS: Ban dang dung truoc ' + best.location_name + '?', 'ok');
      }

      await fetchWithTimeout('/api/realtime/vio/relocalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sid,
          lat: best.lat,
          lon: best.lon,
          heading_deg: _userHeading,
          accuracy_m: Math.max(2, best.distance_m ? best.distance_m * 0.1 : 5),
          source: 'vpr',
        }),
      }, 5000).catch(() => {});

      if (best.floor && best.floor >= 1 && typeof floorState !== 'undefined') {
        floorState.floor = best.floor;
        floorState.confidence = Math.max(floorState.confidence || 0, 0.85);
        floorState.method = 'vps';
        if (typeof _updateFloorHUD === 'function') _updateFloorHUD();
      }
    } catch (e) {
      console.warn('[AREnhanced] Relocalize skipped:', e.message);
    }
  }

  async function startPassthrough() {
    if (PT.running) return;
    await _requestOrientationPermission();

    PT.canvas = document.getElementById('ar-passthrough-canvas');
    if (!PT.canvas) {
      PT.canvas = document.createElement('canvas');
      PT.canvas.id = 'ar-passthrough-canvas';
      PT.canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;z-index:795;';
      document.body.appendChild(PT.canvas);
    }
    PT.canvas.width = window.innerWidth;
    PT.canvas.height = window.innerHeight;
    PT.ctx = PT.canvas.getContext('2d');
    PT.canvas.style.display = 'block';
    PT.cameraSpinner = true;
    const splash = () => {
      const c = PT.ctx, ww = PT.canvas.width, hh = PT.canvas.height;
      c.fillStyle = '#0f172a';
      c.fillRect(0, 0, ww, hh);
      c.fillStyle = '#94a3b8';
      c.font = '600 17px -apple-system,sans-serif';
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText('Đang mở camera…', ww / 2, hh / 2 - 8);
      c.font = '13px sans-serif';
      c.fillStyle = '#64748b';
      c.fillText('Cho phép quyền camera khi được hỏi', ww / 2, hh / 2 + 18);
    };
    splash();

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
      if (typeof toast === 'function') {
        toast('Không thể bật camera: ' + (e && e.message ? e.message : 'lỗi không xác định'), 'warn');
      }
    }
    PT.cameraSpinner = false;

    window.addEventListener('deviceorientation', _ptOnOrientation, { passive: true });
    window.addEventListener('deviceorientationabsolute', _ptOnOrientation, { passive: true });
    PT.lastMotionTs = Date.now();
    PT.lastRelocalizeTs = 0;
    PT.running = true;

    PT._boundResize = () => {
      if (PT.canvas) {
        PT.canvas.width = window.innerWidth;
        PT.canvas.height = window.innerHeight;
      }
    };
    PT._boundOrient = () => setTimeout(PT._boundResize, 300);
    window.addEventListener('resize', PT._boundResize);
    window.addEventListener('orientationchange', PT._boundOrient);

    _drawPassthrough();

    const closeBtn = document.getElementById('ar-close-btn');
    if (closeBtn) closeBtn.classList.add('visible');
    window.dispatchEvent(new CustomEvent('ar-mode-change', { detail: { mode: 'passthrough' } }));
  }

  function stopPassthrough() {
    PT.running = false;
    PT.cameraSpinner = false;
    if (PT.animId) { cancelAnimationFrame(PT.animId); PT.animId = null; }
    if (PT.stream) { PT.stream.getTracks().forEach(t => t.stop()); PT.stream = null; }
    PT.video = null;
    if (PT.canvas) PT.canvas.style.display = 'none';
    window.removeEventListener('deviceorientation', _ptOnOrientation);
    window.removeEventListener('deviceorientationabsolute', _ptOnOrientation);
    if (PT._boundResize) {
      window.removeEventListener('resize', PT._boundResize);
      window.removeEventListener('orientationchange', PT._boundOrient);
      PT._boundResize = null;
      PT._boundOrient = null;
    }
    PT.headingLp = null;
    PT.arrivalAnnounced = false;
    PT.arrivalBannerUntil = 0;
    _stopLabelCanvas();
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
    person:'\u{1F464}', car:'\u{1F697}', bicycle:'\u{1F6B2}',
    stairs:'\u{1FA9C}', door:'\u{1F6AA}', sign:'\u{1FAA7}',
    tree:'\u{1F333}', vps:'\u{1F4CD}', ocr:'\u{1F4DD}', default:'\u{1F4CD}',
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

  function _stopLabelCanvas() {
    LBL.running = false;
    if (LBL.animId) {
      cancelAnimationFrame(LBL.animId);
      LBL.animId = null;
    }
    if (LBL.canvas) LBL.canvas.style.display = 'none';
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
      if (vpsHint.summary) {
        LBL.labels.push({
          text: String(vpsHint.summary).slice(0, 48),
          x: W/2, y: 118,
          color: LABEL_COLORS.vps, icon: LABEL_ICONS.vps,
          ttl: now + 5000,
        });
      }
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
    _layoutIndoorCanvas();
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

    _drawIndoorRoute(ctx, toScreen);

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
    ctx.fillText('\uD83C\uDFE2 T\u1EA7ng ' + _userFloor, 10, 15);

    IND.animId = requestAnimationFrame(_drawIndoorMap);
  }

  function _drawIndoorRoute(ctx, toScreen) {
    const indoorSteps = (_steps || []).filter(s => (s.floor || _userFloor) === _userFloor && s.lat && s.lon);
    if (indoorSteps.length < 2) return;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    const p0 = toScreen(indoorSteps[0].lat, indoorSteps[0].lon);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < indoorSteps.length; i++) {
      const p = toScreen(indoorSteps[i].lat, indoorSteps[i].lon);
      ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = 'rgba(250,204,21,0.95)';
    ctx.lineWidth = 5;
    ctx.setLineDash([10, 6]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function _layoutIndoorCanvas() {
    if (!IND.canvas) return;
    const glass = PT.running;
    const size = glass ? Math.min(Math.floor(window.innerWidth * 0.86), 420) : 220;
    if (IND.canvas.width !== size || IND.canvas.height !== size) {
      IND.canvas.width = size;
      IND.canvas.height = size;
    }
    IND.scale = glass ? 12 : 9;
    IND.canvas.style.width = size + 'px';
    IND.canvas.style.height = size + 'px';
    IND.canvas.style.right = glass ? '50%' : '14px';
    IND.canvas.style.bottom = glass ? '120px' : '90px';
    IND.canvas.style.transform = glass
      ? 'translateX(50%) perspective(700px) rotateX(18deg)'
      : '';
    IND.canvas.style.background = glass ? 'rgba(15,23,42,0.48)' : 'rgba(15,23,42,0.72)';
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
        'backdrop-filter:blur(8px)',
        'pointer-events:none',
      ].join(';');
      document.body.appendChild(IND.canvas);
    }
    _layoutIndoorCanvas();
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
    _steps = steps || [];
    PT.arrivalAnnounced = false;
    PT.arrivalBannerUntil = 0;
    if (IND.running && IND.userPos) {
      const destStep = [..._steps].reverse().find(s => s.lat && s.lon);
      IND.destPos = destStep ? {
        lat: destStep.lat,
        lon: destStep.lon,
        floor: destStep.floor || _userFloor,
      } : IND.destPos;
    }
  }

  function setUserPose(lat, lon, headingDeg, floor) {
    // Derive heading from GPS movement if compass not available
    if (_prevLat !== null && _prevLon !== null) {
      const dlat = lat - _prevLat, dlon = lon - _prevLon;
      if (Math.hypot(dlat, dlon) > 0.00001) {
        _lastGpsHeading = (Math.atan2(dlon, dlat) * 180 / Math.PI + 360) % 360;
      }
    }
    _prevLat = lat; _prevLon = lon;
    _userLat = lat; _userLon = lon;
    if (!PT.running) {
      _userHeading = headingDeg || 0;
    }
    _userFloor = floor || 1;
    if (IND.running && IND.userPos) {
      IND.userPos.lat = lat; IND.userPos.lon = lon; IND.userPos.floor = _userFloor;
    }
  }

  return { init, startPassthrough, stopPassthrough, updateLabels, updateIndoorMap, setRoute, setUserPose };

})();

window.AREnhanced = AREnhanced;
