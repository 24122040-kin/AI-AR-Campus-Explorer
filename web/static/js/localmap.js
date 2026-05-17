/**
 * LocalMap - Interactive map for HCMUS CS2
 * Simplified workflow for creating roads and viewing locations
 */

// Global state
let map, mode = 'view';
let locMarkers = [], edgeLines = [];
let currentFloor = 'all';
let _coordGroups = new Map();
let _campusBoundaryLayer = null;
let _campusRoadLabelMarkers = [];

// Road creation workflow
let roadPointA = null, roadPointB = null;
let roadMarkerA = null, roadMarkerB = null;
let roadPreviewLine = null;
let roadMethod = null; // 'straight' or 'tracking'

// Tracking state
let walkPoints = [];
let walkPolyline = null;
let walkWatchId = null;
let walkDistance = 0;
let walkLastPos = null;

const SNAP_M = 55;
const API = '';

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function haversineM(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const toR = x => x * Math.PI / 180;
  const dLat = toR(lat2 - lat1), dLon = toR(lon2 - lon1);
  const a = Math.sin(dLat/2)**2 + Math.cos(toR(lat1))*Math.cos(toR(lat2))*Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

function coordKey6(loc) {
  return (Number(loc.lat) || 0).toFixed(6) + ',' + (Number(loc.lon) || 0).toFixed(6);
}

function floorColor(floor) {
  return window.FLOOR_COLORS[(floor - 1) % window.FLOOR_COLORS.length] || '#3b82f6';
}

function setStatus(msg) {
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
}

// ============================================================================
// MODE MANAGEMENT
// ============================================================================

function setMode(m) {
  mode = m;
  
  // Update toolbar buttons
  ['btn-view', 'btn-add-road'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.classList.remove('active');
  });
  
  const activeBtn = document.getElementById('btn-' + m);
  if (activeBtn) activeBtn.classList.add('active');
  
  // Show/hide panels
  const roadPanel = document.getElementById('road-panel');
  if (roadPanel) roadPanel.classList.toggle('show', m === 'add-road');
  
  // Reset states
  if (m !== 'add-road') cancelRoad();
  
  // Show step 1 when entering add-road mode
  if (m === 'add-road') {
    showRoadStep(1);
    setStatus('➕ Chọn 2 điểm để tạo đường');
  } else {
    setStatus('👁 Xem bản đồ - click marker để xem thông tin');
  }
}

function closeMap() {
  if (window.parent && window.parent !== window) {
    window.parent.postMessage({type: 'close-localmap'}, '*');
  }
}

// ============================================================================
// LOCATION POPUP (View mode)
// ============================================================================

async function showLocationPopup(loc, latlng) {
  // Fetch image
  let imgHtml = '';
  try {
    const r = await fetch(API + '/api/location/' + loc.id + '/images');
    const d = await r.json();
    if (d.images && d.images.length > 0) {
      const primaryImg = d.images.find(img => img.is_primary) || d.images[0];
      imgHtml = `<img src="/api/image/${primaryImg.id}" style="width:100%;max-width:240px;border-radius:8px;margin-bottom:10px" alt="${loc.name}"/>`;
    }
  } catch(e) {}

  const popupContent = `
    <div style="min-width:220px;max-width:280px;font-family:-apple-system,sans-serif">
      ${imgHtml}
      <div style="margin-bottom:12px">
        <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:4px">${loc.name}</div>
        <div style="font-size:12px;color:#64748b;margin-bottom:2px">
          📍 Tầng ${loc.floor || 1} · ${loc.category || 'địa điểm'}
        </div>
        ${loc.description ? `<div style="font-size:12px;color:#475569;margin-top:6px;font-style:italic">${loc.description}</div>` : ''}
        <div style="font-size:10px;color:#94a3b8;margin-top:6px">
          ID: #${loc.id} · ${Number(loc.lat).toFixed(6)}, ${Number(loc.lon).toFixed(6)}
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <button class="loc-option-btn" data-action="set-start" data-id="${loc.id}" 
                style="width:100%;padding:8px 12px;background:#10b981;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600">
          🚀 Chọn làm điểm xuất phát
        </button>
        <button class="loc-option-btn" data-action="set-dest" data-id="${loc.id}"
                style="width:100%;padding:8px 12px;background:#3b82f6;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600">
          🎯 Chọn làm điểm đến
        </button>
        <button class="loc-option-btn" data-action="delete" data-id="${loc.id}"
                style="width:100%;padding:8px 12px;background:#ef4444;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600">
          🗑️ Xóa điểm này
        </button>
      </div>
    </div>
  `;

  const popup = L.popup({ maxWidth: 320, closeButton: true })
    .setLatLng(latlng)
    .setContent(popupContent)
    .openOn(map);

  // Add event listeners
  setTimeout(() => {
    const el = popup.getElement();
    if (!el) return;
    
    el.querySelectorAll('.loc-option-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const action = btn.getAttribute('data-action');
        const id = parseInt(btn.getAttribute('data-id'), 10);
        
        if (action === 'set-start') {
          // Set as point A for road creation
          setMode('add-road');
          roadPointA = loc;
          if (roadMarkerA) map.removeLayer(roadMarkerA);
          roadMarkerA = L.circleMarker([loc.lat, loc.lon], {
            radius: 17, color: '#22c55e', weight: 3, opacity: 1, fill: false
          }).bindTooltip('A: ' + loc.name, { permanent: true }).addTo(map);
          
          showRoadStep(1);
          const hint = document.getElementById('road-hint');
          if (hint) hint.textContent = `Điểm A: "${loc.name}" - click điểm B...`;
          setStatus(`✅ Đã chọn điểm A: "${loc.name}"`);
          
        } else if (action === 'set-dest') {
          // Set as point B for road creation
          if (!roadPointA) {
            alert('Chọn điểm xuất phát (A) trước');
            return;
          }
          
          if (loc.id === roadPointA.id) {
            alert('Chọn điểm khác với điểm A');
            return;
          }
          
          roadPointB = loc;
          if (roadMarkerB) map.removeLayer(roadMarkerB);
          roadMarkerB = L.circleMarker([loc.lat, loc.lon], {
            radius: 17, color: '#ef4444', weight: 3, opacity: 1, fill: false
          }).bindTooltip('B: ' + loc.name, { permanent: true }).addTo(map);
          
          if (roadPreviewLine) map.removeLayer(roadPreviewLine);
          roadPreviewLine = L.polyline([
            [roadPointA.lat, roadPointA.lon],
            [roadPointB.lat, roadPointB.lon]
          ], { color: '#f59e0b', dashArray: '6,4', weight: 3 }).addTo(map);
          
          // Move to step 2: choose method
          showRoadStep(2);
          setStatus(`✅ Đã chọn 2 điểm: ${roadPointA.name} → ${roadPointB.name}`);
          
        } else if (action === 'delete') {
          if (!confirm(`Xác nhận xóa địa điểm "${loc.name}"?`)) return;
          try {
            const r = await fetch(API + '/api/location/' + id, { method: 'DELETE' });
            const d = await r.json();
            if (d.ok) {
              const idx = window.LOCATIONS.findIndex(l => l.id === id);
              if (idx >= 0) window.LOCATIONS.splice(idx, 1);
              renderLocations();
              setStatus(`✅ Đã xóa "${loc.name}"`);
            } else {
              setStatus('❌ Lỗi xóa địa điểm');
            }
          } catch(e) {
            setStatus('❌ ' + e.message);
          }
        }
        
        map.closePopup();
      });
    });
  }, 50);
}

// ============================================================================
// ROAD CREATION WORKFLOW
// ============================================================================

function showRoadStep(step) {
  // Hide all steps
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById('road-step-' + i);
    if (el) el.style.display = 'none';
  }
  
  // Show current step
  const current = document.getElementById('road-step-' + step);
  if (current) current.style.display = 'flex';
  
  // Special handling for step 2: show floor info if available
  if (step === 2 && roadPointA && roadPointB) {
    const floorA = roadPointA.floor || 1;
    const floorB = roadPointB.floor || 1;
    
    // Add floor info to step 2
    const step2 = document.getElementById('road-step-2');
    if (step2) {
      let floorInfo = step2.querySelector('.floor-info');
      if (!floorInfo) {
        floorInfo = document.createElement('div');
        floorInfo.className = 'floor-info';
        floorInfo.style.cssText = 'padding:8px;background:#334155;border-radius:6px;font-size:12px;color:#e2e8f0;margin-bottom:8px';
        step2.insertBefore(floorInfo, step2.children[1]);
      }
      
      if (floorA === floorB) {
        floorInfo.innerHTML = `📍 Cùng tầng ${floorA} → Dùng <b>Nối thẳng</b> hoặc <b>Tracking</b>`;
      } else {
        floorInfo.innerHTML = `📍 Tầng ${floorA} → Tầng ${floorB} → Nên dùng <b style="color:#fbbf24">🪜 Cầu thang</b>`;
      }
    }
  }
}

function selectRoadPoint(loc) {
  if (!roadPointA) {
    roadPointA = loc;
    if (roadMarkerA) map.removeLayer(roadMarkerA);
    roadMarkerA = L.circleMarker([loc.lat, loc.lon], {
      radius: 17, color: '#22c55e', weight: 3, opacity: 1, fill: false
    }).bindTooltip('A: ' + loc.name, { permanent: true }).addTo(map);
    
    // Update display
    const displayA = document.getElementById('point-a-display');
    if (displayA) displayA.textContent = `✅ ${loc.name} (Tầng ${loc.floor || 1})`;
    
    const hint = document.getElementById('road-hint');
    if (hint) hint.textContent = `Đã chọn điểm A - bây giờ click marker để chọn điểm B...`;
    
    setStatus(`✅ Đã chọn điểm A: "${loc.name}"`);
    
  } else if (!roadPointB) {
    if (loc.id === roadPointA.id) {
      setStatus('Chọn điểm khác với điểm A');
      return;
    }
    
    roadPointB = loc;
    if (roadMarkerB) map.removeLayer(roadMarkerB);
    roadMarkerB = L.circleMarker([loc.lat, loc.lon], {
      radius: 17, color: '#ef4444', weight: 3, opacity: 1, fill: false
    }).bindTooltip('B: ' + loc.name, { permanent: true }).addTo(map);
    
    if (roadPreviewLine) map.removeLayer(roadPreviewLine);
    roadPreviewLine = L.polyline([
      [roadPointA.lat, roadPointA.lon],
      [roadPointB.lat, roadPointB.lon]
    ], { color: '#f59e0b', dashArray: '6,4', weight: 3 }).addTo(map);
    
    // Update display
    const displayB = document.getElementById('point-b-display');
    if (displayB) displayB.textContent = `✅ ${loc.name} (Tầng ${loc.floor || 1})`;
    
    // Move to step 2: choose method
    showRoadStep(2);
    setStatus(`✅ Đã chọn 2 điểm: ${roadPointA.name} → ${roadPointB.name}`);
  }
}

function selectRoadMethod(method) {
  roadMethod = method;
  
  if (method === 'straight') {
    // Skip to form
    showRoadStep(4);
    setStatus('Điền thông tin đường');
    
  } else if (method === 'stairs') {
    // Check if 2 points are on different floors
    if (roadPointA && roadPointB) {
      const floorA = roadPointA.floor || 1;
      const floorB = roadPointB.floor || 1;
      
      if (floorA === floorB) {
        alert('⚠️ Cầu thang phải nối 2 điểm ở tầng khác nhau!\n\nĐiểm A và B đang cùng tầng ' + floorA + '.\nHãy chọn lại 2 điểm ở 2 tầng khác nhau.');
        return;
      }
    }
    
    // Auto-fill form for stairs
    showRoadStep(4);
    
    const roadTypeSelect = document.getElementById('road-type');
    if (roadTypeSelect) roadTypeSelect.value = 'stairs';
    
    const roadNameInput = document.getElementById('road-name');
    if (roadNameInput && roadPointA && roadPointB) {
      const floorA = roadPointA.floor || 1;
      const floorB = roadPointB.floor || 1;
      roadNameInput.value = `Cầu thang ${Math.min(floorA, floorB)}-${Math.max(floorA, floorB)}`;
    }
    
    // Stairs typically have slope
    const slopeInput = document.getElementById('road-slope');
    if (slopeInput) slopeInput.value = '30'; // typical stair angle
    
    // Stairs are usually covered
    const coveredInput = document.getElementById('road-covered');
    if (coveredInput) coveredInput.checked = true;
    
    setStatus('Điền thông tin cầu thang');
    
  } else if (method === 'tracking') {
    // Show tracking controls
    showRoadStep(3);
    setStatus('⚠️ Chỉ dùng tracking cho đường ngoài trời (GPS không hoạt động trong nhà)');
  }
}

function startWalkTracking() {
  if (!navigator.geolocation) {
    alert('GPS không khả dụng');
    return;
  }
  
  // Warning for indoor tracking
  if (roadPointA && roadPointB) {
    const floorA = roadPointA.floor || 1;
    const floorB = roadPointB.floor || 1;
    
    if (floorA > 1 || floorB > 1) {
      const confirmed = confirm(
        '⚠️ CẢNH BÁO: GPS không hoạt động trong nhà!\n\n' +
        'Bạn đang tạo đường cho tầng ' + floorA + ' → tầng ' + floorB + '.\n' +
        'GPS chỉ hoạt động ngoài trời (tầng 1, sân, đường).\n\n' +
        'Nếu đây là đường trong nhà/cầu thang, hãy dùng "Nối thẳng" hoặc "Cầu thang".\n\n' +
        'Vẫn muốn tracking?'
      );
      
      if (!confirmed) {
        return;
      }
    }
  }
  
  walkPoints = [];
  walkDistance = 0;
  walkLastPos = null;
  
  if (walkPolyline) map.removeLayer(walkPolyline);
  walkPolyline = L.polyline([], { color: '#10b981', weight: 4 }).addTo(map);
  
  const btnStart = document.getElementById('btn-start-walk');
  const btnStop = document.getElementById('btn-stop-walk');
  const status = document.getElementById('walk-status');
  
  if (btnStart) btnStart.style.display = 'none';
  if (btnStop) btnStop.style.display = 'inline-block';
  if (status) status.textContent = 'Đang tracking...';
  
  walkWatchId = navigator.geolocation.watchPosition(pos => {
    const la = pos.coords.latitude, lo = pos.coords.longitude;
    if (la == null || lo == null || isNaN(la) || isNaN(lo)) return;
    
    const pt = [la, lo];
    if (walkLastPos) {
      walkDistance += haversineM(walkLastPos[0], walkLastPos[1], la, lo);
    }
    walkLastPos = pt;
    walkPoints.push(pt);
    walkPolyline.addLatLng(pt);
    map.panTo(pt);
    
    if (status) {
      const accuracy = pos.coords.accuracy ? ` (±${Math.round(pos.coords.accuracy)}m)` : '';
      status.textContent = `${walkPoints.length} điểm · ${Math.round(walkDistance)}m${accuracy}`;
    }
  }, err => {
    setStatus('GPS error: ' + err.message);
  }, { enableHighAccuracy: true, maximumAge: 0, timeout: 5000 });
}

function stopWalkTracking() {
  if (walkWatchId !== null) {
    navigator.geolocation.clearWatch(walkWatchId);
    walkWatchId = null;
  }
  
  const btnStart = document.getElementById('btn-start-walk');
  const btnStop = document.getElementById('btn-stop-walk');
  const status = document.getElementById('walk-status');
  
  if (btnStart) btnStart.style.display = 'inline-block';
  if (btnStop) btnStop.style.display = 'none';
  if (status) status.textContent = `Đã dừng · ${walkPoints.length} điểm`;
  
  if (walkPoints.length < 2) {
    alert('Cần ít nhất 2 điểm GPS');
    return;
  }
  
  // Move to form
  showRoadStep(4);
  setStatus('Điền thông tin đường');
}

function resetWalkTracking() {
  if (walkWatchId !== null) {
    navigator.geolocation.clearWatch(walkWatchId);
    walkWatchId = null;
  }
  
  walkPoints = [];
  walkDistance = 0;
  walkLastPos = null;
  
  if (walkPolyline) {
    map.removeLayer(walkPolyline);
    walkPolyline = null;
  }
  
  const btnStart = document.getElementById('btn-start-walk');
  const btnStop = document.getElementById('btn-stop-walk');
  const status = document.getElementById('walk-status');
  
  if (btnStart) btnStart.style.display = 'inline-block';
  if (btnStop) btnStop.style.display = 'none';
  if (status) status.textContent = 'Chưa bắt đầu';
  
  setStatus('Đã reset tracking');
}

async function saveRoad() {
  if (!roadPointA || !roadPointB) {
    alert('Chọn 2 điểm trước');
    return;
  }
  
  const name = document.getElementById('road-name').value.trim() || 'Đường mới';
  const roadType = document.getElementById('road-type').value;
  const surface = document.getElementById('road-surface').value;
  const slope = parseFloat(document.getElementById('road-slope').value) || 0;
  const direction = document.getElementById('road-direction').value;
  const covered = document.getElementById('road-covered').checked;
  const vehicle = document.getElementById('road-vehicle').checked;
  
  const bidirectional = direction === 'both';
  
  // Simplify tracking points if using tracking method
  let geometry = null;
  if (roadMethod === 'tracking' && walkPoints.length >= 2) {
    geometry = douglasPeucker(walkPoints, 0.00002); // 2m epsilon
  }
  
  try {
    // Check for duplicate roads between same 2 points
    const checkUrl = API + `/api/edge/find?from_lat=${roadPointA.lat}&from_lon=${roadPointA.lon}&to_lat=${roadPointB.lat}&to_lon=${roadPointB.lon}`;
    const checkResp = await fetch(checkUrl);
    const checkData = await checkResp.json();
    
    let replaceEdgeId = null;
    
    if (checkData.ok && checkData.edges && checkData.edges.length > 0) {
      // Found existing road(s) between these 2 points
      const existingEdge = checkData.edges[0];
      const edgeNames = checkData.edges.map(e => e.name || 'Đường không tên').join(', ');
      
      const userChoice = confirm(
        `⚠️ Đã có ${checkData.edges.length} đường giữa 2 điểm này:\n` +
        `"${edgeNames}"\n\n` +
        `Bấm OK để THAY THẾ đường cũ\n` +
        `Bấm Cancel để TẠO THÊM đường mới`
      );
      
      if (userChoice) {
        // User chose to replace - delete old edge
        replaceEdgeId = existingEdge.id;
      }
      // If user chose Cancel, replaceEdgeId stays null and we create new edge
    }
    
    // Delete old edge if replacing
    if (replaceEdgeId) {
      await fetch(API + '/api/edge/' + replaceEdgeId, { method: 'DELETE' });
      
      // Remove from local array
      const idx = window.EDGES.findIndex(e => e.id === replaceEdgeId);
      if (idx >= 0) window.EDGES.splice(idx, 1);
    }
    
    // Create new edge
    const r = await fetch(API + '/api/edge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        from_lat: roadPointA.lat,
        from_lon: roadPointA.lon,
        to_lat: roadPointB.lat,
        to_lon: roadPointB.lon,
        road_type: roadType,
        bidirectional,
        from_floor: roadPointA.floor || 1,
        to_floor: roadPointB.floor || 1,
        is_covered: covered,
        surface,
        slope_deg: slope,
        geometry: geometry,
        // Note: vehicle allowed info - can add to schema later
      }),
    });
    
    const d = await r.json();
    if (d.ok) {
      const dm = typeof d.distance_m === 'number' ? d.distance_m : 0;
      window.EDGES.push({
        id: d.id,
        name,
        road_type: roadType,
        from_lat: roadPointA.lat,
        from_lon: roadPointA.lon,
        to_lat: roadPointB.lat,
        to_lon: roadPointB.lon,
        from_floor: roadPointA.floor || 1,
        to_floor: roadPointB.floor || 1,
        distance_m: dm,
        geometry: geometry,
      });
      
      renderEdges();
      
      const action = replaceEdgeId ? 'Đã thay thế' : 'Đã tạo';
      setStatus(`✅ ${action} đường "${name}" (~${Math.round(dm)}m)`);
      cancelRoad();
      setMode('view');
    } else {
      setStatus('❌ Lỗi tạo đường: ' + (d.error || 'Unknown'));
    }
  } catch(e) {
    setStatus('❌ ' + e.message);
  }
}

function cancelRoad() {
  roadPointA = roadPointB = null;
  roadMethod = null;
  walkPoints = [];
  walkDistance = 0;
  walkLastPos = null;
  
  if (walkWatchId !== null) {
    navigator.geolocation.clearWatch(walkWatchId);
    walkWatchId = null;
  }
  
  [roadMarkerA, roadMarkerB, roadPreviewLine, walkPolyline].forEach(l => {
    if (l) map.removeLayer(l);
  });
  
  roadMarkerA = roadMarkerB = roadPreviewLine = walkPolyline = null;
  
  // Reset displays
  const displayA = document.getElementById('point-a-display');
  const displayB = document.getElementById('point-b-display');
  if (displayA) displayA.textContent = 'Chưa chọn - click marker trên bản đồ';
  if (displayB) displayB.textContent = 'Chưa chọn - click marker trên bản đồ';
  
  // Reset form
  const nameInput = document.getElementById('road-name');
  if (nameInput) nameInput.value = '';
  
  const slopeInput = document.getElementById('road-slope');
  if (slopeInput) slopeInput.value = '0';
  
  const coveredInput = document.getElementById('road-covered');
  if (coveredInput) coveredInput.checked = true;
  
  const vehicleInput = document.getElementById('road-vehicle');
  if (vehicleInput) vehicleInput.checked = false;
  
  // Reset tracking UI
  const btnStart = document.getElementById('btn-start-walk');
  const btnStop = document.getElementById('btn-stop-walk');
  const walkStatus = document.getElementById('walk-status');
  if (btnStart) btnStart.style.display = 'inline-block';
  if (btnStop) btnStop.style.display = 'none';
  if (walkStatus) walkStatus.textContent = 'Chưa bắt đầu';
  
  // Hide all steps
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById('road-step-' + i);
    if (el) el.style.display = 'none';
  }
}

// Douglas-Peucker simplification
function douglasPeucker(points, epsilon) {
  if (points.length <= 2) return points;
  let maxDist = 0, maxIdx = 0;
  const first = points[0], last = points[points.length - 1];
  for (let i = 1; i < points.length - 1; i++) {
    const d = pointToSegmentDist(points[i], first, last);
    if (d > maxDist) { maxDist = d; maxIdx = i; }
  }
  if (maxDist > epsilon) {
    const left = douglasPeucker(points.slice(0, maxIdx + 1), epsilon);
    const right = douglasPeucker(points.slice(maxIdx), epsilon);
    return [...left.slice(0, -1), ...right];
  }
  return [first, last];
}

function pointToSegmentDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  if (dx === 0 && dy === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

// ============================================================================
// RENDERING
// ============================================================================

function buildLocationCoordGroups() {
  const g = new Map();
  for (const loc of window.LOCATIONS) {
    const k = coordKey6(loc);
    if (!g.has(k)) g.set(k, []);
    g.get(k).push(loc);
  }
  for (const arr of g.values()) {
    arr.sort((a, b) => (a.floor || 1) - (b.floor || 1) || a.id - b.id);
  }
  return g;
}

function displayLatLonForLocation(loc, coordGroups) {
  const k = coordKey6(loc);
  const grp = coordGroups.get(k) || [loc];
  const idx = Math.max(0, grp.findIndex(x => x.id === loc.id));
  const n = grp.length;
  const lat = Number(loc.lat) || 0, lon = Number(loc.lon) || 0;
  if (n <= 1) return [lat, lon];
  
  const floor = Number(loc.floor) || 1;
  const meters = 2.4 + idx * 3.0 + (floor - 1) * 2.8;
  const angleDeg = -90 + idx * (360 / n);
  const rad = angleDeg * Math.PI / 180;
  const cosLat = Math.cos(lat * Math.PI / 180) || 1e-6;
  const dLat = meters * Math.cos(rad) / 111320;
  const dLon = meters * Math.sin(rad) / (111320 * cosLat);
  return [lat + dLat, lon + dLon];
}

function renderLocations() {
  locMarkers.forEach(m => map.removeLayer(m));
  locMarkers = [];
  _coordGroups = buildLocationCoordGroups();

  const floors = new Set();
  window.LOCATIONS.forEach(loc => floors.add(loc.floor || 1));

  // Update floor filter
  const sel = document.getElementById('floor-sel');
  const prev = sel.value;
  sel.innerHTML = '<option value="all">Tất cả</option>';
  [...floors].sort().forEach(f => {
    const opt = document.createElement('option');
    opt.value = f;
    opt.textContent = `Tầng ${f}`;
    sel.appendChild(opt);
  });
  sel.value = prev;

  window.LOCATIONS.forEach(loc => {
    const floor = loc.floor || 1;
    if (currentFloor !== 'all' && floor != currentFloor) return;
    
    const color = floorColor(floor);
    const lid = String(loc.id != null ? loc.id : '?');
    const fs = lid.length >= 4 ? '8px' : lid.length >= 3 ? '9px' : '11px';
    const [mlat, mlon] = displayLatLonForLocation(loc, _coordGroups);
    
    const icon = L.divIcon({
      className: '',
      html: `<div class="loc-marker-badge" style="background:${color};font-size:${fs};">${lid}</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14],
    });
    
    const m = L.marker([mlat, mlon], { icon })
      .bindPopup(`<b>#${lid} · ${loc.name}</b><br>Tầng ${floor} · ${loc.category}`)
      .addTo(map);
    
    m.setZIndexOffset(500 + floor * 15);
    
    m.on('click', async (e) => {
      L.DomEvent.stopPropagation(e);
      
      if (mode === 'view') {
        await showLocationPopup(loc, [mlat, mlon]);
      } else if (mode === 'add-road') {
        selectRoadPoint(loc);
      }
    });
    
    locMarkers.push(m);
  });
}

function renderEdges() {
  edgeLines.forEach(l => map.removeLayer(l));
  edgeLines = [];
  
  window.EDGES.forEach(e => {
    const fromFloor = e.from_floor || 1;
    const toFloor = e.to_floor || 1;
    if (currentFloor !== 'all' && fromFloor != currentFloor && toFloor != currentFloor) return;
    
    const color = fromFloor !== toFloor ? '#f59e0b' : floorColor(fromFloor);
    const pts = e.geometry && e.geometry.length >= 2
      ? e.geometry.map(p => [p[0], p[1]])
      : [[e.from_lat, e.from_lon], [e.to_lat, e.to_lon]];
    
    const line = L.polyline(pts, {
      color,
      weight: 3,
      opacity: 0.85,
      dashArray: e.road_type === 'stairs' ? '6,4' : null,
    })
      .bindPopup(`<b>${e.name || 'Đường'}</b><br>${e.road_type} · ${Math.round(e.distance_m)}m`)
      .addTo(map);
    
    edgeLines.push(line);
  });
}

function filterFloor(val) {
  currentFloor = val === 'all' ? 'all' : parseInt(val);
  renderLocations();
  renderEdges();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

function initMap() {
  // Create map
  map = L.map('map').setView([10.8720, 106.8042], 17);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 21,
  }).addTo(map);
  
  // Load campus boundary
  fetch(API + '/api/campus/boundary').then(r => r.json()).then(b => {
    if (!b.polygon) return;
    
    const poly = b.polygon.map(p => [p[0], p[1]]);
    _campusBoundaryLayer = L.polygon(poly, {
      color: '#3b82f6',
      weight: 2,
      opacity: 0.8,
      fillColor: '#3b82f6',
      fillOpacity: 0.05,
      dashArray: '8,5',
    }).bindTooltip('ĐHKHTN CS2 — Khuôn viên', { sticky: true }).addTo(map);
  }).catch(() => {});
  
  // Add legend
  const legend = L.control({ position: 'bottomright' });
  legend.onAdd = () => {
    const div = L.DomUtil.create('div', 'legend');
    div.innerHTML = '<b style="font-size:11px">Địa điểm</b><br>' +
      '<span style="color:#cbd5e1">Số trong vòng = <b>ID</b> DB · Màu = tầng</span><br>' +
      window.FLOOR_COLORS.slice(0, 5).map((c, i) =>
        `<span class="legend-dot" style="background:${c}"></span>Tầng ${i + 1}<br>`
      ).join('') +
      '<hr style="border-color:#475569;margin:4px 0"/>' +
      '<span style="color:#f59e0b">━━</span> Đường liên tầng<br>' +
      '<span style="color:#10b981">━━</span> Đường tracking';
    return div;
  };
  legend.addTo(map);
  
  // Render locations and edges
  renderLocations();
  renderEdges();
  
  setStatus(`${window.LOCATIONS.length} địa điểm · ${window.EDGES.length} đường`);
}

// Start when DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initMap);
} else {
  initMap();
}
