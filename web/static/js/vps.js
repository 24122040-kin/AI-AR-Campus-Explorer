/**
 * vps.js — Visual Place Recognition (VPS) for location identification
 *
 * Provides:
 *   vpsIdentify(target)     — open modal to identify 'from' or 'to' location
 *   vpsCaptureLive()        — start live camera for VPS capture
 *   vpsSnapAndQuery()       — capture frame from live camera and query VPS
 *   vpsFromFile(input)      — query VPS from file picker
 *   vpsStopCam()            — stop live camera
 *   closeVpsModal()         — close VPS modal
 *
 * After identification, fills the target input ('from' or 'to') and
 * updates floor state if the matched location has floor info.
 */
'use strict';

let _vpsTarget  = 'from';   // which input to fill: 'from' | 'to'
let _vpsCamStream = null;

// ── Open modal ────────────────────────────────────────────────────────────────
function vpsIdentify(target) {
  _vpsTarget = target;
  const title = target === 'from' ? '📷 Nhận diện vị trí hiện tại' : '📷 Nhận diện điểm đến';
  el('vps-modal-title').textContent = title;
  el('vps-modal-result').style.display = 'none';
  el('vps-modal-result').innerHTML = '';
  el('vps-cam-wrap').style.display = 'none';
  el('vps-modal').style.display = 'flex';
}

function closeVpsModal() {
  vpsStopCam();
  el('vps-modal').style.display = 'none';
}

// ── Live camera capture ───────────────────────────────────────────────────────
async function vpsCaptureLive() {
  if (!navigator.mediaDevices?.getUserMedia) {
    return toast('Camera không khả dụng', 'warn');
  }
  try {
    _vpsCamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
      audio: false,
    });
    const v = el('vps-cam');
    v.srcObject = _vpsCamStream;
    v.setAttribute('playsinline', '');
    try { await v.play(); } catch (e) {}
    el('vps-cam-wrap').style.display = '';
  } catch (err) {
    let msg = err.message;
    if (err.name === 'NotAllowedError') msg = 'Bị từ chối quyền camera';
    if (err.name === 'NotFoundError')   msg = 'Không tìm thấy camera';
    toast('Không mở được camera: ' + msg, 'warn');
  }
}

function vpsStopCam() {
  if (_vpsCamStream) {
    _vpsCamStream.getTracks().forEach(t => t.stop());
    _vpsCamStream = null;
  }
  const v = el('vps-cam');
  if (v) v.srcObject = null;
  el('vps-cam-wrap').style.display = 'none';
}

async function vpsSnapAndQuery() {
  const v = el('vps-cam');
  if (!v || !_vpsCamStream) return toast('Camera chưa mở', 'warn');
  const w = v.videoWidth || 640, h = v.videoHeight || 480;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(v, 0, 0);
  c.toBlob(blob => {
    if (blob) _vpsQueryBlob(blob);
  }, 'image/jpeg', 0.85);
}

// ── File picker ───────────────────────────────────────────────────────────────
function vpsFromFile(input) {
  const file = input.files[0];
  if (!file) return;
  _vpsQueryBlob(file);
}

// ── Core VPS query ────────────────────────────────────────────────────────────
async function _vpsQueryBlob(blob) {
  const resultDiv = el('vps-modal-result');
  resultDiv.style.display = '';
  resultDiv.innerHTML = '<span class="typing-dots" style="font-size:12px;color:var(--text3)">Đang nhận diện</span>';

  const fd = new FormData();
  fd.append('file', blob instanceof File ? blob : new File([blob], 'vps.jpg', { type: 'image/jpeg' }));
  if (curLat) { fd.append('lat', curLat); fd.append('lon', curLon); }

  try {
    const r = await fetchWithTimeout(API + '/api/vpr/query', { method: 'POST', body: fd }, 20000);
    const d = await r.json();

    if (!d.ok) throw new Error(d.detail || 'VPS lỗi');
    if (!d.vpr_ready) {
      resultDiv.innerHTML = `
        <div style="font-size:12px;color:var(--amber)">
          ⚠️ VPR chưa sẵn sàng — cần upload ảnh và Rebuild VPR trước.<br>
          <span style="color:var(--text3)">Tab 📊 → Rebuild VPR</span>
        </div>`;
      return;
    }
    if (!d.matches?.length) {
      resultDiv.innerHTML = '<div style="font-size:12px;color:var(--text3)">Không nhận ra địa điểm này trong database.</div>';
      return;
    }

    // Render top 3 matches
    const top = d.matches.slice(0, 3);
    resultDiv.innerHTML = top.map((m, i) => {
      const score   = Math.round(m.score * 100);
      const floor   = m.floor ? `Tầng ${m.floor}` : '';
      const dist    = m.distance_m != null ? `${Math.round(m.distance_m)}m` : '';
      const imgHtml = m.primary_image_id
        ? `<img src="${API}/api/image/${m.primary_image_id}" style="width:40px;height:40px;object-fit:cover;border-radius:4px;flex-shrink:0" onerror="this.style.display='none'"/>`
        : `<div style="width:40px;height:40px;background:var(--bg4);border-radius:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:16px">📍</div>`;
      const barColor = score >= 70 ? 'var(--green)' : score >= 40 ? 'var(--amber)' : 'var(--text3)';
      return `
        <div style="display:flex;gap:8px;align-items:center;padding:6px;border-radius:6px;
                    background:var(--bg4);margin-bottom:4px;cursor:pointer"
             onclick="vpsSelectMatch(${JSON.stringify(m.location_name).replace(/"/g,'&quot;')},${m.lat},${m.lon},${m.floor||1})">
          ${imgHtml}
          <div style="flex:1;min-width:0">
            <div style="font-size:12px;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${m.location_name}</div>
            <div style="font-size:10px;color:var(--text3)">${[floor, dist].filter(Boolean).join(' · ')}</div>
            <div style="height:3px;background:var(--bg3);border-radius:2px;margin-top:3px">
              <div style="height:100%;width:${score}%;background:${barColor};border-radius:2px"></div>
            </div>
            <div style="font-size:10px;color:${barColor}">${score}% khớp</div>
          </div>
        </div>`;
    }).join('');

    // Auto-select if top match is very confident
    if (top[0].score >= 0.75) {
      _vpsApplyMatch(top[0].location_name, top[0].lat, top[0].lon, top[0].floor || 1);
    }

  } catch (e) {
    resultDiv.innerHTML = `<div style="font-size:12px;color:var(--red)">❌ ${e.message}</div>`;
  }
}

function vpsSelectMatch(name, lat, lon, floor) {
  _vpsApplyMatch(name, lat, lon, floor);
  closeVpsModal();
}

function _vpsApplyMatch(name, lat, lon, floor) {
  // Fill the target input
  const inputId = _vpsTarget === 'from' ? 'n-from' : 'n-to';
  const inp = el(inputId);
  if (inp && typeof _setPlaceDataset === 'function') {
    _setPlaceDataset(inp, { name, lat, lon, floor: floor || 1, id: null });
  } else if (inp) {
    inp.value = name;
    inp.dataset.lat = String(lat);
    inp.dataset.lon = String(lon);
  }

  // If identifying current position ('from'), update GPS state and floor
  if (_vpsTarget === 'from') {
    // Update floor from VPS match
    if (floor && floor >= 1) {
      floorState.floor      = floor;
      floorState.confidence = 0.85;
      floorState.method     = 'vps';
      _updateFloorHUD();
      // Also calibrate server-side
      fetchWithTimeout(API + '/api/realtime/floor/calibrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, floor }),
      }, 5000).catch(() => {});
    }
    // Show result in nav panel
    const panel = el('vps-result-panel');
    const content = el('vps-result-content');
    if (panel && content) {
      panel.style.display = '';
      content.innerHTML = `
        <div style="font-size:12px;color:var(--green);font-weight:600">📍 ${name}</div>
        <div style="font-size:11px;color:var(--text3)">${floor ? `Tầng ${floor} · ` : ''}VPS nhận diện</div>`;
    }
  }

  toast(`📍 VPS: ${name}${floor > 1 ? ` (Tầng ${floor})` : ''}`, 'ok');
}
