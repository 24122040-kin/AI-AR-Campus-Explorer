/**
 * route.js - route finding, location autocomplete, map panel
 * Depends on: globals.js, ar.js
 */
'use strict';

let _lastArPath = null;
let _lastRouteSteps = [];
let _acTimers = {};

/** Bản đồ mở = overlay toàn màn; đồng bộ mapOpen + aria-hidden */
function syncMapPanel(open) {
  mapOpen = open;
  const w = el('map-wrap');
  if (!w) return;
  w.classList.toggle('show', open);
  w.setAttribute('aria-hidden', open ? 'false' : 'true');
}

function _escapeHtml(s) {
  return String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function _acDropdown(inputId) {
  const id = inputId + '-ac';
  let drop = document.getElementById(id);
  if (!drop) {
    drop = document.createElement('div');
    drop.id = id;
    drop.className = 'ac-dropdown';
    const inp = document.getElementById(inputId);
    if (inp && inp.parentNode) {
      inp.parentNode.style.position = 'relative';
      inp.parentNode.appendChild(drop);
    }
  }
  return drop;
}

function _acHide(inputId) {
  const drop = document.getElementById(inputId + '-ac');
  if (drop) drop.innerHTML = '';
}

function _setPlaceDataset(inp, item) {
  inp.value = item.name || '';
  inp.dataset.lat = item.lat;
  inp.dataset.lon = item.lon;
  inp.dataset.floor = item.floor || 1;
  inp.dataset.placeId = item.id || '';
}

async function _searchPlaces(q, limit = 5, locationsOnly = false) {
  const extra = locationsOnly ? '&locations_only=1' : '';
  const r = await fetchWithTimeout(API + '/api/search?q=' + encodeURIComponent(q) + '&limit=' + limit + extra, {}, 6000);
  const d = await r.json();
  if (locationsOnly) return (d.locations || []).slice(0, limit);
  return [...(d.locations || []), ...(d.pois || [])].slice(0, limit);
}

async function _acSearch(inputId) {
  const inp = document.getElementById(inputId);
  if (!inp) return;
  const q = inp.value.trim();
  delete inp.dataset.lat;
  delete inp.dataset.lon;
  if (q.length < 1) { _acHide(inputId); return; }

  let results = [];
  try { results = await _searchPlaces(q, 8, inputId === 'n-from' || inputId === 'n-to'); } catch (e) {}
  const drop = _acDropdown(inputId);
  if (!results.length) {
    drop.innerHTML = '<div class="ac-empty">Không có địa điểm đã lưu khớp từ khóa. Thêm địa điểm (tab Dữ liệu) rồi thử lại.</div>';
    return;
  }

  drop.innerHTML = '';
  results.forEach(item => {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'ac-row';
    const floor = item.floor ? `Tầng ${item.floor}` : '';
    const cat = item.category || item.type || 'địa điểm';
    const imgId = item.primary_image_id_img || item.primary_image_id;
    const thumb = imgId
      ? `<img src="${API}/api/image/${imgId}" alt="" onerror="this.style.display='none'"/>`
      : '<span class="ac-pin">📍</span>';
    row.innerHTML = `
      ${thumb}
      <span class="ac-main">
        <strong>${_escapeHtml(item.name)}</strong>
        <small>${_escapeHtml([cat, floor].filter(Boolean).join(' · '))}</small>
      </span>`;
    row.addEventListener('mousedown', e => {
      e.preventDefault();
      _setPlaceDataset(inp, item);
      _acHide(inputId);
    });
    drop.appendChild(row);
  });
}

function onRouteInput(inputId) {
  clearTimeout(_acTimers[inputId]);
  _acTimers[inputId] = setTimeout(() => _acSearch(inputId), 220);
}

document.addEventListener('click', e => {
  ['n-from', 'n-to'].forEach(id => {
    const inp = document.getElementById(id);
    const drop = document.getElementById(id + '-ac');
    if (drop && inp && !inp.contains(e.target) && !drop.contains(e.target)) _acHide(id);
  });
});

function _selectedPlacePayload(input, prefix, payload) {
  const value = input.value.trim();
  if (!value) return true;
  if (!input.dataset.lat || !input.dataset.lon) return false;
  payload[prefix + '_lat'] = parseFloat(input.dataset.lat);
  payload[prefix + '_lon'] = parseFloat(input.dataset.lon);
  return true;
}

async function findRoute() {
  const fromInp = el('n-from');
  const toInp = el('n-to');
  const from = fromInp.value.trim();
  const to = toInp.value.trim();
  if (!to) return toast('Chọn điểm đến từ danh sách đã lưu', 'warn');

  const p = {
    destination: to,
    gps_accuracy_m: _gpsAccuracyM,
    session_id: sid,
    begin_navigation: true,
  };

  if (from) {
    if (!_selectedPlacePayload(fromInp, 'origin', p)) {
      return toast('Ô xuất phát: chọn một dòng trong gợi ý (địa điểm đã lưu).', 'warn');
    }
  } else if (curLat) {
    p.origin_lat = curLat;
    p.origin_lon = curLon;
  } else {
    return toast('Chưa có GPS. Chọn điểm xuất phát trong gợi ý hoặc bật GPS.', 'warn');
  }

  if (!_selectedPlacePayload(toInp, 'dest', p)) {
    return toast('Ô đến: phải chọn một địa điểm đã lưu trong gợi ý.', 'warn');
  }

  const tv = el('n-time').value;
  if (tv) {
    const [h, m] = tv.split(':').map(Number);
    p.depart_hour = h;
    p.depart_minute = m;
  }

  const weather = el('n-weather')?.value || 'auto';
  if (weather === 'rain') {
    p.weather_severity = 0.85;
    p.avoid_uncovered = true;
  } else if (weather === 'sunny') {
    p.weather_severity = 0.05;
    p.avoid_uncovered = false;
  }

  _acHide('n-from');
  _acHide('n-to');

  const statusEl = el('route-status-line');
  const resultHost = el('route-result-host');
  if (statusEl) statusEl.innerHTML = '<span class="typing-dots">Đang tính tuyến</span>';
  if (resultHost) { resultHost.style.display = 'none'; resultHost.innerHTML = ''; }

  try {
    const r = await fetchWithTimeout(API + '/api/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(p),
    }, 30000);
    const d = await r.json();
    if (!r.ok || !d.ok) throw new Error(d.detail || d.message || 'Không tìm được đường');

    if (statusEl) {
      statusEl.textContent = `✓ ${d.distance_km} km · ~${d.duration_min} phút`;
    }
    if (resultHost) {
      resultHost.style.display = '';
      resultHost.innerHTML = `<div>${d.html_card}</div>`;
    }
    if (mapOpen) el('map-frame').srcdoc = d.map_html;

    if (d.ar_path && d.ar_path.point_count > 0) {
      _lastArPath = d.ar_path;
      _lastRouteSteps = d.steps || [];
      _arFeedRoute(d.ar_path, d.steps || []);
      if (window.AREnhanced) AREnhanced.setRoute(_lastArPath, _lastRouteSteps);
      const arPanel = el('ar-route-panel');
      if (arPanel) arPanel.style.display = '';
      tab('nav');
    }
  } catch (e) {
    if (statusEl) statusEl.innerHTML = '❌ ' + _escapeHtml(e.message);
    toast(e.message, 'err');
  }
}

function askRouteBot() {
  const to = el('n-to').value.trim();
  if (!to) return toast('Chọn điểm đến', 'warn');
  el('ui-msg').value = `Tìm đường đến ${to}${el('n-time').value ? ' lúc ' + el('n-time').value : ''}`;
  sendMsg();
}

let _searchTimer = null;
function debounceSearch() {
  clearTimeout(_searchTimer);
  const q = el('q-search').value.trim();
  if (q.length < 1) { el('search-out').innerHTML = ''; return; }
  _searchTimer = setTimeout(() => doSearch(), 300);
}

async function doSearch() {
  const q = el('q-search').value.trim();
  if (!q) return;
  let all = [];
  try { all = await _searchPlaces(q, 10, true); } catch (e) { return toast('Lỗi tìm kiếm: ' + e.message, 'warn'); }

  const out = el('search-out');
  if (!all.length) {
    out.innerHTML = '<div class="empty-state">Không tìm thấy địa điểm nào trong dữ liệu.</div>';
    return;
  }
  out.innerHTML = `<div class="result-count">${all.length} kết quả</div>`;
  all.forEach(item => {
    const name = item.name || '';
    const floor = item.floor ? `Tầng ${item.floor}` : '';
    const cat = item.category || item.type || '';
    const imgId = item.primary_image_id_img || item.primary_image_id;
    const thumb = imgId
      ? `<img class="search-thumb" src="${API}/api/image/${imgId}" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"/>
         <div class="search-thumb-placeholder" style="display:none">📍</div>`
      : '<div class="search-thumb-placeholder">📍</div>';
    const div = document.createElement('div');
    div.className = 'search-result';
    div.innerHTML = `
      ${thumb}
      <div class="search-info">
        <div class="search-name">${_escapeHtml(name)}</div>
        <div class="search-meta">${_escapeHtml([cat, floor].filter(Boolean).join(' · '))}</div>
        <div class="search-actions">
          <button class="search-action-btn" data-action="go">Đi đến</button>
          <button class="search-action-btn" data-action="map">Xem map</button>
        </div>
      </div>`;
    div.querySelector('[data-action="go"]').addEventListener('click', () => goToLocation(item));
    div.querySelector('[data-action="map"]').addEventListener('click', () => showOnMap(item.lat, item.lon));
    out.appendChild(div);
  });
}

function goToLocation(item) {
  const toInp = el('n-to');
  _setPlaceDataset(toInp, item);
  tab('nav');
  findRoute();
}

function showOnMap(lat, lon) {
  syncMapPanel(true);
  el('map-frame').src = `${API}/api/map?lat=${lat}&lon=${lon}&zoom=19`;
}

function toggleMap() {
  syncMapPanel(!mapOpen);
  if (mapOpen) mapFull();
}
function mapClose() { syncMapPanel(false); }
function mapFull() {
  const lat = curLat || 10.8720;
  const lon = curLon || 106.8042;
  el('map-frame').src = `${API}/api/map?lat=${lat}&lon=${lon}`;
}
function mapRoute() {
  const f = el('mf').value;
  const t = el('mt').value;
  if (!t) return toast('Nhập tên điểm đã lưu (sidebar) rồi Tìm đường — ô map chỉ xem nhanh.', 'warn');
  el('map-frame').src = `${API}/api/route/map?from_q=${encodeURIComponent(f)}&to_q=${encodeURIComponent(t)}`;
}
function openLocalMapRoutePicker() {
  syncMapPanel(true);
  el('map-frame').src = API + '/api/localmap?mode=route-picker';
  tab('nav');
  const st = el('route-status-line');
  if (st) st.textContent = 'Bản đồ: chế độ 🚗 Tuyến 2 điểm — chọn A rồi B, bấm Gửi về form.';
}
function openIso() {
  if (!curLat) return toast('Cần GPS', 'warn');
  el('map-frame').src = `${API}/api/isochrone?lat=${curLat}&lon=${curLon}&minutes=5,10,15`;
  if (!mapOpen) toggleMap();
}

/** Điền 2 địa điểm đã lưu từ Local Map (postMessage) */
function applyRouteDriveFromMap(from, to) {
  const a = el('n-from');
  const b = el('n-to');
  if (!a || !b || !from || !to) return;
  _setPlaceDataset(a, {
    id: from.id,
    name: from.name,
    lat: from.lat,
    lon: from.lon,
    floor: from.floor || 1,
  });
  _setPlaceDataset(b, {
    id: to.id,
    name: to.name,
    lat: to.lat,
    lon: to.lon,
    floor: to.floor || 1,
  });
  tab('nav');
  mapClose();
  toast('Đã điền xuất phát & đến (địa điểm đã lưu). Bấm «Tìm đường».', 'ok');
}

document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || !mapOpen) return;
  const w = el('map-wrap');
  if (w && w.classList.contains('show')) {
    e.preventDefault();
    mapClose();
  }
});

window.addEventListener('message', (e) => {
  if (e.data === 'close-localmap') mapClose();
});
