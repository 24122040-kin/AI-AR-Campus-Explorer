/**
 * data.js - location, edge, image upload management
 * Depends on: globals.js, layout.js, route.js
 */
'use strict';

let _locImgFiles = [];
let _primaryIdx = 0;
let _edgePlaces = { 'e-a': null, 'e-b': null };
let _edgeTimers = {};

function previewLocationImages(input) {
  const files = Array.from(input.files).slice(0, 5);
  _locImgFiles = files.map((f, i) => ({ file: f, url: URL.createObjectURL(f), isPrimary: i === 0 }));
  _primaryIdx = 0;
  _renderImgPreview();
}

function _renderImgPreview() {
  const grid = el('d-img-preview');
  if (!grid) return;
  grid.innerHTML = '';
  _locImgFiles.forEach((item, i) => {
    const div = document.createElement('div');
    div.className = 'img-thumb' + (i === _primaryIdx ? ' primary' : '');
    div.title = i === _primaryIdx ? 'Ảnh chính' : 'Bấm để đặt làm ảnh chính';
    div.innerHTML = `
      <img src="${item.url}" alt="preview"/>
      ${i === _primaryIdx ? '<span class="primary-badge">CHÍNH</span>' : ''}
      <button class="remove-btn" onclick="event.stopPropagation();removeLocImg(${i})">✕</button>`;
    div.addEventListener('click', () => {
      _primaryIdx = i;
      _locImgFiles.forEach((x, j) => x.isPrimary = j === i);
      _renderImgPreview();
    });
    grid.appendChild(div);
  });
}

function removeLocImg(idx) {
  URL.revokeObjectURL(_locImgFiles[idx].url);
  _locImgFiles.splice(idx, 1);
  if (_primaryIdx >= _locImgFiles.length) _primaryIdx = Math.max(0, _locImgFiles.length - 1);
  _renderImgPreview();
}

async function addLocationWithImages() {
  const name = el('d-name').value.trim();
  const lat = parseFloat(el('d-lat').value) || curLat;
  const lon = parseFloat(el('d-lon').value) || curLon;
  const floor = parseInt(el('d-floor').value) || 1;
  if (!name) return toast('Nhập tên địa điểm', 'warn');
  if (!lat || !lon) return toast('Chưa có tọa độ - bật GPS hoặc nhập tay', 'warn');

  const r1 = await fetchWithTimeout(API + '/api/location', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name, lat, lon, floor,
      description: el('d-desc').value,
      category: el('d-cat').value,
      importance: parseInt(el('d-imp').value),
    }),
  }, 10000);
  const d1 = await r1.json();
  if (!d1.ok) return toast('❌ Lỗi tạo địa điểm', 'err');

  if (_locImgFiles.length > 0) {
    const fd = new FormData();
    fd.append('location_id', d1.id);
    fd.append('primary_index', _primaryIdx);
    fd.append('auto_caption', el('d-auto').checked ? 'true' : 'false');
    _locImgFiles.forEach(item => fd.append('files', item.file));
    const r2 = await fetchWithTimeout(API + '/api/location/images', { method: 'POST', body: fd }, 60000);
    const d2 = await r2.json();
    const ok = d2.images?.filter(x => x.ok).length || 0;
    toast(`✅ Đã thêm "${name}" (Tầng ${floor}) + ${ok}/${_locImgFiles.length} ảnh`, 'ok');
  } else {
    toast(`✅ Đã thêm "${name}" (Tầng ${floor})`, 'ok');
  }

  ['d-name', 'd-lat', 'd-lon', 'd-desc'].forEach(id => { if (el(id)) el(id).value = ''; });
  el('d-floor').value = '1';
  _locImgFiles = [];
  _primaryIdx = 0;
  _renderImgPreview();
  loadStatus();
}

function _edgeDropdown(inputId) {
  let drop = document.getElementById(inputId + '-ac');
  if (!drop) {
    drop = document.createElement('div');
    drop.id = inputId + '-ac';
    drop.className = 'ac-dropdown';
    const inp = el(inputId);
    if (inp && inp.parentNode) {
      inp.parentNode.style.position = 'relative';
      inp.parentNode.appendChild(drop);
    }
  }
  return drop;
}

function _hideEdgeDropdown(inputId) {
  const drop = document.getElementById(inputId + '-ac');
  if (drop) drop.innerHTML = '';
}

function _setEdgePlace(inputId, item) {
  _edgePlaces[inputId] = item;
  const inp = el(inputId);
  if (inp) inp.value = item.name || '';
  _hideEdgeDropdown(inputId);
  _renderEdgeSelection();
}

function _renderEdgeSelection() {
  const a = _edgePlaces['e-a'];
  const b = _edgePlaces['e-b'];
  if (el('edge-points-display')) el('edge-points-display').style.display = a || b ? '' : 'none';
  if (el('edge-pt-a')) el('edge-pt-a').textContent = a ? `${a.name} (Tầng ${a.floor || 1})` : 'Chưa chọn điểm A';
  if (el('edge-pt-b')) el('edge-pt-b').textContent = b ? `${b.name} (Tầng ${b.floor || 1})` : 'Chưa chọn điểm B';
  const btn = el('btn-save-edge');
  if (btn) {
    const ready = Boolean(a && b);
    btn.disabled = !ready;
    btn.style.opacity = ready ? '1' : '0.5';
  }
}

function onEdgePlaceInput(inputId) {
  clearTimeout(_edgeTimers[inputId]);
  _edgeTimers[inputId] = setTimeout(async () => {
    const inp = el(inputId);
    const q = inp?.value.trim() || '';
    _edgePlaces[inputId] = null;
    _renderEdgeSelection();
    if (!q) return _hideEdgeDropdown(inputId);
    let results = [];
    try {
      const r = await fetchWithTimeout(API + '/api/search?q=' + encodeURIComponent(q) + '&limit=5', {}, 6000);
      const d = await r.json();
      results = (d.locations || []).slice(0, 5);
    } catch (e) {}
    const drop = _edgeDropdown(inputId);
    if (!results.length) {
      drop.innerHTML = '<div class="ac-empty">Không có địa điểm này. Hãy tạo địa điểm trước rồi mới nối đường.</div>';
      return;
    }
    drop.innerHTML = '';
    results.forEach(item => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'ac-row';
      row.innerHTML = `<span class="ac-pin">📍</span><span class="ac-main"><strong>${item.name}</strong><small>Tầng ${item.floor || 1} · ${item.category || 'địa điểm'}</small></span>`;
      row.addEventListener('mousedown', e => {
        e.preventDefault();
        _setEdgePlace(inputId, item);
      });
      drop.appendChild(row);
    });
  }, 220);
}

async function addEdge() {
  const a = _edgePlaces['e-a'];
  const b = _edgePlaces['e-b'];
  if (!a || !b) return toast('Chọn 2 địa điểm đã tạo trong gợi ý hoặc trên map', 'warn');

  const r = await fetchWithTimeout(API + '/api/edge', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: el('e-name').value.trim(),
      from_lat: a.lat,
      from_lon: a.lon,
      to_lat: b.lat,
      to_lon: b.lon,
      road_type: el('e-type').value,
      bidirectional: true,
      from_floor: parseInt(a.floor) || 1,
      to_floor: parseInt(b.floor) || 1,
      is_covered: el('e-covered').checked,
      has_lighting: el('e-lighting').checked,
      surface: el('e-surface').value,
      slope_deg: parseFloat(el('e-slope').value) || 0,
      width_m: parseFloat(el('e-width').value) || null,
    }),
  }, 10000);
  const d = await r.json();
  if (d.ok) {
    toast('✅ Đã thêm đường', 'ok');
    ['e-name', 'e-a', 'e-b', 'e-width'].forEach(id => { if (el(id)) el(id).value = ''; });
    el('e-slope').value = '0';
    _edgePlaces = { 'e-a': null, 'e-b': null };
    _renderEdgeSelection();
    loadStatus();
  } else {
    toast('❌ Lỗi thêm đường', 'err');
  }
}

async function uploadImgs() {
  const files = el('u-files').files;
  if (!files.length) return toast('Chọn ảnh', 'warn');
  let ok = 0;
  for (const file of files) {
    const fd = new FormData();
    fd.append('file', file);
    if (curLat) { fd.append('lat', curLat); fd.append('lon', curLon); }
    fd.append('caption', el('u-cap').value);
    fd.append('auto_caption', el('u-auto').checked ? 'true' : 'false');
    try {
      const r = await fetchWithTimeout(API + '/api/upload/image', { method: 'POST', body: fd }, 30000);
      const d = await r.json();
      if (d.ok) ok++;
    } catch (e) {}
  }
  toast(`${ok > 0 ? '✅' : '❌'} Upload ${ok}/${files.length} ảnh`, ok > 0 ? 'ok' : 'warn');
  loadStatus();
}

function openLocalMap() {
  mapOpen = true; el('map-wrap').classList.add('show');
  el('map-frame').src = API + '/api/localmap';
}

function openLocalMapForEdge() {
  mapOpen = true; el('map-wrap').classList.add('show');
  el('map-frame').src = API + '/api/localmap?mode=edge-picker';
  tab('data');
}

window.addEventListener('message', evt => {
  const d = evt.data;
  if (!d || typeof d !== 'object') return;
  if (d.type === 'localnav-edge-points') {
    if (d.a) _setEdgePlace('e-a', d.a);
    if (d.b) _setEdgePlace('e-b', d.b);
    toast('Đã chọn 2 điểm. Điền thông tin đường rồi lưu.', 'ok');
    return;
  }
  if (d.type === 'localnav-route-drive' && d.from && d.to) {
    applyRouteDriveFromMap(d.from, d.to);
  }
  if (d.type === 'localnav-set-route-point' && d.field && d.location) {
    const field = d.field; // 'from' or 'to'
    const loc = d.location;
    const inputId = field === 'from' ? 'n-from' : 'n-to';
    const inp = el(inputId);
    if (inp) {
      inp.value = loc.name;
      // Store location data in input element for later use
      inp.dataset.locationId = loc.id;
      inp.dataset.lat = loc.lat;
      inp.dataset.lon = loc.lon;
      inp.dataset.floor = loc.floor || 1;
    }
    toast(`✅ Đã chọn "${loc.name}" làm ${field === 'from' ? 'điểm xuất phát' : 'điểm đến'}`, 'ok');
    mapClose();
  }
});
