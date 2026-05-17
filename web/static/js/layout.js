/**
 * layout.js — Tabs, accordion modules, help system, status panel
 * Depends on: globals.js
 */
'use strict';

// ── Tabs ──────────────────────────────────────────────────────────────────────
function tab(name) {
  ['nav', 'data', 'traffic', 'stats'].forEach((n, i) => {
    document.querySelectorAll('.tab')[i].classList.toggle('active', n === name);
    el('tp-' + n).classList.toggle('active', n === name);
  });
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function toggleSidebar() {
  sideOpen = !sideOpen;
  el('sidebar').classList.toggle('collapsed', !sideOpen);
}

// ── Accordion modules ─────────────────────────────────────────────────────────
function toggleModule(id) {
  const body  = el(id);
  const chev  = el('chev-' + id);
  const open  = body.classList.toggle('open');
  if (chev) chev.classList.toggle('open', open);
}

// ── Help system ───────────────────────────────────────────────────────────────
const HELP_CONTENT = {
  route: {
    title: '🔍 Tìm đường (chỉ địa điểm đã lưu)',
    html: `
      <ul>
        <li><b>Nguồn điểm:</b> Chỉ <b>locations</b> trong database (đã thêm ở tab Dữ liệu). Gõ tên → chọn <em>một dòng</em> trong gợi ý (có tọa độ).</li>
        <li><b>GPS:</b> Để trống «Điểm xuất phát» = dùng vị trí hiện tại nếu đã bật GPS.</li>
        <li><b>Bản đồ nội bộ:</b> Bấm «Chọn 2 điểm trên bản đồ» → chế độ 🚗 → click marker hoặc map gần marker (≤55m) → Gửi về form.</li>
        <li><b>Tracking đi bộ</b> trên local map là để <em>vẽ cạnh / hành lang</em>, không phải tuyến ô tô OSM.</li>
        <li><b>AR / Isochrone:</b> Sau khi tuyến tính xong.</li>
      </ul>
    `
  },
  search: {
    title: '📍 Tìm kiếm địa điểm',
    html: `
      <ul>
        <li>Gõ tên địa điểm → kết quả hiện ngay (tìm kiếm ngữ nghĩa, top 10).</li>
        <li>Kết quả có ảnh thumbnail, tầng, loại địa điểm.</li>
        <li><b>Đi đến:</b> Tự điền vào ô "Đến" và tìm đường.</li>
        <li><b>Xem trên map:</b> Mở bản đồ tại vị trí đó.</li>
        <li>Tìm kiếm ưu tiên: tên chính xác → bắt đầu bằng → chứa từ → mô tả.</li>
      </ul>
    `
  },
  addloc: {
    title: '📌 Thêm địa điểm + ảnh',
    html: `
      <ul>
        <li><b>Tên:</b> Tên rõ ràng, dễ tìm (Tòa B - Phòng 201, Căng tin tầng 1...).</li>
        <li><b>Lat/Lon:</b> Để trống → tự lấy GPS hiện tại. Hoặc copy từ Google Maps.</li>
        <li><b>Tầng:</b> Số tầng (1 = trệt). Quan trọng cho điều hướng trong nhà.</li>
        <li><b>Ảnh (1–5 tấm):</b> Ảnh đầu tiên = ảnh chính hiển thị khi search. Các ảnh còn lại dùng cho nhận diện VPS (Visual Place Recognition).</li>
        <li>Bấm vào ảnh trong preview để đặt làm ảnh chính.</li>
        <li>Ảnh nên chụp từ nhiều góc khác nhau để VPS nhận diện chính xác hơn.</li>
      </ul>
    `
  },
  addedge: {
    title: '🛤️ Thêm đường / hành lang',
    html: `
      <ul>
        <li><b>Điểm đầu/cuối:</b> Nhập "lat,lon" hoặc dùng Local Map Editor để click chọn điểm.</li>
        <li><b>Loại đường:</b> Chọn đúng loại để AR cảnh báo phù hợp (cầu thang, dốc...).</li>
        <li><b>Tầng đầu/cuối:</b> Nếu khác nhau = đường liên tầng (cầu thang, thang máy).</li>
        <li><b>Độ dốc:</b> Góc nghiêng tính bằng độ. Dương = lên dốc, âm = xuống dốc. AR sẽ cảnh báo khi > 10°.</li>
        <li><b>Có mái che:</b> Ảnh hưởng đến gợi ý đường khi trời mưa.</li>
        <li>💡 Dùng <b>Local Map Editor</b> để thêm đường bằng cách click 2 điểm trực quan hơn.</li>
      </ul>
    `
  },
  localmap: {
    title: '🗺️ Local Map Editor',
    html: `
      <ul>
        <li><b>👁 Xem:</b> Địa điểm đã lưu, màu theo tầng.</li>
        <li><b>🚗 Tuyến 2 điểm:</b> Chọn điểm xuất phát & đến (chỉ từ DB) → Gửi về form «Tìm đường» (tuyến lái xe / OSM). Click map sẽ bắt điểm gần nhất trong bán kính 55m.</li>
        <li><b>➕ Nối cạnh:</b> Hai điểm đã lưu → gửi về form thêm đường nội bộ / hành lang.</li>
        <li><b>🚶 Tracking đi bộ:</b> Ghi GPS để lưu path (không phải tuyến ô tô).</li>
      </ul>
    `
  },
  upload: {
    title: '📤 Upload ảnh nhanh',
    html: `
      <p><b>Mục đích:</b> Upload nhanh ảnh từ gallery điện thoại để tự động tạo địa điểm mới vào database.</p>
      <ul>
        <li>Chụp ảnh bằng <b>camera app mặc định</b> của điện thoại (không phải camera trong app này).</li>
        <li>Ảnh đó sẽ có GPS EXIF — server tự đọc tọa độ và tạo địa điểm.</li>
        <li>Khác với "Thêm địa điểm + ảnh" — upload nhanh không cần điền tên, tự đặt tên theo GPS.</li>
        <li>Dùng khi đi dạo trong trường, chụp nhiều ảnh liên tiếp — upload 1 lần.</li>
        <li>Sau khi upload → Tab 📊 → <b>Rebuild VPR</b> để nhận diện được.</li>
      </ul>
    `
  },
  camera: {
    title: '📷 Camera — Nhận diện vị trí',
    html: `
      <ul>
        <li><b>Mở camera:</b> Bật camera sau điện thoại.</li>
        <li><b>Chụp & nhận dạng:</b> Chụp frame → upload → VPR so khớp với ảnh đã index → hiện tên địa điểm.</li>
        <li>Cần có ảnh đã upload trước (VPR index > 0) mới nhận diện được.</li>
        <li>Camera cần HTTPS để hoạt động trên điện thoại.</li>
      </ul>
    `
  },
  traffic: {
    title: '📊 Biểu đồ tắc nghẽn',
    html: `
      <ul>
        <li>Biểu đồ 24h hiển thị mức độ tắc nghẽn theo giờ trong ngày.</li>
        <li>Màu xanh = thông thoáng, vàng = bình thường, đỏ = tắc nghẽn.</li>
        <li>Dữ liệu từ báo cáo của người dùng + pattern mặc định.</li>
        <li><b>Giờ tốt nhất:</b> Gợi ý giờ đi ít tắc nhất trong ±2h.</li>
      </ul>
    `
  },
  'traffic-report': {
    title: '📡 Báo cáo tắc nghẽn',
    html: `
      <ul>
        <li>Chọn giờ và kéo slider mức độ tắc (0 = thông, 1 = kẹt).</li>
        <li>Báo cáo được lưu vào DB và ảnh hưởng đến tính toán tuyến đường.</li>
        <li>Tọa độ tự lấy từ GPS hiện tại.</li>
      </ul>
    `
  },
  stats: {
    title: '📈 Thống kê hệ thống',
    html: `
      <ul>
        <li><b>Địa điểm:</b> Số địa điểm đã thêm vào DB.</li>
        <li><b>Ảnh:</b> Số ảnh đã upload.</li>
        <li><b>VPR index:</b> Số ảnh đã được index vào FAISS (dùng cho nhận diện).</li>
        <li><b>Rebuild VPR:</b> Chạy sau khi upload nhiều ảnh mới để cập nhật index.</li>
        <li>AI risk score: Đánh giá mức độ sẵn sàng của hệ thống AI.</li>
      </ul>
    `
  },
};

function showHelp(key) {
  const h = HELP_CONTENT[key];
  if (!h) return;
  el('help-title').innerHTML = h.title;
  el('help-content').innerHTML = h.html;
  el('help-overlay').style.display = 'flex';
}

function closeHelp() {
  el('help-overlay').style.display = 'none';
}

// Close help on Escape
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeHelp(); });

// ── Status panel ──────────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const r  = await fetchWithTimeout(API + '/api/status', {}, 8000);
    if (!r.ok) return;
    const d  = await r.json();
    const ai = await fetchWithTimeout(API + '/api/ai/readiness', {}, 8000)
      .then(x => x.ok ? x.json() : null)
      .catch(() => null);

    el('b-vpr').textContent = d.vpr_ready ? `VPR ✓ (${d.vpr_index_size})` : 'VPR ✗';
    el('b-vpr').className   = 'badge ' + (d.vpr_ready ? 'ok' : 'warn');
    el('b-rt').textContent  = d.valhalla ? 'Valhalla ✓' : (d.osm_graph_cached ? 'OSM ✓' : 'OSM ✗');
    el('b-rt').className    = 'badge ' + (d.osm_graph_cached ? 'ok' : 'warn');
    el('hdr-stat').textContent = `${d.locations}📍 ${d.images || 0}🖼`;
    el('sv-loc').textContent = d.locations;
    el('sv-img').textContent = d.images || 0;
    el('sv-poi').textContent = d.pois;
    el('sv-vpr').textContent = d.vpr_index_size;

    const riskColor = !ai ? '#64748b'
      : ai.risk_level === 'low' ? 'var(--green)'
      : ai.risk_level === 'medium' ? 'var(--amber)' : 'var(--red)';
    el('sys-info').innerHTML =
      `Device: <b>${d.device}</b> · Model: <b>${d.model}</b><br>` +
      `VPR backend: <b>${d.vpr_backend || 'n/a'}</b><br>` +
      `Sessions: <b>${d.sessions?.total || 0}</b>` +
      (ai ? `<br>AI risk: <b style="color:${riskColor}">${ai.risk_level?.toUpperCase()} (${ai.risk_score}/100)</b>` : '') +
      (ai?.risks?.length ? `<br><span style="color:var(--text3);font-size:10px">${ai.risks[0]}</span>` : '');
  } catch (e) { /* server not ready */ }
}
loadStatus();
setInterval(loadStatus, 30000);

// ── VPR rebuild ───────────────────────────────────────────────────────────────
async function vprRebuild() {
  const r = await fetchWithTimeout(API + '/api/vpr/rebuild', { method: 'POST' }, 10000);
  const d = await r.json();
  toast(d.message || 'Rebuilding...', 'ok');
  if (d.job?.job_id) pollJob(d.job.job_id, 'VPR rebuild');
}

async function pollJob(jobId, label) {
  let tries = 0;
  const tick = async () => {
    tries++;
    try {
      const r   = await fetchWithTimeout(API + `/api/jobs/${jobId}`, {}, 5000);
      const d   = await r.json();
      const job = d.job;
      if (!job) return;
      if (job.status === 'completed') { toast(`✅ ${label} xong`, 'ok'); loadStatus(); return; }
      if (job.status === 'failed')    { toast(`❌ ${label} thất bại: ${job.error || job.message}`, 'warn'); return; }
      if (tries < 120) setTimeout(tick, 2000);
    } catch (e) {
      if (tries < 120) setTimeout(tick, 3000);
    }
  };
  tick();
}

// ── Experimental UI buttons ───────────────────────────────────────────────────
function initExperimentalUI() {
  const camBody = el('mod-camera');
  if (!camBody || el('btn-landmarks')) return;
  const buttons = [
    { id: 'btn-landmarks', text: '🔍 Nhận diện landmark', fn: detectLandmarks },
    { id: 'btn-scene',     text: '🌆 Phân tích cảnh',     fn: analyzeScene },
  ];
  const ref = camBody.querySelector('button.btn.danger');
  if (!ref) return;
  for (const { id, text, fn } of buttons) {
    const btn = document.createElement('button');
    btn.id = id; btn.className = 'btn secondary'; btn.textContent = text;
    btn.onclick = fn; btn.style.marginTop = '6px';
    ref.insertAdjacentElement('afterend', btn);
  }
}
initExperimentalUI();
