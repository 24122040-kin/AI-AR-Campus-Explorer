/**
 * camera.js — Camera capture + 2D location recognition
 * Depends on: globals.js
 */
'use strict';

// ── Camera capture ────────────────────────────────────────────────────────────
function refreshCamMeta() {
  const elMeta = el('cam-meta');
  if (!elMeta) return;
  const latest = capturedFrames.length ? capturedFrames[capturedFrames.length - 1] : null;
  elMeta.textContent = `Frames: ${capturedFrames.length}` +
    (latest?.lat ? ` | GPS ${latest.lat.toFixed(5)}, ${latest.lon.toFixed(5)}` : '');
}

async function startCameraCapture() {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('Trình duyệt không hỗ trợ camera', 'warn'); return;
  }
  if (location.protocol !== 'https:' &&
      location.hostname !== 'localhost' &&
      location.hostname !== '127.0.0.1') {
    // Warn only — camera may still work on LAN (Android Chrome allows it)
    toast('⚠️ Camera hoạt động tốt hơn trên HTTPS. Đang thử mở...', 'warn');
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    const v = el('cam-preview');
    if (v) {
      v.srcObject = camStream;
      v.setAttribute('playsinline', '');
      v.setAttribute('muted', '');
      try { await v.play(); } catch (e) { /* autoplay may be blocked */ }
    }
    toast('📷 Camera sẵn sàng', 'ok');
  } catch (err) {
    let msg = err.message;
    if (err.name === 'NotAllowedError')  msg = 'Bị từ chối quyền camera. Vào Settings > Safari > Camera để cấp quyền.';
    if (err.name === 'NotFoundError')    msg = 'Không tìm thấy camera trên thiết bị.';
    if (err.name === 'NotReadableError') msg = 'Camera đang được dùng bởi app khác.';
    toast('Không mở được camera: ' + msg, 'warn');
  }
}

function stopCameraCapture() {
  if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }
  const v = el('cam-preview');
  if (v) v.srcObject = null;
  toast('Đã dừng camera', 'ok');
}

function captureFrame() {
  const v = el('cam-preview');
  if (!v || !camStream) { toast('Mở camera trước', 'warn'); return; }
  const vw = v.videoWidth || v.clientWidth || 1280;
  const vh = v.videoHeight || v.clientHeight || 720;
  if (vw === 0 || vh === 0) { toast('Camera chưa sẵn sàng, thử lại', 'warn'); return; }
  // Limit to 30 frames to avoid RAM crash on phone
  if (capturedFrames.length >= 30) {
    toast('Đã đạt giới hạn 30 frames. Xóa bớt trước.', 'warn'); return;
  }
  const c = document.createElement('canvas');
  c.width = vw; c.height = vh;
  c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
  c.toBlob(blob => {
    if (!blob) return;
    capturedFrames.push({ blob, name: `capture_${Date.now()}.jpg`, lat: curLat, lon: curLon, ts: new Date().toISOString() });
    refreshCamMeta();
    toast('Đã chụp frame #' + capturedFrames.length, 'ok');
    // Auto-upload for VPR 2D recognition if GPS available
    if (curLat && curLon) _autoUploadFrame(blob);
  }, 'image/jpeg', 0.92);
}

// Auto-upload captured frame for 2D location indexing (VPR)
// Throttled: max 1 upload per 5 seconds to avoid flooding the server
let _lastAutoUploadTs = 0;
async function _autoUploadFrame(blob) {
  const now = Date.now();
  if (now - _lastAutoUploadTs < 5000) return;  // throttle: 1 per 5s
  _lastAutoUploadTs = now;

  const fd = new FormData();
  fd.append('file', new File([blob], `frame_${now}.jpg`, { type: 'image/jpeg' }));
  fd.append('lat', curLat);
  fd.append('lon', curLon);
  fd.append('location_name', `GPS_${curLat.toFixed(4)}_${curLon.toFixed(4)}`);
  fd.append('auto_caption', 'false');
  try {
    await fetchWithTimeout(API + '/api/upload/image', { method: 'POST', body: fd }, 15000);
  } catch (e) { /* silent — frame is still saved locally */ }
}

function clearCapturedFrames() {
  capturedFrames = []; refreshCamMeta(); toast('Đã xóa frames', 'ok');
}

// ── 2D Landmark / Scene detection ─────────────────────────────────────────────
async function detectLandmarks() {
  const source = chatImg || el('u-files')?.files[0] || null;
  if (!source) return toast('Chọn 1 ảnh để nhận diện', 'warn');
  const fd = new FormData(); fd.append('file', source);
  const bot = appendMsg('<span class="typing-dots">Đang nhận diện địa điểm</span>', 'bot');
  try {
    const r = await fetchWithTimeout(API + '/api/experimental/landmarks', { method: 'POST', body: fd }, 20000);
    const d = await r.json();
    if (!r.ok || !d.ok) throw new Error(d.detail || 'Không nhận diện được');
    if (!d.available) { bot.innerHTML = md(`YOLO chưa sẵn sàng.\n${d.message || ''}`); return; }
    if (!d.detections.length) { bot.innerHTML = 'Không phát hiện landmark rõ ràng trong ảnh này.'; return; }
    const lines   = d.detections.slice(0, 8).map((det, i) => `${i + 1}. **${det.label}** - ${(det.confidence * 100).toFixed(1)}%`);
    const preview = d.preview_url ? `<br><a href="${d.preview_url}" target="_blank">Xem ảnh annotate</a><br><img src="${d.preview_url}" style="margin-top:8px;max-width:100%;border-radius:10px;border:1px solid var(--border)"/>` : '';
    const shape   = (d.image_width && d.image_height) ? `\nKích thước: ${d.image_width} x ${d.image_height}` : '';
    bot.innerHTML = md(`Nhận diện ${d.detections.length} đối tượng bằng **${d.model}**:${shape}\n${lines.join('\n')}`) + preview;
  } catch (e) { bot.innerHTML = 'Lỗi nhận diện: ' + e.message; }
}

async function analyzeScene() {
  const source = chatImg || el('u-files')?.files[0] || null;
  if (!source) return toast('Chọn 1 ảnh để phân tích', 'warn');
  const fd = new FormData(); fd.append('file', source);
  const bot = appendMsg('<span class="typing-dots">Đang phân tích cảnh</span>', 'bot');
  try {
    const r = await fetchWithTimeout(API + '/api/experimental/scene', { method: 'POST', body: fd }, 25000);
    const d = await r.json();
    if (!r.ok || !d.ok) throw new Error(d.detail || 'Không phân tích được');
    const marks   = (d.landmarks?.detections || []).slice(0, 6).map((det, i) => `${i + 1}. **${det.label}** - ${(det.confidence * 100).toFixed(1)}%`);
    const texts   = (d.ocr?.blocks || []).slice(0, 6).map((b, i) => `${i + 1}. \`${b.text}\` - ${(b.confidence * 100).toFixed(1)}%`);
    const preview = d.preview_url ? `<br><a href="${d.preview_url}" target="_blank">Xem ảnh kết quả</a><br><img src="${d.preview_url}" style="margin-top:8px;max-width:100%;border-radius:10px;border:1px solid var(--border)"/>` : '';
    bot.innerHTML = md([
      `**Tổng quan**\n${d.summary || 'Không có tóm tắt.'}`,
      marks.length ? `**Landmark**\n${marks.join('\n')}` : '**Landmark**\nKhông thấy mốc rõ ràng.',
      texts.length ? `**Text/OCR**\n${texts.join('\n')}` : '**Text/OCR**\nKhông đọc được text.',
    ].join('\n\n')) + preview;
  } catch (e) { bot.innerHTML = 'Lỗi phân tích cảnh: ' + e.message; }
}
