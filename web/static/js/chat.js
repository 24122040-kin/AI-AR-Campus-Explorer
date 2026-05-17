/**
 * chat.js — Chat UI: sendMsg, image chat, input helpers
 * Depends on: globals.js, websocket.js
 */
'use strict';

// ── Stop bot ──────────────────────────────────────────────────────────────────
function stopBot() {
  // 1. Cancel any in-progress WebSocket stream
  if (ws && ws.readyState === 1) {
    try { ws.send(JSON.stringify({ type: 'stop' })); } catch (e) {}
  }
  // 2. Abort any pending HTTP fetch (via AbortController in fetchWithTimeout)
  // 3. Clear the pending spinner
  if (pendingBotMsg) {
    pendingBotMsg.innerHTML = '<span style="color:var(--text3);font-size:12px">⏹ Đã dừng</span>';
    pendingBotMsg = null;
    pendingBotText = '';
  }
  // 4. Stop TTS
  if (typeof SpeechModule !== 'undefined') SpeechModule.stopSpeaking();
  // 5. Hide stop button
  const stopBtn = el('stop-btn');
  if (stopBtn) stopBtn.style.display = 'none';
}

function _showStopBtn() {
  const stopBtn = el('stop-btn');
  if (stopBtn) stopBtn.style.display = '';
}
function _hideStopBtn() {
  const stopBtn = el('stop-btn');
  if (stopBtn) stopBtn.style.display = 'none';
}

async function sendMsg() {
  const inp = el('ui-msg');
  const txt = inp.value.trim();
  if (!txt && !chatImg) return;
  inp.value = ''; inp.style.height = 'auto';

  // ── Image chat ──────────────────────────────────────────────────────────────
  if (chatImg) {
    appendMsg(`📷 <em>${chatImg.name}</em>${txt ? '<br>' + md(txt) : ''}`, 'user');
    const fd = new FormData();
    fd.append('file', chatImg);
    fd.append('message', txt || 'Day la dau?');
    fd.append('session_id', sid);
    if (curLat) { fd.append('lat', curLat); fd.append('lon', curLon); }
    const bot = appendMsg('<span class="typing-dots">Nhận dạng</span>', 'bot');
    try {
      const r = await fetchWithTimeout(API + '/api/chat/image', { method: 'POST', body: fd }, 45000);
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || d.error || 'Khong phan tich duoc anh');
      if (d.response) {
        bot.innerHTML = md(d.response);
        chatImg = null; el('img-chat').value = ''; return;
      }
      if (d.matches?.length) {
        const b = d.matches[0];
        bot.innerHTML = md(
          `📍 **${b.location_name}**\n` +
          `Tọa độ: \`${b.lat.toFixed(5)}, ${b.lon.toFixed(5)}\`\n` +
          `Độ khớp: ${(b.score * 100).toFixed(1)}%` +
          (b.caption ? `\n${b.caption}` : '')
        );
      } else {
        bot.innerHTML = 'Không nhận ra địa điểm này.';
      }
    } catch (e) { bot.innerHTML = 'Lỗi: ' + e.message; }
    chatImg = null; el('img-chat').value = ''; return;
  }

  // ── Text chat ───────────────────────────────────────────────────────────────
  appendMsg(md(txt), 'user');
  const bot = appendMsg('<span class="typing-dots">Đang suy nghĩ</span>', 'bot');
  _showStopBtn();

  const _httpFallback = async () => {
    const r = await fetchWithTimeout(API + '/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: txt, lat: curLat, lon: curLon, session_id: sid }),
    }, 45000);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || d.error || 'Loi API');
    bot.innerHTML = md(d.response || '');
    _hideStopBtn();
  };

  if (ws && ws.readyState === 1) {
    pendingBotMsg = bot; pendingBotText = '';
    try {
      ws.send(JSON.stringify({ type: 'chat', message: txt, lat: curLat, lon: curLon, session_id: sid }));
    } catch (wsErr) {
      pendingBotMsg = null;
      _hideStopBtn();
      try { await _httpFallback(); } catch (e) { bot.innerHTML = md(`❌ Loi chat: ${e.message}`); _hideStopBtn(); }
    }
  } else {
    try { await _httpFallback(); } catch (e) { bot.innerHTML = md(`❌ Loi chat: ${e.message}`); _hideStopBtn(); }
  }
}

function setChatImg(inp) {
  if (inp.files[0]) { chatImg = inp.files[0]; toast('Ảnh: ' + chatImg.name, 'ok'); }
}
function handleKey(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); } }
let _autoHTimer = null;
function autoH(el) {
  if (!el) return;
  clearTimeout(_autoHTimer);
  _autoHTimer = setTimeout(() => {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }, 32);
}
