/**
 * websocket.js — Chat WebSocket + Realtime state WebSocket
 * Depends on: globals.js
 */
'use strict';

// ── Chat WebSocket (/ws/chat) ─────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  try {
    ws = new WebSocket(`${proto}://${location.host}/ws/chat`);
  } catch (e) {
    setTimeout(connectWS, _wsRetryDelay);
    return;
  }
  ws.onopen = () => { _wsRetryDelay = 2500; };
  ws.onmessage = evt => {
    let d;
    try { d = JSON.parse(evt.data); } catch (e) {
      // Malformed JSON — clear stuck spinner
      if (pendingBotMsg) { pendingBotMsg.innerHTML = '❌ Lỗi phản hồi từ server'; pendingBotMsg = null; pendingBotText = ''; }
      return;
    }
    if (!pendingBotMsg) return;
    if (d.type === 'start')  { pendingBotText = ''; return; }
    if (d.type === 'chunk')  {
      pendingBotText += d.text || '';
      pendingBotMsg.innerHTML = md(pendingBotText);
      pendingBotMsg.scrollIntoView({ block: 'end', behavior: 'smooth' });
      return;
    }
    if (d.type === 'end') {
      pendingBotMsg.innerHTML = md(pendingBotText || d.full || '');
      pendingBotMsg = null; pendingBotText = '';
      if (typeof _hideStopBtn === 'function') _hideStopBtn();
      return;
    }
    if (d.type === 'error') {
      pendingBotMsg.innerHTML = '❌ ' + (d.message || 'Lỗi không xác định');
      pendingBotMsg = null; pendingBotText = '';
      if (typeof _hideStopBtn === 'function') _hideStopBtn();
    }
  };
  ws.onclose = () => {
    _wsRetryDelay = Math.min(_wsRetryDelay * 1.5, 30000);
    setTimeout(connectWS, _wsRetryDelay);
  };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}
connectWS();

// ── Realtime WebSocket (/ws/realtime/{sid}) ───────────────────────────────────
let _rtWs = null;
let _rtRetryDelay = 3000;

function connectRealtimeWS() {
  if (!sid) return;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  try {
    _rtWs = new WebSocket(`${proto}://${location.host}/ws/realtime/${sid}`);
  } catch (e) {
    setTimeout(connectRealtimeWS, _rtRetryDelay);
    return;
  }
  _rtWs.onopen = () => { _rtRetryDelay = 3000; };
  _rtWs.onmessage = evt => {
    try {
      const d = JSON.parse(evt.data);
      if (d.type === 'realtime_state' && d.state) {
        _arUpdateFromRealtimeState(d.state);
        const ns = d.state.latest_nav_event;
        if (ns && ns.type && ns.type !== 'none') {
          el('nav-state').textContent = ns.type;
        }
      } else if (d.type === 'alert' && d.alert) {
        // Proactive alert from server — speak + show banner
        if (typeof SpeechModule !== 'undefined') {
          SpeechModule.handleAlert(d.alert);
        }
      }
    } catch (e) { /* ignore parse errors */ }
  };
  _rtWs.onclose = () => {
    _rtRetryDelay = Math.min(_rtRetryDelay * 1.5, 30000);
    setTimeout(connectRealtimeWS, _rtRetryDelay);
  };
  _rtWs.onerror = () => { try { _rtWs.close(); } catch (e) {} };
}
connectRealtimeWS();
