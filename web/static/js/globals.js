/**
 * globals.js — Shared state, utilities, and boot sequence
 * Loaded first. All other modules depend on these.
 */
'use strict';

const API = '';

// ── Shared state ──────────────────────────────────────────────────────────────
let ws            = null;
let curLat        = null;
let curLon        = null;
let chatImg       = null;
let mapOpen       = false;
let sideOpen      = true;
let _gpsAccuracyM = 5.0;
let pendingBotMsg  = null;
let pendingBotText = '';
let camStream     = null;
let capturedFrames = [];
let streetState   = { sequenceId: null, nodeId: null };
let _wsRetryDelay = 2500;

// Session ID — unique per page load
const sid = 'sid_' + Math.random().toString(36).slice(2, 10);

// ── DOM helper ────────────────────────────────────────────────────────────────
function el(id) { return document.getElementById(id); }

// ── Markdown renderer (minimal) ───────────────────────────────────────────────
function md(t) {
  if (typeof t !== 'string') return '';
  return t
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/[*][*]([^*]+)[*][*]/g, '<strong>$1</strong>')
    .replace(/(?<![*])[*](?![*])([^*]+)(?<![*])[*](?![*])/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^---$/gm, '<hr>')
    .replace(/\n/g, '<br>');
}

// ── Fetch with timeout ────────────────────────────────────────────────────────
async function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timeoutId);
    return response;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error(`Request timeout sau ${timeoutMs / 1000}s`);
    }
    throw err;
  }
}

// ── Chat message helpers ──────────────────────────────────────────────────────
function appendMsg(html, role) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = html;
  const msgs = el('msgs');
  msgs.appendChild(d);
  d.scrollIntoView({ block: 'end', behavior: 'smooth' });
  return d;
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, type = 'ok') {
  const t = el('toast');
  t.textContent = msg;
  t.style.borderColor = type === 'ok'
    ? 'var(--green)'
    : type === 'warn'
    ? 'var(--amber)'
    : 'var(--red)';
  t.classList.add('show');
  // Lỗi (type='err') hiện 6s, cảnh báo 4.5s, thông báo thường 3s
  const duration = type === 'err' ? 6000 : type === 'warn' ? 4500 : 3000;
  clearTimeout(t._hideTimer);
  t._hideTimer = setTimeout(() => t.classList.remove('show'), duration);
}

// ── iOS DeviceMotion permission ───────────────────────────────────────────────
async function requestMotionPermission() {
  if (typeof DeviceMotionEvent !== 'undefined' &&
      typeof DeviceMotionEvent.requestPermission === 'function') {
    try {
      const result = await DeviceMotionEvent.requestPermission();
      if (result !== 'granted') toast('Không có quyền cảm biến chuyển động', 'warn');
    } catch (e) { /* ignore */ }
  }
}

// ── Page load boot ────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  getGPS();
  requestMotionPermission();
  initFloorDetection();
  if (typeof VIOClient !== 'undefined') initVIO();
  if (typeof SpeechModule !== 'undefined') SpeechModule.init();
  if (typeof AREnhanced !== 'undefined') AREnhanced.init();
});

// ── TTS toggle (header button) ────────────────────────────────────────────────
let _ttsOn = true;
function _toggleTts() {
  _ttsOn = !_ttsOn;
  if (typeof SpeechModule !== 'undefined') SpeechModule.setTtsEnabled(_ttsOn);
  const btn = el('tts-btn');
  if (btn) {
    btn.textContent = _ttsOn ? '🔊' : '🔇';
    btn.title = _ttsOn ? 'Tắt đọc to' : 'Bật đọc to';
    btn.style.opacity = _ttsOn ? '1' : '0.45';
  }
  toast(_ttsOn ? '🔊 Đọc to: BẬT' : '🔇 Đọc to: TẮT', 'ok');
}
