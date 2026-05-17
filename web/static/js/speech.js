/**
 * speech.js — Speech-to-Intent + Text-to-Speech (TTS) + Proactive Alerts
 *
 * Speech Recognition:
 *   1. Web Speech API (SpeechRecognition) — primary, zero-latency, on-device
 *   2. Fallback: MediaRecorder → audio blob → POST /api/speech/transcribe (Whisper)
 *
 * TTS (Text-to-Speech):
 *   - SpeechSynthesis API — reads navigation instructions and alerts aloud
 *   - Vietnamese voice preferred; falls back to any available voice
 *   - Urgency-based rate/pitch: high urgency = faster + higher pitch
 *
 * Alert handling:
 *   - Receives {"type":"alert","alert":{...}} from /ws/realtime WebSocket
 *   - Speaks the alert message via TTS
 *   - Shows a dismissible banner in the UI
 *
 * Public API:
 *   SpeechModule.init()
 *   SpeechModule.startListening()   — start mic
 *   SpeechModule.stopListening()    — stop mic
 *   SpeechModule.speak(text, urgency)  — TTS
 *   SpeechModule.handleAlert(alert)    — process incoming alert
 *   SpeechModule.setEnabled(bool)      — master on/off
 *   SpeechModule.isListening()         → bool
 */
'use strict';

const SpeechModule = (() => {

  // ── Config ─────────────────────────────────────────────────────────────────
  const CFG = {
    LANG:              'vi-VN',
    WHISPER_PATH:      '/api/speech/transcribe',
    // Minimum confidence for Web Speech API results
    MIN_CONFIDENCE:    0.3,
    // Max recording duration for Whisper fallback (ms)
    MAX_RECORD_MS:     8000,
    // TTS rate/pitch by urgency
    TTS: {
      high:   { rate: 1.15, pitch: 1.1, volume: 1.0 },
      normal: { rate: 1.0,  pitch: 1.0, volume: 0.95 },
      low:    { rate: 0.95, pitch: 0.95, volume: 0.85 },
    },
    // Alert banner auto-dismiss (ms)
    ALERT_DISMISS_MS: 6000,
  };

  // ── State ──────────────────────────────────────────────────────────────────
  let _enabled       = true;
  let _ttsEnabled    = true;
  let _listening     = false;
  let _recognition   = null;   // SpeechRecognition instance
  let _mediaRecorder = null;   // Whisper fallback
  let _audioChunks   = [];
  let _recordTimer   = null;
  let _viVoice       = null;   // cached Vietnamese TTS voice
  let _speakQueue    = [];     // pending TTS utterances
  let _speaking      = false;

  // ── TTS voice selection ────────────────────────────────────────────────────

  function _pickVoice() {
    if (_viVoice) return _viVoice;
    const voices = window.speechSynthesis?.getVoices() || [];
    // Prefer Vietnamese voices
    _viVoice = voices.find(v => v.lang.startsWith('vi'))
            || voices.find(v => v.lang.startsWith('vi-VN'))
            || voices.find(v => v.default)
            || voices[0]
            || null;
    return _viVoice;
  }

  // Voices load asynchronously on some browsers
  if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = () => { _viVoice = null; _pickVoice(); };
  }

  // ── TTS ────────────────────────────────────────────────────────────────────

  function speak(text, urgency = 'normal') {
    if (!_ttsEnabled || !text || !window.speechSynthesis) return;
    // Cancel any current low-urgency speech if high urgency arrives
    if (urgency === 'high' && _speaking) {
      window.speechSynthesis.cancel();
      _speakQueue = _speakQueue.filter(u => u._urgency === 'high');
      _speaking = false;
    }
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang   = CFG.LANG;
    utt.voice  = _pickVoice();
    const cfg  = CFG.TTS[urgency] || CFG.TTS.normal;
    utt.rate   = cfg.rate;
    utt.pitch  = cfg.pitch;
    utt.volume = cfg.volume;
    utt._urgency = urgency;
    utt.onstart = () => { _speaking = true; };
    utt.onend   = () => {
      _speaking = false;
      if (_speakQueue.length) {
        const next = _speakQueue.shift();
        window.speechSynthesis.speak(next);
      }
    };
    utt.onerror = () => { _speaking = false; };

    if (_speaking && urgency !== 'high') {
      _speakQueue.push(utt);
    } else {
      _speaking = true;
      window.speechSynthesis.speak(utt);
    }
  }

  function stopSpeaking() {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    _speakQueue = [];
    _speaking = false;
  }

  // ── Alert handler ──────────────────────────────────────────────────────────

  function handleAlert(alert) {
    if (!_enabled || !alert) return;
    const msg     = alert.message || '';
    const urgency = alert.urgency || 'normal';

    // Speak the alert
    speak(msg, urgency);

    // Show banner in UI
    _showAlertBanner(alert);
  }

  function _showAlertBanner(alert) {
    const container = el('alert-banner-container');
    if (!container) return;

    const banner = document.createElement('div');
    banner.className = 'alert-banner alert-' + (alert.urgency || 'normal');

    const icon = { high: '⚠️', normal: 'ℹ️', low: '💡' }[alert.urgency] || 'ℹ️';
    const distStr = alert.distance_m != null ? ` (${Math.round(alert.distance_m)}m)` : '';
    banner.innerHTML = `
      <span class="alert-icon">${icon}</span>
      <span class="alert-msg">${alert.message}${distStr}</span>
      <button class="alert-dismiss" onclick="this.parentElement.remove()">✕</button>
    `;

    container.prepend(banner);

    // Auto-dismiss
    const dismissMs = alert.urgency === 'high' ? CFG.ALERT_DISMISS_MS * 1.5 : CFG.ALERT_DISMISS_MS;
    setTimeout(() => banner.remove(), dismissMs);
  }

  // ── Web Speech API ─────────────────────────────────────────────────────────

  function _initWebSpeech() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return null;

    const rec = new SR();
    rec.lang          = CFG.LANG;
    rec.continuous    = false;
    rec.interimResults = false;
    rec.maxAlternatives = 1;

    rec.onresult = evt => {
      const result = evt.results[0];
      if (!result) return;
      const alt = result[0];
      const text = (alt.transcript || '').trim();
      const conf = alt.confidence || 1.0;
      if (text && conf >= CFG.MIN_CONFIDENCE) {
        _onTranscript(text, 'webspeech');
      }
    };

    rec.onerror = evt => {
      if (evt.error === 'not-allowed') {
        toast('Không có quyền microphone', 'warn');
      } else if (evt.error === 'network' || evt.error === 'service-not-allowed' || evt.error === 'audio-capture') {
        try { rec.stop(); } catch (e) {}
        _startWhisperRecording();
        return;
      }
      _setListeningUI(false);
      _listening = false;
    };

    rec.onend = () => {
      _setListeningUI(false);
      _listening = false;
    };

    return rec;
  }

  // ── Whisper fallback (MediaRecorder) ──────────────────────────────────────

  async function _startWhisperRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      toast('Microphone không được hỗ trợ', 'warn'); return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      _audioChunks = [];
      _mediaRecorder = new MediaRecorder(stream, { mimeType: _bestMimeType() });
      _mediaRecorder.ondataavailable = e => { if (e.data.size > 0) _audioChunks.push(e.data); };
      _mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(_audioChunks, { type: _mediaRecorder.mimeType });
        await _sendToWhisper(blob, _mediaRecorder.mimeType);
        _setListeningUI(false);
        _listening = false;
      };
      _mediaRecorder.start();
      // Auto-stop after max duration
      _recordTimer = setTimeout(() => stopListening(), CFG.MAX_RECORD_MS);
    } catch (err) {
      toast('Không mở được microphone: ' + err.message, 'warn');
      _setListeningUI(false);
      _listening = false;
    }
  }

  async function _sendToWhisper(blob, mimeType) {
    const fd = new FormData();
    fd.append('file', new File([blob], 'audio' + _mimeToExt(mimeType), { type: mimeType }));
    fd.append('language', 'vi');
    fd.append('session_id', typeof sid !== 'undefined' ? sid : '');
    try {
      const r = await fetchWithTimeout(CFG.WHISPER_PATH, { method: 'POST', body: fd }, 20000);
      if (!r.ok) { toast('Whisper lỗi: ' + r.status, 'warn'); return; }
      const d = await r.json();
      if (d.text) _onTranscript(d.text, 'whisper');
    } catch (e) {
      toast('Không kết nối được server nhận dạng giọng nói', 'warn');
    }
  }

  function _bestMimeType() {
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
    return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
  }

  function _mimeToExt(mime) {
    if (mime.includes('webm')) return '.webm';
    if (mime.includes('ogg'))  return '.ogg';
    if (mime.includes('mp4'))  return '.mp4';
    return '.webm';
  }

  // ── Transcript handler → intent routing ───────────────────────────────────

  let _lastTranscript = '';
  let _transcriptTimer = null;

  function _onTranscript(text, source) {
    if (!text) return;
    // Debounce: ignore duplicate transcript within 300ms
    if (text === _lastTranscript) return;
    _lastTranscript = text;
    clearTimeout(_transcriptTimer);
    _transcriptTimer = setTimeout(() => { _lastTranscript = ''; }, 300);

    // Show in chat as user message
    if (typeof appendMsg === 'function') appendMsg(md(text), 'user');
    // Feed into chat pipeline (same as typing)
    // Use setTimeout(0) to let the DOM commit inp.value before sendMsg() reads it
    if (typeof el === 'function') {
      const inp = el('ui-msg');
      if (inp) {
        inp.value = text;
        setTimeout(() => { if (typeof sendMsg === 'function') sendMsg(); }, 0);
      }
    }
    // Visual feedback
    toast(`🎤 "${text}"`, 'ok');
  }

  // ── UI helpers ─────────────────────────────────────────────────────────────

  function _setListeningUI(active) {
    const btn = el('mic-btn');
    if (!btn) return;
    btn.classList.toggle('mic-active', active);
    btn.title = active ? 'Đang nghe… (nhấn để dừng)' : 'Nhấn để nói';
    btn.textContent = active ? '🔴' : '🎤';
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  function init() {
    // Try Web Speech API first
    _recognition = _initWebSpeech();
    // Voices may not be loaded yet — trigger load
    if (window.speechSynthesis) window.speechSynthesis.getVoices();
  }

  function startListening() {
    if (!_enabled || _listening) return;

    // Check HTTPS requirement
    if (location.protocol !== 'https:' &&
        location.hostname !== 'localhost' &&
        location.hostname !== '127.0.0.1') {
      toast('Mic cần HTTPS — dùng ngrok để bật mic trên điện thoại', 'warn');
    }

    _listening = true;
    _setListeningUI(true);

    if (_recognition) {
      try {
        _recognition.start();
        return;
      } catch (e) {
        // InvalidStateError = already started, or not-allowed
        if (e.name === 'InvalidStateError') {
          // Re-create recognition instance
          _recognition = _initWebSpeech();
          if (_recognition) {
            try { _recognition.start(); return; } catch (e2) { /* fall through */ }
          }
        }
        console.warn('[Speech] Web Speech failed:', e.message);
      }
    }
    // Whisper fallback
    _startWhisperRecording();
  }

  function stopListening() {
    if (!_listening) return;
    if (_recognition) {
      try { _recognition.stop(); } catch (e) { /* ignore */ }
    }
    if (_mediaRecorder && _mediaRecorder.state === 'recording') {
      clearTimeout(_recordTimer);
      _mediaRecorder.stop();
    }
    _setListeningUI(false);
    _listening = false;
  }

  function toggleListening() {
    _listening ? stopListening() : startListening();
  }

  function setEnabled(v) { _enabled = v; }
  function setTtsEnabled(v) { _ttsEnabled = v; }
  function isListening() { return _listening; }

  return { init, startListening, stopListening, toggleListening,
           speak, stopSpeaking, handleAlert, setEnabled, setTtsEnabled, isListening };

})();

window.SpeechModule = SpeechModule;
