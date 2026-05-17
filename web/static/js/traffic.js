/**
 * traffic.js - Traffic timeline, reporting, best-time, weather
 * Depends on: globals.js
 */
'use strict';

// Weather severity label
function updateWeatherLabel(val) {
  const v = parseFloat(val);
  const lbl = el('tc-w-label');
  if (!lbl) return;
  if (v < 0.2)      lbl.textContent = '\u2600\ufe0f N\u1eafng';
  else if (v < 0.5) lbl.textContent = '\u26c5 M\u00e2y';
  else if (v < 0.8) lbl.textContent = '\U0001f327\ufe0f M\u01b0a nh\u1eb9';
  else              lbl.textContent = '\u26a1 M\u01b0a to';
  lbl.style.color = v < 0.3 ? 'var(--teal)' : v < 0.6 ? 'var(--amber)' : 'var(--red)';
}

async function loadTraffic() {
  try {
    const r = await fetchWithTimeout(API + '/api/traffic/timeline', {}, 8000);
    const d = await r.json();
    el('traffic-html').innerHTML = d.html || '';
  } catch (e) { /* ignore */ }
}
loadTraffic();

async function reportTraffic() {
  if (!curLat) return toast('C\u1ea7n GPS', 'warn');
  const hour = parseInt(el('tc-h').value) || new Date().getHours();
  const congestion = parseFloat(el('tc-c').value);
  const weather = parseFloat(el('tc-w')?.value || 0);
  try {
    const r = await fetchWithTimeout(API + '/api/traffic', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lat: curLat, lon: curLon,
        hour, weekday: new Date().getDay(),
        congestion,
      }),
    }, 8000);
    const d = await r.json();
    if (d.ok) {
      // Also report weather/environment
      await fetchWithTimeout(API + '/api/environment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat: curLat, lon: curLon,
          hour, weekday: new Date().getDay(),
          weather_severity: weather,
          crowd_level: congestion,
        }),
      }, 8000).catch(() => {});
      toast('\U0001f4e1 B\u00e1o c\u00e1o th\u00e0nh c\u00f4ng', 'ok');
      loadTraffic();
    } else {
      toast('\u274c L\u1ed7i', 'warn');
    }
  } catch (e) { toast('\u274c ' + e.message, 'err'); }
}

async function bestTime() {
  const h = parseInt(el('tc-h').value) || new Date().getHours();
  try {
    const r = await fetchWithTimeout(API + '/api/traffic/best-time?hour=' + h, {}, 8000);
    const d = await r.json();
    appendMsg(md(
      '\u23f0 Gi\u1edd t\u1ed1t nh\u1ea5t \u0111\u1ec3 \u0111i l\u00fac **' + h + ':00**:\n' +
      'G\u1ee3i \u00fd: **' + d.recommended_hour + ':00** \u2014 ' + d.status +
      ' (t\u1eafc ' + (d.congestion * 100).toFixed(0) + '%)\n' +
      (d.save_minutes > 0 ? 'Ti\u1ebft ki\u1ec7m ~' + d.save_minutes + ' ph\u00fat so v\u1edbi gi\u1edd d\u1ef1 đ\u1ecbnh.' : '')
    ), 'bot');
  } catch (e) { toast('L\u1ed7i: ' + e.message, 'warn'); }
}
