# 10 — Frontend & Mobile Web App

## Overview

The frontend is a single-page application (SPA) served from `web/ui.html`. It is designed for mobile browsers (Android Chrome, iOS Safari) and runs entirely over HTTP/WebSocket on the local network. No build step is required — all JavaScript is vanilla ES6+ loaded as separate modules.

---

## Architecture / Data Flow

```
web/ui.html  (single page, tabs)
    │
    ├── globals.js     — shared state, utilities, boot
    ├── gps.js         — watchPosition + deviceorientation
    ├── websocket.js   — /ws/chat + /ws/realtime/{sid}
    ├── chat.js        — sendMsg, setChatImg, handleKey
    ├── route.js       — findRoute, doSearch, mapFull, openIso
    ├── camera.js      — getUserMedia, captureFrame, detectLandmarks
    ├── ar.js          — _initAR, toggleAR, _arFeedRoute
    ├── floor.js       — initFloorDetection, barometer, step detector
    ├── vio.js         — initVIO, startVIO/stopVIO, _vioOnGpsFix
    ├── speech.js      — SpeechModule (STT + TTS + alerts)
    ├── traffic.js     — traffic timeline, best-time chart
    ├── data.js        — upload, location/POI/edge forms
    └── layout.js      — tab switching, sidebar, responsive layout
```

---

## Key Modules

### `web/static/js/globals.js`

The foundation module. Loaded first. All other modules depend on it.

#### Shared State

| Variable | Type | Description |
|---|---|---|
| `curLat`, `curLon` | `float \| null` | Current GPS coordinates |
| `sid` | `string` | Session ID, unique per page load (`"sid_" + random`) |
| `chatImg` | `File \| null` | Image attached to next chat message |
| `capturedFrames` | `array` | Camera frames captured this session (max 30) |
| `camStream` | `MediaStream \| null` | Active camera stream |
| `ws` | `WebSocket \| null` | Chat WebSocket connection |
| `_gpsAccuracyM` | `float` | Last GPS accuracy in metres |
| `pendingBotMsg` | `HTMLElement \| null` | DOM element for streaming bot response |
| `_wsRetryDelay` | `int` | Current WebSocket reconnect delay (exponential backoff) |

#### `fetchWithTimeout(url, options, timeoutMs=30000)`
Wraps `fetch()` with `AbortController`. Throws `"Request timeout après Xs"` on timeout. Used by all API calls to prevent hanging requests.

#### `toast(msg, type='ok')`
Shows a dismissible notification banner. Duration by type:
- `'err'` → 6 s
- `'warn'` → 4.5 s
- `'ok'` → 3 s

Border colour: green / amber / red.

#### `md(text)`
Minimal markdown renderer. Supports: `**bold**`, `*italic*`, `` `code` ``, `## heading`, `---` (hr), newlines → `<br>`. HTML-escapes `&`, `<`, `>` first.

#### `requestMotionPermission()`
Calls `DeviceMotionEvent.requestPermission()` on iOS 13+ (required for accelerometer access). Called at page load.

#### Boot sequence (`window.addEventListener('load', ...)`)
1. `getGPS()` — start GPS watcher
2. `requestMotionPermission()` — iOS sensor permission
3. `initFloorDetection()` — start barometer + step detector
4. `initVIO()` — start VIO client (if available)
5. `SpeechModule.init()` — initialise speech recognition

---

### `web/static/js/gps.js`

#### `getGPS()`
Calls `navigator.geolocation.watchPosition()` with `enableHighAccuracy: true`. On each fix:
1. Updates `curLat`, `curLon`, `_gpsAccuracyM`.
2. Updates the GPS dot indicator (`el('gps-dot').classList.add('live')`).
3. Calls `ARRenderer.setUserPose()` if AR is active.
4. Calls `_vioOnGpsFix()` to feed VIO re-localization.

**Heading sources** (in priority order):
1. `deviceorientationabsolute` event → `(360 - e.alpha) % 360` (Android)
2. `deviceorientation` with `webkitCompassHeading` (iOS)
3. `deviceorientation` with `e.alpha` (fallback)

**HTTPS warning**: Shows a toast if running on HTTP (not localhost) — GPS may still work on Android Chrome over LAN HTTP.

---

### `web/static/js/camera.js`

#### `startCameraCapture()`
Calls `getUserMedia({ video: { facingMode: 'environment' } })`. Sets `camStream` and attaches to `<video id="cam-preview">`. Shows a warning toast on HTTP (camera may still work on Android).

#### `captureFrame()`
Draws the current video frame to a canvas, converts to JPEG blob (quality 0.92), and:
1. Pushes to `capturedFrames` (max 30).
2. Calls `_autoUploadFrame(blob)` if GPS is available.

#### `_autoUploadFrame(blob)`
Throttled at 1 upload per 5 seconds. Posts to `POST /api/upload/image` with GPS coordinates. Silent on error — frame is still saved locally.

#### `detectLandmarks()`
Posts the current `chatImg` or selected file to `POST /api/experimental/landmarks`. Displays results in the chat panel with confidence percentages and a link to the annotated preview image.

#### `analyzeScene()`
Posts to `POST /api/experimental/scene`. Displays combined YOLO + OCR results with a summary string.

---

### `web/static/js/ar.js`

#### `_initAR()`
1. Checks `navigator.xr?.isSessionSupported('immersive-ar')`.
2. If supported: starts WebXR session, renders route arrows as 3D objects.
3. If not supported: activates compass 2D fallback mode.

#### `toggleAR()`
Starts or stops the AR view. Requests `DeviceMotion` permission on iOS before starting.

#### `_arFeedRoute(routeData)`
Called after a route is found. Passes `ar_path` data to `ARRenderer.setArPath()`.

#### `_arUpdateFromRealtimeState(state)`
Called by the realtime WebSocket handler. Updates the AR renderer with the latest instruction and VIO pose.

#### `stopAR()`
Ends the WebXR session or stops the compass overlay. Releases camera resources.

---

### `web/static/js/websocket.js`

#### `connectWS()` — Chat WebSocket (`/ws/chat`)
- Connects to `ws://host/ws/chat`.
- Handles streaming bot responses: `start` → clear buffer, `chunk` → append to `pendingBotMsg`, `end` → finalise.
- **Exponential backoff**: `_wsRetryDelay` starts at 2.5 s, multiplies by 1.5 on each failure, capped at 30 s.
- Resets delay to 2.5 s on successful connection.

#### `connectRealtimeWS()` — Realtime WebSocket (`/ws/realtime/{sid}`)
- Connects to `ws://host/ws/realtime/{sid}`.
- On `realtime_state`: calls `_arUpdateFromRealtimeState()` and updates nav state indicator.
- On `alert`: calls `SpeechModule.handleAlert()` to speak the alert and show the banner.
- Same exponential backoff as chat WS.

---

### `web/static/js/chat.js`

#### `sendMsg()`
1. Reads message from `el('ui-msg')`.
2. If WebSocket is open: sends `{"type": "chat", "message": ..., "lat": ..., "lon": ...}`.
3. If WebSocket is closed: falls back to `POST /api/chat` (HTTP).
4. Creates `pendingBotMsg` DOM element with typing indicator.

#### `setChatImg(file)`
Sets `chatImg` to the selected file. Shows a thumbnail preview in the chat input area.

#### `handleKey(event)`
Sends message on Enter (without Shift). Shift+Enter inserts a newline.

---

### `web/static/js/route.js`

#### `findRoute()`
Reads origin/destination from form fields, calls `POST /api/route`. On success:
1. Injects `html_card` into the chat panel.
2. Calls `_arFeedRoute(ar_path)` to update AR.
3. Sets session state to NAVIGATING via `POST /api/realtime/vio/relocalize`.

#### `doSearch(query)`
Calls `GET /api/search?q=...` and displays results as clickable location cards.

#### `mapFull()`
Opens the full Folium map in a modal by calling `GET /api/map`.

#### `openIso()`
Opens the isochrone map for the current GPS position via `GET /api/isochrone`.

---

### `web/static/js/floor.js`

#### `initFloorDetection()`
Calls `_initBarometer()` and `_initStepDetector()`.

#### `_initBarometer()`
Tries two APIs in order:
1. `DevicePressureEvent` (experimental, some Android)
2. `Barometer` (Generic Sensor API, Chrome 67+)

On each reading: updates `floorState.pressureHpa`, sets baseline from median of first 3 readings, calls `_fuseFloor()`, updates HUD, calls `_sendFloorUpdate()`.

#### `_initStepDetector()`
Listens to `DeviceMotionEvent`. Maintains `floorState.accelBuf` (last 3 s of samples). Detects steps (Z-axis peaks > 1.5 m/s² above gravity, debounced at 0.25 s). Detects elevator onset (sustained Z offset ≥ 0.3 m/s²).

#### `_fuseFloor()`
Client-side mirror of the server-side `FloorDetector._recompute()`. Blends barometer and step signals. Updates `floorState.floor`, `floorState.confidence`, `floorState.method`.

#### `_sendFloorUpdate()`
Throttled at 800 ms. Posts to `POST /api/realtime/floor` with current pressure and last accelerometer sample. Updates `floorState` from server response (server-side fusion may be more accurate).

**Floor HUD**: Shows `"🏢 Tầng N"` in the top-right corner. Green if confidence ≥ 0.7, amber if ≥ 0.4, grey otherwise. Hidden if `method === 'none'`.

---

### `web/static/js/vio.js`

#### `initVIO()`
Initialises `VIOClient` (from `vio_client.js`). Registers pose callback to update VIO HUD and AR renderer. Listens for `vio-needs-relocalization` custom event.

#### `startVIO()` / `stopVIO()`
Start/stop the VIO IMU loop and optical flow. On start, immediately sends a GPS re-localization fix if available.

#### `_vioOnGpsFix(lat, lon, accuracy)`
Called by `gps.js` on every GPS fix. Forwards to `VIOClient.relocalize()` to reset drift.

#### `_vioTriggerVprFrame()`
Captures a camera frame and posts it to `POST /api/realtime/frame` for VPR re-localization. Called when `needs_relocalization=true` is received from the server.

**VIO HUD**: Shows `"📡 VIO ✓"` (green) or `"📡 1.5m drift"` (amber). Hidden when VIO is stopped.

---

### `web/static/js/speech.js`

#### `SpeechModule` (IIFE module)

**Speech recognition** (two-tier):
1. **Web Speech API** (`SpeechRecognition`): Zero-latency, on-device. Preferred on Android Chrome and desktop.
2. **MediaRecorder + Whisper fallback**: Records up to 8 s of audio, sends to `POST /api/speech/transcribe`. Used on iOS Safari (no Web Speech API) or when Web Speech API fails.

**TTS** (`SpeechSynthesis`):
- Vietnamese voice preferred (`lang.startsWith('vi')`).
- Urgency-based parameters:
  - `high`: rate=1.15, pitch=1.1 — faster and higher for urgent alerts
  - `normal`: rate=1.0, pitch=1.0
  - `low`: rate=0.95, pitch=0.95
- High-urgency speech cancels current low-urgency speech.
- Queue for pending utterances.

**Alert handling** (`handleAlert(alert)`):
1. Speaks `alert.message` via TTS at the alert's urgency level.
2. Shows a dismissible banner in `#alert-banner-container`.
3. Auto-dismisses after 6 s (9 s for high urgency).

**Transcript routing** (`_onTranscript(text, source)`):
1. Appends text as a user message in the chat panel.
2. Sets `el('ui-msg').value = text`.
3. Calls `sendMsg()` after a `setTimeout(0)` to let the DOM commit.
4. Shows a toast: `"🎤 'text'"`.

---

## Mobile Testing Checklist

### Setup
1. Connect phone to the same WiFi as the server.
2. Open `http://192.168.1.217:8000` in Android Chrome or iOS Safari.

### GPS
- Allow location permission when prompted.
- **Healthy**: `gps-dot` turns green, coordinates update in the header.
- **Broken**: Toast "GPS: Bị từ chối quyền GPS" — check browser permissions.

### Camera
1. Go to the **Data** tab.
2. Click **Mở camera**.
3. Allow camera permission.
4. **Healthy**: Video preview shows rear camera.
5. **Broken**: Toast "Không mở được camera" — check HTTPS or permissions.

### AR
1. Find a route first (Navigation tab).
2. Click the **AR** button.
3. Allow DeviceMotion permission (iOS prompt).
4. **Healthy**: Badge shows `"🥽 WebXR"` or `"🧭 Compass"`. Route arrows visible.
5. **Broken**: AR canvas black — check if route was found; check WebXR support.

### Chat
1. Type a message and press Enter.
2. **Healthy**: Bot response streams in chunks, typing indicator shows.
3. **Broken**: No response — check WebSocket connection; try HTTP fallback.

### Voice
1. Click the microphone button `🎤`.
2. Speak a navigation command in Vietnamese.
3. **Healthy**: Transcript appears in chat input, bot responds.
4. **Broken**: Toast "Không có quyền microphone" — check browser permissions.

### Floor detection
1. If device has a barometer: floor HUD appears automatically.
2. Walk up stairs.
3. **Healthy**: Floor number increments, method shows `"barometer+step"`.
4. **Broken**: HUD hidden — device has no barometer; use manual calibration.

### TTS
1. Click the speaker button `🔊` in the header.
2. Find a route — bot should read the first instruction aloud.
3. **Healthy**: Vietnamese voice speaks the instruction.
4. **Broken**: No sound — check device volume; TTS may not support Vietnamese.

---

## Healthy vs Broken Indicators

| Feature | Healthy | Broken |
|---|---|---|
| GPS | Green dot, coordinates updating | Grey dot, "GPS: Bị từ chối" toast |
| Camera | Video preview active | "Không mở được camera" toast |
| AR | Mode badge visible, arrows rendered | Black canvas, no badge |
| Chat WS | Streaming chunks arrive | Spinner stuck, no response |
| Floor HUD | `"🏢 Tầng N"` visible, green | HUD hidden or grey |
| VIO HUD | `"📡 VIO ✓"` green | `"📡 Xm drift"` amber |
| Speech | Transcript in chat | "Không có quyền microphone" |
| TTS | Vietnamese voice speaks | Silent (check volume/permissions) |

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| GPS dot stays grey | Permission denied or HTTPS required | Allow location in browser settings |
| Camera "NotAllowedError" | Permission denied | Settings → Browser → Camera → Allow |
| AR badge missing | Route not found yet | Find a route first, then open AR |
| WebSocket reconnecting | Server restarted | Automatic — wait for reconnect |
| Floor HUD never appears | No barometer on device | Use `POST /api/realtime/floor/calibrate` manually |
| TTS speaks English | No Vietnamese voice installed | Install Vietnamese TTS on Android: Settings → Accessibility → TTS |
| Whisper fallback slow | Server CPU-only | Set `WHISPER_DEVICE=cuda` or use `WHISPER_MODEL=tiny` |
| `capturedFrames` full | 30-frame limit reached | Click "Xóa frames" button |

---

## Performance Notes

- **GPS update rate**: `watchPosition` with `enableHighAccuracy: true` — typically 1–5 Hz on Android, 1 Hz on iOS.
- **Camera frame capture**: Canvas `toBlob()` at JPEG quality 0.92 — ~50–200 ms for 1280×720.
- **Auto-upload throttle**: 1 per 5 s — prevents flooding the server.
- **WebSocket reconnect**: Exponential backoff from 2.5 s to 30 s max.
- **Floor update throttle**: 800 ms — balances responsiveness with server load.
- **Speech recognition**: Web Speech API is near-instant. Whisper fallback adds 1–3 s latency.
- The app works on HTTP over LAN (Android Chrome allows camera/GPS on LAN). iOS Safari requires HTTPS for camera and DeviceMotion.
