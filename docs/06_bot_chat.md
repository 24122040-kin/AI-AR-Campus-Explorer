# 06 — Navigation Bot & Chat

## Overview

The bot module provides the conversational interface for LocalNavBot. It wraps an LLM (Anthropic Claude, OpenAI, or Ollama) with a Vietnamese navigation persona, structured intent parsing, route finding, VPR-based place identification, and streaming responses. A session state machine tracks navigation progress and triggers live re-routing when the user deviates from the planned route.

---

## Architecture / Data Flow

```
User message (text or image)
        │
        ▼
NavBot.ask()
        │
        ├── 1. Build context
        │       ├── db.nearby_locations(user_lat, user_lon)
        │       ├── db.nearby_pois(user_lat, user_lon)
        │       └── GPS + time string
        │
        ├── 2. LLMClient.parse_intent()  → NavigationIntent
        │       └── JSON extraction via LLM
        │
        ├── 3. Intent dispatch
        │       ├── "route"          → _handle_route()
        │       │       ├── router.resolve_location(origin/dest)
        │       │       ├── router.find_route()
        │       │       └── _attach_images_to_route()
        │       ├── "find_poi"       → db.search_pois() + db.search_locations()
        │       ├── "identify_place" → VPREngine.query()
        │       └── "chat"           → pass-through
        │
        ├── 4. Build visual context (if image provided)
        │       ├── LandmarkDetector.detect()
        │       ├── OCRReader.detect()
        │       └── VPREngine.query()
        │
        ├── 5. Build LLM messages
        │       └── [system + context] + conversation history
        │
        └── 6. LLMClient.chat(stream=True/False)
                        │
                        ▼
                  Response text / AsyncIterator[str]

─────────────────────────────────────────────────────────────

GPS stream (WS /ws/chat)
        │
        ▼
SessionManager.process_gps_update()
        │
        ├── NavSession.update_gps()
        │       ├── snap_point_to_polyline()
        │       ├── step_advance (d < 25 m)
        │       ├── arrived (d_dest < 30 m)
        │       └── off_route (d > 50 m × 3 times)
        │
        └── off_route + pending_reroute?
                └── router.find_route() → new Route
                        └── return "rerouted" event
```

---

## Key Classes and Functions

### `bot/nav_bot.py`

#### `SYSTEM_PROMPT`
Vietnamese navigation persona injected as the system message for every LLM call. Key capabilities declared:
1. Optimal routing based on departure time and congestion
2. Place identification from images
3. Step-by-step directions with illustrative photos
4. Local POI suggestions (restaurants, shortcuts, landmarks)
5. Real-time congestion updates

Response style: concise Vietnamese, landmark-based directions ("đến ngã tư có cây xăng Shell thì rẽ phải"), peak-hour warnings, local shortcut suggestions.

---

#### `LLMClient`

```python
class LLMClient:
    def __init__()                    # reads settings.llm_provider, llm_model, llm_api_key
    async def chat(messages, stream) -> str | AsyncIterator[str]
    async def parse_intent(user_message) -> NavigationIntent
```

**Provider support**:
- `anthropic`: Uses `anthropic.AsyncAnthropic`. System message is extracted from messages and passed as the `system` parameter. Streaming via `client.messages.stream()`.
- `openai` / `ollama`: Uses `openai.AsyncOpenAI` with custom `base_url` for Ollama. Streaming via `stream=True` in `chat.completions.create()`.

**`parse_intent`**: Sends a structured JSON extraction prompt to the LLM. Strips markdown code fences from the response before parsing. Falls back to `intent_type="chat"` on any error.

**`_fallback_intent`**: Rule-based fallback when LLM intent parsing returns `"chat"`:
- Image + keywords like "đây là đâu", "nhận diện" → `"identify_place"`
- Keywords like "đi đến", "tìm đường", "route" → `"route"`

---

#### `NavigationIntent` (dataclass)

```python
@dataclass
class NavigationIntent:
    intent_type: str    # "route" | "identify_place" | "find_poi" | "add_info" | "chat"
    origin: str | None
    destination: str | None
    depart_time_str: str | None   # "08:30" format
    poi_query: str | None
    raw_query: str
    has_image: bool
    lat: float | None
    lon: float | None
```

---

#### `NavBot`

```python
class NavBot:
    def __init__(router: NavRouter, vpr_engine=None)
    async def ask(user_message, image_path, user_lat, user_lon, stream) -> str | AsyncIterator[str]
    async def stream(user_message, image_path, user_lat, user_lon) -> AsyncIterator[str]
    async def _attach_images_to_route(route: Route) -> None
    async def _handle_route(intent, user_lat, user_lon) -> str
    async def _handle_identify(image_path, user_lat, user_lon) -> str
    async def _build_visual_context(image_path, user_lat, user_lon) -> str
    async def rebuild_vpr_index() -> None
    async def add_location_with_images(name, lat, lon, image_paths, ...) -> int
    def reset_history() -> None
```

**`_attach_images_to_route`**: For each step, queries `db.nearby_locations(step.lat, step.lon, radius_deg=0.0005)` (~55 m radius) and attaches up to 2 image paths to `step.image_paths`.

**`_build_visual_context`**: Runs YOLO, OCR, and VPR on the provided image and returns a Vietnamese context string like:
```
Landmark thấy trong ảnh: car, traffic light, building
Text đọc được: Chợ Dĩ An | Đường Lê Lợi
VPR gợi ý: Ngã tư Bình Dương (81%)
```

**`rebuild_vpr_index`**: Fetches all images from DB (joined with locations), creates `ImageMeta` objects, and calls `VPREngine.index_all_images()`.

**Conversation history**: Stored in `_history` as a list of `{"role": "user"/"assistant", "content": ...}`. Truncated to `CONV_MAX_TURNS × 2 = 40` messages.

---

### `bot/session_manager.py`

#### `NavState` (Enum)
```
IDLE → NAVIGATING → REROUTING → ARRIVED
                 ↑                  │
                 └──────────────────┘ (re-route)
```

#### `GPSFix` (dataclass)
```python
@dataclass
class GPSFix:
    lat: float
    lon: float
    accuracy_m: float = 10.0
    timestamp: datetime
    speed_kmh: float = 0.0
    bearing: float = 0.0
```

#### `NavSession`

```python
class NavSession:
    OFF_ROUTE_M = 50      # metres before counting as off-route
    ARRIVE_M = 30         # metres from destination to trigger arrival
    SNAP_TRUST_M = 120.0  # max residual to trust snap position

    async def update_gps(fix: GPSFix) -> dict
    def add_message(role, content) -> None
    def recent_history() -> list[dict]
    def smoothed_speed() -> float
    def eta() -> datetime | None
```

**`update_gps` logic**:
1. Snap GPS to polyline: `snap_point_to_polyline(fix.lat, fix.lon, route.geometry)`.
2. Use snapped position if residual < 120 m, otherwise use raw GPS.
3. Check arrival: `haversine(fix, destination) < 30 m` → `ARRIVED`.
4. Check step advance: `haversine(use_pos, next_step) < 25 m` → increment `current_step_idx`.
5. Check off-route: `distance_to_polyline > 50 m` → increment `deviation_count`. After 3 consecutive deviations → `REROUTING` + `pending_reroute: True`.
6. If not off-route: decrement `deviation_count` (min 0).

**`smoothed_speed`**: Average of non-zero `speed_kmh` values in the last 20 GPS fixes.

**`eta`**: Sums `duration_s` of remaining steps, adjusts by speed ratio (clamped to [0.5, 2.0]×).

---

#### `SessionManager`

```python
class SessionManager:
    def get_or_create(session_id) -> NavSession
    async def process_gps_update(session_id, fix, router) -> dict
    async def start() -> None   # starts cleanup loop
    def stats() -> dict
```

**`process_gps_update`**: Calls `NavSession.update_gps()`. If result is `off_route` with `pending_reroute=True` and a router + destination are available, calls `router.find_route()` from current position to destination. On success, returns `type: "rerouted"` with the new route summary.

**Session cleanup**: Background task runs every 5 minutes, deletes sessions inactive for 2 hours.

---

### `bot/realtime_navigator.py`

#### `RealtimeNavigator.build_instruction(scene_state) -> dict`

Fuses route progress + visual context into a single instruction dict:

```python
{
  "instruction": "Rẽ phải sau khoảng 45 m.",
  "short_instruction": "Rẽ phải sau khoảng 45 m.",
  "reason": "Landmark thấy được: car, traffic light. VPR gợi ý khu vực Ngã tư Bình Dương.",
  "urgency": "high"  # "high" if distance ≤ 50 m or off_route
}
```

Priority:
1. Off-route → "Bạn đang lệch khỏi tuyến, hệ thống sẽ tìm lại đường."
2. Next maneuver with distance → formatted instruction
3. No route → "Tiếp tục đi thẳng theo tuyến hiện tại."

---

### `web/routes/chat.py`

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Single-turn chat (JSON body) |
| `/api/chat/image` | POST | Chat with image (multipart) |
| `/api/chat/stream` | POST | SSE streaming response |
| `/ws/chat` | WS | Bidirectional chat + GPS nav |

**`POST /api/chat`** request:
```json
{
  "message": "Tìm đường đến chợ Dĩ An",
  "lat": 10.9085,
  "lon": 106.760,
  "session_id": "user_abc"
}
```

**`POST /api/chat/stream`** response (SSE):
```
data: {"chunk": "Tuyến đường từ vị trí của bạn"}
data: {"chunk": " đến chợ Dĩ An:"}
data: {"chunk": "\n\n## 2.3 km · ~8 phút"}
data: [DONE]
```

**WebSocket `/ws/chat`** message types:
- `{"type": "chat", "message": "...", "lat": ..., "lon": ...}` → streaming response
- `{"type": "gps", "lat": ..., "lon": ..., "accuracy": ..., "speed": ..., "bearing": ...}` → nav event
- `{"type": "start_nav"}` → set session state to NAVIGATING

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` \| `ollama` |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Model name |
| `LLM_API_KEY` | `""` | API key (Anthropic or OpenAI) |
| `LLM_BASE_URL` | `""` | Base URL for Ollama: `http://localhost:11434/v1` |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per response |
| `LLM_TEMPERATURE` | `0.2` | Response temperature |
| `LLM_TIMEOUT_SECONDS` | `45` | Request timeout |
| `CHAT_MAX_CHARS` | `4000` | Max message length |
| `OLLAMA_ENABLED` | `false` | Enable Ollama provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:3b-instruct` | Ollama model name |

---

## How to Test

### Basic chat

```bash
curl -X POST http://192.168.1.217:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Xin chào", "session_id": "test"}'
```

### Route request with GPS

```bash
curl -X POST http://192.168.1.217:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tìm đường đến chợ Dĩ An",
    "lat": 10.9085,
    "lon": 106.760,
    "session_id": "test"
  }'
```

### Chat with image

```bash
curl -X POST http://192.168.1.217:8000/api/chat/image \
  -F "file=@photo.jpg" \
  -F "message=Đây là đâu?" \
  -F "lat=10.9085" \
  -F "lon=106.760"
```

### Streaming chat (SSE)

```bash
curl -N -X POST http://192.168.1.217:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Đường nào ít tắc nhất lúc 17 giờ?", "session_id": "test"}'
```

### WebSocket chat

```javascript
const ws = new WebSocket('ws://192.168.1.217:8000/ws/chat');
ws.onopen = () => ws.send(JSON.stringify({
  type: 'chat',
  message: 'Đường nào ít tắc nhất?'
}));
ws.onmessage = e => {
  const d = JSON.parse(e.data);
  if (d.type === 'chunk') process.stdout.write(d.text);
  if (d.type === 'end') console.log('\n[done]');
};
```

---

## Healthy Output Examples

**Chat response:**
```json
{
  "response": "Chào bạn! Tôi là LocalNavBot, trợ lý điều hướng địa phương...",
  "ok": true
}
```

**Route response (truncated):**
```json
{
  "response": "## Tuyến đường — 2.3 km · ~8 phút\nXuất phát lúc 08:30\n---\n🚀 **Bước 1**: Xuất phát từ Đường Lê Lợi...",
  "ok": true
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `503 "Router not ready"` | NavRouter not initialised | Wait for startup |
| `504 "AI response timed out"` | LLM API slow or unreachable | Check API key; increase `LLM_TIMEOUT_SECONDS`; use Ollama locally |
| `500 "LLM key error"` | Invalid or missing API key | Set `LLM_API_KEY` in `.env` |
| Empty response | LLM returned empty string | Check model name; verify API quota |
| Intent always `"chat"` | LLM not returning valid JSON | Check `LLM_TEMPERATURE` (should be low, 0.1–0.3); verify model supports JSON |
| Route not found in chat | Destination not geocodable | Add location to DB via `POST /api/location`; enable `ALLOW_REMOTE_GEOCODING=true` |
| Session history not growing | Wrong `session_id` | Use consistent session ID across requests |

---

## Performance Notes

- **LLM latency**: Anthropic Claude Sonnet: ~1–3 s first token, ~5–15 s full response. Ollama local: ~0.5–2 s depending on model and GPU.
- **Intent parsing**: One additional LLM call per message (~0.5–1 s). Can be disabled by using rule-based fallback only.
- **Route attachment**: `_attach_images_to_route()` makes one DB query per step. For a 10-step route: ~10 ms.
- **Streaming**: SSE chunks arrive as the LLM generates them. First chunk typically arrives in 0.5–2 s.
- **Session memory**: Each session stores up to 40 messages (20 turns). Older messages are dropped automatically.
- For production use, consider a dedicated LLM proxy with caching for common queries (e.g., "đường nào ít tắc?").
