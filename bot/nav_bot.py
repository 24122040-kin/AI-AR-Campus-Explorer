"""
bot/nav_bot.py — Navigation assistant bot
Handles natural language → intent parsing → routing → illustrated response
"""
from __future__ import annotations
import json
import base64
import asyncio
from pathlib import Path
from datetime import datetime
from typing import AsyncIterator, Optional
from dataclasses import dataclass, field

from loguru import logger
from PIL import Image

from config.settings import settings
from core.database import db
from core.landmark_detector import LandmarkDetector
from core.ocr_reader import OCRReader
from routing.router import NavRouter, Route, RouteStep


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là LocalNavBot — trợ lý điều hướng thông minh cho khu vực địa phương.
Bạn biết rõ các con đường, ngõ hẻm và địa điểm địa phương không có trên Google Maps.

Năng lực của bạn:
1. Tìm đường tối ưu dựa trên giờ đi và tình trạng tắc nghẽn
2. Nhận dạng địa điểm từ ảnh người dùng gửi
3. Cung cấp hướng dẫn chi tiết từng bước kèm ảnh minh hoạ
4. Gợi ý địa điểm local (quán ăn, đường tắt, điểm mốc)
5. Cập nhật thông tin tắc nghẽn real-time

Phong cách trả lời:
- Ngắn gọn, rõ ràng, dùng tiếng Việt tự nhiên
- Ưu tiên hướng dẫn theo mốc địa danh ("đến ngã tư có cây xăng Shell thì rẽ phải")
- Cảnh báo các điểm hay ùn tắc theo giờ
- Đề xuất đường tắt local khi có

Định dạng tuyến đường:
- Hiển thị tổng quãng đường và thời gian ước tính
- Mỗi bước có tên đường và ảnh minh hoạ nếu có
- Highlight các điểm tắc nghẽn có thể xảy ra

Khi nhận tọa độ GPS hoặc ảnh, tự động nhận dạng vị trí và điều chỉnh hướng dẫn.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Intent types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NavigationIntent:
    intent_type: str           # "route" | "identify_place" | "find_poi" | "add_info" | "chat"
    origin: Optional[str] = None
    destination: Optional[str] = None
    depart_time_str: Optional[str] = None
    poi_query: Optional[str] = None
    raw_query: str = ""
    has_image: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def _img_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _resize_for_api(img_path: str, max_dim: int = 1120) -> str:
    """Resize image for API upload, return base64."""
    img = Image.open(img_path)
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# LLM client wrapper (Anthropic / OpenAI / Ollama)
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=settings.llm_api_key or None)
        elif self.provider in ("openai", "ollama"):
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=settings.llm_api_key or "ollama",
                base_url=settings.llm_base_url or None,
            )
        return self._client

    async def chat(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        client = self._get_client()

        if self.provider == "anthropic":
            # Convert messages: separate system from user/assistant
            sys_msg = ""
            conv = []
            for m in messages:
                if m["role"] == "system":
                    sys_msg = m["content"] if isinstance(m["content"], str) else ""
                else:
                    conv.append(m)

            if stream:
                return self._anthropic_stream(client, sys_msg, conv)

            resp = await client.messages.create(
                model=self.model,
                max_tokens=settings.llm_max_tokens,
                system=sys_msg,
                messages=conv,
                temperature=settings.llm_temperature,
            )
            return resp.content[0].text

        else:  # openai / ollama
            if stream:
                return self._openai_stream(client, messages)
            resp = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
            return resp.choices[0].message.content

    async def _anthropic_stream(self, client, system: str, messages: list) -> AsyncIterator[str]:
        async with client.messages.stream(
            model=self.model,
            max_tokens=settings.llm_max_tokens,
            system=system,
            messages=messages,
            temperature=settings.llm_temperature,
        ) as s:
            async for text in s.text_stream:
                yield text

    async def _openai_stream(self, client, messages: list) -> AsyncIterator[str]:
        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            stream=True,
        )
        async for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def parse_intent(self, user_message: str) -> NavigationIntent:
        """Use LLM to extract structured intent from user message."""
        prompt = f"""Phân tích câu hỏi điều hướng sau và trả về JSON:

Câu hỏi: "{user_message}"

Trả về JSON với các trường:
- intent_type: "route" | "identify_place" | "find_poi" | "add_info" | "chat"
- origin: điểm xuất phát (string hoặc null)
- destination: điểm đến (string hoặc null)
- depart_time_str: giờ đi (ví dụ "08:30" hoặc null)
- poi_query: tìm kiếm địa điểm (string hoặc null)

Chỉ trả về JSON thuần, không giải thích."""

        try:
            result = await self.chat([
                {"role": "system", "content": "You are a JSON extractor. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ])
            text = result if isinstance(result, str) else ""
            # Strip markdown code blocks if present
            text = text.strip().strip("```json").strip("```").strip()
            data = json.loads(text)
            return NavigationIntent(raw_query=user_message, **data)
        except Exception as e:
            logger.debug(f"Intent parse failed: {e}")
            return NavigationIntent(intent_type="chat", raw_query=user_message)


# ─────────────────────────────────────────────────────────────────────────────
# Response builder
# ─────────────────────────────────────────────────────────────────────────────

def _format_route_for_bot(route: Route, images_per_step: int = 1) -> str:
    """Format a Route into a rich markdown string for the bot response."""
    dist_km = route.total_distance_m / 1000
    mins = route.total_duration_min
    analysis = route.analysis or {}
    depart_time = route.depart_time or datetime.now()

    # Traffic warning
    h = depart_time.hour
    warning = ""
    for start_h, end_h, factor in settings.peak_hours:
        if start_h <= h < end_h:
            warning = f"\n⚠️ **Giờ cao điểm** (hệ số tắc nghẽn {factor:.1f}×) — dự kiến chậm hơn bình thường.\n"
            break

    lines = [
        f"## Tuyến đường — {dist_km:.1f} km · ~{int(mins)} phút",
        f"Xuất phát lúc {depart_time.strftime('%H:%M')}",
        (
            f"Phân tích: crowd {analysis.get('avg_crowd_level', 0):.2f}, "
            f"weather {analysis.get('avg_weather_severity', 0):.2f}, "
            f"congestion {analysis.get('avg_congestion', 0):.2f}, "
            f"profile `{analysis.get('selected_profile', analysis.get('strategy', 'default'))}`"
            if analysis else ""
        ),
        "---",
    ]
    if warning:
        lines.insert(2, warning)
    for i, step in enumerate(route.steps, 1):
        icon = {
            "depart": "🚀", "turn_left": "⬅️", "turn_right": "➡️",
            "slight_left": "↖️", "slight_right": "↗️", "straight": "⬆️",
            "u_turn": "🔄", "arrive": "🏁", "arrive_left": "🏁",
            "arrive_right": "🏁", "roundabout_enter": "🔁",
        }.get(step.maneuver, "•")

        lines.append(f"{icon} **Bước {i}**: {step.instruction}")
        if step.image_paths:
            for p in step.image_paths[:images_per_step]:
                if isinstance(p, str) and p.strip():
                    lines.append(f"  ???? `{p}`")
        if step.maneuver != "arrive":
            lines.append("")

    lines += [
        "---",
        f"**Tổng cộng**: {dist_km:.1f} km · {int(mins)} phút",
        f"*Cập nhật lúc {datetime.now().strftime('%H:%M')}*",
    ]
    return "\n".join(lines)


def _attach_images_to_steps(route: Route, vpr_engine=None) -> None:
    """
    For each route step, find the closest stored photos and attach them.
    Uses simple haversine proximity if VPR engine is not provided.
    """
    if vpr_engine is None:
        return
    # This is called async from the bot — we attach in the async context
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Main Bot
# ─────────────────────────────────────────────────────────────────────────────

class NavBot:
    """
    Main navigation bot.
    Integrates: LLM intent parsing → route finding → VPR image attachment → response.
    """

    def __init__(self, router: NavRouter, vpr_engine=None):
        self.router = router
        self.vpr = vpr_engine
        self.llm = LLMClient()
        self._history: list[dict] = []   # conversation history per session
        self._ocr = OCRReader()
        self._landmarks = LandmarkDetector()

    def reset_history(self) -> None:
        self._history.clear()

    # ── Image attachment ──────────────────────────────────────────────

    async def _attach_images_to_route(self, route: Route) -> None:
        """Query DB for images near each step's coordinates."""
        for step in route.steps:
            if step.lat is None or step.lon is None:
                continue
            if step.lat == 0 and step.lon == 0:
                continue
            rows = await db.nearby_locations(step.lat, step.lon, radius_deg=0.0005)
            if rows:
                loc = rows[0]
                imgs = await db.get_images_for_location(loc["id"])
                step.image_paths = [img["filepath"] for img in imgs[:2] if img.get("filepath")]

    # ── Core query handler ────────────────────────────────────────────

    async def ask(
        self,
        user_message: str,
        image_path: str | None = None,
        user_lat: float | None = None,
        user_lon: float | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """
        Process a user message and return bot response.

        Args:
            user_message: Text query from user
            image_path:   Optional path to an image the user sent
            user_lat/lon: Optional current GPS position
            stream:       Whether to stream the response
        """
        # ── 1. Build context ──────────────────────────────────────────
        context_parts: list[str] = []
        if user_lat is not None and user_lon is not None:
            nearby_locs = await db.nearby_locations(user_lat, user_lon)
            nearby_pois = await db.nearby_pois(user_lat, user_lon)
            if nearby_locs:
                loc_names = ", ".join(l["name"] for l in nearby_locs[:5])
                context_parts.append(f"Vị trí hiện tại gần: {loc_names}")
            if nearby_pois:
                poi_names = ", ".join(p["name"] for p in nearby_pois[:5])
                context_parts.append(f"POI gần đây: {poi_names}")
            context_parts.append(
                f"Tọa độ GPS: {user_lat:.6f}, {user_lon:.6f} | "
                f"Giờ hiện tại: {datetime.now().strftime('%H:%M')}"
            )

        # ── 2. Parse intent ───────────────────────────────────────────
        intent = await self.llm.parse_intent(user_message)
        intent = self._fallback_intent(intent, user_message, image_path)
        route_text = ""

        if intent.intent_type == "route" and intent.destination:
            route_text = await self._handle_route(
                intent, user_lat, user_lon
            )
            context_parts.append(route_text)

        elif intent.intent_type == "find_poi" and intent.poi_query:
            pois = await db.search_pois(intent.poi_query)
            locs = await db.search_locations(intent.poi_query)
            all_places = pois + locs
            if all_places:
                names = "\n".join(f"- {p['name']} ({p.get('type','')}) tại ({p['lat']:.5f},{p['lon']:.5f})"
                                   for p in all_places[:8])
                context_parts.append(f"Địa điểm tìm được:\n{names}")
            else:
                context_parts.append("Không tìm thấy địa điểm phù hợp trong DB local.")

        elif intent.intent_type == "identify_place" and image_path:
            context_parts.append(await self._handle_identify(image_path, user_lat, user_lon))

        if image_path:
            visual_context = await self._build_visual_context(image_path, user_lat, user_lon)
            if visual_context:
                context_parts.append(visual_context)

        # ── 3. Build messages ─────────────────────────────────────────
        system_msg = SYSTEM_PROMPT
        if context_parts:
            system_msg += "\n\nContext hiện tại:\n" + "\n".join(context_parts)

        user_content: list[dict] | str
        if image_path:
            b64 = _resize_for_api(image_path)
            user_content = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": user_message},
            ]
        else:
            user_content = user_message

        self._history.append({"role": "user", "content": user_content})

        messages = [{"role": "system", "content": system_msg}] + self._history

        # ── 4. Call LLM ───────────────────────────────────────────────
        response = await self.llm.chat(messages, stream=stream)

        if not stream:
            self._history.append({"role": "assistant", "content": response})

        return response

    def _fallback_intent(
        self,
        intent: NavigationIntent,
        user_message: str,
        image_path: str | None,
    ) -> NavigationIntent:
        if intent.intent_type != "chat":
            return intent
        q = user_message.lower()
        if image_path and any(token in q for token in ["day la dau", "o dau", "nhan dien", "anh", "bien", "biển"]):
            intent.intent_type = "identify_place"
            return intent
        route_tokens = ["đi đến", "di den", "tìm đường", "tim duong", "đường đến", "duong den", "route"]
        if any(token in q for token in route_tokens):
            intent.intent_type = "route"
            if not intent.destination:
                intent.destination = user_message.strip()
        return intent

    async def _handle_route(
        self,
        intent: NavigationIntent,
        user_lat: float | None,
        user_lon: float | None,
    ) -> str:
        """Resolve locations, find route, attach images."""
        # Parse departure time
        depart_time = datetime.now()
        if intent.depart_time_str:
            try:
                h, m = map(int, intent.depart_time_str.split(":"))
                depart_time = depart_time.replace(hour=h, minute=m, second=0)
            except Exception:
                pass

        # Resolve origin
        if intent.origin:
            orig_coords = await self.router.resolve_location(intent.origin)
        elif user_lat is not None and user_lon is not None:
            orig_coords = (user_lat, user_lon)
        else:
            return "❌ Không xác định được vị trí xuất phát. Hãy bật GPS hoặc nhập địa chỉ."

        # Resolve destination
        dest_coords = await self.router.resolve_location(intent.destination)
        if not dest_coords:
            return f"❌ Không tìm thấy '{intent.destination}' trong bản đồ local hoặc OSM."

        if not orig_coords:
            return "❌ Không xác định được điểm xuất phát."

        # Find route
        route = await self.router.find_route(
            orig_coords[0], orig_coords[1],
            dest_coords[0], dest_coords[1],
            depart_time=depart_time,
        )

        if not route:
            return f"❌ Không tìm được tuyến đường đến '{intent.destination}'."

        await self._attach_images_to_route(route)
        return _format_route_for_bot(route)

    async def _handle_identify(
        self,
        image_path: str,
        user_lat: float | None,
        user_lon: float | None,
    ) -> str:
        """Use VPR to identify location from image."""
        if self.vpr is None:
            return "VPR engine chưa được khởi tạo."

        img = Image.open(image_path).convert("RGB")
        matches = self.vpr.query(img, top_k=3, query_lat=user_lat, query_lon=user_lon)

        if not matches:
            return "Không nhận ra địa điểm này trong cơ sở dữ liệu ảnh local."

        best = matches[0]
        nearby = await db.nearby_pois(best.lat, best.lon)
        poi_info = ""
        if nearby:
            poi_info = "\nĐịa điểm gần đây: " + ", ".join(p["name"] for p in nearby[:3])

        result = (
            f"📍 Nhận dạng: **{best.location_name}**\n"
            f"Tọa độ: {best.lat:.6f}, {best.lon:.6f}\n"
            f"Độ tương đồng: {best.score:.1%}"
        )
        if best.caption:
            result += f"\nMô tả: {best.caption}"
        visual_hint = await self._build_visual_context(image_path, user_lat, user_lon)
        result += poi_info
        if visual_hint:
            result += "\n" + visual_hint
        return result

    async def _build_visual_context(
        self,
        image_path: str,
        user_lat: float | None,
        user_lon: float | None,
    ) -> str:
        notes: list[str] = []
        if self._landmarks.available:
            landmark_result = self._landmarks.detect(Path(image_path), conf=settings.yolo_confidence, save_preview=False)
            if landmark_result.detections:
                labels = [det.label for det in landmark_result.detections[:5]]
                notes.append("Landmark thay trong anh: " + ", ".join(labels))
        if self._ocr.available:
            ocr_result = self._ocr.detect(Path(image_path), min_conf=settings.ocr_confidence, save_preview=False)
            if ocr_result.blocks:
                texts = [block.text for block in ocr_result.blocks[:5]]
                notes.append("Text doc duoc: " + " | ".join(texts))
        if self.vpr is not None:
            try:
                img = Image.open(image_path).convert("RGB")
                matches = self.vpr.query(img, top_k=2, query_lat=user_lat, query_lon=user_lon)
                if matches:
                    best = matches[0]
                    notes.append(f"VPR goi y: {best.location_name} ({best.score:.0%})")
            except Exception as e:
                logger.debug(f"Visual VPR context failed: {e}")
        return "\n".join(notes)

    # ── Streaming convenience ─────────────────────────────────────────

    async def stream(
        self,
        user_message: str,
        image_path: str | None = None,
        user_lat: float | None = None,
        user_lon: float | None = None,
    ) -> AsyncIterator[str]:
        response = await self.ask(user_message, image_path, user_lat, user_lon, stream=True)
        if isinstance(response, str):
            yield response
        else:
            async for chunk in response:
                yield chunk

    # ── Data ingestion helpers ────────────────────────────────────────

    async def add_location_with_images(
        self,
        name: str,
        lat: float,
        lon: float,
        image_paths: list[Path],
        description: str = "",
        category: str = "general",
        importance: int = 1,
        captions: list[str] | None = None,
    ) -> int:
        """Add a new location with its photos to the database and VPR index."""
        loc_id = await db.add_location(
            name=name, lat=lat, lon=lon,
            description=description,
            category=category,
            importance=importance,
        )

        for i, img_path in enumerate(image_paths[:4]):   # max 4 images per location
            caption = (captions[i] if captions and i < len(captions) else "")
            img_id = await db.add_image(
                location_id=loc_id,
                filename=img_path.name,
                filepath=str(img_path),
                caption=caption,
            )

            if self.vpr and self.vpr.aggregator._fitted:
                from core.vpr_engine import ImageMeta
                meta = ImageMeta(
                    image_id=img_id, location_id=loc_id,
                    location_name=name, lat=lat, lon=lon,
                    filepath=str(img_path), caption=caption,
                )
                faiss_id = self.vpr.index_image(img_path, meta)
                await db.update_faiss_id(img_id, faiss_id)

        logger.info(f"Added location '{name}' with {len(image_paths)} images.")
        return loc_id

    async def rebuild_vpr_index(self) -> None:
        """Rebuild VPR index from all images in the database."""
        if self.vpr is None:
            return
        from core.vpr_engine import ImageMeta

        all_imgs = await db.fetchall(
            """SELECT i.*, l.name AS loc_name, l.lat, l.lon
               FROM images i JOIN locations l ON i.location_id = l.id"""
        )
        metas = [
            ImageMeta(
                image_id=img["id"],
                location_id=img["location_id"],
                location_name=img["loc_name"],
                lat=img["lat"],
                lon=img["lon"],
                filepath=img["filepath"],
                caption=img.get("caption", ""),
            )
            for img in all_imgs
        ]
        if metas:
            self.vpr.index_all_images(metas)
            logger.info(f"VPR index rebuilt with {len(metas)} images.")
