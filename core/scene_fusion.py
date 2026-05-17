from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings
from core.landmark_detector import LandmarkDetector
from core.ocr_reader import OCRReader


@dataclass
class SceneFusionState:
    last_ocr_at: datetime | None = None
    last_yolo_at: datetime | None = None
    last_vpr_at: datetime | None = None
    last_ocr_blocks: list[dict] = field(default_factory=list)
    last_landmarks: list[dict] = field(default_factory=list)
    last_vpr_hint: dict | None = None


class SceneFusionService:
    def __init__(self, vpr_engine: Any = None):
        self._landmarks = LandmarkDetector()
        self._ocr = OCRReader()
        self._vpr = vpr_engine

    async def build_scene_state(
        self,
        frame_path: Path,
        *,
        gps: dict | None,
        nav_event: dict | None,
        nav_session: Any,
        fused_pose: dict | None,
        fusion_state: SceneFusionState,
    ) -> dict:
        now = datetime.utcnow()
        image_width, image_height = self._image_size(frame_path)
        route_progress = self._build_route_progress(nav_event, nav_session)
        visual = {
            "landmarks": fusion_state.last_landmarks,
            "ocr_blocks": fusion_state.last_ocr_blocks,
            "vpr_hint": fusion_state.last_vpr_hint,
            "confidence": round((fused_pose or {}).get("confidence", 0.0), 3),
        }

        if self._landmarks.available and self._should_run(now, fusion_state.last_yolo_at, 1000 / max(settings.realtime_yolo_fps, 0.2)):
            landmark_result = self._landmarks.detect(frame_path, conf=settings.yolo_confidence, save_preview=False)
            fusion_state.last_yolo_at = now
            image_width = landmark_result.image_width or image_width
            image_height = landmark_result.image_height or image_height
            fusion_state.last_landmarks = [
                {
                    "label": det.label,
                    "confidence": round(det.confidence, 3),
                    "bbox": [round(v, 1) for v in det.bbox],
                }
                for det in landmark_result.detections[:8]
            ]
            visual["landmarks"] = fusion_state.last_landmarks

        if self._ocr.available and self._should_run(now, fusion_state.last_ocr_at, settings.realtime_ocr_interval_ms):
            ocr_result = self._ocr.detect(frame_path, min_conf=settings.ocr_confidence, save_preview=False)
            fusion_state.last_ocr_at = now
            image_width = ocr_result.image_width or image_width
            image_height = ocr_result.image_height or image_height
            fusion_state.last_ocr_blocks = [
                {
                    "text": block.text,
                    "confidence": round(block.confidence, 3),
                    "bbox": [[round(pt[0], 1), round(pt[1], 1)] for pt in block.bbox],
                }
                for block in ocr_result.blocks[:8]
            ]
            visual["ocr_blocks"] = fusion_state.last_ocr_blocks

        if self._vpr is not None and self._should_run(now, fusion_state.last_vpr_at, settings.realtime_vpr_interval_ms):
            try:
                from PIL import Image

                img = Image.open(frame_path).convert("RGB")
                qlat = (fused_pose or {}).get("lat") or (gps or {}).get("lat")
                qlon = (fused_pose or {}).get("lon") or (gps or {}).get("lon")
                matches = self._vpr.query(img, top_k=1, query_lat=qlat, query_lon=qlon)
                if matches:
                    best = matches[0]
                    fusion_state.last_vpr_hint = {
                        "location_name": best.location_name,
                        "score": round(float(best.score), 3),
                        "lat": best.lat,
                        "lon": best.lon,
                        "summary": self._summarize_place(best.location_name, best.score),
                    }
                    visual["vpr_hint"] = fusion_state.last_vpr_hint
            except Exception:
                fusion_state.last_vpr_hint = fusion_state.last_vpr_hint
            finally:
                fusion_state.last_vpr_at = now

        return {
            "timestamp": now.isoformat(),
            "gps": gps or {},
            "fused_pose": fused_pose or {},
            "route_progress": route_progress,
            "visual": visual,
            "image_width": image_width,
            "image_height": image_height,
        }

    @staticmethod
    def _should_run(now: datetime, last_run: datetime | None, interval_ms: float) -> bool:
        if last_run is None:
            return True
        return (now - last_run).total_seconds() * 1000.0 >= interval_ms

    @staticmethod
    def _build_route_progress(nav_event: dict | None, nav_session: Any) -> dict:
        current_route = getattr(nav_session, "current_route", None)
        step_idx = getattr(nav_session, "current_step_idx", 0)
        next_instruction = None
        distance_to_next = None
        if current_route and current_route.steps:
            step_idx = min(step_idx, len(current_route.steps) - 1)
            step = current_route.steps[step_idx]
            next_instruction = step.instruction
            distance_to_next = round(step.distance_m, 1)
        return {
            "state": getattr(getattr(nav_session, "state", None), "value", "idle"),
            "current_step_idx": step_idx,
            "off_route": (nav_event or {}).get("type") == "off_route",
            "distance_to_route_m": (nav_event or {}).get("d_route_m") or (nav_event or {}).get("distance_m"),
            "next_maneuver": next_instruction,
            "distance_to_next_turn_m": distance_to_next,
            "map_match": (nav_event or {}).get("map_match"),
        }

    @staticmethod
    def _image_size(frame_path: Path) -> tuple[int | None, int | None]:
        try:
            from PIL import Image

            with Image.open(frame_path) as img:
                return img.size
        except Exception:
            return None, None

    @staticmethod
    def _summarize_place(location_name: str, score: float) -> str:
        confidence = "cao" if score >= 0.75 else "vua" if score >= 0.55 else "thap"
        return f"Co ve ban dang nhin ve {location_name}. Do tin cay VPS {confidence}."
