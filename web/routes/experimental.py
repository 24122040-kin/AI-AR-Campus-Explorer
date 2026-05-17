from __future__ import annotations

from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from config.settings import settings
from core.landmark_detector import LandmarkDetector
from core.ocr_reader import OCRReader
from web.uploads import MAX_UPLOAD_SIZE_BYTES, validate_upload


router = APIRouter(tags=["experimental"])
_detector = LandmarkDetector()
_ocr_reader = OCRReader()


def _temp_upload_path(prefix: str, suffix: str) -> Path:
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return settings.images_dir / f"{prefix}_{stamp}{suffix}"


async def _persist_temp_upload(file: UploadFile, prefix: str) -> Path:
    suffix = validate_upload(file)
    data = await file.read()
    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(400, "File too large")
    tmp_path = _temp_upload_path(prefix, suffix)
    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(data)
    return tmp_path


@router.get("/api/detections/{name}")
async def get_detection_preview(name: str):
    path = settings.detections_dir / Path(name).name
    if not path.exists():
        raise HTTPException(404, "Detection preview not found")
    return FileResponse(str(path), media_type="image/jpeg")


@router.post("/api/experimental/landmarks")
async def detect_landmarks(
    file: UploadFile = File(...),
    conf: float = Query(settings.yolo_confidence, ge=0.05, le=0.95),
):
    tmp_path = await _persist_temp_upload(file, "detect")
    try:
        result = _detector.detect(tmp_path, conf=conf, save_preview=True)
        preview_name = result.preview_path.name if result.preview_path else None
        return {
            "ok": True,
            "available": _detector.available,
            "model": _detector.model_name if _detector.available else None,
            "detections": [
                {
                    "label": det.label,
                    "confidence": round(det.confidence, 4),
                    "bbox": det.bbox,
                }
                for det in result.detections
            ],
            "image_width": result.image_width,
            "image_height": result.image_height,
            "preview_name": preview_name,
            "preview_url": f"/api/detections/{preview_name}" if preview_name else None,
            "message": None if _detector.available else "Ultralytics YOLO is not installed in this environment.",
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/api/experimental/ocr")
async def detect_ocr(
    file: UploadFile = File(...),
    conf: float = Query(settings.ocr_confidence, ge=0.05, le=0.99),
):
    tmp_path = await _persist_temp_upload(file, "ocr")
    try:
        result = _ocr_reader.detect(tmp_path, min_conf=conf, save_preview=True)
        preview_name = result.preview_path.name if result.preview_path else None
        return {
            "ok": True,
            "available": _ocr_reader.available,
            "backend": _ocr_reader.backend if _ocr_reader.available else None,
            "languages": _ocr_reader.languages,
            "blocks": [
                {"text": block.text, "confidence": round(block.confidence, 4), "bbox": block.bbox}
                for block in result.blocks
            ],
            "image_width": result.image_width,
            "image_height": result.image_height,
            "preview_name": preview_name,
            "preview_url": f"/api/detections/{preview_name}" if preview_name else None,
            "message": None if _ocr_reader.available else (_ocr_reader.init_error or "OCR backend is not available."),
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/api/experimental/scene")
async def analyze_scene(
    file: UploadFile = File(...),
    landmark_conf: float = Query(settings.yolo_confidence, ge=0.05, le=0.95),
    text_conf: float = Query(settings.ocr_confidence, ge=0.05, le=0.99),
):
    tmp_path = await _persist_temp_upload(file, "scene")
    try:
        landmark_result = _detector.detect(tmp_path, conf=landmark_conf, save_preview=True)
        ocr_result = _ocr_reader.detect(tmp_path, min_conf=text_conf, save_preview=True)
        preview_path = landmark_result.preview_path or ocr_result.preview_path
        preview_name = preview_path.name if preview_path else None
        top_labels = [det.label for det in landmark_result.detections[:5]]
        top_texts = [block.text for block in ocr_result.blocks[:5]]
        summary_bits: list[str] = []
        if top_labels:
            summary_bits.append("moc: " + ", ".join(top_labels))
        if top_texts:
            summary_bits.append("text: " + " | ".join(top_texts))
        return {
            "ok": True,
            "landmarks": {
                "available": _detector.available,
                "model": _detector.model_name if _detector.available else None,
                "detections": [
                    {"label": det.label, "confidence": round(det.confidence, 4), "bbox": det.bbox}
                    for det in landmark_result.detections
                ],
            },
            "ocr": {
                "available": _ocr_reader.available,
                "backend": _ocr_reader.backend if _ocr_reader.available else None,
                "blocks": [
                    {"text": block.text, "confidence": round(block.confidence, 4), "bbox": block.bbox}
                    for block in ocr_result.blocks
                ],
            },
            "summary": "; ".join(summary_bits) if summary_bits else "Chua trich xuat duoc landmark hoac text ro rang.",
            "preview_name": preview_name,
            "preview_url": f"/api/detections/{preview_name}" if preview_name else None,
            "image_width": landmark_result.image_width or ocr_result.image_width,
            "image_height": landmark_result.image_height or ocr_result.image_height,
        }
    finally:
        tmp_path.unlink(missing_ok=True)
