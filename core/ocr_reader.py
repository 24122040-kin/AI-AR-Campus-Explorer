from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from config.settings import settings


@dataclass
class OCRTextBlock:
    text: str
    confidence: float
    bbox: list[list[float]]


@dataclass
class OCRResult:
    blocks: list[OCRTextBlock]
    preview_path: Path | None = None
    image_width: int | None = None
    image_height: int | None = None


class OCRReader:
    def __init__(self, languages: list[str] | None = None, output_dir: Path | None = None) -> None:
        self.languages = languages or settings.ocr_language_list or ["en"]
        self.output_dir = output_dir or settings.detections_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        settings.ocr_models_dir.mkdir(parents=True, exist_ok=True)
        self.available = False
        self.backend = settings.ocr_backend
        self._reader: Any = None
        self._init_error: str | None = None
        self._init_reader()

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def _init_reader(self) -> None:
        try:
            import easyocr

            self._reader = easyocr.Reader(
                self.languages,
                gpu=settings.device == "cuda",
                model_storage_directory=str(settings.ocr_models_dir),
                download_enabled=True,
            )
            self.available = True
            self.backend = "easyocr"
        except Exception as e:
            self._reader = None
            self.available = False
            self._init_error = str(e)

    def detect(self, image_path: Path, min_conf: float | None = None, save_preview: bool = False) -> OCRResult:
        if not self.available or self._reader is None:
            return OCRResult(blocks=[])

        threshold = min_conf if min_conf is not None else settings.ocr_confidence
        raw_image = cv2.imread(str(image_path))
        gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY) if raw_image is not None else str(image_path)
        raw = self._reader.readtext(gray_image)
        blocks: list[OCRTextBlock] = []
        preview_path: Path | None = None
        image = raw_image
        image_height = int(image.shape[0]) if image is not None else None
        image_width = int(image.shape[1]) if image is not None else None

        for item in raw:
            bbox, text, conf = item
            confidence = float(conf)
            cleaned = str(text).strip()
            if confidence < threshold or not cleaned:
                continue
            points = [[float(p[0]), float(p[1])] for p in bbox]
            blocks.append(OCRTextBlock(text=cleaned, confidence=confidence, bbox=points))
            if save_preview and image is not None:
                pts = [(int(p[0]), int(p[1])) for p in points]
                for i in range(len(pts)):
                    cv2.line(image, pts[i], pts[(i + 1) % len(pts)], (0, 170, 255), 2)
                cv2.putText(
                    image,
                    cleaned[:32],
                    pts[0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if save_preview and image is not None and blocks:
            preview_path = self.output_dir / f"{image_path.stem}_ocr.jpg"
            cv2.imwrite(str(preview_path), image)

        return OCRResult(
            blocks=blocks,
            preview_path=preview_path,
            image_width=image_width,
            image_height=image_height,
        )
