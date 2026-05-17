from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from config.settings import settings


@dataclass
class LandmarkDetection:
    label: str
    confidence: float
    bbox: list[float]


@dataclass
class LandmarkDetectionResult:
    detections: list[LandmarkDetection]
    preview_path: Path | None = None
    image_width: int | None = None
    image_height: int | None = None


class LandmarkDetector:
    """
    Optional landmark detector.
    Uses Ultralytics YOLO if installed; otherwise reports unavailable cleanly.
    """

    def __init__(self, model_name: str | None = None, output_dir: Path | None = None) -> None:
        configured_name = model_name or settings.yolo_model
        local_model_path = settings.yolo_config_dir / Path(configured_name).name
        self.model_name = str(local_model_path) if local_model_path.exists() else configured_name
        self.output_dir = output_dir or settings.detections_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        settings.yolo_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(settings.yolo_config_dir))
        self._model: Any = None
        self.available = False
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            self.available = True
        except Exception:
            self._model = None
            self.available = False

    def detect(self, image_path: Path, conf: float | None = None, save_preview: bool = False) -> LandmarkDetectionResult:
        if not self.available or self._model is None:
            return LandmarkDetectionResult(detections=[])

        predict_conf = conf if conf is not None else settings.yolo_confidence
        results = self._model.predict(str(image_path), conf=predict_conf, verbose=False)
        detections: list[LandmarkDetection] = []
        preview_path: Path | None = None
        image_width: int | None = None
        image_height: int | None = None
        for result in results:
            if result.orig_shape:
                image_height, image_width = int(result.orig_shape[0]), int(result.orig_shape[1])
            names = result.names
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    LandmarkDetection(
                        label=str(names.get(cls_idx, cls_idx)),
                        confidence=float(box.conf.item()),
                        bbox=[float(v) for v in xyxy],
                    )
                )
            if save_preview:
                plotted = result.plot()
                preview_path = self.output_dir / f"{image_path.stem}_detect.jpg"
                try:
                    import cv2

                    cv2.imwrite(str(preview_path), plotted)
                except Exception:
                    preview_path = None
        return LandmarkDetectionResult(
            detections=detections,
            preview_path=preview_path,
            image_width=image_width,
            image_height=image_height,
        )
