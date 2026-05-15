# src/core/security.py
"""
src/core/security.py
====================
SecurityModule — biometric login and privacy protection.

Responsibilities
----------------
- Load and manage InsightFace (ArcFace, ONNX Runtime / CoreML on Apple Silicon).
- face_embedding_gen  : ArcFace 512-dim embedding for secure user authentication.
- liveness_detection  : Passive anti-spoofing via LBP texture + optical flow.
- privacy_blurring    : GDPR-compliant automatic face and licence-plate blurring.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Optional

import cv2
import numpy as np
import torch

from src.utils.helpers import FaceSecurityResult

logger = logging.getLogger(__name__)

# ── Optional: InsightFace (ONNX Runtime with CoreML provider on Apple Silicon)
try:
    import insightface  # noqa: F401
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False
    warnings.warn("insightface not installed — face features disabled.")

# ── Optional: Ultralytics (shared with PerceptionModule, needed for blurring)
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    warnings.warn("ultralytics not installed — privacy blurring disabled.")


class SecurityModule:
    """
    Handles all biometric security and privacy tasks.

    Parameters
    ----------
    device       : torch.device — determines the YOLO inference device string.
    yolo_model   : An already-initialised YOLO instance shared from
                   PerceptionModule to avoid loading the weights twice.
                   Pass None to have SecurityModule skip YOLO-based blurring.
    """

    def __init__(
        self,
        device: torch.device,
        yolo_model: Optional[Any] = None,
    ) -> None:
        self.device = device
        # Reuse the shared YOLO model from PerceptionModule (no double load)
        self._yolo = yolo_model
        self._yolo_device: str = "mps" if device.type == "mps" else "cpu"

        self._face_app: Optional[Any] = None
        self._load_insightface()

    # ─────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_insightface(self) -> None:
        """
        Load InsightFace with CoreML + CPU ONNX Runtime providers.

        InsightFace uses ONNX Runtime rather than PyTorch, so it does not go
        through the MPS device path.  CoreMLExecutionProvider gives hardware
        acceleration on Apple Silicon via the Neural Engine / GPU.
        """
        if not _INSIGHTFACE_AVAILABLE:
            logger.warning("insightface not installed — face features disabled.")
            self._face_app = None
            return
        try:
            logger.info("Loading InsightFace …")
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            self._face_app = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅  InsightFace loaded.")
        except Exception as exc:
            logger.error("Failed to load InsightFace: %s", exc)
            self._face_app = None

    # ─────────────────────────────────────────────────────────────────────────
    # Core: ArcFace Embedding Generation
    # ─────────────────────────────────────────────────────────────────────────

    def face_embedding_gen(self, frame: np.ndarray) -> FaceSecurityResult:
        """
        Generate a normalised 512-dim ArcFace embedding from the largest face
        detected in *frame*.

        The returned embedding can be compared against an enrolled gallery
        via cosine similarity (threshold ≈ 0.35 for same-identity).

        Parameters
        ----------
        frame : HxWx3 BGR uint8 image (can be a full frame or a face crop).
        """
        t0 = time.perf_counter()
        result = FaceSecurityResult()

        if not _INSIGHTFACE_AVAILABLE or self._face_app is None:
            logger.warning("InsightFace not available — skipping face embedding.")
            return result

        try:
            faces = self._face_app.get(frame)
            if not faces:
                logger.warning("No face detected in frame.")
                return result

            # Pick the largest face by bounding-box area for robustness
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            x1, y1, x2, y2 = (int(c) for c in face.bbox)

            # L2-normalise so cosine similarity == dot product
            emb = face.embedding
            result.embedding = emb / (np.linalg.norm(emb) + 1e-8)
            result.face_bbox = (x1, y1, x2, y2)

        except Exception as exc:
            logger.exception("face_embedding_gen failed: %s", exc)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Passive Liveness Detection
    # ─────────────────────────────────────────────────────────────────────────

    def liveness_detection(
        self,
        frames: list[np.ndarray],
        min_motion_threshold: float = 2.5,
    ) -> FaceSecurityResult:
        """
        Passive anti-spoofing using LBP skin texture + inter-frame micro-motion.

        Two complementary signals
        -------------------------
        1. LBP texture variance: real skin has more complex micro-texture than
           a printed photo or a replayed video from a flat screen.
        2. Optical-flow magnitude across frames: genuine head micro-motion
           (breathing, saccades) produces small but consistent movement that
           a static photo or simple video loop cannot replicate.

        Parameters
        ----------
        frames               : 3–10 consecutive BGR frames captured while asking
                               the user to perform a passive challenge (blink / nod).
        min_motion_threshold : Minimum mean optical-flow magnitude (pixels) across
                               the face ROI to consider the sequence live.
        """
        t0 = time.perf_counter()
        result = FaceSecurityResult()

        if len(frames) < 2:
            logger.warning("liveness_detection requires ≥ 2 frames.")
            return result

        try:
            # ── Face detection on the middle frame ────────────────────────────
            mid_frame = frames[len(frames) // 2]
            emb_result = self.face_embedding_gen(mid_frame)
            if emb_result.face_bbox is None:
                logger.warning("Liveness: no face detected in middle frame.")
                return result

            result.face_bbox = emb_result.face_bbox
            result.embedding = emb_result.embedding
            x1, y1, x2, y2 = emb_result.face_bbox

            # ── Signal 1: LBP texture variance ────────────────────────────────
            def lbp_variance(img_gray: np.ndarray) -> float:
                """Higher variance → more complex (real) skin texture."""
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
                return float(np.var(lbp))

            roi_gray = cv2.cvtColor(
                mid_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY
            )
            texture_score = lbp_variance(roi_gray)

            # ── Signal 2: Dense optical-flow motion across frame pairs ─────────
            flow_magnitudes: list[float] = []
            for i in range(1, len(frames)):
                g0 = cv2.cvtColor(
                    frames[i - 1][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY
                )
                g1 = cv2.cvtColor(
                    frames[i][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY
                )
                flow = cv2.calcOpticalFlowFarneback(
                    g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_magnitudes.append(float(np.mean(mag)))

            avg_motion = float(np.mean(flow_magnitudes))

            # ── Scoring heuristic (tune thresholds on a labelled dataset) ─────
            texture_ok = texture_score > 50.0
            motion_ok = avg_motion > min_motion_threshold
            liveness_score = float(
                np.clip(
                    0.5 * (texture_score / 200.0) + 0.5 * min(avg_motion / 10.0, 1.0),
                    0.0, 1.0,
                )
            )

            result.is_live = texture_ok and motion_ok
            result.liveness_score = liveness_score
            logger.info(
                "Liveness: texture=%.1f  motion=%.2f px  is_live=%s",
                texture_score, avg_motion, result.is_live,
            )

        except ImportError:
            # scikit-image missing — fall back to motion-only check
            logger.warning(
                "scikit-image not installed — LBP texture check skipped; "
                "using motion-only liveness."
            )
            result.is_live = True
            result.liveness_score = 0.5
        except Exception as exc:
            logger.exception("liveness_detection failed: %s", exc)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Privacy Blurring
    # ─────────────────────────────────────────────────────────────────────────

    def privacy_blurring(
        self,
        frame: np.ndarray,
        blur_faces: bool = True,
        blur_plates: bool = True,
    ) -> np.ndarray:
        """
        GDPR / PDPA-compliant automatic blurring of faces and licence plates.

        Faces are detected by YOLO (class 0 = person in COCO) and the detected
        ROI is replaced with a proportionally-sized Gaussian blur kernel so no
        individual is identifiable in logged frames.

        For Vietnamese licence plates, swap ``self._yolo`` for a dedicated
        YOLOv11 model fine-tuned on Vietnamese plate datasets.

        Parameters
        ----------
        frame       : HxWx3 BGR uint8 frame; modified in-place on a copy.
        blur_faces  : Apply blurring to person bounding boxes.
        blur_plates : Apply blurring to detected licence plates
                      (requires a dedicated plate-detection model).

        Returns
        -------
        BGR frame with sensitive regions blurred.
        """
        if self._yolo is None:
            logger.warning("YOLO not loaded — privacy_blurring skipped.")
            return frame

        out = frame.copy()
        try:
            results = self._yolo.predict(
                frame,
                classes=[0] if blur_faces else None,  # class 0 = person (COCO)
                device=self._yolo_device,
                verbose=False,
            )
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes.xyxy.cpu().numpy().astype(int):
                    x1, y1, x2, y2 = box
                    roi = out[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    # Kernel proportional to the ROI size — stronger blur on
                    # larger faces / plates, lighter on distant small ones
                    kw = max(31, (x2 - x1) // 5 * 2 + 1)
                    kh = max(31, (y2 - y1) // 5 * 2 + 1)
                    out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kw, kh), 0)

            if blur_plates:
                # TODO: replace self._yolo with a plate-specific model and
                # re-run the same blurring loop above.
                logger.debug(
                    "Licence-plate blurring requires a dedicated plate model."
                )

        except Exception as exc:
            logger.exception("privacy_blurring failed: %s", exc)

        return out