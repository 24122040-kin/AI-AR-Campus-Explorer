# src/api/routes.py
"""
src/api/routes.py
=================
FastAPI application and all HTTP endpoints for the AI AR Campus Explorer CV API.

Endpoints
---------
POST /api/v1/identify_location  — VPS: building ID + 3-D position from one image.
POST /api/v1/process_frame      — Full AR inference pipeline (depth + VPS + seg).
GET  /health                    — Liveness probe with model-load status.

The CampusCVSystem singleton is created lazily on the first request so the
FastAPI worker can start and pass health checks even before the models finish
loading (useful in Docker / k8s environments with readiness probes).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from src.system import CampusCVSystem

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI AR Campus Explorer — CV API",
    description=(
        "Computer Vision endpoints for the AR Campus Explorer.\n\n"
        "Dev 1 — Trung Hiếu | Optimised for Apple Silicon M1 Pro (MPS backend)"
    ),
    version="1.0.0",
)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton — lazy initialisation
# ─────────────────────────────────────────────────────────────────────────────

_cv_system: Optional[CampusCVSystem] = None


def get_cv_system() -> CampusCVSystem:
    """
    Return the global CampusCVSystem, creating it on the first call.

    Using a module-level singleton avoids reloading multi-GB model weights on
    every request while remaining compatible with single-worker uvicorn and
    gunicorn deployments.
    """
    global _cv_system
    if _cv_system is None:
        logger.info("Initialising CampusCVSystem for the first time …")
        _cv_system = CampusCVSystem(
            model_dir=Path("./models"),
            map_db_path=Path("./data/campus_map.npz"),
        )
    return _cv_system


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _decode_upload(file_bytes: bytes) -> np.ndarray:
    """
    Decode raw upload bytes to a BGR numpy array.
    Raises ValueError if the bytes cannot be decoded as an image.
    """
    arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(
            "Could not decode image. "
            "Ensure the upload is a valid JPEG, PNG, or BMP file."
        )
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/identify_location",
    summary="Identify campus location from a single image",
    response_description=(
        "Building ID, 3-D world position, localisation confidence, "
        "and depth extents."
    ),
)
async def identify_location(
    file: UploadFile = File(..., description="JPEG/PNG frame from the AR camera"),
) -> JSONResponse:
    """
    Run the full VPS pipeline on the uploaded frame.

    Returns the building the user is currently facing, their estimated 3-D
    position in world coordinates, a confidence score, and the depth range
    visible in the frame.
    """
    try:
        frame = _decode_upload(await file.read())
        cv_sys = get_cv_system()
        result = cv_sys.api_identify_location(frame, camera_matrix=None)
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("/api/v1/identify_location unhandled error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/api/v1/process_frame",
    summary="Full AR frame inference (depth + VPS + segmentation)",
    response_description="Aggregated FrameInferenceResult serialised to JSON.",
)
async def process_frame_endpoint(
    file: UploadFile = File(..., description="JPEG/PNG frame from the AR camera"),
    enable_face: bool = Query(
        False, description="Enable face embedding (login flow only)"
    ),
    enable_privacy: bool = Query(
        True, description="Blur faces and licence plates before processing"
    ),
) -> JSONResponse:
    """
    Run the complete per-frame AR inference pipeline:

    1. Privacy blurring → 2. Depth estimation → 3. VPS localisation
    → 4. Building segmentation → 5. Face security (optional).

    Designed to complete in ≤ 100 ms at 720p on an Apple M1 Pro.
    """
    try:
        frame = _decode_upload(await file.read())
        cv_sys = get_cv_system()

        result = cv_sys.process_frame(
            frame,
            camera_matrix=None,         # auto-inferred from frame dimensions
            enable_face_security=enable_face,
            enable_privacy=enable_privacy,
        )

        loc = result.localization
        return JSONResponse(
            content={
                "localized": loc.success if loc else False,
                "building_id": loc.building_id if loc else None,
                "position_xyz": (
                    loc.position_xyz.tolist()
                    if (loc and loc.position_xyz is not None)
                    else None
                ),
                "confidence": loc.confidence if loc else 0.0,
                "matched_keypoints": loc.matched_keypoints if loc else 0,
                "detections": result.detections,
                "depth_near": (
                    result.depth.near_plane if result.depth else None
                ),
                "depth_far": (
                    result.depth.far_plane if result.depth else None
                ),
                "privacy_applied": result.privacy_applied,
                "total_latency_ms": result.total_latency_ms,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("/api/v1/process_frame unhandled error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/health",
    summary="Service liveness and model-load status",
)
async def health() -> JSONResponse:
    """
    Lightweight health probe.

    Returns the active compute device and a per-model boolean indicating
    whether each weight file was successfully loaded.  Use this endpoint
    as a Kubernetes readiness probe once the system has initialised.
    """
    cv_sys = get_cv_system()
    return JSONResponse(
        content={
            "status": "ok",
            "device": str(cv_sys.device),
            "mps_available": torch.backends.mps.is_available(),
            "models": {
                "depth_anything_v2": (
                    cv_sys.reconstruction._depth_model is not None
                ),
                "lightglue": (
                    cv_sys.localization._extractor is not None
                ),
                "yolo_v11": (
                    cv_sys.perception._yolo is not None
                ),
                "insightface": (
                    cv_sys.security._face_app is not None
                ),
            },
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (development only)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Development: python -m src.api.routes
    # Production:  uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --workers 1
    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # set to False in production
        log_level="info",
    )