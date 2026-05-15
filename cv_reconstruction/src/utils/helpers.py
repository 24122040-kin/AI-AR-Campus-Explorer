# src/utils/helpers.py
"""
src/utils/helpers.py
====================
Shared utility functions and result dataclasses for the AI AR Campus Explorer.

All MPS-safe tensor helpers and result types live here so every domain module
can import them without circular dependencies.
No model loading or business logic belongs in this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result Dataclasses  (single source of truth — imported everywhere)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LocalizationResult:
    """Result from VPS / visual localization."""
    success: bool
    position_xyz: Optional[np.ndarray] = None       # World coords (meters)
    rotation_matrix: Optional[np.ndarray] = None    # 3×3 rotation
    confidence: float = 0.0
    matched_keypoints: int = 0
    building_id: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class FaceSecurityResult:
    """Result from the face pipeline (embedding + liveness)."""
    embedding: Optional[np.ndarray] = None          # 512-dim ArcFace vector
    is_live: bool = False
    liveness_score: float = 0.0
    face_bbox: Optional[tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    latency_ms: float = 0.0


@dataclass
class DepthResult:
    """Depth map and derived spatial data."""
    depth_map: Optional[np.ndarray] = None          # HxW float32 (meters)
    point_cloud: Optional[np.ndarray] = None        # Nx6 float32 (XYZRGB)
    near_plane: float = 0.1
    far_plane: float = 100.0
    latency_ms: float = 0.0


@dataclass
class FrameInferenceResult:
    """Aggregated output of CampusCVSystem.process_frame (FastAPI entry-point)."""
    localization: Optional[LocalizationResult] = None
    depth: Optional[DepthResult] = None
    detections: list[dict] = field(default_factory=list)
    face_security: Optional[FaceSecurityResult] = None
    privacy_applied: bool = False
    total_latency_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Device Selection  (MPS → CPU, no CUDA)
# ─────────────────────────────────────────────────────────────────────────────

def select_device() -> torch.device:
    """
    Return the best available PyTorch device for Apple Silicon M1 Pro.

    Priority: MPS (Apple GPU) → CPU.
    CUDA is intentionally excluded — M1 Pro has no NVIDIA GPU.
    Requires PyTorch ≥ 1.12 and macOS ≥ 12.3 for MPS support.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("✅  MPS backend detected — using Apple GPU acceleration.")
    else:
        device = torch.device("cpu")
        logger.warning(
            "⚠️  MPS not available — falling back to CPU. "
            "Ensure PyTorch ≥ 1.12 and macOS ≥ 12.3."
        )
    return device


# ─────────────────────────────────────────────────────────────────────────────
# MPS-safe Tensor Helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_device_dtype(
    tensor: torch.Tensor,
    device: torch.device,
    allow_half: bool = True,
) -> torch.Tensor:
    """
    Move *tensor* to *device* and optionally cast to float16.

    On MPS, certain ops (e.g. complex-valued FFTs) do not yet support float16,
    so we catch RuntimeError and silently fall back to float32.
    On CPU we always stay at float32.

    Parameters
    ----------
    tensor      : Input tensor (any dtype).
    device      : Target torch.device.
    allow_half  : When True, attempt float16 on MPS for memory/speed savings.
    """
    tensor = tensor.to(device)
    if allow_half and device.type == "mps":
        try:
            return tensor.half()
        except RuntimeError:
            logger.debug(
                "float16 not supported for this op on MPS — using float32."
            )
            return tensor.float()
    return tensor.float()


def np_to_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a HxWxC BGR uint8 numpy image to a 1×C×H×W float32 tensor on
    *device*, with pixel values normalised to [0, 1].
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)  # 1×C×H×W


def default_camera_matrix(frame: np.ndarray) -> np.ndarray:
    """
    Build a reasonable pinhole intrinsics matrix from frame dimensions.
    Approximates an iPhone 13 Pro focal length.
    Replace with values from camera_calibration() in production.
    """
    h, w = frame.shape[:2]
    f = max(h, w) * 1.2
    return np.array(
        [[f, 0, w / 2],
         [0, f, h / 2],
         [0, 0,     1]], dtype=np.float32
    )