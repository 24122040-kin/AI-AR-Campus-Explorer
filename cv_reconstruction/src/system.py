# src/system.py
"""
src/system.py
=============
CampusCVSystem — top-level orchestrator for the AI AR Campus Explorer.

This class owns the shared torch.device, initialises each domain module,
and exposes the unified process_frame() pipeline that the FastAPI layer calls.

Dependency graph
----------------
CampusCVSystem
  ├── utils.helpers.select_device()         ← device shared by all modules
  ├── core.reconstruction.ReconstructionModule
  ├── core.localization.LocalizationModule
  ├── core.perception.PerceptionModule
  └── core.security.SecurityModule           ← receives PerceptionModule._yolo
                                               to avoid loading weights twice
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.utils.helpers import (
    FrameInferenceResult,
    select_device,
    default_camera_matrix,
)
from src.core.reconstruction import ReconstructionModule
from src.core.localization import LocalizationModule
from src.core.perception import PerceptionModule
from src.core.security import SecurityModule

# One shared logger for the orchestration layer
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CampusCVSystem")


class CampusCVSystem:
    """
    Production-ready Computer Vision orchestrator for the AI AR Campus Explorer.

    Optimised for Apple Silicon M1 Pro via PyTorch MPS backend.
    Falls back to CPU if MPS is unavailable (e.g. CI runners, Linux servers).

    Parameters
    ----------
    model_dir        : Directory containing all pre-downloaded model weights.
    yolo_weights     : YOLOv11 segmentation weight filename.
    depth_model_size : Depth Anything V2 backbone size: "vits" | "vitb" | "vitl".
    use_half         : Attempt float16 on MPS for memory/speed savings.
    map_db_path      : Path to the pre-built campus map .npz archive.
                       Required for VPS; can be built offline with DUSt3R.

    Usage
    -----
    >>> system = CampusCVSystem(model_dir=Path("./models"))
    >>> result = system.process_frame(frame_bgr)
    """

    def __init__(
        self,
        model_dir: Path = Path("./models"),
        yolo_weights: str = "yolo11n-seg.pt",
        depth_model_size: str = "vitl",
        use_half: bool = True,
        map_db_path: Optional[Path] = None,
    ) -> None:
        self.model_dir = model_dir
        self.use_half = use_half
        self.map_db_path = map_db_path

        # ── 1. Shared device (MPS → CPU) ──────────────────────────────────────
        # select_device() contains the authoritative MPS detection logic.
        # The returned device object is passed into every module constructor
        # so all tensor operations share the same backend.
        self.device: torch.device = select_device()

        # ── 2. Domain modules ─────────────────────────────────────────────────
        self.reconstruction = ReconstructionModule(
            device=self.device,
            model_dir=model_dir,
            depth_model_size=depth_model_size,
            use_half=use_half,
        )

        self.localization = LocalizationModule(
            device=self.device,
            use_half=use_half,
        )

        self.perception = PerceptionModule(
            device=self.device,
            model_dir=model_dir,
            yolo_weights=yolo_weights,
        )

        # SecurityModule reuses the YOLO weights already loaded by
        # PerceptionModule — pass the model reference to avoid a double load.
        self.security = SecurityModule(
            device=self.device,
            yolo_model=self.perception._yolo,
        )

        # ── 3. Campus map (optional — needed for VPS) ─────────────────────────
        if map_db_path and map_db_path.exists():
            self.localization.load_campus_map(map_db_path)
        elif map_db_path:
            logger.warning(
                "Campus map path %s does not exist — VPS will be unavailable "
                "until the map is built with dust3r_reconstruction().",
                map_db_path,
            )

        logger.info(
            "CampusCVSystem ready │ device=%s │ half=%s", self.device, use_half
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — convenience pass-throughs for direct module access
    # ─────────────────────────────────────────────────────────────────────────

    def dust3r_reconstruction(self, images, camera_intrinsics=None):
        """Offline 3-D campus reconstruction. See ReconstructionModule."""
        return self.reconstruction.dust3r_reconstruction(images, camera_intrinsics)

    def extract_depth_maps(self, frame, metric_scale=1.0):
        """Real-time monocular depth estimation. See ReconstructionModule."""
        return self.reconstruction.extract_depth_maps(frame, metric_scale)

    def anchor_placement(self, depth_result, target_pixel, camera_matrix):
        """3-D AR anchor from a tapped pixel. See ReconstructionModule."""
        return self.reconstruction.anchor_placement(
            depth_result, target_pixel, camera_matrix
        )

    def visual_localization_vps(self, query_frame, camera_matrix, top_k=5):
        """Full VPS pipeline. See LocalizationModule."""
        return self.localization.visual_localization_vps(
            query_frame, camera_matrix, top_k
        )

    def pose_estimation_6dof(self, frame, camera_matrix, **kwargs):
        """PnP + RANSAC 6-DoF pose. See LocalizationModule."""
        return self.localization.pose_estimation_6dof(
            frame, camera_matrix, **kwargs
        )

    def feature_tracking(self, prev_frame, curr_frame, prev_pts=None):
        """LK optical-flow tracking. See LocalizationModule."""
        return self.localization.feature_tracking(prev_frame, curr_frame, prev_pts)

    def api_identify_location(self, frame, camera_matrix=None):
        """High-level VPS wrapper. See LocalizationModule."""
        if camera_matrix is None:
            camera_matrix = default_camera_matrix(frame)
        depth = self.reconstruction.extract_depth_maps(frame)
        return self.localization.api_identify_location(frame, camera_matrix, depth)

    def building_segmentation(self, frame):
        """Instance segmentation. See PerceptionModule."""
        return self.perception.building_segmentation(frame)

    def light_estimation(self, frame):
        """Real-world lighting estimation. See PerceptionModule."""
        return self.perception.light_estimation(frame)

    def semantic_mesh_labeling(self, point_cloud, segmentation_result,
                               camera_matrix, pose):
        """2-D mask → 3-D point cloud labeling. See PerceptionModule."""
        return self.perception.semantic_mesh_labeling(
            point_cloud, segmentation_result, camera_matrix, pose
        )

    def face_embedding_gen(self, frame):
        """ArcFace embedding generation. See SecurityModule."""
        return self.security.face_embedding_gen(frame)

    def liveness_detection(self, frames, min_motion_threshold=2.5):
        """Passive anti-spoofing. See SecurityModule."""
        return self.security.liveness_detection(frames, min_motion_threshold)

    def privacy_blurring(self, frame, blur_faces=True, blur_plates=True):
        """GDPR face/plate blurring. See SecurityModule."""
        return self.security.privacy_blurring(frame, blur_faces, blur_plates)

    # ─────────────────────────────────────────────────────────────────────────
    # Master Real-time Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        camera_matrix: Optional[np.ndarray] = None,
        enable_face_security: bool = False,
        enable_privacy: bool = True,
    ) -> FrameInferenceResult:
        """
        Master real-time inference pipeline for a single AR frame.
        Designed to run in ≤ 100 ms on M1 Pro at 720p.

        Execution order is driven by the inter-stage dependency graph:

        Step 1 — Privacy blurring
            Must run first so no raw faces/plates are ever passed downstream
            or stored in logs.

        Step 2 — Depth estimation (ReconstructionModule)
            Produces the DepthResult consumed by anchor_placement and passed
            as metadata to api_identify_location.

        Step 3 — VPS localization (LocalizationModule)
            Uses the live frame and camera intrinsics to return a 6-DoF pose
            and building ID.

        Step 4 — Building segmentation (PerceptionModule)
            Runs YOLO on the (already privacy-blurred) frame to detect and
            segment visible structures.

        Step 5 — Face security (SecurityModule, optional)
            Only executed on the login / re-authentication screen; skipped
            during normal AR navigation to save compute.

        Parameters
        ----------
        frame                : HxWx3 BGR uint8 frame from the AR camera.
        camera_matrix        : 3×3 intrinsics. Auto-filled from frame size if None.
        enable_face_security : Run face embedding extraction (login flow only).
        enable_privacy       : Blur faces and plates before any other processing.
        """
        t0 = time.perf_counter()
        result = FrameInferenceResult()

        # Resolve default camera intrinsics
        if camera_matrix is None:
            camera_matrix = default_camera_matrix(frame)

        # ── Step 1: Privacy blurring ──────────────────────────────────────────
        if enable_privacy:
            try:
                frame = self.security.privacy_blurring(frame)
                result.privacy_applied = True
            except Exception as exc:
                logger.warning("Privacy blurring error: %s", exc)

        # ── Step 2: Depth estimation ──────────────────────────────────────────
        try:
            result.depth = self.reconstruction.extract_depth_maps(frame)
        except Exception as exc:
            logger.warning("Depth estimation error: %s", exc)

        # ── Step 3: VPS localization ──────────────────────────────────────────
        try:
            result.localization = self.localization.visual_localization_vps(
                frame, camera_matrix
            )
        except Exception as exc:
            logger.warning("VPS error: %s", exc)

        # ── Step 4: Building segmentation ────────────────────────────────────
        try:
            seg = self.perception.building_segmentation(frame)
            result.detections = [
                {"label": lbl, "box": box, "score": score}
                for lbl, box, score in zip(
                    seg["labels"], seg["boxes"], seg["scores"]
                )
            ]
        except Exception as exc:
            logger.warning("Segmentation error: %s", exc)

        # ── Step 5: Face security (login flow only) ───────────────────────────
        if enable_face_security:
            try:
                result.face_security = self.security.face_embedding_gen(frame)
            except Exception as exc:
                logger.warning("Face security error: %s", exc)

        result.total_latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("process_frame complete in %.1f ms", result.total_latency_ms)
        return result