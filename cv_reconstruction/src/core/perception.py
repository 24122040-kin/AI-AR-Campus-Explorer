# src/core/perception.py
"""
src/core/perception.py
======================
PerceptionModule — scene understanding and environmental awareness.

Responsibilities
----------------
- Load and manage YOLOv11 (segmentation variant).
- building_segmentation  : Instance + semantic segmentation of buildings.
- semantic_mesh_labeling : Project 2-D masks onto the 3-D point cloud.
- light_estimation       : Real-world lighting cue extraction for AR shadows.
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from src.utils.helpers import to_device_dtype

logger = logging.getLogger(__name__)

# ── Optional: Ultralytics YOLOv11 ────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    warnings.warn("ultralytics not installed — YOLO features disabled.")


class PerceptionModule:
    """
    Handles scene understanding tasks for the AR Campus Explorer.

    Parameters
    ----------
    device       : torch.device — must be the same device used system-wide.
    model_dir    : Directory that contains pre-downloaded YOLO weight files.
    yolo_weights : Filename of the YOLOv11 segmentation weights, e.g.
                   "yolo11n-seg.pt" (auto-downloaded on first use if absent).
    """

    def __init__(
        self,
        device: torch.device,
        model_dir: Path = Path("./models"),
        yolo_weights: str = "yolo11n-seg.pt",
    ) -> None:
        self.device = device
        self.model_dir = model_dir

        self._yolo: Optional[Any] = None
        self._yolo_device: str = "cpu"

        self._load_yolo(yolo_weights)

    # ─────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_yolo(self, weights: str) -> None:
        """
        Load YOLOv11 segmentation model.

        Ultralytics handles MPS dispatch internally via the ``device`` kwarg
        at inference time, so we do not call .to(device) on the model object.
        """
        if not _YOLO_AVAILABLE:
            logger.warning("ultralytics not installed — YOLO disabled.")
            self._yolo = None
            return
        try:
            weights_path = self.model_dir / weights
            if not weights_path.exists():
                logger.warning(
                    "YOLO weights not found at %s — downloading …", weights_path
                )
            self._yolo = YOLO(
                str(weights_path) if weights_path.exists() else weights
            )
            # Ultralytics reads "mps" or "cpu" as the device kwarg at predict()
            self._yolo_device = "mps" if self.device.type == "mps" else "cpu"
            logger.info("✅  YOLOv11 loaded (device kwarg: %s).", self._yolo_device)
        except Exception as exc:
            logger.error("Failed to load YOLO: %s", exc)
            self._yolo = None

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Building / Structure Segmentation
    # ─────────────────────────────────────────────────────────────────────────

    def building_segmentation(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Instance + semantic segmentation of buildings and structures.

        Uses YOLOv11-seg (the segmentation variant).  For campus-specific
        classes, fine-tune the YOLO model on your collected dataset and update
        the weights path in the constructor.

        Returns
        -------
        dict with keys:
          masks   : list of (H, W) bool arrays — one per detected instance.
          labels  : list of class name strings.
          boxes   : list of [x1, y1, x2, y2] bounding boxes.
          scores  : list of confidence floats.
          latency_ms : wall-clock time in milliseconds.
        """
        t0 = time.perf_counter()
        result: dict[str, Any] = {
            "masks": [], "labels": [], "boxes": [], "scores": []
        }

        if self._yolo is None:
            logger.warning("YOLO not loaded — building_segmentation skipped.")
            return result

        try:
            preds = self._yolo.predict(
                frame,
                device=self._yolo_device,
                verbose=False,
                retina_masks=True,
            )
            h_orig, w_orig = frame.shape[:2]

            for pred in preds:
                if pred.masks is None:
                    continue
                for i, mask_tensor in enumerate(pred.masks.data):
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    # Resize each mask to match the original frame resolution
                    mask_np = cv2.resize(
                        mask_np,
                        (w_orig, h_orig),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                    cls_id = int(pred.boxes.cls[i].item())
                    label = self._yolo.names.get(cls_id, str(cls_id))

                    result["masks"].append(mask_np)
                    result["labels"].append(label)
                    result["boxes"].append(
                        pred.boxes.xyxy[i].cpu().numpy().tolist()
                    )
                    result["scores"].append(float(pred.boxes.conf[i]))

        except Exception as exc:
            logger.exception("building_segmentation failed: %s", exc)

        result["latency_ms"] = (time.perf_counter() - t0) * 1000
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Semantic Mesh Labeling
    # ─────────────────────────────────────────────────────────────────────────

    def semantic_mesh_labeling(
        self,
        point_cloud: np.ndarray,
        segmentation_result: dict[str, Any],
        camera_matrix: np.ndarray,
        pose: dict[str, Any],
    ) -> np.ndarray:
        """
        Project 2-D segmentation masks onto a 3-D point cloud to attach
        semantic labels (room / department) to each 3-D point.

        Parameters
        ----------
        point_cloud          : (N, 6) XYZRGB float32 array.
        segmentation_result  : Output dict from building_segmentation().
        camera_matrix        : 3×3 camera intrinsics.
        pose                 : Output dict from pose_estimation_6dof().

        Returns
        -------
        label_array : (N,) int32 — label index per point; -1 = unlabelled.
        """
        if not segmentation_result["masks"]:
            return np.full(len(point_cloud), -1, dtype=np.int32)

        labels = np.full(len(point_cloud), -1, dtype=np.int32)
        try:
            if not pose.get("success"):
                return labels

            R = pose["rotation_mat"]
            t = pose["translation_vec"]

            # Project 3-D world points → 2-D pixel coordinates
            pts3d = point_cloud[:, :3].T                # 3×N
            pts_cam = R @ pts3d + t[:, None]            # 3×N
            z = pts_cam[2, :]
            valid = z > 0.01

            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

            u = np.round(fx * pts_cam[0, :] / (z + 1e-8) + cx).astype(int)
            v = np.round(fy * pts_cam[1, :] / (z + 1e-8) + cy).astype(int)

            if segmentation_result["masks"]:
                h = segmentation_result["masks"][0].shape[0]
                w = segmentation_result["masks"][0].shape[1]
            else:
                h, w = 480, 640

            in_frame = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)

            for lbl_idx, mask in enumerate(segmentation_result["masks"]):
                pts_in_mask = in_frame & mask[
                    np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)
                ]
                labels[pts_in_mask] = lbl_idx

        except Exception as exc:
            logger.exception("semantic_mesh_labeling failed: %s", exc)

        return labels

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Light Estimation
    # ─────────────────────────────────────────────────────────────────────────

    def light_estimation(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Estimate real-world ambient lighting for AR shadow rendering.

        Uses a Sobel-gradient heuristic to infer dominant light direction and
        a luminance percentile to estimate intensity — fast enough for every
        rendered frame without a dedicated neural model.

        Returns
        -------
        dict with keys:
          light_direction  : (3,) unit vector pointing toward the light source.
          intensity        : float in [0, 1] — brightness of the brightest 10 %.
          color_temperature: "warm" | "cool" string heuristic.
          ambient_rgb      : (R, G, B) mean channel values normalised to [0, 1].
          latency_ms       : wall-clock time in milliseconds.
        """
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Sobel gradients → proxy for dominant light direction in image-space
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)

        mean_gx = float(np.mean(gx))
        mean_gy = float(np.mean(gy))
        gz = -1.0  # lights are predominantly above the camera

        norm = np.sqrt(mean_gx**2 + mean_gy**2 + gz**2) + 1e-8
        light_dir = np.array([mean_gx / norm, mean_gy / norm, gz / norm])

        # Intensity from the top 10 % brightest pixels
        flat = gray.flatten()
        threshold = np.percentile(flat, 90)
        intensity = float(np.mean(flat[flat >= threshold])) / 255.0

        # Warm/cool colour temperature from B-vs-R channel mean ratio
        b_mean, g_mean, r_mean = cv2.mean(frame)[:3]
        color_temp = "warm" if r_mean > b_mean else "cool"

        return {
            "light_direction": light_dir,
            "intensity": intensity,
            "color_temperature": color_temp,
            "ambient_rgb": (r_mean / 255.0, g_mean / 255.0, b_mean / 255.0),
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }