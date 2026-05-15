# src/core/reconstruction.py
"""
src/core/reconstruction.py
==========================
ReconstructionModule — 3-D world building and spatial reasoning.

Responsibilities
----------------
- Load and manage Depth Anything V2 (monocular depth estimation).
- dust3r_reconstruction  : Offline multi-view 3-D point cloud from campus photos.
- extract_depth_maps     : Real-time per-frame monocular depth (MPS-optimised).
- _depth_to_pointcloud   : Back-projection helper (depth → XYZRGB cloud).
- anchor_placement       : Compute 3-D AR anchor from a tapped pixel.
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
import torch.nn.functional as F
from PIL import Image

from src.utils.helpers import DepthResult, to_device_dtype

logger = logging.getLogger(__name__)

# ── Optional: open3d (point cloud I/O and visualisation) ─────────────────────
try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False
    warnings.warn("open3d not installed — 3D reconstruction features disabled.")


class ReconstructionModule:
    """
    Handles depth estimation and 3-D reconstruction for the AR Campus Explorer.

    Parameters
    ----------
    device           : torch.device — must be the same device used system-wide.
    model_dir        : Directory that contains pre-downloaded model weights.
    depth_model_size : Depth Anything V2 backbone variant: "vits" | "vitb" | "vitl".
    use_half         : Whether to attempt float16 on MPS (default True).
    """

    def __init__(
        self,
        device: torch.device,
        model_dir: Path = Path("./models"),
        depth_model_size: str = "vitl",
        use_half: bool = True,
    ) -> None:
        self.device = device
        self.model_dir = model_dir
        self.use_half = use_half

        self._depth_model: Optional[Any] = None
        self._depth_processor: Optional[Any] = None

        self._load_depth_model(depth_model_size)

    # ─────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_depth_model(self, size: str) -> None:
        """Load Depth Anything V2 from HuggingFace Hub onto the MPS/CPU device."""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            model_id = f"depth-anything/Depth-Anything-V2-{size.capitalize()}-hf"
            logger.info("Loading Depth Anything V2 (%s) …", size)

            self._depth_processor = AutoImageProcessor.from_pretrained(model_id)
            self._depth_model = AutoModelForDepthEstimation.from_pretrained(model_id)

            # Place on device; optionally cast to float16 for MPS speed gains
            self._depth_model = self._depth_model.to(self.device)
            if self.use_half and self.device.type == "mps":
                self._depth_model = self._depth_model.half()

            self._depth_model.eval()
            logger.info("✅  Depth Anything V2 loaded.")
        except Exception as exc:
            logger.error("Failed to load Depth Anything V2: %s", exc)
            self._depth_model = None
            self._depth_processor = None

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Offline DUSt3R Multi-view Reconstruction
    # ─────────────────────────────────────────────────────────────────────────

    def dust3r_reconstruction(
        self,
        images: list[np.ndarray],
        camera_intrinsics: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """
        Offline 3-D reconstruction via DUSt3R.
        Accepts a list of overlapping campus images (BGR uint8).

        Install DUSt3R separately:
            git clone https://github.com/naver/dust3r && pip install -e dust3r/

        Returns
        -------
        dict with keys:
          point_cloud  : open3d.geometry.PointCloud (if open3d is available)
          poses        : (N, 4, 4) camera-to-world transforms
          dense_pts    : (M, 3) numpy array of reconstructed 3-D points
          confidence   : (M,) per-point confidence scores
          latency_ms   : wall-clock time in milliseconds
        """
        t0 = time.perf_counter()
        result: dict[str, Any] = {}

        if len(images) < 2:
            raise ValueError("DUSt3R requires at least 2 overlapping images.")

        try:
            # DUSt3R must be installed from source (see docstring above)
            from dust3r.inference import inference
            from dust3r.model import AsymmetricCroCo3DStereo
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

            model_path = (
                self.model_dir / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            )
            logger.info("Loading DUSt3R model …")
            dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(str(model_path))
            dust3r_model = dust3r_model.to(self.device).eval()

            # float16 on MPS for memory efficiency during offline reconstruction
            if self.use_half and self.device.type == "mps":
                dust3r_model = dust3r_model.half()

            # ── Step 2: All-pairs image pairing ──────────────────────────────
            pairs = make_pairs(
                images, scene_graph="complete", prefilter=None, symmetrize=True
            )

            # ── Step 3: Dense stereo inference ────────────────────────────────
            logger.info("Running DUSt3R inference on %d pairs …", len(pairs))
            output = inference(pairs, dust3r_model, self.device, batch_size=1)

            # ── Step 4: Global point-cloud alignment (bundle adjustment) ──────
            scene = global_aligner(
                output,
                device=self.device,
                mode=GlobalAlignerMode.PointCloudOptimizer,
            )
            loss = scene.compute_global_alignment(
                init="mst", niter=200, schedule="linear", lr=0.01
            )
            logger.info("DUSt3R alignment loss: %.4f", loss)

            # ── Step 5: Extract dense points, masks, and camera poses ─────────
            pts3d = scene.get_pts3d()   # list of (H, W, 3) tensors
            masks = scene.get_masks()   # list of (H, W) bool tensors
            poses = scene.get_im_poses().detach().cpu().numpy()  # (N, 4, 4)

            dense_pts = np.concatenate(
                [p[m].reshape(-1, 3) for p, m in zip(pts3d, masks)], axis=0
            )
            confs = np.concatenate(
                [
                    scene.get_conf()[i][masks[i]]
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    for i in range(len(masks))
                ],
                axis=0,
            )

            result["poses"] = poses
            result["dense_pts"] = dense_pts
            result["confidence"] = confs

            # ── Step 6: Open3D point cloud (optional) ─────────────────────────
            if _OPEN3D_AVAILABLE:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(dense_pts)
                pcd = pcd.voxel_down_sample(voxel_size=0.02)  # 2 cm resolution
                result["point_cloud"] = pcd
                logger.info(
                    "Point cloud: %d points after downsampling.", len(pcd.points)
                )

        except ImportError:
            logger.error(
                "DUSt3R not installed. "
                "Clone https://github.com/naver/dust3r and run `pip install -e .`"
            )
        except Exception as exc:
            logger.exception("dust3r_reconstruction failed: %s", exc)

        result["latency_ms"] = (time.perf_counter() - t0) * 1000
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Real-time Monocular Depth Estimation
    # ─────────────────────────────────────────────────────────────────────────

    def extract_depth_maps(
        self,
        frame: np.ndarray,
        metric_scale: float = 1.0,
    ) -> DepthResult:
        """
        Monocular depth estimation via Depth Anything V2.

        Parameters
        ----------
        frame        : HxWx3 BGR uint8 frame from the phone camera.
        metric_scale : Multiply raw depth output by this factor to convert to
                       metres. Calibrate once against a known-distance target.

        Returns
        -------
        DepthResult  : Contains the HxW float32 depth map (metres), a sparse
                       XYZRGB point cloud, and near/far plane extents.
        """
        t0 = time.perf_counter()
        result = DepthResult()

        if self._depth_model is None:
            logger.warning("Depth model not loaded — skipping extract_depth_maps.")
            return result

        try:
            # ── Pre-processing ────────────────────────────────────────────────
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self._depth_processor(images=img_pil, return_tensors="pt")

            # Move pixel values to device; use float16 on MPS when allowed
            pixel_values = inputs["pixel_values"].to(self.device)
            if self.use_half and self.device.type == "mps":
                pixel_values = pixel_values.half()

            # ── Inference (no grad for memory efficiency) ─────────────────────
            with torch.no_grad():
                outputs = self._depth_model(pixel_values=pixel_values)
                predicted_depth = outputs.predicted_depth  # (1, H', W')

            # ── Post-processing: upsample back to original resolution ──────────
            h, w = frame.shape[:2]
            depth_up = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth_np = depth_up.float().cpu().numpy() * metric_scale
            result.depth_map = depth_np
            result.near_plane = float(depth_np.min())
            result.far_plane = float(depth_np.max())
            result.point_cloud = self._depth_to_pointcloud(depth_np, frame)

        except Exception as exc:
            logger.exception("extract_depth_maps failed: %s", exc)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: Depth Map → Coloured Point Cloud
    # ─────────────────────────────────────────────────────────────────────────

    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,
        frame: np.ndarray,
        fx: float = 600.0,
        fy: float = 600.0,
        max_points: int = 5000,
    ) -> np.ndarray:
        """
        Back-project a depth map to a coloured point cloud.

        Returns (N, 6) float32 array where each row is [X, Y, Z, R, G, B]
        with RGB normalised to [0, 1].  Uses default focal lengths if camera
        intrinsics are not passed in; call extract_depth_maps with a calibrated
        camera matrix for metric accuracy.
        """
        h, w = depth.shape
        cx, cy = w / 2.0, h / 2.0
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        z = depth.flatten()
        x = (u.flatten() - cx) * z / fx
        y = (v.flatten() - cy) * z / fy

        pts = np.stack([x, y, z], axis=-1)
        colors = frame.reshape(-1, 3)[:, ::-1] / 255.0  # BGR → RGB, [0, 1]

        # Remove sky / background and extremely close points
        valid = (z > 0.1) & (z < 80.0)
        pts, colors = pts[valid], colors[valid]

        # Random subsample to cap memory usage
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts, colors = pts[idx], colors[idx]

        return np.concatenate([pts, colors], axis=-1).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Core: AR Anchor Placement
    # ─────────────────────────────────────────────────────────────────────────

    def anchor_placement(
        self,
        depth_result: DepthResult,
        target_pixel: tuple[int, int],
        camera_matrix: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute the 3-D world position and surface normal for a tapped pixel,
        so the AR engine can attach a virtual label or object at that location.

        Parameters
        ----------
        depth_result   : Output of extract_depth_maps for the current frame.
        target_pixel   : (u, v) pixel coordinate of the tap / detection centre.
        camera_matrix  : 3×3 intrinsics.

        Returns
        -------
        dict with keys: success, world_position (3,), anchor_normal (3,),
                        depth_m, and an optional reason string on failure.
        """
        if depth_result.depth_map is None:
            return {"success": False, "reason": "No depth map available"}

        u, v = target_pixel
        depth = float(depth_result.depth_map[v, u])

        if depth < 0.1:
            return {"success": False, "reason": "Depth too close or invalid"}

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # Estimate surface normal from immediate pixel neighbourhood
        dm = depth_result.depth_map
        h_dm, w_dm = dm.shape
        u1, v1 = min(u + 1, w_dm - 1), v
        u2, v2 = u, min(v + 1, h_dm - 1)

        def back_project(pu: int, pv: int) -> np.ndarray:
            dz = dm[pv, pu]
            return np.array(
                [(pu - cx) * dz / fx, (pv - cy) * dz / fy, dz]
            )

        p0 = np.array([x, y, z])
        p1 = back_project(u1, v1)
        p2 = back_project(u2, v2)

        normal = np.cross(p1 - p0, p2 - p0)
        norm_mag = np.linalg.norm(normal) + 1e-8
        normal /= norm_mag

        return {
            "success": True,
            "world_position": p0,
            "anchor_normal": normal,
            "depth_m": depth,
        }