# src/core/localization.py
"""
src/core/localization.py
========================
LocalizationModule — the campus positioning brain.

Responsibilities
----------------
- Load and manage SuperPoint + LightGlue models.
- Load the pre-built campus map database.
- visual_localization_vps  : full VPS pipeline (coarse retrieval → geometric
                              re-ranking → PnP+RANSAC).
- pose_estimation_6dof     : OpenCV PnP wrapper (also consumed by VPS).
- feature_tracking         : Lucas-Kanade optical flow for inter-frame stability.
- api_identify_location    : thin high-level wrapper combining VPS + depth info.
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

from src.utils.helpers import (
    LocalizationResult,
    DepthResult,
    select_device,
    to_device_dtype,
)

logger = logging.getLogger(__name__)

# ── Optional: LightGlue + SuperPoint ─────────────────────────────────────────
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    _LIGHTGLUE_AVAILABLE = True
except ImportError:
    _LIGHTGLUE_AVAILABLE = False
    warnings.warn("lightglue not installed — VPS/tracking features disabled.")


class LocalizationModule:
    """
    Handles all visual localization tasks for the AR Campus Explorer.

    Parameters
    ----------
    device   : torch.device — must be the same device used system-wide.
    use_half : Whether to attempt float16 on MPS (default True).
    """

    def __init__(self, device: torch.device, use_half: bool = True) -> None:
        self.device = device
        self.use_half = use_half
        self._campus_map: dict[str, Any] = {}

        self._load_lightglue()

    # ─────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_lightglue(self) -> None:
        """Load SuperPoint feature extractor and LightGlue matcher."""
        if not _LIGHTGLUE_AVAILABLE:
            logger.warning("LightGlue unavailable — VPS disabled.")
            self._extractor: Optional[Any] = None
            self._matcher: Optional[Any] = None
            return
        try:
            logger.info("Loading SuperPoint + LightGlue …")
            # NOTE: LightGlue supports float16 on CUDA; on MPS keep float32 for
            # numerical stability — do NOT call .half() on these two models.
            self._extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self._matcher = LightGlue(features="superpoint").eval().to(self.device)
            logger.info("✅  LightGlue loaded.")
        except Exception as exc:
            logger.error("Failed to load LightGlue: %s", exc)
            self._extractor = None
            self._matcher = None

    def load_campus_map(self, path: Path) -> None:
        """
        Load a pre-built campus map database (numpy .npz archive).

        Expected keys
        -------------
        descriptors   : (N_frames, D)  — mean global SuperPoint descriptor per frame.
        keypoints     : object array of (K_i, 2) per-frame keypoint arrays.
        poses         : (N_frames, 4, 4) camera-to-world transforms.
        building_ids  : (N_frames,) string labels.

        Per-frame per-keypoint descriptors and 3-D world points are stored
        under dynamic keys ``kpt_descriptors_<i>`` and ``world_pts_<i>``.
        """
        try:
            data = np.load(path, allow_pickle=True)
            self._campus_map = {
                # Move the global descriptors to the inference device once
                "descriptors": torch.from_numpy(
                    data["descriptors"]
                ).float().to(self.device),
                "keypoints": data["keypoints"],
                "poses": data["poses"],
                "building_ids": data["building_ids"],
            }
            # Store any per-frame arrays that were packed into the archive
            for key in data.files:
                if key.startswith("kpt_descriptors_") or key.startswith("world_pts_"):
                    self._campus_map[key] = data[key]

            logger.info(
                "✅  Campus map loaded: %d reference frames.", len(data["poses"])
            )
        except Exception as exc:
            logger.error("Failed to load campus map DB: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Core: 6-DoF Pose Estimation
    # ─────────────────────────────────────────────────────────────────────────

    def pose_estimation_6dof(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        world_points: Optional[np.ndarray] = None,
        image_points: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """
        6-DoF pose estimation via OpenCV PnP + RANSAC.

        Parameters
        ----------
        frame          : BGR query frame (used only for context/logging).
        camera_matrix  : 3×3 intrinsics (from camera_calibration).
        dist_coeffs    : Distortion coefficients; zeros if None.
        world_points   : (N, 3) 3-D reference points from the campus map.
        image_points   : (N, 2) corresponding 2-D pixel detections.

        Returns
        -------
        dict with keys: success, rotation_vec, translation_vec,
                        rotation_mat, inlier_count, latency_ms.
        """
        t0 = time.perf_counter()
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5)

        result: dict[str, Any] = {"success": False, "latency_ms": 0.0}

        if world_points is None or image_points is None:
            logger.warning("pose_estimation_6dof: no 3D-2D correspondences provided.")
            return result

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                world_points.astype(np.float32),
                image_points.astype(np.float32),
                camera_matrix,
                dist_coeffs,
                iterationsCount=200,
                reprojectionError=4.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success and inliers is not None and len(inliers) >= 6:
                R, _ = cv2.Rodrigues(rvec)
                result.update(
                    {
                        "success": True,
                        "rotation_vec": rvec.flatten(),
                        "translation_vec": tvec.flatten(),
                        "rotation_mat": R,
                        "inlier_count": len(inliers),
                    }
                )
                logger.debug("6DoF PnP: %d inliers.", len(inliers))
            else:
                logger.warning("6DoF PnP failed or too few inliers.")

        except Exception as exc:
            logger.exception("pose_estimation_6dof failed: %s", exc)

        result["latency_ms"] = (time.perf_counter() - t0) * 1000
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Visual Positioning System (VPS)
    # ─────────────────────────────────────────────────────────────────────────

    def visual_localization_vps(
        self,
        query_frame: np.ndarray,
        camera_matrix: np.ndarray,
        top_k: int = 5,
    ) -> LocalizationResult:
        """
        Localize the phone within the pre-built campus 3-D map.

        Pipeline
        --------
        1. Extract SuperPoint features from the live query frame.
        2. Coarse retrieval: cosine similarity of global descriptors vs. campus DB.
        3. Re-rank top-K candidates with LightGlue geometric verification.
        4. Run PnP+RANSAC on the best candidate's 2D-3D correspondences.
        5. Package LocalizationResult.

        Parameters
        ----------
        query_frame   : Live BGR frame from the phone camera.
        camera_matrix : 3×3 intrinsic matrix (from camera_calibration).
        top_k         : Number of DB frames to re-rank with LightGlue.
        """
        t0 = time.perf_counter()
        result = LocalizationResult(success=False)

        if not _LIGHTGLUE_AVAILABLE or self._extractor is None:
            logger.error("LightGlue not available — VPS cannot run.")
            return result

        if not self._campus_map:
            logger.error("Campus map DB not loaded — call load_campus_map() first.")
            return result

        try:
            # ── Step 1: Extract query SuperPoint features ─────────────────────
            gray_query = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)
            tensor_query = (
                torch.from_numpy(gray_query).float() / 255.0
            ).unsqueeze(0).unsqueeze(0).to(self.device)  # 1×1×H×W

            with torch.no_grad():
                feats_query = self._extractor.extract(tensor_query)
            q_desc = feats_query["descriptors"][0]  # (N_q, D)
            q_kpts = feats_query["keypoints"][0]    # (N_q, 2)

            # ── Step 2: Coarse retrieval via cosine similarity ─────────────────
            db_descs = self._campus_map["descriptors"]          # (N_frames, D)
            q_global = F.normalize(q_desc.mean(0, keepdim=True), dim=-1)   # (1, D)
            db_global = F.normalize(db_descs.float(), dim=-1)              # (N, D)
            similarities = (q_global @ db_global.T).squeeze(0)            # (N,)
            top_k_ids = (
                similarities.topk(min(top_k, len(similarities))).indices.tolist()
            )
            logger.debug("VPS: top-%d candidates: %s", top_k, top_k_ids)

            # ── Step 3: LightGlue geometric re-ranking ────────────────────────
            best_result: Optional[dict[str, Any]] = None
            best_inliers = 0
            img_hw = torch.tensor(
                [[gray_query.shape[1], gray_query.shape[0]]]
            ).to(self.device)

            for cand_id in top_k_ids:
                try:
                    db_kpts_np = self._campus_map["keypoints"][cand_id]
                    db_kpts = torch.from_numpy(db_kpts_np).float().to(self.device)

                    db_desc_kpt_np = self._campus_map.get(
                        f"kpt_descriptors_{cand_id}", db_kpts_np
                    )
                    db_desc_kpt = torch.from_numpy(
                        db_desc_kpt_np
                    ).float().to(self.device)

                    feats_db = {
                        "keypoints": db_kpts.unsqueeze(0),
                        "descriptors": db_desc_kpt.unsqueeze(0),
                        "image_size": img_hw,
                    }
                    feats_q_fmt = {
                        "keypoints": q_kpts.unsqueeze(0),
                        "descriptors": q_desc.unsqueeze(0),
                        "image_size": img_hw,
                    }

                    with torch.no_grad():
                        matches_out = self._matcher(
                            {"image0": feats_db, "image1": feats_q_fmt}
                        )
                    matches_out = rbd(matches_out)   # remove batch dimension

                    matches = matches_out["matches"]  # (M, 2)
                    valid = matches[:, 0] >= 0
                    n_matches = int(valid.sum())
                    logger.debug("  Candidate %d: %d matches", cand_id, n_matches)

                    if n_matches < 10:
                        continue

                    # ── Step 4: PnP on matched correspondences ────────────────
                    m0 = matches[valid, 0].cpu().numpy()  # indices into DB frame
                    m1 = matches[valid, 1].cpu().numpy()  # indices into query

                    world_pts = self._campus_map.get(f"world_pts_{cand_id}")
                    if world_pts is None:
                        continue

                    pnp = self.pose_estimation_6dof(
                        frame=query_frame,
                        camera_matrix=camera_matrix,
                        world_points=world_pts[m0].astype(np.float32),
                        image_points=q_kpts[m1].cpu().numpy().astype(np.float32),
                    )

                    if pnp["success"] and pnp["inlier_count"] > best_inliers:
                        best_inliers = pnp["inlier_count"]
                        best_result = {
                            "pnp": pnp,
                            "building_id": str(
                                self._campus_map["building_ids"][cand_id]
                            ),
                            "n_matches": n_matches,
                        }

                except Exception as inner_exc:
                    logger.debug("VPS candidate %d failed: %s", cand_id, inner_exc)
                    continue

            # ── Step 5: Package result ────────────────────────────────────────
            if best_result and best_result["pnp"]["success"]:
                pnp = best_result["pnp"]
                R = pnp["rotation_mat"]
                t = pnp["translation_vec"]
                cam_pos = -R.T @ t  # camera position in world coords

                result.success = True
                result.position_xyz = cam_pos
                result.rotation_matrix = R
                result.confidence = min(best_inliers / 50.0, 1.0)
                result.matched_keypoints = best_result["n_matches"]
                result.building_id = best_result["building_id"]
                logger.info(
                    "VPS ✅  building=%s  pos=(%.2f, %.2f, %.2f)  conf=%.2f",
                    result.building_id, *cam_pos, result.confidence,
                )
            else:
                logger.warning("VPS failed — insufficient matches/inliers.")

        except Exception as exc:
            logger.exception("visual_localization_vps failed: %s", exc)

        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Short-term Feature Tracking (LK Optical Flow)
    # ─────────────────────────────────────────────────────────────────────────

    def feature_tracking(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_pts: Optional[np.ndarray] = None,
        max_corners: int = 500,
    ) -> dict[str, Any]:
        """
        Short-term keypoint tracking via Lucas-Kanade optical flow.
        Used between VPS calls to maintain AR anchor stability.

        Parameters
        ----------
        prev_frame  : Previous BGR frame.
        curr_frame  : Current BGR frame.
        prev_pts    : (N, 1, 2) float32 points to track; detected if None.
        max_corners : Maximum Shi-Tomasi corners to detect when prev_pts is None.

        Returns
        -------
        dict with keys: success, tracked_pts, prev_pts, good_mask,
                        flow_magnitude, n_tracked, latency_ms.
        """
        t0 = time.perf_counter()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01
            ),
        )

        if prev_pts is None or len(prev_pts) < 50:
            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=max_corners,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=7,
            )

        if prev_pts is None:
            return {"success": False, "tracked_pts": None, "good_mask": None}

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **lk_params
        )

        good_mask = (status == 1).flatten()
        flow = curr_pts[good_mask] - prev_pts[good_mask]
        flow_mag = (
            float(np.mean(np.linalg.norm(flow, axis=-1)))
            if flow.size else 0.0
        )

        return {
            "success": True,
            "tracked_pts": curr_pts,
            "prev_pts": prev_pts,
            "good_mask": good_mask,
            "flow_magnitude": flow_mag,
            "n_tracked": int(good_mask.sum()),
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # High-level API wrapper
    # ─────────────────────────────────────────────────────────────────────────

    def api_identify_location(
        self,
        frame: np.ndarray,
        camera_matrix: np.ndarray,
        depth_result: Optional[DepthResult] = None,
    ) -> dict[str, Any]:
        """
        High-level wrapper combining VPS results with depth metadata.
        Suitable for direct exposure via a FastAPI route.
        """
        t0 = time.perf_counter()
        loc = self.visual_localization_vps(frame, camera_matrix)

        return {
            "localized": loc.success,
            "building_id": loc.building_id,
            "position_xyz": (
                loc.position_xyz.tolist() if loc.position_xyz is not None else None
            ),
            "confidence": loc.confidence,
            "depth_near": depth_result.near_plane if depth_result else None,
            "depth_far": depth_result.far_plane if depth_result else None,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }