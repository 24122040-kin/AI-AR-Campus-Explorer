from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

from bot.realtime_navigator import RealtimeNavigator
from bot.session_manager import GPSFix, session_manager
from config.settings import settings
from core.alert_engine import AlertEngine, alert_registry
from core.floor_detector import FloorDetector
from core.scene_fusion import SceneFusionService, SceneFusionState
from core.sensor_fusion import FusionPoseEstimator
from core.vio_fusion import VIOFusion, vio_registry


@dataclass
class RealtimeSession:
    session_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    latest_frame_path: str | None = None
    latest_frame_meta: dict = field(default_factory=dict)
    latest_gps: dict = field(default_factory=dict)
    latest_sensors: dict = field(default_factory=dict)
    latest_scene_state: dict = field(default_factory=dict)
    latest_instruction: dict = field(default_factory=dict)
    latest_nav_event: dict = field(default_factory=dict)
    latest_floor: dict = field(default_factory=lambda: {"floor": 1, "confidence": 0.0, "method": "none"})
    latest_vio_pose: dict = field(default_factory=dict)
    revision: int = 0
    fusion_state: SceneFusionState = field(default_factory=SceneFusionState)
    pose_estimator: FusionPoseEstimator = field(default_factory=FusionPoseEstimator)
    floor_detector: FloorDetector = field(default_factory=FloorDetector)
    # Pending alerts to push to client on next WebSocket tick
    pending_alerts: list[dict] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()
        self.revision += 1

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "latest_frame_path": self.latest_frame_path,
            "latest_frame_meta": self.latest_frame_meta,
            "latest_gps": self.latest_gps,
            "latest_sensors": self.latest_sensors,
            "latest_scene_state": self.latest_scene_state,
            "latest_instruction": self.latest_instruction,
            "latest_nav_event": self.latest_nav_event,
            "latest_floor": self.latest_floor,
            "latest_vio_pose": self.latest_vio_pose,
            "revision": self.revision,
        }

    def pop_alerts(self) -> list[dict]:
        """Drain and return pending alerts (called by WebSocket loop)."""
        alerts = list(self.pending_alerts)
        self.pending_alerts.clear()
        return alerts


class RealtimeSessionManager:
    def __init__(self, router: Any, vpr_engine: Any = None):
        self._router = router
        self._sessions: dict[str, RealtimeSession] = {}
        self._scene_fusion = SceneFusionService(vpr_engine=vpr_engine)
        self._navigator = RealtimeNavigator()
        self._vpr_engine = vpr_engine

    def get_or_create(self, session_id: str) -> RealtimeSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = RealtimeSession(session_id=session_id)
        return self._sessions[session_id]

    def list_sessions(self) -> list[dict]:
        return [session.as_dict() for session in self._sessions.values()]

    def get_state(self, session_id: str) -> dict:
        return self.get_or_create(session_id).as_dict()

    async def update_sensors(self, session_id: str, payload: dict) -> dict:
        session = self.get_or_create(session_id)
        session.latest_sensors = payload

        # Feed accelerometer into floor detector
        ax = payload.get("accel_x")
        ay = payload.get("accel_y")
        az = payload.get("accel_z")
        if ax is not None and ay is not None and az is not None:
            session.floor_detector.update_accel(float(ax), float(ay), float(az))

        # Feed barometer into floor detector
        pressure_hpa = payload.get("pressure_hpa")
        if pressure_hpa is not None:
            session.floor_detector.update_pressure(float(pressure_hpa))

        # Get floor estimate and merge into fused pose
        floor_info = session.floor_detector.get_floor()
        session.latest_floor = floor_info

        pose = session.pose_estimator.update_imu(
            compass_heading=payload.get("compass_heading"),
            gyro_heading=payload.get("gyro_heading"),
            accel_norm=payload.get("accel_norm"),
            floor=floor_info["floor"],
            floor_confidence=floor_info["confidence"],
            floor_method=floor_info["method"],
        )
        session.touch()

        # Evaluate alerts (floor change, VIO drift, battery)
        new_alerts = alert_registry.get_or_create(session_id).evaluate(session.as_dict())
        if new_alerts:
            session.pending_alerts.extend(a.as_dict() for a in new_alerts)

        return {
            "ok": True,
            "session_id": session_id,
            "fused_pose": pose.as_dict(),
            "floor": floor_info,
            "alerts": [a.as_dict() for a in new_alerts],
            "revision": session.revision,
        }

    async def update_floor(self, session_id: str, pressure_hpa: float | None, accel: dict | None) -> dict:
        """
        Dedicated floor-update endpoint — accepts barometer + accel payload.
        Returns the current floor estimate.
        """
        session = self.get_or_create(session_id)

        if pressure_hpa is not None:
            session.floor_detector.update_pressure(pressure_hpa)

        if accel:
            ax = accel.get("x")
            ay = accel.get("y")
            az = accel.get("z")
            if ax is not None and ay is not None and az is not None:
                session.floor_detector.update_accel(float(ax), float(ay), float(az))

        floor_info = session.floor_detector.get_floor()
        session.latest_floor = floor_info

        # Propagate into fused pose
        session.pose_estimator.update_imu(
            floor=floor_info["floor"],
            floor_confidence=floor_info["confidence"],
            floor_method=floor_info["method"],
        )
        session.touch()
        return {"ok": True, "session_id": session_id, "floor": floor_info, "revision": session.revision}

    async def calibrate_floor(self, session_id: str, floor: int) -> dict:
        """Manually set the current floor (resets barometric baseline)."""
        session = self.get_or_create(session_id)
        session.floor_detector.calibrate_floor(floor)
        floor_info = session.floor_detector.get_floor()
        session.latest_floor = floor_info
        session.pose_estimator.update_imu(
            floor=floor_info["floor"],
            floor_confidence=floor_info["confidence"],
            floor_method=floor_info["method"],
        )
        session.touch()
        return {"ok": True, "session_id": session_id, "floor": floor_info, "revision": session.revision}

    # ── VIO methods ───────────────────────────────────────────────────────────

    async def vio_update_imu(self, session_id: str, payload: dict) -> dict:
        """
        High-rate IMU update for VIO dead-reckoning.
        Merges VIO pose into FusedPose and triggers VPR re-localization
        if drift exceeds threshold.
        """
        session = self.get_or_create(session_id)
        vio = vio_registry.get_or_create(session_id)

        vio_pose = vio.update_imu(
            ax=float(payload.get("ax", 0.0)),
            ay=float(payload.get("ay", 0.0)),
            az=float(payload.get("az", 0.0)),
            gyro_z_rad_s=float(payload.get("gyro_z", 0.0)),
            compass_deg=payload.get("compass_deg"),
            dt_s=float(payload.get("dt_s", 0.05)),
        )

        # Merge into fused pose
        fused = session.pose_estimator.update_vio(vio_pose)
        session.latest_vio_pose = vio_pose.as_dict()
        session.touch()

        result = {
            "ok": True,
            "session_id": session_id,
            "vio_pose": vio_pose.as_dict(),
            "fused_pose": fused.as_dict(),
            "needs_relocalization": vio.needs_relocalization,
            "revision": session.revision,
        }

        # Auto-trigger VPR re-localization when drift is too high
        # Only if we have a recent frame to query
        if vio.needs_relocalization and session.latest_frame_path:
            result["vpr_requested"] = True
            
            # Try VPR relocalization if engine available
            if self._vpr_engine is not None:
                from pathlib import Path
                vpr_reloc = await self.vio_try_vpr_relocalize(
                    session_id,
                    Path(session.latest_frame_path),
                    self._vpr_engine,
                )
                
                if vpr_reloc:
                    result["vpr_relocalized"] = True
                    result["vpr_match"] = vpr_reloc.get("vpr_match")
                    result["vio_pose"] = vpr_reloc.get("vio_pose", vio_pose.as_dict())
                    session.latest_vio_pose = result["vio_pose"]

        return result

    async def vio_update_flow(
        self,
        session_id: str,
        flow_x_px: float,
        flow_y_px: float,
        dt_s: float,
    ) -> dict:
        """Optical flow correction from JS client."""
        session = self.get_or_create(session_id)
        vio = vio_registry.get_or_create(session_id)

        vio_pose = vio.update_optical_flow(flow_x_px, flow_y_px, dt_s)
        fused = session.pose_estimator.update_vio(vio_pose)
        session.latest_vio_pose = vio_pose.as_dict()
        session.touch()

        return {
            "ok": True,
            "session_id": session_id,
            "vio_pose": vio_pose.as_dict(),
            "fused_pose": fused.as_dict(),
            "revision": session.revision,
        }

    async def vio_relocalize(
        self,
        session_id: str,
        lat: float,
        lon: float,
        heading_deg: float | None,
        accuracy_m: float,
        source: str = "gps",
    ) -> dict:
        """
        Absolute position reset from GPS or VPR match.
        Also updates the main FusedPose with the corrected position.
        """
        session = self.get_or_create(session_id)
        vio = vio_registry.get_or_create(session_id)

        vio_pose = vio.relocalize(lat, lon, heading_deg, accuracy_m)

        # Also update the GPS-based fused pose
        fused = session.pose_estimator.update_gps(
            lat, lon,
            accuracy_m=accuracy_m,
            bearing=heading_deg,
        )
        fused = session.pose_estimator.update_vio(vio_pose)
        session.latest_vio_pose = vio_pose.as_dict()
        session.touch()

        return {
            "ok": True,
            "session_id": session_id,
            "vio_pose": vio_pose.as_dict(),
            "fused_pose": fused.as_dict(),
            "source": source,
            "revision": session.revision,
        }

    async def vio_get_pose(self, session_id: str) -> dict:
        """Return current VIO pose without updating."""
        vio = vio_registry.get(session_id)
        if vio is None:
            return {"ok": True, "session_id": session_id, "vio_pose": None}
        return {
            "ok": True,
            "session_id": session_id,
            "vio_pose": vio.get_pose().as_dict(),
        }

    async def vio_try_vpr_relocalize(
        self,
        session_id: str,
        frame_path: Path,
        vpr_engine: Any,
    ) -> dict | None:
        """
        Run VPR on the latest frame and relocalize if a confident match is found.
        Called automatically when drift > VPR_DRIFT_TRIGGER_M.
        Returns relocalization result dict or None if VPR unavailable/no match.
        """
        if vpr_engine is None:
            return None
        try:
            from PIL import Image
            from loguru import logger

            vio = vio_registry.get(session_id)
            if vio is None:
                return None

            # Get current approximate position for VPR proximity re-ranking
            pose = vio.get_pose()
            latlon = pose.to_latlon()
            query_lat = latlon[0] if latlon else None
            query_lon = latlon[1] if latlon else None

            logger.info(f"VPR relocalization triggered for session {session_id}, VIO drift: {pose.drift_m:.2f}m")

            img = Image.open(frame_path).convert("RGB")
            matches = vpr_engine.query(img, top_k=5, query_lat=query_lat, query_lon=query_lon)

            if not matches:
                logger.warning(f"VPR found no matches for session {session_id}")
                return None

            best = matches[0]
            logger.info(f"VPR best match: {best.location_name} (score={best.score:.3f}, dist={best.distance_m:.1f}m)")

            # Only relocalize if VPR confidence is high enough
            min_score = getattr(settings, 'vio_vpr_min_score', 0.65)
            if best.score < min_score:
                logger.warning(f"VPR score {best.score:.3f} below threshold {min_score}")
                return None

            # Additional validation: check if match is within reasonable distance
            # If VIO drift is 3m but VPR match is 50m away, it's probably wrong
            max_reasonable_dist = max(10.0, pose.drift_m * 3.0)
            if best.distance_m > max_reasonable_dist:
                logger.warning(f"VPR match too far: {best.distance_m:.1f}m > {max_reasonable_dist:.1f}m")
                # Try second-best match
                if len(matches) > 1:
                    second = matches[1]
                    if second.score >= min_score and second.distance_m <= max_reasonable_dist:
                        logger.info(f"Using second-best match: {second.location_name}")
                        best = second
                    else:
                        return None
                else:
                    return None

            # Relocalize with VPR match
            result = await self.vio_relocalize(
                session_id,
                best.lat, best.lon,
                heading_deg=None,
                accuracy_m=max(1.5, best.distance_m * 0.15),  # More optimistic accuracy
                source="vpr",
            )

            # Add VPR match info to result
            if result:
                result["vpr_match"] = {
                    "location_name": best.location_name,
                    "location_id": best.location_id,
                    "score": round(best.score, 3),
                    "distance_m": round(best.distance_m, 1),
                    "lat": best.lat,
                    "lon": best.lon,
                }
                logger.info(f"VPR relocalization successful: drift reset from {pose.drift_m:.2f}m to 0m")

            return result

        except Exception as e:
            from loguru import logger
            logger.error(f"VPR relocalization failed: {e}")
            return None

    async def ingest_frame(
        self,
        session_id: str,
        frame_path: Path,
        *,
        lat: float | None = None,
        lon: float | None = None,
        accuracy_m: float = 10.0,
        speed_kmh: float = 0.0,
        bearing: float = 0.0,
    ) -> dict:
        session = self.get_or_create(session_id)
        session.latest_frame_path = str(frame_path)
        session.latest_frame_meta = {"received_at": datetime.utcnow().isoformat(), "name": frame_path.name}

        nav_event = {"type": "none"}
        gps_payload = session.latest_gps
        fused_pose = session.pose_estimator._pose.as_dict()
        if lat is not None and lon is not None:
            gps_fix = GPSFix(lat=lat, lon=lon, accuracy_m=accuracy_m, speed_kmh=speed_kmh, bearing=bearing)
            nav_event = await session_manager.process_gps_update(session_id, gps_fix, self._router)
            fused_pose = session.pose_estimator.update_gps(
                lat,
                lon,
                accuracy_m=accuracy_m,
                speed_kmh=speed_kmh,
                bearing=bearing,
            ).as_dict()
            gps_payload = {
                "lat": lat,
                "lon": lon,
                "accuracy_m": accuracy_m,
                "speed_kmh": speed_kmh,
                "bearing": bearing,
            }
            session.latest_gps = gps_payload

        nav_session = session_manager.get_or_create(session_id)
        scene_state = await self._scene_fusion.build_scene_state(
            frame_path,
            gps=gps_payload,
            nav_event=nav_event,
            nav_session=nav_session,
            fused_pose=fused_pose,
            fusion_state=session.fusion_state,
        )
        instruction = self._navigator.build_instruction(scene_state)

        session.latest_nav_event = nav_event
        session.latest_scene_state = scene_state
        session.latest_instruction = instruction
        session.touch()

        # Auto VPR re-localization when VIO drift is too high
        vio = vio_registry.get(session_id)
        vpr_reloc = None
        vpr_triggered = False
        
        if vio is not None and self._vpr_engine is not None:
            # Check if VIO needs relocalization (drift > 2m)
            if vio.needs_relocalization:
                vpr_triggered = True
                vpr_reloc = await self.vio_try_vpr_relocalize(
                    session_id, frame_path, self._vpr_engine
                )
                
                # If VPR relocalization succeeded, update VIO pose in response
                if vpr_reloc:
                    session.latest_vio_pose = vpr_reloc.get("vio_pose", {})
                    # Add alert for successful relocalization
                    from core.alert_engine import Alert
                    reloc_alert = Alert(
                        alert_id=f"vpr_reloc_{session_id}",
                        severity="info",
                        category="vio",
                        title="VIO Relocalized",
                        message=f"Position corrected via VPR: {vpr_reloc.get('vpr_match', {}).get('location_name', 'unknown')}",
                        timestamp=datetime.utcnow(),
                        metadata=vpr_reloc.get("vpr_match", {}),
                    )
                    session.pending_alerts.append(reloc_alert.as_dict())

        # Evaluate proactive alerts
        new_alerts = alert_registry.get_or_create(session_id).evaluate(session.as_dict())
        if new_alerts:
            session.pending_alerts.extend(a.as_dict() for a in new_alerts)

        return {
            "ok": True,
            "session_id": session_id,
            "scene_state": scene_state,
            "instruction": instruction,
            "nav_event": nav_event,
            "vio_pose": session.latest_vio_pose or None,
            "vpr_triggered": vpr_triggered,
            "vpr_relocalized": vpr_reloc is not None,
            "vpr_match": vpr_reloc.get("vpr_match") if vpr_reloc else None,
            "alerts": [a.as_dict() for a in new_alerts],
            "revision": session.revision,
        }

    def build_frame_path(self, suffix: str) -> Path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return settings.realtime_frames_dir / f"rt_{stamp}_{uuid.uuid4().hex[:8]}{suffix}"
