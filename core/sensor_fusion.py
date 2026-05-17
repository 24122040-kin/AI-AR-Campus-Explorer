from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from config.settings import settings

if TYPE_CHECKING:
    from core.vio_fusion import VIOPose


@dataclass
class FusedPose:
    lat: float | None = None
    lon: float | None = None
    accuracy_m: float | None = None
    heading_deg: float | None = None
    speed_kmh: float = 0.0
    confidence: float = 0.0
    source: str = "none"
    updated_at: datetime = field(default_factory=datetime.utcnow)
    # Floor detection fields
    floor: int = 1
    floor_confidence: float = 0.0
    floor_method: str = "none"
    # VIO fields (indoor dead-reckoning)
    vio_px: float | None = None        # ENU East metres from session origin
    vio_py: float | None = None        # ENU North metres from session origin
    vio_drift_m: float = 0.0           # accumulated drift since last absolute fix
    vio_source: str = "none"           # "imu" | "flow" | "vpr" | "gps" | "none"

    def as_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "accuracy_m": self.accuracy_m,
            "heading_deg": self.heading_deg,
            "speed_kmh": self.speed_kmh,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "updated_at": self.updated_at.isoformat(),
            "floor": self.floor,
            "floor_confidence": round(self.floor_confidence, 3),
            "floor_method": self.floor_method,
            "vio_px": round(self.vio_px, 3) if self.vio_px is not None else None,
            "vio_py": round(self.vio_py, 3) if self.vio_py is not None else None,
            "vio_drift_m": round(self.vio_drift_m, 3),
            "vio_source": self.vio_source,
        }


class FusionPoseEstimator:
    def __init__(self):
        self._pose = FusedPose()

    def update_gps(
        self,
        lat: float,
        lon: float,
        *,
        accuracy_m: float = 10.0,
        speed_kmh: float = 0.0,
        bearing: float | None = None,
    ) -> FusedPose:
        alpha = min(max(settings.fusion_position_alpha, 0.0), 1.0)
        if self._pose.lat is None or self._pose.lon is None or settings.sensor_fusion_mode == "raw":
            fused_lat = lat
            fused_lon = lon
        else:
            fused_lat = alpha * lat + (1.0 - alpha) * self._pose.lat
            fused_lon = alpha * lon + (1.0 - alpha) * self._pose.lon

        if bearing is not None and bearing >= 0:
            heading = self._blend_heading(self._pose.heading_deg, bearing)
        else:
            heading = self._pose.heading_deg

        confidence = max(0.1, min(1.0, 1.0 - min(accuracy_m, 50.0) / 60.0))
        self._pose = FusedPose(
            lat=fused_lat,
            lon=fused_lon,
            accuracy_m=accuracy_m,
            heading_deg=heading,
            speed_kmh=speed_kmh,
            confidence=confidence,
            source="gps+fusion" if settings.sensor_fusion_mode != "raw" else "gps",
            updated_at=datetime.utcnow(),
        )
        return self._pose

    def update_imu(
        self,
        *,
        compass_heading: float | None = None,
        gyro_heading: float | None = None,
        accel_norm: float | None = None,
        floor: int | None = None,
        floor_confidence: float | None = None,
        floor_method: str | None = None,
    ) -> FusedPose:
        heading = self._pose.heading_deg
        if compass_heading is not None:
            heading = self._blend_heading(heading, compass_heading)
        if gyro_heading is not None:
            heading = self._blend_heading(heading, gyro_heading)

        confidence = self._pose.confidence
        if accel_norm is not None:
            confidence = max(0.05, min(1.0, confidence - abs(accel_norm - 9.81) / 30.0))

        self._pose.heading_deg = heading
        self._pose.confidence = confidence
        self._pose.source = "gps+imu"
        self._pose.updated_at = datetime.utcnow()

        if floor is not None:
            self._pose.floor = max(1, floor)
        if floor_confidence is not None:
            self._pose.floor_confidence = round(floor_confidence, 3)
        if floor_method is not None:
            self._pose.floor_method = floor_method

        return self._pose

    @staticmethod
    def _blend_heading(old_heading: float | None, new_heading: float) -> float:
        if old_heading is None:
            return new_heading % 360.0
        alpha = min(max(settings.fusion_heading_alpha, 0.0), 1.0)
        old_rad = math.radians(old_heading)
        new_rad = math.radians(new_heading)
        x = (1.0 - alpha) * math.cos(old_rad) + alpha * math.cos(new_rad)
        y = (1.0 - alpha) * math.sin(old_rad) + alpha * math.sin(new_rad)
        return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0

    def update_vio(self, vio_pose: "VIOPose") -> FusedPose:
        """
        Merge VIO position estimate into FusedPose.

        When GPS is unavailable (lat/lon is None) and VIO has an origin,
        we synthesize lat/lon from the ENU offset.
        When GPS is available, VIO provides a secondary confidence signal.
        """
        import math as _math

        # Update heading from VIO (math → compass convention: 90 - math_deg)
        vio_heading_deg = (90.0 - _math.degrees(vio_pose.heading_rad)) % 360.0
        self._pose.heading_deg = self._blend_heading(self._pose.heading_deg, vio_heading_deg)

        # Update speed
        self._pose.speed_kmh = vio_pose.speed_ms * 3.6

        # Synthesize lat/lon from VIO when GPS is absent
        if self._pose.lat is None and vio_pose.origin_lat is not None:
            latlon = vio_pose.to_latlon()
            if latlon is not None:
                self._pose.lat = latlon[0]
                self._pose.lon = latlon[1]
                # Accuracy estimate from covariance
                pos_std_m = _math.sqrt(vio_pose.cov_px + vio_pose.cov_py)
                self._pose.accuracy_m = max(0.5, pos_std_m)
                self._pose.confidence = max(0.05, min(0.9, 1.0 - pos_std_m / 10.0))
                self._pose.source = "vio"

        # Always store VIO fields
        self._pose.vio_px = vio_pose.px
        self._pose.vio_py = vio_pose.py
        self._pose.vio_drift_m = vio_pose.drift_m
        self._pose.vio_source = vio_pose.source
        self._pose.updated_at = datetime.utcnow()
        return self._pose
