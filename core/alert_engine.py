"""
core/alert_engine.py — Proactive Alert Engine

Monitors session state after each frame/sensor update and emits
structured alert objects when conditions are met.

Alert schema:
    {
        "type":     "turn_soon" | "off_route" | "stairs_detected" |
                    "floor_change" | "arrived" | "low_battery" | "vio_drift",
        "message":  str,          # Vietnamese, ready for TTS
        "urgency":  "high" | "normal" | "low",
        "distance_m": float | None,
        "suppress_s": float,      # minimum seconds before re-firing same type
        "ts":       ISO timestamp
    }

Integration:
    AlertEngine.evaluate(session_state) → list[Alert]
    Called inside RealtimeSessionManager.ingest_frame() and update_sensors().
    Alerts are pushed to the client via the /ws/realtime WebSocket as
    {"type": "alert", "alert": {...}}.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# ── Types ─────────────────────────────────────────────────────────────────────

AlertType = Literal[
    "turn_soon",
    "turn_now",
    "off_route",
    "stairs_detected",
    "elevator_detected",
    "floor_change",
    "arrived",
    "low_battery",
    "vio_drift",
    "rerouting",
]


@dataclass
class Alert:
    type: AlertType
    message: str
    urgency: Literal["high", "normal", "low"] = "normal"
    distance_m: float | None = None
    suppress_s: float = 10.0   # cooldown before same alert fires again
    ts: datetime = field(default_factory=datetime.utcnow)

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "message": self.message,
            "urgency": self.urgency,
            "distance_m": self.distance_m,
            "suppress_s": self.suppress_s,
            "ts": self.ts.isoformat(),
        }


# ── Thresholds ────────────────────────────────────────────────────────────────

TURN_SOON_M   = 80.0    # announce turn at this distance
TURN_NOW_M    = 20.0    # urgent re-announce at this distance
OFF_ROUTE_SUPPRESS_S  = 30.0
TURN_SOON_SUPPRESS_S  = 15.0
TURN_NOW_SUPPRESS_S   = 8.0
FLOOR_CHANGE_SUPPRESS_S = 5.0
VIO_DRIFT_SUPPRESS_S  = 20.0
LOW_BATTERY_SUPPRESS_S = 60.0
LOW_BATTERY_THRESHOLD  = 20   # percent


# ── Alert Engine ──────────────────────────────────────────────────────────────

class AlertEngine:
    """
    Stateful per-session alert engine.
    Tracks last-fired timestamps to avoid spamming the same alert.
    """

    def __init__(self) -> None:
        # Maps alert_type → last fired datetime
        self._last_fired: dict[str, datetime] = {}
        # Track previous floor to detect changes
        self._prev_floor: int = 1
        # Track previous nav event type
        self._prev_nav_event: str = "none"

    def evaluate(self, session_state: dict) -> list[Alert]:
        """
        Evaluate the current session state and return any new alerts.
        session_state is the dict returned by RealtimeSession.as_dict().
        """
        alerts: list[Alert] = []
        now = datetime.utcnow()

        instruction  = session_state.get("latest_instruction", {})
        nav_event    = session_state.get("latest_nav_event", {})
        scene_state  = session_state.get("latest_scene_state", {})
        floor_info   = session_state.get("latest_floor", {})
        vio_pose     = session_state.get("latest_vio_pose", {})
        sensors      = session_state.get("latest_sensors", {})

        # ── 1. Turn soon / turn now ───────────────────────────────────────────
        route_progress = scene_state.get("route_progress", {})
        dist_m = route_progress.get("distance_to_next_turn_m")
        maneuver = route_progress.get("next_maneuver", "")

        if dist_m is not None and maneuver:
            if dist_m <= TURN_NOW_M:
                if self._can_fire("turn_now", TURN_NOW_SUPPRESS_S, now):
                    alerts.append(Alert(
                        type="turn_now",
                        message=_vn_turn_message(maneuver, dist_m, urgent=True),
                        urgency="high",
                        distance_m=dist_m,
                        suppress_s=TURN_NOW_SUPPRESS_S,
                    ))
            elif dist_m <= TURN_SOON_M:
                if self._can_fire("turn_soon", TURN_SOON_SUPPRESS_S, now):
                    alerts.append(Alert(
                        type="turn_soon",
                        message=_vn_turn_message(maneuver, dist_m, urgent=False),
                        urgency="normal",
                        distance_m=dist_m,
                        suppress_s=TURN_SOON_SUPPRESS_S,
                    ))

        # ── 2. Off-route ──────────────────────────────────────────────────────
        if route_progress.get("off_route"):
            if self._can_fire("off_route", OFF_ROUTE_SUPPRESS_S, now):
                alerts.append(Alert(
                    type="off_route",
                    message="Bạn đang lệch khỏi tuyến đường. Đang tìm lại đường.",
                    urgency="high",
                    suppress_s=OFF_ROUTE_SUPPRESS_S,
                ))

        # ── 3. Rerouting event ────────────────────────────────────────────────
        nav_type = nav_event.get("type", "none")
        if nav_type == "rerouted" and self._prev_nav_event != "rerouted":
            if self._can_fire("rerouting", 15.0, now):
                alerts.append(Alert(
                    type="rerouting",
                    message="Đã tìm được tuyến đường mới. Hãy làm theo hướng dẫn.",
                    urgency="normal",
                    suppress_s=15.0,
                ))

        # ── 4. Arrived ────────────────────────────────────────────────────────
        if nav_type == "arrived" and self._prev_nav_event != "arrived":
            alerts.append(Alert(
                type="arrived",
                message="Bạn đã đến nơi. Chúc mừng!",
                urgency="normal",
                suppress_s=60.0,
            ))

        self._prev_nav_event = nav_type

        # ── 5. Stairs / elevator detected in scene ────────────────────────────
        visual = scene_state.get("visual", {})
        landmarks = [l.get("label", "").lower() for l in visual.get("landmarks", [])]
        if any("stair" in l or "cầu thang" in l or "stairs" in l for l in landmarks):
            if self._can_fire("stairs_detected", 20.0, now):
                alerts.append(Alert(
                    type="stairs_detected",
                    message="Phát hiện cầu thang phía trước.",
                    urgency="normal",
                    suppress_s=20.0,
                ))
        if any("elevator" in l or "thang máy" in l or "lift" in l for l in landmarks):
            if self._can_fire("elevator_detected", 20.0, now):
                alerts.append(Alert(
                    type="elevator_detected",
                    message="Phát hiện thang máy phía trước.",
                    urgency="low",
                    suppress_s=20.0,
                ))

        # ── 6. Floor change ───────────────────────────────────────────────────
        current_floor = floor_info.get("floor", 1)
        floor_conf    = floor_info.get("confidence", 0.0)
        if (current_floor != self._prev_floor
                and floor_conf >= 0.5
                and self._can_fire("floor_change", FLOOR_CHANGE_SUPPRESS_S, now)):
            direction = "lên" if current_floor > self._prev_floor else "xuống"
            alerts.append(Alert(
                type="floor_change",
                message=f"Bạn đang đi {direction} — Tầng {current_floor}.",
                urgency="normal",
                suppress_s=FLOOR_CHANGE_SUPPRESS_S,
            ))
        self._prev_floor = current_floor

        # ── 7. VIO drift warning ──────────────────────────────────────────────
        if vio_pose:
            drift = vio_pose.get("drift_m", 0.0)
            if drift > 2.0 and self._can_fire("vio_drift", VIO_DRIFT_SUPPRESS_S, now):
                alerts.append(Alert(
                    type="vio_drift",
                    message="Vị trí trong nhà có thể không chính xác. Đang hiệu chỉnh lại.",
                    urgency="low",
                    suppress_s=VIO_DRIFT_SUPPRESS_S,
                ))

        # ── 8. Low battery ────────────────────────────────────────────────────
        battery_level = sensors.get("battery_level")
        if (battery_level is not None
                and battery_level <= LOW_BATTERY_THRESHOLD
                and self._can_fire("low_battery", LOW_BATTERY_SUPPRESS_S, now)):
            alerts.append(Alert(
                type="low_battery",
                message=f"Pin còn {battery_level}%. Hãy sạc điện thoại.",
                urgency="low",
                suppress_s=LOW_BATTERY_SUPPRESS_S,
            ))

        # Record fire times
        for alert in alerts:
            self._last_fired[alert.type] = now

        return alerts

    def reset(self) -> None:
        """Clear all cooldown state (e.g. when session resets)."""
        self._last_fired.clear()
        self._prev_floor = 1
        self._prev_nav_event = "none"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _can_fire(self, alert_type: str, suppress_s: float, now: datetime) -> bool:
        last = self._last_fired.get(alert_type)
        if last is None:
            return True
        elapsed = (now - last).total_seconds()
        return elapsed >= suppress_s


# ── Per-session registry ──────────────────────────────────────────────────────

class AlertEngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, AlertEngine] = {}

    def get_or_create(self, session_id: str) -> AlertEngine:
        if session_id not in self._engines:
            self._engines[session_id] = AlertEngine()
        return self._engines[session_id]

    def delete(self, session_id: str) -> None:
        self._engines.pop(session_id, None)


alert_registry = AlertEngineRegistry()


# ── Vietnamese instruction helpers ────────────────────────────────────────────

_MANEUVER_VN: dict[str, str] = {
    "turn_left":        "Rẽ trái",
    "turn_right":       "Rẽ phải",
    "slight_left":      "Đi nhẹ sang trái",
    "slight_right":     "Đi nhẹ sang phải",
    "sharp_left":       "Rẽ gắt sang trái",
    "sharp_right":      "Rẽ gắt sang phải",
    "u_turn":           "Quay đầu xe",
    "straight":         "Đi thẳng",
    "arrive":           "Đã đến nơi",
    "depart":           "Xuất phát",
    "stairs":           "Đi lên cầu thang",
    "elevator":         "Đi thang máy",
}


def _vn_turn_message(maneuver: str, dist_m: float, urgent: bool) -> str:
    action = _MANEUVER_VN.get(maneuver, maneuver)
    dist_str = f"{int(dist_m)} mét"
    if urgent:
        return f"{action} ngay bây giờ!"
    return f"{action} sau {dist_str}."
