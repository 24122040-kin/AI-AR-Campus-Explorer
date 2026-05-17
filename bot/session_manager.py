"""
bot/session_manager.py — Multi-session management + live re-routing
Handles:
  - Per-session conversation history with sliding window
  - Live GPS stream → deviation detection → synchronous re-route when possible
  - Session state machine (idle → navigating → arrived → rerouting)
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import Any, Callable, Optional
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

from loguru import logger

from config.settings import settings
from routing.geo_utils import distance_point_to_polyline_m, snap_point_to_polyline
from routing.router import Route


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class NavState(str, Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    REROUTING = "rerouting"
    ARRIVED = "arrived"
    PAUSED = "paused"


@dataclass
class GPSFix:
    lat: float
    lon: float
    accuracy_m: float = 10.0
    timestamp: datetime = field(default_factory=datetime.now)
    speed_kmh: float = 0.0
    bearing: float = 0.0


@dataclass
class NavSession:
    session_id: str
    state: NavState = NavState.IDLE
    current_route: Optional[Route] = None
    current_step_idx: int = 0
    gps_history: deque = field(default_factory=lambda: deque(maxlen=20))
    conversation: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    deviation_count: int = 0
    origin: Optional[tuple[float, float]] = None
    destination: Optional[tuple[float, float]] = None

    on_reroute: Optional[Callable] = field(default=None, repr=False)
    on_arrived: Optional[Callable] = field(default=None, repr=False)

    CONV_MAX_TURNS = 20
    OFF_ROUTE_M = 50
    ARRIVE_M = 30
    SNAP_TRUST_M = 120.0

    def add_message(self, role: str, content) -> None:
        self.conversation.append({"role": role, "content": content})
        if len(self.conversation) > self.CONV_MAX_TURNS * 2:
            self.conversation = self.conversation[-self.CONV_MAX_TURNS * 2 :]
        self.last_active = datetime.now()

    def recent_history(self) -> list[dict]:
        return self.conversation[-self.CONV_MAX_TURNS * 2 :]

    async def update_gps(self, fix: GPSFix) -> dict:
        """
        Process a new GPS fix.
        Returns an event dict: step_advance | arrived | off_route | none
        """
        self.gps_history.append(fix)
        self.last_active = datetime.now()

        if self.state != NavState.NAVIGATING or self.current_route is None:
            return {"type": "none"}

        route = self.current_route
        steps = route.steps
        poly = route.geometry or []

        snap_lat, snap_lon, res_m, seg_i = snap_point_to_polyline(fix.lat, fix.lon, poly)
        d_route = distance_point_to_polyline_m(fix.lat, fix.lon, poly)
        use_lat = snap_lat if res_m < self.SNAP_TRUST_M else fix.lat
        use_lon = snap_lon if res_m < self.SNAP_TRUST_M else fix.lon

        dest = route.destination
        d_dest = _haversine(fix.lat, fix.lon, dest[0], dest[1])
        if d_dest < self.ARRIVE_M:
            self.state = NavState.ARRIVED
            if self.on_arrived:
                asyncio.create_task(self.on_arrived(self.session_id))
            return {"type": "arrived", "distance_m": d_dest}

        if self.current_step_idx < len(steps) - 1:
            next_step = steps[self.current_step_idx]
            d_raw = _haversine(fix.lat, fix.lon, next_step.lat, next_step.lon)
            d_snap = _haversine(use_lat, use_lon, next_step.lat, next_step.lon)
            d_step = min(d_raw, d_snap)
            if d_step < 25:
                self.current_step_idx += 1
                return {
                    "type": "step_advance",
                    "step_idx": self.current_step_idx,
                    "instruction": steps[self.current_step_idx].instruction,
                    "image_paths": steps[self.current_step_idx].image_paths,
                    "map_match": {
                        "lat": snap_lat,
                        "lon": snap_lon,
                        "residual_m": round(res_m, 1),
                        "segment_index": seg_i,
                    },
                }

        if d_route > self.OFF_ROUTE_M:
            self.deviation_count += 1
            if self.deviation_count >= 3:
                self.deviation_count = 0
                self.state = NavState.REROUTING
                return {
                    "type": "off_route",
                    "distance_m": round(d_route, 1),
                    "current_lat": fix.lat,
                    "current_lon": fix.lon,
                    "pending_reroute": True,
                    "map_match": {
                        "lat": snap_lat,
                        "lon": snap_lon,
                        "residual_m": round(res_m, 1),
                        "segment_index": seg_i,
                    },
                }
        else:
            self.deviation_count = max(0, self.deviation_count - 1)

        return {
            "type": "none",
            "d_route_m": round(d_route, 1),
            "map_match": {
                "lat": snap_lat,
                "lon": snap_lon,
                "residual_m": round(res_m, 1),
                "segment_index": seg_i,
            },
        }

    def smoothed_speed(self) -> float:
        fixes = list(self.gps_history)
        if len(fixes) < 2:
            return 0.0
        speeds = [f.speed_kmh for f in fixes if f.speed_kmh > 0]
        return sum(speeds) / len(speeds) if speeds else 0.0

    def eta(self) -> Optional[datetime]:
        if self.current_route is None:
            return None
        remaining_steps = self.current_route.steps[self.current_step_idx :]
        remaining_s = sum(s.duration_s for s in remaining_steps)
        speed = self.smoothed_speed()
        if speed > 5:
            planned_speed = 30.0
            factor = planned_speed / speed
            remaining_s *= max(0.5, min(factor, 2.0))
        return datetime.now() + __import__("datetime").timedelta(seconds=remaining_s)


class SessionManager:
    """Manages all active navigation sessions."""

    def __init__(self):
        self._sessions: dict[str, NavSession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._router: Any = None

    def set_nav_router(self, router: Any) -> None:
        """Called at app startup so live re-routing can call find_route."""
        self._router = router

    async def start(self) -> None:
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    def get_or_create(self, session_id: str) -> NavSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = NavSession(session_id=session_id)
            logger.debug(f"New session: {session_id}")
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[NavSession]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def all_sessions(self) -> list[NavSession]:
        return list(self._sessions.values())

    @staticmethod
    def _serialize_route_brief(route: Route) -> dict:
        cap_geom = 800
        geom = route.geometry
        return {
            "distance_km": round(route.total_distance_m / 1000, 3),
            "duration_min": round(route.total_duration_min, 1),
            "steps": [
                {
                    "instruction": s.instruction,
                    "lat": s.lat,
                    "lon": s.lon,
                    "maneuver": s.maneuver,
                    "distance_m": round(s.distance_m, 1),
                }
                for s in route.steps[:50]
            ],
            "geometry": geom[: min(cap_geom, len(geom))],
            "geometry_total_points": len(geom),
            "analysis": route.analysis,
        }

    async def process_gps_update(
        self, session_id: str, fix: GPSFix, router: Any | None
    ) -> dict:
        """
        Run GPS logic; if off-route with pending_reroute and router+destination exist,
        await a new route and return type \"rerouted\".
        """
        sess = self.get_or_create(session_id)
        event = await sess.update_gps(fix)

        if event.get("type") == "off_route" and event.get("pending_reroute"):
            if router is None or sess.destination is None:
                sess.state = NavState.NAVIGATING
                event = {**event, "reroute_failed": True, "reason": "no_router_or_destination"}
                return event

            new_route: Route | None = None
            try:
                new_route = await router.find_route(
                    fix.lat,
                    fix.lon,
                    sess.destination[0],
                    sess.destination[1],
                    datetime.now(),
                )
            except Exception as e:
                logger.warning(f"Reroute find_route failed: {e}")

            if new_route:
                sess.current_route = new_route
                sess.current_step_idx = 0
                sess.state = NavState.NAVIGATING
                sess.origin = (fix.lat, fix.lon)
                return {
                    "type": "rerouted",
                    "previous_off_route_m": event.get("distance_m"),
                    "route": self._serialize_route_brief(new_route),
                }

            sess.state = NavState.NAVIGATING
            return {**event, "reroute_failed": True}

        return event

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            now = datetime.now()
            expired = [
                sid
                for sid, s in self._sessions.items()
                if (now - s.last_active).total_seconds() > 7200
            ]
            for sid in expired:
                logger.debug(f"Expiring inactive session: {sid}")
                self.delete(sid)

    def stats(self) -> dict:
        states: dict[str, int] = {}
        for s in self._sessions.values():
            states[s.state.value] = states.get(s.state.value, 0) + 1
        return {"total": len(self._sessions), "by_state": states}


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


session_manager = SessionManager()
