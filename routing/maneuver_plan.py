from __future__ import annotations

import math

from routing.router import Route


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def build_maneuver_plan(route: Route) -> list[dict]:
    plan: list[dict] = []
    prev_lat, prev_lon = route.origin
    for idx, step in enumerate(route.steps):
        bearing_before = _bearing(prev_lat, prev_lon, step.lat, step.lon)
        if idx + 1 < len(route.steps):
            nxt = route.steps[idx + 1]
            bearing_after = _bearing(step.lat, step.lon, nxt.lat, nxt.lon)
        else:
            bearing_after = bearing_before

        plan.append(
            {
                "maneuver_id": idx,
                "instruction": step.instruction,
                "maneuver": step.maneuver or "straight",
                "anchor_lat": step.lat,
                "anchor_lon": step.lon,
                "distance_m": round(step.distance_m, 1),
                "duration_s": round(step.duration_s, 1),
                "bearing_before": round(bearing_before, 1),
                "bearing_after": round(bearing_after, 1),
                "street_name": step.street_name,
                "instruction_priority": "high" if step.maneuver not in {"straight", "depart"} else "normal",
            }
        )
        prev_lat, prev_lon = step.lat, step.lon
    return plan
