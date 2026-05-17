"""
routing/geo_utils.py — Polyline distance & map-matching helpers (local equirectangular).
Safe for small segments (<~50 km); avoids brittle vertex-only heuristics.
"""
from __future__ import annotations

import math
from typing import Optional

R_EARTH_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R_EARTH_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _to_local_m(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """Equirectangular metres with origin at (lat0, lon0)."""
    cos0 = max(0.2, math.cos(math.radians(lat0)))
    x = R_EARTH_M * math.radians(lon - lon0) * cos0
    y = R_EARTH_M * math.radians(lat - lat0)
    return x, y


def distance_point_to_segment_m(
    lat: float, lon: float,
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Shortest distance from point P to segment A–B (metres), planar approx at mid-latitude."""
    lat0 = (lat1 + lat2 + lat) / 3.0
    px, py = _to_local_m(lat0, lon1, lat, lon)  # use lon1 as ref lon
    ax, ay = _to_local_m(lat0, lon1, lat1, lon1)
    bx, by = _to_local_m(lat0, lon1, lat2, lon2)
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-6:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def distance_point_to_polyline_m(lat: float, lon: float, polyline: list[tuple[float, float]]) -> float:
    """Minimum distance from P to any segment of polyline (vertices connected)."""
    if not polyline:
        return float("inf")
    if len(polyline) == 1:
        return haversine_m(lat, lon, polyline[0][0], polyline[0][1])
    best = float("inf")
    for i in range(len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]
        d = distance_point_to_segment_m(lat, lon, a[0], a[1], b[0], b[1])
        if d < best:
            best = d
    return best


def snap_point_to_polyline(
    lat: float, lon: float, polyline: list[tuple[float, float]],
) -> tuple[float, float, float, int]:
    """
    Project (lat, lon) onto closest point on the polyline.
    Returns (snap_lat, snap_lon, residual_distance_m, segment_index).
    """
    if not polyline:
        return lat, lon, float("inf"), -1
    if len(polyline) == 1:
        d = haversine_m(lat, lon, polyline[0][0], polyline[0][1])
        return polyline[0][0], polyline[0][1], d, 0

    best_d = float("inf")
    best_lat = lat
    best_lon = lon
    best_seg = 0

    for i in range(len(polyline) - 1):
        lat1, lon1 = polyline[i]
        lat2, lon2 = polyline[i + 1]
        lat0 = (lat1 + lat2 + lat) / 3.0
        px, py = _to_local_m(lat0, lon1, lat, lon)
        ax, ay = _to_local_m(lat0, lon1, lat1, lon1)
        bx, by = _to_local_m(lat0, lon1, lat2, lon2)
        abx, aby = bx - ax, by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-6:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
        cx, cy = ax + t * abx, ay + t * aby
        d = math.hypot(px - cx, py - cy)
        if d < best_d:
            best_d = d
            best_seg = i
            # invert local metres → lat/lon
            snap_lat = lat1 + t * (lat2 - lat1)
            snap_lon = lon1 + t * (lon2 - lon1)
            best_lat, best_lon = snap_lat, snap_lon

    return best_lat, best_lon, best_d, best_seg


def min_haversine_to_points(lat: float, lon: float, polyline: list[tuple[float, float]]) -> float:
    return min(haversine_m(lat, lon, p[0], p[1]) for p in polyline) if polyline else float("inf")


def distance_for_navigation(
    raw_lat: float, raw_lon: float,
    polyline: list[tuple[float, float]],
    use_snap_for_step_advance: bool = True,
) -> tuple[float, Optional[tuple[float, float]]]:
    """
    Cross-track distance to polyline + optional snapped position for stable step logic.
    Returns (cross_track_m, (snap_lat, snap_lon) or None if empty polyline).
    """
    if not polyline:
        return float("inf"), None
    slat, slon, res_m, _ = snap_point_to_polyline(raw_lat, raw_lon, polyline)
    return res_m, (slat, slon) if use_snap_for_step_advance else (res_m, None)
