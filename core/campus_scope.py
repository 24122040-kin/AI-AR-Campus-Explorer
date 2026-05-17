"""
core/campus_scope.py — Campus boundary scope filter

Provides a single point-in-polygon check for HCMUS Campus 2.
Used by search, geocoding, and the local map to scope results
to the campus area.

Algorithm: Ray-casting (point-in-polygon).
The polygon is defined in settings.campus_polygon as a list of
[lat, lon] vertices. A ~50 m buffer is already baked into the
polygon so edge locations are always captured.
"""
from __future__ import annotations

from config.settings import settings


def point_in_campus(lat: float, lon: float) -> bool:
    """
    Return True if (lat, lon) is inside (or very close to) the campus polygon.
    Uses bounding-box fast-reject first, then ray-casting.
    """
    if not settings.campus_boundary_enabled:
        return True  # scope disabled — accept everything

    # Fast bbox reject
    if not (settings.campus_bbox_lat_min <= lat <= settings.campus_bbox_lat_max and
            settings.campus_bbox_lon_min <= lon <= settings.campus_bbox_lon_max):
        return False

    # Ray-casting algorithm
    poly = settings.campus_polygon
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = poly[i][0], poly[i][1]
        yj, xj = poly[j][0], poly[j][1]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def clamp_to_campus(lat: float, lon: float) -> tuple[float, float]:
    """
    If the point is outside the campus, return the campus centre.
    Used as a fallback for geocoding results that land outside scope.
    """
    if point_in_campus(lat, lon):
        return lat, lon
    return settings.map_default_lat, settings.map_default_lon


def campus_center() -> tuple[float, float]:
    """Return the geographic centre of the campus polygon."""
    poly = settings.campus_polygon
    lat = sum(p[0] for p in poly) / len(poly)
    lon = sum(p[1] for p in poly) / len(poly)
    return lat, lon


def campus_bbox() -> dict:
    """Return the bounding box as a dict for frontend use."""
    return {
        "lat_min": settings.campus_bbox_lat_min,
        "lat_max": settings.campus_bbox_lat_max,
        "lon_min": settings.campus_bbox_lon_min,
        "lon_max": settings.campus_bbox_lon_max,
        "center_lat": settings.map_default_lat,
        "center_lon": settings.map_default_lon,
        "polygon": settings.campus_polygon,
    }
