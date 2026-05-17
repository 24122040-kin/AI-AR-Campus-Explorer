from __future__ import annotations

import math
from typing import Iterable


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def wgs84_to_ecef(lat: float, lon: float, alt: float = 0.0) -> tuple[float, float, float]:
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    sin_lat = math.sin(lat_r)
    cos_lat = math.cos(lat_r)
    cos_lon = math.cos(lon_r)
    sin_lon = math.sin(lon_r)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt) * cos_lat * cos_lon
    y = (n + alt) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return x, y, z


def ecef_to_enu(
    x: float,
    y: float,
    z: float,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
) -> tuple[float, float, float]:
    ref_x, ref_y, ref_z = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    lat_r = math.radians(ref_lat)
    lon_r = math.radians(ref_lon)
    sin_lat = math.sin(lat_r)
    cos_lat = math.cos(lat_r)
    sin_lon = math.sin(lon_r)
    cos_lon = math.cos(lon_r)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def wgs84_to_enu(
    lat: float,
    lon: float,
    alt: float,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
) -> tuple[float, float, float]:
    return ecef_to_enu(*wgs84_to_ecef(lat, lon, alt), ref_lat=ref_lat, ref_lon=ref_lon, ref_alt=ref_alt)


def route_to_local_frame(
    geometry: Iterable[tuple[float, float]],
    ref_lat: float,
    ref_lon: float,
    ref_alt: float = 0.0,
) -> list[dict]:
    local_points: list[dict] = []
    for idx, (lat, lon) in enumerate(geometry):
        east, north, up = wgs84_to_enu(lat, lon, 0.0, ref_lat, ref_lon, ref_alt)
        local_points.append(
            {
                "index": idx,
                "lat": lat,
                "lon": lon,
                "east_m": round(east, 3),
                "north_m": round(north, 3),
                "up_m": round(up, 3),
            }
        )
    return local_points
