"""
core/traffic_analyzer.py — Real-time traffic heuristic engine
Combines:
  - Crowd-sourced observations (DB)
  - Historical peak-hour patterns
  - Time-of-day/weekday learning
  - Per-segment speed estimation
  - Isochrone generation (reachable area in N minutes)
"""
from __future__ import annotations
import math
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from loguru import logger

from config.settings import settings
from core.database import db


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrafficSegment:
    lat: float
    lon: float
    speed_kmh: float
    congestion: float        # 0.0 = free, 1.0 = gridlock
    hour: int
    weekday: int
    sample_count: int = 1


@dataclass
class CongestionGrid:
    """Spatial grid of average congestion factors."""
    cell_size_deg: float = 0.002   # ~220m per cell
    grid: dict = field(default_factory=lambda: defaultdict(list))

    def _key(self, lat: float, lon: float) -> tuple:
        return (
            round(lat / self.cell_size_deg),
            round(lon / self.cell_size_deg),
        )

    def add(self, lat: float, lon: float, congestion: float) -> None:
        self.grid[self._key(lat, lon)].append(congestion)

    def get(self, lat: float, lon: float) -> float:
        """Return average congestion for the grid cell, or 0.0 if unknown."""
        vals = self.grid.get(self._key(lat, lon), [])
        return float(np.mean(vals)) if vals else 0.0

    def to_heatmap_data(self) -> list[dict]:
        """Return [{lat, lon, intensity}] for Folium HeatMap plugin."""
        result = []
        for (row, col), vals in self.grid.items():
            lat = row * self.cell_size_deg
            lon = col * self.cell_size_deg
            intensity = float(np.mean(vals))
            result.append({"lat": lat, "lon": lon, "intensity": intensity})
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Traffic Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class TrafficAnalyzer:
    """
    Learns and predicts traffic conditions from crowd-sourced data.
    Provides per-hour congestion curves, spatial heatmaps,
    and travel time multipliers for the routing engine.
    """

    # Default Vietnamese traffic patterns if no data collected yet
    # (based on typical HCMC traffic rhythms)
    _DEFAULT_HOURLY = {
        0: 0.05, 1: 0.03, 2: 0.02, 3: 0.02, 4: 0.05,
        5: 0.15, 6: 0.50, 7: 0.85, 8: 0.75, 9: 0.45,
        10: 0.35, 11: 0.55, 12: 0.60, 13: 0.45, 14: 0.35,
        15: 0.45, 16: 0.70, 17: 0.90, 18: 0.85, 19: 0.65,
        20: 0.50, 21: 0.40, 22: 0.30, 23: 0.15,
    }

    # Weekday multiplier: 0=Mon, 6=Sun
    _WEEKDAY_MULT = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.1, 5: 0.8, 6: 0.6}

    def __init__(self):
        self._hourly_cache: dict[tuple, float] = {}   # (hour, weekday) → congestion
        self._grid = CongestionGrid()
        self._last_refresh = datetime.min

    async def refresh(self, force: bool = False) -> None:
        """Reload congestion data from DB into memory cache."""
        age = (datetime.now() - self._last_refresh).total_seconds()
        if not force and age < 300:   # refresh every 5 minutes
            return

        rows = await db.fetchall(
            """SELECT hour, weekday, AVG(congestion) AS avg_c, COUNT(*) AS n
               FROM traffic_observations
               GROUP BY hour, weekday"""
        )
        for r in rows:
            self._hourly_cache[(r["hour"], r["weekday"])] = r["avg_c"] or 0.0

        # Spatial grid (last 7 days)
        since = (datetime.now() - timedelta(days=7)).isoformat()
        spatial = await db.fetchall(
            "SELECT lat, lon, congestion FROM traffic_observations WHERE observed_at > ?",
            (since,),
        )
        self._grid = CongestionGrid()
        for s in spatial:
            if s["congestion"] is not None:
                self._grid.add(s["lat"], s["lon"], s["congestion"])

        self._last_refresh = datetime.now()
        logger.debug(f"Traffic cache refreshed: {len(rows)} hour-day pairs, {len(spatial)} spatial obs")

    def congestion_at(
        self,
        hour: int,
        weekday: int,
        lat: float | None = None,
        lon: float | None = None,
    ) -> float:
        """
        Return congestion level [0.0, 1.0] for given time and optional location.
        Blends DB data with default pattern.
        """
        db_val = self._hourly_cache.get((hour, weekday))

        if db_val is not None:
            base = db_val
        else:
            # Interpolate from default pattern
            base = self._DEFAULT_HOURLY[hour] * self._WEEKDAY_MULT.get(weekday, 1.0)

        # Spatial adjustment
        if lat is not None and lon is not None:
            spatial = self._grid.get(lat, lon)
            if spatial > 0:
                base = 0.6 * base + 0.4 * spatial

        return min(max(base, 0.0), 1.0)

    def speed_multiplier(self, congestion: float) -> float:
        """Convert congestion [0,1] to speed multiplier [0.2, 1.0]."""
        # Greenshields traffic flow model approximation
        return max(0.2, 1.0 - 0.8 * congestion)

    def travel_time_factor(self, congestion: float) -> float:
        """Multiplicative factor on free-flow travel time."""
        return 1.0 / self.speed_multiplier(congestion)

    def full_day_curve(self, weekday: int = 1) -> list[dict]:
        """Return 24-hour congestion profile for charting."""
        return [
            {
                "hour": h,
                "congestion": self.congestion_at(h, weekday),
                "label": f"{h:02d}:00",
                "status": self._status_label(self.congestion_at(h, weekday)),
            }
            for h in range(24)
        ]

    @staticmethod
    def _status_label(c: float) -> str:
        if c < 0.3:  return "thông thoáng"
        if c < 0.55: return "bình thường"
        if c < 0.75: return "hơi đông"
        if c < 0.9:  return "tắc nghẽn"
        return "kẹt xe nặng"

    def best_departure_window(
        self,
        target_hour: int,
        tolerance_hours: int = 2,
        weekday: int = 1,
    ) -> dict:
        """
        Find the least congested 30-min slot within ±tolerance hours of target.
        Returns the recommended departure time and congestion estimate.
        """
        best_h = target_hour
        best_c = self.congestion_at(target_hour, weekday)

        for delta in range(-tolerance_hours * 2, tolerance_hours * 2 + 1):
            h = (target_hour + delta * 0.5)
            h_int = int(h) % 24
            c = self.congestion_at(h_int, weekday)
            if c < best_c:
                best_c = c
                best_h = h_int

        return {
            "recommended_hour": best_h,
            "congestion": best_c,
            "status": self._status_label(best_c),
            "save_minutes": max(0, int((self.congestion_at(target_hour, weekday) - best_c) * 20)),
        }

    def heatmap_data(self) -> list[dict]:
        return self._grid.to_heatmap_data()


# ─────────────────────────────────────────────────────────────────────────────
# Isochrone generator
# ─────────────────────────────────────────────────────────────────────────────

class IsochroneGenerator:
    """
    Generate reachable area polygons from a point within N minutes.
    Uses Dijkstra on the OSM graph with traffic-weighted edge costs.
    """

    def __init__(self, osm_graph, analyzer: TrafficAnalyzer):
        self.osm = osm_graph
        self.analyzer = analyzer

    def generate(
        self,
        lat: float,
        lon: float,
        minutes: list[int],
        depart_time: datetime | None = None,
    ) -> dict[int, list[tuple[float, float]]]:
        """
        Returns {minutes: [(lat,lon),...]} convex-hull-style polygon for each minute value.
        """
        import osmnx as ox
        import networkx as nx

        if depart_time is None:
            depart_time = datetime.now()

        G = self.osm.load()
        center = ox.nearest_nodes(G, lon, lat)
        weekday = depart_time.weekday()
        hour = depart_time.hour
        cong = self.analyzer.congestion_at(hour, weekday, lat, lon)
        factor = self.analyzer.travel_time_factor(cong)

        # Build weighted graph
        for u, v, data in G.edges(data=True):
            base_t = data.get("travel_time", data.get("length", 50) / 8.33)
            data["_iso_w"] = base_t * factor

        results: dict[int, list[tuple[float, float]]] = {}
        for max_min in sorted(minutes):
            max_sec = max_min * 60
            reachable = nx.single_source_dijkstra_path_length(
                G, center, cutoff=max_sec, weight="_iso_w"
            )
            nodes = [G.nodes[n] for n in reachable]
            coords = [(n["y"], n["x"]) for n in nodes]
            if len(coords) >= 3:
                results[max_min] = _convex_hull(coords)
            else:
                results[max_min] = coords

        return results


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Simple Graham scan convex hull."""
    if len(points) < 3:
        return points
    pts = sorted(set(points))

    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# Singleton
traffic_analyzer = TrafficAnalyzer()
