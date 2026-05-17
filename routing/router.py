"""
routing/router.py — Time-aware heuristic routing engine
Primary: Valhalla (self-hosted Docker) with custom costing
Fallback: osmnx + NetworkX A* with custom edge weights
"""
from __future__ import annotations
import math
import hashlib
from datetime import datetime
from typing import Optional

from routing.geo_utils import haversine_m
from dataclasses import dataclass, field

import httpx
import networkx as nx
import osmnx as ox
from loguru import logger

from config.settings import settings
from core.database import db
from core.environmental_analyzer import environmental_analyzer
from core.traffic_analyzer import traffic_analyzer


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RouteStep:
    instruction: str          # human readable, Vietnamese
    distance_m: float
    duration_s: float
    lat: float
    lon: float
    bearing: float = 0.0
    street_name: str = ""
    maneuver: str = ""        # "turn_left" | "turn_right" | "straight" | "arrive"
    image_paths: list[str] = field(default_factory=list)   # illustrative photos


@dataclass
class Route:
    steps: list[RouteStep]
    total_distance_m: float
    total_duration_s: float
    geometry: list[tuple[float, float]]    # [(lat, lon), …] full polyline
    origin: tuple[float, float]
    destination: tuple[float, float]
    depart_time: datetime
    via_pois: list[dict] = field(default_factory=list)
    analysis: dict = field(default_factory=dict)

    @property
    def total_duration_min(self) -> float:
        return self.total_duration_s / 60


@dataclass(frozen=True)
class RouteProfile:
    name: str
    local_bias: float = 1.0
    highway_bias: float = 1.0
    turn_bias: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Traffic / heuristic cost
# ─────────────────────────────────────────────────────────────────────────────

class TrafficHeuristic:
    """
    Compute a time-dependent congestion multiplier.
    Combines:
     1. Static peak-hour schedule (from settings)
     2. Crowd-sourced observations stored in DB
     3. Road-type preference (local roads get a bonus)
    """

    def __init__(self):
        self.peak_schedule = settings.peak_hours
        self._obs_cache: dict[int, float] = {}  # hour → avg congestion

    async def warm_cache(self, weekday: int | None = None) -> None:
        for h in range(24):
            c = await db.avg_congestion(h, weekday)
            self._obs_cache[h] = c

    def congestion_factor(self, depart_time: datetime) -> float:
        """Returns multiplicative speed penalty (>1 = slower)."""
        h = depart_time.hour
        # DB-observed factor (scaled: 0.0–1.0 → 1.0–2.0 multiplier)
        obs_c = self._obs_cache.get(h, 0.0)
        obs_factor = 1.0 + obs_c  # [1.0, 2.0]

        # Static schedule
        sched_factor = 1.0
        for start_h, end_h, factor in self.peak_schedule:
            if start_h <= h < end_h:
                sched_factor = factor
                break

        # Use the max of both signals
        return max(obs_factor, sched_factor)

    def edge_weight(
        self,
        base_time_s: float,
        depart_time: datetime,
        road_type: str = "",
        is_custom_local: bool = False,
        lat: float | None = None,
        lon: float | None = None,
        local_bias: float = 1.0,
        highway_bias: float = 1.0,
        weather_severity: float | None = None,
        avoid_uncovered: bool = False,
        is_covered: bool = True,
        surface: str = "",
    ) -> float:
        """Final edge weight used by the router."""
        cong = self.congestion_factor(depart_time)
        weighted = base_time_s * (
            settings.route_time_weight + settings.route_congestion_weight * cong
        )

        env_penalty, _ = environmental_analyzer.environmental_penalty(lat, lon, depart_time)
        weighted *= 1 + settings.route_crowd_weight * (env_penalty - 1)
        weighted *= 1 + settings.route_weather_weight * (env_penalty - 1)
        if weather_severity is not None:
            weighted *= 1 + settings.route_weather_weight * float(weather_severity)
        if avoid_uncovered and not is_covered:
            weighted *= 1.9
        if avoid_uncovered and surface in ("grass", "dirt", "gravel"):
            weighted *= 1.35

        if is_custom_local:
            weighted *= max(0.5, settings.local_road_bonus * local_bias)

        if road_type in ("motorway", "trunk", "primary"):
            weighted *= settings.highway_penalty * highway_bias

        weighted += base_time_s * settings.route_distance_weight * 0.1

        return weighted


# ─────────────────────────────────────────────────────────────────────────────
# OSM graph (with custom edges injected)
# ─────────────────────────────────────────────────────────────────────────────

class OSMGraph:
    """
    Downloads (or loads cached) OSM road network and patches in
    custom edges (alleys, shortcuts) from the local database.
    """

    def __init__(self, area: str = settings.osm_area):
        self.area = area
        self.cache_dir = settings.osm_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_key = hashlib.md5(area.encode()).hexdigest()[:12]
        self._graph_path = self.cache_dir / f"osm_{self._cache_key}.graphml"
        self.G: nx.MultiDiGraph | None = None

    def load(self) -> nx.MultiDiGraph:
        if self.G is not None:
            return self.G
        if self._graph_path.exists():
            logger.info(f"Loading cached OSM graph from {self._graph_path}…")
            self.G = ox.load_graphml(str(self._graph_path))
        else:
            if not settings.osm_auto_download:
                raise RuntimeError(
                    "No cached OSM graph found. Set OSM_AUTO_DOWNLOAD=true or prebuild the graph cache."
                )
            logger.info(f"Downloading OSM graph for '{self.area}'…")
            self.G = ox.graph_from_place(
                self.area,
                network_type=settings.osm_network_type,
                simplify=True,
                retain_all=False,
            )
            ox.save_graphml(self.G, str(self._graph_path))
            logger.info("OSM graph saved to cache.")
        return self.G

    async def patch_custom_edges(self) -> None:
        """Inject user-defined alleys/shortcuts into the graph."""
        if self.G is None:
            return
        edges = await db.get_all_custom_edges()
        max_snap = settings.custom_edge_snap_max_m
        patched = 0
        skipped = 0
        for e in edges:
            # Find nearest OSM nodes to the custom edge endpoints
            u = ox.nearest_nodes(self.G, e["from_lon"], e["from_lat"])
            v = ox.nearest_nodes(self.G, e["to_lon"],   e["to_lat"])
            nu = self.G.nodes[u]
            nv = self.G.nodes[v]
            d_from = haversine_m(e["from_lat"], e["from_lon"], nu.get("y", 0.0), nu.get("x", 0.0))
            d_to = haversine_m(e["to_lat"], e["to_lon"], nv.get("y", 0.0), nv.get("x", 0.0))
            if d_from > max_snap or d_to > max_snap:
                logger.warning(
                    f"Skip custom edge id={e.get('id')}: snap too far "
                    f"({d_from:.0f} m / {d_to:.0f} m > {max_snap:.0f} m)"
                )
                skipped += 1
                continue
            self.G.add_edge(u, v,
                key=f"custom_{e['id']}",
                length=e["distance_m"] or 50,
                travel_time=e["travel_time_s"] or 10,
                road_type=e.get("road_type", "alley"),
                custom_local=True,
                name=e.get("name", "Custom road"),
                is_covered=bool(e.get("is_covered")),
                surface=e.get("surface", ""),
                slope_deg=e.get("slope_deg", 0),
            )
            if e.get("is_bidirectional"):
                self.G.add_edge(v, u,
                    key=f"custom_{e['id']}_rev",
                    length=e["distance_m"] or 50,
                    travel_time=e["travel_time_s"] or 10,
                    road_type=e.get("road_type", "alley"),
                    custom_local=True,
                    name=e.get("name", "Custom road"),
                    is_covered=bool(e.get("is_covered")),
                    surface=e.get("surface", ""),
                    slope_deg=e.get("slope_deg", 0),
                )
            patched += 1
        logger.debug(f"Patched {patched} custom edges into graph ({skipped} skipped by snap limit).")

    def add_travel_times(self, speed_kph: float = 30.0) -> None:
        """Estimate travel_time for each edge based on length and speed."""
        for u, v, data in self.G.edges(data=True):
            if "travel_time" not in data:
                dist = data.get("length", 0)
                data["travel_time"] = dist / (speed_kph / 3.6)


# ─────────────────────────────────────────────────────────────────────────────
# Valhalla client
# ─────────────────────────────────────────────────────────────────────────────

class ValhallaClient:
    def __init__(self, base_url: str = settings.valhalla_url):
        self.base_url = base_url.rstrip("/")

    async def is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{self.base_url}/status")
                return r.status_code == 200
        except Exception:
            return False

    async def route(
        self,
        origin_lat: float, origin_lon: float,
        dest_lat: float, dest_lon: float,
        depart_time: datetime | None = None,
        costing: str = "auto",
        extra_costing: dict | None = None,
        via_waypoints: list[tuple[float, float]] | None = None,
        alternates: int = 0,
    ) -> dict | None:
        locs: list[dict] = [{"lat": origin_lat, "lon": origin_lon}]
        for wlat, wlon in via_waypoints or []:
            locs.append({"lat": wlat, "lon": wlon, "type": "through"})
        locs.append({"lat": dest_lat, "lon": dest_lon})
        payload: dict = {
            "locations": locs,
            "costing": costing,
            "directions_options": {"language": "vi-VI", "units": "km"},
        }
        if depart_time:
            payload["date_time"] = {
                "type": 1,
                "value": depart_time.strftime("%Y-%m-%dT%H:%M"),
            }
        if extra_costing:
            payload["costing_options"] = {costing: extra_costing}
        if alternates > 0:
            payload["alternates"] = min(alternates, settings.route_alternates_max)

        try:
            async with httpx.AsyncClient(timeout=settings.valhalla_timeout) as c:
                r = await c.post(f"{self.base_url}/route", json=payload)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            logger.warning(f"Valhalla error: {e}")
            if alternates > 0:
                try:
                    payload.pop("alternates", None)
                    async with httpx.AsyncClient(timeout=settings.valhalla_timeout) as c2:
                        r2 = await c2.post(f"{self.base_url}/route", json=payload)
                        r2.raise_for_status()
                        return r2.json()
                except Exception as e2:
                    logger.warning(f"Valhalla retry without alternates: {e2}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Direction instruction generator (Vietnamese)
# ─────────────────────────────────────────────────────────────────────────────

def _vn_instruction(maneuver: str, street: str, dist: float) -> str:
    dist_str = f"{int(dist)} m" if dist < 1000 else f"{dist/1000:.1f} km"
    templates = {
        "turn_left":         f"Rẽ trái vào {street}, đi {dist_str}",
        "turn_right":        f"Rẽ phải vào {street}, đi {dist_str}",
        "slight_left":       f"Đi nhẹ sang trái vào {street}, đi {dist_str}",
        "slight_right":      f"Đi nhẹ sang phải vào {street}, đi {dist_str}",
        "sharp_left":        f"Rẽ gắt sang trái vào {street}",
        "sharp_right":       f"Rẽ gắt sang phải vào {street}",
        "straight":          f"Đi thẳng trên {street}, đi {dist_str}",
        "u_turn":            f"Quay đầu xe",
        "arrive":            f"Đã đến nơi — {street}",
        "arrive_left":       f"Điểm đến ở bên trái — {street}",
        "arrive_right":      f"Điểm đến ở bên phải — {street}",
        "depart":            f"Xuất phát từ {street}, đi {dist_str}",
        "merge":             f"Nhập vào {street}",
        "ramp":              f"Lên nhánh đường {street}",
        "roundabout_enter":  f"Vào vòng xuyến, ra lối {street}",
    }
    return templates.get(maneuver, f"Đi theo {street}, {dist_str}")


# ─────────────────────────────────────────────────────────────────────────────
# osmnx A* fallback router
# ─────────────────────────────────────────────────────────────────────────────

class OSMNXRouter:
    def __init__(self, osm_graph: OSMGraph, heuristic: TrafficHeuristic):
        self.osm = osm_graph
        self.heuristic = heuristic

    def _weighted_graph(self, depart_time: datetime) -> nx.MultiDiGraph:
        G = self.osm.load()
        for u, v, k, data in G.edges(keys=True, data=True):
            base_t = data.get("travel_time", data.get("length", 50) / 8.33)
            road_type = data.get("highway", "")
            is_custom = data.get("custom_local", False)
            lat = G.nodes[u].get("y")
            lon = G.nodes[u].get("x")
            data["_weight"] = self.heuristic.edge_weight(
                base_t, depart_time, road_type, is_custom, lat, lon
            )
        return G

    async def route(
        self,
        origin_lat: float, origin_lon: float,
        dest_lat: float, dest_lon: float,
        depart_time: datetime | None = None,
    ) -> Route | None:
        if depart_time is None:
            depart_time = datetime.now()

        G = self._weighted_graph(depart_time)
        orig_node = ox.nearest_nodes(G, origin_lon, origin_lat)
        dest_node = ox.nearest_nodes(G, dest_lon, dest_lat)

        try:
            node_path = nx.astar_path(
                G, orig_node, dest_node,
                weight="_weight",
                heuristic=lambda u, v: _haversine(
                    G.nodes[u]["y"], G.nodes[u]["x"],
                    G.nodes[v]["y"], G.nodes[v]["x"],
                ) / (settings.osm_network_type == "drive" and 60 or 5),
            )
        except nx.NetworkXNoPath:
            logger.warning("No path found via A*")
            return None

        steps: list[RouteStep] = []
        geometry: list[tuple[float, float]] = []
        total_dist = 0.0
        total_time = 0.0

        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            edge_data = min(G[u][v].values(), key=lambda d: d.get("_weight", 999999))
            dist = edge_data.get("length", 0)
            t    = edge_data.get("travel_time", dist / 8.33)
            street = edge_data.get("name", "") or ""
            if isinstance(street, list):
                street = street[0] if street else ""
            road_type = edge_data.get("road_type", edge_data.get("highway", ""))

            lat_u = G.nodes[u]["y"]
            lon_u = G.nodes[u]["x"]
            lat_v = G.nodes[v]["y"]
            lon_v = G.nodes[v]["x"]
            geometry.append((lat_u, lon_u))

            # Simple maneuver detection
            maneuver = "straight"
            if i > 0:
                prev = node_path[i - 1]
                b1 = _bearing(G.nodes[prev]["y"], G.nodes[prev]["x"], lat_u, lon_u)
                b2 = _bearing(lat_u, lon_u, lat_v, lon_v)
                diff = (b2 - b1 + 360) % 360
                if diff > 315 or diff < 45:
                    maneuver = "straight"
                elif diff < 135:
                    maneuver = "turn_right"
                elif diff < 225:
                    maneuver = "u_turn"
                else:
                    maneuver = "turn_left"
            elif i == 0:
                maneuver = "depart"

            total_dist += dist
            total_time += edge_data.get("_weight", t)

            steps.append(RouteStep(
                instruction=_vn_instruction(maneuver, street or "đường", dist),
                distance_m=dist,
                duration_s=t,
                lat=lat_u,
                lon=lon_u,
                bearing=_bearing(lat_u, lon_u, lat_v, lon_v),
                street_name=street,
                maneuver=maneuver,
            ))

        # Final arrive step
        if node_path:
            last = node_path[-1]
            geometry.append((G.nodes[last]["y"], G.nodes[last]["x"]))
            steps.append(RouteStep(
                instruction="Đã đến điểm đến",
                distance_m=0,
                duration_s=0,
                lat=G.nodes[last]["y"],
                lon=G.nodes[last]["x"],
                maneuver="arrive",
            ))

        return Route(
            steps=steps,
            total_distance_m=total_dist,
            total_duration_s=total_time,
            geometry=geometry,
            origin=(origin_lat, origin_lon),
            destination=(dest_lat, dest_lon),
            depart_time=depart_time,
            analysis=self._build_route_analysis(geometry, depart_time, total_dist, total_time),
        )

    def _build_route_analysis(
        self,
        geometry: list[tuple[float, float]],
        depart_time: datetime,
        total_distance_m: float,
        total_duration_s: float,
    ) -> dict:
        sample = geometry[:: max(1, len(geometry) // 12)] if geometry else []
        crowd_vals = []
        weather_vals = []
        for lat, lon in sample:
            crowd_vals.append(
                environmental_analyzer.crowd_level(lat, lon, depart_time.hour, depart_time.weekday())
            )
            weather_vals.append(
                environmental_analyzer.weather_severity(lat, lon, depart_time.hour, depart_time.weekday())
            )
        avg_crowd = sum(crowd_vals) / len(crowd_vals) if crowd_vals else 0.0
        avg_weather = sum(weather_vals) / len(weather_vals) if weather_vals else 0.0
        return {
            "strategy": "offline_heuristic_astar",
            "distance_km": round(total_distance_m / 1000, 3),
            "duration_min": round(total_duration_s / 60, 2),
            "avg_crowd_level": round(avg_crowd, 3),
            "avg_weather_severity": round(avg_weather, 3),
            "weights": {
                "distance": settings.route_distance_weight,
                "time": settings.route_time_weight,
                "congestion": settings.route_congestion_weight,
                "crowd": settings.route_crowd_weight,
                "weather": settings.route_weather_weight,
            },
        }


class SmartOSMNXRouter(OSMNXRouter):
    @staticmethod
    def _profiles() -> list[RouteProfile]:
        return [
            RouteProfile(name="balanced", local_bias=1.0, highway_bias=1.0, turn_bias=1.0),
            RouteProfile(name="local_friendly", local_bias=0.82, highway_bias=1.12, turn_bias=0.95),
            RouteProfile(name="fast_main", local_bias=1.08, highway_bias=0.9, turn_bias=1.1),
        ][: max(1, settings.route_candidate_profiles)]

    def _weighted_graph_for_profile(
        self,
        depart_time: datetime,
        profile: RouteProfile,
        avoid_discs: list[tuple[float, float, float]] | None = None,
        weather_severity: float | None = None,
        avoid_uncovered: bool = False,
    ) -> nx.MultiDiGraph:
        G = self.osm.load()
        pen = settings.route_avoid_disc_penalty
        for u, v, k, data in G.edges(keys=True, data=True):
            base_t = data.get("travel_time", data.get("length", 50) / 8.33)
            road_type = data.get("highway", "")
            is_custom = data.get("custom_local", False)
            lat_u = G.nodes[u].get("y")
            lon_u = G.nodes[u].get("x")
            lat_v = G.nodes[v].get("y")
            lon_v = G.nodes[v].get("x")
            data["_weight"] = self.heuristic.edge_weight(
                base_t,
                depart_time,
                road_type,
                is_custom,
                lat_u,
                lon_u,
                local_bias=profile.local_bias,
                highway_bias=profile.highway_bias,
                weather_severity=weather_severity,
                avoid_uncovered=avoid_uncovered,
                is_covered=bool(data.get("is_covered", not data.get("custom_local", False))),
                surface=str(data.get("surface", "")),
            )
            if avoid_discs and lat_u is not None and lon_u is not None and lat_v is not None and lon_v is not None:
                mlat = (float(lat_u) + float(lat_v)) / 2.0
                mlon = (float(lon_u) + float(lon_v)) / 2.0
                for alat, alon, rad_m in avoid_discs:
                    if haversine_m(mlat, mlon, alat, alon) < float(rad_m):
                        data["_weight"] *= pen
                        break
        return G

    @staticmethod
    def _geometry_fingerprint(route: Route) -> tuple:
        g = route.geometry
        if len(g) < 2:
            return (round(route.total_distance_m, 0),)
        return (
            round(route.total_distance_m, 0),
            round(g[0][0], 5),
            round(g[0][1], 5),
            round(g[len(g) // 2][0], 5),
            round(g[-1][0], 5),
        )

    @staticmethod
    def _summarize_alternate_route(cand: Route) -> dict:
        cap = 500
        geom = cand.geometry
        return {
            "profile": cand.analysis.get("selected_profile", "unknown"),
            "distance_km": round(cand.total_distance_m / 1000, 3),
            "duration_min": round(cand.total_duration_min, 2),
            "route_score": cand.analysis.get("route_score"),
            "geometry": geom[:cap],
            "geometry_truncated": len(geom) > cap,
            "geometry_point_count": len(geom),
        }

    async def route(
        self,
        origin_lat: float, origin_lon: float,
        dest_lat: float, dest_lon: float,
        depart_time: datetime | None = None,
        avoid_discs: list[tuple[float, float, float]] | None = None,
        alternates_count: int = 0,
        weather_severity: float | None = None,
        avoid_uncovered: bool = False,
    ) -> Route | None:
        if depart_time is None:
            depart_time = datetime.now()

        candidates: list[Route] = []
        for profile in self._profiles():
            G = self._weighted_graph_for_profile(
                depart_time, profile, avoid_discs, weather_severity, avoid_uncovered
            )
            orig_node = ox.nearest_nodes(G, origin_lon, origin_lat)
            dest_node = ox.nearest_nodes(G, dest_lon, dest_lat)
            try:
                node_path = nx.astar_path(
                    G,
                    orig_node,
                    dest_node,
                    weight="_weight",
                    heuristic=lambda u, v: _haversine(
                        G.nodes[u]["y"], G.nodes[u]["x"], G.nodes[v]["y"], G.nodes[v]["x"]
                    ) / (settings.osm_network_type == "drive" and 60 or 5),
                )
            except nx.NetworkXNoPath:
                continue

            route = await self._route_from_path(
                G, node_path, origin_lat, origin_lon, dest_lat, dest_lon, depart_time, profile
            )
            if route:
                candidates.append(route)

        if not candidates:
            logger.warning("No path found via smart A*")
            return None

        scored = sorted(candidates, key=lambda r: r.analysis.get("route_score", float("inf")))
        best_route = scored[0]
        best_route.analysis["candidate_profiles"] = [
            {
                "profile": cand.analysis.get("selected_profile", "unknown"),
                "score": round(cand.analysis.get("route_score", 0.0), 3),
                "duration_min": round(cand.total_duration_min, 2),
                "landmark_density": cand.analysis.get("landmark_density", 0.0),
            }
            for cand in candidates
        ]
        best_route.analysis["strategy"] = "multi_profile_offline_rerank"
        if weather_severity is not None:
            best_route.analysis["weather_severity_request"] = round(float(weather_severity), 3)
            best_route.analysis["avoid_uncovered"] = avoid_uncovered

        want_alt = max(alternates_count, 0)
        if want_alt > 0:
            seen: set[tuple] = {self._geometry_fingerprint(best_route)}
            alt_summaries: list[dict] = []
            for cand in scored[1:]:
                fp = self._geometry_fingerprint(cand)
                if fp in seen:
                    continue
                seen.add(fp)
                alt_summaries.append(self._summarize_alternate_route(cand))
                if len(alt_summaries) >= min(want_alt, settings.route_alternates_max):
                    break
            best_route.analysis["alternates"] = alt_summaries

        return best_route

    async def _route_from_path(
        self,
        G: nx.MultiDiGraph,
        node_path: list[int],
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        depart_time: datetime,
        profile: RouteProfile,
    ) -> Route | None:
        steps: list[RouteStep] = []
        geometry: list[tuple[float, float]] = []
        total_dist = 0.0
        total_time = 0.0
        turn_count = 0
        custom_edge_count = 0
        highway_edge_count = 0

        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            edge_data = min(G[u][v].values(), key=lambda d: d.get("_weight", 999999))
            dist = edge_data.get("length", 0)
            t = edge_data.get("travel_time", dist / 8.33)
            street = edge_data.get("name", "") or ""
            if isinstance(street, list):
                street = street[0] if street else ""
            road_type = edge_data.get("road_type", edge_data.get("highway", ""))

            lat_u = G.nodes[u]["y"]
            lon_u = G.nodes[u]["x"]
            lat_v = G.nodes[v]["y"]
            lon_v = G.nodes[v]["x"]
            geometry.append((lat_u, lon_u))

            maneuver = "straight"
            if i > 0:
                prev = node_path[i - 1]
                b1 = _bearing(G.nodes[prev]["y"], G.nodes[prev]["x"], lat_u, lon_u)
                b2 = _bearing(lat_u, lon_u, lat_v, lon_v)
                diff = (b2 - b1 + 360) % 360
                if diff > 315 or diff < 45:
                    maneuver = "straight"
                elif diff < 135:
                    maneuver = "turn_right"
                elif diff < 225:
                    maneuver = "u_turn"
                else:
                    maneuver = "turn_left"
            elif i == 0:
                maneuver = "depart"

            total_dist += dist
            total_time += edge_data.get("_weight", t)
            if maneuver in ("turn_left", "turn_right"):
                turn_count += 1
                total_time += settings.route_turn_penalty_turn * profile.turn_bias
            elif maneuver == "u_turn":
                turn_count += 1
                total_time += settings.route_turn_penalty_uturn * profile.turn_bias
            if edge_data.get("custom_local"):
                custom_edge_count += 1
            if road_type in ("motorway", "trunk", "primary"):
                highway_edge_count += 1

            steps.append(RouteStep(
                instruction=_vn_instruction(maneuver, street or "đường", dist),
                distance_m=dist,
                duration_s=t,
                lat=lat_u,
                lon=lon_u,
                bearing=_bearing(lat_u, lon_u, lat_v, lon_v),
                street_name=street,
                maneuver=maneuver,
            ))

        if node_path:
            last = node_path[-1]
            geometry.append((G.nodes[last]["y"], G.nodes[last]["x"]))
            steps.append(RouteStep(
                instruction="Đã đến điểm đến",
                distance_m=0,
                duration_s=0,
                lat=G.nodes[last]["y"],
                lon=G.nodes[last]["x"],
                maneuver="arrive",
            ))

        analysis = await self._build_smart_analysis(
            geometry,
            depart_time,
            total_dist,
            total_time,
            turn_count,
            custom_edge_count,
            highway_edge_count,
            len(steps),
            profile,
        )
        return Route(
            steps=steps,
            total_distance_m=total_dist,
            total_duration_s=total_time,
            geometry=geometry,
            origin=(origin_lat, origin_lon),
            destination=(dest_lat, dest_lon),
            depart_time=depart_time,
            analysis=analysis,
        )

    async def _build_smart_analysis(
        self,
        geometry: list[tuple[float, float]],
        depart_time: datetime,
        total_distance_m: float,
        total_duration_s: float,
        turn_count: int,
        custom_edge_count: int,
        highway_edge_count: int,
        step_count: int,
        profile: RouteProfile,
    ) -> dict:
        sample = geometry[:: max(1, len(geometry) // 12)] if geometry else []
        crowd_vals: list[float] = []
        weather_vals: list[float] = []
        congestion_vals: list[float] = []
        landmark_hits = 0
        for lat, lon in sample:
            crowd_vals.append(environmental_analyzer.crowd_level(lat, lon, depart_time.hour, depart_time.weekday()))
            weather_vals.append(environmental_analyzer.weather_severity(lat, lon, depart_time.hour, depart_time.weekday()))
            congestion_vals.append(traffic_analyzer.congestion_at(depart_time.hour, depart_time.weekday(), lat, lon))
            nearby = await db.nearby_locations(lat, lon, radius_deg=0.0007)
            landmark_hits += min(len(nearby), 4)

        avg_crowd = sum(crowd_vals) / len(crowd_vals) if crowd_vals else 0.0
        avg_weather = sum(weather_vals) / len(weather_vals) if weather_vals else 0.0
        avg_congestion = sum(congestion_vals) / len(congestion_vals) if congestion_vals else 0.0
        complexity = turn_count / max(1, step_count)
        custom_ratio = custom_edge_count / max(1, step_count)
        highway_ratio = highway_edge_count / max(1, step_count)
        landmark_density = landmark_hits / max(1, len(sample))
        route_score = (
            settings.route_time_weight * (total_duration_s / 60)
            + settings.route_congestion_weight * (avg_congestion * 25)
            + settings.route_crowd_weight * (avg_crowd * 20)
            + settings.route_weather_weight * (avg_weather * 20)
            + settings.route_complexity_weight * (complexity * 50)
            - settings.route_landmark_weight * (landmark_density * 5)
            - settings.route_locality_weight * (custom_ratio * 15)
            + settings.highway_penalty * highway_ratio
        )
        return {
            "strategy": "multi_profile_offline_rerank",
            "selected_profile": profile.name,
            "distance_km": round(total_distance_m / 1000, 3),
            "duration_min": round(total_duration_s / 60, 2),
            "avg_congestion": round(avg_congestion, 3),
            "avg_crowd_level": round(avg_crowd, 3),
            "avg_weather_severity": round(avg_weather, 3),
            "turn_count": turn_count,
            "complexity": round(complexity, 3),
            "custom_edge_ratio": round(custom_ratio, 3),
            "highway_edge_ratio": round(highway_ratio, 3),
            "landmark_density": round(landmark_density, 3),
            "route_score": round(route_score, 3),
        }


def _merge_route_segments(acc: Route | None, nxt: Route) -> Route:
    """Concatenate two Route objects (e.g. A→via, via→B). Drops duplicate arrive/depart joins."""
    if acc is None:
        return Route(
            steps=list(nxt.steps),
            total_distance_m=nxt.total_distance_m,
            total_duration_s=nxt.total_duration_s,
            geometry=list(nxt.geometry),
            origin=nxt.origin,
            destination=nxt.destination,
            depart_time=nxt.depart_time,
            analysis=dict(nxt.analysis),
            via_pois=list(nxt.via_pois),
        )
    acc_steps = list(acc.steps)
    if acc_steps and acc_steps[-1].maneuver == "arrive":
        acc_steps = acc_steps[:-1]
    nxt_steps = list(nxt.steps)
    if nxt_steps and nxt_steps[0].maneuver == "depart":
        nxt_steps = nxt_steps[1:]
    steps = acc_steps + nxt_steps
    geom = list(acc.geometry)
    if nxt.geometry:
        if geom and nxt.geometry and geom[-1] == nxt.geometry[0]:
            geom.extend(nxt.geometry[1:])
        else:
            geom.extend(nxt.geometry)
    analysis = dict(acc.analysis)
    analysis["chained_segments"] = int(analysis.get("chained_segments", 1)) + 1
    return Route(
        steps=steps,
        total_distance_m=acc.total_distance_m + nxt.total_distance_m,
        total_duration_s=acc.total_duration_s + nxt.total_duration_s,
        geometry=geom,
        origin=acc.origin,
        destination=nxt.destination,
        depart_time=acc.depart_time,
        analysis=analysis,
        via_pois=list(acc.via_pois) + list(nxt.via_pois),
    )


def _enrich_valhalla_route_analysis(route: Route) -> None:
    """Add congestion/crowd/weather samples so Valhalla routes align with offline analysis fields."""
    geom = route.geometry
    if not geom:
        return
    sample = geom[:: max(1, len(geom) // 12)]
    h = route.depart_time.hour
    wd = route.depart_time.weekday()
    congestion_vals = [traffic_analyzer.congestion_at(h, wd, la, lo) for la, lo in sample]
    crowd_vals = [environmental_analyzer.crowd_level(la, lo, h, wd) for la, lo in sample]
    weather_vals = [environmental_analyzer.weather_severity(la, lo, h, wd) for la, lo in sample]
    route.analysis["avg_congestion"] = round(
        sum(congestion_vals) / len(congestion_vals), 3
    ) if congestion_vals else 0.0
    route.analysis["avg_crowd_level"] = round(sum(crowd_vals) / len(crowd_vals), 3) if crowd_vals else 0.0
    route.analysis["avg_weather_severity"] = round(
        sum(weather_vals) / len(weather_vals), 3
    ) if weather_vals else 0.0
    route.analysis["geometry_point_count"] = len(geom)


def _extract_valhalla_alternate_summaries(raw: dict) -> list[dict]:
    summaries: list[dict] = []
    alts = raw.get("alternates")
    if not isinstance(alts, list):
        return summaries
    for a in alts[: settings.route_alternates_max]:
        if not isinstance(a, dict):
            continue
        trip = a.get("trip", a)
        if not isinstance(trip, dict):
            continue
        legs = trip.get("legs", [])
        geom: list[tuple[float, float]] = []
        td = tt = 0.0
        for lg in legs:
            td += lg.get("length", 0) * 1000
            tt += lg.get("time", 0)
            geom.extend(_decode_polyline(lg.get("shape", "") or ""))
        cap = 500
        summaries.append(
            {
                "provider": "valhalla",
                "distance_km": round(td / 1000, 3),
                "duration_min": round(tt / 60, 2),
                "geometry": geom[:cap],
                "geometry_truncated": len(geom) > cap,
                "geometry_point_count": len(geom),
            }
        )
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Main Router (Valhalla → fallback to osmnx)
# ─────────────────────────────────────────────────────────────────────────────

class NavRouter:
    def __init__(self):
        self.osm = OSMGraph()
        self.heuristic = TrafficHeuristic()
        self.valhalla = ValhallaClient()
        self._osmnx_router: SmartOSMNXRouter | None = None
        self._valhalla_ok = False

    async def init(self) -> None:
        """Preload graph, warm traffic cache, check Valhalla."""
        settings.setup_dirs()
        try:
            self.osm.load()
            self.osm.add_travel_times()
            await self.osm.patch_custom_edges()
        except Exception as e:
            logger.warning(f"OSM graph not ready yet: {e}")
        await self.heuristic.warm_cache(datetime.now().weekday())
        await environmental_analyzer.refresh(force=True)

        self._valhalla_ok = await self.valhalla.is_healthy()
        if self._valhalla_ok and not avoid_uncovered:
            logger.info("Valhalla routing engine: OK")
        else:
            logger.info("Valhalla not running — using osmnx A* fallback")
            self._osmnx_router = SmartOSMNXRouter(self.osm, self.heuristic)

    async def find_route(
        self,
        origin_lat: float, origin_lon: float,
        dest_lat: float, dest_lon: float,
        depart_time: datetime | None = None,
        *,
        waypoints: list[tuple[float, float]] | None = None,
        avoid_discs: list[tuple[float, float, float]] | None = None,
        alternates: int = 0,
        gps_accuracy_m: float = 5.0,
        weather_severity: float | None = None,
        avoid_uncovered: bool = False,
    ) -> Route | None:
        if depart_time is None:
            depart_time = datetime.now()

        # ── Smart Indoor Detection ────────────────────────────────────────────
        # Priority 1: Check if both points are in database locations (definite indoor)
        origin_loc = await db.find_location_by_coords(origin_lat, origin_lon, tolerance=0.0001)
        dest_loc = await db.find_location_by_coords(dest_lat, dest_lon, tolerance=0.0001)
        
        if origin_loc and dest_loc:
            # Both points are in database → definitely indoor routing
            logger.info(
                f"Indoor routing: {origin_loc['name']} (floor {origin_loc.get('floor', 1)}) → "
                f"{dest_loc['name']} (floor {dest_loc.get('floor', 1)})"
            )
            indoor_result = await self._try_indoor_route(
                origin_lat, origin_lon, dest_lat, dest_lon
            )
            if indoor_result is not None:
                return indoor_result
            # If indoor routing fails, fall through to outdoor routing
            logger.warning("Indoor routing failed, falling back to outdoor")
        
        # Priority 2: Check GPS accuracy (poor accuracy suggests indoor)
        elif gps_accuracy_m > settings.indoor_gps_accuracy_threshold_m:
            indoor_result = await self._try_indoor_route(
                origin_lat, origin_lon, dest_lat, dest_lon
            )
            if indoor_result is not None:
                return indoor_result

        # ── Outdoor Routing ───────────────────────────────────────────────────
        via = list(waypoints or [])
        points = [(origin_lat, origin_lon)] + via + [(dest_lat, dest_lon)]
        want_alt = max(0, min(int(alternates), settings.route_alternates_max))
        valhalla_alt = want_alt if len(points) == 2 else 0

        if self._valhalla_ok:
            raw = await self.valhalla.route(
                origin_lat,
                origin_lon,
                dest_lat,
                dest_lon,
                depart_time=depart_time,
                extra_costing={
                    "speed_types": ["freeflow", "constrained", "predicted"],
                    "use_living_streets": 0.8,
                },
                via_waypoints=via if via else None,
                alternates=valhalla_alt,
            )
            if raw:
                route = self._parse_valhalla(
                    raw, depart_time, origin_lat, origin_lon, dest_lat, dest_lon
                )
                alt_sum = _extract_valhalla_alternate_summaries(raw)
                if alt_sum:
                    route.analysis["alternates"] = alt_sum
                return route

        if len(points) > 2:
            return await self._find_route_osmnx_chain(
                points, depart_time, avoid_discs, want_alt, weather_severity, avoid_uncovered
            )

        if self._osmnx_router is None:
            self._osmnx_router = SmartOSMNXRouter(self.osm, self.heuristic)
        return await self._osmnx_router.route(
            origin_lat,
            origin_lon,
            dest_lat,
            dest_lon,
            depart_time,
            avoid_discs=avoid_discs,
            alternates_count=want_alt,
            weather_severity=weather_severity,
            avoid_uncovered=avoid_uncovered,
        )

    async def _try_indoor_route(
        self,
        origin_lat: float, origin_lon: float,
        dest_lat: float, dest_lon: float,
    ) -> Route | None:
        """
        Check if both origin and destination fall within a known indoor building.
        If so, run the indoor A* router and wrap the result as a Route object
        so the rest of the navigation pipeline is unaffected.
        Returns None if no indoor map covers the area.
        """
        try:
            from core.indoor_router import indoor_registry, IndoorRouter

            # Find the nearest building to the origin
            nearby = await db.nearby_floor_nodes(
                origin_lat, origin_lon, radius_deg=0.001
            )
            if not nearby:
                return None

            building_id = nearby[0]["building_id"]
            graph = indoor_registry.get(building_id)
            if graph is None:
                return None

            indoor_r = IndoorRouter(graph)
            origin_node = graph.nearest_node(origin_lat, origin_lon)
            dest_node = graph.nearest_node(dest_lat, dest_lon)
            if origin_node is None or dest_node is None:
                return None

            indoor_route = indoor_r.route(origin_node.node_id, dest_node.node_id)
            if indoor_route is None:
                return None

            # Convert IndoorRoute → Route
            steps: list[RouteStep] = []
            geometry: list[tuple[float, float]] = []
            for s in indoor_route.steps:
                steps.append(RouteStep(
                    instruction=s.instruction,
                    distance_m=s.distance_m,
                    duration_s=s.duration_s,
                    lat=s.lat,
                    lon=s.lon,
                    maneuver=s.edge_type,
                    street_name=s.instruction,
                ))
                geometry.append((s.lat, s.lon))

            if not steps:
                return None

            return Route(
                steps=steps,
                total_distance_m=indoor_route.total_distance_m,
                total_duration_s=indoor_route.total_duration_s,
                geometry=geometry,
                origin=(origin_lat, origin_lon),
                destination=(dest_lat, dest_lon),
                depart_time=datetime.now(),
                analysis={
                    "strategy": "indoor_astar",
                    "building_id": building_id,
                    "floors_visited": indoor_route.floors_visited,
                    "summary": " → ".join(
                        [f"Tầng {f}" for f in indoor_route.floors_visited]
                        + ([graph.nodes[indoor_route.destination_node].name]
                           if indoor_route.destination_node in graph.nodes else [])
                    ),
                    "distance_km": round(indoor_route.total_distance_m / 1000, 3),
                    "duration_min": round(indoor_route.total_duration_s / 60, 2),
                },
            )
        except Exception as e:
            logger.debug(f"Indoor route attempt failed: {e}")
            return None

    async def _find_route_osmnx_chain(
        self,
        points: list[tuple[float, float]],
        depart_time: datetime,
        avoid_discs: list[tuple[float, float, float]] | None,
        alternates: int,
        weather_severity: float | None = None,
        avoid_uncovered: bool = False,
    ) -> Route | None:
        if self._osmnx_router is None:
            self._osmnx_router = SmartOSMNXRouter(self.osm, self.heuristic)
        merged: Route | None = None
        via_count = max(0, len(points) - 2)
        for i in range(len(points) - 1):
            o = points[i]
            d = points[i + 1]
            seg = await self._osmnx_router.route(
                o[0], o[1], d[0], d[1], depart_time,
                avoid_discs=avoid_discs,
                alternates_count=0,
                weather_severity=weather_severity,
                avoid_uncovered=avoid_uncovered,
            )
            if not seg:
                return None
            merged = _merge_route_segments(merged, seg)
        if merged is not None:
            merged.analysis["via_count"] = via_count
            merged.analysis["strategy"] = "osmnx_chained"
        return merged

    def _parse_valhalla(
        self, raw: dict, depart_time: datetime,
        olat: float, olon: float, dlat: float, dlon: float
    ) -> Route:
        """Convert Valhalla JSON response to Route."""
        legs = raw.get("trip", {}).get("legs", [])
        steps: list[RouteStep] = []
        geometry: list[tuple[float, float]] = []
        total_dist = 0.0
        total_time = 0.0

        maneuver_map = {
            0: "none", 1: "depart", 2: "depart", 3: "turn_right",
            4: "slight_right", 5: "sharp_right", 6: "u_turn",
            7: "sharp_left", 8: "slight_left", 9: "turn_left",
            10: "straight", 11: "ramp", 12: "ramp", 13: "merge",
            14: "roundabout_enter", 15: "roundabout_enter",
            17: "arrive", 18: "arrive_right", 19: "arrive_left",
        }

        for leg in legs:
            total_dist += leg.get("length", 0) * 1000
            total_time += leg.get("time", 0)
            shape_pts = leg.get("shape", "") or ""
            decoded_leg = _decode_polyline(shape_pts) if shape_pts else []
            geometry.extend(decoded_leg)

            for m in leg.get("maneuvers", []):
                mtype = maneuver_map.get(m.get("type", 10), "straight")
                street = m.get("street_names", [""])[0] if m.get("street_names") else ""
                dist = m.get("length", 0) * 1000
                dur = m.get("time", 0)
                begin_shape = m.get("begin_shape_index", 0)
                if decoded_leg and begin_shape < len(decoded_leg):
                    lat, lon = decoded_leg[begin_shape]
                else:
                    lat, lon = _decode_polyline_point(shape_pts, begin_shape)

                steps.append(RouteStep(
                    instruction=_vn_instruction(mtype, street or "đường", dist),
                    distance_m=dist,
                    duration_s=dur,
                    lat=lat,
                    lon=lon,
                    street_name=street,
                    maneuver=mtype,
                ))

        route = Route(
            steps=steps,
            total_distance_m=total_dist,
            total_duration_s=total_time,
            geometry=geometry,
            origin=(olat, olon),
            destination=(dlat, dlon),
            depart_time=depart_time,
            analysis={
                "strategy": "valhalla_time_aware",
                "distance_km": round(total_dist / 1000, 3),
                "duration_min": round(total_time / 60, 2),
            },
        )
        _enrich_valhalla_route_analysis(route)
        return route

    async def resolve_location(self, query: str) -> tuple[float, float] | None:
        """
        Try to find (lat, lon) for a text query.
        Priority:
          1. Local DB (campus-scoped — always preferred)
          2. Nominatim with campus viewbox bias
          3. Campus centre fallback if query looks like an on-campus name
        """
        from core.campus_scope import campus_bbox, campus_center

        # 1. Local DB
        locs = await db.search_locations(query)
        if locs:
            return locs[0]["lat"], locs[0]["lon"]

        pois = await db.search_pois(query)
        if pois:
            return pois[0]["lat"], pois[0]["lon"]

        # 2. Nominatim — bias search to campus viewbox
        if settings.allow_remote_geocoding:
            bbox = campus_bbox()
            try:
                async with httpx.AsyncClient(headers={"User-Agent": "LocalNavBot/1.0"}) as c:
                    r = await c.get(
                        "https://nominatim.openstreetmap.org/search",
                        params={
                            "q": query,
                            "format": "json",
                            "limit": 5,
                            # Bias results toward campus area
                            "viewbox": f"{bbox['lon_min']},{bbox['lat_max']},{bbox['lon_max']},{bbox['lat_min']}",
                            "bounded": 0,  # 0 = prefer viewbox but don't restrict
                        },
                        timeout=5,
                    )
                    data = r.json()
                    if data:
                        return float(data[0]["lat"]), float(data[0]["lon"])
            except Exception as e:
                logger.warning(f"Geocoding failed: {e}")

        # 3. If query looks like a campus building/room name, return campus centre
        #    so routing at least starts from a sensible point
        campus_keywords = [
            "toa", "phong", "nha", "b1", "b2", "b3", "b4", "b5",
            "a1", "a2", "a3", "c1", "c2", "d1", "d2",
            "thu vien", "can tin", "ky tuc xa", "san the duc",
            "khtn", "hcmus", "dai hoc", "truong",
        ]
        q_lower = query.lower().replace("à","a").replace("á","a").replace("ả","a")\
                               .replace("ã","a").replace("ạ","a").replace("ă","a")\
                               .replace("â","a").replace("è","e").replace("é","e")\
                               .replace("ê","e").replace("ì","i").replace("í","i")\
                               .replace("ò","o").replace("ó","o").replace("ô","o")\
                               .replace("ơ","o").replace("ù","u").replace("ú","u")\
                               .replace("ư","u").replace("đ","d")
        if any(kw in q_lower for kw in campus_keywords):
            clat, clon = campus_center()
            logger.debug(f"resolve_location: campus keyword match for '{query}' → campus centre")
            return clat, clon

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Geometry utilities
# ─────────────────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _decode_polyline_point(encoded: str, index: int) -> tuple[float, float]:
    """Decode a single point from Valhalla's encoded polyline6 string."""
    try:
        pts = _decode_polyline(encoded)
        if index < len(pts):
            return pts[index]
        return (0.0, 0.0)
    except Exception:
        return (0.0, 0.0)


def _decode_polyline(encoded: str, precision: int = 6) -> list[tuple[float, float]]:
    inv = 10 ** (-precision)
    decoded = []
    idx = lat = lon = 0
    while idx < len(encoded):
        for is_lon in (False, True):
            result = shift = 0
            while True:
                b = ord(encoded[idx]) - 63
                idx += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if is_lon:
                lon += delta
            else:
                lat += delta
        decoded.append((lat * inv, lon * inv))
    return decoded
