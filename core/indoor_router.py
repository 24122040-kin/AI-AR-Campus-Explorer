"""
core/indoor_router.py — Indoor floor-plan graph + multi-floor A* routing

GeoJSON schema for a floor plan:
{
  "type": "FeatureCollection",
  "building_id": "main_building",
  "floor": 1,
  "features": [
    {
      "type": "Feature",
      "id": "node_101",
      "geometry": {"type": "Point", "coordinates": [lon, lat]},
      "properties": {
        "node_type": "room|corridor|stairs|elevator|entrance|exit",
        "name": "Phòng 101",
        "floor": 1,
        "accessible": true
      }
    },
    {
      "type": "Feature",
      "id": "edge_101_102",
      "geometry": {"type": "LineString", "coordinates": [[lon1,lat1],[lon2,lat2]]},
      "properties": {
        "edge_type": "corridor|stairs|elevator|door",
        "from_node": "node_101",
        "to_node": "node_102",
        "from_floor": 1,
        "to_floor": 1,
        "distance_m": 12.5,
        "bidirectional": true,
        "accessible": true
      }
    }
  ]
}

Multi-floor edges (stairs/elevator) connect nodes on different floors.
Edge cost model:
  - corridor/door : distance_m / WALK_SPEED_MS
  - stairs up/down: STAIR_TIME_PER_FLOOR * |floor_delta|
  - elevator      : ELEVATOR_WAIT_S + ELEVATOR_TIME_PER_FLOOR * |floor_delta|
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from typing import Any

# ── Cost constants ────────────────────────────────────────────────────────────
WALK_SPEED_MS: float = 1.2          # m/s indoor walking speed
STAIR_TIME_PER_FLOOR: float = 20.0  # seconds per floor via stairs
ELEVATOR_WAIT_S: float = 30.0       # average elevator wait
ELEVATOR_TIME_PER_FLOOR: float = 5.0  # seconds per floor in elevator
DOOR_PENALTY_S: float = 3.0         # time to open/pass a door


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class IndoorNode:
    node_id: str
    building_id: str
    floor: int
    name: str
    node_type: str          # room | corridor | stairs | elevator | entrance | exit
    lat: float
    lon: float
    accessible: bool = True
    properties: dict = field(default_factory=dict)


@dataclass
class IndoorEdge:
    edge_id: str
    from_node: str
    to_node: str
    from_floor: int
    to_floor: int
    edge_type: str          # corridor | stairs | elevator | door
    distance_m: float
    bidirectional: bool = True
    accessible: bool = True
    cost_s: float = 0.0     # pre-computed travel cost in seconds; always recomputed in __post_init__
    # Environmental properties
    is_covered: bool = False
    surface: str = "concrete"
    slope_deg: float = 0.0

    def __post_init__(self) -> None:
        # Always recompute from edge geometry — caller should not set cost_s directly.
        # A real indoor edge can never have 0 cost (even 1 cm corridor takes ~0.008 s).
        self.cost_s = _edge_cost(self)


def _edge_cost(e: IndoorEdge) -> float:
    floor_delta = abs(e.to_floor - e.from_floor)
    if e.edge_type == "stairs":
        return STAIR_TIME_PER_FLOOR * max(1, floor_delta)
    if e.edge_type == "elevator":
        return ELEVATOR_WAIT_S + ELEVATOR_TIME_PER_FLOOR * max(1, floor_delta)
    if e.edge_type == "door":
        return DOOR_PENALTY_S + e.distance_m / WALK_SPEED_MS
    # corridor / default
    return e.distance_m / WALK_SPEED_MS


@dataclass
class IndoorRouteStep:
    instruction: str
    from_node_id: str
    to_node_id: str
    from_floor: int
    to_floor: int
    edge_type: str
    distance_m: float
    duration_s: float
    lat: float
    lon: float


@dataclass
class IndoorRoute:
    steps: list[IndoorRouteStep]
    total_distance_m: float
    total_duration_s: float
    floors_visited: list[int]
    building_id: str
    origin_node: str
    destination_node: str

    def as_dict(self) -> dict:
        return {
            "ok": True,
            "building_id": self.building_id,
            "origin_node": self.origin_node,
            "destination_node": self.destination_node,
            "total_distance_m": round(self.total_distance_m, 1),
            "total_duration_s": round(self.total_duration_s, 1),
            "total_duration_min": round(self.total_duration_s / 60, 2),
            "floors_visited": self.floors_visited,
            "steps": [
                {
                    "instruction": s.instruction,
                    "from_node": s.from_node_id,
                    "to_node": s.to_node_id,
                    "from_floor": s.from_floor,
                    "to_floor": s.to_floor,
                    "edge_type": s.edge_type,
                    "distance_m": round(s.distance_m, 1),
                    "duration_s": round(s.duration_s, 1),
                    "lat": s.lat,
                    "lon": s.lon,
                }
                for s in self.steps
            ],
            "html_card": _render_indoor_html(self),
        }


# ── Graph ─────────────────────────────────────────────────────────────────────

class IndoorGraph:
    """
    In-memory directed graph for one building (all floors).
    Loaded from GeoJSON floor-plan features stored in the DB.
    """

    def __init__(self, building_id: str) -> None:
        self.building_id = building_id
        self.nodes: dict[str, IndoorNode] = {}
        self.adj: dict[str, list[tuple[str, IndoorEdge]]] = {}  # node_id → [(neighbor_id, edge)]

    def add_node(self, node: IndoorNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self.adj:
            self.adj[node.node_id] = []

    def add_edge(self, edge: IndoorEdge) -> None:
        if edge.from_node not in self.adj:
            self.adj[edge.from_node] = []
        self.adj[edge.from_node].append((edge.to_node, edge))
        if edge.bidirectional:
            if edge.to_node not in self.adj:
                self.adj[edge.to_node] = []
            # Reverse edge — same cost as forward (stairs/elevator are symmetric)
            rev = IndoorEdge(
                edge_id=edge.edge_id + "_rev",
                from_node=edge.to_node,
                to_node=edge.from_node,
                from_floor=edge.to_floor,
                to_floor=edge.from_floor,
                edge_type=edge.edge_type,
                distance_m=edge.distance_m,
                bidirectional=False,
                accessible=edge.accessible,
            )
            # cost_s is recomputed in __post_init__ — it will be identical
            # because _edge_cost is symmetric for all edge types.
            self.adj[edge.to_node].append((edge.from_node, rev))

    def load_geojson(self, geojson: dict) -> None:
        """Parse a GeoJSON FeatureCollection into nodes and edges."""
        building_id = geojson.get("building_id", self.building_id)
        for feat in geojson.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            fid = feat.get("id", "")

            if geom.get("type") == "Point":
                coords = geom["coordinates"]  # [lon, lat]
                node = IndoorNode(
                    node_id=fid,
                    building_id=building_id,
                    floor=int(props.get("floor", 1)),
                    name=props.get("name", fid),
                    node_type=props.get("node_type", "corridor"),
                    lat=float(coords[1]),
                    lon=float(coords[0]),
                    accessible=bool(props.get("accessible", True)),
                    properties=props,
                )
                self.add_node(node)

            elif geom.get("type") == "LineString":
                coords = geom["coordinates"]
                mid = coords[len(coords) // 2]
                dist = props.get("distance_m") or _polyline_length_m(coords)
                edge = IndoorEdge(
                    edge_id=fid,
                    from_node=props["from_node"],
                    to_node=props["to_node"],
                    from_floor=int(props.get("from_floor", props.get("floor", 1))),
                    to_floor=int(props.get("to_floor", props.get("floor", 1))),
                    edge_type=props.get("edge_type", "corridor"),
                    distance_m=float(dist),
                    bidirectional=bool(props.get("bidirectional", True)),
                    accessible=bool(props.get("accessible", True)),
                )
                self.add_edge(edge)

    def nodes_on_floor(self, floor: int) -> list[IndoorNode]:
        return [n for n in self.nodes.values() if n.floor == floor]

    def nearest_node(
        self,
        lat: float,
        lon: float,
        floor: int | None = None,
        node_types: list[str] | None = None,
    ) -> IndoorNode | None:
        """Find the closest node by Euclidean distance (lat/lon degrees)."""
        candidates = list(self.nodes.values())
        if floor is not None:
            candidates = [n for n in candidates if n.floor == floor]
        if node_types:
            candidates = [n for n in candidates if n.node_type in node_types]
        if not candidates:
            return None
        return min(candidates, key=lambda n: (n.lat - lat) ** 2 + (n.lon - lon) ** 2)

    def find_node_by_name(self, name: str, floor: int | None = None) -> IndoorNode | None:
        name_lower = name.lower()
        for n in self.nodes.values():
            if floor is not None and n.floor != floor:
                continue
            if name_lower in n.name.lower():
                return n
        return None


# ── A* router ─────────────────────────────────────────────────────────────────

class IndoorRouter:
    """
    A* router over an IndoorGraph.
    Supports multi-floor routing via stairs and elevator nodes.
    """

    def __init__(self, graph: IndoorGraph) -> None:
        self.graph = graph

    def route(
        self,
        origin_node_id: str,
        dest_node_id: str,
        *,
        prefer_accessible: bool = False,
        prefer_elevator: bool = False,
    ) -> IndoorRoute | None:
        """
        Find the shortest-time path between two nodes.
        Returns None if no path exists.
        """
        if origin_node_id not in self.graph.nodes or dest_node_id not in self.graph.nodes:
            return None
        if origin_node_id == dest_node_id:
            origin = self.graph.nodes[origin_node_id]
            return IndoorRoute(
                steps=[],
                total_distance_m=0.0,
                total_duration_s=0.0,
                floors_visited=[origin.floor],
                building_id=self.graph.building_id,
                origin_node=origin_node_id,
                destination_node=dest_node_id,
            )

        dest_node = self.graph.nodes[dest_node_id]

        # Priority queue: (f_cost, g_cost, node_id)
        open_heap: list[tuple[float, float, str]] = []
        heapq.heappush(open_heap, (0.0, 0.0, origin_node_id))

        g_cost: dict[str, float] = {origin_node_id: 0.0}
        came_from: dict[str, tuple[str, IndoorEdge] | None] = {origin_node_id: None}

        while open_heap:
            _, g, current_id = heapq.heappop(open_heap)

            if current_id == dest_node_id:
                return self._reconstruct(came_from, dest_node_id)

            if g > g_cost.get(current_id, float("inf")):
                continue  # stale entry

            for neighbor_id, edge in self.graph.adj.get(current_id, []):
                if prefer_accessible and not edge.accessible:
                    continue

                cost = edge.cost_s
                # Penalise stairs when elevator preferred
                if prefer_elevator and edge.edge_type == "stairs":
                    cost *= 3.0

                new_g = g + cost
                if new_g < g_cost.get(neighbor_id, float("inf")):
                    g_cost[neighbor_id] = new_g
                    came_from[neighbor_id] = (current_id, edge)
                    h = self._heuristic(neighbor_id, dest_node)
                    heapq.heappush(open_heap, (new_g + h, new_g, neighbor_id))

        return None  # no path

    def _heuristic(self, node_id: str, dest: IndoorNode) -> float:
        """Admissible heuristic: Euclidean distance / walk speed + floor penalty."""
        n = self.graph.nodes.get(node_id)
        if n is None:
            return 0.0
        dlat = (n.lat - dest.lat) * 111000
        dlon = (n.lon - dest.lon) * 111000 * math.cos(math.radians(n.lat))
        dist_m = math.sqrt(dlat ** 2 + dlon ** 2)
        floor_penalty = abs(n.floor - dest.floor) * STAIR_TIME_PER_FLOOR
        return dist_m / WALK_SPEED_MS + floor_penalty

    def _reconstruct(
        self,
        came_from: dict[str, tuple[str, IndoorEdge] | None],
        dest_id: str,
    ) -> IndoorRoute:
        path: list[tuple[str, IndoorEdge | None]] = []
        current = dest_id
        while came_from[current] is not None:
            prev_id, edge = came_from[current]  # type: ignore[misc]
            path.append((current, edge))
            current = prev_id
        path.reverse()

        steps: list[IndoorRouteStep] = []
        total_dist = 0.0
        total_time = 0.0
        floors: list[int] = []

        for to_id, edge in path:
            from_node = self.graph.nodes[edge.from_node]
            to_node = self.graph.nodes[edge.to_node]
            if from_node.floor not in floors:
                floors.append(from_node.floor)
            if to_node.floor not in floors:
                floors.append(to_node.floor)
            instruction = _vn_indoor_instruction(edge, from_node, to_node)
            steps.append(IndoorRouteStep(
                instruction=instruction,
                from_node_id=edge.from_node,
                to_node_id=edge.to_node,
                from_floor=edge.from_floor,
                to_floor=edge.to_floor,
                edge_type=edge.edge_type,
                distance_m=edge.distance_m,
                duration_s=edge.cost_s,
                lat=from_node.lat,
                lon=from_node.lon,
            ))
            total_dist += edge.distance_m
            total_time += edge.cost_s

        if path:
            last_node = self.graph.nodes[dest_id]
            if last_node.floor not in floors:
                floors.append(last_node.floor)

        # Walk came_from back to find the true origin node
        current = dest_id
        while came_from.get(current) is not None:
            prev_id, _ = came_from[current]  # type: ignore[misc]
            current = prev_id
        origin_id = current

        return IndoorRoute(
            steps=steps,
            total_distance_m=total_dist,
            total_duration_s=total_time,
            floors_visited=sorted(set(floors)),
            building_id=self.graph.building_id,
            origin_node=origin_id,
            destination_node=dest_id,
        )


# ── Building registry ─────────────────────────────────────────────────────────

class IndoorBuildingRegistry:
    """
    Holds all loaded IndoorGraph instances, keyed by building_id.
    Loaded lazily from DB floor_maps rows.
    """

    def __init__(self) -> None:
        self._graphs: dict[str, IndoorGraph] = {}

    def load_geojson(self, building_id: str, geojson: dict) -> IndoorGraph:
        graph = self._graphs.get(building_id) or IndoorGraph(building_id)
        graph.load_geojson(geojson)
        self._graphs[building_id] = graph
        return graph

    def get(self, building_id: str) -> IndoorGraph | None:
        return self._graphs.get(building_id)

    def list_buildings(self) -> list[str]:
        return list(self._graphs.keys())

    def get_router(self, building_id: str) -> IndoorRouter | None:
        g = self.get(building_id)
        return IndoorRouter(g) if g else None


# Singleton
indoor_registry = IndoorBuildingRegistry()


# ── Build from database ───────────────────────────────────────────────────────

async def build_indoor_graph_from_db(building_id: str = "main_building") -> IndoorGraph:
    """
    Build an IndoorGraph from locations and custom_edges in the database.
    This converts the flat DB schema into a proper multi-floor graph.
    """
    from core.database import db
    
    graph = IndoorGraph(building_id)
    
    # Load all locations as nodes
    locations = await db.fetchall("SELECT * FROM locations ORDER BY id")
    for loc in locations:
        node = IndoorNode(
            node_id=f"loc_{loc['id']}",
            building_id=building_id,
            floor=loc.get("floor", 1),
            name=loc["name"],
            node_type=_infer_node_type(loc),
            lat=float(loc["lat"]),
            lon=float(loc["lon"]),
            accessible=True,
            properties={"location_id": loc["id"], "category": loc.get("category", "")},
        )
        graph.add_node(node)
    
    # Load all custom_edges as edges
    edges = await db.fetchall("SELECT * FROM custom_edges ORDER BY id")
    for e in edges:
        # Find nearest nodes to edge endpoints
        from_node = _find_nearest_node_id(graph, e["from_lat"], e["from_lon"], e.get("from_floor", 1))
        to_node = _find_nearest_node_id(graph, e["to_lat"], e["to_lon"], e.get("to_floor", 1))
        
        if from_node and to_node:
            edge = IndoorEdge(
                edge_id=f"edge_{e['id']}",
                from_node=from_node,
                to_node=to_node,
                from_floor=e.get("from_floor", 1),
                to_floor=e.get("to_floor", 1),
                edge_type=_map_road_type_to_edge_type(e.get("road_type", "corridor")),
                distance_m=float(e.get("distance_m", 10.0)),
                bidirectional=bool(e.get("is_bidirectional", True)),
                accessible=True,
            )
            # Store additional properties for environmental analysis
            edge.is_covered = bool(e.get("is_covered", False))
            edge.surface = e.get("surface", "concrete")
            edge.slope_deg = float(e.get("slope_deg", 0))
            
            graph.add_edge(edge)
    
    # Register in global registry
    indoor_registry._graphs[building_id] = graph
    
    from loguru import logger
    logger.info(
        f"Built indoor graph '{building_id}': "
        f"{len(graph.nodes)} nodes, {sum(len(adj) for adj in graph.adj.values())} edges"
    )
    
    return graph


def _infer_node_type(loc: dict) -> str:
    """Infer node type from location name/category."""
    name = loc.get("name", "").lower()
    category = loc.get("category", "").lower()
    
    if "cầu thang" in name or "stairs" in name:
        return "stairs"
    if "thang máy" in name or "elevator" in name:
        return "elevator"
    if "sảnh" in name or "lobby" in name or "hall" in name:
        return "corridor"
    if "cổng" in name or "entrance" in name or "gate" in name:
        return "entrance"
    if "phòng" in name or "room" in category:
        return "room"
    if "hành lang" in name or "corridor" in name:
        return "corridor"
    
    return "corridor"  # default


def _map_road_type_to_edge_type(road_type: str) -> str:
    """Map database road_type to IndoorEdge edge_type."""
    mapping = {
        "stairs": "stairs",
        "elevator": "elevator",
        "corridor": "corridor",
        "alley": "corridor",
        "path": "corridor",
        "shortcut": "corridor",
        "door": "door",
    }
    return mapping.get(road_type, "corridor")


def _find_nearest_node_id(
    graph: IndoorGraph,
    lat: float,
    lon: float,
    floor: int,
    tolerance: float = 0.0002,  # ~20m - increased from 5m
) -> str | None:
    """Find the nearest node ID within tolerance."""
    candidates = [
        (nid, n) for nid, n in graph.nodes.items()
        if n.floor == floor
    ]
    if not candidates:
        return None
    
    # Return closest by Euclidean distance (no tolerance check - always return closest)
    best = min(candidates, key=lambda x: (x[1].lat - lat) ** 2 + (x[1].lon - lon) ** 2)
    dist = ((best[1].lat - lat) ** 2 + (best[1].lon - lon) ** 2) ** 0.5
    
    # Only apply tolerance if distance is significant
    if dist > tolerance:
        from loguru import logger
        logger.warning(
            f"Edge endpoint ({lat:.6f}, {lon:.6f}) floor {floor} is {dist*111000:.1f}m "
            f"from nearest node {best[0]} ({best[1].name})"
        )
    
    return best[0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _polyline_length_m(coords: list[list[float]]) -> float:
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        dlat = (lat2 - lat1) * 111000
        dlon = (lon2 - lon1) * 111000 * math.cos(math.radians(lat1))
        total += math.sqrt(dlat ** 2 + dlon ** 2)
    return total


def _vn_indoor_instruction(edge: IndoorEdge, from_node: IndoorNode, to_node: IndoorNode) -> str:
    dist_str = f"{int(edge.distance_m)} m" if edge.distance_m >= 1 else ""
    floor_delta = to_node.floor - from_node.floor

    if edge.edge_type == "stairs":
        if floor_delta > 0:
            return f"Đi lên cầu thang {from_node.name} → Tầng {to_node.floor}"
        elif floor_delta < 0:
            return f"Đi xuống cầu thang {from_node.name} → Tầng {to_node.floor}"
        else:
            return f"Qua cầu thang {from_node.name}"

    if edge.edge_type == "elevator":
        if floor_delta > 0:
            return f"Đi thang máy lên Tầng {to_node.floor}"
        elif floor_delta < 0:
            return f"Đi thang máy xuống Tầng {to_node.floor}"
        else:
            return f"Qua thang máy"

    if edge.edge_type == "door":
        return f"Qua cửa vào {to_node.name}"

    # corridor / default
    if to_node.node_type in ("room",):
        return f"Đi đến {to_node.name}{' — ' + dist_str if dist_str else ''}"
    return f"Đi theo hành lang{' — ' + dist_str if dist_str else ''} đến {to_node.name}"


def _render_indoor_html(route: IndoorRoute) -> str:
    """Generate a simple HTML card for the indoor route."""
    floor_badges = "".join(
        f'<span style="background:#334155;border-radius:4px;padding:2px 7px;font-size:11px;margin-right:4px">Tầng {f}</span>'
        for f in route.floors_visited
    )
    steps_html = "".join(
        f'<div style="padding:5px 0;border-bottom:1px solid #334155;font-size:12px">'
        f'<span style="color:#94a3b8;margin-right:6px">{i+1}.</span>{s.instruction}'
        f'<span style="float:right;color:#64748b">{int(s.duration_s)}s</span></div>'
        for i, s in enumerate(route.steps)
    )
    return (
        f'<div style="background:#1e293b;border-radius:10px;padding:14px;color:#e2e8f0">'
        f'<div style="font-weight:700;margin-bottom:8px">🏢 Lộ trình trong nhà</div>'
        f'<div style="margin-bottom:8px">{floor_badges}</div>'
        f'<div style="color:#94a3b8;font-size:12px;margin-bottom:10px">'
        f'Tổng: {route.total_distance_m:.0f} m · {route.total_duration_s/60:.1f} phút</div>'
        f'{steps_html}</div>'
    )
