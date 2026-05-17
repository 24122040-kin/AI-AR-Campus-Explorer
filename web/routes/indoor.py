"""
web/routes/indoor.py — Indoor floor-map management and multi-floor routing API

Endpoints:
  POST   /api/indoor/map                — upload / replace a floor-plan GeoJSON
  GET    /api/indoor/map/{building_id}  — list floors for a building
  GET    /api/indoor/map/{building_id}/{floor} — get raw GeoJSON for one floor
  DELETE /api/indoor/map/{building_id}/{floor} — remove a floor plan
  GET    /api/indoor/buildings          — list all buildings
  POST   /api/indoor/route              — find indoor route (multi-floor A*)
  GET    /api/indoor/nodes              — nearby indoor nodes (for GPS snap)
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from core.database import db
from core.indoor_router import (
    IndoorGraph,
    IndoorRouter,
    indoor_registry,
    _polyline_length_m,
)

router = APIRouter(tags=["indoor"])

# ── Helpers ───────────────────────────────────────────────────────────────────

GPS_ACCURACY_INDOOR_THRESHOLD_M: float = 15.0  # switch to indoor when accuracy > this


def _extract_nodes_from_geojson(building_id: str, floor: int, geojson: dict) -> list[dict]:
    nodes = []
    for feat in geojson.get("features", []):
        if feat.get("geometry", {}).get("type") != "Point":
            continue
        props = feat.get("properties", {})
        coords = feat["geometry"]["coordinates"]
        nodes.append({
            "node_id": feat.get("id", ""),
            "name": props.get("name", feat.get("id", "")),
            "node_type": props.get("node_type", "corridor"),
            "lat": float(coords[1]),
            "lon": float(coords[0]),
            "accessible": bool(props.get("accessible", True)),
            "properties": props,
        })
    return nodes


def _bbox_center(geojson: dict) -> tuple[float, float] | None:
    lats, lons = [], []
    for feat in geojson.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") == "Point":
            c = geom["coordinates"]
            lons.append(c[0]); lats.append(c[1])
        elif geom.get("type") == "LineString":
            for c in geom["coordinates"]:
                lons.append(c[0]); lats.append(c[1])
    if not lats:
        return None
    return (sum(lats) / len(lats), sum(lons) / len(lons))


async def _load_building_into_registry(building_id: str) -> IndoorGraph | None:
    """Load all floor maps for a building from DB into the in-memory registry."""
    rows = await db.fetchall(
        "SELECT floor, geojson FROM floor_maps WHERE building_id=? ORDER BY floor",
        (building_id,),
    )
    if not rows:
        return None
    graph = IndoorGraph(building_id)
    for row in rows:
        try:
            gj = json.loads(row["geojson"])
            graph.load_geojson(gj)
        except Exception:
            pass
    indoor_registry._graphs[building_id] = graph
    return graph


# ── Request / Response models ─────────────────────────────────────────────────

class IndoorRouteRequest(BaseModel):
    building_id: str
    # Origin — either node_id or lat/lon + floor
    origin_node: Optional[str] = None
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    origin_floor: Optional[int] = None
    # Destination — either node_id or name + floor
    dest_node: Optional[str] = None
    dest_name: Optional[str] = None
    dest_floor: Optional[int] = None
    dest_lat: Optional[float] = None
    dest_lon: Optional[float] = None
    # Options
    prefer_accessible: bool = False
    prefer_elevator: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/api/indoor/map")
async def upload_floor_map(
    file: UploadFile = File(...),
    building_id: str = Form(...),
    floor: int = Form(...),
    name: str = Form(""),
):
    """
    Upload a GeoJSON floor plan for one floor of a building.
    Replaces any existing plan for the same building_id + floor.
    """
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, "Floor plan too large (max 10 MB)")

    try:
        geojson = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON: {e}")

    if geojson.get("type") != "FeatureCollection":
        raise HTTPException(400, "GeoJSON must be a FeatureCollection")

    # Inject building_id and floor into the GeoJSON if missing
    geojson.setdefault("building_id", building_id)
    geojson.setdefault("floor", floor)

    center = _bbox_center(geojson)
    lat_c = center[0] if center else None
    lon_c = center[1] if center else None

    map_name = name or f"{building_id} — Tầng {floor}"
    map_id = await db.upsert_floor_map(
        building_id=building_id,
        floor=floor,
        name=map_name,
        geojson=geojson,
        lat_center=lat_c,
        lon_center=lon_c,
    )

    # Sync nodes to floor_nodes table
    nodes = _extract_nodes_from_geojson(building_id, floor, geojson)
    node_count = await db.upsert_floor_nodes(building_id, floor, nodes)

    # Reload building into in-memory registry
    await _load_building_into_registry(building_id)

    return {
        "ok": True,
        "map_id": map_id,
        "building_id": building_id,
        "floor": floor,
        "name": map_name,
        "node_count": node_count,
        "lat_center": lat_c,
        "lon_center": lon_c,
    }


@router.get("/api/indoor/buildings")
async def list_buildings():
    """List all buildings that have at least one floor plan uploaded."""
    buildings = await db.list_buildings()
    return {"ok": True, "buildings": buildings}


@router.get("/api/indoor/map/{building_id}")
async def list_floors(building_id: str):
    """List all floor plans for a building (metadata only, no GeoJSON)."""
    floors = await db.list_floor_maps(building_id)
    if not floors:
        raise HTTPException(404, f"No floor maps found for building '{building_id}'")
    return {"ok": True, "building_id": building_id, "floors": floors}


@router.get("/api/indoor/map/{building_id}/{floor}")
async def get_floor_map(building_id: str, floor: int):
    """Return the raw GeoJSON for a specific floor."""
    row = await db.get_floor_map(building_id, floor)
    if not row:
        raise HTTPException(404, f"Floor {floor} not found for building '{building_id}'")
    geojson = json.loads(row["geojson"])
    return {
        "ok": True,
        "building_id": building_id,
        "floor": floor,
        "name": row["name"],
        "geojson": geojson,
    }


@router.delete("/api/indoor/map/{building_id}/{floor}")
async def delete_floor_map(building_id: str, floor: int):
    """Remove a floor plan and its nodes."""
    row = await db.get_floor_map(building_id, floor)
    if not row:
        raise HTTPException(404, f"Floor {floor} not found for building '{building_id}'")
    await db.execute(
        "DELETE FROM floor_maps WHERE building_id=? AND floor=?", (building_id, floor)
    )
    await db.execute(
        "DELETE FROM floor_nodes WHERE building_id=? AND floor=?", (building_id, floor)
    )
    # Reload registry
    await _load_building_into_registry(building_id)
    return {"ok": True, "building_id": building_id, "floor": floor, "deleted": True}


@router.post("/api/indoor/route")
async def indoor_route(req: IndoorRouteRequest):
    """
    Find the shortest-time indoor route between two points (multi-floor A*).

    Origin and destination can be specified as:
      - node_id (exact match)
      - lat/lon + floor (snapped to nearest node)
      - name + floor (fuzzy name search)

    Returns step-by-step instructions with floor transitions.
    Example output:
      Tầng 1 → Cầu thang A → Tầng 3 → Phòng 302
    """
    # Ensure building is loaded
    graph = indoor_registry.get(req.building_id)
    if graph is None:
        graph = await _load_building_into_registry(req.building_id)
    if graph is None or not graph.nodes:
        raise HTTPException(404, f"No floor maps loaded for building '{req.building_id}'")

    indoor_r = IndoorRouter(graph)

    # ── Resolve origin ────────────────────────────────────────────────────────
    origin_node_id = req.origin_node
    if not origin_node_id:
        if req.origin_lat is not None and req.origin_lon is not None:
            n = graph.nearest_node(req.origin_lat, req.origin_lon, floor=req.origin_floor)
            if n is None:
                raise HTTPException(404, "Cannot find indoor node near origin coordinates")
            origin_node_id = n.node_id
        else:
            raise HTTPException(400, "Provide origin_node or origin_lat/lon")

    # ── Resolve destination ───────────────────────────────────────────────────
    dest_node_id = req.dest_node
    if not dest_node_id:
        if req.dest_name:
            n = graph.find_node_by_name(req.dest_name, floor=req.dest_floor)
            if n is None:
                raise HTTPException(404, f"Cannot find indoor node named '{req.dest_name}'")
            dest_node_id = n.node_id
        elif req.dest_lat is not None and req.dest_lon is not None:
            n = graph.nearest_node(req.dest_lat, req.dest_lon, floor=req.dest_floor)
            if n is None:
                raise HTTPException(404, "Cannot find indoor node near destination coordinates")
            dest_node_id = n.node_id
        else:
            raise HTTPException(400, "Provide dest_node, dest_name, or dest_lat/lon")

    if origin_node_id not in graph.nodes:
        raise HTTPException(404, f"Origin node '{origin_node_id}' not found in building")
    if dest_node_id not in graph.nodes:
        raise HTTPException(404, f"Destination node '{dest_node_id}' not found in building")

    route = indoor_r.route(
        origin_node_id,
        dest_node_id,
        prefer_accessible=req.prefer_accessible,
        prefer_elevator=req.prefer_elevator,
    )

    if route is None:
        raise HTTPException(404, "No indoor route found between the specified nodes")

    result = route.as_dict()

    # Add human-readable summary
    origin_name = graph.nodes[origin_node_id].name
    dest_name = graph.nodes[dest_node_id].name
    result["summary"] = _build_summary(route, graph, origin_name, dest_name)

    return result


@router.get("/api/indoor/nodes")
async def nearby_indoor_nodes(
    lat: float,
    lon: float,
    floor: Optional[int] = None,
    radius: float = 0.001,
):
    """
    Return indoor nodes near a GPS coordinate.
    Used to snap outdoor GPS position to the indoor graph.
    """
    nodes = await db.nearby_floor_nodes(lat, lon, radius_deg=radius, floor=floor)
    return {"ok": True, "nodes": nodes, "count": len(nodes)}


# ── Summary builder ───────────────────────────────────────────────────────────

def _build_summary(route, graph, origin_name: str, dest_name: str) -> str:
    """
    Build a human-readable summary like:
    Tầng 1 → Cầu thang A → Tầng 3 → Phòng 302
    """
    if not route.floors_visited:
        return f"{origin_name} → {dest_name}"

    parts: list[str] = [f"Tầng {route.floors_visited[0]}"]
    prev_floor = route.floors_visited[0]

    for step in route.steps:
        if step.edge_type in ("stairs", "elevator"):
            node = graph.nodes.get(step.from_node_id)
            label = node.name if node else ("Cầu thang" if step.edge_type == "stairs" else "Thang máy")
            if label not in parts:
                parts.append(label)
            if step.to_floor != prev_floor:
                parts.append(f"Tầng {step.to_floor}")
                prev_floor = step.to_floor

    # Final destination
    dest_node = graph.nodes.get(route.destination_node)
    if dest_node and dest_node.name not in parts:
        parts.append(dest_node.name)

    return " → ".join(parts)
