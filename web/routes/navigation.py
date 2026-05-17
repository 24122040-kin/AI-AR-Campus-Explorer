from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from bot.session_manager import GPSFix, NavState, session_manager
from core.route_projection import build_ar_path, build_ar_path_floor_aware, should_use_vio
from core.database import db
from core.traffic_analyzer import traffic_analyzer
from core.vio_fusion import vio_registry
from routing.maneuver_plan import build_maneuver_plan
from routing.route_renderer import render_route_html, render_route_map
from web.state import get_router


router = APIRouter(tags=["navigation"])


class WaypointIn(BaseModel):
    lat: float
    lon: float


class AvoidDiscIn(BaseModel):
    lat: float
    lon: float
    radius_m: float = Field(default=120.0, ge=5.0, le=50_000.0)


class RouteRequest(BaseModel):
    origin: Optional[str] = None
    destination: str
    origin_lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    origin_lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    dest_lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    dest_lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    depart_hour: Optional[int] = None
    depart_minute: Optional[int] = None
    waypoints: list[WaypointIn] = Field(default_factory=list)
    avoid_discs: list[AvoidDiscIn] = Field(default_factory=list)
    alternates: int = Field(default=0, ge=0, le=3)
    session_id: Optional[str] = None
    begin_navigation: bool = False
    # GPS accuracy hint — triggers indoor routing when > threshold
    gps_accuracy_m: float = Field(default=5.0, ge=0.0, le=5000.0)
    weather_severity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avoid_uncovered: bool = False


class GPSUpdateRequest(BaseModel):
    session_id: str
    lat: float = Field(ge=-90.0, le=90.0, description="Latitude in degrees")
    lon: float = Field(ge=-180.0, le=180.0, description="Longitude in degrees")
    accuracy_m: float = Field(default=10.0, ge=0.0, le=10000.0, description="GPS accuracy in meters")
    speed_kmh: float = Field(default=0.0, ge=0.0, le=500.0, description="Speed in km/h")
    bearing: float = Field(default=0.0, ge=0.0, le=360.0, description="Bearing in degrees")


def _route_explanation(analysis: dict) -> dict:
    profile = analysis.get("selected_profile", analysis.get("strategy", "default"))
    reasons = [
        f"profile={profile}",
        f"congestion={analysis.get('avg_congestion', 0):.2f}",
        f"turns={analysis.get('turn_count', 0)}",
        f"landmarks={analysis.get('landmark_density', 0):.2f}",
        f"local_edges={analysis.get('custom_edge_ratio', 0):.2f}",
    ]
    return {
        "headline": f"Tuyen duoc chon theo profile {profile}",
        "reasons": reasons,
        "candidates": analysis.get("candidate_profiles", []),
    }


def _route_realtime_payload(
    route,
    reference_lat: float,
    reference_lon: float,
    session_id: str | None = None,
    gps_accuracy_m: float = 5.0,
) -> dict:
    """
    Build realtime payload with VIO-aware AR path.
    
    If VIO is available and conditions warrant (indoor, poor GPS, multi-floor),
    uses floor-aware AR path with VIO position fallback.
    """
    # Get VIO state if session exists
    vio = vio_registry.get(session_id) if session_id else None
    vio_pose = vio.get_pose() if vio else None
    
    # Get current floor from session (default to 1 if not available)
    current_floor = 1
    if session_id:
        sess = session_manager.get_or_create(session_id)
        # Floor detector would be attached to session in production
        # For now, check if route has floor info
        if route.steps and hasattr(route.steps[0], 'floor'):
            current_floor = route.steps[0].floor or 1
    
    # Check if route is indoor (has location IDs in steps)
    is_indoor = any(hasattr(s, 'location_id') and s.location_id for s in route.steps)
    
    # Decide whether to use VIO
    use_vio_mode = should_use_vio(
        gps_accuracy_m=gps_accuracy_m,
        indoor=is_indoor,
        floor=current_floor,
    )
    
    # Build AR path (floor-aware if multi-floor route detected)
    has_floor_changes = False
    if route.steps:
        floors = {getattr(s, 'floor', 1) for s in route.steps}
        has_floor_changes = len(floors) > 1
    
    if has_floor_changes and use_vio_mode:
        # Use floor-aware AR path for multi-floor indoor routes
        ar_path = build_ar_path_floor_aware(
            route=route,
            current_floor=current_floor,
            ref_lat=reference_lat,
            ref_lon=reference_lon,
            vio_pose=vio_pose,
        )
    else:
        # Use standard AR path for outdoor/single-floor routes
        ar_path = build_ar_path(
            route=route,
            ref_lat=reference_lat,
            ref_lon=reference_lon,
            vio_pose=vio_pose,
            use_vio=use_vio_mode,
            current_floor=current_floor,
        )
    
    return {
        "maneuver_plan": build_maneuver_plan(route),
        "ar_path": ar_path,
    }


@router.post("/api/route")
async def get_route(req: RouteRequest):
    runtime_router = get_router()
    if runtime_router is None:
        raise HTTPException(503, "Router not ready")

    depart = datetime.now()
    if req.depart_hour is not None:
        depart = depart.replace(hour=req.depart_hour, minute=req.depart_minute or 0, second=0)

    if req.origin:
        orig = await runtime_router.resolve_location(req.origin)
    elif req.origin_lat and req.origin_lon:
        orig = (req.origin_lat, req.origin_lon)
    else:
        raise HTTPException(400, "origin required")

    if req.dest_lat is not None and req.dest_lon is not None:
        dest = (req.dest_lat, req.dest_lon)
    else:
        dest = await runtime_router.resolve_location(req.destination)
    if not orig or not dest:
        raise HTTPException(404, "Cannot geocode origin or destination")

    wpl = [(w.lat, w.lon) for w in req.waypoints] if req.waypoints else None
    avoid = (
        [(a.lat, a.lon, a.radius_m) for a in req.avoid_discs] if req.avoid_discs else None
    )
    try:
        route = await runtime_router.find_route(
            orig[0],
            orig[1],
            dest[0],
            dest[1],
            depart,
            waypoints=wpl,
            avoid_discs=avoid,
            alternates=req.alternates,
            gps_accuracy_m=req.gps_accuracy_m,
            weather_severity=req.weather_severity,
            avoid_uncovered=req.avoid_uncovered,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    if not route:
        raise HTTPException(404, "No route found")

    if req.session_id:
        sess = session_manager.get_or_create(req.session_id)
        sess.current_route = route
        sess.origin = (orig[0], orig[1])
        sess.destination = (dest[0], dest[1])
        sess.current_step_idx = 0
        if req.begin_navigation:
            sess.state = NavState.NAVIGATING

    await traffic_analyzer.refresh()
    html_card = render_route_html(route, traffic_analyzer)
    map_html = render_route_map(route, traffic_analyzer)
    realtime_payload = _route_realtime_payload(
        route,
        orig[0],
        orig[1],
        session_id=req.session_id,
        gps_accuracy_m=req.gps_accuracy_m,
    )

    return {
        "ok": True,
        "distance_km": round(route.total_distance_m / 1000, 2),
        "duration_min": round(route.total_duration_min, 1),
        "analysis": route.analysis,
        "alternates": route.analysis.get("alternates", []),
        "explanation": _route_explanation(route.analysis),
        "steps": [
            {
                "instruction": s.instruction,
                "distance_m": round(s.distance_m, 1),
                "duration_s": round(s.duration_s, 1),
                "lat": s.lat,
                "lon": s.lon,
                "maneuver": s.maneuver,
                "street_name": s.street_name,
                "bearing": round(s.bearing, 1),
                "images": s.image_paths,
            }
            for s in route.steps
        ],
        "geometry": route.geometry,
        "maneuver_plan": realtime_payload["maneuver_plan"],
        "ar_path": realtime_payload["ar_path"],
        "html_card": html_card,
        "map_html": map_html,
    }


@router.get("/api/route/map", response_class=HTMLResponse)
async def route_map(from_q: str = "", to_q: str = "", depart_hour: int = -1):
    req = RouteRequest(
        origin=from_q or None,
        destination=to_q,
        depart_hour=depart_hour if depart_hour >= 0 else None,
    )
    result = await get_route(req)
    return HTMLResponse(result["map_html"])


@router.post("/api/gps")
async def gps_update(req: GPSUpdateRequest):
    fix = GPSFix(
        lat=req.lat,
        lon=req.lon,
        accuracy_m=req.accuracy_m,
        speed_kmh=req.speed_kmh,
        bearing=req.bearing,
    )
    event = await session_manager.process_gps_update(req.session_id, fix, get_router())
    sess = session_manager.get_or_create(req.session_id)
    nearby = await db.nearby_locations(req.lat, req.lon, radius_deg=0.0003)
    return {"ok": True, "nav_event": event, "nearby": nearby[:3], "session_state": sess.state.value}
