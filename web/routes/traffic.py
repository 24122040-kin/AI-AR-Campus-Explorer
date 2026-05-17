from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from core.database import db
from core.environmental_analyzer import environmental_analyzer
from core.traffic_analyzer import IsochroneGenerator, traffic_analyzer
from routing.route_renderer import render_traffic_timeline
from web.state import get_router


router = APIRouter(tags=["traffic"])


class TrafficObsRequest(BaseModel):
    lat: float
    lon: float
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    speed_kmh: Optional[float] = None
    congestion: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class EnvironmentObsRequest(BaseModel):
    lat: float
    lon: float
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    crowd_level: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    weather_severity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: str = ""


@router.get("/api/traffic/timeline")
async def traffic_timeline(weekday: Optional[int] = None):
    await traffic_analyzer.refresh()
    wd = weekday if weekday is not None else datetime.now().weekday()
    return {"curve": traffic_analyzer.full_day_curve(wd), "html": render_traffic_timeline(traffic_analyzer, wd)}


@router.get("/api/traffic/best-time")
async def best_time(hour: int, weekday: Optional[int] = None):
    await traffic_analyzer.refresh()
    wd = weekday if weekday is not None else datetime.now().weekday()
    return traffic_analyzer.best_departure_window(hour, 2, wd)


@router.post("/api/traffic")
async def add_traffic(req: TrafficObsRequest):
    obs_id = await db.add_traffic_obs(
        lat=req.lat,
        lon=req.lon,
        hour=req.hour,
        weekday=req.weekday,
        speed_kmh=req.speed_kmh,
        congestion=req.congestion,
    )
    await traffic_analyzer.refresh(force=True)
    runtime_router = get_router()
    if runtime_router:
        await runtime_router.heuristic.warm_cache()
    return {"ok": True, "id": obs_id}


@router.post("/api/environment")
async def add_environment(req: EnvironmentObsRequest):
    obs_id = await db.add_environment_obs(
        lat=req.lat,
        lon=req.lon,
        hour=req.hour,
        weekday=req.weekday,
        crowd_level=req.crowd_level,
        weather_severity=req.weather_severity,
        notes=req.notes,
    )
    await environmental_analyzer.refresh(force=True)
    return {"ok": True, "id": obs_id}


@router.get("/api/traffic/heatmap")
async def traffic_heatmap():
    await traffic_analyzer.refresh()
    return {"data": traffic_analyzer.heatmap_data()}


@router.get("/api/isochrone", response_class=HTMLResponse)
async def isochrone(lat: float, lon: float, minutes: str = "5,10,15", depart_hour: int = -1):
    runtime_router = get_router()
    if runtime_router is None or not runtime_router.osm.G:
        raise HTTPException(503, "Router not ready")

    mins = [int(m) for m in minutes.split(",") if m.strip().isdigit()]
    depart = datetime.now()
    if depart_hour >= 0:
        depart = depart.replace(hour=depart_hour)

    gen = IsochroneGenerator(runtime_router.osm, traffic_analyzer)
    polygons = gen.generate(lat, lon, mins, depart)

    import folium

    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")
    colors = ["#22c55e", "#f59e0b", "#ef4444"]
    for i, (min_val, coords) in enumerate(sorted(polygons.items())):
        if len(coords) >= 3:
            folium.Polygon(
                locations=coords,
                color=colors[min(i, 2)],
                fill_color=colors[min(i, 2)],
                fill_opacity=0.12,
                weight=2,
                tooltip=f"Trong {min_val} phut",
            ).add_to(m)
    folium.Marker([lat, lon], popup="Xuat phat", icon=folium.Icon(color="blue", icon="star", prefix="fa")).add_to(m)
    return m._repr_html_()
