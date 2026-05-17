"""
web/app.py — FastAPI application — Complete version v2
"""
from __future__ import annotations
import asyncio
import json
import mimetypes
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncIterator, TYPE_CHECKING, Any

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from loguru import logger

from config.settings import settings
from core.database import db
from core.environmental_analyzer import environmental_analyzer
from core.realtime_manager import RealtimeSessionManager
from core.route_projection import build_ar_path
from core.traffic_analyzer import TrafficAnalyzer, IsochroneGenerator, traffic_analyzer
from core.image_manager import BatchImageImporter, read_gps_exif
from routing.maneuver_plan import build_maneuver_plan
from routing.router import NavRouter
from routing.route_renderer import render_route_html, render_route_map, render_traffic_timeline
from bot.nav_bot import NavBot
from bot.session_manager import SessionManager, NavState, GPSFix, session_manager
from web.jobs import job_store
from web.state import get_router, get_bot, set_runtime_state
from web.uploads import build_upload_path, ensure_safe_batch_folder, validate_upload, MAX_UPLOAD_SIZE_BYTES
from web.routes.chat import router as chat_router
from web.routes.data import router as data_router
from web.routes.experimental import router as experimental_router
from web.routes.indoor import router as indoor_router
from web.routes.speech import router as speech_router
from web.routes.map_data import router as map_data_router
from web.routes.navigation import router as navigation_router
from web.routes.realtime import router as realtime_router
from web.routes.system import router as system_router
from web.routes.traffic import router as traffic_router
from web.routes.vpr import router as vpr_router

if TYPE_CHECKING:
    from core.vpr_engine import VPREngine

app = FastAPI(title="LocalNavBot", version="2.0.0")
_cors_origins = settings.cors_origin_list or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=("*" not in _cors_origins),  # credentials not allowed with wildcard
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.include_router(experimental_router)
app.include_router(system_router)
app.include_router(vpr_router)
app.include_router(chat_router)
app.include_router(data_router)
app.include_router(navigation_router)
app.include_router(realtime_router)
app.include_router(traffic_router)
app.include_router(map_data_router)
app.include_router(indoor_router)
app.include_router(speech_router)

# Serve ar_renderer.js as a static asset
_web_dir = Path(__file__).parent

@app.get("/ar_renderer.js")
async def serve_ar_renderer():
    p = _web_dir / "ar_renderer.js"
    if not p.exists():
        raise HTTPException(404, "ar_renderer.js not found")
    return FileResponse(str(p), media_type="application/javascript")


@app.get("/vio_client.js")
async def serve_vio_client():
    p = _web_dir / "vio_client.js"
    if not p.exists():
        raise HTTPException(404, "vio_client.js not found")
    return FileResponse(str(p), media_type="application/javascript")


@app.get("/static/css/{filename}")
async def serve_css(filename: str):
    p = _web_dir / "static" / "css" / filename
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="text/css")


@app.get("/static/js/{filename}")
async def serve_js(filename: str):
    p = _web_dir / "static" / "js" / filename
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="application/javascript")

_router: Optional[NavRouter] = None
_vpr: Optional[Any] = None
_bot: Optional[NavBot] = None
_realtime_manager: Optional[RealtimeSessionManager] = None


def _build_vpr() -> Any:
    from core.vpr_engine import VPREngine

    return VPREngine()


async def _build_vpr_async() -> Any:
    """Load VPR engine in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _build_vpr)


def _build_realtime_route_payload(route, reference_lat: float, reference_lon: float) -> dict:
    return {
        "maneuver_plan": build_maneuver_plan(route),
        "ar_path": build_ar_path(route, reference_lat, reference_lon),
    }


@app.on_event("startup")
async def startup():
    global _router, _vpr, _bot, _realtime_manager
    settings.setup_dirs()
    await db.init()
    _router = NavRouter()
    await _router.init()
    try:
        _vpr = await _build_vpr_async()
    except Exception as e:
        logger.warning(f"VPR: {e}")
        _vpr = None
    _bot = NavBot(_router, _vpr)
    _realtime_manager = RealtimeSessionManager(_router, _vpr)
    set_runtime_state(router=_router, vpr=_vpr, bot=_bot, realtime_manager=_realtime_manager)
    session_manager.set_nav_router(_router)
    await traffic_analyzer.refresh(force=True)
    await environmental_analyzer.refresh(force=True)
    await session_manager.start()

    # Pre-load all indoor floor maps into the in-memory registry
    try:
        from core.indoor_router import indoor_registry, IndoorGraph, build_indoor_graph_from_db
        import json as _json
        
        # Method 1: Load from floor_maps table (GeoJSON format)
        buildings = await db.list_buildings()
        for bld in buildings:
            bid = bld["building_id"]
            rows = await db.fetchall(
                "SELECT floor, geojson FROM floor_maps WHERE building_id=? ORDER BY floor",
                (bid,),
            )
            if rows:
                graph = IndoorGraph(bid)
                for row in rows:
                    try:
                        graph.load_geojson(_json.loads(row["geojson"]))
                    except Exception as e:
                        logger.warning(f"Indoor map load error ({bid} floor {row['floor']}): {e}")
                indoor_registry._graphs[bid] = graph
                logger.info(f"Indoor: loaded {len(rows)} floor(s) for building '{bid}'")
        
        # Method 2: Build from locations + custom_edges (if no floor_maps exist)
        if not buildings:
            logger.info("No floor_maps found, building indoor graph from locations + custom_edges...")
            graph = await build_indoor_graph_from_db("main_building")
            if len(graph.nodes) > 0:
                logger.info(
                    f"Indoor: built graph from DB with {len(graph.nodes)} nodes, "
                    f"{sum(len(adj) for adj in graph.adj.values())} edges"
                )
            else:
                logger.warning("No indoor data found in database")
                
    except Exception as e:
        logger.warning(f"Indoor map pre-load: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    logger.info("LocalNavBot v2 ready")


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    session_id: Optional[str] = "default"

class LocationRequest(BaseModel):
    name: str
    lat: float
    lon: float
    description: str = ""
    category: str = "general"
    importance: int = Field(default=1, ge=1, le=5)
    tags: list[str] = []

class POIRequest(BaseModel):
    name: str
    poi_type: str
    lat: float
    lon: float
    address: str = ""
    notes: str = ""

class EdgeRequest(BaseModel):
    from_lat: float; from_lon: float
    to_lat: float;   to_lon: float
    name: str = ""; road_type: str = "alley"
    bidirectional: bool = True; notes: str = ""

class TrafficObsRequest(BaseModel):
    lat: float; lon: float
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

class _WaypointIn(BaseModel):
    lat: float
    lon: float


class _AvoidDiscIn(BaseModel):
    lat: float
    lon: float
    radius_m: float = Field(default=120.0, ge=5.0, le=50_000.0)


class RouteRequest(BaseModel):
    origin: Optional[str] = None
    destination: str
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    depart_hour: Optional[int] = None
    depart_minute: Optional[int] = None
    waypoints: list[_WaypointIn] = Field(default_factory=list)
    avoid_discs: list[_AvoidDiscIn] = Field(default_factory=list)
    alternates: int = Field(default=0, ge=0, le=3)
    session_id: Optional[str] = None
    begin_navigation: bool = False

class GPSUpdateRequest(BaseModel):
    session_id: str
    lat: float; lon: float
    accuracy_m: float = 10.0
    speed_kmh: float = 0.0
    bearing: float = 0.0


async def _run_batch_import_job(job_id: str, folder: Path, auto_caption: bool, min_quality: float) -> None:
    try:
        job_store.update(job_id, status="running", message=f"Importing images from {folder}")
        importer = BatchImageImporter(do_captions=auto_caption, min_quality=min_quality)
        result = await importer.import_folder(folder)
        job_store.update(job_id, status="running", message="Refreshing VPR index after import", result=result)
        if _bot:
            await _bot.rebuild_vpr_index()
        job_store.update(job_id, status="completed", message="Batch import completed", result=result)
    except Exception as e:
        logger.error(traceback.format_exc())
        job_store.update(job_id, status="failed", message="Batch import failed", error=str(e))


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/_legacy/chat")
async def legacy_chat(req: ChatRequest):
    sess = session_manager.get_or_create(req.session_id or "default")
    bot = NavBot(get_router(), _vpr)
    bot._history = sess.recent_history()
    try:
        response = await bot.ask(req.message, user_lat=req.lat, user_lon=req.lon)
        sess.add_message("user", req.message)
        sess.add_message("assistant", response)
        return {"response": response, "ok": True}
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))


@app.post("/api/_legacy/chat/stream")
async def legacy_chat_stream(req: ChatRequest):
    sess = session_manager.get_or_create(req.session_id or "default")
    bot = NavBot(get_router(), _vpr)
    bot._history = sess.recent_history()

    async def sse() -> AsyncIterator[str]:
        full = ""
        try:
            async for chunk in bot.stream(req.message, user_lat=req.lat, user_lon=req.lon):
                full += chunk
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
            sess.add_message("user", req.message)
            sess.add_message("assistant", full)
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.websocket("/ws/_legacy/chat")
async def legacy_ws_chat(websocket: WebSocket):
    await websocket.accept()
    sid = f"ws_{id(websocket)}"
    sess = session_manager.get_or_create(sid)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "chat")

            if msg_type == "gps":
                fix = GPSFix(lat=data["lat"], lon=data["lon"],
                             accuracy_m=data.get("accuracy", 10),
                             speed_kmh=data.get("speed", 0),
                             bearing=data.get("bearing", 0))
                event = await session_manager.process_gps_update(sid, fix, get_router())
                await websocket.send_json({"type": "nav_event", **event})

            elif msg_type == "chat":
                message = data.get("message", "")
                lat, lon = data.get("lat"), data.get("lon")
                bot = NavBot(get_router(), _vpr)
                bot._history = sess.recent_history()
                await websocket.send_json({"type": "start"})
                full = ""
                async for chunk in bot.stream(message, user_lat=lat, user_lon=lon):
                    full += chunk
                    await websocket.send_json({"type": "chunk", "text": chunk})
                sess.add_message("user", message)
                sess.add_message("assistant", full)
                await websocket.send_json({"type": "end", "full": full})

            elif msg_type == "start_nav":
                sess.state = NavState.NAVIGATING
                await websocket.send_json({"type": "nav_started"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── Route ─────────────────────────────────────────────────────────────────────

@app.post("/api/_legacy/route")
async def legacy_get_route(req: RouteRequest):
    router = get_router()
    depart = datetime.now()
    if req.depart_hour is not None:
        depart = depart.replace(hour=req.depart_hour, minute=req.depart_minute or 0, second=0)

    if req.origin:
        orig = await router.resolve_location(req.origin)
    elif req.origin_lat and req.origin_lon:
        orig = (req.origin_lat, req.origin_lon)
    else:
        raise HTTPException(400, "origin required")

    dest = await router.resolve_location(req.destination)
    if not orig or not dest:
        raise HTTPException(404, "Cannot geocode origin or destination")

    wpl = [(w.lat, w.lon) for w in req.waypoints] if req.waypoints else None
    avoid = (
        [(a.lat, a.lon, a.radius_m) for a in req.avoid_discs] if req.avoid_discs else None
    )
    try:
        route = await router.find_route(
            orig[0],
            orig[1],
            dest[0],
            dest[1],
            depart,
            waypoints=wpl,
            avoid_discs=avoid,
            alternates=req.alternates,
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
    realtime_payload = _build_realtime_route_payload(route, orig[0], orig[1])

    return {
        "ok": True,
        "distance_km": round(route.total_distance_m / 1000, 2),
        "duration_min": round(route.total_duration_min, 1),
        "analysis": route.analysis,
        "steps": [
            {"instruction": s.instruction, "distance_m": round(s.distance_m, 1),
             "duration_s": round(s.duration_s, 1), "lat": s.lat, "lon": s.lon,
             "maneuver": s.maneuver, "images": s.image_paths}
            for s in route.steps
        ],
        "geometry": route.geometry,
        "maneuver_plan": realtime_payload["maneuver_plan"],
        "ar_path": realtime_payload["ar_path"],
        "html_card": html_card,
        "map_html": map_html,
    }


@app.get("/api/_legacy/route/map", response_class=HTMLResponse)
async def legacy_route_map(from_q: str = "", to_q: str = "", depart_hour: int = -1):
    req = RouteRequest(
        origin=from_q or None, destination=to_q,
        depart_hour=depart_hour if depart_hour >= 0 else None,
    )
    result = await legacy_get_route(req)
    return HTMLResponse(result["map_html"])


# ── GPS live ──────────────────────────────────────────────────────────────────

@app.post("/api/_legacy/gps")
async def legacy_gps_update(req: GPSUpdateRequest):
    fix = GPSFix(lat=req.lat, lon=req.lon, accuracy_m=req.accuracy_m,
                 speed_kmh=req.speed_kmh, bearing=req.bearing)
    event = await session_manager.process_gps_update(req.session_id, fix, get_router())
    sess = session_manager.get_or_create(req.session_id)
    nearby = await db.nearby_locations(req.lat, req.lon, radius_deg=0.0003)
    return {"ok": True, "nav_event": event, "nearby": nearby[:3],
            "session_state": sess.state.value}


# ── Traffic ───────────────────────────────────────────────────────────────────

@app.get("/api/_legacy/traffic/timeline")
async def legacy_traffic_timeline(weekday: Optional[int] = None):
    await traffic_analyzer.refresh()
    wd = weekday if weekday is not None else datetime.now().weekday()
    return {
        "curve": traffic_analyzer.full_day_curve(wd),
        "html": render_traffic_timeline(traffic_analyzer, wd),
    }


@app.get("/api/_legacy/traffic/best-time")
async def legacy_best_time(hour: int, weekday: Optional[int] = None):
    await traffic_analyzer.refresh()
    wd = weekday if weekday is not None else datetime.now().weekday()
    return traffic_analyzer.best_departure_window(hour, 2, wd)


@app.post("/api/_legacy/traffic")
async def legacy_add_traffic(req: TrafficObsRequest):
    obs_id = await db.add_traffic_obs(lat=req.lat, lon=req.lon, hour=req.hour,
                                      weekday=req.weekday, speed_kmh=req.speed_kmh,
                                      congestion=req.congestion)
    await traffic_analyzer.refresh(force=True)
    if _router:
        await _router.heuristic.warm_cache()
    return {"ok": True, "id": obs_id}


@app.post("/api/_legacy/environment")
async def legacy_add_environment(req: EnvironmentObsRequest):
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


@app.get("/api/_legacy/traffic/heatmap")
async def legacy_traffic_heatmap():
    await traffic_analyzer.refresh()
    return {"data": traffic_analyzer.heatmap_data()}


# ── Isochrone ─────────────────────────────────────────────────────────────────

@app.get("/api/_legacy/isochrone", response_class=HTMLResponse)
async def legacy_isochrone(lat: float, lon: float, minutes: str = "5,10,15", depart_hour: int = -1):
    if _router is None or not _router.osm.G:
        raise HTTPException(503, "Router not ready")
    mins = [int(m) for m in minutes.split(",") if m.strip().isdigit()]
    depart = datetime.now()
    if depart_hour >= 0:
        depart = depart.replace(hour=depart_hour)

    gen = IsochroneGenerator(_router.osm, traffic_analyzer)
    polygons = gen.generate(lat, lon, mins, depart)

    import folium
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")
    colors = ["#22c55e", "#f59e0b", "#ef4444"]
    for i, (min_val, coords) in enumerate(sorted(polygons.items())):
        if len(coords) >= 3:
            folium.Polygon(locations=coords, color=colors[min(i, 2)],
                           fill_color=colors[min(i, 2)], fill_opacity=0.12,
                           weight=2, tooltip=f"Trong {min_val} phút").add_to(m)
    folium.Marker([lat, lon], popup="Xuất phát",
                  icon=folium.Icon(color="blue", icon="star", prefix="fa")).add_to(m)
    return m._repr_html_()


# ── Data ingestion ────────────────────────────────────────────────────────────

@app.post("/api/_legacy/upload/image")
async def legacy_upload_image(
    file: UploadFile = File(...),
    location_id: Optional[int] = Form(None),
    location_name: str = Form(""),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    caption: str = Form(""),
    category: str = Form("general"),
    importance: int = Form(1),
    auto_caption: bool = Form(False),
):
    suffix = validate_upload(file)
    data = await file.read()
    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(400, f"File too large. Max size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.")
    dest = build_upload_path(suffix)
    async with aiofiles.open(dest, "wb") as f:
        await f.write(data)

    gps = read_gps_exif(dest)
    if gps:
        lat = lat or gps[0]
        lon = lon or gps[1]

    if lat is None or lon is None:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "GPS required")

    if location_id is None:
        loc_name = location_name or f"Loc_{dest.stem[:12]}"
        location_id = await db.add_location(name=loc_name, lat=lat, lon=lon,
                                            category=category, importance=importance)

    if auto_caption and not caption:
        from core.image_manager import auto_caption as _ac
        loc = await db.get_location(location_id)
        caption = await _ac(dest, f"Gần {loc['name']}" if loc else "")

    img_id = await db.add_image(location_id=location_id, filename=dest.name,
                                filepath=str(dest), caption=caption)

    faiss_id = -1
    if _vpr and _vpr.aggregator._fitted:
        from core.vpr_engine import ImageMeta
        loc = await db.get_location(location_id)
        meta = ImageMeta(image_id=img_id, location_id=location_id,
                         location_name=loc["name"] if loc else "",
                         lat=lat, lon=lon, filepath=str(dest), caption=caption)
        try:
            faiss_id = _vpr.index_image(dest, meta)
            await db.update_faiss_id(img_id, faiss_id)
            _vpr._index.save()
        except Exception as e:
            logger.warning(f"VPR index: {e}")

    return {"ok": True, "image_id": img_id, "location_id": location_id,
            "faiss_id": faiss_id, "lat": lat, "lon": lon, "caption": caption}


@app.post("/api/_legacy/upload/batch")
async def legacy_batch_import(
    background_tasks: BackgroundTasks,
    folder: str = Form(...),
    auto_caption: bool = Form(False),
    min_quality: float = Form(0.25),
):
    if not 0.0 <= min_quality <= 1.0:
        raise HTTPException(400, "min_quality must be between 0.0 and 1.0")
    p = ensure_safe_batch_folder(folder)
    job = job_store.create("batch_import", f"Queued batch import for {p}")
    background_tasks.add_task(_run_batch_import_job, job.job_id, p, auto_caption, min_quality)
    return {"ok": True, "job": job.as_dict()}


@app.post("/api/_legacy/location")
async def legacy_add_location(req: LocationRequest):
    loc_id = await db.add_location(name=req.name, lat=req.lat, lon=req.lon,
                                   description=req.description, category=req.category,
                                   importance=req.importance, tags=req.tags)
    return {"ok": True, "id": loc_id}


@app.post("/api/_legacy/poi")
async def legacy_add_poi(req: POIRequest):
    poi_id = await db.add_poi(name=req.name, poi_type=req.poi_type, lat=req.lat,
                              lon=req.lon, address=req.address, notes=req.notes)
    return {"ok": True, "id": poi_id}


@app.post("/api/_legacy/edge")
async def legacy_add_edge(req: EdgeRequest):
    eid, _dist_m = await db.add_custom_edge(from_lat=req.from_lat, from_lon=req.from_lon,
                                            to_lat=req.to_lat, to_lon=req.to_lon, name=req.name,
                                            road_type=req.road_type, bidirectional=req.bidirectional)
    if _router and _router.osm.G:
        await _router.osm.patch_custom_edges()
    return {"ok": True, "id": eid}


# ── VPR ───────────────────────────────────────────────────────────────────────

@app.post("/api/_legacy/vpr/query")
async def legacy_vpr_query(file: UploadFile = File(...),
                           lat: Optional[float] = Form(None), lon: Optional[float] = Form(None)):
    from web.routes.vpr import vpr_query

    return await vpr_query(file=file, lat=lat, lon=lon)


@app.post("/api/_legacy/vpr/rebuild")
async def legacy_vpr_rebuild(background_tasks: BackgroundTasks):
    from web.routes.vpr import vpr_rebuild

    return await vpr_rebuild(background_tasks=background_tasks)


# ── Map ───────────────────────────────────────────────────────────────────────

@app.get("/api/_legacy/map", response_class=HTMLResponse)
async def legacy_get_map(lat: float = settings.map_default_lat, lon: float = settings.map_default_lon,
                  zoom: int = settings.map_default_zoom):
    import folium
    from folium.plugins import MarkerCluster, HeatMap, MeasureControl
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap")
    MeasureControl(primary_length_unit="meters").add_to(m)
    cluster = MarkerCluster(name="Địa điểm").add_to(m)
    for loc in await db.fetchall("SELECT * FROM locations LIMIT 1000"):
        imgs = await db.get_images_for_location(loc["id"])
        img_html = (f'<br><img src="/api/image/{imgs[0]["id"]}" width="200" style="border-radius:6px"/>'
                    if imgs else "")
        color = {1:"blue",2:"blue",3:"orange",4:"red",5:"darkred"}.get(loc.get("importance",1),"blue")
        folium.Marker([loc["lat"], loc["lon"]],
            popup=folium.Popup(f"<b>{loc['name']}</b><br>{loc.get('description','')}{img_html}", max_width=240),
            tooltip=loc["name"],
            icon=folium.Icon(color=color, icon="camera", prefix="fa")).add_to(cluster)
    poi_layer = folium.FeatureGroup(name="POI local").add_to(m)
    for poi in await db.fetchall("SELECT * FROM pois WHERE is_active=1 LIMIT 500"):
        folium.CircleMarker([poi["lat"], poi["lon"]], radius=7, color="#e74c3c",
                            fill=True, fill_color="#e74c3c", fill_opacity=0.8,
                            popup=f"{poi['name']} ({poi['type']})").add_to(poi_layer)
    edge_layer = folium.FeatureGroup(name="Đường tắt/hẻm").add_to(m)
    for e in await db.get_all_custom_edges():
        folium.PolyLine([(e["from_lat"], e["from_lon"]), (e["to_lat"], e["to_lon"])],
                        color="#27ae60", weight=4, opacity=0.85).add_to(edge_layer)
    heat = traffic_analyzer.heatmap_data()
    if heat:
        HeatMap([[d["lat"], d["lon"], d["intensity"]] for d in heat],
                name="Tắc nghẽn", radius=22, blur=18,
                gradient={"0.0":"blue","0.4":"lime","0.7":"orange","1.0":"red"}).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


@app.get("/api/_legacy/image/{image_id}")
async def legacy_get_image(image_id: int):
    row = await db.fetchone("SELECT filepath FROM images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(404)
    p = Path(row["filepath"])
    if not p.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(str(p), media_type=mimetypes.guess_type(str(p))[0] or "image/jpeg")


# ── Query ─────────────────────────────────────────────────────────────────────

@app.get("/api/_legacy/nearby")
async def legacy_nearby(lat: float, lon: float, radius: float = 0.01):
    return {"locations": await db.nearby_locations(lat, lon, radius),
            "pois": await db.nearby_pois(lat, lon, radius)}

@app.get("/api/_legacy/search")
async def legacy_search(q: str):
    return {"locations": await db.search_locations(q), "pois": await db.search_pois(q)}

@app.get("/api/_legacy/locations")
async def legacy_list_locations(limit: int = 100, offset: int = 0):
    return await db.fetchall(
        "SELECT l.*, COUNT(i.id) AS image_count FROM locations l "
        "LEFT JOIN images i ON i.location_id=l.id "
        "GROUP BY l.id ORDER BY l.importance DESC LIMIT ? OFFSET ?",
        (limit, offset))

@app.get("/api/_legacy/status")
async def legacy_status():
    valhalla_ok = await _router.valhalla.is_healthy() if _router else False
    vpr_ok = _vpr is not None and _vpr.aggregator._fitted
    return {
        "status": "ok",
        "valhalla": valhalla_ok,
        "osm_graph_cached": bool(_router and _router.osm._graph_path.exists()),
        "vpr_ready": vpr_ok,
        "vpr_index_size": _vpr._index.size if _vpr and _vpr._index else 0,
        "vpr_backend": getattr(getattr(_vpr, "extractor", None), "backend", None),
        "locations": (await db.fetchone("SELECT COUNT(*) AS n FROM locations") or {}).get("n", 0),
        "pois":      (await db.fetchone("SELECT COUNT(*) AS n FROM pois") or {}).get("n", 0),
        "images":    (await db.fetchone("SELECT COUNT(*) AS n FROM images") or {}).get("n", 0),
        "sessions":  session_manager.stats(),
        "device":    settings.device,
        "model":     settings.vpr_model,
        "cors_origins": settings.cors_origin_list,
    }

@app.delete("/api/_legacy/session/{session_id}")
async def legacy_delete_session(session_id: str):
    session_manager.delete(session_id)
    return {"ok": True}


@app.get("/api/_legacy/jobs")
async def legacy_list_jobs():
    return {"jobs": job_store.list()}


@app.get("/api/_legacy/jobs/{job_id}")
async def legacy_get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {"job": job.as_dict()}


# ── Web UI ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "ui.html"
    if html_path.exists():
        try:
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            logger.warning("ui.html is not valid UTF-8; serving with replacement characters")
            return HTMLResponse(html_path.read_text(encoding="utf-8", errors="replace"))
    return HTMLResponse("<h1>LocalNavBot API running. See /docs</h1>")
