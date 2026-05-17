from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from core.database import db
from core.image_manager import BatchImageImporter, read_gps_exif
from web.jobs import job_store
from web.state import get_bot, get_router, get_vpr
from web.uploads import MAX_UPLOAD_SIZE_BYTES, build_upload_path, ensure_safe_batch_folder, validate_upload


router = APIRouter(tags=["data"])


class LocationRequest(BaseModel):
    name: str
    lat: float
    lon: float
    description: str = ""
    category: str = "general"
    importance: int = Field(default=1, ge=1, le=5)
    tags: list[str] = []
    floor: int = Field(default=1, ge=1, le=200)


class POIRequest(BaseModel):
    name: str
    poi_type: str
    lat: float
    lon: float
    address: str = ""
    notes: str = ""


class EdgeRequest(BaseModel):
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float
    name: str = ""
    road_type: str = Field(default="alley", description=(
        "alley|shortcut|path|corridor|stairs|elevator|ramp|bridge"
    ))
    bidirectional: bool = True
    notes: str = ""
    from_floor: int = Field(default=1, ge=1, le=200)
    to_floor: int = Field(default=1, ge=1, le=200)
    geometry: list[list[float]] | None = None  # [[lat,lon], ...] for curved paths
    # Physical properties
    is_covered: bool = False          # has roof/shelter
    width_m: float | None = None      # path width in metres
    surface: str = "concrete"         # concrete|tile|grass|dirt|gravel|wood
    has_lighting: bool = True
    slope_deg: float = 0.0            # incline angle in degrees (+ = uphill)


async def _run_batch_import_job(job_id: str, folder: Path, auto_caption: bool, min_quality: float) -> None:
    try:
        job_store.update(job_id, status="running", message=f"Importing images from {folder}")
        importer = BatchImageImporter(do_captions=auto_caption, min_quality=min_quality)
        result = await importer.import_folder(folder)
        job_store.update(job_id, status="running", message="Refreshing VPR index after import", result=result)
        bot = get_bot()
        if bot is not None:
            await bot.rebuild_vpr_index()
        job_store.update(job_id, status="completed", message="Batch import completed", result=result)
    except Exception as e:
        logger.error(traceback.format_exc())
        job_store.update(job_id, status="failed", message="Batch import failed", error=str(e))


@router.post("/api/upload/image")
async def upload_image(
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
        raise HTTPException(400, "GPS required — chụp ảnh ngoài trời hoặc nhập tọa độ thủ công")

    if location_id is None:
        loc_name = location_name or f"Loc_{dest.stem[:12]}"
        location_id = await db.add_location(
            name=loc_name,
            lat=lat,
            lon=lon,
            category=category,
            importance=importance,
        )

    if auto_caption and not caption:
        from core.image_manager import auto_caption as _ac

        loc = await db.get_location(location_id)
        caption = await _ac(dest, f"Gan {loc['name']}" if loc else "")

    img_id = await db.add_image(location_id=location_id, filename=dest.name, filepath=str(dest), caption=caption)

    runtime_vpr = get_vpr()
    faiss_id = -1
    if runtime_vpr and runtime_vpr.aggregator._fitted:
        from core.vpr_engine import ImageMeta

        loc = await db.get_location(location_id)
        meta = ImageMeta(
            image_id=img_id,
            location_id=location_id,
            location_name=loc["name"] if loc else "",
            lat=lat,
            lon=lon,
            filepath=str(dest),
            caption=caption,
        )
        try:
            faiss_id = runtime_vpr.index_image(dest, meta)
            await db.update_faiss_id(img_id, faiss_id)
            runtime_vpr._index.save()
        except Exception as e:
            logger.warning(f"VPR index: {e}")

    return {
        "ok": True,
        "image_id": img_id,
        "location_id": location_id,
        "faiss_id": faiss_id,
        "lat": lat,
        "lon": lon,
        "caption": caption,
    }


@router.post("/api/upload/batch")
async def batch_import(
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


@router.post("/api/location")
async def add_location(req: LocationRequest):
    loc_id = await db.add_location(
        name=req.name,
        lat=req.lat,
        lon=req.lon,
        description=req.description,
        category=req.category,
        importance=req.importance,
        tags=req.tags,
        floor=req.floor,
    )
    return {"ok": True, "id": loc_id}


@router.post("/api/poi")
async def add_poi(req: POIRequest):
    poi_id = await db.add_poi(
        name=req.name,
        poi_type=req.poi_type,
        lat=req.lat,
        lon=req.lon,
        address=req.address,
        notes=req.notes,
    )
    return {"ok": True, "id": poi_id}


@router.post("/api/edge")
async def add_edge(req: EdgeRequest):
    geom = [tuple(p) for p in req.geometry] if req.geometry else None
    eid, dist_m = await db.add_custom_edge(
        from_lat=req.from_lat,
        from_lon=req.from_lon,
        to_lat=req.to_lat,
        to_lon=req.to_lon,
        name=req.name,
        road_type=req.road_type,
        bidirectional=req.bidirectional,
        from_floor=req.from_floor,
        to_floor=req.to_floor,
        geometry=geom,
    )
    # Store physical properties via direct update
    await db.execute(
        """UPDATE custom_edges
           SET is_covered=?, width_m=?, surface=?, has_lighting=?, slope_deg=?
           WHERE id=?""",
        (int(req.is_covered), req.width_m, req.surface,
         int(req.has_lighting), req.slope_deg, eid),
    )
    runtime_router = get_router()
    if runtime_router and runtime_router.osm.G:
        await runtime_router.osm.patch_custom_edges()
    return {"ok": True, "id": eid, "distance_m": round(dist_m, 1)}


@router.post("/api/location/images")
async def upload_location_images(
    location_id: int = Form(...),
    primary_index: int = Form(0),   # which file (0-based) is the primary image
    files: list[UploadFile] = File(...),
    captions: str = Form(""),       # JSON array of captions, one per file
    auto_caption: bool = Form(False),
):
    """Upload 1–5 images for an existing location. First image is primary by default."""
    import json as _json
    if len(files) > 5:
        raise HTTPException(400, "Maximum 5 images per location")

    loc = await db.get_location(location_id)
    if not loc:
        raise HTTPException(404, "Location not found")

    try:
        caps = _json.loads(captions) if captions.strip() else []
    except Exception:
        caps = []

    results = []
    primary_image_id = None

    for idx, file in enumerate(files):
        suffix = validate_upload(file)
        data = await file.read()
        if len(data) > MAX_UPLOAD_SIZE_BYTES:
            results.append({"ok": False, "error": f"File {idx} too large"})
            continue
        dest = build_upload_path(suffix)
        async with aiofiles.open(dest, "wb") as f:
            await f.write(data)

        caption = caps[idx] if idx < len(caps) else ""
        if auto_caption and not caption:
            from core.image_manager import auto_caption as _ac
            caption = await _ac(dest, f"Gần {loc['name']}")

        img_id = await db.add_image(
            location_id=location_id,
            filename=dest.name,
            filepath=str(dest),
            caption=caption,
        )

        # Index into VPR
        runtime_vpr = get_vpr()
        faiss_id = -1
        if runtime_vpr and runtime_vpr.aggregator._fitted:
            from core.vpr_engine import ImageMeta
            meta = ImageMeta(
                image_id=img_id, location_id=location_id,
                location_name=loc["name"],
                lat=loc["lat"], lon=loc["lon"],
                filepath=str(dest), caption=caption,
            )
            try:
                faiss_id = runtime_vpr.index_image(dest, meta)
                await db.update_faiss_id(img_id, faiss_id)
                runtime_vpr._index.save()
            except Exception as e:
                logger.warning(f"VPR index: {e}")

        if idx == primary_index:
            primary_image_id = img_id

        results.append({"ok": True, "image_id": img_id, "faiss_id": faiss_id})

    # Set primary image
    if primary_image_id:
        await db.set_primary_image(location_id, primary_image_id)

    return {"ok": True, "location_id": location_id, "images": results}
