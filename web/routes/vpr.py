from __future__ import annotations

import io
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from loguru import logger

from core.database import db
from web.jobs import job_store
from web.state import get_bot, get_vpr


router = APIRouter(tags=["vpr"])


async def _run_vpr_rebuild_job(job_id: str) -> None:
    try:
        bot = get_bot()
        if bot is None:
            raise RuntimeError("Bot not ready")
        runtime_vpr = get_vpr()
        job_store.update(job_id, status="running", message="Rebuilding VPR index")
        await bot.rebuild_vpr_index()
        size = runtime_vpr._index.size if runtime_vpr and runtime_vpr._index else 0
        job_store.update(
            job_id,
            status="completed",
            message="VPR rebuild completed",
            result={"vpr_index_size": size},
        )
    except Exception as e:
        logger.exception("VPR rebuild job failed")
        job_store.update(job_id, status="failed", message="VPR rebuild failed", error=str(e))


@router.post("/api/vpr/query")
async def vpr_query(
    file: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    runtime_vpr = get_vpr()
    if runtime_vpr is None:
        raise HTTPException(503, "VPR not available")

    from PIL import Image

    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    matches = runtime_vpr.query(img, top_k=5, query_lat=lat, query_lon=lon)
    results = []
    for m in matches:
        # Get location details including floor
        loc = await db.get_location(m.location_id)
        imgs = await db.get_images_for_location(m.location_id)
        primary = next((i for i in imgs if i.get("is_primary")), imgs[0] if imgs else None)
        results.append({
            "location_name": m.location_name,
            "location_id": m.location_id,
            "lat": m.lat,
            "lon": m.lon,
            "floor": loc.get("floor", 1) if loc else 1,
            "category": loc.get("category", "") if loc else "",
            "description": loc.get("description", "") if loc else "",
            "score": round(m.score, 4),
            "distance_m": round(m.distance_m, 1) if m.distance_m != float("inf") else None,
            "caption": m.caption,
            "primary_image_id": primary["id"] if primary else None,
        })
    return {"ok": True, "matches": results, "vpr_ready": runtime_vpr.aggregator._fitted}


@router.post("/api/vpr/rebuild")
async def vpr_rebuild(background_tasks: BackgroundTasks):
    job = job_store.create("vpr_rebuild", "Queued VPR rebuild")
    background_tasks.add_task(_run_vpr_rebuild_job, job.job_id)
    return {"ok": True, "message": "Rebuilding VPR index in background", "job": job.as_dict()}
