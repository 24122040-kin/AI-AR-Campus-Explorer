from __future__ import annotations

from fastapi import APIRouter, HTTPException

from bot.session_manager import session_manager
from config.settings import settings
from core.database import db
from web.jobs import job_store
from web.state import get_router, get_vpr


router = APIRouter(tags=["system"])


def _estimate_ai_risk_score(*, llm_configured: bool, vpr_ready: bool, valhalla_ready: bool, has_data: bool) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if not llm_configured:
        score += 35
        reasons.append("LLM API key/base URL not configured for selected provider.")
    if not vpr_ready:
        score += 25
        reasons.append("VPR is not fully ready; image-based localization may degrade.")
    if not valhalla_ready:
        score += 20
        reasons.append("Valhalla is offline; routing quality depends on osmnx fallback.")
    if not has_data:
        score += 20
        reasons.append("Local place/image database is sparse; recommendations may be weak.")
    return min(score, 100), reasons


@router.get("/api/status")
async def status():
    runtime_router = get_router()
    runtime_vpr = get_vpr()
    valhalla_ok = await runtime_router.valhalla.is_healthy() if runtime_router else False
    vpr_ok = runtime_vpr is not None and runtime_vpr.aggregator._fitted
    return {
        "status": "ok",
        "valhalla": valhalla_ok,
        "osm_graph_cached": bool(runtime_router and runtime_router.osm._graph_path.exists()),
        "vpr_ready": vpr_ok,
        "vpr_index_size": runtime_vpr._index.size if runtime_vpr and runtime_vpr._index else 0,
        "vpr_backend": getattr(getattr(runtime_vpr, "extractor", None), "backend", None),
        "locations": (await db.fetchone("SELECT COUNT(*) AS n FROM locations") or {}).get("n", 0),
        "pois": (await db.fetchone("SELECT COUNT(*) AS n FROM pois") or {}).get("n", 0),
        "images": (await db.fetchone("SELECT COUNT(*) AS n FROM images") or {}).get("n", 0),
        "sessions": session_manager.stats(),
        "device": settings.device,
        "model": settings.vpr_model,
        "cors_origins": settings.cors_origin_list,
    }


@router.get("/api/ai/readiness")
async def ai_readiness():
    runtime_router = get_router()
    runtime_vpr = get_vpr()
    valhalla_ok = await runtime_router.valhalla.is_healthy() if runtime_router else False
    vpr_ok = runtime_vpr is not None and runtime_vpr.aggregator._fitted
    location_count = (await db.fetchone("SELECT COUNT(*) AS n FROM locations") or {}).get("n", 0)
    image_count = (await db.fetchone("SELECT COUNT(*) AS n FROM images") or {}).get("n", 0)
    has_data = (location_count + image_count) >= 10

    llm_configured = bool(settings.llm_api_key.strip())
    if settings.llm_provider == "ollama":
        llm_configured = bool(settings.llm_base_url.strip())

    risk_score, risks = _estimate_ai_risk_score(
        llm_configured=llm_configured,
        vpr_ready=vpr_ok,
        valhalla_ready=valhalla_ok,
        has_data=has_data,
    )

    return {
        "ok": True,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "llm_configured": llm_configured,
        "vpr_ready": vpr_ok,
        "valhalla_ready": valhalla_ok,
        "has_minimum_data": has_data,
        "risk_score": risk_score,
        "risk_level": "low" if risk_score < 25 else "medium" if risk_score < 55 else "high",
        "risks": risks,
        "prevention": [
            "Set valid provider credentials in .env and restart.",
            "Rebuild VPR index after importing image batches.",
            "Keep Valhalla running for better ETA/route quality.",
            "Collect more local locations and photos to improve AI grounding.",
        ],
    }


@router.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    session_manager.delete(session_id)
    return {"ok": True}


@router.get("/api/jobs")
async def list_jobs():
    return {"jobs": job_store.list()}


@router.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return {"job": job.as_dict()}
