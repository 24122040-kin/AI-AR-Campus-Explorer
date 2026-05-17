from __future__ import annotations

import asyncio
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from bot.nav_bot import NavBot
from bot.session_manager import GPSFix, NavState, session_manager
from config.settings import settings
from web.state import get_router, get_vpr
from web.uploads import MAX_UPLOAD_SIZE_BYTES, validate_upload


router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    session_id: Optional[str] = "default"


def _validate_chat_request(req: ChatRequest) -> None:
    if not req.message or not req.message.strip():
        raise HTTPException(400, "Message is required.")
    if len(req.message) > settings.chat_max_chars:
        raise HTTPException(400, f"Message too long. Max {settings.chat_max_chars} characters.")
    if req.lat is not None and not (-90.0 <= req.lat <= 90.0):
        raise HTTPException(400, "Invalid latitude.")
    if req.lon is not None and not (-180.0 <= req.lon <= 180.0):
        raise HTTPException(400, "Invalid longitude.")


@router.post("/api/chat")
async def chat(req: ChatRequest):
    runtime_router = get_router()
    runtime_vpr = get_vpr()
    if runtime_router is None:
        raise HTTPException(503, "Router not ready")
    _validate_chat_request(req)

    sess = session_manager.get_or_create(req.session_id or "default")
    bot = NavBot(runtime_router, runtime_vpr)
    bot._history = sess.recent_history()
    try:
        response = await asyncio.wait_for(
            bot.ask(req.message, user_lat=req.lat, user_lon=req.lon),
            timeout=settings.llm_timeout_seconds,
        )
        sess.add_message("user", req.message)
        sess.add_message("assistant", response)
        return {"response": response, "ok": True}
    except asyncio.TimeoutError:
        raise HTTPException(504, f"AI response timed out after {settings.llm_timeout_seconds}s.")
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))


@router.post("/api/chat/image")
async def chat_with_image(
    file: UploadFile = File(...),
    message: str = Form("Day la dau?"),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    session_id: str = Form("default"),
):
    runtime_router = get_router()
    runtime_vpr = get_vpr()
    if runtime_router is None:
        raise HTTPException(503, "Router not ready")
    if not message or not message.strip():
        message = "Day la dau?"

    suffix = validate_upload(file)
    data = await file.read()
    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(400, f"File too large. Max size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.")

    tmp_path = settings.detections_dir / f"chat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(data)

    sess = session_manager.get_or_create(session_id or "default")
    bot = NavBot(runtime_router, runtime_vpr)
    bot._history = sess.recent_history()
    try:
        response = await asyncio.wait_for(
            bot.ask(message, image_path=str(tmp_path), user_lat=lat, user_lon=lon),
            timeout=settings.llm_timeout_seconds,
        )
        sess.add_message("user", message)
        sess.add_message("assistant", response)
        return {"response": response, "ok": True}
    except asyncio.TimeoutError:
        raise HTTPException(504, f"AI response timed out after {settings.llm_timeout_seconds}s.")
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    runtime_router = get_router()
    runtime_vpr = get_vpr()
    if runtime_router is None:
        raise HTTPException(503, "Router not ready")
    _validate_chat_request(req)

    sess = session_manager.get_or_create(req.session_id or "default")
    bot = NavBot(runtime_router, runtime_vpr)
    bot._history = sess.recent_history()

    async def sse() -> AsyncIterator[str]:
        full = ""
        try:
            # bot.stream() is an async generator — cannot use wait_for on it directly
            async for chunk in bot.stream(req.message, user_lat=req.lat, user_lon=req.lon):
                full += chunk
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
            sess.add_message("user", req.message)
            sess.add_message("assistant", full)
            yield "data: [DONE]\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': f'AI response timed out after {settings.llm_timeout_seconds}s.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    runtime_router = get_router()
    runtime_vpr = get_vpr()
    if runtime_router is None:
        await websocket.close(code=1013, reason="Router not ready")
        return

    await websocket.accept()
    sid = f"ws_{id(websocket)}"
    sess = session_manager.get_or_create(sid)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "chat")

            if msg_type == "gps":
                fix = GPSFix(
                    lat=data["lat"],
                    lon=data["lon"],
                    accuracy_m=data.get("accuracy", 10),
                    speed_kmh=data.get("speed", 0),
                    bearing=data.get("bearing", 0),
                )
                event = await session_manager.process_gps_update(sid, fix, get_router())
                await websocket.send_json({"type": "nav_event", **event})

            elif msg_type == "chat":
                message = data.get("message", "")
                lat, lon = data.get("lat"), data.get("lon")
                bot = NavBot(runtime_router, runtime_vpr)
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
