from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from config.settings import settings
from web.state import get_realtime_manager
from web.uploads import MAX_UPLOAD_SIZE_BYTES, validate_upload


router = APIRouter(tags=["realtime"])


class RealtimeSensorRequest(BaseModel):
    session_id: str
    compass_heading: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    gyro_heading: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    accel_norm: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    # Raw accelerometer axes (m/s²) — used by floor detector
    accel_x: Optional[float] = Field(default=None, ge=-100.0, le=100.0)
    accel_y: Optional[float] = Field(default=None, ge=-100.0, le=100.0)
    accel_z: Optional[float] = Field(default=None, ge=-100.0, le=100.0)
    # Barometer (hPa)
    pressure_hpa: Optional[float] = Field(default=None, ge=300.0, le=1100.0)


class FloorUpdateRequest(BaseModel):
    session_id: str
    pressure_hpa: Optional[float] = Field(default=None, ge=300.0, le=1100.0)
    accel_x: Optional[float] = Field(default=None, ge=-100.0, le=100.0)
    accel_y: Optional[float] = Field(default=None, ge=-100.0, le=100.0)
    accel_z: Optional[float] = Field(default=None, ge=-100.0, le=100.0)


class FloorCalibrateRequest(BaseModel):
    session_id: str
    floor: int = Field(ge=1, le=200)


class VIOImuRequest(BaseModel):
    """High-rate IMU update for VIO dead-reckoning."""
    session_id: str
    ax: float = Field(ge=-100.0, le=100.0)          # m/s²
    ay: float = Field(ge=-100.0, le=100.0)
    az: float = Field(ge=-100.0, le=100.0)
    gyro_z: float = Field(ge=-50.0, le=50.0)        # rad/s yaw rate
    compass_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    dt_s: float = Field(ge=0.001, le=1.0)           # seconds since last call


class VIOFlowRequest(BaseModel):
    """Optical flow correction from JS client."""
    session_id: str
    flow_x_px: float = Field(ge=-500.0, le=500.0)   # mean feature displacement X
    flow_y_px: float = Field(ge=-500.0, le=500.0)   # mean feature displacement Y
    dt_s: float = Field(ge=0.01, le=2.0)


class VIORelocalizeRequest(BaseModel):
    """Absolute position reset from GPS or VPR."""
    session_id: str
    lat: float
    lon: float
    heading_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    accuracy_m: float = Field(default=5.0, ge=0.1, le=100.0)
    source: str = "gps"   # "gps" | "vpr"


@router.post("/api/realtime/frame")
async def realtime_frame(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    accuracy_m: float = Form(10.0),
    speed_kmh: float = Form(0.0),
    bearing: float = Form(0.0),
):
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        raise HTTPException(503, "Realtime manager not ready")

    suffix = validate_upload(file)
    data = await file.read()
    max_bytes = settings.realtime_frame_max_mb * 1024 * 1024
    if len(data) > min(MAX_UPLOAD_SIZE_BYTES, max_bytes):
        raise HTTPException(400, f"Frame too large. Max {settings.realtime_frame_max_mb} MB.")

    frame_path = manager.build_frame_path(suffix)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(frame_path, "wb") as fh:
        await fh.write(data)

    try:
        return await manager.ingest_frame(
            session_id,
            Path(frame_path),
            lat=lat,
            lon=lon,
            accuracy_m=accuracy_m,
            speed_kmh=speed_kmh,
            bearing=bearing,
        )
    except Exception:
        Path(frame_path).unlink(missing_ok=True)
        raise


@router.post("/api/realtime/sensors")
async def realtime_sensors(req: RealtimeSensorRequest):
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        raise HTTPException(503, "Realtime manager not ready")
    return await manager.update_sensors(req.session_id, req.model_dump())


@router.post("/api/realtime/floor")
async def realtime_floor(req: FloorUpdateRequest):
    """
    Dedicated floor-detection endpoint.
    Accepts barometer (hPa) and raw accelerometer axes (m/s²).
    Returns: { floor, confidence, method }
    """
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        raise HTTPException(503, "Realtime manager not ready")
    accel = None
    if req.accel_x is not None and req.accel_y is not None and req.accel_z is not None:
        accel = {"x": req.accel_x, "y": req.accel_y, "z": req.accel_z}
    return await manager.update_floor(req.session_id, req.pressure_hpa, accel)


@router.post("/api/realtime/floor/calibrate")
async def realtime_floor_calibrate(req: FloorCalibrateRequest):
    """
    Manually set the current floor number.
    Resets the barometric baseline so future readings are relative to this floor.
    """
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        raise HTTPException(503, "Realtime manager not ready")
    return await manager.calibrate_floor(req.session_id, req.floor)


# ── VIO endpoints ─────────────────────────────────────────────────────────────

@router.post("/api/realtime/vio/imu")
async def vio_imu(req: VIOImuRequest):
    """
    High-rate IMU update for VIO dead-reckoning.
    Call at 10–50 Hz from DeviceMotionEvent.
    Returns updated VIO pose.
    """
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        raise HTTPException(503, "Realtime manager not ready")
    return await manager.vio_update_imu(req.session_id, req.model_dump())


@router.post("/api/realtime/vio/flow")
async def vio_flow(req: VIOFlowRequest):
    """
    Optical flow correction from JS client (~5 Hz).
    Accepts mean feature displacement in pixels between consecutive frames.
    """
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        raise HTTPException(503, "Realtime manager not ready")
    return await manager.vio_update_flow(
        req.session_id, req.flow_x_px, req.flow_y_px, req.dt_s
    )


@router.post("/api/realtime/vio/relocalize")
async def vio_relocalize(req: VIORelocalizeRequest):
    """
    Absolute position reset from GPS or VPR match.
    Resets drift counter and corrects EKF position.
    """
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        raise HTTPException(503, "Realtime manager not ready")
    return await manager.vio_relocalize(
        req.session_id, req.lat, req.lon,
        req.heading_deg, req.accuracy_m, req.source,
    )


@router.get("/api/realtime/vio/pose/{session_id}")
async def vio_pose(session_id: str):
    """Return current VIO pose for a session."""
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        raise HTTPException(503, "Realtime manager not ready")
    return await manager.vio_get_pose(session_id)


@router.get("/api/realtime/state/{session_id}")
async def realtime_state(session_id: str):
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        raise HTTPException(503, "Realtime manager not ready")
    return {"ok": True, "state": manager.get_state(session_id)}


@router.get("/api/realtime/sessions")
async def realtime_sessions():
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        raise HTTPException(503, "Realtime manager not ready")
    return {"ok": True, "sessions": manager.list_sessions()}


@router.websocket("/ws/realtime/{session_id}")
async def ws_realtime(websocket: WebSocket, session_id: str):
    manager = get_realtime_manager()
    if manager is None or not settings.realtime_enabled:
        await websocket.close(code=1013, reason="Realtime manager not ready")
        return

    await websocket.accept()
    last_revision = -1
    try:
        while True:
            session = manager.get_or_create(session_id)
            state = session.as_dict()
            revision = state.get("revision", 0)

            # Push state update when revision changes
            if revision != last_revision:
                last_revision = revision
                await websocket.send_json({"type": "realtime_state", "state": state})

            # Push any pending alerts immediately (independent of revision)
            pending = session.pop_alerts()
            for alert in pending:
                await websocket.send_json({"type": "alert", "alert": alert})

            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=0.35)
            except asyncio.TimeoutError:
                continue

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "sensors":
                payload = dict(message)
                payload["session_id"] = session_id
                result = await manager.update_sensors(session_id, payload)
                await websocket.send_json({"type": "sensor_update", **result})
            elif message.get("type") == "floor":
                accel = None
                if message.get("accel_x") is not None:
                    accel = {
                        "x": message.get("accel_x"),
                        "y": message.get("accel_y"),
                        "z": message.get("accel_z"),
                    }
                result = await manager.update_floor(
                    session_id,
                    message.get("pressure_hpa"),
                    accel,
                )
                await websocket.send_json({"type": "floor_update", **result})
            elif message.get("type") == "vio_imu":
                result = await manager.vio_update_imu(session_id, message)
                await websocket.send_json({"type": "vio_pose", **result})
            elif message.get("type") == "vio_flow":
                result = await manager.vio_update_flow(
                    session_id,
                    message.get("flow_x_px", 0.0),
                    message.get("flow_y_px", 0.0),
                    message.get("dt_s", 0.1),
                )
                await websocket.send_json({"type": "vio_pose", **result})
    except WebSocketDisconnect:
        return
