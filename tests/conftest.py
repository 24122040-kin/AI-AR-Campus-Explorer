"""
tests/conftest.py — Shared fixtures cho toàn bộ test suite
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Thêm root vào sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Env mặc định cho test ─────────────────────────────────────────────────────
os.environ.setdefault("LLM_API_KEY", "test-key-xxx")
os.environ.setdefault("LLM_PROVIDER", "anthropic")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("VPR_BACKEND", "orb")
os.environ.setdefault("VALHALLA_URL", "http://localhost:9999")  # intentionally unreachable
os.environ.setdefault("USE_OSMNX_FALLBACK", "true")
os.environ.setdefault("OSM_AUTO_DOWNLOAD", "false")
os.environ.setdefault("REALTIME_ENABLED", "true")
os.environ.setdefault("DEBUG", "true")


# ── Async event loop ──────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop để tránh tạo mới mỗi test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Temp DB ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def tmp_db_path(tmp_path_factory) -> Path:
    """SQLite DB tạm thời cho test session."""
    return tmp_path_factory.mktemp("data") / "test_navbot.db"


# ── Mock VPR Engine ───────────────────────────────────────────────────────────
@pytest.fixture
def mock_vpr():
    """VPR engine giả — không load model thật."""
    vpr = MagicMock()
    vpr.aggregator._fitted = False
    vpr._index = MagicMock()
    vpr._index.size = 0
    vpr.extractor = MagicMock()
    vpr.extractor.backend = "orb"
    return vpr


# ── Mock NavRouter ────────────────────────────────────────────────────────────
@pytest.fixture
def mock_router():
    """NavRouter giả — không cần OSM graph."""
    router = MagicMock()
    router.valhalla = MagicMock()
    router.valhalla.is_healthy = AsyncMock(return_value=False)
    router.osm = MagicMock()
    router.osm.G = None
    router.osm._graph_path = Path("/tmp/nonexistent.graphml")
    router.heuristic = MagicMock()
    router.heuristic.warm_cache = AsyncMock()
    router.resolve_location = AsyncMock(return_value=(10.9085, 106.76))
    router.find_route = AsyncMock(return_value=None)
    return router


# ── FastAPI test client ───────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def app_with_mocks(tmp_db_path):
    """
    Khởi tạo FastAPI app với tất cả dependencies được mock.
    Dùng scope=session để tránh khởi tạo lại nhiều lần.
    """
    with (
        patch("core.database.db") as mock_db,
        patch("web.state.get_router") as mock_get_router,
        patch("web.state.get_vpr") as mock_get_vpr,
        patch("web.state.get_bot") as mock_get_bot,
        patch("web.state.get_realtime_manager") as mock_get_rm,
        patch("bot.session_manager.session_manager") as mock_sm,
    ):
        # Setup mock DB
        mock_db.init = AsyncMock()
        mock_db.fetchone = AsyncMock(return_value={"n": 0})
        mock_db.fetchall = AsyncMock(return_value=[])
        mock_db.nearby_locations = AsyncMock(return_value=[])
        mock_db.nearby_pois = AsyncMock(return_value=[])
        mock_db.search_locations = AsyncMock(return_value=[])
        mock_db.search_pois = AsyncMock(return_value=[])
        mock_db.add_location = AsyncMock(return_value=1)
        mock_db.add_poi = AsyncMock(return_value=1)
        mock_db.add_traffic_obs = AsyncMock(return_value=1)
        mock_db.add_environment_obs = AsyncMock(return_value=1)
        mock_db.list_buildings = AsyncMock(return_value=[])

        # Setup mock router
        router = MagicMock()
        router.valhalla = MagicMock()
        router.valhalla.is_healthy = AsyncMock(return_value=False)
        router.osm = MagicMock()
        router.osm.G = None
        router.osm._graph_path = Path("/tmp/nonexistent.graphml")
        router.heuristic = MagicMock()
        router.heuristic.warm_cache = AsyncMock()
        router.resolve_location = AsyncMock(return_value=(10.9085, 106.76))
        router.find_route = AsyncMock(return_value=None)
        mock_get_router.return_value = router

        # Setup mock VPR
        vpr = MagicMock()
        vpr.aggregator._fitted = False
        vpr._index = MagicMock()
        vpr._index.size = 0
        vpr.extractor = MagicMock()
        vpr.extractor.backend = "orb"
        mock_get_vpr.return_value = vpr

        # Setup mock session manager
        mock_sm.get_or_create = MagicMock(return_value=MagicMock(
            recent_history=MagicMock(return_value=[]),
            add_message=MagicMock(),
            state=MagicMock(value="idle"),
        ))
        mock_sm.process_gps_update = AsyncMock(return_value={"type": "none"})
        mock_sm.stats = MagicMock(return_value={"active": 0})
        mock_sm.delete = MagicMock()

        # Setup mock realtime manager
        rm = MagicMock()
        rm.get_or_create = MagicMock(return_value=MagicMock(
            as_dict=MagicMock(return_value={"revision": 0}),
            pop_alerts=MagicMock(return_value=[]),
        ))
        rm.update_sensors = AsyncMock(return_value={"ok": True})
        rm.update_floor = AsyncMock(return_value={"floor": {"floor": 1, "confidence": 0.8, "method": "barometer"}})
        rm.calibrate_floor = AsyncMock(return_value={"ok": True})
        rm.vio_update_imu = AsyncMock(return_value={"ok": True, "pose": {}})
        rm.vio_update_flow = AsyncMock(return_value={"ok": True, "pose": {}})
        rm.vio_relocalize = AsyncMock(return_value={"ok": True})
        rm.vio_get_pose = AsyncMock(return_value={"ok": True, "pose": {}})
        rm.get_state = MagicMock(return_value={})
        rm.list_sessions = MagicMock(return_value=[])
        rm.build_frame_path = MagicMock(return_value=Path(tempfile.mktemp(suffix=".jpg")))
        rm.ingest_frame = AsyncMock(return_value={"ok": True})
        mock_get_rm.return_value = rm

        from web.app import app
        yield app


@pytest.fixture
def client(app_with_mocks):
    """Synchronous test client."""
    from fastapi.testclient import TestClient
    with TestClient(app_with_mocks, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
async def async_client(app_with_mocks) -> AsyncGenerator:
    """Async test client cho WebSocket tests."""
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mocks),
        base_url="http://testserver",
    ) as c:
        yield c


# ── Sample image fixture ──────────────────────────────────────────────────────
@pytest.fixture
def sample_image_bytes() -> bytes:
    """Tạo ảnh JPEG nhỏ hợp lệ (1x1 pixel) để test upload."""
    import io
    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_image_file(sample_image_bytes, tmp_path) -> Path:
    """Lưu ảnh test ra file tạm."""
    p = tmp_path / "test.jpg"
    p.write_bytes(sample_image_bytes)
    return p
