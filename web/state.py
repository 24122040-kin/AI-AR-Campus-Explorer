from __future__ import annotations

from typing import Any

_router = None
_vpr: Any = None
_bot = None
_realtime_manager = None


def set_runtime_state(*, router=None, vpr=None, bot=None, realtime_manager=None) -> None:
    global _router, _vpr, _bot, _realtime_manager
    if router is not None:
        _router = router
    if vpr is not None:
        _vpr = vpr
    if bot is not None:
        _bot = bot
    if realtime_manager is not None:
        _realtime_manager = realtime_manager


def get_router():
    return _router


def get_vpr():
    return _vpr


def get_bot():
    return _bot


def get_realtime_manager():
    return _realtime_manager
