"""
web/routes/speech.py — Speech transcription endpoint

POST /api/speech/transcribe
    Accepts an audio blob (webm/ogg/wav/mp4) from the browser.
    Transcribes using openai-whisper (GPU if available, CPU fallback).
    Returns: { "ok": true, "text": "rẽ phải sau 20 mét", "language": "vi" }

The transcribed text is NOT automatically sent to the chat pipeline —
the client decides what to do with it (send to chat, trigger navigation, etc.).

Whisper model selection (via settings.whisper_model):
    "tiny"   — fastest, ~39 MB, good for short commands
    "base"   — balanced, ~74 MB  (default)
    "small"  — better accuracy, ~244 MB
    "medium" — best for Vietnamese, ~769 MB
    "large"  — highest accuracy, ~1.5 GB

Lazy loading: model is loaded on first request and cached in memory.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from config.settings import settings

router = APIRouter(tags=["speech"])

# ── Whisper model cache ───────────────────────────────────────────────────────
_whisper_model = None
_whisper_lock  = None   # asyncio.Lock, created lazily


def _get_lock():
    import asyncio
    global _whisper_lock
    if _whisper_lock is None:
        _whisper_lock = asyncio.Lock()
    return _whisper_lock


async def _load_whisper():
    """Load Whisper model lazily (once per process)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    async with _get_lock():
        if _whisper_model is not None:
            return _whisper_model
        try:
            import whisper
            model_name = settings.whisper_model
            logger.info(f"Loading Whisper model '{model_name}'…")
            _whisper_model = whisper.load_model(
                model_name,
                device=settings.effective_whisper_device,
            )
            logger.info(f"Whisper '{model_name}' ready on {settings.effective_whisper_device}")
        except ImportError:
            raise HTTPException(
                503,
                "openai-whisper not installed. Run: pip install openai-whisper"
            )
        except Exception as e:
            raise HTTPException(503, f"Whisper load failed: {e}")
    return _whisper_model


# ── Allowed audio MIME types ──────────────────────────────────────────────────
_ALLOWED_AUDIO = {
    "audio/webm", "audio/ogg", "audio/wav", "audio/wave",
    "audio/mp4", "audio/mpeg", "audio/mp3", "audio/x-m4a",
    "audio/aac", "video/webm",   # Chrome records as video/webm
    "application/octet-stream",  # fallback when browser omits content-type
}
_MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/api/speech/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),   # "vi", "en", None=auto-detect
    session_id: Optional[str] = Form(default=None),
):
    """
    Transcribe an audio blob to text using Whisper.

    - language: ISO 639-1 code ("vi" for Vietnamese, "en" for English).
                Leave empty for automatic language detection.
    - Returns the transcribed text, detected language, and duration.
    """
    # Validate content type (lenient — browsers vary)
    ct = (file.content_type or "").lower().split(";")[0].strip()
    if ct and ct not in _ALLOWED_AUDIO:
        raise HTTPException(400, f"Unsupported audio type: {ct}")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty audio file")
    if len(data) > _MAX_AUDIO_BYTES:
        raise HTTPException(400, f"Audio too large (max {_MAX_AUDIO_BYTES // (1024*1024)} MB)")

    model = await _load_whisper()

    # Write to a temp file — Whisper needs a file path
    suffix = _guess_suffix(file.filename or "", ct)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        import asyncio
        # Run Whisper in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                tmp_path,
                language=language or None,
                task="transcribe",
                fp16=(settings.whisper_device != "cpu"),
                condition_on_previous_text=False,
                # Optimise for short voice commands
                initial_prompt="Điều hướng, tìm đường, địa điểm.",
            ),
        )
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    text = (result.get("text") or "").strip()
    detected_lang = result.get("language", "unknown")

    logger.info(f"[speech] session={session_id} lang={detected_lang} text={text!r}")

    return {
        "ok": True,
        "text": text,
        "language": detected_lang,
        "session_id": session_id,
    }


# ── Utility ───────────────────────────────────────────────────────────────────

def _guess_suffix(filename: str, content_type: str) -> str:
    """Pick a file extension Whisper / ffmpeg can handle."""
    if filename:
        ext = Path(filename).suffix.lower()
        if ext in (".webm", ".ogg", ".wav", ".mp4", ".mp3", ".m4a", ".aac"):
            return ext
    ct_map = {
        "audio/webm": ".webm", "video/webm": ".webm",
        "audio/ogg": ".ogg",   "audio/wav": ".wav",
        "audio/wave": ".wav",  "audio/mp4": ".mp4",
        "audio/mpeg": ".mp3",  "audio/mp3": ".mp3",
        "audio/x-m4a": ".m4a","audio/aac": ".aac",
    }
    return ct_map.get(content_type, ".webm")
