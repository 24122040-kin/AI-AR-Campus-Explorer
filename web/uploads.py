from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile

from config.settings import settings


ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024


def validate_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(400, f"Unsupported image extension: {suffix or 'missing'}")

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported content type: {content_type}")

    return suffix


def build_upload_path(suffix: str) -> Path:
    name = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return settings.images_dir / f"{name}{suffix}"


def ensure_safe_batch_folder(folder: str) -> Path:
    target = Path(folder).expanduser().resolve()
    allowed_roots = [
        settings.data_dir.resolve(),
        settings.images_dir.resolve(),
        settings.db_path.parent.resolve(),
    ]
    if not any(root == target or root in target.parents for root in allowed_roots):
        raise HTTPException(
            400,
            "Batch import folder must be inside the project data directories for safety.",
        )
    if not target.exists():
        raise HTTPException(404, f"Folder not found: {folder}")
    if not target.is_dir():
        raise HTTPException(400, "Batch import target must be a directory.")
    return target
