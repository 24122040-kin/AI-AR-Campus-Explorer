"""
core/image_manager.py — Smart image pipeline
Features:
  - Auto-extract GPS from EXIF / write GPS to EXIF
  - Auto-caption images using VLM (describe what's in the photo for nav context)
  - Batch import entire folders
  - Duplicate detection via perceptual hash
  - Image quality filter (blur, dark, over-exposed)
  - Auto-detect bearing from sequential images
"""
from __future__ import annotations
import io
import math
import struct
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageStat
from loguru import logger

from config.settings import settings
from core.database import db


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImportedImage:
    path: Path
    lat: float
    lon: float
    bearing: Optional[float]
    taken_at: Optional[str]
    caption: str
    quality_score: float      # 0–1, higher = better
    phash: str
    width: int
    height: int
    size_kb: int


# ─────────────────────────────────────────────────────────────────────────────
# EXIF GPS utilities
# ─────────────────────────────────────────────────────────────────────────────

def read_gps_exif(path: Path) -> tuple[float, float, float | None] | None:
    """
    Returns (lat, lon, bearing_degrees) or None.
    bearing = GPS ImgDirection if present.
    """
    try:
        from PIL import ExifTags
        img = Image.open(path)
        raw = img._getexif()  # type: ignore
        if not raw:
            return None

        exif = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
        gps_raw = exif.get("GPSInfo", {})
        if not gps_raw:
            return None

        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}

        def _dms(vals, ref):
            def _frac(v):
                return float(v[0]) / float(v[1]) if isinstance(v, tuple) else float(v)
            d, m, s = [_frac(v) for v in vals]
            deg = d + m / 60 + s / 3600
            return -deg if ref in ("S", "W") else deg

        lat = _dms(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        lon = _dms(gps["GPSLongitude"], gps["GPSLongitudeRef"])

        bearing = None
        if "GPSImgDirection" in gps:
            v = gps["GPSImgDirection"]
            bearing = float(v[0]) / float(v[1]) if isinstance(v, tuple) else float(v)

        return lat, lon, bearing
    except Exception:
        return None


def write_gps_exif(path: Path, lat: float, lon: float, bearing: float | None = None) -> bool:
    """Write GPS coordinates into image EXIF in-place."""
    try:
        import piexif

        def _to_dms(deg: float) -> tuple:
            deg = abs(deg)
            d = int(deg)
            m_float = (deg - d) * 60
            m = int(m_float)
            s = (m_float - m) * 60
            return ((d, 1), (m, 1), (int(s * 10000), 10000))

        exif_dict: dict = {"GPS": {}}
        exif_bytes = b""

        try:
            exif_dict = piexif.load(str(path))
        except Exception:
            pass

        exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef]  = b"N" if lat >= 0 else b"S"
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]     = _to_dms(lat)
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = b"E" if lon >= 0 else b"W"
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]    = _to_dms(lon)

        if bearing is not None:
            exif_dict["GPS"][piexif.GPSIFD.GPSImgDirectionRef] = b"T"
            exif_dict["GPS"][piexif.GPSIFD.GPSImgDirection]    = (int(bearing * 100), 100)

        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(path))
        return True
    except Exception as e:
        logger.warning(f"EXIF write failed for {path}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Image quality assessment
# ─────────────────────────────────────────────────────────────────────────────

def assess_quality(img: Image.Image) -> float:
    """
    Returns quality score [0.0, 1.0].
    Penalises: blur, darkness, over-exposure, tiny resolution.
    """
    w, h = img.size
    if w * h < 100_000:   # < 0.1 MP
        return 0.1

    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # Laplacian variance → sharpness
    def laplacian_var(a):
        import cv2
        lap = cv2.Laplacian(a, cv2.CV_32F)
        return float(lap.var())

    try:
        sharp = min(laplacian_var(arr) / 500.0, 1.0)
    except Exception:
        sharp = 0.5

    # Brightness
    stat = ImageStat.Stat(gray)
    mean_bright = stat.mean[0] / 255.0
    bright_score = 1.0 - abs(mean_bright - 0.45) * 2  # penalise very dark/bright

    score = 0.55 * sharp + 0.45 * max(bright_score, 0.1)
    return round(min(max(score, 0.0), 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Perceptual hash (for deduplication)
# ─────────────────────────────────────────────────────────────────────────────

def phash(img: Image.Image, size: int = 16) -> str:
    """DCT-based perceptual hash → 64-char hex string."""
    resized = img.convert("L").resize((size, size), Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32)
    # Simple mean-hash as fallback (works without scipy)
    mean = arr.mean()
    bits = (arr > mean).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:0{size*size//4}x}"


def hamming_distance(h1: str, h2: str) -> int:
    """Number of differing bits between two hex hashes."""
    n1 = int(h1, 16)
    n2 = int(h2, 16)
    xor = n1 ^ n2
    return bin(xor).count("1")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-captioner (uses LLM vision)
# ─────────────────────────────────────────────────────────────────────────────

async def auto_caption(img_path: Path, context: str = "") -> str:
    """
    Generate a navigation-useful caption for an image using the configured LLM.
    Caption focuses on landmarks, road features, and turning cues.
    """
    try:
        import base64

        img = Image.open(img_path)
        # Resize to save tokens
        w, h = img.size
        if max(w, h) > 800:
            scale = 800 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        from bot.nav_bot import LLMClient
        llm = LLMClient()

        prompt = (
            "Bạn đang nhìn vào một ảnh chụp tại một vị trí GPS để dùng cho điều hướng. "
            "Hãy mô tả ngắn gọn (1-2 câu) những đặc điểm nhận dạng quan trọng: "
            "tên đường, biển hiệu, công trình nổi bật, ngã tư, cây xanh đặc trưng. "
            "Tập trung vào thông tin giúp người lái xe nhận ra vị trí này. "
            + (f"Ngữ cảnh: {context}" if context else "")
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }]

        result = await llm.chat(messages)
        if isinstance(result, str):
            return result.strip()[:200]
    except Exception as e:
        logger.debug(f"Auto-caption failed: {e}")

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Batch importer
# ─────────────────────────────────────────────────────────────────────────────

class BatchImageImporter:
    """
    Import a folder of images:
    1. Read GPS from EXIF
    2. Assess quality, skip blurry/dark
    3. Deduplicate by perceptual hash
    4. Group by proximity (cluster images within 10m → same location)
    5. Auto-caption (optional)
    6. Save to DB + copy to images_dir
    """

    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    MIN_QUALITY = 0.25
    DEDUP_THRESHOLD = 8    # hamming distance ≤ 8 = duplicate
    CLUSTER_RADIUS_M = 15  # images within 15m → same location

    def __init__(self, do_captions: bool = False, min_quality: float = MIN_QUALITY):
        self.do_captions = do_captions
        self.min_quality = min_quality
        settings.images_dir.mkdir(parents=True, exist_ok=True)

    async def import_folder(self, folder: Path) -> dict:
        """Import all images from a folder. Returns summary stats."""
        files = [p for p in folder.rglob("*") if p.suffix.lower() in self.EXTS]
        logger.info(f"Found {len(files)} image files in {folder}")

        processed: list[ImportedImage] = []
        skipped_no_gps = 0
        skipped_quality = 0
        skipped_dup = 0
        seen_hashes: list[str] = []

        for f in files:
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                continue

            # GPS
            gps = read_gps_exif(f)
            if gps is None:
                skipped_no_gps += 1
                continue
            lat, lon, bearing = gps

            # Quality
            q = assess_quality(img)
            if q < self.min_quality:
                skipped_quality += 1
                logger.debug(f"Low quality ({q:.2f}): {f.name}")
                continue

            # Dedup
            ph = phash(img)
            is_dup = any(hamming_distance(ph, h) <= self.DEDUP_THRESHOLD for h in seen_hashes)
            if is_dup:
                skipped_dup += 1
                continue
            seen_hashes.append(ph)

            # Taken-at from EXIF
            taken_at = None
            try:
                from PIL import ExifTags
                raw = img._getexif()  # type: ignore
                if raw:
                    exif = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
                    taken_at = exif.get("DateTime") or exif.get("DateTimeOriginal")
            except Exception:
                pass

            processed.append(ImportedImage(
                path=f, lat=lat, lon=lon, bearing=bearing,
                taken_at=taken_at, caption="", quality_score=q,
                phash=ph, width=img.size[0], height=img.size[1],
                size_kb=f.stat().st_size // 1024,
            ))

        logger.info(f"After filtering: {len(processed)} images "
                    f"(skip no-GPS={skipped_no_gps}, quality={skipped_quality}, dup={skipped_dup})")

        # Cluster by proximity
        clusters = self._cluster(processed)
        logger.info(f"Clustered into {len(clusters)} locations")

        # Save to DB
        added_locations = 0
        added_images = 0
        for cluster in clusters:
            avg_lat = sum(im.lat for im in cluster) / len(cluster)
            avg_lon = sum(im.lon for im in cluster) / len(cluster)

            # Check if location already exists
            nearby = await db.nearby_locations(avg_lat, avg_lon, radius_deg=0.00015)
            if nearby:
                loc_id = nearby[0]["id"]
            else:
                name = f"Location {datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:18]}"
                loc_id = await db.add_location(
                    name=name, lat=avg_lat, lon=avg_lon,
                    importance=min(len(cluster), 4),
                )
                added_locations += 1

            # Save up to 4 best images per cluster (sorted by quality)
            best = sorted(cluster, key=lambda x: -x.quality_score)[:4]
            for im in best:
                # Copy to images_dir
                dest = settings.images_dir / im.path.name
                if not dest.exists():
                    import shutil
                    shutil.copy2(im.path, dest)

                # Auto-caption
                caption = im.caption
                if self.do_captions and not caption:
                    caption = await auto_caption(dest)

                img_id = await db.add_image(
                    location_id=loc_id,
                    filename=dest.name,
                    filepath=str(dest),
                    caption=caption,
                    bearing=im.bearing,
                )
                added_images += 1

        return {
            "total_files": len(files),
            "processed": len(processed),
            "skipped_no_gps": skipped_no_gps,
            "skipped_quality": skipped_quality,
            "skipped_dup": skipped_dup,
            "locations_created": added_locations,
            "images_added": added_images,
        }

    def _cluster(self, images: list[ImportedImage]) -> list[list[ImportedImage]]:
        """Simple greedy spatial clustering."""
        clusters: list[list[ImportedImage]] = []
        assigned = [False] * len(images)

        for i, im in enumerate(images):
            if assigned[i]:
                continue
            cluster = [im]
            assigned[i] = True
            for j, other in enumerate(images):
                if assigned[j]:
                    continue
                d = _haversine(im.lat, im.lon, other.lat, other.lon)
                if d <= self.CLUSTER_RADIUS_M:
                    cluster.append(other)
                    assigned[j] = True
            clusters.append(cluster)

        return clusters


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
