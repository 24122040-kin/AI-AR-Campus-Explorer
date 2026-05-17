"""
core/vpr_engine.py — Visual Place Recognition with DINOv2 + VLAD aggregation (AnyLoc-style)
GPU-accelerated. Works with ordinary 2D photos tagged with GPS.
"""
from __future__ import annotations
import json
import math
import numpy as np
from pathlib import Path
from typing import Optional, NamedTuple, Any
from dataclasses import dataclass, field

from PIL import Image, ExifTags
from loguru import logger

from config.settings import settings


def _import_torch():
    import torch

    return torch


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageMeta:
    image_id: int
    location_id: int
    location_name: str
    lat: float
    lon: float
    filepath: str
    caption: str = ""
    faiss_idx: int = -1


class VPRMatch(NamedTuple):
    image_id: int
    location_id: int
    location_name: str
    lat: float
    lon: float
    filepath: str
    caption: str
    score: float            # cosine similarity [0, 1]
    distance_m: float       # geodesic distance from query GPS (if given)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2Extractor:
    """
    Extracts per-patch DINOv2 features from images.
    Uses the `value` facet of the specified intermediate layer (AnyLoc recipe).
    """

    TRANSFORM_MEAN = (0.485, 0.456, 0.406)
    TRANSFORM_STD  = (0.229, 0.224, 0.225)
    # Standard input sizes (multiples of patch size 14)
    _SIZES = [224, 322, 448, 518]

    def __init__(
        self,
        model_name: str = settings.vpr_model,
        layer: int = settings.vpr_layer,
        facet: str = settings.vpr_facet,
        device: str = settings.device,
    ):
        self.backend = "dinov2"
        self.layer = layer
        self.facet = facet
        torch = _import_torch()
        self._torch = torch
        self.device = torch.device(device)

        logger.info(f"Loading {model_name} on {device}…")
        # Suppress xFormers unavailability warnings — they are cosmetic only;
        # DINOv2 runs correctly without xFormers (slightly slower attention).
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*xFormers.*")
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

        if settings.torch_dtype == "float16" and self.device.type == "cuda":
            self.model = self.model.half()
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        # Register hook on the chosen transformer block
        self._feats: list[Any] = []
        self._register_hook()

        # Determine patch feature dimension dynamically
        dummy = torch.zeros(1, 3, 224, 224, device=self.device, dtype=self._dtype)
        with torch.no_grad():
            self.model(dummy)
        self.feat_dim: int = self._feats[0].shape[-1]
        self._feats.clear()
        logger.info(f"DINOv2 feature dim = {self.feat_dim}")

    def _register_hook(self) -> None:
        def _hook(module, inp, out):
            # out shape: (B, num_tokens, dim)
            if self.facet == "value":
                # Access attention module's v projection
                self._feats.append(out.detach())
            else:
                self._feats.append(out.detach())

        block = self.model.blocks[self.layer]
        if self.facet == "key":
            block.attn.proj.register_forward_hook(_hook)
        elif self.facet == "value":
            block.attn.proj.register_forward_hook(_hook)
        else:  # token / output
            block.register_forward_hook(_hook)

    def _preprocess(self, img: Image.Image, size: int = 448):
        from torchvision import transforms
        tfm = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.TRANSFORM_MEAN, std=self.TRANSFORM_STD),
        ])
        return tfm(img.convert("RGB")).unsqueeze(0).to(self.device, dtype=self._dtype)

    def extract(self, img: Image.Image, size: int = 448) -> np.ndarray:
        """
        Returns: (num_patches, feat_dim) float32 array on CPU.
        """
        torch = self._torch
        self._feats.clear()
        x = self._preprocess(img, size)
        with torch.no_grad():
            _ = self.model(x)
        feats = self._feats[0]             # (1, N_tokens, D)
        feats = feats[:, 1:, :]            # drop CLS token
        feats = feats.squeeze(0)           # (N_tokens, D)
        return feats.float().cpu().numpy()


class ORBExtractor:
    """CPU-friendly fallback extractor for fully offline Windows setups."""

    def __init__(self, nfeatures: int = 2048):
        import cv2

        self.backend = "orb"
        self.cv2 = cv2
        self.extractor = cv2.ORB_create(nfeatures=nfeatures)
        self.feat_dim = 32

    def extract(self, img: Image.Image, size: int = 448) -> np.ndarray:
        arr = np.array(img.convert("RGB"))
        gray = self.cv2.cvtColor(arr, self.cv2.COLOR_RGB2GRAY)
        keypoints, desc = self.extractor.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            desc = np.zeros((1, self.feat_dim), dtype=np.float32)
        else:
            desc = desc.astype(np.float32)
        return desc


# ─────────────────────────────────────────────────────────────────────────────
# VLAD aggregation
# ─────────────────────────────────────────────────────────────────────────────

class VLADAggregator:
    """
    Builds VLAD vocabulary via k-means and aggregates patch features into a
    single compact descriptor per image.
    """

    def __init__(self, num_clusters: int = settings.vpr_num_clusters):
        self.num_clusters = num_clusters
        self.centroids: Optional[np.ndarray] = None   # (K, D)
        self._fitted = False

    def fit(self, all_feats: np.ndarray) -> None:
        """Run mini-batch k-means on collected patch features."""
        from sklearn.cluster import MiniBatchKMeans
        logger.info(f"Fitting VLAD vocabulary with {self.num_clusters} clusters on "
                    f"{len(all_feats)} descriptors…")
        km = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            random_state=42,
            batch_size=4096,
            n_init=5,
            max_iter=300,
        )
        km.fit(all_feats)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self._fitted = True
        logger.info("VLAD vocabulary ready.")

    def encode(self, patch_feats: np.ndarray) -> np.ndarray:
        """
        Args: patch_feats (N_patches, D)
        Returns: vlad (K*D,) L2-normalised float32 vector
        """
        assert self._fitted, "Call fit() or load() before encode()"
        K, D = self.centroids.shape
        # Nearest centroid assignment
        diffs = patch_feats[:, None, :] - self.centroids[None, :, :]   # (N, K, D)
        dists = np.linalg.norm(diffs, axis=-1)                          # (N, K)
        assigns = np.argmin(dists, axis=-1)                             # (N,)
        # VLAD accumulation
        vlad = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            mask = assigns == k
            if mask.sum() > 0:
                residuals = patch_feats[mask] - self.centroids[k]
                vlad[k] = residuals.sum(axis=0)
        # Intra-normalisation + L2
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        norm = np.linalg.norm(vlad)
        if norm > 1e-8:
            vlad /= norm
        return vlad.flatten()

    def save(self, path: Path) -> None:
        np.save(str(path), self.centroids)

    def load(self, path: Path) -> None:
        self.centroids = np.load(str(path)).astype(np.float32)
        self._fitted = True


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index manager
# ─────────────────────────────────────────────────────────────────────────────

class FAISSIndex:
    """Flat inner-product index (= cosine similarity after L2 normalisation)."""

    def __init__(
        self,
        dim: int,
        index_path: Path = settings.faiss_index_path,
        meta_path: Path = settings.faiss_meta_path,
    ):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.meta: list[ImageMeta] = []
        self._index = None
        self._backend = "faiss"
        self._vectors: list[np.ndarray] = []
        self._load_or_create()

    def _load_or_create(self) -> None:
        try:
            import faiss
        except ImportError:
            self._backend = "numpy"
            if self.meta_path.exists():
                raw = json.loads(self.meta_path.read_text())
                self.meta = [ImageMeta(**m) for m in raw]
            if self.index_path.exists():
                with self.index_path.open("rb") as f:
                    self._vectors = [v for v in np.load(f, allow_pickle=False)]
            logger.warning("faiss not installed; using slower numpy similarity index")
            return

        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            raw = json.loads(self.meta_path.read_text())
            self.meta = [ImageMeta(**m) for m in raw]
            logger.info(f"FAISS index loaded: {self._index.ntotal} vectors")
        else:
            import faiss
            self._index = faiss.IndexFlatIP(self.dim)   # inner product (cosine)
            logger.info(f"FAISS index created (dim={self.dim})")

    def add(self, vector: np.ndarray, meta: ImageMeta) -> int:
        vec = vector.reshape(1, -1).astype(np.float32)
        idx = self.size
        if self._backend == "faiss":
            import faiss
            faiss.normalize_L2(vec)
            self._index.add(vec)
        else:
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            vec = vec / np.clip(norm, 1e-8, None)
            self._vectors.append(vec[0])
        meta.faiss_idx = idx
        self.meta.append(meta)
        return idx

    def search(
        self, query_vec: np.ndarray, top_k: int = settings.vpr_top_k
    ) -> list[tuple[float, ImageMeta]]:
        vec = query_vec.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.size)
        if k == 0:
            return []
        results = []
        if self._backend == "faiss":
            import faiss
            faiss.normalize_L2(vec)
            scores, idxs = self._index.search(vec, k)
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                results.append((float(score), self.meta[idx]))
        else:
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            q = vec / np.clip(norm, 1e-8, None)
            mat = np.vstack(self._vectors)
            sims = mat @ q[0]
            idxs = np.argsort(-sims)[:k]
            for idx in idxs:
                results.append((float(sims[idx]), self.meta[int(idx)]))
        return results

    def save(self) -> None:
        if self._backend == "faiss":
            import faiss
            faiss.write_index(self._index, str(self.index_path))
        else:
            arr = np.vstack(self._vectors) if self._vectors else np.zeros((0, self.dim), dtype=np.float32)
            with self.index_path.open("wb") as f:
                np.save(f, arr)
        raw = [vars(m) for m in self.meta]
        self.meta_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
        logger.info(f"Vector index saved ({self.size} vectors)")

    @property
    def size(self) -> int:
        if self._backend == "faiss":
            return self._index.ntotal if self._index else 0
        return len(self._vectors)


# ─────────────────────────────────────────────────────────────────────────────
# High-level VPR Engine
# ─────────────────────────────────────────────────────────────────────────────

class VPREngine:
    """
    Full pipeline: image → DINOv2 patches → VLAD → FAISS search → VPRMatch list
    """

    VOCAB_PATH  = settings.data_dir / "vlad_centroids.npy"
    FAISS_DIM_PATH = settings.data_dir / "vlad_dim.txt"

    def __init__(self):
        self.extractor = self._make_extractor()
        self.aggregator = VLADAggregator()
        self.vlad_dim: int = 0
        self._index: Optional[FAISSIndex] = None
        self._load_vocab_if_exists()

    def _make_extractor(self):
        backend = settings.vpr_backend.lower()
        if backend == "orb":
            return ORBExtractor()
        if backend in ("auto", "dinov2"):
            try:
                return DINOv2Extractor()
            except Exception as e:
                logger.warning(f"DINOv2 unavailable, falling back to ORB: {e}")
                return ORBExtractor()
        raise ValueError(f"Unsupported VPR backend: {settings.vpr_backend}")

    # ── Vocab / Index ─────────────────────────────────────────────────

    def _load_vocab_if_exists(self) -> None:
        if self.VOCAB_PATH.exists():
            self.aggregator.load(self.VOCAB_PATH)
            if self.FAISS_DIM_PATH.exists():
                self.vlad_dim = int(self.FAISS_DIM_PATH.read_text())
                self._index = FAISSIndex(self.vlad_dim)
                logger.info(f"VPR ready: vocab+index loaded (dim={self.vlad_dim})")

    def _ensure_index(self) -> FAISSIndex:
        assert self._index is not None, (
            "FAISS index not initialised. "
            "Run index_all_images() or add_image() to build it."
        )
        return self._index

    # ── GPS extraction from EXIF ──────────────────────────────────────

    @staticmethod
    def gps_from_exif(img_path: Path) -> tuple[float, float] | None:
        """Extract (lat, lon) from image EXIF if available."""
        try:
            img = Image.open(img_path)
            exif_data = img._getexif()  # type: ignore
            if not exif_data:
                return None
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
            gps = exif.get("GPSInfo", {})
            if not gps:
                return None
            gps_named = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}

            def _to_deg(vals, ref):
                d, m, s = [float(v[0]) / float(v[1]) for v in vals]
                deg = d + m / 60 + s / 3600
                if ref in ("S", "W"):
                    deg = -deg
                return deg

            lat = _to_deg(gps_named["GPSLatitude"], gps_named["GPSLatitudeRef"])
            lon = _to_deg(gps_named["GPSLongitude"], gps_named["GPSLongitudeRef"])
            return lat, lon
        except Exception:
            return None

    # ── Indexing ──────────────────────────────────────────────────────

    def build_vocabulary(self, image_paths: list[Path], sample_per_image: int = 200) -> None:
        """Collect patch features from images and fit VLAD vocabulary."""
        all_feats: list[np.ndarray] = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                feats = np.asarray(self.extractor.extract(img), dtype=np.float32)
                # random subsample to keep memory bounded
                if len(feats) > sample_per_image:
                    idx = np.random.choice(len(feats), sample_per_image, replace=False)
                    feats = feats[idx]
                all_feats.append(feats)
            except Exception as e:
                logger.warning(f"Skip {p}: {e}")
        self.aggregator.fit(np.vstack(all_feats))
        self.aggregator.save(self.VOCAB_PATH)

        # Compute vlad dim
        test_feats = all_feats[0] if all_feats else np.zeros((1, self.extractor.feat_dim))
        vlad_vec = self.aggregator.encode(test_feats)
        self.vlad_dim = len(vlad_vec)
        self.FAISS_DIM_PATH.write_text(str(self.vlad_dim))
        self._index = FAISSIndex(self.vlad_dim)
        logger.info(f"VLAD dim = {self.vlad_dim}")

    def index_image(self, img_path: Path, meta: ImageMeta) -> int:
        """Encode one image and add to FAISS. Returns FAISS index id."""
        assert self.aggregator._fitted, "Build vocabulary first"
        img = Image.open(img_path).convert("RGB")
        feats = np.asarray(self.extractor.extract(img), dtype=np.float32)
        vlad_vec = self.aggregator.encode(feats)
        idx = self._ensure_index().add(vlad_vec, meta)
        return idx

    def index_all_images(self, metas: list[ImageMeta]) -> None:
        """Full pipeline: build vocab then index all images."""
        paths = [Path(m.filepath) for m in metas]
        self.build_vocabulary(paths)
        for meta in metas:
            try:
                self.index_image(Path(meta.filepath), meta)
                logger.debug(f"Indexed {meta.filepath}")
            except Exception as e:
                logger.warning(f"Failed to index {meta.filepath}: {e}")
        self._ensure_index().save()
        logger.info(f"Indexed {self._ensure_index().size} images.")

    # ── Query ─────────────────────────────────────────────────────────

    def query(
        self,
        img: Image.Image,
        top_k: int = settings.vpr_top_k,
        query_lat: float | None = None,
        query_lon: float | None = None,
    ) -> list[VPRMatch]:
        """
        Find the most visually similar stored places.
        Optionally re-rank by GPS proximity when coordinates are provided.
        """
        if not self.aggregator._fitted:
            return []
        feats = np.asarray(self.extractor.extract(img), dtype=np.float32)
        vlad_vec = self.aggregator.encode(feats)
        raw = self._ensure_index().search(vlad_vec, top_k * 2)

        results: list[VPRMatch] = []
        for score, meta in raw:
            dist_m = float("inf")
            if query_lat is not None and query_lon is not None:
                dist_m = _haversine(query_lat, query_lon, meta.lat, meta.lon)
            results.append(VPRMatch(
                image_id=meta.image_id,
                location_id=meta.location_id,
                location_name=meta.location_name,
                lat=meta.lat,
                lon=meta.lon,
                filepath=meta.filepath,
                caption=meta.caption,
                score=score,
                distance_m=dist_m,
            ))

        # Combined score: visual similarity + proximity weight
        if query_lat is not None:
            max_dist = max(r.distance_m for r in results) + 1e-6
            results.sort(
                key=lambda r: -(0.7 * r.score + 0.3 * (1 - min(r.distance_m, max_dist) / max_dist))
            )
        return results[:top_k]

    def query_from_path(
        self,
        img_path: Path,
        top_k: int = settings.vpr_top_k,
    ) -> list[VPRMatch]:
        img = Image.open(img_path).convert("RGB")
        gps = self.gps_from_exif(img_path)
        lat, lon = (gps if gps else (None, None))
        return self.query(img, top_k=top_k, query_lat=lat, query_lon=lon)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
