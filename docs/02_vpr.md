# 02 — Visual Place Recognition (VPR)

## Overview

The VPR module enables the system to identify a physical location from a camera image. It implements the **AnyLoc** recipe: DINOv2 patch features → VLAD aggregation → FAISS cosine similarity search. This allows LocalNavBot to answer "where am I?" from a photo, re-localize the VIO dead-reckoning system when GPS drift is detected, and attach illustrative photos to route steps.

The pipeline is GPU-accelerated (float16 on CUDA) with a CPU fallback using ORB descriptors when PyTorch or DINOv2 is unavailable.

---

## Architecture / Data Flow

```
Query Image (PIL)
      │
      ▼
DINOv2Extractor.extract()          ← hook on transformer block[layer]
      │  (N_patches × feat_dim)    ← float32 numpy array
      ▼
VLADAggregator.encode()
      │  (K × feat_dim) → flatten → L2-normalise
      │  → (K*feat_dim,) VLAD descriptor
      ▼
FAISSIndex.search(top_k * 2)       ← IndexFlatIP cosine similarity
      │  [(score, ImageMeta), ...]
      ▼
GPS re-ranking (if query_lat/lon provided)
      │  combined_score = 0.7 × visual + 0.3 × proximity
      ▼
VPRMatch list (top_k results)

─────────────────────────────────────────────────────
Index build path:

All DB images
      │
      ▼
build_vocabulary()                 ← MiniBatchKMeans on patch features
      │  saves vlad_centroids.npy
      ▼
index_image() × N                  ← encode + FAISSIndex.add()
      │
      ▼
FAISSIndex.save()                  ← vpr_index.faiss + vpr_meta.json
```

---

## Key Classes and Functions

### `core/vpr_engine.py`

#### `DINOv2Extractor`

```python
class DINOv2Extractor:
    def __init__(model_name, layer, facet, device)
    def extract(img: Image.Image, size: int = 448) -> np.ndarray  # (N_patches, feat_dim)
```

- Loads `dinov2_vitg14` (or `vitl14`/`vitb14`) from `torch.hub`.
- Registers a forward hook on `model.blocks[layer].attn.proj` to capture intermediate features.
- Drops the CLS token; returns patch tokens as `(N_patches, feat_dim)` float32 array.
- On CUDA with `torch_dtype=float16`, the model runs in half precision for ~2× speed.
- Default: layer 31, facet `"value"`, input size 448×448.
- `feat_dim` is determined dynamically by running a dummy forward pass.

**ViT-G/14 dimensions**: 1536-dim features, ~1024 patches at 448px input.

#### `ORBExtractor` (CPU fallback)

```python
class ORBExtractor:
    def extract(img: Image.Image, size: int = 448) -> np.ndarray  # (N_keypoints, 32)
```

Uses OpenCV `ORB_create(nfeatures=2048)`. Returns binary descriptors as float32. Used when `VPR_BACKEND=orb` or when DINOv2 fails to load.

---

#### `VLADAggregator`

```python
class VLADAggregator:
    def fit(all_feats: np.ndarray) -> None          # MiniBatchKMeans
    def encode(patch_feats: np.ndarray) -> np.ndarray  # (K*D,) L2-normalised
    def save(path: Path) -> None
    def load(path: Path) -> None
```

**Vocabulary building** (`fit`):
- Runs `MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=5, max_iter=300)` on all collected patch features.
- `K = settings.vpr_num_clusters` (default 32).

**VLAD encoding** (`encode`):
1. Assign each patch to its nearest centroid.
2. Accumulate residuals: `vlad[k] = Σ (patch - centroid_k)` for all patches assigned to k.
3. Intra-normalise: `sign(vlad) × sqrt(|vlad|)` (power normalisation).
4. L2-normalise the full vector.
5. Output: `(K × D,)` = `32 × 1536 = 49152` dimensions for ViT-G/14.

---

#### `FAISSIndex`

```python
class FAISSIndex:
    def add(vector: np.ndarray, meta: ImageMeta) -> int
    def search(query_vec: np.ndarray, top_k: int) -> list[tuple[float, ImageMeta]]
    def save() -> None
    @property size: int
```

- Uses `faiss.IndexFlatIP` (inner product = cosine similarity after L2 normalisation).
- Falls back to numpy matrix multiplication if `faiss` is not installed.
- Vectors are L2-normalised before insertion and before search.
- Metadata is stored in a parallel list and serialised to `vpr_meta.json`.
- Index is saved to `vpr_index.faiss` (binary format).

#### `ImageMeta` (dataclass)
```python
@dataclass
class ImageMeta:
    image_id: int
    location_id: int
    location_name: str
    lat: float
    lon: float
    filepath: str
    caption: str
    faiss_idx: int = -1
```

#### `VPRMatch` (NamedTuple)
```python
class VPRMatch(NamedTuple):
    image_id: int
    location_id: int
    location_name: str
    lat: float
    lon: float
    filepath: str
    caption: str
    score: float        # cosine similarity [0, 1]
    distance_m: float   # geodesic distance from query GPS
```

---

#### `VPREngine`

```python
class VPREngine:
    def build_vocabulary(image_paths: list[Path], sample_per_image: int = 200) -> None
    def index_image(img_path: Path, meta: ImageMeta) -> int
    def index_all_images(metas: list[ImageMeta]) -> None
    def query(img: Image.Image, top_k: int, query_lat, query_lon) -> list[VPRMatch]
    def query_from_path(img_path: Path, top_k: int) -> list[VPRMatch]
    @staticmethod gps_from_exif(img_path: Path) -> tuple[float, float] | None
```

**`build_vocabulary`**: Collects up to `sample_per_image` random patches from each image, stacks them, and calls `VLADAggregator.fit()`. Saves centroids to `data/vlad_centroids.npy`.

**`index_image`**: Encodes one image to a VLAD vector and adds it to the FAISS index. Returns the FAISS row index.

**`query`**: 
1. Extracts VLAD descriptor from query image.
2. Searches FAISS for `top_k * 2` candidates.
3. If GPS provided, re-ranks by combined score: `0.7 × visual_score + 0.3 × (1 - normalised_distance)`.
4. Returns top `top_k` matches.

**GPS re-ranking**: Normalises distances by the maximum distance in the candidate set. This means a visually strong match that is also geographically close scores highest.

---

### `web/routes/vpr.py`

#### `POST /api/vpr/query`

```
Content-Type: multipart/form-data
Fields:
  file  (required) — image file
  lat   (optional) — float, current GPS latitude
  lon   (optional) — float, current GPS longitude
```

Returns:
```json
{
  "ok": true,
  "matches": [
    {
      "location_name": "Chợ Dĩ An",
      "location_id": 42,
      "lat": 10.9085,
      "lon": 106.760,
      "score": 0.8734,
      "distance_m": 12.5,
      "caption": "Cổng chính chợ Dĩ An, biển hiệu đỏ",
      "images": ["data/images/cho_di_an_01.jpg"]
    }
  ]
}
```

#### `POST /api/vpr/rebuild`

Triggers a background job to rebuild the entire VPR index from all images in the database. Returns a job ID for polling.

```json
{
  "ok": true,
  "message": "Rebuilding VPR index in background",
  "job": {"job_id": "abc123", "status": "queued"}
}
```

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `VPR_MODEL` | `dinov2_vitg14` | DINOv2 variant: `vitg14` \| `vitl14` \| `vitb14` |
| `VPR_BACKEND` | `auto` | `auto` \| `dinov2` \| `orb` |
| `VPR_LAYER` | `31` | Transformer block index for feature extraction |
| `VPR_FACET` | `value` | Feature facet: `value` \| `key` \| `token` |
| `VPR_NUM_CLUSTERS` | `32` | VLAD vocabulary size (K) |
| `VPR_TOP_K` | `5` | Number of matches to return |
| `DEVICE` | `cuda` | `cuda` \| `cpu` \| `mps` |
| `TORCH_DTYPE` | `float16` | `float16` \| `float32` |

---

## How to Test

### Query with an image

```bash
curl -X POST http://192.168.1.217:8000/api/vpr/query \
  -F "file=@/path/to/photo.jpg" \
  -F "lat=10.9085" \
  -F "lon=106.760"
```

### Rebuild index

```bash
curl -X POST http://192.168.1.217:8000/api/vpr/rebuild
```

### Check VPR status

```bash
curl http://192.168.1.217:8000/api/status | python -m json.tool
# Look for: "vpr_ready": true, "vpr_index_size": 150
```

### Poll rebuild job

```bash
curl http://192.168.1.217:8000/api/jobs/{job_id}
```

---

## Healthy Output Examples

**Status check:**
```json
{
  "vpr_ready": true,
  "vpr_index_size": 150,
  "vpr_backend": "dinov2"
}
```

**Query result (healthy):**
```json
{
  "ok": true,
  "matches": [
    {"location_name": "Ngã tư Bình Dương", "score": 0.8921, "distance_m": 8.3}
  ]
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `503 "VPR not available"` | VPREngine failed to initialise | Check GPU/CUDA availability; set `VPR_BACKEND=orb` for CPU-only |
| `vpr_ready: false` | Vocabulary not built yet | Upload images and call `POST /api/vpr/rebuild` |
| Empty `matches` array | No images indexed, or all scores below threshold | Rebuild index; check `vpr_index_size > 0` |
| `"Build vocabulary first"` | `index_image()` called before `build_vocabulary()` | Always rebuild after adding new images |
| DINOv2 download fails | No internet access | Pre-download model or set `VPR_BACKEND=orb` |
| FAISS not installed | `faiss-cpu` or `faiss-gpu` missing | `pip install faiss-cpu`; numpy fallback is automatic but slower |
| Low scores (< 0.5) | Images too different from database | Add more diverse reference images; reduce `VPR_NUM_CLUSTERS` |

---

## Performance Notes

- **DINOv2 ViT-G/14 on RTX 3060**: ~80–150 ms per query image (448px, float16).
- **DINOv2 ViT-B/14 on CPU**: ~2–5 s per query — use `VPR_MODEL=dinov2_vitb14` for CPU.
- **ORB fallback**: ~5–20 ms per query, but significantly lower accuracy.
- **FAISS IndexFlatIP**: O(N × D) search — for 10,000 images at D=49152, ~50 ms on CPU.
- **Vocabulary build**: ~30–120 s for 500 images on GPU; run once, cached to disk.
- **Memory**: ViT-G/14 float16 ≈ 2.5 GB VRAM; ViT-B/14 float16 ≈ 350 MB VRAM.
- The numpy fallback index is ~5–10× slower than FAISS for large collections (>1000 images).
