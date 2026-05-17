"""
config/settings.py — Central configuration for LocalNavBot
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────
    app_name: str = "LocalNavBot"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    # Comma-separated list of allowed CORS origins.
    # Use "*" to allow all origins (convenient for mobile/LAN access).
    cors_origins: str = "*"

    # ── GPU ──────────────────────────────────────────────────────────
    device: str = "cuda"           # "cuda" | "cpu" | "mps"
    gpu_id: int = 0
    torch_dtype: str = "float16"   # "float16" | "float32" | "bfloat16"

    # ── Paths ─────────────────────────────────────────────────────────
    data_dir: Path = BASE_DIR / "data"
    images_dir: Path = BASE_DIR / "data" / "images"
    detections_dir: Path = BASE_DIR / "data" / "detections"
    realtime_frames_dir: Path = BASE_DIR / "data" / "realtime_frames"
    yolo_config_dir: Path = BASE_DIR / "data" / "yolo"
    ocr_models_dir: Path = BASE_DIR / "data" / "ocr_models"
    db_path: Path = BASE_DIR / "data" / "navbot.db"
    faiss_index_path: Path = BASE_DIR / "data" / "vpr_index.faiss"
    faiss_meta_path: Path = BASE_DIR / "data" / "vpr_meta.json"
    osm_cache_dir: Path = BASE_DIR / "data" / "osm_cache"

    # ── VPR / AnyLoc ─────────────────────────────────────────────────
    vpr_model: str = "dinov2_vitg14"   # dinov2_vitg14 | dinov2_vitl14 | dinov2_vitb14
    vpr_backend: str = "auto"          # auto | dinov2 | orb
    vpr_layer: int = 31
    vpr_facet: str = "value"
    vpr_num_clusters: int = 32          # VLAD clusters
    vpr_embed_dim: int = 1536           # ViT-G/14 feature dim
    vpr_top_k: int = 5                  # number of candidates to retrieve
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.25
    ocr_backend: str = "easyocr"
    ocr_languages: str = "en,vi"
    ocr_confidence: float = 0.35

    # ── Routing / Valhalla ────────────────────────────────────────────
    valhalla_url: str = "http://localhost:8002"
    valhalla_timeout: int = 10
    # Fallback: pure osmnx A* if Valhalla is not running
    use_osmnx_fallback: bool = True
    osm_auto_download: bool = False
    osm_area: str = "Dĩ An, Bình Dương, Vietnam"   # default area to cache
    osm_network_type: str = "drive"                  # drive | walk | bike | all
    allow_remote_geocoding: bool = False
    # Custom edges: skip injection if nearest OSM node farther than this (metres)
    custom_edge_snap_max_m: float = 80.0
    # OSMnx fallback: multiply edge weight if edge midpoint lies inside avoid disc
    route_avoid_disc_penalty: float = 5.0
    # Max alternate route summaries attached to primary route.analysis (OSMnx / Valhalla)
    route_alternates_max: int = 2

    # ── Traffic / Heuristic ───────────────────────────────────────────
    # Peak hours: each entry is (start_h, end_h, congestion_factor)
    # factor > 1.0 → slower / penalised
    peak_hours: list[tuple[int, int, float]] = [
        (6, 8, 1.8),    # sáng sớm cao điểm
        (11, 13, 1.3),  # trưa
        (17, 19, 2.0),  # chiều cao điểm
        (21, 23, 1.1),  # tối muộn
    ]
    local_road_bonus: float = 0.85   # < 1.0 → prefer local roads
    highway_penalty: float = 1.2     # > 1.0 → avoid big highways when possible
    crowd_penalty: float = 0.6
    weather_penalty: float = 0.45
    route_distance_weight: float = 0.15
    route_time_weight: float = 0.5
    route_congestion_weight: float = 0.2
    route_crowd_weight: float = 0.1
    route_weather_weight: float = 0.05
    route_complexity_weight: float = 0.08
    route_landmark_weight: float = 0.07
    route_locality_weight: float = 0.06
    route_turn_penalty_slight: float = 5.0
    route_turn_penalty_turn: float = 12.0
    route_turn_penalty_uturn: float = 30.0
    route_candidate_profiles: int = 3

    # ── Bot / LLM ─────────────────────────────────────────────────────
    realtime_enabled: bool = True
    ar_enabled: bool = True
    vps_enabled: bool = True
    slam_enabled: bool = False
    depth_enabled: bool = False
    lane_enabled: bool = False
    ollama_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b-instruct"
    realtime_frame_interval_ms: int = 400
    realtime_yolo_fps: float = 3.0
    realtime_ocr_interval_ms: int = 1500
    realtime_vpr_interval_ms: int = 2500
    realtime_frame_max_mb: int = 8
    sensor_fusion_mode: Literal["raw", "complementary", "kalman_lite"] = "complementary"
    map_match_mode: Literal["lightweight", "hmm"] = "lightweight"
    fusion_heading_alpha: float = 0.82
    fusion_position_alpha: float = 0.7

    llm_provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_api_key: str = ""
    llm_base_url: str = ""           # for Ollama: http://localhost:11434/v1
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 45
    chat_max_chars: int = 4000

    # ── Map display ───────────────────────────────────────────────────
    map_default_lat: float = 10.8720   # HCMUS Campus 2, Linh Trung, Thu Duc
    map_default_lon: float = 106.8042
    map_default_zoom: int = 17

    # ── Campus scope (HCMUS Campus 2, Linh Trung, Thu Duc) ───────────
    # Boundary defined by the internal roads of VNUHCM urban area:
    #   North  : Đường Marie Curie
    #   South  : Đường Isaac Newton
    #   East   : Quảng trường Sáng tạo / eastern boundary
    #   West   : Western fence of HCMUS CS2
    #
    # Polygon expanded ~30 m outward so edge locations are captured.
    # Outliers just outside (cafes on Marie Curie, bus stops) are fine.
    campus_boundary_enabled: bool = True
    # Fast bounding-box pre-filter
    campus_bbox_lat_min: float = 10.8690
    campus_bbox_lat_max: float = 10.8755
    campus_bbox_lon_min: float = 106.8015
    campus_bbox_lon_max: float = 106.8070
    # Convex polygon vertices [lat, lon], clockwise from NW
    # NW corner: intersection of Marie Curie & western fence
    # NE corner: Marie Curie & Quảng trường Sáng tạo
    # SE corner: Isaac Newton & Quảng trường Sáng tạo
    # SW corner: Isaac Newton & western fence
    campus_polygon: list[list[float]] = [
        [10.8752, 106.8018],  # NW — Marie Curie / west fence
        [10.8750, 106.8067],  # NE — Marie Curie / Quảng trường Sáng tạo
        [10.8693, 106.8065],  # SE — Isaac Newton / Quảng trường Sáng tạo
        [10.8695, 106.8017],  # SW — Isaac Newton / west fence
    ]

    # ── Indoor routing ────────────────────────────────────────────────
    # GPS accuracy threshold (metres) above which the router switches to
    # indoor mode if a floor map covers the area.
    indoor_gps_accuracy_threshold_m: float = 15.0

    # ── VIO (Visual-Inertial Odometry) ────────────────────────────────
    # Minimum VPR cosine similarity score to accept a re-localization fix.
    vio_vpr_min_score: float = 0.72
    # Drift threshold (metres) above which VPR re-localization is triggered.
    vio_drift_trigger_m: float = 2.0
    # Optical flow: pixels per metre at 1 m distance (calibrate per device).
    vio_flow_px_per_m: float = 554.0

    # ── Speech / Whisper ──────────────────────────────────────────────
    # Whisper model size: tiny | base | small | medium | large
    whisper_model: str = "base"
    # Device for Whisper inference — inherits from `device` by default.
    # Override with WHISPER_DEVICE=cpu if GPU OOM during transcription.
    whisper_device: str = ""   # empty = auto-inherit from `device`

    @property
    def effective_whisper_device(self) -> str:
        """Return the device to use for Whisper (falls back to main device)."""
        if self.whisper_device.strip():
            return self.whisper_device.strip()
        # On CPU-only systems always use cpu
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
        except ImportError:
            return "cpu"
        return self.device

    def setup_dirs(self) -> None:
        for d in [
            self.data_dir,
            self.images_dir,
            self.detections_dir,
            self.realtime_frames_dir,
            self.yolo_config_dir,
            self.ocr_models_dir,
            self.osm_cache_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def cors_origin_list(self) -> list[str]:
        raw = [v.strip() for v in self.cors_origins.split(",")]
        cleaned = [v for v in raw if v]
        # If wildcard is present, return it alone so FastAPI CORS middleware
        # uses allow_origins=["*"] which enables all origins.
        if "*" in cleaned:
            return ["*"]
        return cleaned

    @property
    def ocr_language_list(self) -> list[str]:
        raw = [v.strip() for v in self.ocr_languages.split(",")]
        return [v for v in raw if v]


settings = Settings()
