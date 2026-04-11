from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    host: str
    port: int
    debug: bool
    log_level: str

    request_timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    cache_ttl_seconds: int
    max_parallel_analyses: int

    off_user_agent: str
    serpapi_key: str
    public_base_url: str

    enable_yolo: bool
    yolo_model_path: str
    yolo_conf_threshold: float
    tesseract_cmd: str

    uploads_dir: Path
    incoming_dir: Path
    crops_dir: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    uploads_dir = PROJECT_ROOT / "uploads"
    incoming_dir = uploads_dir / "incoming"
    crops_dir = uploads_dir / "crops"

    incoming_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    default_model_path = str((PROJECT_ROOT.parent / "yolov8l-seg.pt").resolve())

    return Settings(
        app_name=os.getenv("APP_NAME", "ProductIntelligence_V2"),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        debug=_to_bool(os.getenv("DEBUG"), default=False),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10")),
        max_retries=int(os.getenv("MAX_RETRIES", "2")),
        retry_backoff_seconds=float(os.getenv("RETRY_BACKOFF_SECONDS", "0.5")),
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "900")),
        max_parallel_analyses=max(1, int(os.getenv("MAX_PARALLEL_ANALYSES", "4"))),
        off_user_agent=os.getenv("OFF_USER_AGENT", "ProductIntelligenceV2/1.0"),
        serpapi_key=os.getenv("SERPAPI_KEY", "").strip(),
        public_base_url=os.getenv("PUBLIC_BASE_URL", "").rstrip("/"),
        enable_yolo=_to_bool(os.getenv("ENABLE_YOLO"), default=True),
        yolo_model_path=os.getenv("YOLO_MODEL_PATH", default_model_path),
        yolo_conf_threshold=float(os.getenv("YOLO_CONF_THRESHOLD", "0.2")),
        tesseract_cmd=os.getenv("TESSERACT_CMD", "").strip(),
        uploads_dir=uploads_dir,
        incoming_dir=incoming_dir,
        crops_dir=crops_dir,
    )
