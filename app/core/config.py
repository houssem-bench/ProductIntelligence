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
    database_url: str

    off_user_agent: str
    serpapi_key: str
    public_base_url: str
    enable_ngrok: bool
    ngrok_auth_token: str
    ngrok_domain: str
    lens_max_matches: int
    lens_country: str
    lens_safe: str
    grok_api_key: str
    grok_model: str
    grok_base_url: str

    enable_yolo: bool
    yolo_model_path: str
    yolo_conf_threshold: float
    tesseract_cmd: str

    uploads_dir: Path
    incoming_dir: Path
    crops_dir: Path
    phone_capture_ttl_seconds: int
    phone_capture_max_upload_mb: int
    phone_capture_poll_interval_ms: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    uploads_dir = PROJECT_ROOT / "uploads"
    incoming_dir = uploads_dir / "incoming"
    crops_dir = uploads_dir / "crops"

    incoming_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    default_model_path = str((PROJECT_ROOT / "yolov8l-world.pt").resolve())
    default_database_url = f"sqlite:///{(PROJECT_ROOT / 'product_cache.db').resolve()}"

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
        database_url=os.getenv("DATABASE_URL", default_database_url).strip() or default_database_url,
        off_user_agent=os.getenv("OFF_USER_AGENT", "ProductIntelligenceV2/1.0"),
        serpapi_key=os.getenv("SERPAPI_KEY", "e8715f247716f03e6f730763c1e60df0ec78c84c90d508b9c3f7dea7696dcb50").strip(),
        public_base_url=(os.getenv("PUBLIC_BASE_URL") or os.getenv("NGROK_BASE_URL", "https://subattenuate-joanne-vacuous.ngrok-free.dev")).rstrip("/"),
        enable_ngrok=_to_bool(os.getenv("ENABLE_NGROK"), default=False),
        ngrok_auth_token=os.getenv("NGROK_AUTHTOKEN", "").strip(),
        ngrok_domain=os.getenv("NGROK_DOMAIN", "https://subattenuate-joanne-vacuous.ngrok-free.dev").strip(),
        lens_max_matches=max(1, int(os.getenv("LENS_MAX_MATCHES", "5"))),
        lens_country=os.getenv("LENS_COUNTRY", "TN").strip(),
        lens_safe=os.getenv("LENS_SAFE", "off").strip().lower(),
        grok_api_key=os.getenv("GROK_API_KEY", "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").strip(),
        grok_model=os.getenv("GROK_MODEL", "llama-3.3-70b-versatile").strip(),
        grok_base_url=(os.getenv("GROK_BASE_URL", "https://api.groq.com/openai/v1").strip() or "https://api.groq.com/openai/v11").rstrip("/"),
        enable_yolo=_to_bool(os.getenv("ENABLE_YOLO"), default=True),
        yolo_model_path=os.getenv("YOLO_MODEL_PATH", default_model_path),
        yolo_conf_threshold=float(os.getenv("YOLO_CONF_THRESHOLD", "0.015")),
        tesseract_cmd=os.getenv("TESSERACT_CMD", "").strip(),
        uploads_dir=uploads_dir,
        incoming_dir=incoming_dir,
        crops_dir=crops_dir,
        phone_capture_ttl_seconds=max(30, int(os.getenv("PHONE_CAPTURE_TTL_SECONDS", "300"))),
        phone_capture_max_upload_mb=max(1, int(os.getenv("PHONE_CAPTURE_MAX_UPLOAD_MB", "15"))),
        phone_capture_poll_interval_ms=max(500, int(os.getenv("PHONE_CAPTURE_POLL_INTERVAL_MS", "2000"))),
    )
