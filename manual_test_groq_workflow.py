from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.core.config import Settings
from app.core.http import HttpClient
from app.services.grok_service import GrokService


# Fill these values directly here (no .env needed).
GROQ_API_KEY = "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"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Any title you want to test.
INPUT_TITLE = "French Click - Danao Lait Et Jus De Fruits Peche Abricot 1L OLD"


def build_settings() -> Settings:
    project_root = Path(__file__).resolve().parent
    uploads_dir = project_root / "uploads"
    incoming_dir = uploads_dir / "incoming"
    crops_dir = uploads_dir / "crops"

    incoming_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_name="manual-groq-test",
        host="127.0.0.1",
        port=8000,
        debug=False,
        log_level="INFO",
        request_timeout_seconds=30.0,
        max_retries=1,
        retry_backoff_seconds=0.2,
        cache_ttl_seconds=60,
        max_parallel_analyses=1,
        off_user_agent="manual-groq-test/1.0",
        serpapi_key="",
        public_base_url="",
        enable_ngrok=False,
        ngrok_auth_token="",
        ngrok_domain="",
        lens_max_matches=5,
        lens_country="TN",
        lens_safe="off",
        grok_api_key=GROQ_API_KEY,
        grok_model=GROQ_MODEL,
        grok_base_url=GROQ_BASE_URL,
        enable_yolo=False,
        yolo_model_path="yolov8l-world.pt",
        yolo_conf_threshold=0.02,
        tesseract_cmd="",
        uploads_dir=uploads_dir,
        incoming_dir=incoming_dir,
        crops_dir=crops_dir,
    )


async def run() -> None:
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("REPLACE_WITH"):
        raise RuntimeError("Set GROQ_API_KEY in this file before running.")

    settings = build_settings()
    http = HttpClient(settings)
    service = GrokService(http, settings)

    try:
        result = await service.extract_product_title_from_url_results(
            url_results=[
                {
                    "title": INPUT_TITLE,
                    "url": "https://example.com/product",
                }
            ]
        )
    finally:
        await http.close()

    print("Input title:")
    print(INPUT_TITLE)
    print("\nExtracted JSON:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
