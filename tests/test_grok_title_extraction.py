from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings
from app.core.http import HttpClient
from app.services.grok_service import GrokService


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.last_url = None
        self.last_json_body = None
        self.last_headers = None

    async def post_json(self, url, *, json_body, headers=None):
        self.last_url = url
        self.last_json_body = json_body
        self.last_headers = headers
        return self.response


def _build_settings(
    *,
    grok_api_key: str = "test-key",
    grok_base_url: str = "https://api.x.ai/v1",
    grok_model: str = "grok-3-latest",
) -> Settings:
    return Settings(
        app_name="test",
        host="127.0.0.1",
        port=8000,
        debug=False,
        log_level="INFO",
        request_timeout_seconds=5.0,
        max_retries=1,
        retry_backoff_seconds=0.1,
        cache_ttl_seconds=30,
        max_parallel_analyses=2,
        database_url="sqlite:///test_grok_title_extraction.db",
        off_user_agent="test-agent",
        serpapi_key="",
        public_base_url="",
        enable_ngrok=False,
        ngrok_auth_token="",
        ngrok_domain="",
        lens_max_matches=5,
        lens_country="TN",
        lens_safe="off",
        grok_api_key=grok_api_key,
        grok_model=grok_model,
        grok_base_url=grok_base_url,
        enable_yolo=False,
        yolo_model_path="yolov8l-world.pt",
        yolo_conf_threshold=0.02,
        tesseract_cmd="",
        uploads_dir=Path("uploads"),
        incoming_dir=Path("uploads/incoming"),
        crops_dir=Path("uploads/crops"),
        phone_capture_ttl_seconds=300,
        phone_capture_max_upload_mb=15,
        phone_capture_poll_interval_ms=2000,
    )


def test_extract_product_title_from_url_results_builds_prompt_and_returns_json():
    fake_http = FakeHttpClient(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"brand":"Danao","company":"Delice",'
                            '"product_name":"Peche Abricot","confidence":0.95}'
                        )
                    }
                }
            ]
        }
    )
    service = GrokService(fake_http, _build_settings(grok_base_url="https://api.x.ai/v1"))

    result = asyncio.run(
        service.extract_product_title_from_url_results(
            url_results=[
                {
                    "title": "DANAO PECHE ABRICOT 20CL DELICE - Izy by Zedna",
                    "url": "https://shop.example/item",
                }
            ]
        )
    )

    assert result == {
        "brand": "Danao",
        "company": "Delice",
        "product_name": "Peche Abricot",
        "confidence": 0.95,
    }
    assert fake_http.last_url == "https://api.x.ai/v1/chat/completions"
    assert "DANAO PECHE ABRICOT 20CL DELICE - Izy by Zedna" in fake_http.last_json_body["messages"][1]["content"]
    assert "You are an information extraction system for e-commerce product titles." in fake_http.last_json_body["messages"][1]["content"]


def test_extract_product_title_from_url_results_returns_none_when_not_ready():
    fake_http = FakeHttpClient({"choices": []})
    service = GrokService(fake_http, _build_settings(grok_api_key=""))

    result = asyncio.run(
        service.extract_product_title_from_url_results(
            url_results=[{"title": "Some title"}]
        )
    )

    assert result is None
    assert fake_http.last_url is None


def test_extract_product_title_from_url_results_returns_none_when_no_title_present():
    fake_http = FakeHttpClient({"choices": []})
    service = GrokService(fake_http, _build_settings())

    result = asyncio.run(
        service.extract_product_title_from_url_results(
            url_results=[{"url": "https://shop.example/item"}]
        )
    )

    assert result is None
    assert fake_http.last_url is None


@pytest.mark.integration
def test_extract_product_title_from_url_results_real_api_optional():
    run_enabled = os.getenv("RUN_GROQ_INTEGRATION") == "1" or os.getenv("RUN_GROK_INTEGRATION") == "1"
    if not run_enabled:
        pytest.skip("Set RUN_GROQ_INTEGRATION=1 to run real Groq integration test")

    grok_api_key = (os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or "").strip()
    if not grok_api_key:
        pytest.skip("Set GROQ_API_KEY (or GROK_API_KEY) to run real Groq integration test")

    grok_base_url = (os.getenv("GROQ_BASE_URL") or os.getenv("GROK_BASE_URL") or "https://api.groq.com/openai/v1").strip()
    grok_model = (os.getenv("GROQ_MODEL") or os.getenv("GROK_MODEL") or "llama-3.3-70b-versatile").strip()

    async def _run() -> dict | None:
        settings = _build_settings(
            grok_api_key=grok_api_key,
            grok_base_url=grok_base_url,
            grok_model=grok_model,
        )
        http = HttpClient(settings)
        service = GrokService(http, settings)
        try:
            return await service.extract_product_title_from_url_results(
                url_results=[
                    {
                        "title": "DANAO PECHE ABRICOT 20CL DELICE - Izy by Zedna",
                        "url": "https://shop.example/item",
                    }
                ]
            )
        finally:
            await http.close()

    result = asyncio.run(_run())

    assert isinstance(result, dict)
    assert set(result.keys()) == {"brand", "company", "product_name", "confidence"}
    assert isinstance(result["brand"], (str, type(None)))
    assert isinstance(result["company"], (str, type(None)))
    assert isinstance(result["product_name"], (str, type(None)))
    assert isinstance(result["confidence"], (float, int, type(None)))
    if result["confidence"] is not None:
        assert 0.0 <= float(result["confidence"]) <= 1.0
