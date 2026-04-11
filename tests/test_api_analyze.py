from __future__ import annotations

import asyncio
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import BBox, ProductAnalysis, SegmentedProduct
from app.services.segmentation_service import SessionProduct


def test_analyze_endpoint_returns_segmented_products(monkeypatch):
    async def fake_segment_upload(_image, *, segmentation_mode: str = "auto", expected_products=None):
        return "session-123", [
            SegmentedProduct(
                product_id="1",
                label="product",
                confidence=0.9,
                bbox=BBox(x=0, y=0, width=100, height=100),
                crop_url="/uploads/crops/session-123/1.jpg",
            )
        ]

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.container.segmentation_service, "segment_upload", fake_segment_upload)
        response = client.post(
            "/analyze",
            files={"image": ("sample.jpg", b"fake-image-content", "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "session-123"
    assert payload["segmentation_mode"] == "auto"
    assert payload["total_products"] == 1
    assert payload["products"][0]["product_id"] == "1"


def test_analyze_endpoint_rejects_invalid_segmentation_mode():
    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            data={"segmentation_mode": "bad-mode"},
            files={"image": ("sample.jpg", b"fake-image-content", "image/jpeg")},
        )

    assert response.status_code == 400
    payload = response.json()
    assert "request_id" in payload
    assert "segmentation_mode" in payload["detail"]


def test_analyze_endpoint_accepts_single_mode(monkeypatch):
    captured = {"mode": None, "expected": None}

    async def fake_segment_upload(_image, *, segmentation_mode: str = "auto", expected_products=None):
        captured["mode"] = segmentation_mode
        captured["expected"] = expected_products
        return "single-session", [
            SegmentedProduct(
                product_id="1",
                label="product",
                confidence=1.0,
                bbox=BBox(x=0, y=0, width=120, height=140),
                crop_url="/uploads/crops/single-session/1.jpg",
            )
        ]

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.container.segmentation_service, "segment_upload", fake_segment_upload)
        response = client.post(
            "/analyze",
            data={"segmentation_mode": "single"},
            files={"image": ("sample.jpg", b"fake-image-content", "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "single-session"
    assert payload["segmentation_mode"] == "single"
    assert payload["total_products"] == 1
    assert captured["mode"] == "single"
    assert captured["expected"] is None


def test_analyze_endpoint_rejects_non_image():
    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            files={"image": ("sample.txt", b"not-image", "text/plain")},
        )

    assert response.status_code == 400
    payload = response.json()
    assert "request_id" in payload
    assert response.headers.get("x-request-id")


def test_analyze_selected_invalid_json_has_request_id():
    with TestClient(app) as client:
        response = client.post(
            "/analyze/selected",
            data='{"session_id":',
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422
    payload = response.json()
    assert "request_id" in payload
    assert isinstance(payload.get("detail"), list)
    assert response.headers.get("x-request-id")


def test_analyze_selected_respects_max_parallel_analyses(monkeypatch):
    with TestClient(app) as client:
        patched_settings = replace(app.state.container.settings, max_parallel_analyses=2)
        monkeypatch.setattr(app.state.container, "settings", patched_settings)

        session_products = {
            str(i): SessionProduct(
                product_id=str(i),
                label="product",
                confidence=0.9,
                bbox=BBox(x=i * 10, y=i * 10, width=100, height=120),
                crop_path=f"tmp/{i}.jpg",
                source_image_path=f"tmp/source_{i}.jpg",
                crop_url=f"/uploads/crops/session-1/{i}.jpg",
            )
            for i in range(1, 5)
        }
        monkeypatch.setattr(
            app.state.container.segmentation_service,
            "get_session_products",
            lambda _session_id: session_products,
        )

        state = {"running": 0, "peak": 0}
        lock = asyncio.Lock()

        async def fake_analyze_product(
            *,
            product_id: str,
            image_path: str,
            original_image_path: str | None = None,
            crop_url: str,
            detected_label: str,
        ):
            async with lock:
                state["running"] += 1
                state["peak"] = max(state["peak"], state["running"])

            await asyncio.sleep(0.03)

            async with lock:
                state["running"] -= 1

            return ProductAnalysis(
                product_id=product_id,
                source="ocr",
                confidence=0.55,
                category="unknown",
                name=detected_label,
                brand=None,
                ingredients=[],
                additives=[],
                debug={},
            )

        monkeypatch.setattr(app.state.container.pipeline, "analyze_product", fake_analyze_product)

        response = client.post(
            "/analyze/selected",
            json={
                "session_id": "session-1",
                "product_ids": ["1", "2", "3", "4"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analyzed_count"] == 4
    assert state["peak"] <= 2
