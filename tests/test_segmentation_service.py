from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.services.segmentation_service import Detection, SegmentationService


def _build_settings(tmp_path: Path) -> Settings:
    uploads_dir = tmp_path / "uploads"
    incoming_dir = uploads_dir / "incoming"
    crops_dir = uploads_dir / "crops"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

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
        off_user_agent="test-agent",
        serpapi_key="",
        public_base_url="",
        enable_ngrok=False,
        ngrok_auth_token="",
        ngrok_domain="",
        lens_max_matches=5,
        lens_country="TN",
        lens_safe="off",
        grok_api_key="",
        grok_model="grok-3-latest",
        grok_base_url="https://api.x.ai/v1",
        enable_yolo=False,
        yolo_model_path="yolov8l-seg.pt",
        yolo_conf_threshold=0.2,
        tesseract_cmd="",
        uploads_dir=uploads_dir,
        incoming_dir=incoming_dir,
        crops_dir=crops_dir,
    )


def test_detect_products_prefers_yolo_when_available(tmp_path, monkeypatch):
    service = SegmentationService(_build_settings(tmp_path))
    service._yolo = object()

    image = np.zeros((120, 120, 3), dtype=np.uint8)

    monkeypatch.setattr(service, "_detect_with_yolo", lambda _image: [Detection((10, 8, 40, 36), 0.91, "product")])

    detections = service._detect_products(image)

    assert len(detections) == 1
    assert detections[0].bbox == (10, 8, 40, 36)
    assert detections[0].confidence == 0.91


def test_detect_products_returns_empty_when_yolo_empty(tmp_path, monkeypatch):
    service = SegmentationService(_build_settings(tmp_path))
    service._yolo = object()
    image = np.zeros((320, 480, 3), dtype=np.uint8)

    monkeypatch.setattr(service, "_detect_with_yolo", lambda _image: [])

    detections = service._detect_products(image)

    assert detections == []


def test_detect_products_returns_empty_when_both_detectors_empty(tmp_path, monkeypatch):
    service = SegmentationService(_build_settings(tmp_path))
    service._yolo = object()
    image = np.zeros((320, 480, 3), dtype=np.uint8)

    monkeypatch.setattr(service, "_detect_with_yolo", lambda _image: [])

    detections = service._detect_products(image)

    assert detections == []


def test_single_mode_skips_segmentation_and_uses_full_image(tmp_path, monkeypatch):
    service = SegmentationService(_build_settings(tmp_path))
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    called = {"yolo": False}

    def _fake_yolo(_image):
        called["yolo"] = True
        return [Detection((10, 10, 120, 100), 0.81, "product")]

    monkeypatch.setattr(service, "_detect_with_yolo", _fake_yolo)

    detections = service._detect_products(image, segmentation_mode="single")

    assert len(detections) == 1
    assert detections[0].bbox == (0, 0, 500, 300)
    assert detections[0].confidence == 1.0
    assert called["yolo"] is False
