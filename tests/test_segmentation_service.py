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
    monkeypatch.setattr(service, "_detect_with_contours", lambda _image: [Detection((0, 0, 100, 100), 0.35, "product")])

    detections = service._detect_products(image)

    assert len(detections) == 1
    assert detections[0].bbox == (10, 8, 40, 36)
    assert detections[0].confidence == 0.91


def test_detect_products_uses_watershed_before_grid(tmp_path, monkeypatch):
    service = SegmentationService(_build_settings(tmp_path))
    image = np.zeros((400, 600, 3), dtype=np.uint8)

    monkeypatch.setattr(service, "_detect_with_contours", lambda _image: [Detection((0, 0, 580, 390), 0.5, "product")])
    monkeypatch.setattr(
        service,
        "_detect_with_watershed",
        lambda _image: [
            Detection((30, 20, 180, 200), 0.61, "product"),
            Detection((250, 30, 180, 210), 0.59, "product"),
        ],
    )

    grid_called = {"value": False}

    def _fake_grid(_width: int, _height: int):
        grid_called["value"] = True
        return [Detection((0, 0, 100, 100), 0.35, "product")]

    monkeypatch.setattr(service, "_grid_fallback", _fake_grid)

    detections = service._detect_products(image)

    assert len(detections) >= 2
    assert grid_called["value"] is False


def test_detect_products_uses_grid_only_when_no_candidates(tmp_path, monkeypatch):
    service = SegmentationService(_build_settings(tmp_path))
    image = np.zeros((320, 480, 3), dtype=np.uint8)

    monkeypatch.setattr(service, "_detect_with_contours", lambda _image: [])
    monkeypatch.setattr(service, "_detect_with_watershed", lambda _image: [])

    sentinel = [Detection((1, 2, 3, 4), 0.35, "product")]
    monkeypatch.setattr(service, "_grid_fallback", lambda _w, _h: sentinel)

    detections = service._detect_products(image)

    assert len(detections) == 1
    assert detections[0].bbox == (1, 2, 3, 4)


def test_single_mode_uses_full_image_without_detectors(tmp_path, monkeypatch):
    service = SegmentationService(_build_settings(tmp_path))
    service._yolo = object()

    image = np.zeros((300, 500, 3), dtype=np.uint8)
    called = {"yolo": False, "contours": False, "watershed": False}

    def _fake_yolo(_image):
        called["yolo"] = True
        return [Detection((10, 10, 30, 30), 0.8, "product")]

    def _fake_contours(_image):
        called["contours"] = True
        return [Detection((20, 20, 50, 50), 0.4, "product")]

    def _fake_watershed(_image):
        called["watershed"] = True
        return [Detection((30, 30, 40, 40), 0.5, "product")]

    monkeypatch.setattr(service, "_detect_with_yolo", _fake_yolo)
    monkeypatch.setattr(service, "_detect_with_contours", _fake_contours)
    monkeypatch.setattr(service, "_detect_with_watershed", _fake_watershed)

    detections = service._detect_products(image, segmentation_mode="single")

    assert len(detections) == 1
    assert detections[0].bbox == (0, 0, 500, 300)
    assert detections[0].confidence == 1.0
    assert called["yolo"] is False
    assert called["contours"] is False
    assert called["watershed"] is False
