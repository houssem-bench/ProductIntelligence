from __future__ import annotations

import cv2

from app.services.barcode_service import BarcodeService


def test_normalize_numeric_code_accepts_valid_ean13():
    service = BarcodeService()
    assert service._normalize_numeric_code("3600551119816") == "3600551119816"


def test_normalize_numeric_code_normalizes_upc_to_ean13():
    service = BarcodeService()
    assert service._normalize_numeric_code("036000291452") == "0036000291452"


def test_normalize_numeric_code_rejects_invalid_checksum():
    service = BarcodeService()
    assert service._normalize_numeric_code("3600551119817") is None


def test_get_readiness_reports_disabled_by_config():
    service = BarcodeService(enabled=False)
    ready, reason, backends = service.get_readiness()

    assert ready is False
    assert reason == "disabled_by_config"
    assert backends["enabled"] is False


def test_extract_short_circuits_when_disabled(monkeypatch):
    service = BarcodeService(enabled=False)

    called = {"imread": False}

    def fake_imread(_path):
        called["imread"] = True
        return None

    monkeypatch.setattr(cv2, "imread", fake_imread)

    result = service.extract("anything.jpg")

    assert result is None
    assert called["imread"] is False
