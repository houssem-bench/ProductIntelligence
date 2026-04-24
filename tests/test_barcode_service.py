from __future__ import annotations

import cv2
import numpy as np
import pytest

import app.services.barcode_service as barcode_module
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


def test_extract_stops_after_max_candidates(monkeypatch):
    service = BarcodeService(max_candidates=2, max_decode_seconds=10.0)

    monkeypatch.setattr(cv2, "imread", lambda _path: np.zeros((10, 10, 3), dtype=np.uint8))
    monkeypatch.setattr(service, "_iter_candidates", lambda _image: [np.zeros((10, 10, 3), dtype=np.uint8)] * 6)

    calls = {"qr": 0, "barcode": 0, "pyzbar": 0}

    def fake_qr(_img):
        calls["qr"] += 1
        return None

    def fake_barcode(_img):
        calls["barcode"] += 1
        return None

    def fake_pyzbar(_img):
        calls["pyzbar"] += 1
        return None

    monkeypatch.setattr(service, "_decode_qr", fake_qr)
    monkeypatch.setattr(service, "_decode_barcode", fake_barcode)
    monkeypatch.setattr(service, "_decode_with_pyzbar", fake_pyzbar)

    assert service.extract("x.jpg") is None
    assert calls["qr"] == 4
    assert calls["barcode"] == 4
    assert calls["pyzbar"] == 4


def test_decode_with_pyzbar_uses_symbol_filter_when_available(monkeypatch):
    if barcode_module.zbar_decode is None:
        pytest.skip("pyzbar unavailable")

    calls: list[dict[str, object]] = []

    def fake_decode(_img, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(barcode_module, "zbar_decode", fake_decode)

    service = BarcodeService()
    result = service._decode_with_pyzbar(np.zeros((16, 16, 3), dtype=np.uint8))

    assert result is None
    assert len(calls) == 1
    if barcode_module.ZBarSymbol is None:
        assert calls[0] == {}
    else:
        assert "symbols" in calls[0]
        assert len(calls[0]["symbols"]) >= 4
