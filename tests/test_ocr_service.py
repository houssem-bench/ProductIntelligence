from __future__ import annotations

from types import SimpleNamespace

import cv2

from app.services.ocr_service import OCRService


def test_get_readiness_reports_disabled_by_config():
    service = OCRService(SimpleNamespace(enable_ocr=False, tesseract_cmd=""))

    ready, reason = service.get_readiness()

    assert ready is False
    assert reason == "disabled_by_config"


def test_extract_short_circuits_when_disabled(monkeypatch):
    service = OCRService(SimpleNamespace(enable_ocr=False, tesseract_cmd=""))

    called = {"imread": False}

    def fake_imread(_path):
        called["imread"] = True
        return None

    monkeypatch.setattr(cv2, "imread", fake_imread)

    result = service.extract("anything.jpg")

    assert result is None
    assert called["imread"] is False