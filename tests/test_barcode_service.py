from __future__ import annotations

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
