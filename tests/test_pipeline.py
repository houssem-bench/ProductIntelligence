from __future__ import annotations

import asyncio

import pytest

from app.pipeline.orchestrator import PipelineOrchestrator
from app.services.barcode_service import BarcodeDetection
from app.services.lens_service import LensResolution
from app.services.ocr_service import OCRExtraction
from app.models.schemas import ProductFacts


class FakeBarcodeService:
    def __init__(self, result):
        self.result = result

    def extract(self, image_path: str):
        return self.result


class FakeLensService:
    def __init__(self, result):
        self.result = result
        self.called = False

    async def resolve_name(self, crop_url: str, detected_label: str):
        self.called = True
        return self.result


class FakeOCRService:
    def __init__(self, result):
        self.result = result
        self.called = False

    def extract(self, image_path: str):
        self.called = True
        return self.result


class FakeFactsProvider:
    def __init__(self, barcode_result=None, name_result=None):
        self.barcode_result = barcode_result
        self.name_result = name_result

    async def fetch_by_barcode(self, barcode: str):
        return self.barcode_result

    async def fetch_by_name(self, name: str):
        return self.name_result


def test_pipeline_uses_barcode_first():
    barcode_service = FakeBarcodeService(BarcodeDetection(code="3017620422003", format_type="ean13"))
    lens_service = FakeLensService(None)
    ocr_service = FakeOCRService(None)
    facts = ProductFacts(name="Nutella", category="food", ingredients=["sugar", "oil"], additives=[])
    provider = FakeFactsProvider(barcode_result=facts)

    pipeline = PipelineOrchestrator(barcode_service, lens_service, ocr_service, provider)
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="1",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/1.jpg",
            detected_label="product",
        )
    )

    assert result.source == "barcode"
    assert lens_service.called is False
    assert ocr_service.called is False


def test_pipeline_uses_lens_before_ocr():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(LensResolution(title="Shampoo X", candidates=["Shampoo X"]))
    ocr_service = FakeOCRService(None)
    facts = ProductFacts(name="Shampoo X", category="cosmetic", ingredients=["aqua"], additives=[])
    provider = FakeFactsProvider(name_result=facts)

    pipeline = PipelineOrchestrator(barcode_service, lens_service, ocr_service, provider)
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="2",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/2.jpg",
            detected_label="product",
        )
    )

    assert result.source == "lens"
    assert lens_service.called is True
    assert ocr_service.called is False


def test_pipeline_falls_back_to_ocr_last():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(None)
    ocr_service = FakeOCRService(
        OCRExtraction(name="Unknown", category="food", ingredients=["sugar", "cocoa"], raw_text="ingredients: sugar, cocoa")
    )
    provider = FakeFactsProvider(name_result=None)

    pipeline = PipelineOrchestrator(barcode_service, lens_service, ocr_service, provider)
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="3",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/3.jpg",
            detected_label="product",
        )
    )

    assert result.source == "ocr"
    assert ocr_service.called is True
    assert len(result.ingredients) == 2
