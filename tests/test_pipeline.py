from __future__ import annotations

import asyncio

import pytest

from app.pipeline.orchestrator import PipelineOrchestrator
from app.models.schemas import ProductAnalysis
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
        self.barcode_calls = 0
        self.name_calls = 0

    async def fetch_by_barcode(self, barcode: str):
        self.barcode_calls += 1
        return self.barcode_result

    async def fetch_by_name(self, name: str):
        self.name_calls += 1
        return self.name_result


class FakeProductCacheService:
    def __init__(self, cached_by_ean=None, cached_by_fingerprint=None):
        self.cached_by_ean = cached_by_ean
        self.cached_by_fingerprint = cached_by_fingerprint
        self.saved_calls = 0

    def build_fingerprint(self, name: str | None, brand: str | None) -> str:
        value = f"{(name or '').strip().lower()}|{(brand or '').strip().lower()}"
        return value

    def get_cached_analysis_by_ean(self, ean: str):
        return self.cached_by_ean

    def get_cached_analysis_by_fingerprint(self, fingerprint: str):
        return self.cached_by_fingerprint

    def save_analysis(self, analysis, *, ean: str | None, extraction_method: str, metadata_json=None):
        self.saved_calls += 1
        return analysis


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


def test_pipeline_uses_cached_result_and_skips_downstream_stages():
    barcode_service = FakeBarcodeService(BarcodeDetection(code="3017620422003", format_type="ean13"))
    lens_service = FakeLensService(None)
    ocr_service = FakeOCRService(None)
    provider = FakeFactsProvider()

    cache_service = FakeProductCacheService(
        cached_by_ean=ProductAnalysis(
            product_id="1",
            source="barcode",
            confidence=0.9,
            category="food",
            name="Nutella",
            brand=None,
            ingredients=["sugar"],
            additives=[],
            barcode="3017620422003",
            debug={},
        )
    )

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        product_cache_service=cache_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="99",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/99.jpg",
            detected_label="product",
        )
    )

    assert result.source == "barcode"
    assert lens_service.called is False
    assert ocr_service.called is False
    assert provider.barcode_calls == 0
    assert cache_service.saved_calls == 0


def test_pipeline_persists_new_analysis_result():
    barcode_service = FakeBarcodeService(BarcodeDetection(code="3017620422003", format_type="ean13"))
    lens_service = FakeLensService(None)
    ocr_service = FakeOCRService(None)
    facts = ProductFacts(name="Nutella", category="food", ingredients=["sugar", "oil"], additives=[])
    provider = FakeFactsProvider(barcode_result=facts)
    cache_service = FakeProductCacheService(cached_by_ean=None)

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        product_cache_service=cache_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="1",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/1.jpg",
            detected_label="product",
        )
    )

    assert result.source == "barcode"
    assert cache_service.saved_calls == 1
