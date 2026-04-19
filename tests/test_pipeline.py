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


class FakeGrokService:
    def __init__(self, result=None):
        self.result = result
        self.called = False

    async def extract_product_title_from_url_results(self, *, url_results):
        self.called = True
        return self.result


class FakeFactsProvider:
    def __init__(self, barcode_result=None, name_result=None, name_results_by_query=None):
        self.barcode_result = barcode_result
        self.name_result = name_result
        self.name_results_by_query = name_results_by_query or {}
        self.barcode_calls = 0
        self.name_calls = 0
        self.name_queries: list[str] = []
        self.name_preferred_categories: list[str | None] = []

    async def fetch_by_barcode(self, barcode: str):
        self.barcode_calls += 1
        return self.barcode_result

    async def fetch_by_name(self, name: str, *, preferred_category: str | None = None):
        self.name_calls += 1
        self.name_queries.append(name)
        self.name_preferred_categories.append(preferred_category)
        if name in self.name_results_by_query:
            return self.name_results_by_query[name]
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


def test_pipeline_uses_groq_brand_company_before_lens_title_lookup():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(LensResolution(title="DANAO PECHE ABRICOT 20CL DELICE", candidates=["DANAO PECHE ABRICOT 20CL DELICE"]))
    ocr_service = FakeOCRService(None)

    facts = ProductFacts(name="Danao Peche Abricot", category="food", ingredients=["water", "milk"], additives=[])
    provider = FakeFactsProvider(
        name_results_by_query={
            "Peche Abricot Danao Delice": facts,
        }
    )
    grok_service = FakeGrokService(
        {
            "brand": "Danao",
            "company": "Delice",
            "product_name": "Peche Abricot",
            "confidence": 0.95,
        }
    )

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        grok_service=grok_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="7",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/7.jpg",
            detected_label="product",
        )
    )

    assert result.source == "lens"
    assert grok_service.called is True
    assert ocr_service.called is False
    assert provider.name_queries[0] == "Peche Abricot Danao Delice"
    assert "DANAO PECHE ABRICOT 20CL DELICE" not in provider.name_queries


def test_pipeline_groq_null_brand_company_falls_back_to_first_word_query():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(LensResolution(title="DANAO PECHE ABRICOT 20CL", candidates=["DANAO PECHE ABRICOT 20CL"]))
    ocr_service = FakeOCRService(None)

    facts = ProductFacts(name="Danao Drink", category="food", ingredients=["water"], additives=[])
    provider = FakeFactsProvider(
        name_results_by_query={
            "DANAO": facts,
        }
    )
    grok_service = FakeGrokService(
        {
            "brand": None,
            "company": None,
            "product_name": None,
            "confidence": 0.5,
        }
    )

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        grok_service=grok_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="8",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/8.jpg",
            detected_label="food product",
        )
    )

    assert result.source == "lens"
    assert provider.name_queries[0] == "DANAO"
    assert provider.name_preferred_categories[0] == "food"


def test_pipeline_rejects_broad_brand_match_without_product_overlap():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(LensResolution(title="Thon entier a l'huile vegetale Sidi Ali 140g", candidates=["Thon entier a l'huile vegetale Sidi Ali 140g"]))
    ocr_service = FakeOCRService(None)

    broad_brand_only = ProductFacts(
        name="Sidi Ali Eau Minerale",
        category="food",
        ingredients=["water", "calcium"],
        additives=[],
    )
    specific = ProductFacts(
        name="Thon entier Sidi Ali",
        category="food",
        ingredients=["thon", "huile vegetale", "sel"],
        additives=[],
    )
    provider = FakeFactsProvider(
        name_results_by_query={
            "Thon entier Sidi Ali": broad_brand_only,
            "Thon entier Sidi": specific,
        }
    )
    grok_service = FakeGrokService(
        {
            "brand": "Sidi",
            "company": "Ali",
            "product_name": "Thon entier",
            "confidence": 0.92,
        }
    )

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        grok_service=grok_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="10",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/10.jpg",
            detected_label="food product",
        )
    )

    assert result.source == "lens"
    assert result.name == "Thon entier Sidi Ali"
    assert provider.name_queries[:2] == ["Thon entier Sidi Ali", "Thon entier Sidi"]


def test_pipeline_does_not_use_product_name_only_when_brand_or_company_present():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(LensResolution(title="Double concentre tomate Sicam", candidates=["Double concentre tomate Sicam"]))
    ocr_service = FakeOCRService(None)

    specific = ProductFacts(
        name="Double concentre de tomate Sicam",
        category="food",
        ingredients=["tomate", "sel"],
        additives=[],
    )
    provider = FakeFactsProvider(
        name_results_by_query={
            "Tomate Sicam": specific,
        }
    )
    grok_service = FakeGrokService(
        {
            "brand": "Sicam",
            "company": None,
            "product_name": "Tomate",
            "confidence": 0.88,
        }
    )

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        grok_service=grok_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="11",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/11.jpg",
            detected_label="food product",
        )
    )

    assert result.source == "lens"
    assert "Tomate" not in provider.name_queries
    assert provider.name_queries[0] == "Tomate Sicam"


def test_pipeline_accepts_approximate_product_name_token_match():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(LensResolution(title="Double concentre tomate Sicam", candidates=["Double concentre tomate Sicam"]))
    ocr_service = FakeOCRService(None)

    specific = ProductFacts(
        name="Double concentrated tomato paste Sicam",
        category="food",
        ingredients=["tomato", "salt"],
        additives=[],
    )
    provider = FakeFactsProvider(
        name_results_by_query={
            "Tomate Sicam": specific,
        }
    )
    grok_service = FakeGrokService(
        {
            "brand": "Sicam",
            "company": None,
            "product_name": "Tomate",
            "confidence": 0.82,
        }
    )

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        grok_service=grok_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="12",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/12.jpg",
            detected_label="food product",
        )
    )

    assert result.source == "lens"
    assert result.name == "Double concentrated tomato paste Sicam"


def test_pipeline_accepts_groq_match_when_facts_name_unknown_but_brand_matches():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(LensResolution(title="Major Chocolate Biscuits", candidates=["Major Chocolate Biscuits"]))
    ocr_service = FakeOCRService(None)

    facts = ProductFacts(
        name="Unknown",
        brand="Major",
        category="food",
        ingredients=["flour", "sugar", "cocoa"],
        additives=[],
    )
    provider = FakeFactsProvider(name_results_by_query={"Chocolate Biscuits Major": facts})
    grok_service = FakeGrokService(
        {
            "brand": "Major",
            "company": None,
            "product_name": "Chocolate Biscuits",
            "confidence": 0.9,
        }
    )

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        grok_service=grok_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="13",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/13.jpg",
            detected_label="food product",
        )
    )

    assert result.source == "lens"
    assert result.brand == "Major"


def test_pipeline_rejects_unknown_name_when_brand_does_not_match():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(LensResolution(title="Major Chocolate Biscuits", candidates=["Major Chocolate Biscuits"]))
    ocr_service = FakeOCRService(None)

    facts = ProductFacts(
        name="Unknown",
        brand="OtherBrand",
        category="food",
        ingredients=["flour", "sugar"],
        additives=[],
    )
    provider = FakeFactsProvider(
        name_results_by_query={
            "Chocolate Biscuits Major": facts,
            "Major Chocolate Biscuits": None,
        }
    )
    grok_service = FakeGrokService(
        {
            "brand": "Major",
            "company": None,
            "product_name": "Chocolate Biscuits",
            "confidence": 0.9,
        }
    )

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        grok_service=grok_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="14",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/14.jpg",
            detected_label="food product",
        )
    )

    assert result.source == "failed"


def test_pipeline_infers_food_category_from_groq_payload_when_label_is_generic():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(LensResolution(title="Chocolate Biscuits MAJOR - Ayshek", candidates=["Chocolate Biscuits MAJOR - Ayshek"]))
    ocr_service = FakeOCRService(None)

    facts = ProductFacts(
        name="Chocolate Biscuits Major",
        brand="Major",
        category="food",
        ingredients=["flour", "sugar", "cocoa"],
        additives=[],
    )
    provider = FakeFactsProvider(name_results_by_query={"Chocolate Biscuits MAJOR": facts})
    grok_service = FakeGrokService(
        {
            "brand": "MAJOR",
            "company": None,
            "product_name": "Chocolate Biscuits",
            "confidence": 0.9,
        }
    )

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        grok_service=grok_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="15",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/15.jpg",
            detected_label="product",
        )
    )

    assert result.source == "lens"
    assert provider.name_queries[0] == "Chocolate Biscuits MAJOR"
    assert provider.name_preferred_categories[0] == "food"


def test_pipeline_skips_raw_lens_lookup_after_confident_structured_groq_miss():
    barcode_service = FakeBarcodeService(None)
    raw_title = "Pasta do Zebow Colgate Max Fresh Clean Mint Colgate"
    lens_service = FakeLensService(LensResolution(title=raw_title, candidates=[raw_title]))
    ocr_service = FakeOCRService(None)

    provider = FakeFactsProvider(name_results_by_query={"Max Fresh Clean Mint Colgate": None})
    grok_service = FakeGrokService(
        {
            "brand": "Colgate",
            "company": None,
            "product_name": "Max Fresh Clean Mint",
            "confidence": 0.9,
        }
    )

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        grok_service=grok_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="16",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/16.jpg",
            detected_label="product",
        )
    )

    assert result.source == "failed"
    assert provider.name_queries == ["Max Fresh Clean Mint Colgate"]
    assert raw_title not in provider.name_queries


def test_pipeline_lens_lookup_uses_cosmetic_category_hint():
    barcode_service = FakeBarcodeService(None)
    lens_service = FakeLensService(LensResolution(title="Shampoo X", candidates=["Shampoo X"]))
    ocr_service = FakeOCRService(None)
    facts = ProductFacts(name="Shampoo X", category="cosmetic", ingredients=["aqua"], additives=[])
    provider = FakeFactsProvider(name_result=facts)
    grok_service = FakeGrokService(None)

    pipeline = PipelineOrchestrator(
        barcode_service,
        lens_service,
        ocr_service,
        provider,
        grok_service=grok_service,
    )
    result = asyncio.run(
        pipeline.analyze_product(
            product_id="9",
            image_path="fake.jpg",
            crop_url="/uploads/crops/x/9.jpg",
            detected_label="cosmetic product",
        )
    )

    assert result.source == "lens"
    assert "Shampoo X" in provider.name_queries
    assert "cosmetic" in provider.name_preferred_categories


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
