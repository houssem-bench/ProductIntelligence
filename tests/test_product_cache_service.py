from __future__ import annotations

from pathlib import Path

from app.db.session import Database
from app.models.schemas import ProductAnalysis
from app.repositories.product_repository import ProductRepository
from app.services.product_cache_service import ProductCacheService


def _build_service(db_path: Path) -> ProductCacheService:
    database = Database(f"sqlite:///{db_path}")
    database.create_tables()
    return ProductCacheService(database.session_factory(), ProductRepository())


def test_search_by_ean_returns_cached_full_json(tmp_path):
    service = _build_service(tmp_path / "ean_cache.db")
    analysis = ProductAnalysis(
        product_id="1",
        source="barcode",
        confidence=0.94,
        category="food",
        name="Nutella",
        brand="Ferrero",
        ingredients=["sugar", "palm oil"],
        additives=[],
        barcode="3017620422003",
        debug={"stage": "barcode"},
    )

    service.save_analysis(
        analysis,
        ean="3017620422003",
        extraction_method="pipeline:barcode:crop",
    )

    cached = service.get_cached_analysis_by_ean("3017620422003")
    assert cached is not None
    assert cached.name == "Nutella"
    assert cached.barcode == "3017620422003"


def test_search_by_fingerprint_returns_cached_product(tmp_path):
    service = _build_service(tmp_path / "fingerprint_cache.db")
    analysis = ProductAnalysis(
        product_id="2",
        source="ocr",
        confidence=0.72,
        category="cosmetic",
        name="Shampoo X",
        brand="BrandCo",
        ingredients=["aqua"],
        additives=[],
        debug={"stage": "ocr"},
    )

    service.save_analysis(
        analysis,
        ean=None,
        extraction_method="pipeline:ocr:name_lookup",
    )

    fingerprint = service.build_fingerprint("Shampoo X", "BrandCo")
    cached = service.get_cached_analysis_by_fingerprint(fingerprint)
    assert cached is not None
    assert cached.name == "Shampoo X"
    assert cached.source == "ocr"


def test_full_json_is_persisted(tmp_path):
    service = _build_service(tmp_path / "json_cache.db")
    analysis = ProductAnalysis(
        product_id="3",
        source="lens",
        confidence=0.83,
        category="food",
        name="Chocolate Bar",
        brand="SweetCo",
        ingredients=["cocoa", "sugar"],
        additives=[],
        lens_title="Chocolate Bar SweetCo",
        debug={"stage": "lens"},
    )

    saved = service.save_analysis(
        analysis,
        ean=None,
        extraction_method="pipeline:lens:name_lookup",
    )

    row = service.get_product_by_id(saved.id)
    assert row is not None
    assert isinstance(row.full_json, dict)
    assert row.full_json["name"] == "Chocolate Bar"
    assert row.full_json["source"] == "lens"
    assert row.full_json["ingredients"] == ["cocoa", "sugar"]


def test_new_product_is_saved_and_listed(tmp_path):
    service = _build_service(tmp_path / "list_cache.db")
    analysis = ProductAnalysis(
        product_id="4",
        source="failed",
        confidence=0.0,
        category="unknown",
        name="Unknown Pack",
        brand=None,
        ingredients=[],
        additives=[],
        debug={"stage": "fallback"},
    )

    service.save_analysis(
        analysis,
        ean=None,
        extraction_method="pipeline:fallback_failed",
    )

    rows, total = service.list_products(skip=0, limit=10)
    assert total == 1
    assert len(rows) == 1
    assert rows[0].name == "Unknown Pack"
    assert rows[0].analysis_source == "fallback"
