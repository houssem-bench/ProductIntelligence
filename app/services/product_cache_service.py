from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import ProductRecord
from app.models.schemas import ProductAnalysis
from app.repositories.product_repository import ProductRepository


logger = logging.getLogger(__name__)


class ProductCacheService:
    def __init__(self, session_factory: Callable[[], Session], repository: ProductRepository) -> None:
        self._session_factory = session_factory
        self._repository = repository

    def build_fingerprint(self, name: str | None, brand: str | None) -> str:
        normalized_name = self._normalize_text(name)
        normalized_brand = self._normalize_text(brand)
        raw_value = f"{normalized_name}|{normalized_brand}"
        return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()

    def get_cached_analysis_by_ean(self, ean: str) -> ProductAnalysis | None:
        with self._session_factory() as db:
            row = self._repository.get_by_ean(db, ean)
            if row is None:
                return None
            return self._analysis_from_row(row)

    def get_cached_analysis_by_fingerprint(self, fingerprint: str) -> ProductAnalysis | None:
        with self._session_factory() as db:
            row = self._repository.get_by_fingerprint(db, fingerprint)
            if row is None:
                return None
            return self._analysis_from_row(row)

    def save_analysis(
        self,
        analysis: ProductAnalysis,
        *,
        ean: str | None,
        extraction_method: str,
        metadata_json: dict[str, Any] | None = None,
    ) -> ProductRecord:
        normalized_name = self._normalize_text(analysis.name)
        normalized_brand = self._normalize_text(analysis.brand)
        if not ean and not normalized_name and not normalized_brand:
            raise ValueError("Cannot persist product without ean or identifiable name/brand")

        full_json = analysis.model_dump(mode="json")
        full_json["ean"] = full_json.get("barcode") or ean

        fingerprint = self.build_fingerprint(analysis.name, analysis.brand)
        analysis_source = "fallback" if analysis.source == "failed" else analysis.source

        with self._session_factory() as db:
            row = self._repository.upsert(
                db,
                name=analysis.name,
                brand=analysis.brand,
                category=analysis.category,
                ingredients=list(analysis.ingredients),
                additives=list(analysis.additives),
                analysis_source=analysis_source,
                confidence=float(analysis.confidence),
                ean=full_json.get("ean"),
                fingerprint=fingerprint,
                full_json=full_json,
                extraction_method=extraction_method,
                metadata_json=metadata_json,
            )
            db.commit()
            return row

    def list_products(
        self,
        *,
        skip: int,
        limit: int,
        ean: str | None = None,
        name: str | None = None,
        created_from: datetime | None = None,
    ) -> tuple[list[ProductRecord], int]:
        with self._session_factory() as db:
            return self._repository.list_products(
                db,
                skip=skip,
                limit=limit,
                ean=ean,
                name=name,
                created_from=created_from,
            )

    def get_product_by_id(self, product_id: int) -> ProductRecord | None:
        with self._session_factory() as db:
            return self._repository.get_by_id(db, product_id)

    def _analysis_from_row(self, row: ProductRecord) -> ProductAnalysis:
        full_json = row.full_json
        if isinstance(full_json, dict):
            try:
                return ProductAnalysis(**full_json)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("[CACHE] Invalid cached full_json id=%s error=%s", row.id, exc)

        return ProductAnalysis(
            product_id="0",
            source=row.analysis_source,
            confidence=row.confidence,
            category=row.category,
            name=row.name,
            brand=row.brand,
            ingredients=list(row.ingredients or []),
            additives=list(row.additives or []),
            barcode=row.ean,
            debug={"cache_fallback": True},
        )

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        if not value:
            return ""
        return " ".join(value.strip().lower().split())
