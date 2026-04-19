from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Select, desc, func, select
from sqlalchemy.orm import Session

from app.db.models import ProductRecord


class ProductRepository:
    def get_by_id(self, db: Session, product_id: int) -> ProductRecord | None:
        return db.get(ProductRecord, product_id)

    def get_by_ean(self, db: Session, ean: str) -> ProductRecord | None:
        stmt = select(ProductRecord).where(ProductRecord.ean == ean)
        return db.scalar(stmt)

    def get_by_fingerprint(self, db: Session, fingerprint: str) -> ProductRecord | None:
        stmt = select(ProductRecord).where(ProductRecord.fingerprint == fingerprint)
        return db.scalar(stmt)

    def list_products(
        self,
        db: Session,
        *,
        skip: int,
        limit: int,
        ean: str | None = None,
        name: str | None = None,
        created_from: datetime | None = None,
    ) -> tuple[list[ProductRecord], int]:
        stmt: Select[tuple[ProductRecord]] = select(ProductRecord)

        if ean:
            stmt = stmt.where(ProductRecord.ean == ean)
        if name:
            stmt = stmt.where(ProductRecord.name.ilike(f"%{name}%"))
        if created_from:
            stmt = stmt.where(ProductRecord.created_at >= created_from)

        ordered_stmt = stmt.order_by(desc(ProductRecord.created_at))
        total_stmt = select(func.count()).select_from(stmt.subquery())
        total = int(db.scalar(total_stmt) or 0)
        items = db.scalars(ordered_stmt.offset(skip).limit(limit)).all()
        return items, total

    def upsert(
        self,
        db: Session,
        *,
        name: str,
        brand: str | None,
        category: str,
        ingredients: list[str],
        additives: list[str],
        analysis_source: str,
        confidence: float,
        ean: str | None,
        fingerprint: str,
        full_json: dict[str, Any],
        extraction_method: str,
        metadata_json: dict[str, Any] | None = None,
    ) -> ProductRecord:
        existing = None
        if ean:
            existing = self.get_by_ean(db, ean)
        if existing is None:
            existing = self.get_by_fingerprint(db, fingerprint)

        if existing is None:
            existing = ProductRecord(
                name=name,
                brand=brand,
                category=category,
                ingredients=ingredients,
                additives=additives,
                analysis_source=analysis_source,
                confidence=confidence,
                ean=ean,
                fingerprint=fingerprint,
                full_json=full_json,
                extraction_method=extraction_method,
                metadata_json=metadata_json,
            )
            db.add(existing)
            db.flush()
            return existing

        existing.name = name
        existing.brand = brand
        existing.category = category
        existing.ingredients = ingredients
        existing.additives = additives
        existing.analysis_source = analysis_source
        existing.confidence = confidence
        existing.ean = ean or existing.ean
        existing.fingerprint = fingerprint
        existing.full_json = full_json
        existing.extraction_method = extraction_method
        existing.metadata_json = metadata_json
        db.flush()
        return existing
