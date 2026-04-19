from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.container import ServiceContainer
from app.models.schemas import ProductCacheDetailResponse, ProductCacheItem, ProductCacheListResponse


router = APIRouter(tags=["products"])


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


@router.get("/products", response_model=ProductCacheListResponse)
async def list_products(
    request: Request,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    ean: str | None = Query(default=None),
    name: str | None = Query(default=None),
    created_from: datetime | None = Query(default=None),
    container: ServiceContainer = Depends(get_container),
) -> ProductCacheListResponse:
    rows, total = container.product_cache_service.list_products(
        skip=skip,
        limit=limit,
        ean=ean,
        name=name,
        created_from=created_from,
    )

    items = [
        ProductCacheItem(
            id=row.id,
            name=row.name,
            brand=row.brand,
            category=row.category,
            analysis_source=row.analysis_source,
            confidence=row.confidence,
            ean=row.ean,
            fingerprint=row.fingerprint,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]

    _ = request
    return ProductCacheListResponse(total=total, skip=skip, limit=limit, items=items)


@router.get("/products/{product_id}", response_model=ProductCacheDetailResponse)
async def get_product_detail(
    product_id: int,
    container: ServiceContainer = Depends(get_container),
) -> ProductCacheDetailResponse:
    row = container.product_cache_service.get_product_by_id(product_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Product not found")

    full_json = row.full_json if isinstance(row.full_json, dict) else {}
    return ProductCacheDetailResponse(id=row.id, full_json=full_json)
