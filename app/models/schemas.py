from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class SegmentedProduct(BaseModel):
    product_id: str
    label: str = "product"
    confidence: float
    bbox: BBox
    crop_url: str


class SegmentationResponse(BaseModel):
    request_id: str
    session_id: str
    segmentation_mode: str = "auto"
    expected_products: int | None = None
    total_products: int
    products: list[SegmentedProduct]


class AnalyzeSelectionRequest(BaseModel):
    session_id: str
    product_ids: list[str] = Field(default_factory=list)


class ProductFacts(BaseModel):
    name: str
    brand: str | None = None
    category: str = "unknown"
    ingredients: list[str] = Field(default_factory=list)
    additives: list[str] = Field(default_factory=list)
    source_detail: str | None = None


class ProductAnalysis(BaseModel):
    product_id: str
    source: str
    confidence: float
    category: str
    name: str
    brand: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    additives: list[str] = Field(default_factory=list)
    barcode: str | None = None
    lens_title: str | None = None
    debug: dict[str, Any] = Field(default_factory=dict)


class AnalysisBatchResponse(BaseModel):
    request_id: str
    session_id: str
    analyzed_count: int
    results: list[ProductAnalysis]


class PhoneCaptureSessionResponse(BaseModel):
    token: str
    status: str
    expires_at: datetime
    phone_page_url: str
    upload_url: str
    status_url: str
    consume_url: str
    poll_interval_ms: int


class PhoneCaptureStatusResponse(BaseModel):
    token: str
    status: str
    expires_at: datetime
    has_image: bool
    filename: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str


class ProductCacheItem(BaseModel):
    id: int
    name: str
    brand: str | None = None
    category: str
    analysis_source: str
    confidence: float
    ean: str | None = None
    fingerprint: str
    created_at: datetime
    updated_at: datetime


class ProductCacheListResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: list[ProductCacheItem] = Field(default_factory=list)


class ProductCacheDetailResponse(BaseModel):
    id: int
    full_json: dict[str, Any]
