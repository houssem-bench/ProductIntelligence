from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.core.container import ServiceContainer
from app.models.schemas import AnalysisBatchResponse, AnalyzeSelectionRequest, ProductAnalysis, SegmentationResponse


logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


@router.post("/analyze", response_model=SegmentationResponse)
async def segment_products(
    request: Request,
    image: UploadFile = File(...),
    segmentation_mode: str = Form("auto"),
    expected_products: int | None = Form(None),
    container: ServiceContainer = Depends(get_container),
) -> SegmentationResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    normalized_mode = segmentation_mode.strip().lower()
    if normalized_mode not in {"auto", "single", "multi"}:
        raise HTTPException(status_code=400, detail="segmentation_mode must be one of: auto, single, multi")

    if expected_products is not None and expected_products < 1:
        raise HTTPException(status_code=400, detail="expected_products must be >= 1")

    try:
        session_id, products = await container.segmentation_service.segment_upload(
            image,
            segmentation_mode=normalized_mode,
            expected_products=expected_products,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request_id = getattr(request.state, "request_id", "-")
    return SegmentationResponse(
        request_id=request_id,
        session_id=session_id,
        segmentation_mode=normalized_mode,
        expected_products=expected_products,
        total_products=len(products),
        products=products,
    )


@router.post("/analyze/selected", response_model=AnalysisBatchResponse)
async def analyze_selected_products(
    payload: AnalyzeSelectionRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> AnalysisBatchResponse:
    session_products = container.segmentation_service.get_session_products(payload.session_id)
    if session_products is None:
        raise HTTPException(status_code=404, detail="Session not found")

    selected_ids = payload.product_ids
    if not selected_ids:
        raise HTTPException(status_code=400, detail="No product selected")

    invalid_ids = [pid for pid in selected_ids if pid not in session_products]
    if invalid_ids:
        raise HTTPException(status_code=400, detail=f"Invalid product IDs: {invalid_ids}")

    semaphore = asyncio.Semaphore(container.settings.max_parallel_analyses)

    async def run_one(product_id: str) -> ProductAnalysis:
        async with semaphore:
            product = session_products[product_id]
            logger.info("Start analysis session=%s product_id=%s", payload.session_id, product_id)
            try:
                return await container.pipeline.analyze_product(
                    product_id=product.product_id,
                    image_path=product.crop_path,
                    original_image_path=product.source_image_path,
                    crop_url=product.crop_url,
                    detected_label=product.label,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Product analysis failed session=%s product_id=%s", payload.session_id, product_id)
                return ProductAnalysis(
                    product_id=product.product_id,
                    source="failed",
                    confidence=0.0,
                    category="unknown",
                    name=product.label,
                    brand=None,
                    ingredients=[],
                    additives=[],
                    debug={"error": str(exc)},
                )

    results = await asyncio.gather(*(run_one(pid) for pid in selected_ids))
    request_id = getattr(request.state, "request_id", "-")

    return AnalysisBatchResponse(
        request_id=request_id,
        session_id=payload.session_id,
        analyzed_count=len(results),
        results=results,
    )
