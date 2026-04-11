from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.container import ServiceContainer
from app.models.schemas import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="ProductIntelligence_V2")


@router.get("/health/readiness")
async def readiness(request: Request) -> dict[str, dict[str, object]]:
    container: ServiceContainer = request.app.state.container

    barcode_ok, barcode_reason, barcode_backends = container.barcode_service.get_readiness()
    lens_ok, lens_reason = container.lens_service.get_readiness()
    ocr_ok, ocr_reason = container.ocr_service.get_readiness()

    return {
        "barcode": {
            "ok": barcode_ok,
            "reason": barcode_reason,
            "backends": barcode_backends,
        },
        "lens": {
            "ok": lens_ok,
            "reason": lens_reason,
        },
        "ocr": {
            "ok": ocr_ok,
            "reason": ocr_reason,
        },
    }
