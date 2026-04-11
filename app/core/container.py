from __future__ import annotations

from app.core.config import Settings
from app.core.http import HttpClient
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.lens_provider import LensProvider
from app.providers.product_facts_provider import ProductFactsProvider
from app.services.barcode_service import BarcodeService
from app.services.lens_service import LensService
from app.services.ocr_service import OCRService
from app.services.segmentation_service import SegmentationService
from app.utils.cache import SimpleTTLCache


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = SimpleTTLCache[dict](ttl_seconds=settings.cache_ttl_seconds)
        self.http = HttpClient(settings)

        self.product_facts_provider = ProductFactsProvider(self.http, settings, self.cache)
        self.lens_provider = LensProvider(self.http, settings, self.cache)

        self.barcode_service = BarcodeService()
        self.lens_service = LensService(self.lens_provider, settings)
        self.ocr_service = OCRService(settings)
        self.segmentation_service = SegmentationService(settings)

        self.pipeline = PipelineOrchestrator(
            barcode_service=self.barcode_service,
            lens_service=self.lens_service,
            ocr_service=self.ocr_service,
            facts_provider=self.product_facts_provider,
        )

    async def close(self) -> None:
        await self.http.close()
