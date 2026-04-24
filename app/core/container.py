from __future__ import annotations

from app.core.config import Settings
from app.core.http import HttpClient
from app.db.session import Database
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.lens_provider import LensProvider
from app.providers.product_facts_provider import ProductFactsProvider
from app.repositories.product_repository import ProductRepository
from app.services.barcode_service import BarcodeService
from app.services.grok_service import GrokService
from app.services.lens_service import LensService
from app.services.ocr_service import OCRService
from app.services.phone_capture_service import PhoneCaptureService
from app.services.product_cache_service import ProductCacheService
from app.services.products_list_service import ProductsListService
from app.services.segmentation_service import SegmentationService
from app.utils.cache import SimpleTTLCache


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = SimpleTTLCache[dict](ttl_seconds=settings.cache_ttl_seconds)
        self.http = HttpClient(settings)
        self.database = Database(settings.database_url)
        self.database.create_tables()

        self.product_facts_provider = ProductFactsProvider(self.http, settings, self.cache)
        self.lens_provider = LensProvider(self.http, settings, self.cache)
        self.product_repository = ProductRepository()

        self.barcode_service = BarcodeService(enabled=settings.enable_barcode)
        self.lens_service = LensService(self.lens_provider, settings)
        self.ocr_service = OCRService(settings)
        self.segmentation_service = SegmentationService(settings)
        self.phone_capture_service = PhoneCaptureService(settings)
        self.grok_service = GrokService(self.http, settings)
        self.products_list_service = ProductsListService(self.grok_service)
        self.product_cache_service = ProductCacheService(
            self.database.session_factory(),
            self.product_repository,
        )

        self.pipeline = PipelineOrchestrator(
            barcode_service=self.barcode_service,
            lens_service=self.lens_service,
            ocr_service=self.ocr_service,
            facts_provider=self.product_facts_provider,
            product_cache_service=self.product_cache_service,
            grok_service=self.grok_service,
        )

    async def close(self) -> None:
        self.lens_service.close()
        self.database.dispose()
        await self.http.close()
