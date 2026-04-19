from __future__ import annotations

import logging

from app.models.schemas import ProductAnalysis
from app.providers.product_facts_provider import ProductFactsProvider
from app.services.barcode_service import BarcodeService
from app.services.lens_service import LensService
from app.services.ocr_service import OCRService
from app.services.product_cache_service import ProductCacheService


logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(
        self,
        barcode_service: BarcodeService,
        lens_service: LensService,
        ocr_service: OCRService,
        facts_provider: ProductFactsProvider,
        product_cache_service: ProductCacheService | None = None,
    ) -> None:
        self._barcode = barcode_service
        self._lens = lens_service
        self._ocr = ocr_service
        self._facts = facts_provider
        self._product_cache = product_cache_service

    async def analyze_product(
        self,
        *,
        product_id: str,
        image_path: str,
        original_image_path: str | None = None,
        crop_url: str,
        detected_label: str,
    ) -> ProductAnalysis:
        debug: dict[str, object] = {}
        barcode = self._barcode.extract(image_path)
        barcode_from = "crop"

        if barcode is None and original_image_path and original_image_path != image_path:
            barcode = self._barcode.extract(original_image_path)
            if barcode:
                barcode_from = "original"

        cached_result = self._lookup_cached_result(
            product_id=product_id,
            barcode_code=barcode.code if barcode else None,
            detected_label=detected_label,
            debug=debug,
        )
        if cached_result:
            return cached_result

        # 1) BARCODE / QR
        logger.info("[BARCODE] stage start product_id=%s", product_id)
        debug["barcode_status"] = "detected" if barcode else "not_detected"
        debug["barcode_source"] = barcode_from if barcode else "none"
        if barcode:
            debug["barcode_format"] = barcode.format_type
            facts = await self._facts.fetch_by_barcode(barcode.code)
            if facts and facts.ingredients:
                logger.info("[BARCODE] success product_id=%s", product_id)
                result = ProductAnalysis(
                    product_id=product_id,
                    source="barcode",
                    confidence=self._score_confidence("barcode", len(facts.ingredients), True),
                    category=facts.category,
                    name=facts.name,
                    brand=facts.brand,
                    ingredients=facts.ingredients,
                    additives=facts.additives,
                    barcode=barcode.code,
                    debug=debug,
                )
                return self._persist_result(
                    result,
                    ean=barcode.code,
                    extraction_method=f"pipeline:barcode:{barcode_from}",
                )
            debug["barcode_lookup"] = "no_ingredients"

        # 2) GOOGLE LENS
        logger.info("[LENS] stage start product_id=%s", product_id)
        if hasattr(self._lens, "get_readiness"):
            ready, reason = self._lens.get_readiness()
            debug["lens_status"] = reason
        else:
            ready = True
        lens = await self._lens.resolve_name(crop_url, detected_label)
        if lens:
            debug["lens_candidates"] = lens.candidates
            if lens.upload_route:
                debug["lens_upload_route"] = lens.upload_route
            if lens.public_image_url:
                debug["lens_public_image_url"] = lens.public_image_url
            facts = await self._facts.fetch_by_name(lens.title)
            if facts and facts.ingredients:
                logger.info("[LENS] success product_id=%s", product_id)
                result = ProductAnalysis(
                    product_id=product_id,
                    source="lens",
                    confidence=self._score_confidence("lens", len(facts.ingredients), True),
                    category=facts.category,
                    name=facts.name,
                    brand=facts.brand,
                    ingredients=facts.ingredients,
                    additives=facts.additives,
                    lens_title=lens.title,
                    debug=debug,
                )
                return self._persist_result(
                    result,
                    ean=barcode.code if barcode else None,
                    extraction_method="pipeline:lens:name_lookup",
                )
            debug["lens_lookup"] = "no_ingredients"
        elif ready:
            debug["lens_status"] = "no_match"

        # 3) OCR (last fallback)
        logger.info("[OCR] stage start product_id=%s", product_id)
        ocr = self._ocr.extract(image_path)
        if ocr:
            debug["ocr_text_chars"] = len(ocr.raw_text)

            if ocr.name:
                facts = await self._facts.fetch_by_name(ocr.name)
                if facts and facts.ingredients:
                    logger.info("[OCR] success via name lookup product_id=%s", product_id)
                    result = ProductAnalysis(
                        product_id=product_id,
                        source="ocr",
                        confidence=self._score_confidence("ocr", len(facts.ingredients), True),
                        category=facts.category,
                        name=facts.name,
                        brand=facts.brand,
                        ingredients=facts.ingredients,
                        additives=facts.additives,
                        debug=debug,
                    )
                    return self._persist_result(
                        result,
                        ean=barcode.code if barcode else None,
                        extraction_method="pipeline:ocr:name_lookup",
                    )

            if ocr.ingredients:
                logger.info("[OCR] success via local extraction product_id=%s", product_id)
                result = ProductAnalysis(
                    product_id=product_id,
                    source="ocr",
                    confidence=self._score_confidence("ocr", len(ocr.ingredients), False),
                    category=ocr.category,
                    name=ocr.name or detected_label,
                    brand=None,
                    ingredients=ocr.ingredients,
                    additives=[],
                    debug=debug,
                )
                return self._persist_result(
                    result,
                    ean=barcode.code if barcode else None,
                    extraction_method="pipeline:ocr:local_extraction",
                )
        else:
            debug["ocr_status"] = "unavailable_or_no_text"

        logger.warning("Pipeline exhausted fallbacks product_id=%s", product_id)
        result = ProductAnalysis(
            product_id=product_id,
            source="failed",
            confidence=0.0,
            category="unknown",
            name=detected_label,
            brand=None,
            ingredients=[],
            additives=[],
            debug=debug,
        )
        return self._persist_result(
            result,
            ean=barcode.code if barcode else None,
            extraction_method="pipeline:fallback_failed",
        )

    def _lookup_cached_result(
        self,
        *,
        product_id: str,
        barcode_code: str | None,
        detected_label: str,
        debug: dict[str, object],
    ) -> ProductAnalysis | None:
        if self._product_cache is None:
            return None

        if barcode_code:
            cached = self._product_cache.get_cached_analysis_by_ean(barcode_code)
            if cached:
                logger.info("[CACHE] hit via EAN=%s product_id=%s", barcode_code, product_id)
                return self._finalize_cached_result(cached, product_id, debug, "ean")

        normalized_label = " ".join((detected_label or "").strip().lower().split())
        if normalized_label and normalized_label not in {"product", "unknown"}:
            fingerprint = self._product_cache.build_fingerprint(detected_label, None)
            debug["lookup_fingerprint"] = fingerprint
            cached = self._product_cache.get_cached_analysis_by_fingerprint(fingerprint)
            if cached:
                logger.info("[CACHE] hit via fingerprint product_id=%s", product_id)
                return self._finalize_cached_result(cached, product_id, debug, "fingerprint")

        return None

    def _finalize_cached_result(
        self,
        cached: ProductAnalysis,
        product_id: str,
        debug: dict[str, object],
        lookup: str,
    ) -> ProductAnalysis:
        cached_debug: dict[str, object] = dict(cached.debug)
        cached_debug.update(debug)
        cached_debug["cache_hit"] = True
        cached_debug["cache_lookup"] = lookup
        return cached.model_copy(update={"product_id": product_id, "debug": cached_debug})

    def _persist_result(
        self,
        result: ProductAnalysis,
        *,
        ean: str | None,
        extraction_method: str,
    ) -> ProductAnalysis:
        if self._product_cache is None:
            return result

        try:
            self._product_cache.save_analysis(
                result,
                ean=ean,
                extraction_method=extraction_method,
                metadata_json={
                    "pipeline_source": result.source,
                    "cached": False,
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("[CACHE] Persist failed product_id=%s error=%s", result.product_id, exc)

        return result

    def _score_confidence(self, source: str, ingredients_count: int, api_backed: bool) -> float:
        base = {
            "barcode": 0.86,
            "lens": 0.73,
            "ocr": 0.58,
        }.get(source, 0.4)

        ingredient_bonus = min(0.12, ingredients_count * 0.01)
        api_bonus = 0.04 if api_backed else 0.0
        return round(min(0.99, base + ingredient_bonus + api_bonus), 3)
