from __future__ import annotations

import logging
import re
from typing import Any

from app.core.config import Settings
from app.core.http import HttpClient
from app.models.schemas import ProductFacts
from app.utils.cache import SimpleTTLCache


logger = logging.getLogger(__name__)


OFF_BARCODE_URL = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
OBF_BARCODE_URL = "https://world.openbeautyfacts.org/api/v0/product/{barcode}.json"
OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
OBF_SEARCH_URL = "https://world.openbeautyfacts.org/cgi/search.pl"


class ProductFactsProvider:
    def __init__(self, http: HttpClient, settings: Settings, cache: SimpleTTLCache[dict]) -> None:
        self._http = http
        self._settings = settings
        self._cache = cache

    async def fetch_by_barcode(self, barcode: str) -> ProductFacts | None:
        cache_key = f"barcode::{barcode}"
        cached = self._cache.get(cache_key)
        if cached:
            return ProductFacts(**cached)

        headers = {"User-Agent": self._settings.off_user_agent}

        off_data = await self._http.get_json(OFF_BARCODE_URL.format(barcode=barcode), headers=headers)
        if off_data and off_data.get("status") == 1:
            parsed = self._parse_off(off_data.get("product", {}))
            if parsed and parsed.ingredients:
                self._cache.set(cache_key, parsed.model_dump())
                return parsed

        obf_data = await self._http.get_json(OBF_BARCODE_URL.format(barcode=barcode), headers=headers)
        if obf_data and obf_data.get("status") == 1:
            parsed = self._parse_obf(obf_data.get("product", {}))
            if parsed and parsed.ingredients:
                self._cache.set(cache_key, parsed.model_dump())
                return parsed

        return None

    async def fetch_by_name(self, name: str) -> ProductFacts | None:
        candidates = self._prepare_name_candidates(name)
        if not candidates:
            return None

        for candidate in candidates:
            cache_key = f"name::{candidate.lower()}"
            cached = self._cache.get(cache_key)
            if cached:
                return ProductFacts(**cached)

        headers = {"User-Agent": self._settings.off_user_agent}

        for candidate in candidates:
            off_params = {
                "search_terms": candidate,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": 3,
            }
            off_data = await self._http.get_json(OFF_SEARCH_URL, params=off_params, headers=headers)
            if off_data and off_data.get("products"):
                best = self._pick_best_product(off_data.get("products", []))
                parsed = self._parse_off(best)
                if parsed and parsed.ingredients:
                    self._cache_name_result(candidates, parsed)
                    return parsed

            obf_params = {
                "search_terms": candidate,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": 3,
            }
            obf_data = await self._http.get_json(OBF_SEARCH_URL, params=obf_params, headers=headers)
            if obf_data and obf_data.get("products"):
                best = self._pick_best_product(obf_data.get("products", []))
                parsed = self._parse_obf(best)
                if parsed and parsed.ingredients:
                    self._cache_name_result(candidates, parsed)
                    return parsed

        return None

    def _cache_name_result(self, candidates: list[str], parsed: ProductFacts) -> None:
        payload = parsed.model_dump()
        for candidate in candidates:
            self._cache.set(f"name::{candidate.lower()}", payload)

    @staticmethod
    def _prepare_name_candidates(name: str) -> list[str]:
        base = " ".join(name.strip().split())
        if not base:
            return []

        # Keep the original title first, then progressively cleaner forms.
        variants = [base]
        cleaned = base

        cleanup_patterns = [
            r"[\(\[\{].*?[\)\]\}]",  # remove bracketed metadata
            r"\b\d+\s?[xX]\s?\d+(?:[\.,]\d+)?\s?(?:ml|l|cl|dl|g|kg|mg|oz)\b",
            r"\b\d+(?:[\.,]\d+)?\s?(?:ml|l|cl|dl|g|kg|mg|oz)\b",
            r"\b(?:pack|lot|set)\s+of\s+\d+\b",
            r"\b\d+\s?(?:pcs?|pieces?)\b",
        ]
        for pattern in cleanup_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"[\|/_-]+", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        if cleaned and cleaned.lower() != base.lower():
            variants.append(cleaned)

        tokens = cleaned.split()
        if len(tokens) > 6:
            variants.append(" ".join(tokens[:6]))

        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            key = variant.casefold()
            if not variant or key in seen:
                continue
            seen.add(key)
            deduped.append(variant)

        return deduped

    def _pick_best_product(self, products: list[dict[str, Any]]) -> dict[str, Any]:
        if not products:
            return {}
        sorted_products = sorted(
            products,
            key=lambda p: len(p.get("ingredients", [])) if isinstance(p.get("ingredients"), list) else 0,
            reverse=True,
        )
        return sorted_products[0]

    def _parse_off(self, product: dict[str, Any]) -> ProductFacts | None:
        ingredients: list[str] = []
        additives: list[str] = []

        for item in product.get("ingredients", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            ingredient_id = str(item.get("id") or "")
            if text and not ingredient_id.startswith("fr:"):
                ingredients.append(text)
            if ingredient_id.startswith("en:e"):
                additives.append(ingredient_id.replace("en:", "").upper())

        if not ingredients:
            fallback_text = str(product.get("ingredients_text") or "").strip()
            if fallback_text:
                ingredients = [part.strip() for part in fallback_text.split(",") if part.strip()]

        if not ingredients:
            return None

        return ProductFacts(
            name=str(product.get("product_name") or "Unknown"),
            brand=str(product.get("brands") or "") or None,
            category="food",
            ingredients=list(dict.fromkeys(ingredients)),
            additives=list(dict.fromkeys(additives)),
            source_detail="openfoodfacts",
        )

    def _parse_obf(self, product: dict[str, Any]) -> ProductFacts | None:
        ingredients: list[str] = []
        for item in product.get("ingredients", []):
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    ingredients.append(text)

        if not ingredients:
            fallback_text = str(product.get("ingredients_text") or "").strip()
            if fallback_text:
                ingredients = [part.strip() for part in fallback_text.split(",") if part.strip()]

        if not ingredients:
            return None

        return ProductFacts(
            name=str(product.get("product_name") or "Unknown"),
            brand=str(product.get("brands") or "") or None,
            category="cosmetic",
            ingredients=list(dict.fromkeys(ingredients)),
            additives=[],
            source_detail="openbeautyfacts",
        )
