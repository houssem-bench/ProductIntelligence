from __future__ import annotations

import logging
from typing import Any

from app.models.schemas import ProductAnalysis
from app.services.grok_service import GrokService


logger = logging.getLogger(__name__)


class ProductsListService:
    def __init__(self, grok_service: GrokService) -> None:
        self._grok_service = grok_service

    async def build_products_list(self, analyses: list[ProductAnalysis]) -> dict[str, Any]:
        base_payload = {
            "products_list": [self._build_base_product(item) for item in analyses if item.ingredients],
        }

        enriched = await self._grok_service.enrich_products_list(
            products_list_payload=base_payload,
            analyses=analyses,
        )
        if not enriched or not isinstance(enriched.get("products_list"), list):
            return base_payload

        merged_products = []
        enriched_by_id = {
            str(item.get("product_id")): item
            for item in enriched["products_list"]
            if isinstance(item, dict) and item.get("product_id") is not None
        }

        for base_product in base_payload["products_list"]:
            product_id = str(base_product.get("product_id"))
            enriched_product = enriched_by_id.get(product_id)
            if not isinstance(enriched_product, dict):
                merged_products.append(base_product)
                continue
            merged_products.append(self._merge_product(base_product, enriched_product))

        return {"products_list": merged_products}

    def _build_base_product(self, analysis: ProductAnalysis) -> dict[str, Any]:
        product_usage, exposure_type = self._map_usage_and_exposure(analysis.category)

        ingredients = []
        for ingredient in analysis.ingredients:
            name = str(ingredient).strip()
            if name:
                ingredients.append({"name": name})

        product_payload: dict[str, Any] = {
            "product_id": analysis.product_id,
            "ingredient_list": ingredients,
        }

        if product_usage:
            product_payload["product_usage"] = product_usage
        if exposure_type:
            product_payload["exposure_type"] = exposure_type

        return product_payload

    def _merge_product(self, base_product: dict[str, Any], enriched_product: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {
            "product_id": base_product.get("product_id"),
        }

        usage = self._safe_text(enriched_product.get("product_usage")) or self._safe_text(base_product.get("product_usage"))
        exposure = self._safe_text(enriched_product.get("exposure_type")) or self._safe_text(base_product.get("exposure_type"))

        if usage:
            merged["product_usage"] = usage
        if exposure:
            merged["exposure_type"] = exposure

        base_ingredients = base_product.get("ingredient_list")
        enriched_ingredients = enriched_product.get("ingredient_list")

        if not isinstance(base_ingredients, list):
            base_ingredients = []
        if not isinstance(enriched_ingredients, list):
            enriched_ingredients = []

        enriched_by_name: dict[str, dict[str, Any]] = {}
        for item in enriched_ingredients:
            if not isinstance(item, dict):
                continue
            ingredient_name = self._safe_text(item.get("name"))
            if not ingredient_name:
                continue
            enriched_by_name[self._normalize_key(ingredient_name)] = item

        merged_ingredients = []
        for item in base_ingredients:
            if not isinstance(item, dict):
                continue
            ingredient_name = self._safe_text(item.get("name"))
            if not ingredient_name:
                continue

            enriched_item = enriched_by_name.get(self._normalize_key(ingredient_name), {})
            merged_item = {"name": ingredient_name}

            for key in ("code", "dose", "product_prevalence", "additional_info"):
                value = self._safe_text(enriched_item.get(key))
                if value:
                    merged_item[key] = value

            merged_ingredients.append(merged_item)

        merged["ingredient_list"] = merged_ingredients
        return merged

    @staticmethod
    def _map_usage_and_exposure(category: str) -> tuple[str | None, str | None]:
        normalized = (category or "").strip().lower()

        if normalized == "food":
            return "food", "oral"
        if normalized in {"cosmetic", "cosmetics", "beauty"}:
            return "cosmetics", "skin"
        if normalized in {"detergent", "cleaning", "household"}:
            return "detergent", "skin"

        return None, None

    @staticmethod
    def _safe_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalize_key(value: str) -> str:
        return " ".join(value.lower().split())
