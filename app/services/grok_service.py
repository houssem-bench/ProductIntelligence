from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import Settings
from app.core.http import HttpClient
from app.models.schemas import ProductAnalysis


logger = logging.getLogger(__name__)


class GrokService:
    def __init__(self, http: HttpClient, settings: Settings) -> None:
        self._http = http
        self._settings = settings

    def get_readiness(self) -> tuple[bool, str]:
        if not self._settings.grok_api_key:
            return False, "missing_grok_api_key"
        return True, "ready"

    async def enrich_products_list(
        self,
        *,
        products_list_payload: dict[str, Any],
        analyses: list[ProductAnalysis],
    ) -> dict[str, Any] | None:
        ready, reason = self.get_readiness()
        if not ready:
            logger.info("[GROK] Skipped: %s", reason)
            return None

        url = f"{self._settings.grok_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.grok_api_key}",
            "Content-Type": "application/json",
        }

        analysis_context = [
            {
                "product_id": item.product_id,
                "name": item.name,
                "brand": item.brand,
                "category": item.category,
                "source": item.source,
                "ingredients": item.ingredients,
                "additives": item.additives,
                "barcode": item.barcode,
                "lens_title": item.lens_title,
            }
            for item in analyses
        ]

        payload = {
            "model": self._settings.grok_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON formatter for product ingredient risk summaries. "
                        "Return JSON only, no markdown, no explanations. "
                        "Never invent uncertain data. If data is missing, omit the field."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Using ONLY the provided facts, enrich this JSON structure.\n"
                        "Required top-level key: products_list (array).\n"
                        "Each product may include: product_id, product_usage, ingredient_list, exposure_type.\n"
                        "ingredient_list items may include: name, code, dose, product_prevalence, additional_info.\n"
                        "Do not create unknown values. Omit empty fields.\n"
                        "Keep ingredient names aligned with provided ingredient names whenever possible.\n\n"
                        f"BASE_JSON:\n{json.dumps(products_list_payload)}\n\n"
                        f"ANALYSIS_CONTEXT:\n{json.dumps(analysis_context)}"
                    ),
                },
            ],
        }

        data = await self._http.post_json(url, json_body=payload, headers=headers)
        if not data:
            return None

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            return None

        return self._extract_json_object(content)

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any] | None:
        text = content.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or start >= end:
                return None
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                logger.warning("[GROK] Invalid JSON response")
                return None
