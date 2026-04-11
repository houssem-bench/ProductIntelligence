from __future__ import annotations

import logging

from app.core.config import Settings
from app.core.http import HttpClient
from app.utils.cache import SimpleTTLCache


logger = logging.getLogger(__name__)


class LensProvider:
    def __init__(self, http: HttpClient, settings: Settings, cache: SimpleTTLCache[dict]) -> None:
        self._http = http
        self._settings = settings
        self._cache = cache

    async def search_titles(self, image_url: str, max_results: int = 5) -> list[str]:
        cache_key = f"lens::{image_url}::{max_results}"
        cached = self._cache.get(cache_key)
        if cached:
            return list(cached.get("titles", []))

        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": self._settings.serpapi_key,
            "type": "products",
        }
        data = await self._http.get_json("https://serpapi.com/search.json", params=params)
        if not data:
            return []

        visual_matches = data.get("visual_matches", [])
        titles = []
        for item in visual_matches[:max_results]:
            title = str(item.get("title") or "").strip()
            if title:
                titles.append(title)

        self._cache.set(cache_key, {"titles": titles})
        logger.info("[LENS] Provider returned matches=%s", len(titles))
        return titles
