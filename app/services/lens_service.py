from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.providers.lens_provider import LensProvider


logger = logging.getLogger(__name__)


@dataclass
class LensResolution:
    title: str
    candidates: list[str]


class LensService:
    def __init__(self, provider: LensProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    def get_readiness(self) -> tuple[bool, str]:
        if not self._settings.serpapi_key:
            return False, "missing_serpapi_key"
        if not self._settings.public_base_url:
            return False, "missing_public_base_url"
        return True, "ready"

    async def resolve_name(self, crop_url: str, detected_label: str) -> LensResolution | None:
        ready, reason = self.get_readiness()
        if not ready:
            logger.info("[LENS] Skipped: %s", reason)
            return None

        public_url = f"{self._settings.public_base_url}{crop_url}"
        titles = await self._provider.search_titles(public_url, max_results=5)

        if not titles:
            logger.info("[LENS] No visual match, fallback to OCR")
            return None

        title = titles[0] if titles else detected_label
        logger.info("[LENS] Best match title=%s", title)
        return LensResolution(title=title, candidates=titles)
