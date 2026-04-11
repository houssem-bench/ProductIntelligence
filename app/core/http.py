from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.core.config import Settings


logger = logging.getLogger(__name__)


class HttpClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    async def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        retries = self._settings.max_retries
        backoff = self._settings.retry_backoff_seconds

        for attempt in range(retries + 1):
            try:
                response = await self._client.get(url, params=params, headers=headers)
                response.raise_for_status()
                if "application/json" not in response.headers.get("Content-Type", "").lower():
                    return None
                return response.json()
            except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
                if attempt >= retries:
                    logger.error("HTTP GET failed url=%s error=%s", url, exc)
                    return None
                sleep_seconds = backoff * (2 ** attempt)
                await asyncio.sleep(sleep_seconds)

        return None
