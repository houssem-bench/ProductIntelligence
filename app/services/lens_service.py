from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

from app.core.config import Settings
from app.providers.lens_provider import LensProvider

try:
    from pyngrok import ngrok
except Exception:  # pragma: no cover - optional dependency at runtime
    ngrok = None


logger = logging.getLogger(__name__)


@dataclass
class LensResolution:
    title: str
    candidates: list[str]
    upload_route: str | None = None
    public_image_url: str | None = None


class LensService:
    def __init__(self, provider: LensProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings
        self._ngrok_public_base_url: str | None = None
        self._ngrok_lock = threading.Lock()

    def get_readiness(self) -> tuple[bool, str]:
        if not self._settings.serpapi_key:
            return False, "missing_serpapi_key"
        if self._settings.public_base_url:
            return True, "ready"
        if not self._settings.enable_ngrok:
            return False, "missing_public_base_url"
        if ngrok is None:
            return False, "missing_pyngrok_dependency"
        return True, "ready_via_ngrok"

    async def resolve_name(self, crop_url: str, detected_label: str) -> LensResolution | None:
        ready, reason = self.get_readiness()
        if not ready:
            logger.info("[LENS] Skipped: %s", reason)
            return None

        public_base_url = await self._get_public_base_url()
        if not public_base_url:
            logger.info("[LENS] Skipped: missing_public_base_url")
            return None

        upload_route = self._normalize_upload_route(crop_url)
        public_url = self._build_public_image_url(public_base_url, crop_url)
        if not public_url:
            logger.info("[LENS] Skipped: missing_crop_url")
            return None

        titles = await self._provider.search_titles(public_url, max_results=self._settings.lens_max_matches)

        if not titles:
            logger.info("[LENS] No visual match, fallback to OCR")
            return None

        title = titles[0] if titles else detected_label
        logger.info("[LENS] Best match title=%s", title)
        return LensResolution(
            title=title,
            candidates=titles,
            upload_route=upload_route,
            public_image_url=public_url,
        )

    async def _get_public_base_url(self) -> str | None:
        if self._settings.public_base_url:
            return self._settings.public_base_url

        if self._ngrok_public_base_url:
            return self._ngrok_public_base_url

        if not self._settings.enable_ngrok or ngrok is None:
            return None

        return await asyncio.to_thread(self._start_ngrok_tunnel)

    def _build_public_image_url(self, public_base_url: str, crop_url: str) -> str | None:
        route = (crop_url or "").strip()
        if not route:
            return None

        if route.startswith("http://") or route.startswith("https://"):
            return route

        upload_route = self._normalize_upload_route(route)
        if not upload_route:
            return None

        return f"{public_base_url}{upload_route}"

    def _normalize_upload_route(self, crop_url: str | None) -> str | None:
        if not crop_url:
            return None

        normalized = str(crop_url).strip().replace("\\", "/")
        if not normalized:
            return None

        if normalized.startswith("http://") or normalized.startswith("https://"):
            return None

        marker = "/uploads/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            return f"/uploads/{relative.lstrip('/')}"

        if normalized.startswith("uploads/"):
            return f"/uploads/{normalized[len('uploads/'):].lstrip('/')}"

        if normalized.startswith("/"):
            return normalized

        return f"/{normalized.lstrip('/')}"

    def _start_ngrok_tunnel(self) -> str | None:
        if ngrok is None:
            return None

        with self._ngrok_lock:
            if self._ngrok_public_base_url:
                return self._ngrok_public_base_url

            tunnel_addr = f"{self._settings.host}:{self._settings.port}"
            options: dict[str, object] = {
                "addr": tunnel_addr,
                "proto": "http",
                "bind_tls": True,
            }
            if self._settings.ngrok_domain:
                options["domain"] = self._settings.ngrok_domain

            try:
                if self._settings.ngrok_auth_token:
                    ngrok.set_auth_token(self._settings.ngrok_auth_token)

                tunnel = ngrok.connect(**options)
                self._ngrok_public_base_url = str(tunnel.public_url).rstrip("/")
                logger.info("[LENS] ngrok tunnel ready public_base_url=%s", self._ngrok_public_base_url)
            except Exception as exc:
                logger.error("[LENS] ngrok tunnel failed error=%s", exc)
                return None

            return self._ngrok_public_base_url

    def close(self) -> None:
        if ngrok is None or not self._ngrok_public_base_url:
            return

        try:
            ngrok.disconnect(self._ngrok_public_base_url)
        except Exception as exc:
            logger.warning("[LENS] ngrok disconnect failed error=%s", exc)
        finally:
            self._ngrok_public_base_url = None
