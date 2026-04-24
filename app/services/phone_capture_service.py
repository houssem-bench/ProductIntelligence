from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from threading import Lock

from app.core.config import Settings


@dataclass
class PhoneCaptureSession:
    token: str
    created_at: datetime
    expires_at: datetime
    status: str = "pending"
    filename: str | None = None
    content_type: str | None = None
    image_bytes: bytes | None = None
    updated_at: datetime | None = None
    consumed_at: datetime | None = None


class PhoneCaptureService:
    def __init__(self, settings: Settings) -> None:
        self._ttl_seconds = settings.phone_capture_ttl_seconds
        self._max_upload_bytes = settings.phone_capture_max_upload_mb * 1024 * 1024
        self._poll_interval_ms = settings.phone_capture_poll_interval_ms

        self._lock = Lock()
        self._sessions: dict[str, PhoneCaptureSession] = {}

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _cleanup_locked(self, now: datetime) -> None:
        stale_tokens: list[str] = []

        for token, session in self._sessions.items():
            if session.status in {"pending", "ready"} and session.expires_at <= now:
                session.status = "expired"
                session.image_bytes = None
                session.content_type = None
                session.updated_at = now

            if session.status in {"expired", "consumed"}:
                terminal_at = session.updated_at or session.expires_at
                if now - terminal_at > timedelta(minutes=30):
                    stale_tokens.append(token)

        for token in stale_tokens:
            self._sessions.pop(token, None)

    def _get_session_locked(self, token: str, now: datetime) -> PhoneCaptureSession | None:
        self._cleanup_locked(now)
        session = self._sessions.get(token)
        if session is None:
            return None

        if session.status in {"pending", "ready"} and session.expires_at <= now:
            session.status = "expired"
            session.image_bytes = None
            session.content_type = None
            session.updated_at = now

        return session

    @staticmethod
    def _public_session_copy(session: PhoneCaptureSession) -> PhoneCaptureSession:
        return replace(session, image_bytes=None)

    def create_session(self) -> PhoneCaptureSession:
        with self._lock:
            now = self._utc_now()
            self._cleanup_locked(now)

            token = secrets.token_urlsafe(24)
            while token in self._sessions:
                token = secrets.token_urlsafe(24)

            expires_at = now + timedelta(seconds=self._ttl_seconds)
            session = PhoneCaptureSession(
                token=token,
                created_at=now,
                expires_at=expires_at,
                status="pending",
                updated_at=now,
            )
            self._sessions[token] = session
            return self._public_session_copy(session)

    def get_status(self, token: str) -> PhoneCaptureSession | None:
        with self._lock:
            session = self._get_session_locked(token, self._utc_now())
            if session is None:
                return None
            return self._public_session_copy(session)

    def store_upload(
        self,
        token: str,
        *,
        image_bytes: bytes,
        content_type: str,
        filename: str | None,
    ) -> PhoneCaptureSession:
        if not image_bytes:
            raise ValueError("Uploaded file is empty")

        if len(image_bytes) > self._max_upload_bytes:
            raise ValueError("Uploaded file is too large")

        if not content_type.startswith("image/"):
            raise ValueError("File must be an image")

        with self._lock:
            now = self._utc_now()
            session = self._get_session_locked(token, now)
            if session is None:
                raise KeyError("Session not found")

            if session.status == "expired":
                raise RuntimeError("Session expired")
            if session.status == "consumed":
                raise RuntimeError("Session already consumed")
            if session.status == "ready":
                raise RuntimeError("Image already uploaded")

            session.image_bytes = image_bytes
            session.content_type = content_type
            session.filename = (filename or "phone-capture.jpg").strip() or "phone-capture.jpg"
            session.status = "ready"
            session.updated_at = now
            return self._public_session_copy(session)

    def consume_upload(self, token: str) -> tuple[bytes, str, str, PhoneCaptureSession]:
        with self._lock:
            now = self._utc_now()
            session = self._get_session_locked(token, now)
            if session is None:
                raise KeyError("Session not found")

            if session.status == "expired":
                raise RuntimeError("Session expired")
            if session.status == "consumed":
                raise RuntimeError("Session already consumed")
            if session.status != "ready" or not session.image_bytes:
                raise RuntimeError("Image not uploaded yet")

            payload = session.image_bytes
            content_type = session.content_type or "image/jpeg"
            filename = session.filename or "phone-capture.jpg"

            session.image_bytes = None
            session.status = "consumed"
            session.consumed_at = now
            session.updated_at = now

            return payload, content_type, filename, self._public_session_copy(session)

    def get_poll_interval_ms(self) -> int:
        return self._poll_interval_ms

    def get_ttl_seconds(self) -> int:
        return self._ttl_seconds
