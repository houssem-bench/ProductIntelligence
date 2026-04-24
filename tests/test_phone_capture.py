from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from fastapi.testclient import TestClient

from app.main import app
from app.services.phone_capture_service import PhoneCaptureService


def _patch_phone_capture_service(monkeypatch) -> None:
    settings = replace(
        app.state.container.settings,
        phone_capture_ttl_seconds=120,
        phone_capture_max_upload_mb=5,
        phone_capture_poll_interval_ms=700,
    )
    monkeypatch.setattr(app.state.container, "settings", settings)
    monkeypatch.setattr(app.state.container, "phone_capture_service", PhoneCaptureService(settings))


def test_create_phone_capture_session(monkeypatch):
    with TestClient(app) as client:
        _patch_phone_capture_service(monkeypatch)
        response = client.post("/phone-capture/session")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["token"]
    assert payload["phone_page_url"].endswith(payload["token"])
    assert payload["poll_interval_ms"] == 700


def test_phone_capture_rejects_non_image_upload(monkeypatch):
    with TestClient(app) as client:
        _patch_phone_capture_service(monkeypatch)
        create_response = client.post("/phone-capture/session")
        token = create_response.json()["token"]

        upload_response = client.post(
            f"/phone-capture/session/{token}/upload",
            files={"image": ("bad.txt", b"plain-text", "text/plain")},
        )

    assert upload_response.status_code == 400
    assert "image" in upload_response.json()["detail"].lower()


def test_phone_capture_status_and_consume_once(monkeypatch):
    image_bytes = b"fake-jpeg-bytes"

    with TestClient(app) as client:
        _patch_phone_capture_service(monkeypatch)
        create_response = client.post("/phone-capture/session")
        token = create_response.json()["token"]

        upload_response = client.post(
            f"/phone-capture/session/{token}/upload",
            files={"image": ("photo.jpg", image_bytes, "image/jpeg")},
        )
        assert upload_response.status_code == 200
        assert upload_response.json()["status"] == "ready"

        status_response = client.get(f"/phone-capture/session/{token}/status")
        assert status_response.status_code == 200
        status_payload = status_response.json()
        assert status_payload["status"] == "ready"
        assert status_payload["has_image"] is True

        consume_response = client.get(f"/phone-capture/session/{token}/consume")
        assert consume_response.status_code == 200
        assert consume_response.content == image_bytes

        consumed_status_response = client.get(f"/phone-capture/session/{token}/status")
        assert consumed_status_response.status_code == 200
        consumed_payload = consumed_status_response.json()
        assert consumed_payload["status"] == "consumed"
        assert consumed_payload["has_image"] is False

        consume_again_response = client.get(f"/phone-capture/session/{token}/consume")

    assert consume_again_response.status_code == 410


def test_phone_capture_expired_session(monkeypatch):
    with TestClient(app) as client:
        _patch_phone_capture_service(monkeypatch)
        create_response = client.post("/phone-capture/session")
        token = create_response.json()["token"]

        service = app.state.container.phone_capture_service
        with service._lock:  # pylint: disable=protected-access
            session = service._sessions[token]  # pylint: disable=protected-access
            session.expires_at = session.created_at - timedelta(seconds=1)

        status_response = client.get(f"/phone-capture/session/{token}/status")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "expired"

        upload_response = client.post(
            f"/phone-capture/session/{token}/upload",
            files={"image": ("photo.jpg", b"img", "image/jpeg")},
        )

    assert upload_response.status_code == 410
