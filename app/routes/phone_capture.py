from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from app.core.container import ServiceContainer
from app.models.schemas import PhoneCaptureSessionResponse, PhoneCaptureStatusResponse


router = APIRouter(tags=["phone-capture"])


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def _absolute_url(request: Request, path: str) -> str:
    return f"{str(request.base_url).rstrip('/')}{path}"


def _phone_base_url(request: Request, container: ServiceContainer) -> str:
    requested_base = str(request.base_url).rstrip("/")
    host = (request.url.hostname or "").lower()
    configured_base = container.settings.public_base_url.strip().rstrip("/")

    if host in {"127.0.0.1", "localhost"} and configured_base.startswith("http"):
        return configured_base

    return requested_base


def _session_response(
    *,
    request: Request,
    phone_base_url: str,
    token: str,
    status: str,
    expires_at,
    poll_interval_ms: int,
) -> PhoneCaptureSessionResponse:
    return PhoneCaptureSessionResponse(
        token=token,
        status=status,
        expires_at=expires_at,
        phone_page_url=f"{phone_base_url}/phone-capture/{token}",
        upload_url=_absolute_url(request, f"/phone-capture/session/{token}/upload"),
        status_url=_absolute_url(request, f"/phone-capture/session/{token}/status"),
        consume_url=_absolute_url(request, f"/phone-capture/session/{token}/consume"),
        poll_interval_ms=poll_interval_ms,
    )


@router.post("/phone-capture/session", response_model=PhoneCaptureSessionResponse)
async def create_phone_capture_session(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> PhoneCaptureSessionResponse:
    session = container.phone_capture_service.create_session()
    return _session_response(
        request=request,
      phone_base_url=_phone_base_url(request, container),
        token=session.token,
        status=session.status,
        expires_at=session.expires_at,
        poll_interval_ms=container.phone_capture_service.get_poll_interval_ms(),
    )


@router.get("/phone-capture/session/{token}/status", response_model=PhoneCaptureStatusResponse)
async def get_phone_capture_status(
    token: str,
    container: ServiceContainer = Depends(get_container),
) -> PhoneCaptureStatusResponse:
    session = container.phone_capture_service.get_status(token)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return PhoneCaptureStatusResponse(
        token=session.token,
        status=session.status,
        expires_at=session.expires_at,
        has_image=session.status == "ready",
        filename=session.filename,
    )


@router.post("/phone-capture/session/{token}/upload", response_model=PhoneCaptureStatusResponse)
async def upload_phone_capture(
    token: str,
    image: UploadFile = File(...),
    container: ServiceContainer = Depends(get_container),
) -> PhoneCaptureStatusResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await image.read()

    try:
        session = container.phone_capture_service.store_upload(
            token,
            image_bytes=image_bytes,
            content_type=image.content_type,
            filename=image.filename,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        if "expired" in detail.lower() or "consumed" in detail.lower():
            raise HTTPException(status_code=410, detail=detail) from exc
        raise HTTPException(status_code=409, detail=detail) from exc

    return PhoneCaptureStatusResponse(
        token=session.token,
        status=session.status,
        expires_at=session.expires_at,
        has_image=session.status == "ready",
        filename=session.filename,
    )


@router.get("/phone-capture/session/{token}/consume")
async def consume_phone_capture(
    token: str,
    container: ServiceContainer = Depends(get_container),
) -> Response:
    try:
        payload, content_type, filename, _session = container.phone_capture_service.consume_upload(token)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except RuntimeError as exc:
        detail = str(exc)
        if "expired" in detail.lower() or "consumed" in detail.lower():
            raise HTTPException(status_code=410, detail=detail) from exc
        raise HTTPException(status_code=409, detail=detail) from exc

    safe_filename = (filename or "phone-capture.jpg").replace('"', "")
    return Response(
        content=payload,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{safe_filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/phone-capture/{token}", include_in_schema=False, response_class=HTMLResponse)
async def phone_capture_page(
    token: str,
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> HTMLResponse:
    session = container.phone_capture_service.get_status(token)

    if session is None:
        initial_state = "invalid"
        expires_at = ""
    else:
        initial_state = session.status
        expires_at = session.expires_at.isoformat()

    upload_url = _absolute_url(request, f"/phone-capture/session/{token}/upload")

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Phone Capture</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --line: #d8e1ef;
      --text: #12223a;
      --muted: #4f6078;
      --primary: #14532d;
      --primary-soft: #dcfce7;
      --danger: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--text);
      background: linear-gradient(165deg, #f1f5ff 0%, var(--bg) 50%, #eefaf2 100%);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 20px;
    }}
    .card {{
      width: 100%;
      max-width: 520px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
      padding: 18px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 1.4rem; }}
    p {{ margin: 0 0 12px; color: var(--muted); }}
    .row {{ display: grid; gap: 10px; }}
    input[type=file] {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #f8fbff;
    }}
    button {{
      border: 0;
      border-radius: 10px;
      padding: 11px 14px;
      font-weight: 700;
      color: #fff;
      background: linear-gradient(120deg, var(--primary), #166534);
      cursor: pointer;
    }}
    button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .status {{
      border-radius: 10px;
      padding: 10px;
      font-size: 0.95rem;
      background: var(--primary-soft);
      color: #166534;
    }}
    .status.error {{ background: #fee2e2; color: var(--danger); }}
    .hidden {{ display: none; }}
    .small {{ font-size: 0.85rem; color: var(--muted); }}
  </style>
</head>
<body>
  <main class=\"card\">
    <h1>Capture from Phone</h1>
    <p>Take one photo and send it to your desktop session.</p>
    <div id=\"status\" class=\"status\">Preparing capture...</div>
    <div id=\"capturePanel\" class=\"row\">
      <input id=\"photoInput\" type=\"file\" accept=\"image/*\" capture=\"environment\" />
      <button id=\"sendBtn\" type=\"button\">Send Photo</button>
      <div class=\"small\" id=\"meta\"></div>
    </div>
  </main>

  <script>
    const initialState = {json.dumps(initial_state)};
    const expiresAt = {json.dumps(expires_at)};
    const uploadUrl = {json.dumps(upload_url)};

    const statusEl = document.getElementById('status');
    const panelEl = document.getElementById('capturePanel');
    const photoInput = document.getElementById('photoInput');
    const sendBtn = document.getElementById('sendBtn');
    const metaEl = document.getElementById('meta');

    function setStatus(text, isError = false) {{
      statusEl.textContent = text;
      statusEl.classList.toggle('error', isError);
    }}

    function lockPanel() {{
      photoInput.disabled = true;
      sendBtn.disabled = true;
    }}

    function boot() {{
      if (initialState === 'invalid') {{
        setStatus('This capture link is invalid.', true);
        lockPanel();
        return;
      }}

      if (initialState === 'expired') {{
        setStatus('This capture session has expired. Please start a new one from desktop.', true);
        lockPanel();
        return;
      }}

      if (initialState === 'consumed') {{
        setStatus('This capture session is already completed.', true);
        lockPanel();
        return;
      }}

      if (initialState === 'ready') {{
        setStatus('A photo was already sent for this session.', true);
        lockPanel();
        return;
      }}

      if (expiresAt) {{
        const localText = new Date(expiresAt).toLocaleString();
        metaEl.textContent = `Session expires at ${{localText}}`;
      }}

      setStatus('Pick or capture one image, then tap Send Photo.');
    }}

    async function parseJsonSafe(response) {{
      const body = await response.text();
      try {{
        return JSON.parse(body);
      }} catch {{
        return {{ detail: body || `HTTP ${{response.status}}` }};
      }}
    }}

    sendBtn.addEventListener('click', async () => {{
      const file = photoInput.files && photoInput.files[0] ? photoInput.files[0] : null;
      if (!file) {{
        setStatus('Please capture an image first.', true);
        return;
      }}

      if (!file.type.startsWith('image/')) {{
        setStatus('Only image files are accepted.', true);
        return;
      }}

      sendBtn.disabled = true;
      setStatus('Uploading photo...');

      const formData = new FormData();
      formData.append('image', file);

      try {{
        const response = await fetch(uploadUrl, {{ method: 'POST', body: formData }});
        const data = await parseJsonSafe(response);

        if (!response.ok) {{
          throw new Error(data.detail || 'Upload failed');
        }}

        setStatus('Photo sent. You can return to your desktop now.');
        lockPanel();
      }} catch (error) {{
        setStatus(`Error: ${{error.message}}`, true);
        sendBtn.disabled = false;
      }}
    }});

    boot();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)
