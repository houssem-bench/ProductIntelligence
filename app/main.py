from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import PROJECT_ROOT, get_settings
from app.core.container import ServiceContainer
from app.logging.setup import configure_logging
from app.routes.analyze import router as analyze_router
from app.routes.health import router as health_router
from app.utils.request_context import clear_request_id, set_request_id


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        set_request_id(request_id)

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            clear_request_id()


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


def _log_startup_readiness(container: ServiceContainer) -> None:
    barcode_ok, barcode_reason, barcode_backends = container.barcode_service.get_readiness()
    lens_ok, lens_reason = container.lens_service.get_readiness()
    ocr_ok, ocr_reason = container.ocr_service.get_readiness()

    barcode_state = "OK" if barcode_ok else "KO"
    lens_state = "OK" if lens_ok else "KO"
    ocr_state = "OK" if ocr_ok else "KO"

    backend_summary = ", ".join(f"{name}={str(enabled).lower()}" for name, enabled in barcode_backends.items())

    logger.info("[READINESS] Barcode: %s (%s) [%s]", barcode_state, barcode_reason, backend_summary)
    logger.info("[READINESS] Lens: %s (%s)", lens_state, lens_reason)
    logger.info("[READINESS] OCR: %s (%s)", ocr_state, ocr_reason)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.container = ServiceContainer(settings)
    _log_startup_readiness(_app.state.container)
    yield
    container: ServiceContainer = _app.state.container
    await container.close()


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(settings.uploads_dir)), name="uploads")

frontend_dir = PROJECT_ROOT / "frontend"
frontend_dir.mkdir(parents=True, exist_ok=True)


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    index_file = frontend_dir / "index.html"
    return FileResponse(index_file)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    logger.info("Validation error req=%s path=%s", request_id, request.url.path)
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "request_id": request_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    detail = exc.detail if isinstance(exc.detail, (str, list, dict)) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "detail": detail,
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("Unhandled exception req=%s path=%s", request_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
        },
    )


app.include_router(health_router)
app.include_router(analyze_router)
