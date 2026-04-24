# ProductIntelligence_V2

Clean and maintainable V2 of Product Intelligence, redesigned for modularity, security, and performance.

## Overview

ProductIntelligence_V2 analyzes images containing multiple products and extracts ingredient information using a staged fallback pipeline.

### Core goals achieved

- Clean modular architecture.
- No API keys hardcoded.
- `.env`-driven configuration.
- Structured logging with request IDs.
- Optimized fallback order: Barcode -> Lens -> OCR.
- Minimal frontend for product selection before analysis.
- Optional one-shot "Capture from Phone" flow to send a single photo from smartphone to desktop.
- Simple tests for pipeline behavior and `/analyze` endpoint.

## Architecture

```text
ProductIntelligence_V2/
├── app/
│   ├── main.py
│   ├── routes/
│   │   ├── analyze.py
│   │   └── health.py
│   ├── core/
│   │   ├── config.py
│   │   ├── container.py
│   │   └── http.py
│   ├── pipeline/
│   │   └── orchestrator.py
│   ├── services/
│   │   ├── segmentation_service.py
│   │   ├── barcode_service.py
│   │   ├── lens_service.py
│   │   └── ocr_service.py
│   ├── providers/
│   │   ├── product_facts_provider.py
│   │   └── lens_provider.py
│   ├── models/
│   │   └── schemas.py
│   ├── utils/
│   │   ├── request_context.py
│   │   └── cache.py
│   └── logging/
│       └── setup.py
├── frontend/
│   └── index.html
├── uploads/
│   ├── incoming/
│   └── crops/
├── tests/
│   ├── test_pipeline.py
│   └── test_api_analyze.py
├── .env.example
├── requirements.txt
└── README.md
```

## Pipeline (optimized)

New execution order:

1. **[BARCODE]** QR/Barcode detection + OFF/OBF lookup.
2. **[LENS]** Google Lens title resolution + OFF/OBF lookup by name.
3. **[OCR]** OCR extraction only as last fallback.

This order reduces expensive OCR usage and improves average response time.

## Request flow

1. `POST /analyze` with image.
  - Supports `segmentation_mode` = `auto` | `single` | `multi`.
  - Optional `expected_products` for multi mode.
  - Input image can come from desktop file upload/drag-drop or one-shot phone capture transfer.
2. Backend segments products and returns crops + session ID.
3. Frontend displays all crops with checkboxes.
4. User selects products to analyze.
5. `POST /analyze/selected` with selected `product_ids`.
6. Backend runs pipeline only for selected products.

Phone capture helper flow:

1. User clicks `Capture from Phone` in desktop UI.
2. Backend creates a short-lived one-shot session token.
3. Phone opens the generated link/QR page and uploads one image.
4. Desktop auto-downloads that image and injects it into the same analyze flow.

## API Endpoints

- `GET /health`
  - Service health check.

- `GET /health/readiness`
  - Runtime readiness snapshot for `barcode`, `lens`, and `ocr`.
  - Returns `ok` flag + `reason` to explain why a step is disabled.

- `POST /analyze`
  - Input: multipart image file + optional form fields (`segmentation_mode`, `expected_products`).
  - Output: `session_id` + segmented products (`product_id`, `bbox`, `crop_url`, confidence).

- `POST /analyze/selected`
  - Input JSON:

    ```json
    {
      "session_id": "...",
      "product_ids": ["1", "2"]
    }

    ```

  - Output: analysis results for selected products only.
  - Side effect: logs a `products_list` JSON payload in the backend console after each batch analysis.

- `POST /phone-capture/session`
  - Creates short-lived one-shot session for smartphone capture handoff.

- `GET /phone-capture/session/{token}/status`
  - Returns session status (`pending`, `ready`, `consumed`, `expired`).

- `POST /phone-capture/session/{token}/upload`
  - Phone uploads single image for desktop session.

- `GET /phone-capture/session/{token}/consume`
  - Desktop consumes uploaded image once and reuses existing `/analyze` flow.

## Security

- No secret in code.
- Uses environment variables loaded via `python-dotenv`.
- `.env.example` provided.

## Performance choices

- HTTP timeout + retry with exponential backoff (`httpx`).
- Simple in-memory TTL cache for Lens and product lookup responses.
- Bounded concurrency for selected products (`MAX_PARALLEL_ANALYSES`).

## Startup diagnostics

- On startup, backend logs explicit readiness lines:
  - `[READINESS] Barcode: OK/KO (...)`
  - `[READINESS] Lens: OK/KO (...)`
  - `[READINESS] OCR: OK/KO (...)`
- This makes it easy to see immediately if Lens/OCR are disabled by missing env/runtime.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```bash
copy .env.example .env
```

Run server:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- `http://127.0.0.1:8000`

## Environment variables

See `.env.example`. Main variables:

- `SERPAPI_KEY`
- `PUBLIC_BASE_URL`
- `NGROK_BASE_URL`
- `ENABLE_NGROK`
- `NGROK_AUTHTOKEN`
- `NGROK_DOMAIN`
- `LENS_MAX_MATCHES`
- `LENS_COUNTRY`
- `LENS_SAFE`
- `GROK_API_KEY`
- `GROK_MODEL`
- `GROK_BASE_URL`
- `GROQ_API_KEY` (supported alias for integration tests)
- `GROQ_MODEL` (supported alias for integration tests)
- `GROQ_BASE_URL` (supported alias for integration tests, e.g. `https://api.groq.com/openai/v1`)
- `RUN_GROQ_INTEGRATION` (set to `1` to run real Groq test)
- `REQUEST_TIMEOUT_SECONDS`
- `MAX_RETRIES`
- `CACHE_TTL_SECONDS`
- `MAX_PARALLEL_ANALYSES`
- `ENABLE_YOLO`
- `YOLO_MODEL_PATH`
- `TESSERACT_CMD`
- `PHONE_CAPTURE_TTL_SECONDS`
- `PHONE_CAPTURE_MAX_UPLOAD_MB`
- `PHONE_CAPTURE_POLL_INTERVAL_MS`

## Tests

```bash
pytest -q
```

Run Groq real integration test (optional):

```bash
RUN_GROQ_INTEGRATION=1 GROQ_API_KEY=your_key_here GROQ_MODEL=llama-3.3-70b-versatile GROQ_BASE_URL=https://api.groq.com/openai/v1 pytest tests/test_grok_title_extraction.py -m integration -q -s
```

PowerShell:

```powershell
$env:RUN_GROQ_INTEGRATION = "1"
$env:GROQ_API_KEY = "your_key_here"
$env:GROQ_MODEL = "llama-3.3-70b-versatile"
$env:GROQ_BASE_URL = "https://api.groq.com/openai/v1"
pytest tests/test_grok_title_extraction.py -m integration -q -s
```

Included tests:

- Pipeline ordering and fallback behavior.
- `/analyze` endpoint response contract.

## Current limitations

- OCR depends on local Tesseract availability.
- Lens step requires `SERPAPI_KEY` plus a public image URL base (`PUBLIC_BASE_URL` or `NGROK_BASE_URL`) reachable from SerpAPI, or `ENABLE_NGROK=true` so the service can create one at runtime.
- In-memory session/cache storage is not distributed.

## Future improvements

- Replace in-memory session store with Redis.
- Add async background queue for heavy image workloads.
- Add calibrated confidence model from real benchmark dataset.
- Add richer OCR model stack and multilingual tuning.
