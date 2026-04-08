# Product Segmentation + Lens + EDC Web Analyzer

This project is a Flask web application that analyzes food-product images end-to-end:

1. Segments one image into possible product objects.
2. Reverse-searches the selected crop with Google Lens through SerpAPI.
3. Resolves product ingredients with Open Food Facts.
4. Verifies additive chemicals against TEDX endocrine disruptor data.
5. Enriches matched chemicals with effects lists and PubChem metadata.

## What Is Implemented

- Flask website UI for upload, segmentation preview, product selection, and EDC result view.
- YOLOv8 segmentation with contour fallback when ultralytics is unavailable.
- Nested-segmentation merge logic to reduce false split detections from package artwork.
- Selected-product Lens workflow (crop -> Lens title -> OFF -> EDC).
- Dedicated EDC helper module separated from route/controller code.

## Project Structure

- Main web server: [server_test_seg_edc.py](server_test_seg_edc.py)
- Website template/UI: [templates/index.html](templates/index.html)
- Segmentation pipeline: [ImageC/main.py](ImageC/main.py)
- EDC matching helper: [edc_helper.py](edc_helper.py)
- Open Food Facts search logic: [testSearch3.py](testSearch3.py)
- PubChem enrichment logic: [pubchem_enrichment.py](pubchem_enrichment.py)
- Dependencies list: [requirements.txt](requirements.txt)

## Runtime Flow (Inner Workings)

### 1) Browser Upload -> Segmentation

- Frontend action from [templates/index.html](templates/index.html) sends image file to `/analyze`.
- Route `/analyze` in [server_test_seg_edc.py](server_test_seg_edc.py) calls:
	- `workflow.analyze_image(...)` from [ImageC/main.py](ImageC/main.py)
- Segmentation result includes product objects with:
	- label
	- confidence
	- bounding box
	- crop path

### 2) Segmentation Internals

Inside [ImageC/main.py](ImageC/main.py):

- `ProductWorkflow._detect_with_yolo(...)` performs model inference.
- Post-processing steps:
	- `_suppress_nested_detections(...)`
	- `_suppress_high_overlap_conflicts(...)`
	- `_suppress_mask_overlap_conflicts(...)`
- Nested detections now merge into a single object using:
	- `_merge_confidence(...)`
	- `_merge_detection_geometry(...)`

This is specifically designed to handle realistic food imagery printed on product packaging.

### 3) Selected Product -> Lens Resolution

- Frontend sends selected `crop_path` and `detected_label` to `/scrape-ingredients`.
- Route `/scrape-ingredients` in [server_test_seg_edc.py](server_test_seg_edc.py) calls:
	- `resolve_product_name_with_lens(...)`
	- which calls `search_google_lens_matches(...)`
- If Lens is unavailable or fails, fallback is the segmentation label.

### 4) Product Name -> Open Food Facts

- `search_for_ingredients_with_edc(...)` in [server_test_seg_edc.py](server_test_seg_edc.py) calls:
	- `search_products(...)` from [testSearch3.py](testSearch3.py)
- [testSearch3.py](testSearch3.py) builds a best product candidate and extracts:
	- ingredients
	- additives (E-numbers)
	- nutrition-derived risk score

### 5) Additives -> EDC Verification

- [server_test_seg_edc.py](server_test_seg_edc.py) delegates EDC operations to [edc_helper.py](edc_helper.py).
- Main EDC helper calls:
	- `load_reference_data(...)`
	- `build_edc_list(...)`
- EDC helper internals include:
	- `load_tedx_set(...)`
	- `load_effects_list(...)`
	- `lookup_effects(...)`
	- `resolve_additive_to_chemical(...)`
	- `is_edc(...)`

### 6) Chemical Enrichment (PubChem)

- [edc_helper.py](edc_helper.py) calls `enrich_chemical(...)` from [pubchem_enrichment.py](pubchem_enrichment.py).
- Enrichment includes:
	- CID resolution
	- structure properties
	- synonyms
	- selected hazard/toxicity text sections

## File Call Map (Who Calls What)

### Request entrypoints

- `/` -> `home()` in [server_test_seg_edc.py](server_test_seg_edc.py) -> `render_template("index.html")`.
- `/analyze` -> `analyze_image()` in [server_test_seg_edc.py](server_test_seg_edc.py) -> `ProductWorkflow.analyze_image()` in [ImageC/main.py](ImageC/main.py).
- `/scrape-ingredients` -> `scrape_ingredients()` in [server_test_seg_edc.py](server_test_seg_edc.py).

### Core call chain for selected product analysis

1. `scrape_ingredients()` in [server_test_seg_edc.py](server_test_seg_edc.py)
2. `resolve_product_name_with_lens()` in [server_test_seg_edc.py](server_test_seg_edc.py)
3. `search_google_lens_matches()` in [server_test_seg_edc.py](server_test_seg_edc.py)
4. `search_for_ingredients_with_edc()` in [server_test_seg_edc.py](server_test_seg_edc.py)
5. `search_products()` in [testSearch3.py](testSearch3.py)
6. `load_reference_data()` in [edc_helper.py](edc_helper.py)
7. `build_edc_list()` in [edc_helper.py](edc_helper.py)
8. `resolve_additive_to_chemical()` in [edc_helper.py](edc_helper.py)
9. `enrich_chemical()` in [pubchem_enrichment.py](pubchem_enrichment.py)

## Data Files and Their Roles

- TEDX reference chemicals: [tedx.xls](tedx.xls)
- Additional effects datasets:
	- [list1.xlsx](list1.xlsx)
	- [list2.xlsx](list2.xlsx)
	- [list3.xlsx](list3.xlsx)
- Runtime ingredient log: [ingredients.csv](ingredients.csv)
- Generated image and metadata storage:
	- [uploads](uploads)

## API Endpoints

- `GET /`
	- Serves the website UI from [templates/index.html](templates/index.html)

- `POST /analyze`
	- Input: multipart form-data with `image`
	- Output: segmentation payload and `visual_matches` compatibility field

- `POST /scrape-ingredients`
	- Input options:
		- `product_name`, or
		- `crop_path` + `detected_label`
	- Output:
		- resolved product name
		- ingredient text
		- `edc_list`
		- Lens metadata (`lens_matches`, `lens_error`, `public_image_url`)

- `GET /uploads/<path>`
	- Serves stored crops/originals under uploads

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure model and dataset files exist in project root:
	 - `yolov8n-seg.pt`
	 - `tedx.xls`
	 - `list1.xlsx`, `list2.xlsx`, `list3.xlsx`

## Configuration

Current config is set in [server_test_seg_edc.py](server_test_seg_edc.py):

- `SERPAPI_KEY`
- `NGROK_BASE_URL`
- `LENS_MAX_MATCHES`
- `EDC_THRESHOLD`

Recommended production-style setup is environment variables for keys and tunnel URL.

## Run

```bash
python server_test_seg_edc.py
```

Open:

- http://127.0.0.1:3001

## Error Handling and Fallbacks

- If ultralytics is missing:
	- Segmentation endpoint reports unavailable status.
- If Lens cannot run:
	- Product resolution falls back to detected label.
- If OFF request fails:
	- Returns structured error payload.
- If TEDX/list files fail to load:
	- Returns `tedx_error` and/or `list*_error` fields.

## Known Notes

- [requirements.txt](requirements.txt) contains legacy packages not required by every path.
- [server.py](server.py) exists as an older/alternate server and is not the primary web app documented here.

## Verification Checklist

1. Start server and load `/`.
2. Upload an image and confirm detected product cards appear.
3. Select one card and confirm Lens + OFF + EDC response is shown.
4. Confirm ingredient history appends to [ingredients.csv](ingredients.csv).
5. Confirm crop files and metadata are written under [uploads](uploads).