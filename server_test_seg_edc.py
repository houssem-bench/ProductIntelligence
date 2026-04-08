from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory
from serpapi import GoogleSearch

from middleware.edc_helper import build_edc_list as build_edc_list_from_helper
from middleware.edc_helper import load_reference_data as load_reference_data_from_helper
from ImageC.main import ProductWorkflow, ULTRALYTICS_AVAILABLE
from middleware.testSearch3 import search_products


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ORIGINAL_UPLOADS = os.path.join(UPLOAD_FOLDER, "incoming")
os.makedirs(ORIGINAL_UPLOADS, exist_ok=True)

INGREDIENTS_CSV = "ingredients.csv"
TEDX_PATH = "tedx.xls"
LIST_PATHS = {
    "list1": "list1.xlsx",
    "list2": "list2.xlsx",
    "list3": "list3.xlsx",
}
EDC_THRESHOLD = 90
SERPAPI_KEY = "e8715f247716f03e6f730763c1e60df0ec78c84c90d508b9c3f7dea7696dcb50"

# 🔴 IMPORTANT: Replace this with your ngrok HTTPS URL
NGROK_BASE_URL = "https://subattenuate-joanne-vacuous.ngrok-free.dev"
LENS_MAX_MATCHES = 5

workflow = ProductWorkflow(model_name="yolov8n-seg.pt")


def init_csv_file() -> None:
    if not os.path.exists(INGREDIENTS_CSV):
        with open(INGREDIENTS_CSV, "w", newline="", encoding="utf-8") as f:
            f.write("Product Name,Ingredients,Scanned At\n")


def save_ingredients_to_csv(product_name: str, ingredients: str) -> bool:
    init_csv_file()
    try:
        with open(INGREDIENTS_CSV, "a", newline="", encoding="utf-8") as f:
            ingredients_clean = ingredients.replace("\n", " ").replace("\r", " ")
            f.write(f'"{product_name}","{ingredients_clean}","{datetime.now().isoformat()}"\n')
        return True
    except Exception:
        return False


def load_reference_data() -> tuple[set[str], dict[str, dict[str, dict[str, str]]], dict[str, str | None]]:
    # Backward-compatible wrapper for previous local smoke tests.
    return load_reference_data_from_helper(TEDX_PATH, LIST_PATHS)


def build_edc_list(
    best_details: dict[str, Any],
    tedx_set: set[str],
    effects_maps: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    return build_edc_list_from_helper(best_details, tedx_set, effects_maps, LIST_PATHS, EDC_THRESHOLD)


def crop_path_to_upload_route(crop_path: str | None) -> str | None:
    if not crop_path:
        return None

    normalized = str(crop_path).replace("\\", "/")
    marker = "/uploads/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
    elif normalized.startswith("uploads/"):
        relative = normalized[len("uploads/"):]
    else:
        relative = normalized

    return f"/uploads/{relative.lstrip('/')}"


def build_public_url_from_route(upload_route: str | None) -> str | None:
    if not upload_route or not NGROK_BASE_URL:
        return None
    return f"{NGROK_BASE_URL}{upload_route}"


def search_google_lens_matches(public_image_url: str, max_matches: int = LENS_MAX_MATCHES) -> tuple[list[dict[str, Any]], str | None]:
    if not SERPAPI_KEY:
        return [], "Missing SERPAPI_KEY environment variable"

    params = {
        "engine": "google_lens",
        "url": public_image_url,
        "api_key": SERPAPI_KEY,
        "type": "products",
        "safe": "off",
        "country": "TN",
    }

    try:
        results = GoogleSearch(params).get_dict()
    except Exception as exc:  # pylint: disable=broad-except
        return [], f"SerpAPI request failed: {exc}"

    matches = results.get("visual_matches", [])
    if not isinstance(matches, list):
        return [], "SerpAPI returned invalid visual_matches payload"

    return matches[:max_matches], None


def resolve_product_name_with_lens(crop_path: str | None, detected_label: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    upload_route = crop_path_to_upload_route(crop_path)
    public_image_url = build_public_url_from_route(upload_route)

    meta: dict[str, Any] = {
        "upload_route": upload_route,
        "public_image_url": public_image_url,
        "lens_error": None,
    }

    if not public_image_url:
        if not NGROK_BASE_URL:
            meta["lens_error"] = "Missing NGROK_BASE_URL environment variable"
        else:
            meta["lens_error"] = "Missing crop path for Lens search"
        return detected_label, [], meta

    matches, lens_error = search_google_lens_matches(public_image_url)
    meta["lens_error"] = lens_error

    if matches:
        best_title = str(matches[0].get("title") or "").strip()
        if best_title:
            return best_title, matches, meta

    return detected_label, matches, meta


def search_for_ingredients_with_edc(product_name: str) -> dict[str, Any]:
    result = search_products(product_name)
    if not result:
        return {"error": "No products found"}
    if isinstance(result, dict) and result.get("error"):
        return result

    best_details = result.get("best_product_details", {})
    ingredients_list = best_details.get("ingredients", [])
    additives_list = best_details.get("additives", [])

    tedx_set, effects_maps, ref_errors = load_reference_data_from_helper(TEDX_PATH, LIST_PATHS)
    edc_list = build_edc_list_from_helper(best_details, tedx_set, effects_maps, LIST_PATHS, EDC_THRESHOLD) if tedx_set else []

    ingredients_text = ""
    if ingredients_list:
        ingredients_text += "Ingredients:\n" + ", ".join(ingredients_list)
    if additives_list:
        ingredients_text += "\n\nAdditives:\n" + ", ".join(additives_list)

    return {
        "source": "Open Food Facts + TEDX/Lists + PubChem",
        "product_name": best_details.get("product_name", product_name),
        "ingredients": ingredients_text.strip(),
        "best_product_details": best_details,
        "edc_list": edc_list,
        **ref_errors,
    }


def generate_html_results(segmentation_products: list[dict[str, Any]]) -> str:
    cards = []
    for idx, product in enumerate(segmentation_products, 1):
        title = product.get("label", "product")
        confidence = product.get("confidence", 0)
        cards.append(
            f"<li><strong>#{idx}</strong> {title} (confidence: {confidence}) "
            f"<button onclick=\"scrapeIngredients('{title}')\">Ingredients</button></li>"
        )

    return f"""<!doctype html>
<html>
  <head><meta charset=\"utf-8\"><title>Segmentation Test</title></head>
  <body>
    <h2>Segmentation Results</h2>
    <ul>{''.join(cards)}</ul>
    <pre id=\"result\"></pre>
    <script>
      async function scrapeIngredients(productName) {{
        const response = await fetch('/scrape-ingredients', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ product_name: productName }})
        }});
        const data = await response.json();
        document.getElementById('result').textContent = JSON.stringify(data, null, 2);
      }}
    </script>
  </body>
</html>"""


@app.route("/", methods=["GET"])
def home() -> str:
    return render_template(
        "index.html",
        ultralytics_available=ULTRALYTICS_AVAILABLE,
        lens_ready=bool(SERPAPI_KEY and NGROK_BASE_URL),
        ngrok_base_url=NGROK_BASE_URL,
    )


@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_file(filename: str):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/analyze", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    if not ULTRALYTICS_AVAILABLE:
        return jsonify({"error": "Segmentation unavailable: ultralytics is not installed."}), 500

    image_file = request.files["image"]
    safe_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{image_file.filename}"
    incoming_path = os.path.join(ORIGINAL_UPLOADS, safe_name)
    image_file.save(incoming_path)

    try:
        analysis = workflow.analyze_image(incoming_path).to_dict()
        products = analysis.get("products", [])
        enriched_products: list[dict[str, Any]] = []

        for product in products:
            product_copy = dict(product)
            upload_route = crop_path_to_upload_route(product_copy.get("crop_path"))
            product_copy["crop_route"] = upload_route
            product_copy["lens_ready"] = bool(upload_route and NGROK_BASE_URL and SERPAPI_KEY)
            enriched_products.append(product_copy)

        analysis["products"] = enriched_products
        return jsonify(
            {
                "status": analysis.get("status"),
                "total_products": analysis.get("total_products"),
                "segmentation": analysis,
                # Kept for frontend compatibility with existing flow.
                "visual_matches": enriched_products,
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Segmentation failed: {exc}"}), 500


@app.route("/results", methods=["POST"])
def results_page():
    data = request.get_json(silent=True) or {}
    visual_matches = data.get("visual_matches", [])
    return generate_html_results(visual_matches), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/scrape-ingredients", methods=["POST"])
def scrape_ingredients():
    data = request.get_json(silent=True) or {}
    product_name = str(data.get("product_name", "")).strip()
    crop_path = str(data.get("crop_path", "")).strip() or None
    detected_label = str(data.get("detected_label", "")).strip() or "product"

    if not product_name and not crop_path:
        return jsonify({"error": "Provide product_name or crop_path with detected_label"}), 400

    lens_matches: list[dict[str, Any]] = []
    lens_meta: dict[str, Any] = {
        "upload_route": crop_path_to_upload_route(crop_path),
        "public_image_url": None,
        "lens_error": None,
    }
    resolved_name = product_name

    # Preferred workflow for selected segmentation result: crop -> Lens -> OFF -> EDC.
    if not resolved_name:
        resolved_name, lens_matches, lens_meta = resolve_product_name_with_lens(crop_path, detected_label)

    enriched = search_for_ingredients_with_edc(resolved_name)
    if enriched.get("error"):
        error_payload = {
            **enriched,
            "resolved_product_name": resolved_name,
            "detected_label": detected_label,
            "lens_matches": lens_matches,
            **lens_meta,
        }
        return jsonify(error_payload), 400

    saved = save_ingredients_to_csv(resolved_name, enriched.get("ingredients", ""))
    enriched["saved_to_csv"] = saved
    enriched["scanned_at"] = datetime.now().isoformat()
    enriched["resolved_product_name"] = resolved_name
    enriched["detected_label"] = detected_label
    enriched["lens_matches"] = lens_matches
    enriched.update(lens_meta)

    return jsonify(enriched), 200


if __name__ == "__main__":
    app.run(port=3001, debug=True)
