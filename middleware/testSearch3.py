from typing import Optional

import pandas as pd
import requests
from rapidfuzz import process


OPENFOODFACTS_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
PUBCHEM_NAME_TO_IUPAC_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{query}/property/IUPACName/JSON"
)
REQUEST_HEADERS = {"User-Agent": "product-edc-screen/1.0"}


def simplify_product(product: dict) -> dict:
    return {
        "product_name": product.get("product_name"),
        "nova_group": product.get("nova_group"),
        "nutriscore": product.get("nutriscore_grade"),
        "additives_count": len(product.get("additives_tags", [])),
        "allergens_count": len(product.get("allergens_tags", [])),
    }


def calculate_risk(product: dict) -> int:
    score = 0

    if product.get("nova_group") == 4:
        score += 3
    elif product.get("nova_group") == 3:
        score += 2

    additives_count = len(product.get("additives", []))
    if additives_count >= 5:
        score += 3
    elif additives_count >= 2:
        score += 2
    elif additives_count == 1:
        score += 1

    if product.get("nutriscore") in ["d", "e"]:
        score += 2
    elif product.get("nutriscore") == "c":
        score += 1

    return score


def detailed_product(product: dict) -> dict:
    ingredients = []
    additives = []

    for item in product.get("ingredients", []):
        text = item.get("text")
        ingredient_id = item.get("id", "")

        if text and not ingredient_id.startswith("fr:"):
            ingredients.append(text)

        if ingredient_id.startswith("en:e"):
            additives.append(ingredient_id.replace("en:", "").upper())

    nova = product.get("nova_group")
    nutriscore = product.get("nutriscore_grade")

    return {
        "product_name": product.get("product_name"),
        "ingredients": list(dict.fromkeys(ingredients)),
        "additives": list(dict.fromkeys(additives)),
        "nova_group": nova,
        "nutriscore": nutriscore,
        "risk_score": calculate_risk(
            {
                "nova_group": nova,
                "additives": additives,
                "allergens": product.get("allergens_tags", []),
                "nutriscore": nutriscore,
            }
        ),
    }


def search_products(product_name: str) -> Optional[dict]:
    params = {
        "search_terms": product_name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 10,
        "tagtype_0": "countries",
        "tag_contains_0": "contains",
        "tag_0": "tunisia",
    }

    try:
        response = requests.get(
            OPENFOODFACTS_SEARCH_URL,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=12,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            return {
                "error": "OpenFoodFacts did not return JSON.",
                "status_code": response.status_code,
                "content_type": content_type,
                "response_preview": response.text[:200].strip(),
            }

        data = response.json()
    except requests.RequestException as exc:
        return {"error": f"OpenFoodFacts request failed: {exc}"}
    except ValueError as exc:
        return {
            "error": f"OpenFoodFacts returned invalid JSON: {exc}",
            "response_preview": response.text[:200].strip() if "response" in locals() else "",
        }

    products = data.get("products", [])
    if not products:
        return None

    products = sorted(
        products,
        key=lambda p: (p.get("nova_group") is not None, len(p.get("ingredients", []))),
        reverse=True,
    )

    top3 = [simplify_product(p) for p in products[:3]]
    best = detailed_product(products[0])

    return {
        "top_3": top3,
        "best_product_details": best,
    }


def e_to_chemical(e_number: str) -> Optional[str]:
    query = e_number.replace("E", "E ")
    url = PUBCHEM_NAME_TO_IUPAC_URL.format(query=requests.utils.quote(query))

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=8)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            return None

        data = response.json()
        return data["PropertyTable"]["Properties"][0]["IUPACName"]
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None


def load_tedx(path: str = "tedx.xls") -> set:
    df = pd.read_excel(path)
    normalized_columns = {
        col: str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    }
    df = df.rename(columns=normalized_columns)

    chemical_col = None
    for candidate in ("chemical_name", "name"):
        if candidate in df.columns:
            chemical_col = candidate
            break

    if not chemical_col:
        raise ValueError("tedx.xls must contain a chemical name column")

    return set(df[chemical_col].dropna().astype(str).str.lower().str.strip())


def is_edc(chemical: str, tedx_set: set, threshold: int = 85) -> bool:
    if not chemical or not tedx_set:
        return False

    match = process.extractOne(chemical.lower(), tedx_set)
    if not match:
        return False

    _, score, _ = match
    return score >= threshold


def build_edc_hits(best_product_details: dict, tedx_set: set, threshold: int = 85) -> list:
    hits = []

    for additive in best_product_details.get("additives", []):
        if not additive.startswith("E"):
            continue

        chemical_name = e_to_chemical(additive)
        if not chemical_name:
            continue

        if is_edc(chemical_name, tedx_set, threshold=threshold):
            hits.append(
                {
                    "additive": additive,
                    "chemical_name": chemical_name,
                    "matched_in_tedx": True,
                    "threshold": threshold,
                }
            )

    return hits


def search_products_with_edc(product_name: str, tedx_path: str = "tedx.xls") -> Optional[dict]:
    result = search_products(product_name)
    if not result or result.get("error"):
        return result

    try:
        tedx_set = load_tedx(tedx_path)
    except (FileNotFoundError, ValueError) as exc:
        result["edc_error"] = str(exc)
        result["edc_list"] = []
        return result

    best = result.get("best_product_details", {})
    result["edc_list"] = build_edc_hits(best, tedx_set)
    return result


if __name__ == "__main__":
    print(search_products_with_edc("brownies"))