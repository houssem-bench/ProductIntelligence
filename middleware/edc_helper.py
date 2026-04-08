from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd
from rapidfuzz import process

from middleware.pubchem_enrichment import enrich_chemical


# E-number fallback map for common additives if direct PubChem lookup by code fails.
STATIC_E_TO_CHEMICAL = {
    "E202": "potassium sorbate",
    "E330": "citric acid",
    "E450I": "disodium diphosphate",
    "E415": "xanthan gum",
    "E420": "sorbitol",
    "E503II": "ammonium bicarbonate",
    "E500II": "sodium bicarbonate",
}


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_tedx_set(path: str) -> set[str]:
    df = pd.read_excel(path)
    normalized_columns = {col: str(col).strip().lower().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=normalized_columns)

    for candidate in ("chemical_name", "name"):
        if candidate in df.columns:
            return set(df[candidate].dropna().astype(str).map(normalize_name))

    raise ValueError("tedx.xls must contain a chemical name column")


def load_effects_list(path: str) -> dict[str, dict[str, str]]:
    df = pd.read_excel(path)
    normalized_columns = {col: str(col).strip().lower().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=normalized_columns)

    name_col = "name_and_abbreviation"
    health_col = "health_effects"
    env_col = "environmental_effects" if "environmental_effects" in df.columns else "evironmental_effects"

    if name_col not in df.columns or health_col not in df.columns or env_col not in df.columns:
        raise ValueError(
            f"{path} must contain Name and abbreviation, health effects, and environmental effects columns"
        )

    effects_map: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        source_name = str(row.get(name_col, "")).strip()
        if not source_name:
            continue

        effects_map[normalize_name(source_name)] = {
            "source_name": source_name,
            "health_effects": str(row.get(health_col, "")).strip(),
            "environmental_effects": str(row.get(env_col, "")).strip(),
        }

    return effects_map


def lookup_effects(
    chemical_name: str,
    effects_map: dict[str, dict[str, str]],
    threshold: int,
) -> dict[str, str] | None:
    if not chemical_name or not effects_map:
        return None

    key = normalize_name(chemical_name)
    if key in effects_map:
        return effects_map[key]

    match = process.extractOne(key, effects_map.keys())
    if not match:
        return None

    matched_key, score, _ = match
    if score < threshold:
        return None

    return effects_map[matched_key]


def resolve_additive_to_chemical(
    additive: str,
    pubchem_cache: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    additive_upper = str(additive).upper().strip()

    if additive_upper in STATIC_E_TO_CHEMICAL:
        chemical = STATIC_E_TO_CHEMICAL[additive_upper]
        if chemical not in pubchem_cache:
            pubchem_cache[chemical] = enrich_chemical(chemical)
        return chemical, pubchem_cache[chemical]

    candidates = [additive_upper, additive_upper.replace("E", "E-"), additive_upper.replace("E", "INS NO.")]

    for candidate in candidates:
        info = enrich_chemical(candidate)
        if info.get("cid"):
            chemical_name = info.get("identity", {}).get("iupac_name") or candidate
            pubchem_cache[chemical_name] = info
            return chemical_name, info

    return None, None


def is_edc(chemical_name: str, tedx_set: set[str], threshold: int) -> tuple[bool, float | None, str | None]:
    if not chemical_name or not tedx_set:
        return False, None, None

    match = process.extractOne(normalize_name(chemical_name), tedx_set)
    if not match:
        return False, None, None

    matched_name, score, _ = match
    return score >= threshold, float(score), matched_name


def build_edc_list(
    best_details: dict[str, Any],
    tedx_set: set[str],
    effects_maps: dict[str, dict[str, dict[str, str]]],
    list_paths: dict[str, str],
    threshold: int,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    pubchem_cache: dict[str, dict[str, Any]] = {}

    for additive in best_details.get("additives", []):
        if not str(additive).startswith("E"):
            continue

        chemical_name, pubchem_data = resolve_additive_to_chemical(str(additive), pubchem_cache)
        if not chemical_name:
            continue

        matched, score, matched_name = is_edc(chemical_name, tedx_set, threshold)
        if not matched:
            continue

        list_effects = {
            list_key: lookup_effects(chemical_name, effects_maps.get(list_key, {}), threshold)
            for list_key in list_paths
        }

        hits.append(
            {
                "additive": additive,
                "chemical_name": chemical_name,
                "matched_in_tedx": True,
                "matched_name": matched_name,
                "match_score": score,
                "threshold": threshold,
                "health_effects": {k: (v.get("health_effects") if v else None) for k, v in list_effects.items()},
                "environmental_effects": {k: (v.get("environmental_effects") if v else None) for k, v in list_effects.items()},
                "source_names": {k: (v.get("source_name") if v else None) for k, v in list_effects.items()},
                "pubchem": pubchem_data,
            }
        )

    return hits


def load_reference_data(
    tedx_path: str,
    list_paths: dict[str, str],
) -> tuple[set[str], dict[str, dict[str, dict[str, str]]], dict[str, str | None]]:
    errors: dict[str, str | None] = {"tedx_error": None}
    for list_key in list_paths:
        errors[f"{list_key}_error"] = None

    tedx_set: set[str] = set()
    effects_maps: dict[str, dict[str, dict[str, str]]] = {list_key: {} for list_key in list_paths}

    try:
        tedx_set = load_tedx_set(tedx_path)
    except Exception as exc:  # pylint: disable=broad-except
        errors["tedx_error"] = str(exc)

    for list_key, list_path in list_paths.items():
        try:
            effects_maps[list_key] = load_effects_list(list_path)
        except Exception as exc:  # pylint: disable=broad-except
            errors[f"{list_key}_error"] = str(exc)

    return tedx_set, effects_maps, errors
