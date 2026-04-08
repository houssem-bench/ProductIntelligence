from __future__ import annotations

from typing import Any

import requests


PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest"
HEADERS = {"User-Agent": "product-edc-screen/1.0"}


def _get_json(url: str, timeout: int = 20) -> dict[str, Any] | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        if "application/json" not in response.headers.get("Content-Type", "").lower():
            return None
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def resolve_cid_by_name(name: str) -> int | None:
    if not name:
        return None

    url = f"{PUBCHEM_BASE}/pug/compound/name/{requests.utils.quote(name)}/cids/JSON"
    data = _get_json(url)
    if not data:
        return None

    cid_list = data.get("IdentifierList", {}).get("CID", [])
    if not cid_list:
        return None

    return int(cid_list[0])


def fetch_properties_by_cid(cid: int) -> dict[str, Any]:
    prop_fields = "IUPACName,MolecularFormula,MolecularWeight,CanonicalSMILES,InChIKey"
    url = f"{PUBCHEM_BASE}/pug/compound/cid/{cid}/property/{prop_fields}/JSON"
    data = _get_json(url)
    if not data:
        return {}

    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return {}

    row = props[0]
    return {
        "iupac_name": row.get("IUPACName"),
        "molecular_formula": row.get("MolecularFormula"),
        "molecular_weight": row.get("MolecularWeight"),
        "canonical_smiles": row.get("ConnectivitySMILES") or row.get("CanonicalSMILES"),
        "inchikey": row.get("InChIKey"),
    }


def fetch_synonyms_by_cid(cid: int, max_items: int = 25) -> list[str]:
    url = f"{PUBCHEM_BASE}/pug/compound/cid/{cid}/synonyms/JSON"
    data = _get_json(url)
    if not data:
        return []

    info = data.get("InformationList", {}).get("Information", [])
    if not info:
        return []

    synonyms = info[0].get("Synonym", [])
    return [str(s) for s in synonyms[:max_items]]


def _extract_string_values(node: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for info in node.get("Information", []):
        for value in info.get("Value", {}).get("StringWithMarkup", []):
            text = value.get("String")
            if text:
                values.append(text)
    return values


def _collect_section_text(section: dict[str, Any], wanted: set[str], out: dict[str, list[str]]) -> None:
    heading = section.get("TOCHeading")
    if heading in wanted:
        out.setdefault(heading, [])
        out[heading].extend(_extract_string_values(section))

    for child in section.get("Section", []):
        _collect_section_text(child, wanted, out)


def fetch_safety_sections_by_cid(cid: int) -> dict[str, list[str]]:
    url = f"{PUBCHEM_BASE}/pug_view/data/compound/{cid}/JSON"
    data = _get_json(url, timeout=30)
    if not data:
        return {}

    wanted = {"Primary Hazards", "Safety and Hazards", "Toxicity"}
    collected: dict[str, list[str]] = {}

    for section in data.get("Record", {}).get("Section", []):
        _collect_section_text(section, wanted, collected)

    # Deduplicate while keeping order
    for key, values in list(collected.items()):
        deduped = list(dict.fromkeys(v.strip() for v in values if v and v.strip()))
        collected[key] = deduped[:20]

    return collected


def enrich_chemical(chemical_name: str) -> dict[str, Any]:
    cid = resolve_cid_by_name(chemical_name)
    if not cid:
        return {
            "query": chemical_name,
            "cid": None,
            "identity": {},
            "structure": {},
            "safety": {},
            "synonyms": [],
            "error": "No CID found",
        }

    properties = fetch_properties_by_cid(cid)
    synonyms = fetch_synonyms_by_cid(cid)
    safety_sections = fetch_safety_sections_by_cid(cid)

    return {
        "query": chemical_name,
        "cid": cid,
        "identity": {
            "iupac_name": properties.get("iupac_name"),
        },
        "structure": {
            "molecular_formula": properties.get("molecular_formula"),
            "molecular_weight": properties.get("molecular_weight"),
            "canonical_smiles": properties.get("canonical_smiles"),
            "inchikey": properties.get("inchikey"),
        },
        "safety": {
            "primary_hazards": safety_sections.get("Primary Hazards", []),
            "safety_and_hazards": safety_sections.get("Safety and Hazards", []),
            "toxicity": safety_sections.get("Toxicity", []),
        },
        "synonyms": synonyms,
        "error": None,
    }
